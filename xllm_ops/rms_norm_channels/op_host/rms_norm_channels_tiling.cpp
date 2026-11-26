/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <algorithm>
#include <cmath>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "rms_norm_channels_tiling.h"
#include "../op_kernel/rms_norm_channels_common.h"

namespace optiling {
namespace {
// Keep in sync with the device-side tile caps.
constexpr int64_t MAX_N_TILE = 512;
constexpr int64_t MAX_C_CHUNK = 256;
constexpr int64_t TILE_ALIGN = 16;
constexpr int64_t MIN_N_TILE = 64;
constexpr int64_t RMS_NORM_CHANNELS_RANK = 5;
constexpr uint32_t RMS_NORM_CHANNELS_SPLIT_WORKSPACE_LIMIT = 8U * 1000U * 1000U;

int64_t AlignUpI64(int64_t value, int64_t align)
{
    return (value + align - 1) / align * align;
}

uint64_t AlignUpBytes32(uint64_t bytes)
{
    return (bytes + 31ULL) / 32ULL * 32ULL;
}

// DataCopyPad Normal mode pads every block to 32 B, so rows in UB advance by
// this aligned element count rather than by the logical length.
int64_t RowStride(int64_t n, int64_t elemBytes)
{
    return ((n * elemBytes + 31) / 32 * 32) / elemBytes;
}

struct TilingParams {
    int64_t numBlocks = 1;
    int64_t nTile = 1;
    int64_t cChunk = 1;
    int64_t tilesPerBatch = 1;
    int64_t totalTiles = 1;
    uint32_t tilingKey = RMS_NORM_CHANNELS_TK_STREAM;
    int64_t cPerBlock = 1;
    int64_t workspaceStride = 1;
    uint64_t workspaceFloats = 0;
};

// Shape-family routing ported from the validated direct-launch implementation.
// Every branch is a measured winner on Ascend950; changing a threshold needs a
// fresh A/B, see the operator README.
TilingParams ComputeTiling(int64_t b, int64_t c, int64_t n, int64_t dtypeSize, int64_t coreNum,
    uint64_t ubSize)
{
    TilingParams result;
    if (b <= 0 || c <= 0 || n <= 0 || dtypeSize <= 0 || coreNum <= 0) {
        return result;
    }
    if (ubSize == 0) {
        ubSize = 192ULL * 1024ULL;
    }

    if (c == 1) {
        int64_t target = (n + coreNum - 1) / coreNum;
        target = AlignUpI64(std::max<int64_t>(MIN_N_TILE, target), TILE_ALIGN);
        target = std::min<int64_t>(target, MAX_N_TILE);
        target = std::min<int64_t>(target, n);
        const int64_t nTile = target > 0 ? target : 1;
        const int64_t tilesPerBatch = (n + nTile - 1) / nTile;
        const int64_t totalTiles = b * tilesPerBatch;
        result.numBlocks = std::max<int64_t>(1, std::min<int64_t>(coreNum, totalTiles));
        result.nTile = nTile;
        result.cChunk = 1;
        result.tilesPerBatch = tilesPerBatch;
        result.totalTiles = totalTiles;
        result.tilingKey = RMS_NORM_CHANNELS_TK_SINGLE_CHANNEL;
        result.cPerBlock = 1;
        result.workspaceStride = n;
        return result;
    }

    // SIMT path for the very short N family: one warp per position.  The
    // measured A/B on Ascend950PR wins only for N <= 16 (N=1 is contiguous,
    // N=16 has a 32 B row stride); larger N stays on SPLIT_C / SMALL_C.
    const bool simtEligible = c <= 256 && n <= 16 && b * n <= 8192;
    if (simtEligible) {
        const int64_t totalPositions = b * n;
        result.numBlocks = totalPositions;
        result.nTile = 32;
        result.cChunk = totalPositions;
        result.tilesPerBatch = 1;
        result.totalTiles = totalPositions;
        result.tilingKey = RMS_NORM_CHANNELS_TK_SIMT_WARP;
        result.cPerBlock = 32;
        result.workspaceStride = n;
        return result;
    }

    const uint64_t budget = ubSize * 85ULL / 100ULL;
    // SPLIT_C pays one extra join pass over the partial rows.  It wins when the
    // serial C loop dominates: large C with a small position count.
    const bool splitLargeC = c >= 1024 && n <= 1024;
    // C=256/512 with a single batch and a very short position axis still has
    // enough serial rows to repay the split, while the partial tile stays tiny.
    const bool splitC256 = c >= 256 && c <= 512 && b == 1 && n <= 160;

    if (splitLargeC || splitC256) {
        int64_t numBlocks = std::max<int64_t>(1, std::min<int64_t>(coreNum, c));
        const int64_t cPerBlock = (c + numBlocks - 1) / numBlocks;
        const int64_t workspaceStride =
            AlignUpI64(n * static_cast<int64_t>(sizeof(float)), 32) / static_cast<int64_t>(sizeof(float));
        const double workspaceFloats = static_cast<double>(b) * numBlocks * workspaceStride;
        if (workspaceFloats <= static_cast<double>(RMS_NORM_CHANNELS_SPLIT_WORKSPACE_LIMIT)) {
            int64_t nTile = std::min<int64_t>(n, MAX_N_TILE);
            if (nTile < n) {
                nTile = AlignUpI64(nTile, TILE_ALIGN);
            }
            auto splitBytes = [&](int64_t nt, int64_t cc) -> uint64_t {
                const uint64_t rowStrideT = static_cast<uint64_t>(RowStride(nt, dtypeSize));
                const uint64_t rowStrideF = static_cast<uint64_t>(RowStride(nt, sizeof(float)));
                return static_cast<uint64_t>(cc) * rowStrideT * static_cast<uint64_t>(dtypeSize) +
                       static_cast<uint64_t>(numBlocks) * rowStrideF * sizeof(float) +
                       3ULL * AlignUpBytes32(static_cast<uint64_t>(nt) * sizeof(float)) +
                       AlignUpBytes32(static_cast<uint64_t>(cPerBlock) * dtypeSize + 32ULL) +
                       AlignUpBytes32(static_cast<uint64_t>(cPerBlock) * sizeof(float) + 32ULL) + 2048ULL;
            };
            int64_t cChunk = std::min<int64_t>(cPerBlock, 128);
            bool fitted = false;
            for (int64_t nt = nTile; nt >= 1; nt = ((nt > 16) ? nt - 16 : nt - 1)) {
                for (int64_t cc = std::min<int64_t>(cPerBlock, 128); cc >= 1;
                     cc = ((cc > 8) ? cc - 4 : cc - 1)) {
                    if (splitBytes(nt, cc) <= budget) {
                        nTile = nt;
                        cChunk = cc;
                        fitted = true;
                        break;
                    }
                }
                if (fitted) {
                    break;
                }
            }
            if (fitted) {
                result.numBlocks = numBlocks;
                result.nTile = nTile;
                result.cChunk = cChunk;
                result.tilingKey = RMS_NORM_CHANNELS_TK_SPLIT_C;
                result.cPerBlock = cPerBlock;
                result.workspaceStride = workspaceStride;
                result.workspaceFloats = static_cast<uint64_t>(workspaceFloats);
                result.tilesPerBatch = 1;
                result.totalTiles = 0;
                return result;
            }
        }
    }

    if (c <= 512) {
        int64_t target = (n + coreNum - 1) / coreNum;
        target = AlignUpI64(std::max<int64_t>(MIN_N_TILE, target), TILE_ALIGN);
        target = std::min<int64_t>(target, n);
        auto smallBytes = [&](int64_t nt) -> uint64_t {
            const uint64_t rowStride = static_cast<uint64_t>(RowStride(nt, dtypeSize));
            return static_cast<uint64_t>(c) * rowStride * static_cast<uint64_t>(dtypeSize) +
                   3ULL * AlignUpBytes32(static_cast<uint64_t>(nt) * sizeof(float)) +
                   AlignUpBytes32(static_cast<uint64_t>(c) * dtypeSize + 32ULL) +
                   AlignUpBytes32(static_cast<uint64_t>(c) * sizeof(float) + 32ULL) + 2048ULL;
        };
        while (target > 16 && smallBytes(target) > budget) {
            target = (target > 32) ? target - 16 : target / 2;
        }
        target = std::max<int64_t>(1, target);
        const int64_t nTile = std::min<int64_t>(target, n);
        const int64_t tilesPerBatch = (n + nTile - 1) / nTile;
        const int64_t totalTiles = b * tilesPerBatch;
        result.numBlocks = std::max<int64_t>(1, std::min<int64_t>(coreNum, totalTiles));
        result.nTile = nTile;
        result.cChunk = c;
        result.tilesPerBatch = tilesPerBatch;
        result.totalTiles = totalTiles;
        result.tilingKey = RMS_NORM_CHANNELS_TK_SMALL_C;
        result.cPerBlock = c;
        result.workspaceStride = n;
        return result;
    }

    int64_t target = (n + coreNum - 1) / coreNum;
    target = AlignUpI64(std::max<int64_t>(MIN_N_TILE, target), TILE_ALIGN);
    target = std::min<int64_t>(target, MAX_N_TILE);
    target = std::min<int64_t>(target, n);
    const int64_t nTile = target;
    auto streamBytes = [&](int64_t cc) -> uint64_t {
        const uint64_t rowStride = static_cast<uint64_t>(RowStride(nTile, dtypeSize));
        return 2ULL * static_cast<uint64_t>(cc) * rowStride * static_cast<uint64_t>(dtypeSize) +
               AlignUpBytes32(static_cast<uint64_t>(cc) * dtypeSize + 32ULL) +
               AlignUpBytes32(static_cast<uint64_t>(cc) * sizeof(float) + 32ULL) +
               3ULL * AlignUpBytes32(static_cast<uint64_t>(nTile) * sizeof(float)) + 2048ULL;
    };
    int64_t cChunk = std::min<int64_t>(c, MAX_C_CHUNK);
    while (cChunk > 8 && streamBytes(cChunk) > budget) {
        cChunk -= 8;
    }
    cChunk = std::max<int64_t>(1, cChunk);
    const int64_t tilesPerBatch = (n + nTile - 1) / nTile;
    const int64_t totalTiles = b * tilesPerBatch;
    result.numBlocks = std::max<int64_t>(1, std::min<int64_t>(coreNum, totalTiles));
    result.nTile = nTile;
    result.cChunk = cChunk;
    result.tilesPerBatch = tilesPerBatch;
    result.totalTiles = totalTiles;
    result.tilingKey = RMS_NORM_CHANNELS_TK_STREAM;
    result.cPerBlock = c;
    result.workspaceStride = n;
    return result;
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetOutputShape(0) == nullptr || context->GetInputDesc(0) == nullptr ||
        context->GetInputDesc(1) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& value = context->GetInputShape(0)->GetOriginShape();
    if (value.GetDimNum() != RMS_NORM_CHANNELS_RANK) {
        return ge::GRAPH_FAILED;
    }
    int64_t dims[RMS_NORM_CHANNELS_RANK];
    for (int64_t i = 0; i < RMS_NORM_CHANNELS_RANK; ++i) {
        dims[i] = value.GetDim(i);
        if (dims[i] < 0) {
            // The kernel needs the concrete tile sizes; dynamic shapes are
            // rejected instead of silently normalizing garbage extents.
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t b = dims[0];
    const int64_t c = dims[1];
    if (b == 0 || c == 0 || dims[2] == 0 || dims[3] == 0 || dims[4] == 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t n = dims[2] * dims[3] * dims[4];
    if (n <= 0) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(1)->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& gamma = context->GetInputShape(1)->GetOriginShape();
    if (gamma.GetDimNum() != 1 || gamma.GetDim(0) != c) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& out = context->GetOutputShape(0)->GetOriginShape();
    if (out.GetDimNum() != RMS_NORM_CHANNELS_RANK) {
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 0; i < RMS_NORM_CHANNELS_RANK; ++i) {
        if (out.GetDim(i) != dims[i]) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto* epsilon = context->GetAttrs()->GetAttrPointer<float>(0);
    if (epsilon == nullptr || !std::isfinite(*epsilon) || *epsilon < 0.0f) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const int64_t dtypeSize = dtype == ge::DT_FLOAT ? 4 : 2;
    const int64_t coreNum = std::max<int64_t>(1, platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    const TilingParams params = ComputeTiling(b, c, n, dtypeSize, coreNum, ubSize);

    optiling::RmsNormChannelsTilingData tiling;
    tiling.set_b(static_cast<uint64_t>(b));
    tiling.set_c(static_cast<uint64_t>(c));
    tiling.set_n(static_cast<uint64_t>(n));
    tiling.set_totalTiles(static_cast<uint64_t>(params.totalTiles));
    tiling.set_tilesPerBatch(static_cast<uint64_t>(params.tilesPerBatch));
    tiling.set_numBlocks(static_cast<uint64_t>(params.numBlocks));
    tiling.set_cPerBlock(static_cast<uint64_t>(params.cPerBlock));
    tiling.set_workspaceStride(static_cast<uint64_t>(params.workspaceStride));
    tiling.set_nTile(static_cast<uint32_t>(params.nTile));
    tiling.set_cChunk(static_cast<uint32_t>(params.cChunk));
    tiling.set_tilingKey(params.tilingKey);
    tiling.set_epsilon(*epsilon);

    context->SetBlockDim(static_cast<uint32_t>(params.numBlocks));
    if (params.tilingKey == RMS_NORM_CHANNELS_TK_SPLIT_C) {
        // The split path joins per-block partial sums with a cross-core barrier,
        // so every block must be resident at the same time (batch mode).
        if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    context->SetTilingKey(0);
    auto* buffer = context->GetRawTilingData();
    if (buffer == nullptr || buffer->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(buffer->GetData(), buffer->GetCapacity());
    buffer->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = platform.GetLibApiWorkSpaceSize() +
                   static_cast<size_t>(params.workspaceFloats) * sizeof(float);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(RmsNormChannels).Tiling(TilingFunc);
} // namespace optiling
