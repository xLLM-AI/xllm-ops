/** Host side CrossAttention tiling (plain C++, no device code). */
#include "cross_attention_tiling.h"

#include <cmath>
#include <cstdio>

namespace Xllm {
namespace {
constexpr uint32_t CA_HEAD_DIM_FIXED = 128;
constexpr uint32_t CA_KV_TILE = 128;
constexpr uint32_t CA_BLOCK_CUBE = 16;
constexpr uint32_t CA_MAX_R = 64;
constexpr uint32_t CA_MAX_CORE = 32;

inline uint64_t Align64(uint64_t v)
{
    return (v + 63ull) & ~63ull;
}

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1u) / b;
}

inline uint32_t ClampU32(uint32_t v, uint32_t lo, uint32_t hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}
} // namespace

int32_t ComputeCrossAttentionTiling(const CrossAttentionShape& shape, float scaleValue, uint32_t aicCoreNum,
                                    CrossAttentionTilingData& out)
{
    if (shape.headDim != CA_HEAD_DIM_FIXED) {
        printf("[CrossAttention] unsupported headDim=%u (only 128)\n", shape.headDim);
        return 1;
    }
    if (shape.blockSize != CA_KV_TILE) {
        printf("[CrossAttention] unsupported blockSize=%u (must equal kvTile=128)\n", shape.blockSize);
        return 2;
    }
    if (shape.kvHeads == 0 || shape.qHeads == 0 || shape.qHeads % shape.kvHeads != 0) {
        printf("[CrossAttention] qHeads=%u must be a multiple of kvHeads=%u\n", shape.qHeads, shape.kvHeads);
        return 3;
    }
    const uint32_t groupSize = shape.qHeads / shape.kvHeads;
    if (groupSize > CA_BLOCK_CUBE) {
        printf("[CrossAttention] groupSize=%u exceeds M fractal limit %u\n", groupSize, CA_BLOCK_CUBE);
        return 4;
    }
    if (shape.qSeqlen == 0 || shape.batch == 0) {
        printf("[CrossAttention] empty problem: batch=%u qSeqlen=%u\n", shape.batch, shape.qSeqlen);
        return 5;
    }
    if (shape.maxNumBlocksPerBatch == 0) {
        printf("[CrossAttention] maxNumBlocksPerBatch must be >= 1\n");
        return 6;
    }

    // q rows per task: when groupSize divides the 16-row fractal, fill all CA_MAX_R rows so one K/V panel
    // is reused by several 16-row fractals; otherwise fall back to a single fractal row group.
    const uint32_t qRowsPerTile = (CA_BLOCK_CUBE % groupSize == 0u)
                                      ? (CA_BLOCK_CUBE / groupSize) * (CA_MAX_R / CA_BLOCK_CUBE)
                                      : (CA_BLOCK_CUBE / groupSize);
    const uint32_t qRowTileNum = CeilDiv(shape.qSeqlen, qRowsPerTile);
    const uint32_t rowsPerTask = qRowsPerTile * groupSize;

    const uint32_t baseTaskNum = shape.batch * shape.kvHeads * qRowTileNum;
    uint32_t coreNum = aicCoreNum == 0 ? 32u : aicCoreNum;
    if (coreNum > CA_MAX_CORE) {
        coreNum = CA_MAX_CORE;
    }
    const uint32_t kSegMax = shape.maxNumBlocksPerBatch < 4u ? shape.maxNumBlocksPerBatch : 4u;
    const uint32_t kSeg = ClampU32(coreNum / baseTaskNum, 1u, kSegMax);
    const uint32_t totalTaskNum = baseTaskNum * kSeg;
    const uint32_t usedCoreNum = totalTaskNum < coreNum ? totalTaskNum : coreNum;

    out.batch = shape.batch;
    out.qHeads = shape.qHeads;
    out.kvHeads = shape.kvHeads;
    out.groupSize = groupSize;
    out.headDim = shape.headDim;
    out.qSeqlen = shape.qSeqlen;
    out.blockSize = shape.blockSize;
    out.maxNumBlocksPerBatch = shape.maxNumBlocksPerBatch;
    out.scaleValue = (scaleValue > 0.0f) ? scaleValue : static_cast<float>(1.0 / std::sqrt(static_cast<double>(shape.headDim)));
    out.qRowsPerTile = qRowsPerTile;
    out.qRowTileNum = qRowTileNum;
    out.rowsPerTask = rowsPerTask;
    out.kvTile = CA_KV_TILE;
    out.kSeg = kSeg;
    out.mergeSegBlocks = CeilDiv(shape.maxNumBlocksPerBatch, kSeg);
    out.totalTaskNum = totalTaskNum;
    out.usedCoreNum = usedCoreNum;
    out.dtype = shape.dtype;

    const uint64_t slotElems = static_cast<uint64_t>(rowsPerTask) * CA_HEAD_DIM_FIXED;
    // The m/l area is stored per AIV sub: each sub writes its own rows, a shared block would be overwritten
    out.mergeSlotBytes = (kSeg > 1u) ? Align64((slotElems + 4ull * CA_MAX_R) * sizeof(float)) : 0ull;
    // The CV direct-connect path needs no GM staging: S goes L0C->UB, O accumulates in L0C and lands in UB
    // once per task, P is written by AIV from UB into L1.A1. The workspace therefore only holds the merge
    // slots used when kSeg > 1; sStageOff / oStageOff / pStageOff stay as compatible fields the kernel
    // never reads.
    const uint64_t mergeOff = 0ull;
    const uint64_t mergeTotal = (kSeg > 1u) ? (static_cast<uint64_t>(totalTaskNum) * out.mergeSlotBytes) : 0ull;
    out.sStageOff = Align64(mergeOff + mergeTotal);
    out.oStageOff = out.sStageOff;
    out.pStageOff = out.sStageOff;
    out.userWsBytes = out.pStageOff + 64ull;
    out.tilingVersion = CA_TILING_VERSION;
    out.reserved0 = 0;
    return 0;
}

} // namespace Xllm
