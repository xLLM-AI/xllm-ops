/**
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file sigmoid_mul_tiling.cpp
 * \brief out = sigmoid(x1) * x2. Two independent tilings selected by shape:
 *        key 1000 - x1/x2 same shape, flat elementwise stream;
 *        key 1001 - x1 is [M, 1] against x2 [M, N], one row of x2 per vector tile.
 */
#include "sigmoid_mul_tiling.h"

#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr int64_t kElemsAlign = 256;      // 512 B for bf16: keeps every DMA 32 B aligned
constexpr int64_t kMaxTileElems = 8192;   // see UB budget below
constexpr int64_t kMinTileElems = 2048;
// Small shapes stay dispatch-bound (the kernel finishes in well under a microsecond),
// so fanning them out over every AIV only pays per-core prologue and UB footprint.
// Keep a meaningful amount of work per core; large shapes still reach the full AIV count.
constexpr int64_t kMinElemsPerCore = 4096;
constexpr uint64_t kUbReserveBytes = 16 * 1024;
// x1/x2 in, out (double buffered bf16) + two fp32 scratch buffers:
// 2 * 2 * 2 + 2 * 2 + 2 * 4 = 20 bytes per element.
constexpr int64_t kBytesPerElem = 20;

constexpr uint32_t kTilingKeyFlat = 1000;
constexpr uint32_t kTilingKeyBroadcast = 1001;

int64_t UbTileElems(uint64_t ubSize)
{
    int64_t tileElems = kMaxTileElems;
    if (ubSize > kUbReserveBytes) {
        int64_t fitElems = (static_cast<int64_t>(ubSize) - kUbReserveBytes) / kBytesPerElem;
        fitElems = fitElems / kElemsAlign * kElemsAlign;
        if (fitElems < tileElems) {
            tileElems = fitElems;
        }
    }
    if (tileElems < kMinTileElems) {
        tileElems = kMinTileElems;
    }
    return tileElems;
}

} // namespace

namespace optiling {
static void SaveTiling(gert::TilingContext* context, SigmoidMulTilingData& tiling,
                       uint32_t blockDim, uint32_t tilingKey)
{
    context->SetBlockDim(blockDim);
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
}


// x1 and x2 share a shape: one flat M * N stream, no row structure to carry.
static void TilingFlat(gert::TilingContext* context, int64_t total, uint32_t aivNum, int64_t ubTile)
{
    int64_t blockDim = 1;
    if (total > 0) {
        blockDim = (total + kMinElemsPerCore - 1) / kMinElemsPerCore;
        if (blockDim > static_cast<int64_t>(aivNum)) {
            blockDim = static_cast<int64_t>(aivNum);
        }
    }
    int64_t elemsPerCore = 0;
    if (total > 0) {
        elemsPerCore = (total + blockDim - 1) / blockDim;
        elemsPerCore = (elemsPerCore + kElemsAlign - 1) / kElemsAlign * kElemsAlign;
    }

    // Never reserve UB for more elements than a core actually owns: a core with a
    // single tile would otherwise allocate the full 8K-element working set for a few
    // hundred elements and push that partial tile down the padded-copy path.
    int64_t tileElems = ubTile;
    if (elemsPerCore > 0 && elemsPerCore < tileElems) {
        tileElems = elemsPerCore;
    }
    tileElems = tileElems / kElemsAlign * kElemsAlign;
    if (tileElems < kElemsAlign) {
        tileElems = kElemsAlign;
    }

    SigmoidMulTilingData tiling;
    tiling.set_totalElements(total);
    tiling.set_elemsPerCore(elemsPerCore);
    tiling.set_tileElems(static_cast<int32_t>(tileElems));
    tiling.set_broadcast(0);
    tiling.set_rows(0);
    tiling.set_cols(0);
    tiling.set_rowsPerCore(0);
    SaveTiling(context, tiling, static_cast<uint32_t>(blockDim), kTilingKeyFlat);
}

// x1 is [M, 1]: a core owns whole rows, so each of its N-element rows is tiled on its
// own and the per-row sigmoid scalar is computed once and broadcast over the row.
static void TilingBroadcast(gert::TilingContext* context, int64_t rows, int64_t cols,
                            uint32_t aivNum, int64_t ubTile)
{
    int64_t total = rows * cols;
    int64_t blockDim = (total + kMinElemsPerCore - 1) / kMinElemsPerCore;
    if (blockDim > static_cast<int64_t>(aivNum)) {
        blockDim = static_cast<int64_t>(aivNum);
    }
    if (blockDim > rows) {
        blockDim = rows; // every core needs at least one row
    }
    if (blockDim < 1) {
        blockDim = 1;
    }
    int64_t rowsPerCore = (rows + blockDim - 1) / blockDim;

    int64_t tileElems = ubTile;
    if (cols < tileElems) {
        tileElems = cols;
    }
    tileElems = tileElems / kElemsAlign * kElemsAlign;
    if (tileElems < kElemsAlign) {
        tileElems = kElemsAlign;
    }

    SigmoidMulTilingData tiling;
    tiling.set_totalElements(total);
    tiling.set_elemsPerCore(0);
    tiling.set_tileElems(static_cast<int32_t>(tileElems));
    tiling.set_broadcast(1);
    tiling.set_rows(rows);
    tiling.set_cols(cols);
    tiling.set_rowsPerCore(rowsPerCore);
    SaveTiling(context, tiling, static_cast<uint32_t>(blockDim), kTilingKeyBroadcast);
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* x1Shape = context->GetInputShape(0);
    const gert::StorageShape* x2Shape = context->GetInputShape(1);
    if (x1Shape == nullptr || x2Shape == nullptr) {
        OP_LOGE(context->GetNodeName(), "SigmoidMul: missing input shape");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(0)->GetDataType() != ge::DT_BF16) {
        OP_LOGE(context->GetNodeName(), "SigmoidMul: only bf16 is supported");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& s1 = x1Shape->GetStorageShape();
    const gert::Shape& s2 = x2Shape->GetStorageShape();
    int64_t total1 = s1.GetShapeSize();
    int64_t total2 = s2.GetShapeSize();
    if (total1 < 0 || total2 < 0) {
        OP_LOGE(context->GetNodeName(), "SigmoidMul: invalid element count");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = 1;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t ubTile = UbTileElems(ubSize);

    if (total1 == total2) {
        TilingFlat(context, total1, aivNum, ubTile);
        return ge::GRAPH_SUCCESS;
    }

    // Only the [M, 1] x [M, N] form is broadcast; anything else is a shape mismatch.
    if (s1.GetDimNum() == 2 && s2.GetDimNum() == 2 && s1.GetDim(1) == 1 && s2.GetDim(1) > 1 &&
        s1.GetDim(0) == s2.GetDim(0)) {
        TilingBroadcast(context, s2.GetDim(0), s2.GetDim(1), aivNum, ubTile);
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE(context->GetNodeName(),
            "SigmoidMul: x1 [%ld, %ld] and x2 [%ld, %ld] are neither equal-shaped nor a "
            "[M, 1] x [M, N] broadcast",
            s1.GetDimNum() == 2 ? s1.GetDim(0) : -1, s1.GetDimNum() == 2 ? s1.GetDim(1) : -1,
            s2.GetDimNum() == 2 ? s2.GetDim(0) : -1, s2.GetDimNum() == 2 ? s2.GetDim(1) : -1);
    return ge::GRAPH_FAILED;
}

IMPL_OP_OPTILING(SigmoidMul).Tiling(TilingFunc);
} // namespace optiling
