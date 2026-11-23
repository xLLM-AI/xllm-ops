/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "t5_split_qkv_append_cache_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int64_t kMaxBatch = 64;
constexpr int64_t kMaxHeadNum = 128;
constexpr int64_t kMaxHistoryLen = 8192;
constexpr int64_t kMaxHeadDim = 256;
constexpr int64_t kMaxOutputElements = 1LL << 28;
constexpr int64_t kChunkElements = 32768;
constexpr int64_t kMaxBlockDim = 48;
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* qkvStorage = context->GetInputShape(0);
    const gert::StorageShape* pastKeyStorage = context->GetInputShape(1);
    const gert::StorageShape* pastValueStorage = context->GetInputShape(2);
    if (qkvStorage == nullptr || pastKeyStorage == nullptr || pastValueStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dataType = context->GetInputDesc(0)->GetDataType();
    if (dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDesc(1)->GetDataType() != dataType ||
        context->GetInputDesc(2)->GetDataType() != dataType) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& qkvShapeRef = qkvStorage->GetStorageShape();
    const gert::Shape& pastShapeRef = pastKeyStorage->GetStorageShape();
    if (qkvShapeRef.GetDimNum() != 3 || pastShapeRef.GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    if (pastValueStorage->GetStorageShape() != pastShapeRef) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = pastShapeRef.GetDim(0);
    const int64_t headNum = pastShapeRef.GetDim(1);
    const int64_t historyLen = pastShapeRef.GetDim(2);
    const int64_t headDim = pastShapeRef.GetDim(3);
    if (batch < 1 || batch > kMaxBatch || headNum < 1 || headNum > kMaxHeadNum ||
        historyLen < 0 || historyLen > kMaxHistoryLen || headDim < 1 || headDim > kMaxHeadDim) {
        return ge::GRAPH_FAILED;
    }
    if (qkvShapeRef.GetDim(0) != batch || qkvShapeRef.GetDim(1) != 1 ||
        qkvShapeRef.GetDim(2) != 3 * headNum * headDim) {
        return ge::GRAPH_FAILED;
    }
    if (batch * headNum * headDim * (2 * historyLen + 3) > kMaxOutputElements) {
        return ge::GRAPH_FAILED;
    }

    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t headNumAttr = *attrs->GetAttrPointer<int64_t>(0);
    const int64_t headDimAttr = *attrs->GetAttrPointer<int64_t>(1);
    if (headNumAttr != headNum || headDimAttr != headDim) {
        return ge::GRAPH_FAILED;
    }

    const int64_t qJobs = batch * ((headNum * headDim + kChunkElements - 1) / kChunkElements);
    const int64_t cacheJobs =
        batch * headNum * (((historyLen + 1) * headDim + kChunkElements - 1) / kChunkElements);
    const int64_t jobs = qJobs + 2 * cacheJobs;

    T5SplitQkvAppendCacheTilingData tiling;
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_headNum(static_cast<uint32_t>(headNum));
    tiling.set_historyLen(static_cast<uint32_t>(historyLen));
    tiling.set_headDim(static_cast<uint32_t>(headDim));

    context->SetBlockDim(static_cast<uint32_t>(std::min<int64_t>(kMaxBlockDim, jobs)));
    context->GetWorkspaceSizes(1)[0] = 0;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(T5SplitQkvAppendCache)
    .Tiling(TilingFunc);
} // namespace optiling
