/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// Ascend950 ACLNN adapter for the validated native CV direct-connect implementation.
#include "../cross_attention_tiling.h"
#include "cross_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdio>
#include <limits>

namespace optiling {
namespace {
bool FitsU32(int64_t value)
{
    return value > 0 && static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max();
}
} // namespace

ge::graphStatus TilingCrossAttentionAscend950(gert::TilingContext* context)
{
    const auto* query = context->GetInputShape(0);
    const auto* key = context->GetInputShape(1);
    const auto* value = context->GetInputShape(2);
    const auto* table = context->GetInputShape(3);
    const auto* lens = context->GetInputShape(4);
    if (query == nullptr || key == nullptr || value == nullptr || table == nullptr || lens == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& q = query->GetStorageShape();
    const auto& k = key->GetStorageShape();
    const auto& v = value->GetStorageShape();
    const auto& bt = table->GetStorageShape();
    const auto& lengths = lens->GetStorageShape();
    if (q.GetDimNum() != 4 || k.GetDimNum() != 4 || v.GetDimNum() != 4 ||
        bt.GetDimNum() != 2 || lengths.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (!FitsU32(q.GetDim(i)) || !FitsU32(k.GetDim(i)) || k.GetDim(i) != v.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    if (q.GetDim(0) != bt.GetDim(0) || q.GetDim(0) != lengths.GetDim(0) ||
        q.GetDim(3) != k.GetDim(3) || !FitsU32(bt.GetDim(1))) {
        return ge::GRAPH_FAILED;
    }
    const auto queryType = context->GetInputDesc(0)->GetDataType();
    if ((queryType != ge::DT_FLOAT16 && queryType != ge::DT_BF16) ||
        queryType != context->GetInputDesc(1)->GetDataType() ||
        queryType != context->GetInputDesc(2)->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    const Xllm::CrossAttentionShape shape{
        static_cast<uint32_t>(q.GetDim(0)), static_cast<uint32_t>(q.GetDim(1)),
        static_cast<uint32_t>(q.GetDim(2)), static_cast<uint32_t>(k.GetDim(2)),
        static_cast<uint32_t>(q.GetDim(3)), static_cast<uint32_t>(k.GetDim(1)),
        static_cast<uint32_t>(bt.GetDim(1)),
        queryType == ge::DT_BF16 ? CA_DTYPE_BF16 : CA_DTYPE_FP16};
    float scaleValue = 0.0f;
    if (context->GetAttrs() != nullptr) {
        const auto* scale = context->GetAttrs()->GetAttrPointer<float>(0);
        if (scale != nullptr) {
            scaleValue = *scale;
        }
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    Xllm::CrossAttentionTilingData native{};
    if (Xllm::ComputeCrossAttentionTiling(shape, scaleValue, platform.GetCoreNumAic(), native) != 0) {
        return ge::GRAPH_FAILED;
    }
    CrossAttentionArch35TilingData serialized;
    serialized.set_batch(native.batch);
    serialized.set_qHeads(native.qHeads);
    serialized.set_kvHeads(native.kvHeads);
    serialized.set_groupSize(native.groupSize);
    serialized.set_headDim(native.headDim);
    serialized.set_qSeqlen(native.qSeqlen);
    serialized.set_blockSize(native.blockSize);
    serialized.set_maxNumBlocksPerBatch(native.maxNumBlocksPerBatch);
    serialized.set_scaleValue(native.scaleValue);
    serialized.set_qRowsPerTile(native.qRowsPerTile);
    serialized.set_qRowTileNum(native.qRowTileNum);
    serialized.set_rowsPerTask(native.rowsPerTask);
    serialized.set_kvTile(native.kvTile);
    serialized.set_kSeg(native.kSeg);
    serialized.set_mergeSegBlocks(native.mergeSegBlocks);
    serialized.set_totalTaskNum(native.totalTaskNum);
    serialized.set_usedCoreNum(native.usedCoreNum);
    serialized.set_dtype(native.dtype);
    serialized.set_mergeSlotBytes(native.mergeSlotBytes);
    serialized.set_sStageOff(native.sStageOff);
    serialized.set_pStageOff(native.pStageOff);
    serialized.set_oStageOff(native.oStageOff);
    serialized.set_userWsBytes(native.userWsBytes);
    serialized.set_tilingVersion(native.tilingVersion);
    serialized.set_reserved0(native.reserved0);
    auto* raw = context->GetRawTilingData();
    if (serialized.GetDataSize() != sizeof(native) || raw->GetCapacity() < serialized.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    serialized.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(serialized.GetDataSize());
    context->SetBlockDim(native.usedCoreNum);
    context->SetTilingKey(3510);
    auto* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) + native.userWsBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
