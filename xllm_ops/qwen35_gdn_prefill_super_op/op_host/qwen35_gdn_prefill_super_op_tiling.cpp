/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "qwen35_gdn_prefill_super_op_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kChunkSize = 128;
constexpr uint32_t kValueHeads = 24;
constexpr uint32_t kHalfBytes = 2;
constexpr uint32_t kVectorCoreCount = 40;
constexpr uint32_t kConvDimTiles = 2;

enum InputIndex {
    MIXED_QKV_INDEX = 0,
};

enum AttrIndex {
    NUM_MATRICES_ATTR = 0,
    CONV_STATE_INDEX_ATTR,
    SSM_STATE_INDEX_ATTR,
};

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t CalcUserWorkspaceBytes(uint32_t blockDim)
{
    const uint64_t tileBytes = static_cast<uint64_t>(kChunkSize) * kChunkSize * kHalfBytes;
    constexpr uint64_t kWorkspaceTileCount = 11;
    return static_cast<uint64_t>(blockDim) * kWorkspaceTileCount * tileBytes;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = std::max<uint32_t>(platform.GetCoreNumAic(), 1);
    const gert::StorageShape *mixedShape = context->GetInputShape(MIXED_QKV_INDEX);
    if (mixedShape == nullptr || mixedShape->GetOriginShape().GetDimNum() != 2 ||
        mixedShape->GetOriginShape().GetDim(1) != 5120) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalTokens64 = mixedShape->GetOriginShape().GetDim(0);
    if (totalTokens64 <= 0 || totalTokens64 > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalTokens = static_cast<uint32_t>(totalTokens64);
    const uint32_t tokenCoreBudget = kVectorCoreCount / kConvDimTiles;
    const uint32_t tokenBlockSize = CeilDiv(totalTokens, tokenCoreBudget);
    const uint32_t tokenBlockCount = CeilDiv(totalTokens, tokenBlockSize);

    auto attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR) == nullptr ||
        attrs->GetAttrPointer<int64_t>(CONV_STATE_INDEX_ATTR) == nullptr ||
        attrs->GetAttrPointer<int64_t>(SSM_STATE_INDEX_ATTR) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numMatricesAttr = *attrs->GetAttrPointer<int64_t>(NUM_MATRICES_ATTR);
    const int64_t convStateIndex = *attrs->GetAttrPointer<int64_t>(CONV_STATE_INDEX_ATTR);
    const int64_t ssmStateIndex = *attrs->GetAttrPointer<int64_t>(SSM_STATE_INDEX_ATTR);
    if (numMatricesAttr <= 0 || convStateIndex < 0 || ssmStateIndex < 0) {
        return ge::GRAPH_FAILED;
    }

    Qwen35GdnPrefillSuperOpTilingData tiling;
    tiling.set_block_dim(blockDim);
    tiling.set_num_matrices(static_cast<uint32_t>(numMatricesAttr));
    tiling.set_num_heads(kValueHeads);
    tiling.set_num_key_heads(8);
    tiling.set_token_block_size(tokenBlockSize);
    tiling.set_token_block_count(tokenBlockCount);
    tiling.set_conv_state_index(convStateIndex);
    tiling.set_ssm_state_index(ssmStateIndex);
    tiling.set_batch_size(1);
    tiling.set_seq_len(totalTokens);
    tiling.set_total_tokens(totalTokens);

    context->SetBlockDim(blockDim);
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = CalcUserWorkspaceBytes(blockDim) + platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

struct Qwen35GdnPrefillSuperOpCompileInfo {};

static ge::graphStatus TilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Qwen35GdnPrefillSuperOp)
    .Tiling(TilingFunc)
    .TilingParse<Qwen35GdnPrefillSuperOpCompileInfo>(TilingParse);
}  // namespace optiling
