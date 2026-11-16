/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <algorithm>
#include <cstring>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "mega_kda_decode_policy.h"

namespace optiling {
namespace {
bool Optional(size_t index)
{
    return index == mega_kda::kConvBias || index == mega_kda::kAcceptedTokens;
}

const gert::StorageShape* InputShape(gert::TilingContext* context, size_t index)
{
    return Optional(index) ? context->GetOptionalInputShape(index) : context->GetRequiredInputShape(index);
}

bool Match(const gert::Shape& shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) {
        return false;
    }
    size_t i = 0;
    for (auto dim : dims) {
        if (shape.GetDim(i++) != dim) {
            return false;
        }
    }
    return true;
}
} // namespace

static ge::graphStatus MegaKdaDecodeTiling(gert::TilingContext* context)
{
    using namespace mega_kda;
    const auto* attrs = context->GetAttrs();
    if (!attrs || !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (!attrs->GetInt(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    for (size_t i = 0; i < kInputCount; ++i) {
        if (!Optional(i) && !InputShape(context, i)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto& qkv = InputShape(context, kQkv)->GetStorageShape();
    const auto& conv = InputShape(context, kConvStateIn)->GetStorageShape();
    const auto& ssm = InputShape(context, kSsmStateIn)->GetStorageShape();
    const auto& offsets = InputShape(context, kCuSeqlens)->GetStorageShape();
    if (qkv.GetDimNum() != 2 || conv.GetDimNum() != 3 || ssm.GetDimNum() != 4 ||
        offsets.GetDimNum() != 1 || offsets.GetDim(0) < 2) {
        return ge::GRAPH_FAILED;
    }
    TilingData t{offsets.GetDim(0) - 1, qkv.GetDim(0), ssm.GetDim(1),
        *attrs->GetInt(kMaxQueryTokens), conv.GetDim(1), conv.GetDim(0),
        *attrs->GetInt(kConvOutputSlots), ssm.GetDim(0), *attrs->GetInt(kSsmOutputSlots)};
    const auto key = SelectKey(*attrs->GetInt(kMode), InputShape(context, kAcceptedTokens) != nullptr, t);
    if (!key) {
        return ge::GRAPH_FAILED;
    }
    const auto c = 3 * t.heads * kHeadDim;
    auto input = [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* shape = InputShape(context, i);
        const auto* desc = Optional(i) ? context->GetOptionalInputDesc(i) : context->GetRequiredInputDesc(i);
        return shape && desc && desc->GetDataType() == dtype &&
            Match(shape->GetStorageShape(), dims);
    };
    if (!input(kQkv, {t.tokens, c}, ge::DT_BF16) ||
        !input(kGate, {t.tokens, t.heads, 128}, ge::DT_BF16) ||
        !input(kBeta, {t.tokens, t.heads}, ge::DT_BF16) ||
        !input(kALog, {t.heads}, ge::DT_FLOAT) ||
        !input(kGateBias, {t.heads, 128}, ge::DT_FLOAT) ||
        !input(kConvWeight, {4, c}, ge::DT_BF16) ||
        !input(kConvStateIn, {t.conv_input_slots, t.conv_history, c}, ge::DT_BF16) ||
        !input(kSsmStateIn, {t.ssm_input_slots, t.heads, 128, 128}, ge::DT_FLOAT) ||
        !input(kCuSeqlens, {t.batch + 1}, ge::DT_INT32) ||
        !input(kConvReadIndices, {t.batch}, ge::DT_INT32) ||
        !input(kConvWriteIndices, {t.batch}, ge::DT_INT32) ||
        !input(kSsmReadIndices, {t.batch, t.max_query_tokens}, ge::DT_INT32) ||
        !input(kSsmWriteIndices, {t.batch, t.max_query_tokens}, ge::DT_INT32) ||
        (InputShape(context, kConvBias) && !input(kConvBias, {c}, ge::DT_BF16)) ||
        (InputShape(context, kAcceptedTokens) && !input(kAcceptedTokens, {t.batch}, ge::DT_INT32))) {
        return ge::GRAPH_FAILED;
    }
    auto output = [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* shape = context->GetOutputShape(i);
        const auto* desc = context->GetOutputDesc(i);
        return shape && desc && desc->GetDataType() == dtype && Match(shape->GetStorageShape(), dims);
    };
    if (!output(0, {t.tokens, t.heads, 128}, ge::DT_BF16) ||
        !output(1, {t.conv_output_slots, t.conv_history, c}, ge::DT_BF16) ||
        !output(2, {t.ssm_output_slots, t.heads, 128, 128}, ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const auto cores = platform.GetCoreNumAiv();
    const uint32_t blocks = std::min<int64_t>(cores, t.batch * t.heads * 8);
    auto* raw = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!blocks || !raw || !raw->GetData() || raw->GetCapacity() < sizeof(t) || !workspace) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &t, sizeof(t));
    raw->SetDataSize(sizeof(t));
    workspace[0] = 0;
    context->SetTilingKey(key);
    context->SetBlockDim(blocks);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MegaKdaDecode).Tiling(MegaKdaDecodeTiling);
} // namespace optiling
