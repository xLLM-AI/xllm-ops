/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <algorithm>
#include <cstring>
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "mega_kda_prefill_policy.h"

namespace optiling {
namespace {
bool Match(const gert::Shape& shape, std::initializer_list<int64_t> dims)
{
    if (shape.GetDimNum() != dims.size()) return false;
    size_t i = 0;
    for (auto dim : dims) if (shape.GetDim(i++) != dim) return false;
    return true;
}
}
static ge::graphStatus MegaKdaPrefillTiling(gert::TilingContext* context)
{
    using namespace kda_prefill;
    const auto* attrs = context->GetAttrs();
    if (!attrs || !context->GetPlatformInfo()) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 3; ++i) if (!attrs->GetInt(i)) return ge::GRAPH_FAILED;
    auto shape = [&](size_t i) {
        return i == kConvBias ? context->GetOptionalInputShape(i) : context->GetRequiredInputShape(i);
    };
    for (size_t i = 0; i < kInputCount; ++i) {
        if (i != kConvBias && !shape(i)) return ge::GRAPH_FAILED;
    }
    const auto& qkv = shape(kQkv)->GetStorageShape();
    const auto& conv = shape(kConvStateIn)->GetStorageShape();
    const auto& ssm = shape(kSsmStateIn)->GetStorageShape();
    const auto& cu = shape(kCuSeqlens)->GetStorageShape();
    if (qkv.GetDimNum() != 2 || conv.GetDimNum() != 3 || ssm.GetDimNum() != 4 ||
        cu.GetDimNum() != 1) return ge::GRAPH_FAILED;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    TilingData t{cu.GetDim(0) - 1, qkv.GetDim(0), ssm.GetDim(1), conv.GetDim(0),
        *attrs->GetInt(kConvOutputSlots), ssm.GetDim(0), *attrs->GetInt(kSsmOutputSlots),
        platform.GetCoreNumAic(), static_cast<uint64_t>(*attrs->GetInt(kFftsAddr))};
    if (!ValidTiling(t)) return ge::GRAPH_FAILED;
    const auto c = t.heads * 384;
    auto input = [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* desc = i == kConvBias ? context->GetOptionalInputDesc(i) : context->GetRequiredInputDesc(i);
        return shape(i) && desc && desc->GetDataType() == dtype && Match(shape(i)->GetStorageShape(), dims);
    };
    if (!input(kQkv, {t.tokens, c}, ge::DT_BF16) ||
        !input(kGate, {t.tokens, t.heads, 128}, ge::DT_BF16) ||
        !input(kBeta, {t.tokens, t.heads}, ge::DT_FLOAT) ||
        !input(kALog, {t.heads}, ge::DT_FLOAT) ||
        !input(kGateBias, {t.heads, 128}, ge::DT_FLOAT) ||
        !input(kConvWeight, {4, c}, ge::DT_BF16) ||
        (shape(kConvBias) && !input(kConvBias, {c}, ge::DT_BF16)) ||
        !input(kConvStateIn, {t.conv_input_slots, 3, c}, ge::DT_BF16) ||
        !input(kSsmStateIn, {t.ssm_input_slots, t.heads, 128, 128}, ge::DT_FLOAT) ||
        !input(kCuSeqlens, {t.batch + 1}, ge::DT_INT32)) return ge::GRAPH_FAILED;
    for (auto i : {kConvReadIndices, kConvWriteIndices, kSsmReadIndices, kSsmWriteIndices}) {
        if (!input(i, {t.batch}, ge::DT_INT32)) return ge::GRAPH_FAILED;
    }
    auto output = [&](size_t i, std::initializer_list<int64_t> dims, ge::DataType dtype) {
        const auto* s = context->GetOutputShape(i);
        const auto* d = context->GetOutputDesc(i);
        return s && d && d->GetDataType() == dtype && Match(s->GetStorageShape(), dims);
    };
    if (!output(0, {t.tokens, t.heads, 128}, ge::DT_BF16) ||
        !output(1, {t.conv_output_slots, 3, c}, ge::DT_BF16) ||
        !output(2, {t.ssm_output_slots, t.heads, 128, 128}, ge::DT_FLOAT)) return ge::GRAPH_FAILED;
    auto* raw = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!raw || !raw->GetData() || raw->GetCapacity() < sizeof(t) || !workspace) return ge::GRAPH_FAILED;
    std::memcpy(raw->GetData(), &t, sizeof(t));
    raw->SetDataSize(sizeof(t));
    workspace[0] = Workspace(t).bytes + platform.GetLibApiWorkSpaceSize();
    context->SetTilingKey(1);
    context->SetBlockDim(t.blocks);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(MegaKdaPrefill).Tiling(MegaKdaPrefillTiling);
} // namespace optiling
