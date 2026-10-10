/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "register/op_impl_registry.h"
#include "../op_kernel/mega_kda_decode_tiling_data.h"

namespace ops {
static ge::graphStatus InferShapeMegaKdaDecode(gert::InferShapeContext* context)
{
    using namespace mega_kda;
    const auto* qkv = context->GetRequiredInputShape(kQkv);
    const auto* conv = context->GetRequiredInputShape(kConvStateIn);
    const auto* ssm = context->GetRequiredInputShape(kSsmStateIn);
    const auto* attrs = context->GetAttrs();
    if (!qkv || !conv || !ssm || !attrs || qkv->GetDimNum() != 2 ||
        conv->GetDimNum() != 3 || ssm->GetDimNum() != 4 ||
        !attrs->GetInt(kConvOutputSlots) || !attrs->GetInt(kSsmOutputSlots) ||
        *attrs->GetInt(kConvOutputSlots) < 1 || *attrs->GetInt(kSsmOutputSlots) < 1) {
        return ge::GRAPH_FAILED;
    }
    auto* out = context->GetOutputShape(0);
    auto* convOut = context->GetOutputShape(1);
    auto* ssmOut = context->GetOutputShape(2);
    if (!out || !convOut || !ssmOut) {
        return ge::GRAPH_FAILED;
    }
    *out = gert::Shape({qkv->GetDim(0), ssm->GetDim(1), kHeadDim});
    *convOut = *conv;
    convOut->SetDim(0, *attrs->GetInt(kConvOutputSlots));
    *ssmOut = *ssm;
    ssmOut->SetDim(0, *attrs->GetInt(kSsmOutputSlots));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeMegaKdaDecode(gert::InferDataTypeContext* context)
{
    if (context->SetOutputDataType(0, ge::DT_BF16) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(1, ge::DT_BF16) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(2, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MegaKdaDecode)
    .InferShape(InferShapeMegaKdaDecode).InferDataType(InferDataTypeMegaKdaDecode);
} // namespace ops
