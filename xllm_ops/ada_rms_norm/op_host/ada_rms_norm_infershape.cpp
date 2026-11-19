/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* x = context->GetInputShape(0);
    if (x->GetDimNum() != 2 && x->GetDimNum() != 3 &&
        !(x->GetDimNum() == 1 && x->GetDim(0) == -2)) {
        return ge::GRAPH_FAILED;
    }
    *context->GetOutputShape(0) = *x;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDataType(0);
    if (dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AdaRmsNorm).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
