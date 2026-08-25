/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

static ge::graphStatus InferShapeMegaGdnDraftDecode(
    gert::InferShapeContext* context) {
  const gert::Shape* qkv_shape = context->GetInputShape(0);
  const gert::Shape* z_shape = context->GetInputShape(1);
  const gert::Shape* conv_state_shape = context->GetInputShape(5);
  const gert::Shape* ssm_state_shape = context->GetInputShape(8);
  OP_CHECK_NULL_WITH_CONTEXT(context, qkv_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, z_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, conv_state_shape);
  OP_CHECK_NULL_WITH_CONTEXT(context, ssm_state_shape);

  *context->GetOutputShape(0) = *qkv_shape;
  *context->GetOutputShape(1) = *conv_state_shape;
  *context->GetOutputShape(2) = *ssm_state_shape;
  *context->GetOutputShape(3) = *z_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MegaGdnDraftDecode)
    .InferShape(InferShapeMegaGdnDraftDecode);

}  // namespace ops
