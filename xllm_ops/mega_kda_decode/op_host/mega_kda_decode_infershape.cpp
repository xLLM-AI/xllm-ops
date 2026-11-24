/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "../op_kernel/mega_kda_decode_tiling_data.h"
#include "register/op_impl_registry.h"

namespace ops {
static ge::graphStatus infer_shape_mega_kda_decode(
    gert::InferShapeContext* context) {
  using namespace mega_kda;
  const auto* qkv =
      context->GetRequiredInputShape(static_cast<uint32_t>(Input::QKV));
  const auto* conv = context->GetRequiredInputShape(
      static_cast<uint32_t>(Input::CONV_STATE_IN));
  const auto* ssm = context->GetRequiredInputShape(
      static_cast<uint32_t>(Input::SSM_STATE_IN));
  const auto* attrs = context->GetAttrs();
  if (!qkv || !conv || !ssm || !attrs || qkv->GetDimNum() != 2 ||
      conv->GetDimNum() != 3 || ssm->GetDimNum() != 4 ||
      !attrs->GetInt(static_cast<uint32_t>(Attr::CONV_OUTPUT_SLOTS)) ||
      !attrs->GetInt(static_cast<uint32_t>(Attr::SSM_OUTPUT_SLOTS)) ||
      *attrs->GetInt(static_cast<uint32_t>(Attr::CONV_OUTPUT_SLOTS)) < 1 ||
      *attrs->GetInt(static_cast<uint32_t>(Attr::SSM_OUTPUT_SLOTS)) < 1) {
    return ge::GRAPH_FAILED;
  }
  auto* out = context->GetOutputShape(0);
  auto* conv_out = context->GetOutputShape(1);
  auto* ssm_out = context->GetOutputShape(2);
  if (!out || !conv_out || !ssm_out) {
    return ge::GRAPH_FAILED;
  }
  *out = gert::Shape({qkv->GetDim(0), ssm->GetDim(1), kHeadDim});
  *conv_out = *conv;
  conv_out->SetDim(
      0, *attrs->GetInt(static_cast<uint32_t>(Attr::CONV_OUTPUT_SLOTS)));
  *ssm_out = *ssm;
  ssm_out->SetDim(
      0, *attrs->GetInt(static_cast<uint32_t>(Attr::SSM_OUTPUT_SLOTS)));
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus infer_data_type_mega_kda_decode(
    gert::InferDataTypeContext* context) {
  if (context->SetOutputDataType(0, ge::DT_BF16) != ge::GRAPH_SUCCESS ||
      context->SetOutputDataType(1, ge::DT_BF16) != ge::GRAPH_SUCCESS ||
      context->SetOutputDataType(2, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(MegaKdaDecode)
    .InferShape(infer_shape_mega_kda_decode)
    .InferDataType(infer_data_type_mega_kda_decode);
}  // namespace ops
