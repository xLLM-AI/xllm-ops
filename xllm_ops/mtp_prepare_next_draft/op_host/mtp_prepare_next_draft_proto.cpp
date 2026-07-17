/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_impl_registry.h"

namespace ge {
namespace {

void Set1D(gert::Shape* shape, int64_t dim0) {
  shape->SetDimNum(1);
  shape->SetDim(0, dim0);
}

}  // namespace

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* token_shape = context->GetInputShape(0);
  const gert::Shape* embedding_shape = context->GetInputShape(1);
  if (token_shape == nullptr || embedding_shape == nullptr ||
      token_shape->GetDimNum() != 2 || embedding_shape->GetDimNum() != 3) {
    return GRAPH_FAILED;
  }

  const int64_t batch = token_shape->GetDim(0);
  const int64_t doubled_batch = batch < 0 ? -1 : batch * 2;
  const int64_t hidden = embedding_shape->GetDim(2);

  Set1D(context->GetOutputShape(0), doubled_batch);
  gert::Shape* draft_embedding_shape = context->GetOutputShape(1);
  draft_embedding_shape->SetDimNum(2);
  draft_embedding_shape->SetDim(0, doubled_batch);
  draft_embedding_shape->SetDim(1, hidden);
  Set1D(context->GetOutputShape(2), doubled_batch);
  Set1D(context->GetOutputShape(3), batch);
  Set1D(context->GetOutputShape(4), doubled_batch);
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_INT32);
  context->SetOutputDataType(1, context->GetInputDataType(1));
  context->SetOutputDataType(2, ge::DT_INT32);
  context->SetOutputDataType(3, ge::DT_INT32);
  context->SetOutputDataType(4, ge::DT_INT32);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MtpPrepareNextDraft)
    .InferShape(InferShape)
    .InferDataType(InferDataType);

}  // namespace ge
