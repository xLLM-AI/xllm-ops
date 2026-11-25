/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_expert_selection_threshold_contract.h"

namespace ge {
namespace {

ge::graphStatus InferShape(gert::InferShapeContext* context) {
  if (context == nullptr ||
      context->GetInputShape(ops::ecr_threshold::kInputProb) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ops::ecr_threshold::Contract contract;
  if (!ops::ecr_threshold::BuildContract(
          context,
          *context->GetInputShape(ops::ecr_threshold::kInputProb),
          true,
          contract)) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < ops::ecr_threshold::kOutputCount; ++i) {
    gert::Shape* output = context->GetOutputShape(i);
    if (output == nullptr) {
      return ge::GRAPH_FAILED;
    }
    output->SetDimNum(2);
    output->SetDim(0, contract.e);
    output->SetDim(1, static_cast<int64_t>(contract.b));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < ops::ecr_threshold::kInputCount; ++i) {
    if (context->GetInputDataType(i) != ops::ecr_threshold::InputType(i)) {
      return ge::GRAPH_FAILED;
    }
  }
  for (size_t i = 0; i < ops::ecr_threshold::kOutputCount; ++i) {
    context->SetOutputDataType(i, ops::ecr_threshold::OutputType(i));
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_INFERSHAPE(EcrExpertSelectionThreshold)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
