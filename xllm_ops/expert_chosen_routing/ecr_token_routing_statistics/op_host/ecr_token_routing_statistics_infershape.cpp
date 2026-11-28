/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_token_routing_statistics_contract.h"

namespace ge {
namespace {

ge::graphStatus InferShape(gert::InferShapeContext* context) {
  if (context == nullptr ||
      context->GetInputShape(ops::ecr_statistics::kInputProb) == nullptr ||
      context->GetInputShape(ops::ecr_statistics::kInputTauVal) == nullptr ||
      context->GetInputShape(ops::ecr_statistics::kInputTauIdx) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ops::ecr_statistics::Contract contract;
  if (!ops::ecr_statistics::BuildContract(
          context,
          *context->GetInputShape(ops::ecr_statistics::kInputProb),
          context->GetInputShape(ops::ecr_statistics::kInputTauVal),
          context->GetInputShape(ops::ecr_statistics::kInputTauIdx),
          true,
          contract)) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < ops::ecr_statistics::kOutputCount; ++i) {
    gert::Shape* output = context->GetOutputShape(i);
    if (output == nullptr) {
      return ge::GRAPH_FAILED;
    }
    output->SetDimNum(1);
    output->SetDim(0, contract.s);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < ops::ecr_statistics::kInputCount; ++i) {
    if (context->GetInputDataType(i) != ops::ecr_statistics::InputType(i)) {
      return ge::GRAPH_FAILED;
    }
  }
  for (size_t i = 0; i < ops::ecr_statistics::kOutputCount; ++i) {
    context->SetOutputDataType(i, ops::ecr_statistics::OutputType(i));
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_INFERSHAPE(EcrTokenRoutingStatistics)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
