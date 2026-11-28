/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_routing_normalize_compact_contract.h"

namespace ge {
namespace {

ge::graphStatus InferShape(gert::InferShapeContext* context) {
  if (context == nullptr ||
      context->GetInputShape(ops::ecr_compact::kInputProb) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  ops::ecr_compact::Contract contract;
  if (!ops::ecr_compact::BuildContract(
          context,
          *context->GetInputShape(ops::ecr_compact::kInputProb),
          true,
          contract)) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 1; i < ops::ecr_compact::kInputCount; ++i) {
    const gert::Shape* shape = context->GetInputShape(i);
    if (shape == nullptr) {
      return ge::GRAPH_FAILED;
    }
    const int64_t first = i < 3 ? contract.e : contract.s;
    if (!ops::ecr_compact::Matches(
            *shape, first, static_cast<int64_t>(contract.b), i >= 3, true)) {
      return ge::GRAPH_FAILED;
    }
  }
  for (size_t i = 0; i < ops::ecr_compact::kOutputCount; ++i) {
    gert::Shape* output = context->GetOutputShape(i);
    if (output == nullptr) {
      return ge::GRAPH_FAILED;
    }
    output->SetDimNum(2);
    output->SetDim(0, contract.e);
    output->SetDim(1, contract.capacity_c);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  if (context == nullptr) {
    return ge::GRAPH_FAILED;
  }
  for (size_t i = 0; i < ops::ecr_compact::kInputCount; ++i) {
    if (context->GetInputDataType(i) != ops::ecr_compact::InputType(i)) {
      return ge::GRAPH_FAILED;
    }
  }
  for (size_t i = 0; i < ops::ecr_compact::kOutputCount; ++i) {
    context->SetOutputDataType(i, ops::ecr_compact::OutputType(i));
  }
  return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_INFERSHAPE(EcrRoutingNormalizeCompact)
    .InferShape(InferShape)
    .InferDataType(InferDataType);
}  // namespace ge
