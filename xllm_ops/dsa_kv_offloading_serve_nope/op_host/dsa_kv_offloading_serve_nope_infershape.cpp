/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "register/op_impl_registry.h"
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
  const auto* tokens = context->GetInputShape(4);
  if (tokens == nullptr || tokens->GetDimNum() != 3) return GRAPH_FAILED;
  if (context->GetInputShape(0) == nullptr ||
      context->GetOutputShape(0) == nullptr)
    return GRAPH_FAILED;
  if (context->GetInputShape(8) == nullptr ||
      context->GetOutputShape(1) == nullptr)
    return GRAPH_FAILED;
  if (context->GetInputShape(7) == nullptr ||
      context->GetOutputShape(2) == nullptr)
    return GRAPH_FAILED;
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  *context->GetOutputShape(1) = *context->GetInputShape(8);
  *context->GetOutputShape(2) = *context->GetInputShape(7);
  return GRAPH_SUCCESS;
}
static graphStatus InferType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  context->SetOutputDataType(1, context->GetInputDataType(8));
  context->SetOutputDataType(2, context->GetInputDataType(7));
  return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DsaKvOffloadingServeNope)
    .InferShape(InferShape)
    .InferDataType(InferType);
}  // namespace ge
