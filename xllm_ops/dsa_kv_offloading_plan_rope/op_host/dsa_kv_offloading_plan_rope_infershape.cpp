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
  const auto* tokens = context->GetInputShape(1);
  if (tokens == nullptr || tokens->GetDimNum() != 3) return GRAPH_FAILED;
  if (context->GetInputShape(2) == nullptr ||
      context->GetOutputShape(0) == nullptr)
    return GRAPH_FAILED;
  if (context->GetInputShape(3) == nullptr ||
      context->GetOutputShape(1) == nullptr)
    return GRAPH_FAILED;
  if (context->GetInputShape(4) == nullptr ||
      context->GetOutputShape(2) == nullptr)
    return GRAPH_FAILED;
  if (context->GetOutputShape(3) == nullptr) return GRAPH_FAILED;
  if (context->GetOutputShape(6) == nullptr) return GRAPH_FAILED;
  if (context->GetOutputShape(7) == nullptr) return GRAPH_FAILED;
  if (context->GetOutputShape(4) == nullptr ||
      context->GetOutputShape(5) == nullptr)
    return GRAPH_FAILED;
  const int64_t rows = tokens->GetDim(0);
  *context->GetOutputShape(0) = *context->GetInputShape(2);
  *context->GetOutputShape(1) = *context->GetInputShape(3);
  *context->GetOutputShape(2) = *context->GetInputShape(4);
  *context->GetOutputShape(3) = *context->GetInputShape(1);
  auto& lengths = *context->GetOutputShape(4);
  lengths.SetDimNum(1);
  lengths.SetDim(0, rows);
  auto& table = *context->GetOutputShape(5);
  table.SetDimNum(2);
  table.SetDim(0, rows);
  table.SetDim(1, 17);
  *context->GetOutputShape(6) = *context->GetInputShape(1);
  *context->GetOutputShape(7) = *context->GetInputShape(1);
  return GRAPH_SUCCESS;
}
static graphStatus InferType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, ge::DT_INT32);
  context->SetOutputDataType(1, ge::DT_INT32);
  context->SetOutputDataType(2, ge::DT_INT32);
  context->SetOutputDataType(3, ge::DT_INT32);
  context->SetOutputDataType(4, ge::DT_INT32);
  context->SetOutputDataType(5, ge::DT_INT32);
  context->SetOutputDataType(6, ge::DT_INT32);
  context->SetOutputDataType(7, ge::DT_INT32);
  return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(DsaKvOffloadingPlanRope)
    .InferShape(InferShape)
    .InferDataType(InferType);
}  // namespace ge
