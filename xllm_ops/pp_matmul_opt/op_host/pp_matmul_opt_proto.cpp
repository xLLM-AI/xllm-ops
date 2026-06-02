/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pp_matmul_opt_tiling.h"
#include "register/op_def_registry.h"

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *a_shape = context->GetInputShape(0);
  const gert::Shape *b_shape = context->GetInputShape(1);
  gert::Shape *c_shape = context->GetOutputShape(0);
  *c_shape = *a_shape;
  size_t b_dim = b_shape->GetDimNum();
  size_t c_dim = c_shape->GetDimNum();
  uint32_t N = b_shape->GetDim(b_dim - 2);
  c_shape->SetDim(c_dim - 1, N);
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
  const auto inputDataType = context->GetInputDataType(0);
  context->SetOutputDataType(0, inputDataType);
  return ge::GRAPH_SUCCESS;
}

IMPL_OP(PpMatmulOpt).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ge
