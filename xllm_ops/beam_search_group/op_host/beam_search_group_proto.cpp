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

#include "beam_search_group_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* x1_shape = context->GetInputShape(0);
  gert::Shape* y0_shape = context->GetOutputShape(0);
  *y0_shape = *x1_shape;
  gert::Shape* y1_shape = context->GetOutputShape(1);
  *y1_shape = *x1_shape;
  gert::Shape* y2_shape = context->GetOutputShape(2);
  *y2_shape = *x1_shape;
  gert::Shape* y3_shape = context->GetOutputShape(3);
  *y3_shape = *x1_shape;
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BeamSearchGroup).InferShape(InferShape);
}  // namespace ge

