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

#include "../op_kernel/kda_replayssm_verify_record_tiling_data.h"
#include "register/op_def_registry.h"

namespace ge {
namespace {

graphStatus infer_shape(gert::InferShapeContext* context) {
  using namespace kda_replayssm;
  const auto* query = context->GetInputShape(0);
  const auto* state = context->GetInputShape(5);
  auto* output = context->GetOutputShape(3);
  if (query == nullptr || query->GetDimNum() != 3 || state == nullptr ||
      state->GetDimNum() != 4 || output == nullptr) {
    return GRAPH_FAILED;
  }
  const gert::Shape record_shape{
      state->GetDim(0), kWindow, kHeads, kKeyDimension};
  for (size_t index = 0; index < 3; ++index) {
    auto* record = context->GetOutputShape(index);
    if (record == nullptr) {
      return GRAPH_FAILED;
    }
    *record = record_shape;
  }
  *output = gert::Shape{query->GetDim(0), kHeads, kValueDimension};
  return GRAPH_SUCCESS;
}

}  // namespace
IMPL_OP_INFERSHAPE(KdaReplayssmVerifyRecord).InferShape(infer_shape);

}  // namespace ge
