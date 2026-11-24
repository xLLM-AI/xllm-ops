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

#include "register/op_def_registry.h"

namespace ge {
namespace {

graphStatus infer_shape(gert::InferShapeContext* context) {
  for (size_t layer = 0;; ++layer) {
    const auto* state = context->GetDynamicInputShape(0, layer);
    auto* output = context->GetOutputShape(layer);
    if (state == nullptr || output == nullptr) {
      return layer > 0 && state == nullptr && output == nullptr ? GRAPH_SUCCESS
                                                                : GRAPH_FAILED;
    }
    *output = *state;
  }
}

}  // namespace
IMPL_OP_INFERSHAPE(KdaReplayssmCorrectionFold).InferShape(infer_shape);

}  // namespace ge
