/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "register/op_def_registry.h"

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  // The runtime preallocates final beam-select outputs with the requested
  // result width. Keep those shapes unchanged here so the op can consume
  // widened outputs such as num_return_sequences=512.
  (void)context;
  return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(OnerecFinalBeamSelect)
    .InferShape(InferShape);
}  // namespace ge