/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "register/op_def_registry.h"

namespace {

enum InputIndex {
  LOGITS = 0,
};

enum AttrIndex {
  TOP_K = 1,
};

enum OutputIndex {
  OUT_TOKENS = 0,
  OUT_LOGPROBS,
};

}  // namespace

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* logits_shape = context->GetInputShape(LOGITS);
  const int64_t top_k =
      static_cast<int64_t>(*context->GetAttrs()->GetAttrPointer<int>(TOP_K));

  gert::Shape* out_tokens_shape = context->GetOutputShape(OUT_TOKENS);
  out_tokens_shape->SetDimNum(2);
  out_tokens_shape->SetDim(0, logits_shape->GetDim(0));
  out_tokens_shape->SetDim(1, top_k);

  gert::Shape* out_logprobs_shape = context->GetOutputShape(OUT_LOGPROBS);
  out_logprobs_shape->SetDimNum(2);
  out_logprobs_shape->SetDim(0, logits_shape->GetDim(0));
  out_logprobs_shape->SetDim(1, top_k);
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(OUT_TOKENS, ge::DT_INT32);
  context->SetOutputDataType(OUT_LOGPROBS, ge::DT_FLOAT);
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RecConstrainedTopK)
    .InferShape(InferShape).InferDataType(InferDataType);
}  // namespace ge