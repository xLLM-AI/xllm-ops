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

#include "register/op_def_registry.h"

namespace ops {
class XAttentionTl : public OpDef {
 public:
  explicit XAttentionTl(const char* name) : OpDef(name) {
    this->Input("q_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("k_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("v_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("q_unshared_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_k_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_v_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("output_shared_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("output_unshared_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("shared_exp_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_exp_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("shared_max_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Input("unshared_max_handle")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND});
    // this->SetInferShape(ge::InferShape);
    // this->AICore().SetTiling(optiling::TilingForXAttentionTlFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(XAttentionTl);
}  // namespace ops
