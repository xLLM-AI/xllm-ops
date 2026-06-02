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
class PpMatmulOpt : public OpDef {
public:
  explicit PpMatmulOpt(const char *name) : OpDef(name) {
    this->Input("a")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("b")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("c")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

    // this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

    // this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(PpMatmulOpt);
} // namespace ops
