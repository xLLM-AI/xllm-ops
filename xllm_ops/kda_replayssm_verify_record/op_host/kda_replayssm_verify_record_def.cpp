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

#include <initializer_list>

#include "register/op_def_registry.h"

namespace ops {

class KdaReplayssmVerifyRecord final : public OpDef {
 public:
  explicit KdaReplayssmVerifyRecord(const char* name) : OpDef(name) {
    for (const char* input : {"query", "key", "value"}) {
      this->Input(input)
          .ParamType(REQUIRED)
          .DataType({ge::DT_BF16})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND})
          .IgnoreContiguous();
    }
    for (const char* input : {"gate", "beta"}) {
      this->Input(input)
          .ParamType(REQUIRED)
          .DataType({ge::DT_BF16})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    }
    for (const char* input : {"state", "a_log", "dt_bias"}) {
      this->Input(input)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    }
    this->Input("state_indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("scale").AttrType(REQUIRED).Float();
    this->Attr("qkv_token_stride").AttrType(REQUIRED).Int();

    for (const char* output : {"normalized_keys", "corrections", "decays"}) {
      this->Output(output)
          .ParamType(REQUIRED)
          .DataType({ge::DT_FLOAT})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    }
    this->Output("output")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});

    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(false)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false)
        .ExtendCfgInfo("coreType.value", "VectorCore");
    this->AICore().AddConfig("ascend910_93", config);
  }
};

OP_ADD(KdaReplayssmVerifyRecord);

}  // namespace ops
