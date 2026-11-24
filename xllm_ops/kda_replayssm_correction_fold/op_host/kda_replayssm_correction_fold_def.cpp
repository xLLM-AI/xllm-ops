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

class KdaReplayssmCorrectionFold final : public OpDef {
 public:
  explicit KdaReplayssmCorrectionFold(const char* name) : OpDef(name) {
    for (const char* input :
         {"states", "normalizedKeys", "corrections", "decays"}) {
      this->Input(input)
          .ParamType(DYNAMIC)
          .DataType({ge::DT_FLOAT})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    }
    for (const char* input : {"stateIndices", "acceptedLengths"}) {
      this->Input(input)
          .ParamType(REQUIRED)
          .DataType({ge::DT_INT32})
          .Format({ge::FORMAT_ND})
          .UnknownShapeFormat({ge::FORMAT_ND});
    }
    this->Output("statesOut")
        .ParamType(DYNAMIC)
        .DataType({ge::DT_FLOAT})
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

OP_ADD(KdaReplayssmCorrectionFold);

}  // namespace ops
