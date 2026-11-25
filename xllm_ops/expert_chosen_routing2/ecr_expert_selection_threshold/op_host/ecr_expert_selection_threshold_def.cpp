/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_def_registry.h"

namespace ops {
class EcrExpertSelectionThreshold : public OpDef {
 public:
  explicit EcrExpertSelectionThreshold(const char* name) : OpDef(name) {
    this->Input("prob")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();

    this->Output("tau_val")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("tau_idx")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("n_gt")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();

    this->Attr("seq_lengths").AttrType(REQUIRED).ListInt();
    this->Attr("top_p").AttrType(REQUIRED).Float(1.0f);

    OpAICoreConfig aicoreConfig;
    aicoreConfig.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("coreType.value", "AiCore")
        .ExtendCfgInfo("opFile.value", "ecr_expert_selection_threshold");
    this->AICore().AddConfig("ascend950", aicoreConfig);
  }
};
OP_ADD(EcrExpertSelectionThreshold);
}  // namespace ops
