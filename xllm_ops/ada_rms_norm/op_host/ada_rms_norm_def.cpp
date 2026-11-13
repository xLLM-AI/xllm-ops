/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {
class AdaRmsNorm : public OpDef {
public:
    explicit AdaRmsNorm(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> types = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
        const std::vector<ge::Format> formats(types.size(), ge::FORMAT_ND);
        for (const char* input : {"x", "scale", "shift"}) {
            this->Input(input).ParamType(REQUIRED).DataType(types).Format(formats)
                .UnknownShapeFormat(formats).AutoContiguous();
        }
        this->Input("weight").ParamType(OPTIONAL).DataType(types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Output("out").ParamType(REQUIRED).DataType(types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6f);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(AdaRmsNorm);
} // namespace ops
