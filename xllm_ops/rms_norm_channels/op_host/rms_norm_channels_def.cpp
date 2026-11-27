/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <vector>
#include "register/op_def_registry.h"

namespace ops {
class RmsNormChannels : public OpDef {
public:
    explicit RmsNormChannels(const char* name) : OpDef(name)
    {
        // value/gamma/out share one dtype; the reduction itself runs in FP32.
        const std::vector<ge::DataType> types = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
        const std::vector<ge::Format> formats(types.size(), ge::FORMAT_ND);
        this->Input("value").ParamType(REQUIRED).DataType(types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Input("gamma").ParamType(REQUIRED).DataType(types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Output("out").ParamType(REQUIRED).DataType(types).Format(formats)
            .UnknownShapeFormat(formats).AutoContiguous();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-12f);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend950", config);
    }
};
OP_ADD(RmsNormChannels);
} // namespace ops
