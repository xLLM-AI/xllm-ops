#include "register/op_def_registry.h"

namespace ops {

class Qwen35GdnDecodeSuperOpBatch : public OpDef {
public:
    explicit Qwen35GdnDecodeSuperOpBatch(const char* name) : OpDef(name)
    {
        this->Input("qkv").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("z").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();
        this->Input("convWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("convState")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("aLog")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("dtBias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("ssmState")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("stateIndices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("normWeight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("convOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("convStateOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("ssmStateOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND}).AutoContiguous();

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};

OP_ADD(Qwen35GdnDecodeSuperOpBatch);

} // namespace ops
