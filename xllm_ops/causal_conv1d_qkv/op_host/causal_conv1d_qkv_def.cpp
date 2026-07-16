#include "register/op_def_registry.h"
#include "../../causal_conv1d/op_kernel/causal_conv1d_tiling_data.h"

namespace ops {

class CausalConv1dQkv : public OpDef {
public:
    explicit CausalConv1dQkv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("convStates")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("queryStartLoc")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("cacheIndices")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("initialStateMode")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Input("numAcceptedTokens")
            .ParamType(OPTIONAL)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        this->Attr("activationMode").AttrType(OPTIONAL).Int(CAUSAL_CONV1D_ACTIVATION_SILU_PACKED_QKV);
        this->Attr("padSlotId").AttrType(OPTIONAL).Int(-1);
        this->Attr("runMode").AttrType(OPTIONAL).Int(0);
        this->Attr("qDim").AttrType(OPTIONAL).Int(1024);
        this->Attr("kDim").AttrType(OPTIONAL).Int(1024);
        this->Attr("vDim").AttrType(OPTIONAL).Int(3072);
        this->Attr("headDim").AttrType(OPTIONAL).Int(128);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("coreType.value", "AiCore");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};
OP_ADD(CausalConv1dQkv);

} // namespace ops
