/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include "register/op_def_registry.h"

namespace ops {
class MegaKdaDecode : public OpDef {
public:
    explicit MegaKdaDecode(const char* name) : OpDef(name)
    {
        Input("qkv").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("gate").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("beta").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("a_log").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Input("gate_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Input("conv_weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("conv_bias").ParamType(OPTIONAL).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("conv_state_in").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Input("ssm_state_in").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Input("cu_seqlens").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("conv_read_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("conv_write_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("ssm_read_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("ssm_write_indices").ParamType(REQUIRED).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Input("num_accepted_tokens").ParamType(OPTIONAL).DataType({ge::DT_INT32}).FormatList({ge::FORMAT_ND});
        Output("output").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Output("conv_state_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).FormatList({ge::FORMAT_ND});
        Output("ssm_state_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
        Attr("mode").AttrType(OPTIONAL).Int(0);
        Attr("max_query_tokens").AttrType(OPTIONAL).Int(1);
        Attr("conv_output_slots").AttrType(REQUIRED).Int();
        Attr("ssm_output_slots").AttrType(REQUIRED).Int();
        AICore().AddConfig("ascend910b");
    }
};
OP_ADD(MegaKdaDecode);
} // namespace ops
