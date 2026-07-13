/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "register/op_def_registry.h"

namespace ops {
class Qwen35GdnPrefillSuperOp : public OpDef {
public:
    explicit Qwen35GdnPrefillSuperOp(const char *name) : OpDef(name)
    {
        this->Input("mixed_qkv").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("z").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("b").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("a").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("conv_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("conv_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("a_log").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("dt_bias").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("ssm_state")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("norm_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("mask_lower").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("mask_full").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("minus_identity").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Input("cu_seqlens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("packed_qkv").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("g").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("initial_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("mega_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("g_sum").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("g_t").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("beta_t").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("mega_a").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("a_inv_f32").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("a_inv").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("w").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("u").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("h").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("v_new").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("final_state").ParamType(REQUIRED).DataType({ge::DT_FLOAT16}).Format({ge::FORMAT_ND});
        this->Output("conv_state_out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});
        this->Output("ssm_state_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND});

        this->Attr("num_matrices").AttrType(REQUIRED).Int(0);
        this->Attr("conv_state_index").AttrType(REQUIRED).Int(0);
        this->Attr("ssm_state_index").AttrType(REQUIRED).Int(0);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(Qwen35GdnPrefillSuperOp);
}  // namespace ops
