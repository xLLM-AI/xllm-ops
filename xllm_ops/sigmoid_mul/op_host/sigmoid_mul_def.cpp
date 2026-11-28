/*!
 * \file sigmoid_mul_def.cpp
 * \brief SigmoidMul operator definition: out = x * sigmoid(gate), bf16, 2-D.
 */
#include "register/op_def_registry.h"

namespace ops {
class SigmoidMul : public OpDef {
public:
    explicit SigmoidMul(const char* name) : OpDef(name)
    {
        this->Input("x")                                       // value tensor [M, N]
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gate")                                    // gate [M, N] or [M, 1]
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")                                      // out = x * sigmoid(gate)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig cfg;
        cfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)   // infer-shape requires exactly 2-D
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", cfg);
        this->AICore().AddConfig("ascend910_93", cfg);
    }
};
OP_ADD(SigmoidMul);
} // namespace ops
