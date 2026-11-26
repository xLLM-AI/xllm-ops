/*!
 * \file sigmoid_mul_def.cpp
 * \brief SigmoidMul operator definition: inputs x/gate, output y.
 */
#include "register/op_def_registry.h"

namespace ops {
class SigmoidMul : public OpDef {
public:
    explicit SigmoidMul(const char* name) : OpDef(name)
    {
        this->Input("x")                                       // element-wise input [B, W]
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gate")                                    // gate [B, W] or [B, 1]
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")                                      // out = x * sigmoid(gate)
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
        this->AICore().AddConfig("ascend910_93", aicoreConfig910B);
    }
};
OP_ADD(SigmoidMul);
} // namespace ops
