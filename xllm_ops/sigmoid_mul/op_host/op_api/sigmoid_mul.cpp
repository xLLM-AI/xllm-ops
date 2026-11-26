/**
 * @file sigmoid_mul.cpp
 * @brief ACLNN L0 API implementation - SigmoidMul
 *
 * out = x * sigmoid(gate); gate supports [B, W] or [B, 1] (broadcast).
 */
#include "sigmoid_mul.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SigmoidMul);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_BF16, DataType::DT_FLOAT16, DataType::DT_FLOAT
};

static bool IsAiCoreSupport(const aclTensor* x, const aclTensor* gate)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201 || npuArch == NpuArch::DAV_3510,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SigmoidMul not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
             CheckType(gate->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SigmoidMul not supported: dtype x=%d, gate=%d. Supported: BF16, FP16, FP32.",
                     static_cast<int>(x->GetDataType()), static_cast<int>(gate->GetDataType())),
             return false);
    return true;
}

static bool SigmoidMulInferShape(const op::Shape& xShape, const op::Shape& gateShape, op::Shape& outShape)
{
    OP_CHECK(BroadcastInferShape(xShape, gateShape, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Shape broadcast failed."), return false);
    return true;
}

static const aclTensor* SigmoidMulAiCore(const aclTensor* x, const aclTensor* gate,
                                         const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(SigmoidMulAiCore, x, gate, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SigmoidMul,
        OP_INPUT(x, gate), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SigmoidMulAiCore failed."), return nullptr);
    return out;
}

const aclTensor* SigmoidMul(const aclTensor* x, const aclTensor* gate, aclOpExecutor* executor)
{
    Shape outShape;
    const aclTensor* out = nullptr;

    OP_CHECK(SigmoidMulInferShape(x->GetViewShape(), gate->GetViewShape(), outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."), return nullptr);

    out = executor->AllocTensor(outShape, x->GetDataType());

    OP_CHECK(IsAiCoreSupport(x, gate),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."), return nullptr);

    return SigmoidMulAiCore(x, gate, out, executor);
}

} // namespace l0op
