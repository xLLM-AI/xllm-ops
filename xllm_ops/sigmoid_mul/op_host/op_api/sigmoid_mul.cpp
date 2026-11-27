/**
 * @file sigmoid_mul.cpp
 * @brief ACLNN L0 API implementation - SigmoidMul (bf16, 2-D).
 *
 * out = x * sigmoid(gate); gate is [M, N] (element-wise) or [M, 1] (row broadcast).
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
    DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* x, const aclTensor* gate)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SigmoidMul not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(x->GetDataType(), AICORE_DTYPE_SUPPORT_LIST) &&
             CheckType(gate->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SigmoidMul not supported: dtype x=%d, gate=%d. Supported: BF16.",
                     static_cast<int>(x->GetDataType()), static_cast<int>(gate->GetDataType())),
             return false);
    return true;
}

// Only 2-D element-wise (same shape) or trailing-dim-1 row broadcast [M, 1].
static bool IsSupportedGateLayout(const op::Shape& xShape, const op::Shape& gateShape)
{
    if (xShape.GetDimNum() != 2 || gateShape.GetDimNum() != 2) {
        return false;
    }
    const int64_t M = xShape.GetDim(0);
    const int64_t N = xShape.GetDim(1);
    const int64_t gM = gateShape.GetDim(0);
    const int64_t gN = gateShape.GetDim(1);
    const bool sameShape = (M == gM && N == gN);
    const bool rowBroadcast = (gN == 1 && gM == M);
    return sameShape || rowBroadcast;
}

static bool SigmoidMulInferShape(const op::Shape& xShape, const op::Shape& gateShape, op::Shape& outShape)
{
    OP_CHECK(IsSupportedGateLayout(xShape, gateShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SigmoidMul: gate must be same shape as x or [M, 1] (2-D only)."),
             return false);
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
