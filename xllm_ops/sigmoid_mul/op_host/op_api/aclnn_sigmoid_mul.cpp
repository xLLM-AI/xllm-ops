/**
 * @file aclnn_sigmoid_mul.cpp
 * @brief ACLNN L2 API implementation - SigmoidMul
 */
#include "aclnn_sigmoid_mul.h"
#include "sigmoid_mul.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_BF16, DataType::DT_FLOAT16, DataType::DT_FLOAT
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool HasEmptyTensor(const aclTensor* x, const aclTensor* gate)
{
    return x->IsEmpty() || gate->IsEmpty();
}

static bool CheckNotNull(const aclTensor* x, const aclTensor* gate, const aclTensor* out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(gate, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* gate, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_MATCH(x, gate->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(out, x->GetDataType(), return false);
    OP_CHECK(IsDtypeSupported(x->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: dtype=%d. Supported: BF16, FP16, FP32.",
                     static_cast<int>(x->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* gate, const aclTensor* out)
{
    auto format1 = x->GetStorageFormat();
    auto format2 = gate->GetStorageFormat();
    auto formatOut = out->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(format1) || IsPrivateFormat(format2) || IsPrivateFormat(formatOut)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: x=%d, gate=%d, out=%d",
                     static_cast<int>(format1), static_cast<int>(format2), static_cast<int>(formatOut)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* gate, const aclTensor* out)
{
    OP_CHECK_MAX_DIM(x, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(gate, ACLNN_MAX_SHAPE_RANK, return false);
    OP_CHECK_MAX_DIM(out, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* gate, const aclTensor* out)
{
    CHECK_COND(CheckNotNull(x, gate, out), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(x, gate, out), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: x_dtype=%d, gate_dtype=%d, out_dtype=%d",
               static_cast<int>(x->GetDataType()), static_cast<int>(gate->GetDataType()),
               static_cast<int>(out->GetDataType()));
    CHECK_COND(CheckFormat(x, gate, out), ACLNN_ERR_PARAM_INVALID,
               "CheckFormat failed");
    CHECK_COND(CheckShape(x, gate, out), ACLNN_ERR_PARAM_INVALID,
               "CheckShape failed: x_dim=%zu, gate_dim=%zu, out_dim=%zu",
               x->GetViewShape().GetDimNum(), gate->GetViewShape().GetDimNum(),
               out->GetViewShape().GetDimNum());
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSigmoidMulGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* gate,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSigmoidMul, DFX_IN(x, gate), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, gate, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (HasEmptyTensor(x, gate)) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xContiguous = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(xContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gateContiguous = l0op::Contiguous(gate, uniqueExecutor.get());
    CHECK_RET(gateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opResult = l0op::SigmoidMul(xContiguous, gateContiguous, uniqueExecutor.get());
    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSigmoidMul(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSigmoidMul);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
