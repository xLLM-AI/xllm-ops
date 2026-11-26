/**
 * @file aclnn_sigmoid_mul.h
 * @brief ACLNN L2 API declaration - SigmoidMul
 */
#ifndef ACLNN_SIGMOID_MUL_H_
#define ACLNN_SIGMOID_MUL_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the workspace size for aclnnSigmoidMul
 * @param x [in] input tensor [B, W]
 * @param gate [in] gate tensor [B, W] or [B, 1]
 * @param out [in] output tensor
 * @param workspaceSize [out] workspace size
 * @param executor [out] executor
 * @return aclnnStatus
 */
ACLNN_API aclnnStatus aclnnSigmoidMulGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* gate,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
 * @brief Execute SigmoidMul: out = x * sigmoid(gate)
 */
ACLNN_API aclnnStatus aclnnSigmoidMul(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SIGMOID_MUL_H_
