/**
 * @file sigmoid_mul.h
 * @brief ACLNN L0 API declaration - SigmoidMul
 */
#ifndef OP_API_INC_LEVEL0_SIGMOID_MUL_H_
#define OP_API_INC_LEVEL0_SIGMOID_MUL_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* SigmoidMul(const aclTensor* x, const aclTensor* gate, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_SIGMOID_MUL_H_
