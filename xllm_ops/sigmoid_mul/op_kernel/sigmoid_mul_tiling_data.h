/*!
 * \file sigmoid_mul_tiling_data.h
 * \brief tiling data struct for SigmoidMul
 */
#ifndef _SIGMOID_MUL_TILING_DATA_H_
#define _SIGMOID_MUL_TILING_DATA_H_

struct SigmoidMulTilingData {
    int64_t totalNum = 0;     // total elements of x (B * W)
    int64_t rowWidth = 0;     // broadcast row width (x last dim); 1 when gate is element-wise
    int64_t blockFactor = 0;  // elements per core (flat) or rows per core (broadcast)
    int64_t ubFactor = 0;     // elements per UB loop iteration
    int32_t isBroadcast = 0;  // 1: gate is [.., 1] (row-broadcast); 0: gate same shape as x
};
#endif
