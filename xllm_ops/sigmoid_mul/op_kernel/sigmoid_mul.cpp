/*!
 * \file sigmoid_mul_arch22.cpp
 * \brief SigmoidMul operator kernel entry (arch22).
 */
#include "sigmoid_mul.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void sigmoid_mul(GM_ADDR x, GM_ADDR gate, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidMulTilingData);
    GET_TILING_DATA_WITH_STRUCT(SigmoidMulTilingData, tilingData, tiling);
    NsSigmoidMul::SigmoidMul<D_T_X, BUFFER_MODE> op;
    op.Init(x, gate, y, &tilingData);
    op.Process();
}
