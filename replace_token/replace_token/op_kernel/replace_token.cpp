#include "kernel_operator.h"
#include "./replace_token.h"

extern "C" __global__ __aicore__ void replace_token(GM_ADDR forkedTokenIds, GM_ADDR lastStepOutPutTokenIds,GM_ADDR out,GM_ADDR workspace,GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    ReplaceToken op;
    op.Init(forkedTokenIds, lastStepOutPutTokenIds, out, tiling_data.sequenceLength);
    op.Process();
    
}
