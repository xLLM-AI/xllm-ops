/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <type_traits>
#include "ada_rms_norm.h"

extern "C" __global__ __aicore__ void ada_rms_norm(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR weight,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.rows == 0) {
        return;
    }
    AdaRmsNormData params{};
    params.rows = tilingData.rows;
    params.rowsPerCore = tilingData.rowsPerCore;
    params.seqLength = tilingData.seqLength;
    params.scaleBatchStride = tilingData.scaleBatchStride;
    params.scaleSeqStride = tilingData.scaleSeqStride;
    params.shiftBatchStride = tilingData.shiftBatchStride;
    params.shiftSeqStride = tilingData.shiftSeqStride;
    params.hiddenDim = tilingData.hiddenDim;
    params.rawRowBytes = tilingData.rawRowBytes;
    params.floatRowBytes = tilingData.floatRowBytes;
    params.hasWeight = tilingData.hasWeight;
    params.epsilon = tilingData.epsilon;
    AscendC::TPipe pipe;
    KernelAdaRmsNorm<DTYPE_X, std::is_same_v<DTYPE_X, float>> op(&pipe);
    op.Init(x, scale, shift, weight, out, params);
    op.Process();
}
