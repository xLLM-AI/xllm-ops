/**
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file sigmoid_mul.cpp
 * \brief SigmoidMul kernel entry: out = sigmoid(x1) * x2.
 *        Tiling key 1000: x1/x2 same shape (flat elementwise).
 *        Tiling key 1001: x1 [M, 1] broadcast against x2 [M, N].
 *        The keys are literals because the compiler discovers the per-key binaries by
 *        scanning TILING_KEY_IS.
 */
#include "sigmoid_mul.h"
#include "sigmoid_mul_bcast.h"

extern "C" __global__ __aicore__ void sigmoid_mul(GM_ADDR x1,
                                                  GM_ADDR x2,
                                                  GM_ADDR out,
                                                  GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    (void)workspace;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1000)) {
        SigmoidMulKernel::KernelSigmoidMul<DTYPE_X1> op;
        op.Init(x1, x2, out, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1001)) {
        SigmoidMulBcastKernel::KernelSigmoidMulBcast<DTYPE_X1> op;
        op.Init(x1, x2, out, tilingData);
        op.Process();
    }
}
