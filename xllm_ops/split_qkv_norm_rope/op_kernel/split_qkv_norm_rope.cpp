/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*!
 * \file split_qkv_norm_rope.cpp
 * \brief SplitQkvNormRope kernel 入口（Ascend 950 / AIV only）。
 */

#include "split_qkv_norm_rope.h"

using namespace AscendC;

// DTYPE_QKV 由框架按 bin 的 dtype 组合展开（bf16 -> bfloat16_t，fp16 -> half）。
extern "C" __global__ __aicore__ void split_qkv_norm_rope(
    GM_ADDR qkv, GM_ADDR qGamma, GM_ADDR kGamma, GM_ADDR cosGm, GM_ADDR sinGm,
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    kernels::KernelSplitQkvNormRope<DTYPE_QKV> op;
    op.Init(qkv, qGamma, kGamma, cosGm, sinGm, q, k, v, &pipe, &tilingData);
    op.Process();
}
