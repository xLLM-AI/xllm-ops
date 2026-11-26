/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "kernel_operator.h"
#include "t5_split_qkv_append_cache.h"

extern "C" __global__ __aicore__ void t5_split_qkv_append_cache(
    GM_ADDR qkv, GM_ADDR pastKey, GM_ADDR pastValue, GM_ADDR query,
    GM_ADDR key, GM_ADDR value, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    T5SplitQkvAppendCacheKernel kernel;
    kernel.Init(qkv, pastKey, pastValue, query, key, value,
                tilingData.batch, tilingData.headNum,
                tilingData.historyLen, tilingData.headDim);
    kernel.Process();
}
