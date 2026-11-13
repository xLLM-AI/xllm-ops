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

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/cross_attention_fusion_helper.h"
#else
#include "cross_attention_catlass_helper.h"
#include "lib/matmul_intf.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void cross_attention(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                        GM_ADDR block_table, GM_ADDR actual_kv_lens, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    if (TILING_KEY_IS(3510)) {
        KERNEL_TASK_TYPE(3510, KERNEL_TYPE_MIX_AIC_1_2);
        AscendC::InitSocState();
        const auto* tilingDataPtr = reinterpret_cast<const __gm__ Xllm::CrossAttentionTilingData*>(tiling);
        GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
        if ASCEND_IS_AIC {
            CallCrossAttentionArch35Aic<DTYPE_QUERY, DTYPE_BLOCK_TABLE, DTYPE_ACTUAL_KV_LENS>(
                query, key, value, block_table, actual_kv_lens, userWorkspace, tilingDataPtr);
        }
        if ASCEND_IS_AIV {
            CallCrossAttentionArch35<DTYPE_QUERY, DTYPE_BLOCK_TABLE, DTYPE_ACTUAL_KV_LENS>(
                query, key, value, block_table, actual_kv_lens, attn_out, userWorkspace, tilingDataPtr);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
#else
    GET_TILING_DATA(tiling_data, tiling);
    GM_ADDR s = workspace;
    GM_ADDR p = workspace + tiling_data.mm1OutSize;
    GM_ADDR oTemp = p + tiling_data.pOutSize;
    CrossAttnKernelParams params{query, key, value, block_table, actual_kv_lens,
                                 s, p, oTemp, attn_out, tiling};

    CallCrossAttention<DTYPE_QUERY, DTYPE_BLOCK_TABLE, DTYPE_ACTUAL_KV_LENS>(params, &tiling_data);
#endif
}
