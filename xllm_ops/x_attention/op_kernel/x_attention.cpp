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

// A5(Ascend950/DAV_3510) arch guard.
// Device side must use __NPU_ARCH__ (per catlass migration guide); host side uses
// CATLASS_ARCH. Accept either so the A5 path is selected regardless of which macro
// the toolchain injects for the kernel translation unit.
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510))
#define XA_ARCH35 1
#endif

// Device(kernel) side lacks -DCATLASS_ARCH (host-only inject). Derive it from
// __NPU_ARCH__ HERE, before ANY include, so every catlass forwarding header in
// this translation unit dispatches to the ascend950 specialization consistently.
#if defined(XA_ARCH35) && !defined(CATLASS_ARCH)
#define CATLASS_ARCH 3510
#endif

// A3(AtlasA2/A3, __NPU_ARCH__ == 2201) arch guard. Derive CATLASS_ARCH so the A3
// device translation unit resolves the catlass forwarding headers consistently.
#if !defined(XA_ARCH35) && !defined(CATLASS_ARCH) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#define CATLASS_ARCH 2201
#endif

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

#if defined(XA_ARCH35)
#include "arch35/x_attention_catlass_helper.h"
#else
#include "x_attention_catlass_helper.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void x_attention(GM_ADDR query, GM_ADDR shared_key_block, GM_ADDR shared_value_block,
                        GM_ADDR unshared_key_block, GM_ADDR unshared_value_block, GM_ADDR unshared_block_table,
                        GM_ADDR shared_kv_lens, GM_ADDR decode_step, GM_ADDR shared_block_table, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling) {
#if defined(XA_ARCH35)
    // ===== A5(Ascend950/DAV_3510) path =====
    // workspace layout: [sharedO, sharedMax, sharedSum, unsharedO, unsharedMax, unsharedSum]
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR sharedO = workspace;
    GM_ADDR sharedMax = sharedO + tiling_data.qOSize;
    GM_ADDR sharedSum = sharedMax + tiling_data.sumMaxSize;
    GM_ADDR unsharedO = sharedSum + tiling_data.sumMaxSize;
    GM_ADDR unsharedMax = unsharedO + tiling_data.qOSize;
    GM_ADDR unsharedSum = unsharedMax + tiling_data.sumMaxSize;
    int64_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

    XAttnKernelCommonParams params{query, shared_key_block, shared_value_block, unshared_key_block, unshared_value_block,
     shared_block_table, unshared_block_table, shared_kv_lens, decode_step, sharedO, sharedMax, sharedSum, unsharedO, unsharedMax,
     unsharedSum, attn_out, tiling};

    if (coreIdx < tiling_data.sharedInfo.usedCoreNum) {
        CallSharedInferKernel<DTYPE_QUERY, DTYPE_SHARED_KV_LENS>(params, &tiling_data);
    } else {
        CallUnsharedInferKernel<DTYPE_QUERY, DTYPE_SHARED_KV_LENS, DTYPE_UNSHARED_BLOCK_TABLE>(params, &tiling_data);
    }
    AscendC::SyncAll<false>();
    CallCombineScale<DTYPE_QUERY>(params, &tiling_data);
#else
    // ===== A3(AtlasA2/A3) path =====
    // workspace use; [s,p,oTemp,oUpdate,shared_workspace,unshared_workspace]
    #define CALL_XATTN_KERNEL(INPUT_TYPE, SHARED_PAGED_FLAG, UNSHARED_PAGED_FLAG) \
        do { \
            if (coreIdx < tiling_data.sharedCoreNum) { \
                CallSharedInferKernelShort<INPUT_TYPE, SHARED_PAGED_FLAG>(params, &tiling_data); \
            } else { \
                CallUnsharedInferKernel<INPUT_TYPE, UNSHARED_PAGED_FLAG>(params, &tiling_data); \
            } \
            AscendC::SyncAll<false>(); \
            CallCombineScale<INPUT_TYPE>(params, &tiling_data); \
        } while (0)

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR s = workspace;
    GM_ADDR p = workspace + tiling_data.mm1OutSize;
    GM_ADDR oTemp = p + tiling_data.smOnlineOutSize;
    GM_ADDR oUpdate = oTemp + tiling_data.mm2OutSize;
    GM_ADDR shared_workspace = oUpdate + tiling_data.updateSize;
    GM_ADDR unshared_workspace = shared_workspace + tiling_data.sharedWorkspaceSize;
    int64_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

    XAttnKernelParams params{query, shared_key_block, shared_value_block, unshared_key_block, unshared_value_block,
     shared_block_table, unshared_block_table, shared_kv_lens, decode_step, s, p, oTemp, oUpdate, shared_workspace,
         unshared_workspace, attn_out, tiling};
    if (TILING_KEY_IS(4)) {          // 0b0100
        CALL_XATTN_KERNEL(half, false, true);
    } else if (TILING_KEY_IS(6)) {   // 0b0110
        CALL_XATTN_KERNEL(bfloat16_t, false, true);
    } else if (TILING_KEY_IS(8)) {   // 0b1000
        CALL_XATTN_KERNEL(half, true, false);
    } else if (TILING_KEY_IS(10)) {  // 0b1010
        CALL_XATTN_KERNEL(bfloat16_t, true, false);
    }
#endif
}
