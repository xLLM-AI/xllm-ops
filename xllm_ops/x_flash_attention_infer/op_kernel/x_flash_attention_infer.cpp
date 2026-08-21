/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// A5(Ascend950/DAV_3510) arch guard.
// Device side must use __NPU_ARCH__ (per catlass migration guide); host side uses
// CATLASS_ARCH. Accept either so the A5 path is selected regardless of which macro
// the toolchain injects for the kernel translation unit.
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)) || (defined(CATLASS_ARCH) && (CATLASS_ARCH == 3510))
#define XFA_ARCH35 1
#endif

// Device(kernel) side lacks -DCATLASS_ARCH (host-only inject). Derive it from
// __NPU_ARCH__ HERE, before ANY include, so every catlass forwarding header in
// this translation unit (incl. common.h below) dispatches to the ascend950
// specialization consistently.
#if defined(XFA_ARCH35) && !defined(CATLASS_ARCH)
#define CATLASS_ARCH 3510
#endif

// A3(AtlasA2/A3, __NPU_ARCH__ == 2201) arch guard.
// The A3 path (non-XFA_ARCH35) still pulls catlass forwarding headers via
// x_flash_attention_infer.h; the new catlass tile-copy forwarders dispatch
// ONLY when CATLASS_ARCH is explicitly 2201/3510 (host-only inject on device).
// Derive CATLASS_ARCH from __NPU_ARCH__ HERE (before ANY include) so the A3
// device translation unit resolves CopyGmToL1/CopyL1ToL0A/ScaleGranularity/...
// without affecting the already-validated A5(3510) path above.
#if !defined(XFA_ARCH35) && !defined(CATLASS_ARCH) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
#define CATLASS_ARCH 2201
#endif

#if defined(XFA_ARCH35)
// arch35 (example49-ported) defines its own FAIKernelParams / helpers, which
// clash with x_flash_attention_infer_common.h. So DO NOT pull common.h here;
// instead declare only the TILING_KEY constants this branch dispatches on.
// Values mirror x_flash_attention_infer_common.h:85-88 (keep in sync).
#ifndef QFP16_KVFP16_TND_CAUSALMASK_FD_TILING
#define QFP16_KVFP16_TND_CAUSALMASK_FD_TILING   1000000000000001113
#endif
#ifndef QFP16_KVFP16_KVNZ_CAUSALMASK_FD_TILING
#define QFP16_KVFP16_KVNZ_CAUSALMASK_FD_TILING  1000000000000001213
#endif
#ifndef QBF16_KVBF16_TND_CAUSALMASK_FD_TILING
#define QBF16_KVBF16_TND_CAUSALMASK_FD_TILING   1000000000000001123
#endif
#ifndef QBF16_KVBF16_KVNZ_CAUSALMASK_FD_TILING
#define QBF16_KVBF16_KVNZ_CAUSALMASK_FD_TILING  1000000000000001223
#endif
#include "arch35/a5_x_flash_attention_infer.h"
#else
#include "x_flash_attention_infer.h"
#include "x_flash_attention_infer_fd.h"
#endif

extern "C" __global__ __aicore__ void x_flash_attention_infer(GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache,
                        GM_ADDR mask, GM_ADDR block_table, GM_ADDR actual_q_lens,
                        GM_ADDR actual_kv_lens, GM_ADDR extra_tiling, GM_ADDR attn_out, GM_ADDR workspace, GM_ADDR tiling) {
#if defined(XFA_ARCH35)
    // A5(Ascend950/DAV_3510): dispatch into arch35 example49-ported FAInferKernel.
    // Host tiling A5 branch (bnAxisStartIdx/sparseStartIdx) is filled in stage-3.
    SetAtomicNone();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(QFP16_KVFP16_TND_CAUSALMASK_FD_TILING) ||
        TILING_KEY_IS(QFP16_KVFP16_KVNZ_CAUSALMASK_FD_TILING)) {
        XllmOps::XfaArch35::FAInferA5Dispatch<half, true, true>(
            query, key_cache, value_cache, mask, block_table, actual_q_lens, actual_kv_lens, attn_out, tiling);
    } else if (TILING_KEY_IS(QBF16_KVBF16_TND_CAUSALMASK_FD_TILING) ||
               TILING_KEY_IS(QBF16_KVBF16_KVNZ_CAUSALMASK_FD_TILING)) {
        XllmOps::XfaArch35::FAInferA5Dispatch<bfloat16_t, true, true>(
            query, key_cache, value_cache, mask, block_table, actual_q_lens, actual_kv_lens, attn_out, tiling);
    }
#else
    // workspace use; [s,p,oTemp,oUpdate,shared_workspace,unshared_workspace]
    SetAtomicNone();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(tiling_data, tiling);

    GM_ADDR s = workspace;
    GM_ADDR p = workspace + tiling_data.mm1OutSize;
    GM_ADDR oTemp = p + tiling_data.smOnlineOutSize;
    GM_ADDR oUpdate = oTemp + tiling_data.mm2OutSize;
    GM_ADDR gmlse = oUpdate + tiling_data.updateSize;
    GM_ADDR glo = gmlse + tiling_data.splitLseTotalSize;
    __gm__ SplitKvExtraInfo *extraInfo = reinterpret_cast<__gm__ SplitKvExtraInfo *>(extra_tiling);
    auto coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    if (TILING_KEY_IS(QFP16_KVFP16_TND_CAUSALMASK_FD_TILING)) {  // FD fp16
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<half, half, layout::ColumnMajor, layout::RowMajor ,true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    } else if (TILING_KEY_IS(QBF16_KVBF16_TND_CAUSALMASK_FD_TILING)) {  // FD bf16
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<bfloat16_t, bfloat16_t, layout::ColumnMajor, layout::RowMajor, true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    } else if (TILING_KEY_IS(QFP16_KVFP16_KVNZ_CAUSALMASK_FD_TILING)) {
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<half, half, layout::nZ, layout::zN, true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    } else if (TILING_KEY_IS(QBF16_KVBF16_KVNZ_CAUSALMASK_FD_TILING)) {
        if (extraInfo->coreInfo[coreIdx].startBIdx != UINT32_MAX) {
            FDInfer<bfloat16_t, bfloat16_t, layout::nZ, layout::zN, true, FaiKenel::MaskType::MASK_CAUSAL>(query, key_cache, value_cache, mask, block_table,
            attn_out, actual_q_lens, actual_kv_lens, s, p, oTemp, oUpdate, gmlse, glo, tiling, extraInfo);
        } else {
            AscendC::SyncAll();
        }
    }
#endif
}
