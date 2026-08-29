/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lightning_indexer_v2_template_tiling_key.h"
#if (__CCE_AICORE__ == 310)
    #include "arch35/lightning_indexer_v2_kernel_arch35.h"
#else
    #include "arch22/lightning_indexer_v2_kernel_arch22.h"
#endif
using namespace LIV2Kernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                             \
    do {                                                                                                         \
        templateClass<LIV2Type<__VA_ARGS__>> op;                                                                 \
        GET_TILING_DATA_WITH_STRUCT(LIV2TilingData, tiling_data_in, tiling);                                     \
        const LIV2TilingData *__restrict tiling_data = &tiling_data_in;                                          \
        op.Init(q, k, w, cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK,                               \
                blocktable, outputIdxOffset, metadata, sparseIndices, sparseValues, user,                        \
                tiling_data, &tPipe);                                                                            \
        op.Process();                                                                                            \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void lightning_indexer_v2(__gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w,
                                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                                __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                                __gm__ uint8_t *cmpResidualK, __gm__ uint8_t *blocktable,
                                                __gm__ uint8_t *outputIdxOffset,  __gm__ uint8_t *metadata,
                                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                                __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
                                                
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (ORIG_DTYPE_Q == DT_BF16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerV2Kernel, bfloat16_t, bfloat16_t, int32_t, float, uint32_t,
            PAGE_ATTENTION, LI_V2_LAYOUT(LAYOUT_T), LI_V2_LAYOUT(K_LAYOUT_T), true);
    } else if (ORIG_DTYPE_Q == DT_FLOAT16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerV2Kernel, half, half, int32_t, float, uint32_t,
            PAGE_ATTENTION, LI_V2_LAYOUT(LAYOUT_T), LI_V2_LAYOUT(K_LAYOUT_T), true);
    }
}
