/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias.cpp
 * \brief
 */
#include "add_rms_norm_bias.h"
#include "add_rms_norm_bias_split_d.h"
#include "add_rms_norm_bias_merge_n.h"
#include "add_rms_norm_bias_multi_n.h"
#include "add_rms_norm_bias_single_n.h"

using namespace AscendC;

#define GENERAL_OP_IMPL(templateClass, ...)              \
    do {                                                 \
        templateClass<__VA_ARGS__> op(&pipe);            \
        op.Init(x1, x2, gamma, beta, y, rstd, x, &tilingData); \
        op.Process();                                    \
    } while (0)


#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
// arch35（Ascend950/dav-3510）RegBase/VF 实现：按场景分三个模板，见 arch35/README.md
#include "arch35/add_rms_norm_bias_regbase.h"
#include "arch35/add_rms_norm_bias_single_n_vf.h"
#include "arch35/add_rms_norm_bias_split_d_vf.h"
#endif


#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
// beta 有无在入口按 tiling 判定后选模板实例，VF 函数体内无运行时分支
#define VF_OP_IMPL(templateClass, dtype)                             \
    do {                                                             \
        if (tilingData.nullptr_beta == 0) {                          \
            templateClass<dtype, true> op(&pipe);                    \
            op.Init(x1, x2, gamma, beta, y, rstd, x, &tilingData);   \
            op.Process();                                            \
        } else {                                                     \
            templateClass<dtype, false> op(&pipe);                   \
            op.Init(x1, x2, gamma, beta, y, rstd, x, &tilingData);   \
            op.Process();                                            \
        }                                                            \
    } while (0)

#define VF_PIPEFREE_OP_IMPL(templateClass, dtype)                    \
    do {                                                             \
        if (tilingData.nullptr_beta == 0) {                          \
            templateClass<dtype, true> op;                           \
            op.Init(x1, x2, gamma, beta, y, rstd, x, &tilingData);   \
            op.Process();                                            \
        } else {                                                     \
            templateClass<dtype, false> op;                          \
            op.Init(x1, x2, gamma, beta, y, rstd, x, &tilingData);   \
            op.Process();                                            \
        }                                                            \
    } while (0)
#endif

extern "C" __global__ __aicore__ void add_rms_norm_bias(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // Pipe-free SINGLE_N must return before a TPipe object is constructed.
    if (TILING_KEY_IS(16)) {
        VF_PIPEFREE_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasSingleNVF, half);
        return;
    } else if (TILING_KEY_IS(26)) {
        VF_PIPEFREE_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasSingleNVF, float);
        return;
    } else if (TILING_KEY_IS(36)) {
        VF_PIPEFREE_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasSingleNVF, bfloat16_t);
        return;
    }
#endif
    TPipe pipe;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    // ---- arch35 VF 路径（tilingKey = dtype*10 + {5,6,7}）----
    if (TILING_KEY_IS(15)) {
        VF_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasRegBase, half);
        return;
    } else if (TILING_KEY_IS(25)) {
        VF_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasRegBase, float);
        return;
    } else if (TILING_KEY_IS(35)) {
        VF_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasRegBase, bfloat16_t);
        return;
    } else if (TILING_KEY_IS(11)) {
        VF_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasSplitDVF, half);
        return;
    } else if (TILING_KEY_IS(21)) {
        VF_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasSplitDVF, float);
        return;
    } else if (TILING_KEY_IS(31)) {
        VF_OP_IMPL(AddRmsNormBias::KernelAddRmsNormBiasSplitDVF, bfloat16_t);
        return;
    }
#endif
    if (TILING_KEY_IS(10)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBias, half);
    } else if (TILING_KEY_IS(20)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBias, float);
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBias, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(11)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSplitD, half);
    } else if (TILING_KEY_IS(21)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSplitD, float);
    } else if (TILING_KEY_IS(31)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSplitD, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(12)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMergeN, half);
    } else if (TILING_KEY_IS(22)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMergeN, float);
    } else if (TILING_KEY_IS(32)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMergeN, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(13)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSingleN, half);
    } else if (TILING_KEY_IS(23)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSingleN, float);
    } else if (TILING_KEY_IS(33)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        GENERAL_OP_IMPL(KernelAddRmsNormBiasSingleN, bfloat16_t);
#endif
    } else if (TILING_KEY_IS(14)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMultiN, half);
    } else if (TILING_KEY_IS(34)) {
        GENERAL_OP_IMPL(KernelAddRmsNormBiasMultiN, bfloat16_t);
    }
}
