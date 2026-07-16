/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arch32.h
 * \brief moe_fused_reducesum_div architecture adaptation for A2 (__CCE_AICORE__ == 220)
 *        and A3 (__NPU_ARCH__ == 3003/3113).
 *
 * On these architectures the TQue EnQue/DeQue semantics already order MTE2 before the
 * vector Cast, vector before scalar reads, and vector before MTE3. ReduceSum also takes
 * the "front-n-elements" path and never reads its sharedTmpBuffer. Therefore every
 * hazard-sync primitive below is a no-op and no dedicated work buffer is allocated,
 * keeping more UB available for compute rows.
 */

#ifndef MOE_FUSED_REDUCESUM_DIV_ARCH32_H
#define MOE_FUSED_REDUCESUM_DIV_ARCH32_H

#include "kernel_operator.h"

using namespace AscendC;

namespace kernels {

struct ArchAdapter {
    // A2/A3 never allocate the dedicated ReduceSum tmp buffer: ReduceSum ignores it.
    static constexpr bool kNeedWorkBuf = false;

    template <typename T>
    __aicore__ inline static void InitWorkBuf(TPipe & /*pipe*/, TBuf<TPosition::VECCALC> & /*workBUF*/,
                                              uint32_t /*elemCount*/) {}

    // MTE2 -> V ordering is covered by inputQUE EnQue/DeQue.
    __aicore__ inline static void SyncMte2ToV() {}

    // ReduceSum -> GetValue (V -> S) ordering is implicit.
    __aicore__ inline static void SyncVToS() {}

    // Cast -> DataCopyPad (V -> MTE3) ordering is covered by outputQUE EnQue/DeQue.
    __aicore__ inline static void SyncVToMte3() {}

    // A2/A3: reuse outputLocal as ReduceSum scratch (it is never read by ReduceSum).
    __aicore__ inline static LocalTensor<float> ReduceScratch(const LocalTensor<float> &workLocal,
                                                              const LocalTensor<float> &outputLocal) {
        (void)workLocal;
        return outputLocal;
    }
};

} // namespace kernels

#endif // MOE_FUSED_REDUCESUM_DIV_ARCH32_H