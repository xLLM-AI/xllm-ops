/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arch35.h
 * \brief moe_fused_reducesum_div architecture adaptation for ascend950 / A5
 *        (__NPU_ARCH__ == 3510).
 *
 * On A5 the vector/scalar/MTE pipes run asynchronously and ReduceSum is emulated via
 * Reg vector APIs that truly consume their sharedTmpBuffer ("accumulation method 2").
 * Therefore this adapter:
 *   - allocates a dedicated ReduceSum work buffer (workBUF) and uses it as scratch,
 *     so ReduceSum cannot clobber the compute buffer (outputLocal);
 *   - inserts explicit MTE2->V / V->S / V->MTE3 hazard syncs around buffers that live
 *     outside the TQue pipeline (castBUF is a plain TBuf).
 */

#ifndef MOE_FUSED_REDUCESUM_DIV_ARCH35_H
#define MOE_FUSED_REDUCESUM_DIV_ARCH35_H

#include "kernel_operator.h"

using namespace AscendC;

namespace kernels {

struct ArchAdapter {
    // A5 needs the dedicated ReduceSum tmp buffer.
    static constexpr bool kNeedWorkBuf = true;

    template <typename T>
    __aicore__ inline static void InitWorkBuf(TPipe &pipe, TBuf<TPosition::VECCALC> &workBUF,
                                              uint32_t elemCount) {
        pipe.InitBuffer(workBUF, elemCount * sizeof(float));
    }

    // castBUF is a plain TBuf outside inputQUE: the widening Cast (V) must wait for the
    // staging DataCopyPad (MTE2) to finish writing castBUF before reading it.
    __aicore__ inline static void SyncMte2ToV() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev);
        WaitFlag<HardEvent::MTE2_V>(ev);
    }

    // ReduceSum (V) writes the row sum into UBUF; the following GetValue is a scalar (S)
    // read. Without this sync the scalar read may observe stale UBUF data (typically 0),
    // turning the reciprocal into inf/nan.
    __aicore__ inline static void SyncVToS() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(ev);
        WaitFlag<HardEvent::V_S>(ev);
    }

    // Cast (V) writes castDst, then DataCopyPad (MTE3) reads it. castBUF is a TBuf (no
    // queue), so an explicit V->MTE3 sync is required; a bare PipeBarrier<PIPE_V> does
    // NOT block the MTE3 pipe (observed as un-normalized output for small n).
    __aicore__ inline static void SyncVToMte3() {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ev);
        WaitFlag<HardEvent::V_MTE3>(ev);
    }

    // A5: ReduceSum truly consumes the sharedTmpBuffer, use the dedicated workLocal so it
    // cannot clobber outputLocal.
    __aicore__ inline static LocalTensor<float> ReduceScratch(const LocalTensor<float> &workLocal,
                                                              const LocalTensor<float> &outputLocal) {
        (void)outputLocal;
        return workLocal;
    }
};

} // namespace kernels

#endif // MOE_FUSED_REDUCESUM_DIV_ARCH35_H