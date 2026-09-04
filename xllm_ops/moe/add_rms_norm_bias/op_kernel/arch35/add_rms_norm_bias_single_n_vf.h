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
 * \file add_rms_norm_bias_single_n_vf.h
 * \brief SINGLE_N 分支的 VF/RegBase 特化：每核只处理一行（block_factor==1，即 num_row <= 核数）。
 *
 * 与 R-full-load 版（add_rms_norm_bias_regbase.h）的区别，也是本文件存在的理由：
 *   R-full-load 为多行复用设计——gamma/beta 各占一条常驻 TQue、双缓冲、rstd 对齐 buffer、
 *   binary-add 分级缓冲。这些固定成本在每核跑几十行时可忽略，但 SINGLE_N 每核只有一行、
 *   全程 2~4us，摊不掉（实测反而比老实现慢 20~25%）。
 *   这里改回老 SINGLE_N 的骨架：单块 TBuf 手工切片、无队列、无双缓冲，
 *   只把计算链换成 VF —— 干掉原实现里 Mul/Muls/ReduceSum/Adds/Sqrt/Duplicate/Div 之间的
 *   6 个 PipeBarrier<PIPE_V>，以及 rstd 的 GetValue V->S->V 标量往返。
 */
#ifndef ADD_RMS_NORM_BIAS_SINGLE_N_VF_H_
#define ADD_RMS_NORM_BIAS_SINGLE_N_VF_H_

#include "vf_reduce_common.h"

namespace AddRmsNormBias {
using namespace AscendC;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;

template <typename T, bool HAS_BETA>
class KernelAddRmsNormBiasSingleNVF {
    static constexpr uint32_t STATIC_UB_BYTES = 191U * 1024U;
    static constexpr uint32_t STATIC_FIXED_BYTES = 2048U;
    static constexpr uint32_t VL_F32 = 256 / sizeof(float);   // dav-3510: VREG 256B
    static constexpr uint32_t BLK_F32 = 32 / sizeof(float);
    static constexpr uint32_t T_SLOT_COUNT = HAS_BETA ? 4U : 3U;
    static constexpr uint32_t ALIGN_ELEMS = 32U / sizeof(T);
    static constexpr uint32_t STATIC_SLOT_ELEMS =
        ((STATIC_UB_BYTES - STATIC_FIXED_BYTES) / (T_SLOT_COUNT * sizeof(T) + sizeof(float)) /
         ALIGN_ELEMS) * ALIGN_ELEMS;
    static constexpr uint32_t FP32_BYTE_OFFSET = T_SLOT_COUNT * STATIC_SLOT_ELEMS * sizeof(T);
    static constexpr uint32_t REDUCE_BYTE_OFFSET = FP32_BYTE_OFFSET + STATIC_SLOT_ELEMS * sizeof(float);
    static constexpr uint32_t RSTD_BYTE_OFFSET = REDUCE_BYTE_OFFSET + VL_F32 * sizeof(float);
    static constexpr uint32_t BINARY_BYTE_OFFSET = RSTD_BYTE_OFFSET + VL_F32 * sizeof(float);
    static constexpr uint32_t BETA_BYTE_OFFSET = 3U * STATIC_SLOT_ELEMS * sizeof(T);
    static constexpr uint32_t BETA_ELEMS = HAS_BETA ? STATIC_SLOT_ELEMS : 0U;
    static constexpr uint32_t BINARY_ELEMS = (STATIC_UB_BYTES - BINARY_BYTE_OFFSET) / sizeof(float);
    static constexpr event_t MTE2_V_EVENT = static_cast<event_t>(EVENT_ID0);
    static constexpr event_t V_MTE3_EVENT = static_cast<event_t>(EVENT_ID0);

    // The statically addressed UB arena is shared by every dtype/beta instantiation.
    // Prove byte alignment, interval adjacency/non-overlap, element units and capacity here.
    static_assert(STATIC_UB_BYTES % 32U == 0U, "SINGLEN-STATIC UB size must be 32B aligned");
    static_assert(STATIC_SLOT_ELEMS > 0U, "SINGLEN-STATIC slot must not be empty");
    static_assert((STATIC_SLOT_ELEMS * sizeof(T)) % 32U == 0U,
                  "SINGLEN-STATIC T slot must be 32B aligned");
    static_assert(FP32_BYTE_OFFSET % 32U == 0U && REDUCE_BYTE_OFFSET % 32U == 0U &&
                  RSTD_BYTE_OFFSET % 32U == 0U && BINARY_BYTE_OFFSET % 32U == 0U,
                  "SINGLEN-STATIC offsets must be 32B aligned");
    static_assert(FP32_BYTE_OFFSET == T_SLOT_COUNT * STATIC_SLOT_ELEMS * sizeof(T),
                  "SINGLEN-STATIC T element-count unit drift");
    static_assert(REDUCE_BYTE_OFFSET - FP32_BYTE_OFFSET == STATIC_SLOT_ELEMS * sizeof(float),
                  "SINGLEN-STATIC fp32 element-count unit drift");
    static_assert(RSTD_BYTE_OFFSET - REDUCE_BYTE_OFFSET == VL_F32 * sizeof(float),
                  "SINGLEN-STATIC reduce element-count unit drift");
    static_assert(BINARY_BYTE_OFFSET - RSTD_BYTE_OFFSET == VL_F32 * sizeof(float),
                  "SINGLEN-STATIC rstd element-count unit drift");
    static_assert((!HAS_BETA && BETA_BYTE_OFFSET == FP32_BYTE_OFFSET && BETA_ELEMS == 0U) ||
                  (HAS_BETA && BETA_BYTE_OFFSET + BETA_ELEMS * sizeof(T) == FP32_BYTE_OFFSET),
                  "SINGLEN-STATIC beta/fp32 intervals overlap");
    static_assert(BINARY_BYTE_OFFSET < STATIC_UB_BYTES && BINARY_ELEMS > 0U,
                  "SINGLEN-STATIC binary interval must not be empty");
    static_assert(BINARY_BYTE_OFFSET + BINARY_ELEMS * sizeof(float) <= STATIC_UB_BYTES,
                  "SINGLEN-STATIC interval overflow");

public:
    __aicore__ inline KernelAddRmsNormBiasSingleNVF() = default;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd,
                                GM_ADDR x, const AddRMSNormBiasTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numCol = tiling->num_col;
        numColAlign = tiling->num_col_align;
        epsilon = tiling->epsilon;
        avgFactor = (numCol != 0) ? 1.0f / static_cast<float>(numCol) : 0.0f;
        binAddQuotient = tiling->bin_add_quotient;

        blockIdx_ = GetBlockIdx();
        x1Gm.SetGlobalBuffer((__gm__ T*)x1 + blockIdx_ * numCol, numCol);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2 + blockIdx_ * numCol, numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        if constexpr (HAS_BETA) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta, numCol);
        }
        yGm.SetGlobalBuffer((__gm__ T*)y + blockIdx_ * numCol, numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + blockIdx_, 1);
        xOutGm.SetGlobalBuffer((__gm__ T*)x + blockIdx_ * numCol, numCol);

        ASSERT(numColAlign <= STATIC_SLOT_ELEMS && "host SINGLEN-STATIC capacity guard drift");
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> x1Local(TPosition::VECCALC, 0U, STATIC_SLOT_ELEMS);
        LocalTensor<T> x2Local(TPosition::VECCALC, STATIC_SLOT_ELEMS * sizeof(T), STATIC_SLOT_ELEMS);
        LocalTensor<T> gammaLocal(TPosition::VECCALC, 2U * STATIC_SLOT_ELEMS * sizeof(T), STATIC_SLOT_ELEMS);
        LocalTensor<T> betaLocal(TPosition::VECCALC, BETA_BYTE_OFFSET, BETA_ELEMS);
        LocalTensor<float> xFp32(TPosition::VECCALC, FP32_BYTE_OFFSET, STATIC_SLOT_ELEMS);
        LocalTensor<float> reduceLocal(TPosition::VECCALC, REDUCE_BYTE_OFFSET, VL_F32);
        LocalTensor<float> rstdLocal(TPosition::VECCALC, RSTD_BYTE_OFFSET, VL_F32);
        LocalTensor<float> binAddBuf(TPosition::VECCALC, BINARY_BYTE_OFFSET, BINARY_ELEMS);

        // ---- 搬入 x1/x2/gamma/beta（一次性发出，等一次）----
        DataCopyPadIn(x1Local, x1Gm);
        DataCopyPadIn(x2Local, x2Gm);
        DataCopyPadIn(gammaLocal, gammaGm);
        if constexpr (HAS_BETA) {
            DataCopyPadIn(betaLocal, betaGm);
        }
        SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
        WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);

        // ---- VF ①：x = x1 + x2，同时落 T 版（输出）与 fp32 版（后续复用）----
        AddAndSpill(x1Local, x2Local, xFp32);

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        DataCopyPadOut(xOutGm, x1Local);

        // ---- 平方和 + rstd：全部在寄存器里完成 ----
        // 替代原实现的 Mul/Muls/ReduceSumCustom/Adds/Sqrt/Duplicate/Div + 6 个 PipeBarrier
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
            xFp32, reduceLocal, binAddBuf, static_cast<uint16_t>(1), numColAlign, numCol,
            binAddQuotient, BLK_F32);
        NormCommon::ComputeRstdNewtonRaphson<true, true>(reduceLocal, rstdLocal, 1, epsilon, avgFactor, VL_F32);

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        DataCopyExtParams rstdParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(rstdGm, rstdLocal, rstdParams);

        // ---- VF ②：y = x*rstd*gamma (+beta)，rstd 直接广播进 vreg，无标量往返 ----
        if constexpr (HAS_BETA) {
            CalcYBeta(xFp32, gammaLocal, betaLocal, rstdLocal, x2Local);
        } else {
            CalcYNoBeta(xFp32, gammaLocal, rstdLocal, x2Local);
        }

        SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
        DataCopyPadOut(yGm, x2Local);
    }

private:
    __aicore__ inline void DataCopyPadIn(LocalTensor<T>& dst, GlobalTensor<T>& src)
    {
        DataCopyExtParams p{1, static_cast<uint32_t>(numCol * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0.0)};
        DataCopyPad(dst, src, p, pad);
    }
    __aicore__ inline void DataCopyPadOut(GlobalTensor<T>& dst, LocalTensor<T>& src)
    {
        DataCopyExtParams p{1, static_cast<uint32_t>(numCol * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst, src, p);
    }

    __aicore__ inline void AddAndSpill(LocalTensor<T>& a, LocalTensor<T>& b, LocalTensor<float>& fp32Out)
    {
        __ubuf__ T* aPtr = (__ubuf__ T*)a.GetPhyAddr();
        __ubuf__ T* bPtr = (__ubuf__ T*)b.GetPhyAddr();
        __ubuf__ float* fPtr = (__ubuf__ float*)fp32Out.GetPhyAddr();
        uint32_t sreg = numColAlign;
        uint16_t loops = static_cast<uint16_t>((numColAlign + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> r1, r2, sum;
            MaskReg preg;
            for (uint16_t i = 0; i < loops; ++i) {
                uint32_t off = i * VL_F32;
                preg = AscendC::Reg::UpdateMask<float>(sreg);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(aPtr, r1, preg, off);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(bPtr, r2, preg, off);
                Add(sum, r1, r2, preg);
                NormCommon::NormCommonRegbase::StoreRegForDtype<T>(aPtr, sum, preg, off);
                AscendC::Reg::StoreAlign<float, StoreDist::DIST_NORM_B32>(fPtr + off, sum, preg);
            }
        }
    }

    __aicore__ inline void CalcYBeta(LocalTensor<float>& xFp32, LocalTensor<T>& gammaLocal,
                                     LocalTensor<T>& betaLocal, LocalTensor<float>& rstdLocal, LocalTensor<T>& yLocal)
    {
        __ubuf__ float* xPtr = (__ubuf__ float*)xFp32.GetPhyAddr();
        __ubuf__ T* gPtr = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ T* bPtr = (__ubuf__ T*)betaLocal.GetPhyAddr();
        __ubuf__ float* rPtr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ T* yPtr = (__ubuf__ T*)yLocal.GetPhyAddr();
        uint32_t sreg = numColAlign;
        uint16_t loops = static_cast<uint16_t>((numColAlign + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> xr, gr, br, rstdR, t1, t2;
            MaskReg preg;
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdR, rPtr);
            for (uint16_t i = 0; i < loops; ++i) {
                uint32_t off = i * VL_F32;
                preg = AscendC::Reg::UpdateMask<float>(sreg);
                NormCommon::NormCommonRegbase::LoadRegForDtype<float>(xPtr, xr, preg, off);
                Mul(t1, xr, rstdR, preg);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gPtr, gr, preg, off);
                Mul(t2, t1, gr, preg);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(bPtr, br, preg, off);
                Add(t2, t2, br, preg);
                NormCommon::NormCommonRegbase::StoreRegForDtype<T>(yPtr, t2, preg, off);
            }
        }
    }

    __aicore__ inline void CalcYNoBeta(LocalTensor<float>& xFp32, LocalTensor<T>& gammaLocal,
                                       LocalTensor<float>& rstdLocal, LocalTensor<T>& yLocal)
    {
        __ubuf__ float* xPtr = (__ubuf__ float*)xFp32.GetPhyAddr();
        __ubuf__ T* gPtr = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ float* rPtr = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        __ubuf__ T* yPtr = (__ubuf__ T*)yLocal.GetPhyAddr();
        uint32_t sreg = numColAlign;
        uint16_t loops = static_cast<uint16_t>((numColAlign + VL_F32 - 1) / VL_F32);
        __VEC_SCOPE__
        {
            RegTensor<float> xr, gr, rstdR, t1, t2;
            MaskReg preg;
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdR, rPtr);
            for (uint16_t i = 0; i < loops; ++i) {
                uint32_t off = i * VL_F32;
                preg = AscendC::Reg::UpdateMask<float>(sreg);
                NormCommon::NormCommonRegbase::LoadRegForDtype<float>(xPtr, xr, preg, off);
                Mul(t1, xr, rstdR, preg);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gPtr, gr, preg, off);
                Mul(t2, t1, gr, preg);
                NormCommon::NormCommonRegbase::StoreRegForDtype<T>(yPtr, t2, preg, off);
            }
        }
    }

    GlobalTensor<T> x1Gm, x2Gm, gammaGm, betaGm, yGm, xOutGm;
    GlobalTensor<float> rstdGm;
    uint32_t numCol{0};
    uint32_t numColAlign{0};
    uint32_t binAddQuotient{1};
    float epsilon{0};
    float avgFactor{0};
    uint32_t blockIdx_{0};
};
}  // namespace AddRmsNormBias
#endif  // ADD_RMS_NORM_BIAS_SINGLE_N_VF_H_
