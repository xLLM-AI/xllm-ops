/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file add_rms_norm_regbase.h
 * \brief
 */
#ifndef ADD_RMS_NORM_BIAS_REGBASE_H
#define ADD_RMS_NORM_BIAS_REGBASE_H
#include "kernel_operator.h"
#include "vf_reduce_common.h"

namespace AddRmsNormBias {
using namespace AscendC;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr int32_t UNROLL_NUM = 2;

constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;

using AscendC::Reg::LoadAlign;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreAlign;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
constexpr static uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr static uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}
template <typename T, bool HAS_BETA>
class KernelAddRmsNormBiasRegBase {
    // T=float 时 x 输出与 fp32 中间量逐字节相同，合并成同一块 UB，省掉一份行大小缓冲
    static constexpr bool X_FP32 = is_same<T, float>::value;

public:
    __aicore__ inline KernelAddRmsNormBiasRegBase(TPipe* pipe) { pPipe = pipe; }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd,
                                GM_ADDR x, const AddRMSNormBiasTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        blockFactor = tiling->block_factor;
        binAddQuotient = tiling->bin_add_quotient;
        rowFactor = tiling->row_factor;
        epsilon = tiling->epsilon;
        numColAlign = tiling->num_col_align;
        avgFactor = tiling->avg_factor;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor : numRow - (GetBlockNum() - 1) * blockFactor;
        uint64_t rstdUbSizeAlignSize =
            ((rowFactor + VL_FP32 - 1) / VL_FP32) * VL_FP32 * sizeof(float);
        uint16_t binaryAddQuotientLoop = (binAddQuotient + VL_FP32 - 1) / VL_FP32;
        uint32_t binaryAddBufLen = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32 * sizeof(float) *
                                   rowFactor;

        xGm1.SetGlobalBuffer((__gm__ T*)x1 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        xGm2.SetGlobalBuffer((__gm__ T*)x2 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        if constexpr (HAS_BETA) {
            betaGm.SetGlobalBuffer((__gm__ T*)beta, numCol);
        }
        yGm.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, blockFactor);
        xOutGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);

        pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(inQueueGamma, BUFFER_NUM, numColAlign * sizeof(T));
        if constexpr (HAS_BETA) {
            pPipe->InitBuffer(inQueueBeta, BUFFER_NUM, numColAlign * sizeof(T));
        }
        // y / x 都是写一次即搬走，双缓冲收益远小于输入侧，改单缓冲换 rowFactor
        pPipe->InitBuffer(outQueueY, BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueX, BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xReduceBuff, rstdUbSizeAlignSize);
        if constexpr (!X_FP32) {
            pPipe->InitBuffer(xFp32Buff, numColAlign * sizeof(float) * rowFactor);
        }
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufLen);
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        LocalTensor<T> betaLocal;
        if constexpr (HAS_BETA) {
            CopyInBeta();
            betaLocal = inQueueBeta.DeQue<T>();
        }
        uint32_t rowLoopCount = static_cast<uint32_t>((rowWork + rowFactor - 1) / rowFactor);
        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopCount; rowLoopIdx++) {
            uint64_t rowLoopOffset = rowLoopIdx * rowFactor * numCol;
            uint32_t curRows = Min(rowWork - rowLoopIdx * rowFactor, rowFactor);
            Compute(rowLoopIdx, gammaLocal, betaLocal, curRows, rowLoopOffset);
        }
        inQueueGamma.FreeTensor(gammaLocal);
        if constexpr (HAS_BETA) {
            inQueueBeta.FreeTensor(betaLocal);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t rowLoopIdx, LocalTensor<T> gammaLocal, LocalTensor<T> betaLocal,
                                   uint32_t curRows, uint64_t rowLoopOffset)
    {
        CopyInXMutiMoveAlign(rowLoopOffset, numColAlign, curRows);
        LocalTensor<T> xLocal1 = inQueueX1.DeQue<T>();
        LocalTensor<T> xLocal2 = inQueueX2.DeQue<T>();
        LocalTensor<T> xOutLocal = outQueueX.AllocTensor<T>();
        LocalTensor<float> xFp32Local;
        if constexpr (X_FP32) {
            xFp32Local = xOutLocal.template ReinterpretCast<float>();
            AddSpillFp32(xLocal1, xLocal2, xFp32Local, curRows, numColAlign);
        } else {
            xFp32Local = xFp32Buff.Get<float>();
            AddSpillCast(xLocal1, xLocal2, xOutLocal, xFp32Local, curRows, numColAlign);
        }
        inQueueX1.FreeTensor(xLocal1);
        inQueueX2.FreeTensor(xLocal2);
        if constexpr (!X_FP32) {
            outQueueX.EnQue<T>(xOutLocal);
            CopyOutX(rowLoopOffset, curRows, numColAlign);
        }

        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<float> xReduceLocal = xReduceBuff.Get<float>();
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
            xFp32Local, xReduceLocal, binaryAddBuf, static_cast<uint16_t>(curRows), numColAlign, numCol,
            static_cast<uint32_t>(binAddQuotient), static_cast<uint32_t>(BLK_B32));
        NormCommon::ComputeRstdNewtonRaphson<true, true>(xReduceLocal, rstdLocal, curRows, epsilon, avgFactor, VL_FP32);
        outQueueRstd.EnQue<float>(rstdLocal);

        rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyExtParams rstdCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(curRows * sizeof(float)),
                                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
        DataCopyPad(rstdGm[rowLoopIdx * rowFactor], rstdLocal, rstdCopyParams);

        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        // BR-FULLLOAD-HOIST is deliberately narrow; all other full-load shapes retain row-first order.
        if (numCol <= 512U && curRows >= 2U) {
            CalculateYColumnOuter(xFp32Local, gammaLocal, betaLocal, yLocal, rstdLocal, curRows,
                                  numColAlign, numCol);
        } else if constexpr (HAS_BETA) {
            CalculateYBeta(xFp32Local, gammaLocal, betaLocal, yLocal, rstdLocal, curRows, numColAlign, numCol);
        } else {
            CalculateYNoBeta(xFp32Local, gammaLocal, yLocal, rstdLocal, curRows, numColAlign, numCol);
        }
        outQueueRstd.FreeTensor(rstdLocal);
        outQueueY.EnQue<T>(yLocal);
        CopyOutY(rowLoopOffset, curRows, numColAlign);
        if constexpr (X_FP32) {
            // xOutLocal 即 xFp32Local，必须等 CalculateY 读完再交给 MTE3
            outQueueX.EnQue<T>(xOutLocal);
            CopyOutX(rowLoopOffset, curRows, numColAlign);
        }
    }

    // Column-block outer VF: gamma/beta are loaded once and reused by every row pair in this UB tile.
    __aicore__ inline void CalculateYColumnOuter(LocalTensor<float>& xFp32Local, LocalTensor<T>& gammaLocal,
                                                 LocalTensor<T>& betaLocal, LocalTensor<T>& yLocal,
                                                 LocalTensor<float>& rstdLocal, uint32_t curRows,
                                                 uint32_t numColAlign, uint32_t reduceNum)
    {
        __local_mem__ float* xPtr = (__local_mem__ float*)xFp32Local.GetPhyAddr();
        __local_mem__ T* gammaPtr = (__local_mem__ T*)gammaLocal.GetPhyAddr();
        __local_mem__ T* betaPtr = (__local_mem__ T*)betaLocal.GetPhyAddr();
        __local_mem__ T* yPtr = (__local_mem__ T*)yLocal.GetPhyAddr();
        __local_mem__ float* rstdPtr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        const uint16_t colLoops = static_cast<uint16_t>((reduceNum + VL_FP32 - 1U) / VL_FP32);
        const uint16_t rowPairs = static_cast<uint16_t>(curRows / 2U);
        const bool hasTailRow = (curRows & 1U) != 0U;
        __VEC_SCOPE__
        {
            RegTensor<float> gammaReg, betaReg, x0Reg, x1Reg, rstd0Reg, rstd1Reg, y0Reg, y1Reg;
            MaskReg mask;
            uint32_t remaining = reduceNum;
            for (uint16_t col = 0; col < colLoops; ++col) {
                const uint32_t colOffset = static_cast<uint32_t>(col) * VL_FP32;
                mask = AscendC::Reg::UpdateMask<float>(remaining);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, colOffset);
                if constexpr (HAS_BETA) {
                    NormCommon::NormCommonRegbase::LoadRegForDtype<T>(betaPtr, betaReg, mask, colOffset);
                }
                for (uint16_t pair = 0; pair < rowPairs; ++pair) {
                    const uint32_t row0 = static_cast<uint32_t>(pair) * 2U;
                    const uint32_t row1 = row0 + 1U;
                    AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd0Reg, rstdPtr + row0);
                    AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdPtr + row1);
                    NormCommon::NormCommonRegbase::LoadRegForDtype<float>(
                        xPtr, x0Reg, mask, row0 * numColAlign + colOffset);
                    NormCommon::NormCommonRegbase::LoadRegForDtype<float>(
                        xPtr, x1Reg, mask, row1 * numColAlign + colOffset);
                    AscendC::Reg::Mul(y0Reg, x0Reg, rstd0Reg, mask);
                    AscendC::Reg::Mul(y1Reg, x1Reg, rstd1Reg, mask);
                    AscendC::Reg::Mul(y0Reg, y0Reg, gammaReg, mask);
                    AscendC::Reg::Mul(y1Reg, y1Reg, gammaReg, mask);
                    if constexpr (HAS_BETA) {
                        AscendC::Reg::Add(y0Reg, y0Reg, betaReg, mask);
                        AscendC::Reg::Add(y1Reg, y1Reg, betaReg, mask);
                    }
                    NormCommon::NormCommonRegbase::StoreRegForDtype<T>(
                        yPtr, y0Reg, mask, row0 * numColAlign + colOffset);
                    NormCommon::NormCommonRegbase::StoreRegForDtype<T>(
                        yPtr, y1Reg, mask, row1 * numColAlign + colOffset);
                }
                if (hasTailRow) {
                    const uint32_t row = curRows - 1U;
                    AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd0Reg, rstdPtr + row);
                    NormCommon::NormCommonRegbase::LoadRegForDtype<float>(
                        xPtr, x0Reg, mask, row * numColAlign + colOffset);
                    AscendC::Reg::Mul(y0Reg, x0Reg, rstd0Reg, mask);
                    AscendC::Reg::Mul(y0Reg, y0Reg, gammaReg, mask);
                    if constexpr (HAS_BETA) {
                        AscendC::Reg::Add(y0Reg, y0Reg, betaReg, mask);
                    }
                    NormCommon::NormCommonRegbase::StoreRegForDtype<T>(
                        yPtr, y0Reg, mask, row * numColAlign + colOffset);
                }
            }
        }
    }

    // x = x1 + x2。T=float：只落一份（xFp32 与 x 输出同一块 UB）
    __aicore__ inline void AddSpillFp32(LocalTensor<T>& xLocal1, LocalTensor<T>& xLocal2,
                                        LocalTensor<float>& xFp32Local, uint32_t curRows, uint32_t numColAlign)
    {
        __ubuf__ T* x1InUb = (__ubuf__ T*)xLocal1.GetPhyAddr();
        __ubuf__ T* x2InUb = (__ubuf__ T*)xLocal2.GetPhyAddr();
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        uint32_t sreg = curRows * numColAlign;
        uint16_t loopCount = (sreg + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(sreg);
                LoadRegForDtype<T>(x1InUb, x1, pregLoop, offset);
                LoadRegForDtype<T>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(xFp32Tmp + offset, xSum, pregLoop);
            }
        }
    }

    // T=half/bfloat16：x 输出按 T 落一份，fp32 中间量另落一份
    __aicore__ inline void AddSpillCast(LocalTensor<T>& xLocal1, LocalTensor<T>& xLocal2, LocalTensor<T>& xOutLocal,
                                        LocalTensor<float>& xFp32Local, uint32_t curRows, uint32_t numColAlign)
    {
        __ubuf__ T* x1InUb = (__ubuf__ T*)xLocal1.GetPhyAddr();
        __ubuf__ T* x2InUb = (__ubuf__ T*)xLocal2.GetPhyAddr();
        __ubuf__ T* xOutInUb = (__ubuf__ T*)xOutLocal.GetPhyAddr();
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        uint32_t sreg = curRows * numColAlign;
        uint16_t loopCount = (sreg + VL_FP32 - 1) / VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(sreg);
                LoadRegForDtype<T>(x1InUb, x1, pregLoop, offset);
                LoadRegForDtype<T>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreRegForDtype<T>(xOutInUb, xSum, pregLoop, offset);
                StoreAlign<float, StoreDist::DIST_NORM_B32>(xFp32Tmp + offset, xSum, pregLoop);
            }
        }
    }

    // y = x * rstd * gamma + beta
    __aicore__ inline void CalculateYBeta(LocalTensor<float>& xFp32Local, LocalTensor<T>& gammaLocal,
                                          LocalTensor<T>& betaLocal, LocalTensor<T>& yLocal,
                                          LocalTensor<float>& rstdLocal, uint32_t curRows, uint32_t numColAlign,
                                          uint32_t reduceNum)
    {
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        __ubuf__ T* gammaInUb = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ T* betaInUb = (__ubuf__ T*)betaLocal.GetPhyAddr();
        __ubuf__ T* yInUb = (__ubuf__ T*)yLocal.GetPhyAddr();
        __ubuf__ float* rstdInUb = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((reduceNum + VL_FP32 - 1) / VL_FP32);
        uint16_t loopRowsFold = loopRows / 2;
        uint16_t loopRowsHasLast = loopRows % 2;
        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> gammaReg;
            RegTensor<float> betaReg;
            RegTensor<float> rstd1Reg;
            RegTensor<float> rstd2Reg;
            RegTensor<float> mul1Reg;
            RegTensor<float> mul1UnrollReg;
            RegTensor<float> mul2Reg;
            RegTensor<float> mul2UnrollReg;
            for (uint16_t i = 0; i < loopRowsFold; ++i) {
                uint32_t sregCount = reduceNum;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (2 * i + 1));
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset1 = (2 * i) * numColAlign + r * VL_FP32;
                    uint32_t offset2 = (2 * i + 1) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset1);
                    LoadRegForDtype<float>(xFp32Tmp, x2Reg, regCurLoop, offset2);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    Mul(mul1UnrollReg, x2Reg, rstd2Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    Mul(mul2UnrollReg, mul1UnrollReg, gammaReg, regCurLoop);
                    LoadRegForDtype<T>(betaInUb, betaReg, regCurLoop, r * VL_FP32);
                    Add(mul2Reg, mul2Reg, betaReg, regCurLoop);
                    Add(mul2UnrollReg, mul2UnrollReg, betaReg, regCurLoop);
                    StoreRegForDtype<T>(yInUb, mul2Reg, regCurLoop, offset1);
                    StoreRegForDtype<T>(yInUb, mul2UnrollReg, regCurLoop, offset2);
                }
            }
            for (uint16_t i = 0; i < loopRowsHasLast; ++i) {
                uint32_t sregCount = reduceNum;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * loopRowsFold);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset = (2 * loopRowsFold) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    LoadRegForDtype<T>(betaInUb, betaReg, regCurLoop, r * VL_FP32);
                    Add(mul2Reg, mul2Reg, betaReg, regCurLoop);
                    StoreRegForDtype<T>(yInUb, mul2Reg, regCurLoop, offset);
                }
            }
        }
    }

    // y = x * rstd * gamma
    __aicore__ inline void CalculateYNoBeta(LocalTensor<float>& xFp32Local, LocalTensor<T>& gammaLocal,
                                            LocalTensor<T>& yLocal, LocalTensor<float>& rstdLocal, uint32_t curRows,
                                            uint32_t numColAlign, uint32_t reduceNum)
    {
        __ubuf__ float* xFp32Tmp = (__ubuf__ float*)xFp32Local.GetPhyAddr();
        __ubuf__ T* gammaInUb = (__ubuf__ T*)gammaLocal.GetPhyAddr();
        __ubuf__ T* yInUb = (__ubuf__ T*)yLocal.GetPhyAddr();
        __ubuf__ float* rstdInUb = (__ubuf__ float*)rstdLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((reduceNum + VL_FP32 - 1) / VL_FP32);
        uint16_t loopRowsFold = loopRows / 2;
        uint16_t loopRowsHasLast = loopRows % 2;
        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> gammaReg;
            RegTensor<float> rstd1Reg;
            RegTensor<float> rstd2Reg;
            RegTensor<float> mul1Reg;
            RegTensor<float> mul1UnrollReg;
            RegTensor<float> mul2Reg;
            RegTensor<float> mul2UnrollReg;
            for (uint16_t i = 0; i < loopRowsFold; ++i) {
                uint32_t sregCount = reduceNum;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * i);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (2 * i + 1));
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset1 = (2 * i) * numColAlign + r * VL_FP32;
                    uint32_t offset2 = (2 * i + 1) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset1);
                    LoadRegForDtype<float>(xFp32Tmp, x2Reg, regCurLoop, offset2);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    Mul(mul1UnrollReg, x2Reg, rstd2Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    Mul(mul2UnrollReg, mul1UnrollReg, gammaReg, regCurLoop);
                    StoreRegForDtype<T>(yInUb, mul2Reg, regCurLoop, offset1);
                    StoreRegForDtype<T>(yInUb, mul2UnrollReg, regCurLoop, offset2);
                }
            }
            for (uint16_t i = 0; i < loopRowsHasLast; ++i) {
                uint32_t sregCount = reduceNum;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + 2 * loopRowsFold);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset = (2 * loopRowsFold) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    StoreRegForDtype<T>(yInUb, mul2Reg, regCurLoop, offset);
                }
            }
        }
    }

    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t offset, uint32_t curCols, uint32_t curRows = 0)
    {
        LocalTensor<T> xLocal1 = inQueueX1.AllocTensor<T>();
        LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(curRows),                                               // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)),                                    // blockLen
            static_cast<uint32_t>(0),                                                     // srcStride
            static_cast<uint32_t>((numColAlign - curCols) * sizeof(T) / ALIGN_32_FACTOR), // dstStride
            0                                                                             // rsv
        };
        DataCopyPadExtParams<T> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T>(0.0)      // paddingValue
        };
        DataCopyPad(xLocal1, xGm1[offset], extParams, padParams);
        DataCopyPad(xLocal2, xGm2[offset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                  // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(0),                  // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPadExtParams<T> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T>(0.0)      // paddingValue
        };
        DataCopyPad(gammaLocal, gammaGm, copyParams, padParams);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyInBeta()
    {
        LocalTensor<T> betaLocal = inQueueBeta.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                  // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(0),                  // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPadExtParams<T> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T>(0.0)      // paddingValue
        };
        DataCopyPad(betaLocal, betaGm, copyParams, padParams);
        inQueueBeta.EnQue(betaLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),            // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(srcStride),          // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<T> xLocal = outQueueX.DeQue<T>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),            // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(srcStride),          // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPad(xOutGm[offset], xLocal, copyParams);
        outQueueX.FreeTensor(xLocal);
    }

private:
    TPipe* pPipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueBeta;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueRstd;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueX;
    TBuf<TPosition::VECCALC> xReduceBuff;
    TBuf<TPosition::VECCALC> xFp32Buff;
    TBuf<TPosition::VECCALC> binaryAddBuf;
    GlobalTensor<T> xGm1;
    GlobalTensor<T> xGm2;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> betaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xOutGm;
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t rowFactor;
    uint64_t binAddQuotient;
    float epsilon;
    float avgFactor;
    uint64_t rowWork{1};
};
} // namespace AddRmsNormBias
#endif // ADD_RMS_NORM_BIAS_REGBASE_H
