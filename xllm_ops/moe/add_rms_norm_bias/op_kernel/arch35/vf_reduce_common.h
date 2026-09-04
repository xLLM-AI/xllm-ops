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
 * \file vf_reduce_common.h
 * \brief arch35（dav-3510）VF/RegBase 计算链公共 helper，仅保留本算子三个 VF 模板实际使用的符号。
 *        提取自 CANN ops-nn norm/ 的 arch35 头文件（reduce_common_regbase.h、platform shim），
 *        来源许可为 CANN Open Software License Agreement Version 2.0，与本仓一致。
 */
#ifndef ADD_RMS_NORM_BIAS_VF_REDUCE_COMMON_H
#define ADD_RMS_NORM_BIAS_VF_REDUCE_COMMON_H

#include "kernel_operator.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace platform {
// dav-3510 架构常量，与 Ops::Base::platform_util.h 在 950 上提供的取值一致
__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSize()
{
    return 256U;
}
}

namespace NormCommon {
using namespace AscendC;
using AscendC::Reg::CreateMask;
using AscendC::Reg::LoadDist;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::MaskReg;
using AscendC::Reg::MemType;
using AscendC::Reg::RegTensor;
using AscendC::Reg::StoreDist;
using AscendC::Reg::UpdateMask;

constexpr int32_t V_LENGTH = static_cast<int32_t>(platform::GetVRegSize() / sizeof(float));
constexpr uint16_t DICHOTOMY_ADD_COEFF = 2;

constexpr AscendC::Reg::CastTrait castTraitB162B32 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
template <bool NEED_MAX = true>
__aicore__ inline void ComputeRstdNewtonRaphsonReg(RegTensor<float>& var, RegTensor<float>& rstd, MaskReg& preg,
                                                   float epsilon)
{
    static constexpr float POS_INF = 3.40282366920938E+38;
    static constexpr float SCALAR1 = -0.5;
    static constexpr float SCALAR2 = 1.5;
    static constexpr float SCALAR3 = 0.5;
    static constexpr float SCALAR0 = -99.99;

    RegTensor<float> r;
    RegTensor<float> y;
    RegTensor<float> s;
    RegTensor<float> t;
    RegTensor<float> one;
    RegTensor<float> scalar1;
    RegTensor<float> t1;
    RegTensor<float> t3;
    RegTensor<float> t4;
    RegTensor<float> scalarInf;
    RegTensor<float> scalarZero;
    MaskReg cmpRegZero;
    MaskReg cmpRegInf;

    Duplicate(scalarInf, POS_INF, preg);
    Duplicate(scalarZero, float(0.0), preg);
    Duplicate(one, float(1.0), preg);
    Duplicate(scalar1, SCALAR3, preg);
    Duplicate(t1, SCALAR2, preg);
    Duplicate(s, float(1.0), preg);

    Adds(var, var, epsilon, preg);
    if constexpr (NEED_MAX) {
        Maxs(var, var, SCALAR0, preg);
    }
    Div(r, one, var, preg);
    Sqrt(y, r, preg);
    Muls(t, var, SCALAR1, preg);
    Mul(t, t, y, preg);
    Mula(t1, t, y, preg);
    Mul(rstd, y, t1, preg);
    Muls(t3, var, float(-1.0), preg);
    Mula(s, t3, r, preg);
    Muls(t4, rstd, float(-1.0), preg);
    Mula(r, t4, rstd, preg);
    Mula(s, var, r, preg);
    Mul(s, s, rstd, preg);
    Mula(rstd, s, scalar1, preg);
    CompareScalar(cmpRegZero, var, POS_INF, preg);
    Select(rstd, scalarZero, rstd, cmpRegZero);
    CompareScalar(cmpRegInf, var, float(0.0), preg);
    Select(rstd, scalarInf, rstd, cmpRegInf);
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphson(__local_mem__ float* src, __local_mem__ float* dst, uint32_t rowCount,
                                                float epsilon, float avgFactor = 1.0f, uint32_t vectorLen = V_LENGTH)
{
    uint16_t loopRows = static_cast<uint16_t>((rowCount + vectorLen - 1) / vectorLen);
    __VEC_SCOPE__
    {
        RegTensor<float> var;
        RegTensor<float> rstd;
        MaskReg pregLoop;

        uint32_t sreg = rowCount;
        for (uint16_t i = 0; i < loopRows; ++i) {
            pregLoop = UpdateMask<float>(sreg);
            DataCopy(var, src + i * vectorLen);
            if constexpr (NEED_AVG_FACTOR) {
                Muls(var, var, avgFactor, pregLoop);
            }
            ComputeRstdNewtonRaphsonReg<NEED_MAX>(var, rstd, pregLoop, epsilon);
            DataCopy(dst + i * vectorLen, rstd, pregLoop);
        }
    }
}

template <bool NEED_MAX = true, bool NEED_AVG_FACTOR = false>
__aicore__ inline void ComputeRstdNewtonRaphson(LocalTensor<float> srcLocal, LocalTensor<float> dstLocal,
                                                uint32_t rowCount, float epsilon, float avgFactor = 1.0f,
                                                uint32_t vectorLen = V_LENGTH)
{
    __local_mem__ float* src = (__local_mem__ float*)srcLocal.GetPhyAddr();
    __local_mem__ float* dst = (__local_mem__ float*)dstLocal.GetPhyAddr();
    ComputeRstdNewtonRaphson<NEED_MAX, NEED_AVG_FACTOR>(src, dst, rowCount, epsilon, avgFactor, vectorLen);
}

namespace NormCommonRegbase {

template <typename T, LoadDist FLOAT_LOAD_DIST = LoadDist::DIST_NORM,
          LoadDist NON_FLOAT_LOAD_DIST = LoadDist::DIST_UNPACK_B16>
__aicore__ inline void LoadRegForDtype(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, FLOAT_LOAD_DIST>(dst, src + offset);
    } else {
        RegTensor<T> srcReg;
        DataCopy<T, NON_FLOAT_LOAD_DIST>(srcReg, src + offset);
        Cast<float, T, castTraitB162B32>(dst, srcReg, preg);
    }
}

template <typename T, StoreDist FLOAT_STORE_DIST = StoreDist::DIST_NORM,
          StoreDist NON_FLOAT_STORE_DIST = StoreDist::DIST_PACK_B32>
__aicore__ inline void StoreRegForDtype(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, FLOAT_STORE_DIST>(dst + offset, src, preg);
    } else {
        RegTensor<T> dstReg;
        Cast<T, float, castTraitB322B16>(dstReg, src, preg);
        DataCopy<T, NON_FLOAT_STORE_DIST>(dst + offset, dstReg, preg);
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSumLessThanVL(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                          uint16_t rows, uint32_t rowStride, uint32_t reduceNum)
{
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> sumReg;
        MaskReg pregLoop = UpdateMask<float>(reduceNum);
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        for (uint16_t i = 0; i < rows; ++i) {
            LoadRegForDtype<T>(xPtr, xReg, pregLoop, static_cast<uint32_t>(i) * rowStride);
            Mul(xReg, xReg, xReg, pregLoop);
            ReduceSum(sumReg, xReg, pregLoop);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, sumReg, pregOne);
        }
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                             uint16_t rows, uint32_t rowStride, uint32_t reduceNum)
{
    uint32_t tailLen = reduceNum - V_LENGTH;
    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregTail = UpdateMask<float>(tailLen);
        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            LoadRegForDtype<T>(xPtr, xReg, pregFull, baseOffset);
            LoadRegForDtype<T>(xPtr + V_LENGTH, xFoldReg, pregTail, baseOffset);
            Mul(xReg, xReg, xReg, pregFull);
            Mul(xFoldReg, xFoldReg, xFoldReg, pregTail);
            ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                       pregTail);
            Add(sumReg, xReg, xFoldReg, pregFull);
            ReduceSum(reduceReg, sumReg, pregFull);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
        }
    }
}

template <typename T, int32_t LAST_LOOP_NUMS>
__aicore__ inline void CalculateSquareReduceSumCommon(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                      __local_mem__ float* tmpPtr, uint16_t rows, uint32_t rowStride,
                                                      uint32_t reduceNum, uint32_t foldPoint, uint32_t tmpStride)
{
    uint16_t foldLoops = static_cast<uint16_t>((foldPoint + V_LENGTH - 1) / V_LENGTH);
    uint32_t lastNum = foldPoint / V_LENGTH;
    uint32_t tail = (reduceNum > foldPoint) ? reduceNum - foldPoint : 0;
    uint16_t tailCeilLoops = static_cast<uint16_t>((tail + V_LENGTH - 1) / V_LENGTH);
    uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>(foldLoops - tailCeilLoops);

    __VEC_SCOPE__
    {
        RegTensor<float> xReg;
        RegTensor<float> xFoldReg;
        RegTensor<float> sumReg;
        RegTensor<float> reduceReg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
        MaskReg pregLoop;

        for (uint16_t i = 0; i < rows; ++i) {
            uint32_t baseOffset = static_cast<uint32_t>(i) * rowStride;
            uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
            uint32_t sregTail = tail;
            for (uint16_t j = 0; j < tailCeilLoops; ++j) {
                pregLoop = UpdateMask<float>(sregTail);
                uint32_t offset = static_cast<uint32_t>(j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                LoadRegForDtype<T>(xPtr + foldPoint, xFoldReg, pregFull, offset);
                Mul(xFoldReg, xFoldReg, xFoldReg, pregLoop);
                Add(sumReg, xReg, xFoldReg, pregFull);
                ReduceSum(reduceReg, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + j, reduceReg, pregOne);
            }
            for (uint16_t j = 0; j < firstFlodWithOutAddLoops; ++j) {
                uint32_t offset = static_cast<uint32_t>(tailCeilLoops + j) * V_LENGTH + baseOffset;
                LoadRegForDtype<T>(xPtr, xReg, pregFull, offset);
                Mul(xReg, xReg, xReg, pregFull);
                ReduceSum(reduceReg, xReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(tmpPtr + tmpOffset + tailCeilLoops + j, reduceReg,
                                                                   pregOne);
            }
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        if constexpr (LAST_LOOP_NUMS == 1) {
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                DataCopy(xReg, tmpPtr + static_cast<uint32_t>(i) * tmpStride);
                ReduceSum(reduceReg, xReg, pregLast);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        } else if constexpr (LAST_LOOP_NUMS == DICHOTOMY_ADD_COEFF) {
            lastNum -= V_LENGTH;
            MaskReg pregLast = UpdateMask<float>(lastNum);
            for (uint16_t i = 0; i < rows; ++i) {
                uint32_t tmpOffset = static_cast<uint32_t>(i) * tmpStride;
                DataCopy(xReg, tmpPtr + tmpOffset);
                DataCopy(xFoldReg, tmpPtr + tmpOffset + V_LENGTH);
                ShiftLefts((RegTensor<uint32_t>&)xFoldReg, (RegTensor<uint32_t>&)xFoldReg, static_cast<int16_t>(0),
                           pregLast);
                Add(sumReg, xReg, xFoldReg, pregFull);
                ReduceSum(reduceReg, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(dstPtr + i, reduceReg, pregOne);
            }
        }
    }
}

template <typename T>
// Squares input values inside this function, then reduces each row.
__aicore__ inline void CalculateSquareReduceSum(__local_mem__ T* xPtr, __local_mem__ float* dstPtr,
                                                __local_mem__ float* tmpPtr, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t tmpStride,
                                                uint32_t branchNum = 0)
{
    uint32_t reduceBranchNum = branchNum == 0 ? reduceNum : branchNum;
    if (reduceBranchNum <= V_LENGTH) {
        CalculateSquareReduceSumLessThanVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH + V_LENGTH) {
        CalculateSquareReduceSumLessThanTwoVL<T>(xPtr, dstPtr, rows, rowStride, reduceNum);
    } else if (reduceBranchNum <= V_LENGTH * V_LENGTH * DICHOTOMY_ADD_COEFF) {
        CalculateSquareReduceSumCommon<T, 1>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride);
    } else {
        CalculateSquareReduceSumCommon<T, DICHOTOMY_ADD_COEFF>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum,
                                                               foldPoint, tmpStride);
    }
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(LocalTensor<T>& xLocal, LocalTensor<float>& dstLocal,
                                                LocalTensor<float>& tmpLocal, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t blockAlign,
                                                uint32_t branchNum = 0)
{
    __local_mem__ T* xPtr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ float* dstPtr = (__local_mem__ float*)dstLocal.GetPhyAddr();
    __local_mem__ float* tmpPtr = (__local_mem__ float*)tmpLocal.GetPhyAddr();
    uint32_t foldLoops = (foldPoint + V_LENGTH - 1) / V_LENGTH;
    uint32_t tmpStride = (foldLoops + blockAlign - 1) / blockAlign * blockAlign;
    CalculateSquareReduceSum<T>(xPtr, dstPtr, tmpPtr, rows, rowStride, reduceNum, foldPoint, tmpStride, branchNum);
}

template <typename T>
__aicore__ inline void CalculateSquareReduceSum(LocalTensor<T>& xLocal, LocalTensor<float>& dstLocal,
                                                TBuf<TPosition::VECCALC>& tmpBuf, uint16_t rows, uint32_t rowStride,
                                                uint32_t reduceNum, uint32_t foldPoint, uint32_t blockAlign,
                                                uint32_t branchNum = 0)
{
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    CalculateSquareReduceSum<T>(xLocal, dstLocal, tmpLocal, rows, rowStride, reduceNum, foldPoint, blockAlign,
                                branchNum);
}
} // namespace NormCommonRegbase
} // namespace NormCommon

#endif // dav-3510 Reg device scope

#endif // ADD_RMS_NORM_BIAS_VF_REDUCE_COMMON_H
