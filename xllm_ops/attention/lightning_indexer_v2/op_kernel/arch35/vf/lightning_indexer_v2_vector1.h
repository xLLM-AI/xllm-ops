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
 * \file lightning_indexer_v2_vector1.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_VECTOR1_H
#define LIGHTNING_INDEXER_V2_VECTOR1_H

#include "kernel_operator.h"
#include "common/lightning_indexer_v2_vector1_base.h"

namespace liV2Vector1 {

__aicore__ inline void UIntToFloatReturnValue(const LocalTensor<float> &out_,
                                              const LocalTensor<uint32_t> &in,
                                              const uint32_t topK,
                                              const uint32_t negInfBits)
{
    auto outBuf = (__local_mem__ float*)out_.GetPhyAddr();
    auto inBuf = (__local_mem__ uint32_t*)in.GetPhyAddr();

    const uint16_t repeatSize32 = 128;
    uint16_t topkLoopNum = (topK + repeatSize32 - 1) / repeatSize32;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<uint32_t> regIn[2];
        AscendC::MicroAPI::RegTensor<float> regOut[2];
        AscendC::MicroAPI::RegTensor<uint32_t> regNegInf;
        AscendC::MicroAPI::RegTensor<uint32_t> regZero;
        AscendC::MicroAPI::MaskReg maskInvalid[2];
        AscendC::MicroAPI::MaskReg maskAllB32 =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::Duplicate(regNegInf, negInfBits, maskAllB32);
        AscendC::MicroAPI::Duplicate(regZero, (uint32_t)0, maskAllB32);

        UIntSortConstCtx<float> uint32Ctx;
        InitUIntSortConstCtx(uint32Ctx, maskAllB32);
        for (uint16_t i = 0; i < topkLoopNum; ++i) {
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[0], inBuf + i * repeatSize32);
            AscendC::MicroAPI::LoadAlign<uint32_t>(regIn[1], inBuf + i * repeatSize32 + 64);

            MicroAPI::Compare<uint32_t, CMPMODE::EQ>(maskInvalid[0], regIn[0], regZero, maskAllB32);
            MicroAPI::Compare<uint32_t, CMPMODE::EQ>(maskInvalid[1], regIn[1], regZero, maskAllB32);

            UIntToSortableKey<float>(regOut[0], regIn[0], uint32Ctx, maskAllB32);
            UIntToSortableKey<float>(regOut[1], regIn[1], uint32Ctx, maskAllB32);

            MicroAPI::Select((AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[0], regNegInf,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[0], maskInvalid[0]);
            MicroAPI::Select((AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[1], regNegInf,
                    (AscendC::MicroAPI::RegTensor<uint32_t>&)regOut[1], maskInvalid[1]);
            AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(outBuf + i * repeatSize32,
                                                                                          regOut[0],
                                                                                          maskAllB32);
            AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(
                outBuf + i * repeatSize32 + 64,
                regOut[1],
                maskAllB32);
        }
    }
}

// uint32 out
__simd_callee__ inline void ReduceSumFinalize(AscendC::MicroAPI::RegTensor<float> (&regSum0)[2],
                                          AscendC::MicroAPI::RegTensor<float> (&regSum1)[2],
                                          AscendC::MicroAPI::MaskReg& maskAllB32,
                                          FloatSortConstCtx<float>& fp32Ctx,
                                          __ubuf__ uint32_t* out_)
{
    AscendC::MicroAPI::Add(regSum0[0], regSum0[0], regSum1[0], maskAllB32);
    AscendC::MicroAPI::Add(regSum0[1], regSum0[1], regSum1[1], maskAllB32);
    AscendC::MicroAPI::RegTensor<uint32_t> regOut[2];
    FloatX2ToSortableKey<float>(regOut[0], regOut[1], regSum0[0], regSum0[1], fp32Ctx, maskAllB32);

    AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out_, regOut[0], maskAllB32);
    AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out_ + 64, regOut[1], maskAllB32);
}

__simd_callee__ inline void ReduceSum2Finalize(AscendC::MicroAPI::RegTensor<float> (&regSum0)[2],
                                           AscendC::MicroAPI::RegTensor<float> (&regSum1)[2],
                                           AscendC::MicroAPI::MaskReg& maskAllB32,
                                           FloatSortConstCtx<float>& fp32Ctx,
                                           __ubuf__ uint32_t* out0_,
                                           __ubuf__ uint32_t* out1_)
{
    AscendC::MicroAPI::RegTensor<uint32_t> regOut0[2];
    AscendC::MicroAPI::RegTensor<uint32_t> regOut1[2];

    FloatX2ToSortableKey<float>(regOut0[0], regOut0[1], regSum0[0], regSum0[1], fp32Ctx, maskAllB32);
    FloatX2ToSortableKey<float>(regOut1[0], regOut1[1], regSum1[0], regSum1[1], fp32Ctx, maskAllB32);
    AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0_, regOut0[0], maskAllB32);
    AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out0_ + 64, regOut0[1],
        maskAllB32);
    AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1_, regOut1[0], maskAllB32);
    AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_NORM>(out1_ + 64, regOut1[1],
        maskAllB32);
}

template <typename QK_T>
__simd_callee__ inline void ReduceSumLoopBodyEven(AscendC::MicroAPI::RegTensor<float> (&regQK)[2],
                                          AscendC::MicroAPI::RegTensor<float>& regwBrc,
                                          AscendC::MicroAPI::RegTensor<float>& regW,
                                          AscendC::MicroAPI::RegTensor<float> (&regSum0)[2],
                                          AscendC::MicroAPI::RegTensor<float> (&regSum1)[2],
                                          AscendC::MicroAPI::MaskReg& maskAllB32,
                                          __ubuf__ QK_T* qk_,
                                          const uint32_t qkVLStride,
                                          const int gSize)
{
    constexpr static MicroAPI::CastTrait castTraitInt32ToFP32 = {MicroAPI::RegLayout::UNKNOWN,
        MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + qkVLStride);

        BroadcastLane(regwBrc, regW, i);
        WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + 128 + qkVLStride);

        BroadcastLane(regwBrc, regW, i + 1);
        WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }
}

template <typename QK_T>
__simd_callee__ inline void ReduceSumLoopBodyOdd(AscendC::MicroAPI::RegTensor<float> (&regQK)[2],
                                          AscendC::MicroAPI::RegTensor<float>& regwBrc,
                                          AscendC::MicroAPI::RegTensor<float>& regW,
                                          AscendC::MicroAPI::RegTensor<float> (&regSum0)[2],
                                          AscendC::MicroAPI::RegTensor<float> (&regSum1)[2],
                                          AscendC::MicroAPI::MaskReg& maskAllB32,
                                          __ubuf__ QK_T* qk_,
                                          const uint32_t qkVLStride,
                                          const int gSize)
{
    constexpr static MicroAPI::CastTrait castTraitInt32ToFP32 = {MicroAPI::RegLayout::UNKNOWN,
        MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    uint16_t i = 0;
    for (; i + 1 < (uint16_t)(gSize); i += 2) {
        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + qkVLStride);

        BroadcastLane(regwBrc, regW, i);
        WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);

        MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * i + 128);
        MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * i + 128 + qkVLStride);

        BroadcastLane(regwBrc, regW, i + 1);
        WeightedAccum(regSum1, regQK, regwBrc, maskAllB32);
    }

    const uint16_t tailIdx = (uint16_t)(gSize -1);
    MicroAPI::LoadAlign<float>(regQK[0], qk_ + 128 * tailIdx);
    MicroAPI::LoadAlign<float>(regQK[1], qk_ + 128 * tailIdx + qkVLStride);
    BroadcastLane(regwBrc, regW, tailIdx);
    WeightedAccum(regSum0, regQK, regwBrc, maskAllB32);
}

template <typename QK_T>
__simd_callee__ inline void ReduceSum2LoopBody(AscendC::MicroAPI::RegTensor<float> (&regQK0)[2],
                                           AscendC::MicroAPI::RegTensor<float> (&regQK1)[2],
                                           AscendC::MicroAPI::RegTensor<float> (&regwBrc)[2],
                                           AscendC::MicroAPI::RegTensor<float> (&regW)[2],
                                           AscendC::MicroAPI::RegTensor<float> (&regSum0)[2],
                                           AscendC::MicroAPI::RegTensor<float> (&regSum1)[2],
                                           AscendC::MicroAPI::MaskReg& maskAllB32,
                                           __ubuf__ QK_T* qk0_,
                                           __ubuf__ QK_T* qk1_,
                                           const uint32_t qkVLStride,
                                           __ubuf__ float* brcWeight_,
                                           const int gSize)
{
    constexpr static MicroAPI::CastTrait castTraitInt32ToFP32 = {MicroAPI::RegLayout::UNKNOWN,
        MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

    for (uint16_t i = (uint16_t)(0); i < (uint16_t)(gSize); i++) {
        MicroAPI::LoadAlign<float>(regQK0[0], qk0_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK0[1], qk0_ + 128 * i + qkVLStride);
        MicroAPI::LoadAlign<float>(regQK1[0], qk1_ + 128 * i);
        MicroAPI::LoadAlign<float>(regQK1[1], qk1_ + 128 * i + qkVLStride);

        BroadcastLane(regwBrc[0], regW[0], i);
        BroadcastLane(regwBrc[1], brcWeight_, i);

        AscendC::MicroAPI::Relu(regQK0[0], regQK0[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK0[1], regQK0[1], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[0], regQK1[0], maskAllB32);
        AscendC::MicroAPI::Relu(regQK1[1], regQK1[1], maskAllB32);

        AscendC::MicroAPI::MulAddDst(regSum0[0], regQK0[0], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum0[1], regQK0[1], regwBrc[0], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[0], regQK1[0], regwBrc[1], maskAllB32);
        AscendC::MicroAPI::MulAddDst(regSum1[1], regQK1[1], regwBrc[1], maskAllB32);
    }
}

template <typename QK_T>
__simd_callee__ inline void MulWeightAndReduceSumEvenVF(__ubuf__ uint32_t* out_,
                                          __ubuf__ QK_T* qk_,
                                          const uint32_t qkVLStride,
                                          __ubuf__ float* weight_,
                                          const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc;
    AscendC::MicroAPI::RegTensor<float> regQK[2];
    AscendC::MicroAPI::RegTensor<float> regW;
    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<float> fp32Ctx;
    InitFloatSortConstCtx(fp32Ctx, maskAllB32);

    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(regW, weight_);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    ReduceSumLoopBodyEven<QK_T>(regQK, regwBrc, regW, regSum0, regSum1, maskAllB32,
        qk_, qkVLStride, gSize);

    ReduceSumFinalize(regSum0, regSum1, maskAllB32, fp32Ctx, out_);
}

template <typename QK_T>
__simd_callee__ inline void MulWeightAndReduceSumOddVF(__ubuf__ uint32_t* out_,
                                          __ubuf__ QK_T* qk_,
                                          const uint32_t qkVLStride,
                                          __ubuf__ float* weight_,
                                          const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc;
    AscendC::MicroAPI::RegTensor<float> regQK[2];
    AscendC::MicroAPI::RegTensor<float> regW;
    AscendC::MicroAPI::RegTensor<float> regSum0[2];
    AscendC::MicroAPI::RegTensor<float> regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<float> fp32Ctx;
    InitFloatSortConstCtx(fp32Ctx, maskAllB32);

    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(regW, weight_);

    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    ReduceSumLoopBodyOdd<QK_T>(regQK, regwBrc, regW, regSum0, regSum1, maskAllB32,
        qk_, qkVLStride, gSize);

    ReduceSumFinalize(regSum0, regSum1, maskAllB32, fp32Ctx, out_);
}

// float in uint32 out
template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum2(__ubuf__ uint32_t* out0_,
                                           __ubuf__ uint32_t* out1_,
                                           uint32_t outStride,
                                           __ubuf__ QK_T* qk0_,
                                           __ubuf__ QK_T* qk1_,
                                           uint32_t qkVLStride,
                                           uint32_t qkStride,
                                           __ubuf__ float* weight0_,
                                           __ubuf__ float* weight1_,
                                           uint32_t weightStride,
                                           const int gSize)
{
    AscendC::MicroAPI::RegTensor<float> regwBrc[2];
    AscendC::MicroAPI::RegTensor<float> regQK0[2], regQK1[2];
    AscendC::MicroAPI::RegTensor<float> regW[2];

    AscendC::MicroAPI::RegTensor<float> regSum0[2], regSum1[2];
    AscendC::MicroAPI::MaskReg maskAllB32 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

    FloatSortConstCtx<float> fp32Ctx;
    InitFloatSortConstCtx(fp32Ctx, maskAllB32);

    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(regW[0], weight0_);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(regW[1], weight1_);

    // regW[0]与weight1混合使用
    AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM>(weight1_, regW[1], maskAllB32);
    AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
    DuplicateZero(regSum0, maskAllB32);
    DuplicateZero(regSum1, maskAllB32);

    ReduceSum2LoopBody<QK_T>(regQK0, regQK1, regwBrc, regW,
        regSum0, regSum1, maskAllB32, qk0_, qk1_, qkVLStride, weight1_, gSize);

    ReduceSum2Finalize(regSum0, regSum1, maskAllB32, fp32Ctx, out0_, out1_);
}


template <typename QK_T>
__simd_vf__ inline void MulWeightAndReduceSum(__ubuf__ uint32_t* out_,
                                              __ubuf__ QK_T* qk_,
                                              const uint32_t qkVLStride,
                                              __ubuf__ float* weight_,
                                              const int gSize)
{
    if (gSize % 2 == 0) {
        MulWeightAndReduceSumEvenVF<QK_T>(out_, qk_, qkVLStride, weight_, gSize);
    } else {
        MulWeightAndReduceSumOddVF<QK_T>(out_, qk_, qkVLStride, weight_, gSize);
    }
}

template<typename QK_T, typename W_T, typename SCORE_T>
__aicore__ inline void BatchMulWeightAndReduceSum(const LocalTensor<SCORE_T> &out_, // out [S2Base] [128 ]
                                              uint32_t outStride,
                                              const LocalTensor<QK_T> &qk_, // q*k^t [G, S2Base] [64 128]
                                              uint32_t qkVLStride,
                                              uint32_t qkStride,
                                              const LocalTensor<W_T> &weight_, // w [G] [64 ]
                                              uint32_t weightStride,
                                              const int gSize, // G 64
                                              const int row)
{
    // 暂只支持这两种情况, 后续改成循环
    if (row != 2 && row != 1) {
        return;
    }
    auto weight = (__ubuf__ W_T *)weight_.GetPhyAddr();
    auto qk = (__ubuf__ QK_T *)qk_.GetPhyAddr();
    auto out = (__ubuf__ SCORE_T *)out_.GetPhyAddr();

    if (row == 2) {
        auto weight1 = weight + weightStride;
        auto qk1 = qk + qkStride;
        auto out1 = out + outStride;

        MulWeightAndReduceSum2(out, out1, outStride,
                            qk, qk1, qkVLStride, qkStride,
                            weight, weight1, weightStride,
                            gSize);
    } else {
        MulWeightAndReduceSum(out, qk, qkVLStride, weight, gSize);
    }
}
}

#endif