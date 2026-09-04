/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Row-owned split-D Reg VF implementation for dav-3510.
 * One AIV owns every element and output of a row; R is only tiled inside UB.
 */
#ifndef ADD_RMS_NORM_BIAS_SPLIT_D_VF_H_
#define ADD_RMS_NORM_BIAS_SPLIT_D_VF_H_

#include "kernel_operator.h"
#include "vf_reduce_common.h"

namespace AddRmsNormBias {
using namespace AscendC;
using AscendC::Reg::LoadDist;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using RmsNorm::DataCopyCustom;

template <typename T, bool HAS_BETA>
class KernelAddRmsNormBiasSplitDVF {
    static constexpr uint32_t VL_FP32 = 256U / sizeof(float);

public:
    __aicore__ inline explicit KernelAddRmsNormBiasSplitDVF(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR rstd,
                                GM_ADDR x, const AddRMSNormBiasTilingData* tiling)
    {
        numRow_ = tiling->num_row;
        numCol_ = tiling->num_col;
        blockFactor_ = tiling->block_factor;
        ubFactor_ = tiling->ub_factor;
        epsilon_ = tiling->epsilon;
        blockIdx_ = GetBlockIdx();
        rowWork_ = blockIdx_ + 1U < GetBlockNum() ? blockFactor_
                                                  : numRow_ - (GetBlockNum() - 1U) * blockFactor_;
        const uint64_t rowBase = static_cast<uint64_t>(blockIdx_) * blockFactor_ * numCol_;
        x1Gm_.SetGlobalBuffer((__gm__ T*)x1 + rowBase, static_cast<uint64_t>(rowWork_) * numCol_);
        x2Gm_.SetGlobalBuffer((__gm__ T*)x2 + rowBase, static_cast<uint64_t>(rowWork_) * numCol_);
        gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, numCol_);
        if constexpr (HAS_BETA) {
            betaGm_.SetGlobalBuffer((__gm__ T*)beta, numCol_);
        }
        yGm_.SetGlobalBuffer((__gm__ T*)y + rowBase, static_cast<uint64_t>(rowWork_) * numCol_);
        xOutGm_.SetGlobalBuffer((__gm__ T*)x + rowBase, static_cast<uint64_t>(rowWork_) * numCol_);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + static_cast<uint64_t>(blockIdx_) * blockFactor_, rowWork_);

        pipe_->InitBuffer(xQueue_, 1, 2U * ubFactor_ * sizeof(T));
        pipe_->InitBuffer(paramQueue_, 1, (HAS_BETA ? 2U : 1U) * ubFactor_ * sizeof(T));
        pipe_->InitBuffer(xOutQueue_, 1, ubFactor_ * sizeof(T));
        pipe_->InitBuffer(yQueue_, 1, ubFactor_ * sizeof(T));
        pipe_->InitBuffer(reduceBuf_, VL_FP32 * sizeof(float));
        pipe_->InitBuffer(rstdBuf_, VL_FP32 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t chunkCount = (numCol_ + ubFactor_ - 1U) / ubFactor_;
        for (uint32_t row = 0; row < rowWork_; ++row) {
            LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();
            LocalTensor<float> rstdLocal = rstdBuf_.Get<float>();
            ZeroScalarVF(reduceLocal);
            for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
                const uint32_t valid = MinU32(ubFactor_, numCol_ - chunk * ubFactor_);
                StatisticChunkVF(row, chunk, valid, reduceLocal);
            }
            // avgFactor is already applied before every per-chunk ReduceSum, matching the legacy split-D order.
            NormCommon::ComputeRstdNewtonRaphson<true, true>(reduceLocal, rstdLocal, 1, epsilon_, 1.0f, VL_FP32);
            DataCopyExtParams rstdParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(rstdGm_[row], rstdLocal, rstdParams);
            for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
                const uint32_t valid = MinU32(ubFactor_, numCol_ - chunk * ubFactor_);
                AffineChunkVF(row, chunk, valid, rstdLocal);
            }
        }
    }

private:
    __aicore__ inline static uint32_t MinU32(uint32_t a, uint32_t b) { return a < b ? a : b; }

    __aicore__ inline void CopyInPair(LocalTensor<T>& pair, uint32_t row, uint32_t chunk, uint32_t valid)
    {
        const uint64_t offset = static_cast<uint64_t>(row) * numCol_ + static_cast<uint64_t>(chunk) * ubFactor_;
        DataCopyCustom<T>(pair, x1Gm_[offset], valid);
        DataCopyCustom<T>(pair[ubFactor_], x2Gm_[offset], valid);
    }

    __aicore__ inline void StatisticChunkVF(uint32_t row, uint32_t chunk, uint32_t valid,
                                            LocalTensor<float>& accumulated)
    {
        LocalTensor<T> pair = xQueue_.AllocTensor<T>();
        CopyInPair(pair, row, chunk, valid);
        xQueue_.EnQue(pair);
        pair = xQueue_.DeQue<T>();
        LocalTensor<T> xOut = xOutQueue_.AllocTensor<T>();
        __local_mem__ T* x1Ptr = (__local_mem__ T*)pair.GetPhyAddr();
        __local_mem__ T* x2Ptr = x1Ptr + ubFactor_;
        __local_mem__ T* xOutPtr = (__local_mem__ T*)xOut.GetPhyAddr();
        __local_mem__ float* accPtr = (__local_mem__ float*)accumulated.GetPhyAddr();
        const float avg = numCol_ == 0U ? 0.0f : 1.0f / static_cast<float>(numCol_);
        const uint16_t loops = static_cast<uint16_t>((valid + VL_FP32 - 1U) / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg, x2Reg, xReg, squareReg, partReg, totalReg;
            MaskReg mask, oneMask;
            uint32_t one = 1;
            oneMask = AscendC::Reg::UpdateMask<float>(one);
            AscendC::Reg::Duplicate(totalReg, 0.0f);
            uint32_t remain = valid;
            for (uint16_t i = 0; i < loops; ++i) {
                const uint32_t offset = static_cast<uint32_t>(i) * VL_FP32;
                mask = AscendC::Reg::UpdateMask<float>(remain);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(x1Ptr, x1Reg, mask, offset);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(x2Ptr, x2Reg, mask, offset);
                AscendC::Reg::Add(xReg, x1Reg, x2Reg, mask);
                NormCommon::NormCommonRegbase::StoreRegForDtype<T>(xOutPtr, xReg, mask, offset);
                AscendC::Reg::Mul(squareReg, xReg, xReg, mask);
                AscendC::Reg::Muls(squareReg, squareReg, avg, mask);
                AscendC::Reg::ReduceSum(partReg, squareReg, mask);
                AscendC::Reg::Add(totalReg, totalReg, partReg, oneMask);
            }
            AscendC::Reg::DataCopy(partReg, accPtr);
            AscendC::Reg::Add(totalReg, totalReg, partReg, oneMask);
            AscendC::Reg::DataCopy(accPtr, totalReg, oneMask);
        }
        xQueue_.FreeTensor(pair);
        xOutQueue_.EnQue(xOut);
        xOut = xOutQueue_.DeQue<T>();
        const uint64_t offset = static_cast<uint64_t>(row) * numCol_ + static_cast<uint64_t>(chunk) * ubFactor_;
        DataCopyCustom<T>(xOutGm_[offset], xOut, valid);
        xOutQueue_.FreeTensor(xOut);
    }

    __aicore__ inline void AffineChunkVF(uint32_t row, uint32_t chunk, uint32_t valid,
                                         LocalTensor<float>& rstdLocal)
    {
        const uint64_t offset = static_cast<uint64_t>(row) * numCol_ + static_cast<uint64_t>(chunk) * ubFactor_;
        // Recompute x from its immutable inputs instead of reading xOutGm back immediately after MTE3.
        // This preserves row ownership and removes a cross-pipe GM RAW hazard without SyncAll/workspace.
        LocalTensor<T> pair = xQueue_.AllocTensor<T>();
        CopyInPair(pair, row, chunk, valid);
        xQueue_.EnQue(pair);
        pair = xQueue_.DeQue<T>();
        LocalTensor<T> params = paramQueue_.AllocTensor<T>();
        DataCopyCustom<T>(params, gammaGm_[static_cast<uint64_t>(chunk) * ubFactor_], valid);
        if constexpr (HAS_BETA) {
            DataCopyCustom<T>(params[ubFactor_], betaGm_[static_cast<uint64_t>(chunk) * ubFactor_], valid);
        }
        paramQueue_.EnQue(params);
        params = paramQueue_.DeQue<T>();
        LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
        __local_mem__ T* x1Ptr = (__local_mem__ T*)pair.GetPhyAddr();
        __local_mem__ T* x2Ptr = x1Ptr + ubFactor_;
        __local_mem__ T* gammaPtr = (__local_mem__ T*)params.GetPhyAddr();
        __local_mem__ T* betaPtr = gammaPtr + ubFactor_;
        __local_mem__ T* yPtr = (__local_mem__ T*)yLocal.GetPhyAddr();
        __local_mem__ float* rstdPtr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        const uint16_t loops = static_cast<uint16_t>((valid + VL_FP32 - 1U) / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1Reg, x2Reg, xReg, gammaReg, betaReg, rstdReg, tmpReg;
            MaskReg mask;
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdPtr);
            uint32_t remain = valid;
            for (uint16_t i = 0; i < loops; ++i) {
                const uint32_t elemOffset = static_cast<uint32_t>(i) * VL_FP32;
                mask = AscendC::Reg::UpdateMask<float>(remain);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(x1Ptr, x1Reg, mask, elemOffset);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(x2Ptr, x2Reg, mask, elemOffset);
                AscendC::Reg::Add(xReg, x1Reg, x2Reg, mask);
                NormCommon::NormCommonRegbase::LoadRegForDtype<T>(gammaPtr, gammaReg, mask, elemOffset);
                AscendC::Reg::Mul(tmpReg, xReg, rstdReg, mask);
                AscendC::Reg::Mul(tmpReg, tmpReg, gammaReg, mask);
                if constexpr (HAS_BETA) {
                    NormCommon::NormCommonRegbase::LoadRegForDtype<T>(betaPtr, betaReg, mask, elemOffset);
                    AscendC::Reg::Add(tmpReg, tmpReg, betaReg, mask);
                }
                NormCommon::NormCommonRegbase::StoreRegForDtype<T>(yPtr, tmpReg, mask, elemOffset);
            }
        }
        xQueue_.FreeTensor(pair);
        paramQueue_.FreeTensor(params);
        yQueue_.EnQue(yLocal);
        yLocal = yQueue_.DeQue<T>();
        DataCopyCustom<T>(yGm_[offset], yLocal, valid);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void ZeroScalarVF(LocalTensor<float>& tensor)
    {
        __local_mem__ float* ptr = (__local_mem__ float*)tensor.GetPhyAddr();
        __VEC_SCOPE__
        {
            RegTensor<float> zero;
            MaskReg oneMask;
            uint32_t one = 1;
            oneMask = AscendC::Reg::UpdateMask<float>(one);
            AscendC::Reg::Duplicate(zero, 0.0f);
            AscendC::Reg::DataCopy(ptr, zero, oneMask);
        }
    }

    TPipe* pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> paramQueue_;
    TQue<QuePosition::VECOUT, 1> xOutQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<TPosition::VECCALC> reduceBuf_;
    TBuf<TPosition::VECCALC> rstdBuf_;
    GlobalTensor<T> x1Gm_, x2Gm_, gammaGm_, betaGm_, yGm_, xOutGm_;
    GlobalTensor<float> rstdGm_;
    uint32_t numRow_{0}, numCol_{0}, blockFactor_{0}, ubFactor_{0};
    uint32_t blockIdx_{0}, rowWork_{0};
    float epsilon_{0.0f};
};
}  // namespace AddRmsNormBias

#endif  // ADD_RMS_NORM_BIAS_SPLIT_D_VF_H_
