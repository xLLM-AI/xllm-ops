/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from the actual add_custom/op_kernel/add_custom_kernel.asc.
// Retains TPipe/TQue CopyIn -> Compute -> CopyOut; each core processes full rows.
#pragma once
#include <cstdint>

// Contract cap, independent of hardware capacity; host also validates actual UB.
constexpr uint32_t ADA_RMS_NORM_MAX_D = 8192;
// Conservative software reservation for ASC MemBase intrinsic scratch.
constexpr uint64_t ADA_RMS_NORM_UB_RESERVE = 32 * 1024;
constexpr uint32_t ADA_RMS_NORM_ALIGNMENT = 32;
constexpr uint32_t ADA_RMS_NORM_SCALAR_BYTES = 32;

// Kernel-local parameters decoded from the registered CANN tiling data.
// Broadcast strides are in elements.
struct AdaRmsNormData {
    uint64_t rows;
    uint64_t rowsPerCore;
    uint64_t seqLength;
    uint64_t scaleBatchStride;
    uint64_t scaleSeqStride;
    uint64_t shiftBatchStride;
    uint64_t shiftSeqStride;
    uint32_t hiddenDim;
    uint32_t rawRowBytes;
    uint32_t floatRowBytes;
    uint32_t hasWeight;
    float epsilon;
};

#if defined(__CCE_AICORE__) || defined(__NPU_ARCH__)
#include "kernel_operator.h"

template <typename T, bool IS_FLOAT>
class KernelAdaRmsNorm {
public:
    __aicore__ inline explicit KernelAdaRmsNorm(AscendC::TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift,
        GM_ADDR weight, GM_ADDR out, AdaRmsNormData tiling)
    {
        tiling_ = tiling;
        xGm_.SetGlobalBuffer((__gm__ T*)x);
        scaleGm_.SetGlobalBuffer((__gm__ T*)scale);
        shiftGm_.SetGlobalBuffer((__gm__ T*)shift);
        outGm_.SetGlobalBuffer((__gm__ T*)out);
        if (tiling_.hasWeight != 0) {
            weightGm_.SetGlobalBuffer((__gm__ T*)weight);
        }
        pipe_->InitBuffer(inQueue_, 1, tiling_.rawRowBytes);
        pipe_->InitBuffer(outQueue_, 1, tiling_.rawRowBytes);
        pipe_->InitBuffer(xFloatBuf_, tiling_.floatRowBytes);
        pipe_->InitBuffer(operandBuf_, tiling_.floatRowBytes);
        // A whole FP32 row is a conservative sharedTmpBuffer for ReduceSum.
        pipe_->InitBuffer(reduceBuf_, tiling_.floatRowBytes);
        pipe_->InitBuffer(scalarBuf_, ADA_RMS_NORM_SCALAR_BYTES);
        firstRow_ = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tiling_.rowsPerCore;
        lastRow_ = firstRow_ + tiling_.rowsPerCore;
        if (lastRow_ > tiling_.rows) {
            lastRow_ = tiling_.rows;
        }
    }

    __aicore__ inline void Process()
    {
        for (uint64_t row = firstRow_; row < lastRow_; ++row) {
            ComputeRow(row);
        }
    }

private:
    __aicore__ inline void CopyInAndPromote(
        const AscendC::GlobalTensor<T>& gm, uint64_t offset,
        const AscendC::LocalTensor<float>& dst)
    {
        auto raw = inQueue_.template AllocTensor<T>();
        AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(tiling_.hiddenDim * sizeof(T)), 0, 0, 0};
        // API pads to the transfer block. All compute counts use D, excluding padding.
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(raw, gm[offset], copy, pad);
        inQueue_.EnQue(raw);
        auto ready = inQueue_.template DeQue<T>();
        if constexpr (IS_FLOAT) {
            AscendC::Adds(dst, ready, 0.0f, static_cast<int32_t>(tiling_.hiddenDim));
        } else {
            AscendC::Cast(dst, ready, AscendC::RoundMode::CAST_NONE, tiling_.hiddenDim);
        }
        // TQue frees after its vector consumer; next AllocTensor waits before MTE2 reuse.
        inQueue_.FreeTensor(ready);
    }

    __aicore__ inline uint64_t BroadcastOffset(
        uint64_t row, uint64_t batchStride, uint64_t seqStride)
    {
        return (row / tiling_.seqLength) * batchStride +
               (row % tiling_.seqLength) * seqStride;
    }

    __aicore__ inline void ComputeRow(uint64_t row)
    {
        const int32_t count = static_cast<int32_t>(tiling_.hiddenDim);
        const uint64_t xOffset = row * tiling_.hiddenDim;
        auto xFloat = xFloatBuf_.Get<float>();
        auto operand = operandBuf_.Get<float>();
        auto reduceTmp = reduceBuf_.Get<float>();
        auto invRms = scalarBuf_.Get<float>();

        CopyInAndPromote(xGm_, xOffset, xFloat);
        // RMS only: mean(x*x); no mean subtraction and no variance computation.
        AscendC::Mul(operand, xFloat, xFloat, count);
        AscendC::ReduceSum<float, true>(invRms, operand, reduceTmp, count);
        AscendC::Muls(invRms, invRms, 1.0f / static_cast<float>(count), 1);
        AscendC::Adds(invRms, invRms, tiling_.epsilon, 1);
        AscendC::Rsqrt(invRms, invRms, 1);
        // A5 overload broadcasts the lowest source lane entirely in PIPE_V.
        // No scalar GetValue or V->S->V dependency is introduced.
        AscendC::Duplicate(operand, invRms, count);
        AscendC::Mul(xFloat, xFloat, operand, count);

        if (tiling_.hasWeight != 0) {
            CopyInAndPromote(weightGm_, 0, operand);
            AscendC::Mul(xFloat, xFloat, operand, count);
        }
        const uint64_t scaleOffset = BroadcastOffset(
            row, tiling_.scaleBatchStride, tiling_.scaleSeqStride);
        const uint64_t shiftOffset = BroadcastOffset(
            row, tiling_.shiftBatchStride, tiling_.shiftSeqStride);
        CopyInAndPromote(scaleGm_, scaleOffset, operand);
        AscendC::Adds(operand, operand, 1.0f, count);
        AscendC::Mul(xFloat, xFloat, operand, count);
        CopyInAndPromote(shiftGm_, shiftOffset, operand);
        AscendC::Add(xFloat, xFloat, operand, count);

        auto output = outQueue_.template AllocTensor<T>();
        if constexpr (IS_FLOAT) {
            AscendC::Adds(output, xFloat, 0.0f, count);
        } else {
            AscendC::Cast(output, xFloat, AscendC::RoundMode::CAST_ROUND, tiling_.hiddenDim);
        }
        outQueue_.EnQue(output);
        auto ready = outQueue_.template DeQue<T>();
        AscendC::DataCopyExtParams copy{
            1, static_cast<uint32_t>(tiling_.hiddenDim * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(outGm_[xOffset], ready, copy);
        outQueue_.FreeTensor(ready);
    }

    AscendC::TPipe* pipe_;
    AdaRmsNormData tiling_{};
    AscendC::GlobalTensor<T> xGm_, scaleGm_, shiftGm_, weightGm_, outGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_, operandBuf_, reduceBuf_, scalarBuf_;
    uint64_t firstRow_ = 0;
    uint64_t lastRow_ = 0;
};

#endif // device compilation
