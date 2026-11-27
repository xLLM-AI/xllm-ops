/**
 * Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file sigmoid_mul.h
 * \brief out = sigmoid(x1) * x2, bf16, flat elementwise, software-pipelined.
 */
#ifndef SIGMOID_MUL_H_
#define SIGMOID_MUL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SigmoidMulKernel {
using namespace AscendC;

constexpr int64_t kPipeDepth = 2;
// AscendC::Sigmoid needs a scratch buffer for the dividend, and it loops internally
// when that buffer is smaller than calCount. Keep it fixed: sizing it with the tile
// would grow the UB budget with the tile size for no gain.
constexpr int64_t kSigmoidTmpElems = 2048;

template <typename T>
class KernelSigmoidMul {
public:
    __aicore__ inline KernelSigmoidMul() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR out,
                                const SigmoidMulTilingData& tiling)
    {
        int64_t total = tiling.totalElements;
        int64_t perCore = tiling.elemsPerCore;
        tileLen_ = tiling.tileElems;

        int64_t start = static_cast<int64_t>(GetBlockIdx()) * perCore;
        int64_t end = start + perCore;
        if (start > total) {
            start = total;
        }
        if (end > total) {
            end = total;
        }
        len_ = end - start;

        gmX1_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1) + start, len_);
        gmX2_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2) + start, len_);
        gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out) + start, len_);

        pipe_.InitBuffer(inQue1_, kPipeDepth, tileLen_ * sizeof(T));
        pipe_.InitBuffer(inQue2_, kPipeDepth, tileLen_ * sizeof(T));
        pipe_.InitBuffer(outQue_, kPipeDepth, tileLen_ * sizeof(T));
        pipe_.InitBuffer(sigmoidBuf_, tileLen_ * sizeof(float));
        pipe_.InitBuffer(yBuf_, tileLen_ * sizeof(float));
        int64_t sigmoidTmpElems = tileLen_ < kSigmoidTmpElems ? tileLen_ : kSigmoidTmpElems;
        pipe_.InitBuffer(sigmoidTmpBuf_, sigmoidTmpElems * sizeof(float));
    }

    // One tile ahead: the MTE2 copy of tile t+1 overlaps the vector work of tile t,
    // and tile t's MTE3 store overlaps tile t+1's vector work.
    __aicore__ inline void Process()
    {
        if (len_ <= 0) {
            return;
        }
        int64_t tileCount = (len_ + tileLen_ - 1) / tileLen_;
        CopyIn(0);
        for (int64_t t = 0; t < tileCount; ++t) {
            if (t + 1 < tileCount) {
                CopyIn(t + 1);
            }
            Compute(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline int64_t TileLength(int64_t t) const
    {
        int64_t remain = len_ - t * tileLen_;
        return remain < tileLen_ ? remain : tileLen_;
    }

    __aicore__ inline void CopyIn(int64_t t)
    {
        int64_t offset = t * tileLen_;
        int64_t count = TileLength(t);
        LocalTensor<T> a = inQue1_.AllocTensor<T>();
        LocalTensor<T> b = inQue2_.AllocTensor<T>();
        if (count == tileLen_) {
            DataCopy(a, gmX1_[offset], static_cast<int32_t>(count));
            DataCopy(b, gmX2_[offset], static_cast<int32_t>(count));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
            DataCopyPad(a, gmX1_[offset], params, noPad);
            DataCopyPad(b, gmX2_[offset], params, noPad);
        }
        inQue1_.EnQue(a);
        inQue2_.EnQue(b);
    }

    __aicore__ inline void Compute(int64_t t)
    {
        uint32_t count = static_cast<uint32_t>(TileLength(t));
        LocalTensor<T> a = inQue1_.DeQue<T>();
        LocalTensor<T> b = inQue2_.DeQue<T>();
        LocalTensor<float> sigmoid = sigmoidBuf_.Get<float>();
        LocalTensor<float> y = yBuf_.Get<float>();

        // x2 is only needed by the final multiply, so lift it to fp32 up front and
        // hand its VECIN slot back before the sigmoid sequence runs.
        Cast(y, b, RoundMode::CAST_NONE, count);
        inQue2_.FreeTensor(b);

        // sigmoid(x1) in fp32. Use the library entry point rather than a hand-rolled
        // 1/(1+exp(-x)) so the rounding matches the reference op exactly.
        Cast(sigmoid, a, RoundMode::CAST_NONE, count);
        inQue1_.FreeTensor(a);
        Sigmoid<float, true>(sigmoid, sigmoid, sigmoidTmpBuf_.Get<uint8_t>(), count);
        Mul(y, y, sigmoid, count);

        LocalTensor<T> o = outQue_.AllocTensor<T>();
        Cast(o, y, RoundMode::CAST_RINT, count);
        outQue_.EnQue(o);
    }

    __aicore__ inline void CopyOut(int64_t t)
    {
        int64_t offset = t * tileLen_;
        int64_t count = TileLength(t);
        LocalTensor<T> o = outQue_.DeQue<T>();
        if (count == tileLen_) {
            DataCopy(gmOut_[offset], o, static_cast<int32_t>(count));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(gmOut_[offset], o, params);
        }
        outQue_.FreeTensor(o);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kPipeDepth> inQue1_;
    TQue<QuePosition::VECIN, kPipeDepth> inQue2_;
    TQue<QuePosition::VECOUT, kPipeDepth> outQue_;
    TBuf<QuePosition::VECCALC> sigmoidBuf_;
    TBuf<QuePosition::VECCALC> sigmoidTmpBuf_;
    TBuf<QuePosition::VECCALC> yBuf_;
    GlobalTensor<T> gmX1_;
    GlobalTensor<T> gmX2_;
    GlobalTensor<T> gmOut_;
    int64_t len_ = 0;
    int32_t tileLen_ = 0;
};
} // namespace SigmoidMulKernel

#endif // SIGMOID_MUL_H_
