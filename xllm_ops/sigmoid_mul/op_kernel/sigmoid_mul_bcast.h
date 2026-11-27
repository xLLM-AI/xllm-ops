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
 * \file sigmoid_mul_bcast.h
 * \brief out[r][c] = sigmoid(x1[r]) * x2[r][c] for x1 [M, 1], x2 [M, N], bf16.
 */
#ifndef SIGMOID_MUL_BCAST_H_
#define SIGMOID_MUL_BCAST_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SigmoidMulBcastKernel {
using namespace AscendC;

constexpr int64_t kPipeDepth = 2;
// Rows whose sigmoid is computed in one vector pass before the row loop consumes them.
constexpr int64_t kRowChunk = 512;

template <typename T>
class KernelSigmoidMulBcast {
public:
    __aicore__ inline KernelSigmoidMulBcast() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR out,
                                const SigmoidMulTilingData& tiling)
    {
        rows_ = tiling.rows;
        cols_ = tiling.cols;
        tileLen_ = tiling.tileElems;

        int64_t perCore = tiling.rowsPerCore;
        int64_t start = static_cast<int64_t>(GetBlockIdx()) * perCore;
        int64_t end = start + perCore;
        rowStart_ = start < rows_ ? start : rows_;
        rowEnd_ = end < rows_ ? end : rows_;
        if (rowEnd_ < rowStart_) {
            rowEnd_ = rowStart_;
        }

        gmX1_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1), rows_);
        gmX2_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2), rows_ * cols_);
        gmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), rows_ * cols_);

        pipe_.InitBuffer(x1Que_, 1, kRowChunk * sizeof(T));
        pipe_.InitBuffer(sigBuf_, kRowChunk * sizeof(float));
        pipe_.InitBuffer(x2Que_, kPipeDepth, tileLen_ * sizeof(T));
        pipe_.InitBuffer(outQue_, kPipeDepth, tileLen_ * sizeof(T));
        pipe_.InitBuffer(x2fBuf_, tileLen_ * sizeof(float));
    }

    // A tile never spans rows, so the tile sequence is (row, column-tile) flattened and
    // the copy of the next tile - possibly the first tile of the next row - runs ahead
    // of the vector work of the current one.
    __aicore__ inline void Process()
    {
        if (rowEnd_ <= rowStart_ || cols_ <= 0) {
            return;
        }
        int64_t tilesPerRow = (cols_ + tileLen_ - 1) / tileLen_;
        for (int64_t row0 = rowStart_; row0 < rowEnd_; row0 += kRowChunk) {
            int64_t chunkRows = rowEnd_ - row0 < kRowChunk ? rowEnd_ - row0 : kRowChunk;
            LoadRowSigmoid(row0, chunkRows);

            int64_t tileCount = chunkRows * tilesPerRow;
            CopyIn(0, row0);
            for (int64_t k = 0; k < tileCount; ++k) {
                if (k + 1 < tileCount) {
                    CopyIn(k + 1, row0);
                }
                Compute(k, tilesPerRow);
                CopyOut(k, row0);
            }
        }
    }

private:
    __aicore__ inline int64_t TileOffset(int64_t k, int64_t row0, int64_t tilesPerRow) const
    {
        int64_t row = row0 + k / tilesPerRow;
        return (row * cols_) + ((k % tilesPerRow) * tileLen_);
    }

    __aicore__ inline int64_t TileLength(int64_t k, int64_t tilesPerRow) const
    {
        int64_t off = (k % tilesPerRow) * tileLen_;
        int64_t remain = cols_ - off;
        return remain < tileLen_ ? remain : tileLen_;
    }

    // sigmoid of the chunk's rows into sigBuf_, so the row loop only reads a scalar.
    __aicore__ inline void LoadRowSigmoid(int64_t row0, int64_t chunkRows)
    {
        LocalTensor<T> x1t = x1Que_.AllocTensor<T>();
        if (chunkRows == kRowChunk) {
            DataCopy(x1t, gmX1_[row0], static_cast<int32_t>(chunkRows));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(chunkRows * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
            DataCopyPad(x1t, gmX1_[row0], params, noPad);
        }
        x1Que_.EnQue(x1t);

        LocalTensor<T> in = x1Que_.DeQue<T>();
        LocalTensor<float> sig = sigBuf_.Get<float>();
        uint32_t n = static_cast<uint32_t>(chunkRows);
        Cast(sig, in, RoundMode::CAST_NONE, n);
        x1Que_.FreeTensor(in);
        Muls(sig, sig, static_cast<float>(-1.0), n);
        Exp(sig, sig, static_cast<int32_t>(n));
        Adds(sig, sig, static_cast<float>(1.0), n);
        Reciprocal(sig, sig, n);

        // The row loop reads these through the scalar unit.
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    __aicore__ inline void CopyIn(int64_t k, int64_t row0)
    {
        int64_t tilesPerRow = (cols_ + tileLen_ - 1) / tileLen_;
        int64_t offset = TileOffset(k, row0, tilesPerRow);
        int64_t count = TileLength(k, tilesPerRow);

        LocalTensor<T> b = x2Que_.AllocTensor<T>();
        if (count == tileLen_) {
            DataCopy(b, gmX2_[offset], static_cast<int32_t>(count));
        } else {
            DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> noPad{false, 0, 0, 0};
            DataCopyPad(b, gmX2_[offset], params, noPad);
        }
        x2Que_.EnQue(b);
    }

    __aicore__ inline void Compute(int64_t k, int64_t tilesPerRow)
    {
        int64_t count = TileLength(k, tilesPerRow);
        float s = sigBuf_.Get<float>().GetValue(static_cast<uint32_t>(k / tilesPerRow));

        LocalTensor<T> b = x2Que_.DeQue<T>();
        LocalTensor<float> f = x2fBuf_.Get<float>();
        Cast(f, b, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        x2Que_.FreeTensor(b);
        Muls(f, f, s, static_cast<uint32_t>(count));

        LocalTensor<T> o = outQue_.AllocTensor<T>();
        Cast(o, f, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
        outQue_.EnQue(o);
    }

    __aicore__ inline void CopyOut(int64_t k, int64_t row0)
    {
        int64_t tilesPerRow = (cols_ + tileLen_ - 1) / tileLen_;
        int64_t offset = TileOffset(k, row0, tilesPerRow);
        int64_t count = TileLength(k, tilesPerRow);

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
    TQue<QuePosition::VECIN, 1> x1Que_;
    TQue<QuePosition::VECIN, kPipeDepth> x2Que_;
    TQue<QuePosition::VECOUT, kPipeDepth> outQue_;
    TBuf<QuePosition::VECCALC> sigBuf_;
    TBuf<QuePosition::VECCALC> x2fBuf_;
    GlobalTensor<T> gmX1_;
    GlobalTensor<T> gmX2_;
    GlobalTensor<T> gmOut_;
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    int64_t rowStart_ = 0;
    int64_t rowEnd_ = 0;
    int32_t tileLen_ = 0;
};
} // namespace SigmoidMulBcastKernel

#endif // SIGMOID_MUL_BCAST_H_
