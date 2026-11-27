/*!
 * \file sigmoid_mul.h
 * \brief SigmoidMul operator kernel class (arch22).
 *
 * out = x * sigmoid(gate); gate is [B, W] (element-wise) or [.., 1] (row-broadcast).
 * bf16/fp16 inputs are promoted to fp32 for the sigmoid + mul math, then narrowed
 * back to T, because the V220 vector arithmetic unit only accepts half/float.
 */
#ifndef SIGMOID_MUL_H
#define SIGMOID_MUL_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sigmoid_mul_tiling_data.h"
#include "sigmoid_mul_tiling_key.h"
#include "adv_api/activation/sigmoid.h"

namespace NsSigmoidMul {

using namespace AscendC;

template <typename T, int MODE>
class SigmoidMul {
    static constexpr int32_t BUFFER_NUM = 2;

public:
    __aicore__ inline SigmoidMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gate, GM_ADDR y, const SigmoidMulTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessFlat();
    __aicore__ inline void ProcessBroadcast();
    __aicore__ inline void CopyInFlat(int64_t progress, int64_t currentNum);
    __aicore__ inline void ComputeFlat(int64_t currentNum);
    __aicore__ inline void CopyOutFlat(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyInChunk(int64_t offset, int64_t currentNum);
    __aicore__ inline void ComputeChunk(float gateScalar, int64_t currentNum);
    __aicore__ inline void CopyOutChunk(int64_t offset, int64_t currentNum);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueueX;
    TQue<QuePosition::VECIN, 1> inputQueueG;
    TQue<QuePosition::VECOUT, 1> outputQueueZ;
    TBuf<QuePosition::VECCALC> x32Buf_;
    TBuf<QuePosition::VECCALC> g32Buf_;
    TBuf<QuePosition::VECCALC> z32Buf_;
    TBuf<QuePosition::VECCALC> sigGateBuf_;  // broadcast: sigmoid(gate[0..gateVecLen_)) as fp32
    TBuf<QuePosition::VECCALC> sigRowBuf_;   // broadcast: materialized sigmoid chunk (chunk fp32)

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMG;
    GlobalTensor<T> outputGMZ;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t rowWidth_ = 0;
    int64_t rows_ = 0;
    int64_t gateVecLen_ = 0;  // rows_ padded up to a vector-safe multiple of 64
    int64_t rowsPerCore_ = 0;
    int64_t rowStart_ = 0;
    int64_t coreRows_ = 0;    // number of gate rows owned by this core
    bool isBroadcast_ = false;
};

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::Init(GM_ADDR x, GM_ADDR gate, GM_ADDR y,
    const SigmoidMulTilingData* tilingData)
{
    isBroadcast_ = (tilingData->isBroadcast != 0);
    rowWidth_ = tilingData->rowWidth;
    ubLength_ = tilingData->ubFactor;
    const int64_t coreIdx = AscendC::GetBlockIdx();
    const int64_t coreNum = AscendC::GetBlockNum();

    if constexpr (MODE == 1) {
        rows_ = tilingData->totalNum / tilingData->rowWidth;
        rowsPerCore_ = (rows_ + coreNum - 1) / coreNum;
        rowStart_ = coreIdx * rowsPerCore_;
        const int64_t rowEnd = (rowStart_ + rowsPerCore_ > rows_) ? rows_ : (rowStart_ + rowsPerCore_);
        coreRows_ = (rowEnd > rowStart_) ? (rowEnd - rowStart_) : 0;  // idle cores own 0 rows
        blockLength_ = coreRows_ * rowWidth_;
        inputGMX.SetGlobalBuffer((__gm__ T*)x + rowStart_ * rowWidth_, blockLength_);
        inputGMG.SetGlobalBuffer((__gm__ T*)gate, rows_);
        outputGMZ.SetGlobalBuffer((__gm__ T*)y + rowStart_ * rowWidth_, blockLength_);
        // one row is processed in chunks of ubLength_ (<= rowWidth_) to fit UB
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(x32Buf_, BUFFER_NUM * ubLength_ * sizeof(float));
        pipe.InitBuffer(sigRowBuf_, BUFFER_NUM * ubLength_ * sizeof(float));
        pipe.InitBuffer(z32Buf_, BUFFER_NUM * ubLength_ * sizeof(float));
        gateVecLen_ = ((coreRows_ + 127) / 128) * 128;  // vector-safe count for this core's gate slice
        if (gateVecLen_ < 128) {
            gateVecLen_ = 128;  // keep InitBuffer sizes valid even for idle cores
        }
        pipe.InitBuffer(inputQueueG, 1, gateVecLen_ * sizeof(T));
        pipe.InitBuffer(sigGateBuf_, gateVecLen_ * sizeof(float));
    } else {
        const int64_t total = tilingData->totalNum;
        const int64_t offset = coreIdx * tilingData->blockFactor;
        const int64_t remainder = total - offset;
        blockLength_ = (remainder > tilingData->blockFactor) ? tilingData->blockFactor : remainder;
        inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, blockLength_);
        inputGMG.SetGlobalBuffer((__gm__ T*)gate + offset, blockLength_);
        outputGMZ.SetGlobalBuffer((__gm__ T*)y + offset, blockLength_);
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(inputQueueG, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueZ, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(x32Buf_, BUFFER_NUM * ubLength_ * sizeof(float));
        pipe.InitBuffer(g32Buf_, BUFFER_NUM * ubLength_ * sizeof(float));
        pipe.InitBuffer(z32Buf_, BUFFER_NUM * ubLength_ * sizeof(float));
    }
}

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::Process()
{
    if constexpr (MODE == 1) {
        ProcessBroadcast();
    } else {
        ProcessFlat();
    }
}

// ---------------------------------------------------------------------------
// Flat mode: gate has the same element count as x.
// ---------------------------------------------------------------------------
template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::ProcessFlat()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyInFlat(i, currentNum);
        ComputeFlat(currentNum);
        CopyOutFlat(i, currentNum);
    }
}

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::CopyInFlat(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::LocalTensor<T> gLocal = inputQueueG.template AllocTensor<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    AscendC::DataCopyPad(gLocal, inputGMG[progress * ubLength_], copyParams, {false, 0, 0, static_cast<T>(0)});
    inputQueueX.EnQue(xLocal);
    inputQueueG.EnQue(gLocal);
}

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::ComputeFlat(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> gLocal = inputQueueG.template DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template AllocTensor<T>();
    AscendC::LocalTensor<float> x32 = x32Buf_.template AllocTensor<float>();
    AscendC::LocalTensor<float> g32 = g32Buf_.template AllocTensor<float>();
    AscendC::LocalTensor<float> z32 = z32Buf_.template AllocTensor<float>();

    if constexpr (std::is_same_v<T, float>) {
        // fp32 inputs need no promotion/cast.
        AscendC::Sigmoid(gLocal, gLocal, currentNum);
        AscendC::Mul(zLocal, xLocal, gLocal, currentNum);
    } else {
        AscendC::Cast(x32, xLocal, RoundMode::CAST_NONE, currentNum);
        AscendC::Cast(g32, gLocal, RoundMode::CAST_NONE, currentNum);
        AscendC::Sigmoid(g32, g32, currentNum);
        AscendC::Mul(z32, x32, g32, currentNum);
        AscendC::Cast(zLocal, z32, RoundMode::CAST_RINT, currentNum);
    }

    outputQueueZ.template EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueG.FreeTensor(gLocal);
    x32Buf_.FreeTensor(x32);
    g32Buf_.FreeTensor(g32);
    z32Buf_.FreeTensor(z32);
}

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::CopyOutFlat(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template DeQue<T>();
    AscendC::DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;
    AscendC::DataCopyPad(outputGMZ[progress * ubLength_], zLocal, copyParams);
    outputQueueZ.FreeTensor(zLocal);
}

// ---------------------------------------------------------------------------
// Broadcast mode: gate is [.., 1]; each core owns a set of rows.
// ---------------------------------------------------------------------------
template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::ProcessBroadcast()
{
    const int64_t coreIdx = AscendC::GetBlockIdx();
    if (coreRows_ <= 0) {
        return;  // this core owns no rows (more cores than rows)
    }

    // Precompute sigmoid of this core's gate slice [rowStart_, rowStart_+coreRows_)
    // once into fp32 using a TQue-backed source (same pattern as the flat path).
    AscendC::LocalTensor<T> gateAll = inputQueueG.template AllocTensor<T>();
    AscendC::DataCopyExtParams gp;
    gp.blockCount = 1;
    gp.blockLen = coreRows_ * sizeof(T);
    gp.srcStride = 0;
    gp.dstStride = 0;
    gp.rsv = 0;
    AscendC::DataCopyPad(gateAll, inputGMG[rowStart_], gp, {false, 0, 0, static_cast<T>(0)});
    inputQueueG.EnQue(gateAll);
    AscendC::LocalTensor<T> gateIn = inputQueueG.template DeQue<T>();
    AscendC::LocalTensor<float> sigGate = sigGateBuf_.template AllocTensor<float>();
    if constexpr (std::is_same_v<T, float>) {
        AscendC::DataCopy(sigGate, gateIn, static_cast<uint32_t>(gateVecLen_));
    } else {
        AscendC::Cast(sigGate, gateIn, RoundMode::CAST_NONE, gateVecLen_);
    }
    AscendC::Sigmoid(sigGate, sigGate, gateVecLen_);
    // V -> S sync so scalar reads below observe the sigmoid result.
    event_t eid = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));
    AscendC::SetFlag<AscendC::HardEvent::V_S>(eid);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(eid);
    inputQueueG.FreeTensor(gateIn);

    for (int64_t r = 0; r < rowsPerCore_; r++) {
        const int64_t row = coreIdx * rowsPerCore_ + r;
        if (row >= rows_) {
            break;
        }
        const float gateScalar = sigGate.GetValue(static_cast<uint32_t>(row - rowStart_));
        const int64_t localRowBase = (row - rowStart_) * rowWidth_;
        const int64_t chunks = (rowWidth_ + ubLength_ - 1) / ubLength_;
        for (int64_t c = 0; c < chunks; c++) {
            const int64_t currentNum = (c == (chunks - 1)) ? (rowWidth_ - ubLength_ * c) : ubLength_;
            const int64_t offset = localRowBase + c * ubLength_;
            CopyInChunk(offset, currentNum);
            ComputeChunk(gateScalar, currentNum);
            CopyOutChunk(offset, currentNum);
        }
    }
}

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::CopyInChunk(int64_t offset, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
    AscendC::DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = currentNum * sizeof(T);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[offset], p, {false, 0, 0, static_cast<T>(0)});
    inputQueueX.EnQue(xLocal);
}

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::ComputeChunk(float gateScalar, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template AllocTensor<T>();
    AscendC::LocalTensor<float> x32 = x32Buf_.template AllocTensor<float>();
    AscendC::LocalTensor<float> sig32 = sigRowBuf_.template AllocTensor<float>();
    AscendC::LocalTensor<float> z32 = z32Buf_.template AllocTensor<float>();

    if constexpr (std::is_same_v<T, float>) {
        AscendC::Duplicate(sig32, gateScalar, currentNum);
        AscendC::Mul(zLocal, xLocal, sig32, currentNum);
    } else {
        AscendC::Cast(x32, xLocal, RoundMode::CAST_NONE, currentNum);
        AscendC::Duplicate(sig32, gateScalar, currentNum);
        AscendC::Mul(z32, x32, sig32, currentNum);
        AscendC::Cast(zLocal, z32, RoundMode::CAST_RINT, currentNum);
    }

    outputQueueZ.template EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    x32Buf_.FreeTensor(x32);
    sigRowBuf_.FreeTensor(sig32);
    z32Buf_.FreeTensor(z32);
}

template <typename T, int MODE>
__aicore__ inline void SigmoidMul<T, MODE>::CopyOutChunk(int64_t offset, int64_t currentNum)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.template DeQue<T>();
    AscendC::DataCopyExtParams p;
    p.blockCount = 1;
    p.blockLen = currentNum * sizeof(T);
    p.srcStride = 0;
    p.dstStride = 0;
    p.rsv = 0;
    AscendC::DataCopyPad(outputGMZ[offset], zLocal, p);
    outputQueueZ.FreeTensor(zLocal);
}

} // namespace NsSigmoidMul
#endif // SIGMOID_MUL_H
