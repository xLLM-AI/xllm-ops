/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

/*!
 * \file rms_norm_channels_kernel_impl.h
 * \brief Device-side kernels for RMSNorm over the channel axis of a [B, C, T, H, W] tensor.
 *
 * Ported from the validated direct-launch implementation.  The input is contiguous
 * [B, C, N] with N = T * H * W; the channel axis has stride N, so a position tile is
 * a [C, Ntile] strided matrix.  The host tiling key selects one of the shape families:
 *   1001 SMALL_C         C fits in UB, keep the whole [C, Ntile] tile (one input read).
 *   2001 SPLIT_C         Large C with small N: split C over the vector cores, write
 *                        per-block partial sums, join them after a global barrier.
 *   3001 STREAM          Large C and large N: stream C chunks twice (sum, normalize).
 *   4001 SINGLE_CHANNEL  C == 1, the reduction collapses to x*x.
 *   5001 SIMT_WARP       Very short N, one SIMT warp per output position.
 */

#ifndef RMS_NORM_CHANNELS_KERNEL_IMPL_H
#define RMS_NORM_CHANNELS_KERNEL_IMPL_H

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <type_traits>

#include "kernel_operator.h"
#include "rms_norm_channels_common.h"

namespace RmsNormChannelsKernel {


constexpr int64_t MAX_N_TILE = 512;
constexpr int64_t MAX_C_CHUNK = 256;
constexpr int64_t TILE_ALIGN = 16;
constexpr int64_t MIN_N_TILE = 64;

__aicore__ inline uint32_t AlignUp32(uint64_t bytes)
{
    return static_cast<uint32_t>((bytes + 31ULL) / 32ULL * 32ULL);
}

__aicore__ inline int64_t MinI64(int64_t a, int64_t b)
{
    return a < b ? a : b;
}

// DataCopyPad's Normal mode pads every block to a 32 B boundary.  Rows in UB
// therefore advance by this aligned element count, not by the logical length.
__aicore__ inline int64_t RowStride(int64_t n, int64_t elemBytes)
{
    return ((n * elemBytes + 31) / 32 * 32) / elemBytes;
}

template <typename T>
__aicore__ inline void LoadGammaFloat(const AscendC::LocalTensor<T> &gammaT,
    const AscendC::LocalTensor<float> &gammaF, int64_t cc)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Muls(gammaF, gammaT, 1.0f, static_cast<int32_t>(cc));
    } else {
        AscendC::Cast(gammaF, gammaT, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(cc));
    }
}

template <typename T>
__aicore__ inline void SquareAccumulateRow(const AscendC::LocalTensor<T> &row,
    const AscendC::LocalTensor<float> &xFloat, const AscendC::LocalTensor<float> &acc, int64_t n)
{
    if constexpr (std::is_same<T, float>::value) {
        // acc += row * row in one fused vector instruction.
        AscendC::MulAddDst(acc, row, row, static_cast<int32_t>(n));
    } else {
        AscendC::Cast(xFloat, row, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
        AscendC::MulAddDst(acc, xFloat, xFloat, static_cast<int32_t>(n));
    }
}

template <typename T>
__aicore__ inline void NormalizeRow(const AscendC::LocalTensor<T> &row,
    const AscendC::LocalTensor<float> &xFloat, const AscendC::LocalTensor<float> &rstd, float gammaValue,
    int64_t n)
{
    if constexpr (std::is_same<T, float>::value) {
        AscendC::Mul(row, row, rstd, static_cast<int32_t>(n));
        AscendC::Muls(row, row, gammaValue, static_cast<int32_t>(n));
    } else {
        AscendC::Cast(xFloat, row, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
        AscendC::Mul(xFloat, xFloat, rstd, static_cast<int32_t>(n));
        AscendC::Muls(xFloat, xFloat, gammaValue, static_cast<int32_t>(n));
        constexpr auto roundMode =
            std::is_same<T, bfloat16_t>::value ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
        AscendC::Cast(row, xFloat, roundMode, static_cast<uint32_t>(n));
    }
}

// ---------------------------------------------------------------------------
// SMALL_C: keep the complete [C, Ntile] tile in UB.
// ---------------------------------------------------------------------------
template <typename T>
class KernelRmsNormChannelsSmallC {
public:
    __aicore__ inline KernelRmsNormChannelsSmallC() {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR gamma, GM_ADDR out, int64_t b, int64_t c, int64_t n,
        int64_t totalTiles, int64_t tilesPerBatch, int64_t numBlocks, uint32_t nTile, float eps)
    {
        b_ = b;
        c_ = c;
        n_ = n;
        totalTiles_ = totalTiles;
        tilesPerBatch_ = tilesPerBatch;
        numBlocks_ = numBlocks;
        nTile_ = nTile;
        eps_ = eps;
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());

        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

        const uint32_t rowStride = static_cast<uint32_t>(RowStride(nTile_, sizeof(T)));
        const uint32_t tileBytes = AlignUp32(static_cast<uint64_t>(c_) * rowStride * sizeof(T));
        const uint32_t vectorBytes = AlignUp32(static_cast<uint64_t>(nTile_) * sizeof(float));
        const uint32_t gammaBytes = AlignUp32(static_cast<uint64_t>(c_) * sizeof(T));
        const uint32_t gammaFBytes = AlignUp32(static_cast<uint64_t>(c_) * sizeof(float));

        pipe_.InitBuffer(xBuf_, tileBytes);
        pipe_.InitBuffer(accBuf_, vectorBytes);
        pipe_.InitBuffer(rstdBuf_, vectorBytes);
        pipe_.InitBuffer(xFloatBuf_, vectorBytes);
        pipe_.InitBuffer(gammaBuf_, gammaBytes);
        pipe_.InitBuffer(gammaFBuf_, gammaFBytes);
    }

    __aicore__ inline void Process()
    {
        for (int64_t tile = blockIdx_; tile < totalTiles_; tile += numBlocks_) {
            const int64_t batch = tile / tilesPerBatch_;
            const int64_t n0 = (tile % tilesPerBatch_) * static_cast<int64_t>(nTile_);
            const int64_t remain = n_ - n0;
            const int64_t n = remain < static_cast<int64_t>(nTile_) ? remain : static_cast<int64_t>(nTile_);
            const int64_t base = batch * c_ * n_ + n0;
            ComputeTile(base, n);
        }
    }

private:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<T> &dst, int64_t base, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(c_), static_cast<uint32_t>(n * sizeof(T)),
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, valueGm_[base], copyParams, padParams);
    }

    __aicore__ inline void CopyInGamma(const AscendC::LocalTensor<T> &dst)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, gammaGm_[0], copyParams, padParams);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<T> &src, int64_t base, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(c_), static_cast<uint32_t>(n * sizeof(T)), 0,
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(outGm_[base], src, copyParams);
    }

    __aicore__ inline void ComputeTile(int64_t base, int64_t n)
    {
        const int64_t rowStride = RowStride(n, static_cast<int64_t>(sizeof(T)));
        auto xLocal = xBuf_.Get<T>();
        CopyIn(xLocal, base, n);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        // gammaT is reloaded for every tile: order the previous tile's vector
        // Cast (which read gammaT) before this MTE2 write.
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        auto gammaT = gammaBuf_.Get<T>();
        CopyInGamma(gammaT);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        auto gammaF = gammaFBuf_.Get<float>();
        LoadGammaFloat(gammaT, gammaF, c_);

        auto acc = accBuf_.Get<float>();
        auto xFloat = xFloatBuf_.Get<float>();
        AscendC::Duplicate(acc, 0.0f, static_cast<int32_t>(n));
        for (int64_t i = 0; i < c_; ++i) {
            const int32_t offset = static_cast<int32_t>(i * rowStride);
            SquareAccumulateRow<T>(xLocal[offset], xFloat, acc, n);
        }

        const float invC = 1.0f / static_cast<float>(c_);
        AscendC::Muls(acc, acc, invC, static_cast<int32_t>(n));
        AscendC::Adds(acc, acc, eps_, static_cast<int32_t>(n));
        auto rstd = rstdBuf_.Get<float>();
        AscendC::Rsqrt<float, AscendC::DEFAULT_RSQRT_CONFIG>(rstd, acc, static_cast<int32_t>(n));

        // gammaF was produced by a vector Cast/Muls; read scalars only after a
        // V->S fence.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        for (int64_t i = 0; i < c_; ++i) {
            const int32_t offset = static_cast<int32_t>(i * rowStride);
            const float gammaValue = gammaF.GetValue(static_cast<uint32_t>(i));
            NormalizeRow<T>(xLocal[offset], xFloat, rstd, gammaValue, n);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        CopyOut(xLocal, base, n);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> gammaGm_;
    AscendC::GlobalTensor<T> outGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rstdBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaFBuf_;

    int64_t b_ = 0;
    int64_t c_ = 0;
    int64_t n_ = 0;
    int64_t totalTiles_ = 0;
    int64_t tilesPerBatch_ = 0;
    int64_t numBlocks_ = 0;
    int64_t blockIdx_ = 0;
    uint32_t nTile_ = 0;
    float eps_ = 1e-12f;
};

// ---------------------------------------------------------------------------
// SPLIT_C phase 1: per-block partial sum of squares for its C range.

// ---------------------------------------------------------------------------
// SINGLE_CHANNEL: C == 1.  The reduction is just x*x, so a dedicated one-row
// path drops the generic C loop, the zero-initialised accumulator and the
// 1/C scaling.
// ---------------------------------------------------------------------------
template <typename T>
class KernelRmsNormChannelsSingleChannel {
public:
    __aicore__ inline KernelRmsNormChannelsSingleChannel() {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR gamma, GM_ADDR out, int64_t b, int64_t c, int64_t n,
        int64_t totalTiles, int64_t tilesPerBatch, int64_t numBlocks, uint32_t nTile, float eps)
    {
        b_ = b;
        c_ = c;
        n_ = n;
        totalTiles_ = totalTiles;
        tilesPerBatch_ = tilesPerBatch;
        numBlocks_ = numBlocks;
        nTile_ = nTile;
        eps_ = eps;
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());

        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

        pipe_.InitBuffer(xBuf_, AlignUp32(static_cast<uint64_t>(nTile_) * sizeof(T)));
        pipe_.InitBuffer(xFloatBuf_, AlignUp32(static_cast<uint64_t>(nTile_) * sizeof(float)));
        pipe_.InitBuffer(rstdBuf_, AlignUp32(static_cast<uint64_t>(nTile_) * sizeof(float)));
        pipe_.InitBuffer(gammaBuf_, 32U);
        pipe_.InitBuffer(gammaFBuf_, 32U);
    }

    __aicore__ inline void Process()
    {
        auto gammaT = gammaBuf_.Get<T>();
        CopyInGamma(gammaT);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        auto gammaF = gammaFBuf_.Get<float>();
        LoadGammaFloat(gammaT, gammaF, 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
        const float gammaValue = gammaF.GetValue(0);

        auto xFloat = xFloatBuf_.Get<float>();
        auto rstd = rstdBuf_.Get<float>();
        for (int64_t tile = blockIdx_; tile < totalTiles_; tile += numBlocks_) {
            const int64_t batch = tile / tilesPerBatch_;
            const int64_t n0 = (tile % tilesPerBatch_) * static_cast<int64_t>(nTile_);
            const int64_t remain = n_ - n0;
            const int64_t n = remain < static_cast<int64_t>(nTile_) ? remain : static_cast<int64_t>(nTile_);
            const int64_t base = batch * c_ * n_ + n0;

            auto xLocal = xBuf_.Get<T>();
            CopyIn(xLocal, base, n);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

            if constexpr (std::is_same<T, float>::value) {
                AscendC::Muls(xFloat, xLocal, 1.0f, static_cast<int32_t>(n));
            } else {
                AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
            }
            AscendC::Mul(rstd, xFloat, xFloat, static_cast<int32_t>(n));
            AscendC::Adds(rstd, rstd, eps_, static_cast<int32_t>(n));
            AscendC::Rsqrt<float, AscendC::DEFAULT_RSQRT_CONFIG>(rstd, rstd, static_cast<int32_t>(n));
            AscendC::Mul(xFloat, xFloat, rstd, static_cast<int32_t>(n));
            AscendC::Muls(xFloat, xFloat, gammaValue, static_cast<int32_t>(n));
            if constexpr (std::is_same<T, float>::value) {
                AscendC::Muls(xLocal, xFloat, 1.0f, static_cast<int32_t>(n));
            } else {
                constexpr auto roundMode = std::is_same<T, bfloat16_t>::value
                                               ? AscendC::RoundMode::CAST_RINT
                                               : AscendC::RoundMode::CAST_NONE;
                AscendC::Cast(xLocal, xFloat, roundMode, static_cast<uint32_t>(n));
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            CopyOut(xLocal, base, n);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<T> &dst, int64_t base, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(T)),
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, valueGm_[base], copyParams, padParams);
    }

    __aicore__ inline void CopyInGamma(const AscendC::LocalTensor<T> &dst)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, gammaGm_[0], copyParams, padParams);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<T> &src, int64_t base, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(n * sizeof(T)), 0,
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(outGm_[base], src, copyParams);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> gammaGm_;
    AscendC::GlobalTensor<T> outGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rstdBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaFBuf_;

    int64_t b_ = 0;
    int64_t c_ = 0;
    int64_t n_ = 0;
    int64_t totalTiles_ = 0;
    int64_t tilesPerBatch_ = 0;
    int64_t numBlocks_ = 0;
    int64_t blockIdx_ = 0;
    uint32_t nTile_ = 0;
    float eps_ = 1e-12f;
};

// workspace layout: [B][numBlocks][workspaceStride] float32
// ---------------------------------------------------------------------------
template <typename T>
class KernelRmsNormChannelsSplitSum {
public:
    __aicore__ inline KernelRmsNormChannelsSplitSum() {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR workspace, int64_t b, int64_t c, int64_t n,
        int64_t numBlocks, int64_t cPerBlock, uint32_t cChunk, uint32_t nTile, int64_t workspaceStride)
    {
        b_ = b;
        c_ = c;
        n_ = n;
        numBlocks_ = numBlocks;
        cPerBlock_ = cPerBlock;
        cChunk_ = cChunk;
        nTile_ = nTile;
        workspaceStride_ = workspaceStride;
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());

        cStart_ = blockIdx_ * cPerBlock_;
        cEnd_ = MinI64(c_, cStart_ + cPerBlock_);

        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        const uint32_t rowStride = static_cast<uint32_t>(RowStride(nTile_, sizeof(T)));
        const uint32_t tileBytes = AlignUp32(static_cast<uint64_t>(cChunk_) * rowStride * sizeof(T));
        const uint32_t vectorBytes = AlignUp32(static_cast<uint64_t>(nTile_) * sizeof(float));
        const uint32_t accBytes = AlignUp32(static_cast<uint64_t>(n_) * sizeof(float));

        pipe_.InitBuffer(xBuf_, tileBytes);
        pipe_.InitBuffer(accBuf_, accBytes);
        pipe_.InitBuffer(xFloatBuf_, vectorBytes);
    }

    __aicore__ inline void Process()
    {
        auto acc = accBuf_.Get<float>();
        for (int64_t batch = 0; batch < b_; ++batch) {
            AscendC::Duplicate(acc, 0.0f, static_cast<int32_t>(n_));
            const int64_t batchBase = batch * c_ * n_;
            for (int64_t n0 = 0; n0 < n_; n0 += static_cast<int64_t>(nTile_)) {
                const int64_t remain = n_ - n0;
                const int64_t n = remain < static_cast<int64_t>(nTile_) ? remain : static_cast<int64_t>(nTile_);
                for (int64_t c0 = cStart_; c0 < cEnd_; c0 += static_cast<int64_t>(cChunk_)) {
                    const int64_t cc = MinI64(static_cast<int64_t>(cChunk_), cEnd_ - c0);
                    auto xLocal = xBuf_.Get<T>();
                    CopyIn(xLocal, batchBase + c0 * n_ + n0, cc, n);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    Accumulate(xLocal, acc[static_cast<int32_t>(n0)], cc, n);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            const int64_t workspaceOffset =
                (batch * numBlocks_ + blockIdx_) * workspaceStride_;
            CopyOutWorkspace(workspaceGm_[workspaceOffset], acc);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<T> &dst, int64_t base, int64_t cc, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(cc), static_cast<uint32_t>(n * sizeof(T)),
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, valueGm_[base], copyParams, padParams);
    }

    __aicore__ inline void Accumulate(const AscendC::LocalTensor<T> &src,
        const AscendC::LocalTensor<float> &acc, int64_t cc, int64_t n)
    {
        const int64_t rowStride = RowStride(n, static_cast<int64_t>(sizeof(T)));
        auto xFloat = xFloatBuf_.Get<float>();
        for (int64_t i = 0; i < cc; ++i) {
            const int32_t offset = static_cast<int32_t>(i * rowStride);
            SquareAccumulateRow<T>(src[offset], xFloat, acc, n);
        }
    }

    __aicore__ inline void CopyOutWorkspace(const AscendC::GlobalTensor<float> &dst,
        const AscendC::LocalTensor<float> &src)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(dst, src, copyParams);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_;

    int64_t b_ = 0;
    int64_t c_ = 0;
    int64_t n_ = 0;
    int64_t numBlocks_ = 0;
    int64_t cPerBlock_ = 0;
    int64_t cStart_ = 0;
    int64_t cEnd_ = 0;
    int64_t workspaceStride_ = 0;
    int64_t blockIdx_ = 0;
    uint32_t cChunk_ = 0;
    uint32_t nTile_ = 0;
};

// ---------------------------------------------------------------------------
// SPLIT_C phase 2: join partials and normalize this block's C range.
// ---------------------------------------------------------------------------
template <typename T>
class KernelRmsNormChannelsSplitNorm {
public:
    __aicore__ inline KernelRmsNormChannelsSplitNorm() {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR gamma, GM_ADDR out, GM_ADDR workspace, int64_t b,
        int64_t c, int64_t n, int64_t numBlocks, int64_t cPerBlock, uint32_t cChunk, uint32_t nTile,
        int64_t workspaceStride, float eps)
    {
        b_ = b;
        c_ = c;
        n_ = n;
        numBlocks_ = numBlocks;
        cPerBlock_ = cPerBlock;
        cChunk_ = cChunk;
        nTile_ = nTile;
        workspaceStride_ = workspaceStride;
        eps_ = eps;
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());

        cStart_ = blockIdx_ * cPerBlock_;
        cEnd_ = MinI64(c_, cStart_ + cPerBlock_);
        gammaCount_ = cEnd_ > cStart_ ? cEnd_ - cStart_ : 0;

        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

        const uint32_t rowStride = static_cast<uint32_t>(RowStride(nTile_, sizeof(T)));
        const uint32_t partialStride = static_cast<uint32_t>(RowStride(nTile_, sizeof(float)));
        const uint32_t tileBytes = AlignUp32(static_cast<uint64_t>(cChunk_) * rowStride * sizeof(T));
        const uint32_t partialBytes = AlignUp32(static_cast<uint64_t>(numBlocks_) * partialStride * sizeof(float));
        const uint32_t vectorBytes = AlignUp32(static_cast<uint64_t>(nTile_) * sizeof(float));
        const uint32_t gammaBytes = AlignUp32(static_cast<uint64_t>(gammaCount_) * sizeof(T) + 32ULL);
        const uint32_t gammaFBytes = AlignUp32(static_cast<uint64_t>(gammaCount_) * sizeof(float) + 32ULL);

        pipe_.InitBuffer(xBuf_, tileBytes);
        pipe_.InitBuffer(partialBuf_, partialBytes);
        pipe_.InitBuffer(accBuf_, vectorBytes);
        pipe_.InitBuffer(rstdBuf_, vectorBytes);
        pipe_.InitBuffer(xFloatBuf_, vectorBytes);
        pipe_.InitBuffer(gammaBuf_, gammaBytes);
        pipe_.InitBuffer(gammaFBuf_, gammaFBytes);
    }

    __aicore__ inline void Process()
    {
        auto gammaF = gammaFBuf_.Get<float>();
        if (gammaCount_ > 0) {
            auto gammaT = gammaBuf_.Get<T>();
            CopyInGamma(gammaT);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            LoadGammaFloat(gammaT, gammaF, gammaCount_);
        }

        auto acc = accBuf_.Get<float>();
        auto rstd = rstdBuf_.Get<float>();
        auto xFloat = xFloatBuf_.Get<float>();
        for (int64_t batch = 0; batch < b_; ++batch) {
            const int64_t batchBase = batch * c_ * n_;
            for (int64_t n0 = 0; n0 < n_; n0 += static_cast<int64_t>(nTile_)) {
                const int64_t remain = n_ - n0;
                const int64_t n = remain < static_cast<int64_t>(nTile_) ? remain : static_cast<int64_t>(nTile_);
                const int64_t partialStride = RowStride(n, static_cast<int64_t>(sizeof(float)));
                auto partial = partialBuf_.Get<float>();
                CopyInPartials(partial, (batch * numBlocks_) * workspaceStride_ + n0, n);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                AscendC::Duplicate(acc, 0.0f, static_cast<int32_t>(n));
                for (int64_t j = 0; j < numBlocks_; ++j) {
                    const int32_t offset = static_cast<int32_t>(j * partialStride);
                    AscendC::Add(acc, acc, partial[offset], static_cast<int32_t>(n));
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

                const float invC = 1.0f / static_cast<float>(c_);
                AscendC::Muls(acc, acc, invC, static_cast<int32_t>(n));
                AscendC::Adds(acc, acc, eps_, static_cast<int32_t>(n));
                AscendC::Rsqrt<float, AscendC::DEFAULT_RSQRT_CONFIG>(rstd, acc, static_cast<int32_t>(n));

                AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);

                for (int64_t c0 = cStart_; c0 < cEnd_; c0 += static_cast<int64_t>(cChunk_)) {
                    const int64_t cc = MinI64(static_cast<int64_t>(cChunk_), cEnd_ - c0);
                    auto xLocal = xBuf_.Get<T>();
                    CopyIn(xLocal, batchBase + c0 * n_ + n0, cc, n);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                    const int64_t rowStride = RowStride(n, static_cast<int64_t>(sizeof(T)));
                    const int64_t rowIndex = c0 - cStart_;
                    for (int64_t i = 0; i < cc; ++i) {
                        const int32_t offset = static_cast<int32_t>(i * rowStride);
                        const float gammaValue = gammaF.GetValue(static_cast<uint32_t>(rowIndex + i));
                        NormalizeRow<T>(xLocal[offset], xFloat, rstd, gammaValue, n);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                    CopyOut(xLocal, batchBase + c0 * n_ + n0, cc, n);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                }
            }
        }
    }

private:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<T> &dst, int64_t base, int64_t cc, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(cc), static_cast<uint32_t>(n * sizeof(T)),
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, valueGm_[base], copyParams, padParams);
    }

    __aicore__ inline void CopyInPartials(const AscendC::LocalTensor<float> &dst, int64_t base, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(numBlocks_), static_cast<uint32_t>(n * sizeof(float)),
            static_cast<int64_t>((workspaceStride_ - n) * sizeof(float)), 0, 0};
        AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<float, AscendC::PaddingMode::Normal>(dst, workspaceGm_[base], copyParams, padParams);
    }

    __aicore__ inline void CopyInGamma(const AscendC::LocalTensor<T> &dst)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(gammaCount_ * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, gammaGm_[cStart_], copyParams, padParams);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<T> &src, int64_t base, int64_t cc, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(cc), static_cast<uint32_t>(n * sizeof(T)), 0,
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(outGm_[base], src, copyParams);
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> gammaGm_;
    AscendC::GlobalTensor<T> outGm_;
    AscendC::GlobalTensor<float> workspaceGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> partialBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rstdBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaFBuf_;

    int64_t b_ = 0;
    int64_t c_ = 0;
    int64_t n_ = 0;
    int64_t numBlocks_ = 0;
    int64_t cPerBlock_ = 0;
    int64_t cStart_ = 0;
    int64_t cEnd_ = 0;
    int64_t gammaCount_ = 0;
    int64_t workspaceStride_ = 0;
    int64_t blockIdx_ = 0;
    uint32_t cChunk_ = 0;
    uint32_t nTile_ = 0;
    float eps_ = 1e-12f;
};

// ---------------------------------------------------------------------------
// STREAM: large C, large N.  Two streaming passes over C.
// ---------------------------------------------------------------------------
template <typename T>
class KernelRmsNormChannelsStream {
public:
    __aicore__ inline KernelRmsNormChannelsStream() {}

    __aicore__ inline void Init(GM_ADDR value, GM_ADDR gamma, GM_ADDR out, int64_t b, int64_t c, int64_t n,
        int64_t totalTiles, int64_t tilesPerBatch, int64_t numBlocks, uint32_t nTile, uint32_t cChunk, float eps)
    {
        b_ = b;
        c_ = c;
        n_ = n;
        totalTiles_ = totalTiles;
        tilesPerBatch_ = tilesPerBatch;
        numBlocks_ = numBlocks;
        nTile_ = nTile;
        cChunk_ = cChunk;
        eps_ = eps;
        blockIdx_ = static_cast<int64_t>(AscendC::GetBlockIdx());

        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(value));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(gamma));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out));

        const uint32_t rowStride = static_cast<uint32_t>(RowStride(nTile_, sizeof(T)));
        const uint32_t tileBytes = AlignUp32(static_cast<uint64_t>(cChunk_) * rowStride * sizeof(T));
        const uint32_t gammaBytes = AlignUp32(static_cast<uint64_t>(cChunk_) * sizeof(T));
        const uint32_t vectorBytes = AlignUp32(static_cast<uint64_t>(nTile_) * sizeof(float));
        const uint32_t gammaFBytes = AlignUp32(static_cast<uint64_t>(cChunk_) * sizeof(float));

        pipe_.InitBuffer(inQueue_, 1, tileBytes);
        pipe_.InitBuffer(outQueue_, 1, tileBytes);
        pipe_.InitBuffer(gammaQueue_, 1, gammaBytes);
        pipe_.InitBuffer(xFloatBuf_, vectorBytes);
        pipe_.InitBuffer(accBuf_, vectorBytes);
        pipe_.InitBuffer(rstdBuf_, vectorBytes);
        pipe_.InitBuffer(gammaFBuf_, gammaFBytes);
    }

    __aicore__ inline void Process()
    {
        for (int64_t tile = blockIdx_; tile < totalTiles_; tile += numBlocks_) {
            const int64_t batch = tile / tilesPerBatch_;
            const int64_t n0 = (tile % tilesPerBatch_) * static_cast<int64_t>(nTile_);
            const int64_t remain = n_ - n0;
            const int64_t n = remain < static_cast<int64_t>(nTile_) ? remain : static_cast<int64_t>(nTile_);
            const int64_t base = batch * c_ * n_ + n0;
            ComputeTile(base, n);
        }
    }

private:
    __aicore__ inline void CopyIn(const AscendC::LocalTensor<T> &dst, int64_t base, int64_t c0, int64_t cc, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(cc), static_cast<uint32_t>(n * sizeof(T)),
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(
            dst, valueGm_[base + c0 * n_], copyParams, padParams);
    }

    __aicore__ inline void CopyInGamma(const AscendC::LocalTensor<T> &dst, int64_t c0, int64_t cc)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(1), static_cast<uint32_t>(cc * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(dst, gammaGm_[c0], copyParams, padParams);
    }

    __aicore__ inline void CopyOut(const AscendC::LocalTensor<T> &src, int64_t base, int64_t c0, int64_t cc, int64_t n)
    {
        AscendC::DataCopyExtParams copyParams{
            static_cast<uint16_t>(cc), static_cast<uint32_t>(n * sizeof(T)), 0,
            static_cast<int64_t>((n_ - n) * sizeof(T)), 0};
        AscendC::DataCopyPad<T, AscendC::PaddingMode::Normal>(outGm_[base + c0 * n_], src, copyParams);
    }

    __aicore__ inline void ComputeTile(int64_t base, int64_t n)
    {
        const int64_t rowStride = RowStride(n, static_cast<int64_t>(sizeof(T)));
        auto acc = accBuf_.Get<float>();
        AscendC::Duplicate(acc, 0.0f, static_cast<int32_t>(n));

        for (int64_t c0 = 0; c0 < c_; c0 += static_cast<int64_t>(cChunk_)) {
            const int64_t cc = MinI64(static_cast<int64_t>(cChunk_), c_ - c0);
            auto xLocal = inQueue_.AllocTensor<T>();
            CopyIn(xLocal, base, c0, cc, n);
            inQueue_.EnQue(xLocal);
            xLocal = inQueue_.DeQue<T>();
            const auto xFloat = xFloatBuf_.Get<float>();
            for (int64_t i = 0; i < cc; ++i) {
                const int32_t offset = static_cast<int32_t>(i * rowStride);
                SquareAccumulateRow<T>(xLocal[offset], xFloat, acc, n);
            }
            inQueue_.FreeTensor(xLocal);
        }

        const float invC = 1.0f / static_cast<float>(c_);
        AscendC::Muls(acc, acc, invC, static_cast<int32_t>(n));
        AscendC::Adds(acc, acc, eps_, static_cast<int32_t>(n));
        auto rstd = rstdBuf_.Get<float>();
        AscendC::Rsqrt<float, AscendC::DEFAULT_RSQRT_CONFIG>(rstd, acc, static_cast<int32_t>(n));

        for (int64_t c0 = 0; c0 < c_; c0 += static_cast<int64_t>(cChunk_)) {
            const int64_t cc = MinI64(static_cast<int64_t>(cChunk_), c_ - c0);
            auto xLocal = inQueue_.AllocTensor<T>();
            CopyIn(xLocal, base, c0, cc, n);
            inQueue_.EnQue(xLocal);
            xLocal = inQueue_.DeQue<T>();

            auto gammaT = gammaQueue_.AllocTensor<T>();
            CopyInGamma(gammaT, c0, cc);
            gammaQueue_.EnQue(gammaT);
            gammaT = gammaQueue_.DeQue<T>();
            auto gammaF = gammaFBuf_.Get<float>();
            LoadGammaFloat(gammaT, gammaF, cc);
            gammaQueue_.FreeTensor(gammaT);

            auto outLocal = outQueue_.AllocTensor<T>();
            auto xFloat = xFloatBuf_.Get<float>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID7);
            for (int64_t i = 0; i < cc; ++i) {
                const int32_t offset = static_cast<int32_t>(i * rowStride);
                const float gammaValue = gammaF.GetValue(static_cast<uint32_t>(i));
                if constexpr (std::is_same<T, float>::value) {
                    AscendC::Mul(outLocal[offset], xLocal[offset], rstd, static_cast<int32_t>(n));
                    AscendC::Muls(outLocal[offset], outLocal[offset], gammaValue, static_cast<int32_t>(n));
                } else {
                    AscendC::Cast(xFloat, xLocal[offset], AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(n));
                    AscendC::Mul(xFloat, xFloat, rstd, static_cast<int32_t>(n));
                    AscendC::Muls(xFloat, xFloat, gammaValue, static_cast<int32_t>(n));
                    constexpr auto roundMode =
                        std::is_same<T, bfloat16_t>::value ? AscendC::RoundMode::CAST_RINT
                                                           : AscendC::RoundMode::CAST_NONE;
                    AscendC::Cast(outLocal[offset], xFloat, roundMode, static_cast<uint32_t>(n));
                }
            }
            outQueue_.EnQue(outLocal);
            inQueue_.FreeTensor(xLocal);

            outLocal = outQueue_.DeQue<T>();
            CopyOut(outLocal, base, c0, cc, n);
            outQueue_.FreeTensor(outLocal);
        }
    }

    AscendC::TPipe pipe_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<T> gammaGm_;
    AscendC::GlobalTensor<T> outGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> gammaQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rstdBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaFBuf_;

    int64_t b_ = 0;
    int64_t c_ = 0;
    int64_t n_ = 0;
    int64_t totalTiles_ = 0;
    int64_t tilesPerBatch_ = 0;
    int64_t numBlocks_ = 0;
    int64_t blockIdx_ = 0;
    uint32_t nTile_ = 0;
    uint32_t cChunk_ = 0;
    float eps_ = 1e-12f;
};

} // namespace RmsNormChannelsKernel

#endif // RMS_NORM_CHANNELS_KERNEL_IMPL_H

