/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include "kernel_operator.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
// The mixed A5 kernel uses __DAV_C310_VEC__ rather than __DAV_VEC__.  Put the
// TCVT implementation before pto-inst: pto-inst declares TCVT_IMPL without
// defaults, while tcvt_common defines the same overloads with defaults.
#include <pto/npu/a5/TCvt.hpp>
#endif
#include <pto/pto-inst.hpp>
#include "mega_kda_prefill_tiling_data.h"
#include "kda_kernel_utils.h"
#include "kda_prefill_silu_a5.h"

namespace kda_prefill {
using namespace pto;
using namespace mega_kda_prefill_utils;
template <typename T, int R, int C, int VR = R, int VC = C>
using ND = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, VR, VC>;
using GS = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using ST = pto::Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T> using GM = pto::GlobalTensor<T, GS, ST>;

template <typename T, typename TileT>
AICORE inline void Load(TileT& tile, __gm__ T* ptr, int rows, int cols, int stride)
{
    SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
    GM<T> gm(ptr, GS(1, 1, 1, rows, cols), ST(stride));
    TLOAD(tile, gm);
    SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
}
template <typename T, typename TileT>
AICORE inline void Store(__gm__ T* ptr, TileT& tile, int rows, int cols, int stride)
{
    SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
    GM<T> gm(ptr, GS(1, 1, 1, rows, cols), ST(stride));
    TSTORE(gm, tile);
    SetWaitFlag<PIPE_MTE3, PIPE_V>(0);
}
struct Args {
    __gm__ bfloat16_t *qkv, *gate;
    __gm__ float *beta, *a_log, *gate_bias;
    __gm__ bfloat16_t *weight, *bias, *conv_in;
    __gm__ float* ssm_in;
    __gm__ int32_t *cu, *conv_read, *conv_write, *ssm_read, *ssm_write;
    __gm__ bfloat16_t *output, *conv_out;
    __gm__ float* ssm_out;
};

AICORE inline bool ValidMetadata(const Args& a, const TilingData& t)
{
    if (a.cu[0] != 0 || a.cu[t.batch] != t.tokens) return false;
    for (int64_t b = 0; b < t.batch; ++b) {
        if (a.cu[b] < 0 || a.cu[b] > a.cu[b + 1] || a.cu[b + 1] > t.tokens ||
            a.conv_read[b] < -1 || a.conv_read[b] >= t.conv_input_slots ||
            a.ssm_read[b] < -1 || a.ssm_read[b] >= t.ssm_input_slots ||
            a.conv_write[b] < -1 || a.conv_write[b] >= t.conv_output_slots ||
            a.ssm_write[b] < -1 || a.ssm_write[b] >= t.ssm_output_slots) return false;
        if (a.cu[b] == a.cu[b + 1]) continue;
        for (int64_t j = 0; j < b; ++j) {
            if (a.cu[j] == a.cu[j + 1]) continue;
            if ((a.conv_write[b] >= 0 && a.conv_write[b] == a.conv_write[j]) ||
                (a.ssm_write[b] >= 0 && a.ssm_write[b] == a.ssm_write[j])) return false;
        }
    }
    return true;
}

// A2 uses dedicated flags. A5 uses the PTO MIX protocol after each stage
// has drained its local handshakes.
AICORE inline void Rendezvous()
{
#if __CCE_AICORE__ == 310
    // A5 uses hardware barriers and paired AIV intra-block notifications.
    pto::SYNCALL<pto::SyncCoreType::Mix>();
#else
    pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
    wait_flag_dev(5);
    ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0, 6));
    wait_flag_dev(6);
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(2, 7));
#elif defined(__DAV_VEC__)
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(2, 5));
    wait_flag_dev(7);
#endif
#endif
}

struct FrontScratch {
    ND<float, 1, 128> acc, x, weight, tmp, norm_broadcast, gate, cumulative;
    ND<bfloat16_t, 1, 128> bf;
    ND<float, 1, 64> reduce;
    ND<float, 1, 8, 1, 1> scalar;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 1, 1> column;
    AICORE inline void Init()
    {
        TASSIGN(acc, 0); TASSIGN(x, 512); TASSIGN(weight, 1024); TASSIGN(tmp, 1536);
        TASSIGN(norm_broadcast, 2048); TASSIGN(gate, 2560); TASSIGN(cumulative, 3072);
        TASSIGN(bf, 3584); TASSIGN(reduce, 4096);
        TASSIGN(scalar, 4352); TASSIGN(column, 4352);
    }
    AICORE inline void BfLoad(ND<float, 1, 128>& dst, __gm__ bfloat16_t* src)
    {
        Load(bf, src, 1, 128, 128);
        TCVT(dst, bf, RoundMode::CAST_NONE); PipeBarrierVec();
    }
};

AICORE inline void Conv(FrontScratch& s, const Args& a, const TilingData& t,
                        int64_t b, int64_t token, int64_t channel)
{
    const int64_t c = t.heads * 384, read = a.conv_read[b], begin = a.cu[b];
#if __CCE_AICORE__ == 310
    const int part = channel / (t.heads * 128);
    ND<float, 1, 128> cached;
    TASSIGN(cached, 14336 + part * 512);
    TMOV(s.acc, cached); PipeBarrierVec();
    TEXPANDS(s.norm_broadcast, 0.0f); PipeBarrierVec();
#else
    if (a.bias) s.BfLoad(s.acc, a.bias + channel);
    else { TEXPANDS(s.acc, 0.0f); PipeBarrierVec(); }
#endif
    for (int tap = 0; tap < 4; ++tap) {
        const int64_t pos = token + tap - 3;
        if (pos < begin && read < 0) { TEXPANDS(s.x, 0.0f); PipeBarrierVec(); }
        else {
            auto* p = pos < begin ? a.conv_in + (read * 3 + pos - begin + 3) * c + channel
                                 : a.qkv + pos * c + channel;
            s.BfLoad(s.x, p);
        }
#if __CCE_AICORE__ == 310
        TASSIGN(cached, 8192 + (part * 4 + tap) * 512);
#if defined(__DAV_C310_VEC__)
        MulAddCompensatedRowsA5<decltype(s.acc), decltype(cached)>(
            s.acc.data(), s.norm_broadcast.data(), s.acc.data(),
            s.norm_broadcast.data(), s.x.data(), cached.data(), 1);
#else
        TMUL(s.tmp, s.x, cached); PipeBarrierVec();
        TADD(s.acc, s.acc, s.tmp); PipeBarrierVec();
#endif
#else
        s.BfLoad(s.weight, a.weight + tap * c + channel);
        TMUL(s.tmp, s.x, s.weight); PipeBarrierVec();
        TADD(s.acc, s.acc, s.tmp); PipeBarrierVec();
#endif
    }
#if defined(__DAV_C310_VEC__)
    SiluRowsA5<decltype(s.acc)>(s.acc.data(), s.acc.data(), s.norm_broadcast.data(), 1);
#else
    TMULS(s.tmp, s.acc, -1.0f); PipeBarrierVec();
    TEXP(s.tmp, s.tmp); PipeBarrierVec();
    TADDS(s.tmp, s.tmp, 1.0f); PipeBarrierVec();
    TDIV(s.acc, s.acc, s.tmp); PipeBarrierVec();
#endif
    TCVT(s.bf, s.acc, RoundMode::CAST_RINT); PipeBarrierVec();
    TCVT(s.acc, s.bf, RoundMode::CAST_NONE); PipeBarrierVec();
}

AICORE inline void Normalize(FrontScratch& s)
{
#if defined(__DAV_C310_VEC__)
    decltype(s.column) low, epsilon;
    TASSIGN(low,4608);
    TASSIGN(epsilon,4864);
    TABS(s.tmp,s.acc); PipeBarrierVec();
    TROWMAX(s.column,s.tmp,s.reduce); PipeBarrierVec();
    SplitSquaresRangeA5<decltype(s.acc),decltype(s.column)>(s.tmp.data(),s.norm_broadcast.data(),
        s.acc.data(),epsilon.data(),s.acc.data(),s.column.data(),1);
    TROWSUM(s.column,s.tmp,s.reduce); PipeBarrierVec();
    TROWSUM(low,s.norm_broadcast,s.reduce); PipeBarrierVec();
    NormDenominatorWithEpsilonA5<decltype(s.column)>(s.column.data(),low.data(),s.column.data(),low.data(),epsilon.data(),1);
    NormalizeSplitA5<decltype(s.acc),decltype(s.column)>(s.acc.data(),s.acc.data(),s.column.data(),low.data(),1);
    TCVT(s.bf,s.acc,RoundMode::CAST_RINT); PipeBarrierVec();
#else
    TMUL(s.tmp, s.acc, s.acc); PipeBarrierVec();
    TROWSUM(s.column, s.tmp, s.reduce); PipeBarrierVec();
    TADDS(s.scalar, s.scalar, 1.0e-6f); PipeBarrierVec();
    TSQRT(s.scalar, s.scalar); PipeBarrierVec();
    TROWEXPAND(s.norm_broadcast, s.column); PipeBarrierVec();
    TDIV(s.acc, s.acc, s.norm_broadcast); PipeBarrierVec();
    // The chunk stages consume BF16 normalized Q/K, not unrounded FP32 vectors.
    TCVT(s.bf, s.acc, RoundMode::CAST_RINT); PipeBarrierVec();
#endif
}

#if __CCE_AICORE__ == 310
constexpr int kFrontendRowsA5 = 64;

// Keep the conversion chain in VF registers and feed the existing
// multiply/TwoSum sequence directly.  The three vcvt forms intentionally
// expand BF16 directly to FP32; only the final
// FP32 sum/low parts are written to UB.
template <typename Matrix, typename BfMatrix, typename Row>
__tf__ PTO_INTERNAL void ConvertMulAddCompensatedRowsA5(
    typename Matrix::TileDType __out__ sum_out,
    typename Matrix::TileDType __out__ low_out,
    typename Matrix::TileDType __in__ sum_in,
    typename Matrix::TileDType __in__ low_in,
    typename BfMatrix::TileDType __in__ bf_input,
    typename Row::TileDType __in__ weight, uint16_t rows)
{
    using namespace pto;
    auto* bp = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(bf_input);
    auto* sp = (__ubuf__ float*)__cce_get_tile_ptr(sum_in);
    auto* lp = (__ubuf__ float*)__cce_get_tile_ptr(low_in);
    auto* wp = (__ubuf__ float*)__cce_get_tile_ptr(weight);
    auto* so = (__ubuf__ float*)__cce_get_tile_ptr(sum_out);
    auto* lo = (__ubuf__ float*)__cce_get_tile_ptr(low_out);
    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> bf;
        RegTensor<float> x, product, a, s, v, e, tmp, l, w;
        MaskReg bfMask = PSetWithType<half>(PAT_ALL);
        MaskReg fp32Mask = PSetWithType<float>(PAT_ALL);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            vlds(w, wp, half * 64, NORM);
            for (uint16_t row = 0; row < rows; ++row) {
                const uint32_t offset = row * 128 + half * 64;
                vlds(bf, bp, offset, UNPK_B16);
                vcvt(x, bf, bfMask, PART_EVEN);
                vmul(product, x, w, fp32Mask, MODE_ZEROING);
                vlds(a, sp, offset, NORM);
                vlds(l, lp, offset, NORM);
                vadd(s, a, product, fp32Mask, MODE_ZEROING);
                vsub(v, s, a, fp32Mask, MODE_ZEROING);
                vsub(tmp, s, v, fp32Mask, MODE_ZEROING);
                vsub(e, a, tmp, fp32Mask, MODE_ZEROING);
                vsub(tmp, product, v, fp32Mask, MODE_ZEROING);
                vadd(e, e, tmp, fp32Mask, MODE_ZEROING);
                vadd(l, l, e, fp32Mask, MODE_ZEROING);
                vsts(s, so, offset, dist, fp32Mask);
                vsts(l, lo, offset, dist, fp32Mask);
            }
        }
    }
}

// Fuse the four-tap convolution's input movement and register chain.  The
// caller starts at token-3, so one BF16 tile contains every input needed by
// the output rows: [token-3, token+rows-1], plus the three look-ahead rows.
// Keeping each row's sum/low in registers across all four taps removes the
// three intermediate UB round trips while preserving the staged arithmetic.
// In particular, every tap still executes BF16->FP32->FP16->FP32, an
// independent FP32 vmul, and the full TwoSum update in the original order.
template <typename Matrix, typename LowMatrix, typename BfMatrix, typename Row>
__tf__ PTO_INTERNAL void ConvertConvRowsRegbaseA5(
    typename Matrix::TileDType __out__ sum_out,
    typename LowMatrix::TileDType __out__ low_out,
    typename Matrix::TileDType __in__ sum_in,
    typename LowMatrix::TileDType __in__ low_in,
    typename BfMatrix::TileDType __in__ bf_input,
    typename Row::TileDType __in__ weights, uint16_t rows)
{
    using namespace pto;
    auto* bp = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(bf_input);
    auto* sp = (__ubuf__ float*)__cce_get_tile_ptr(sum_in);
    auto* lp = (__ubuf__ float*)__cce_get_tile_ptr(low_in);
    auto* wp = (__ubuf__ float*)__cce_get_tile_ptr(weights);
    auto* so = (__ubuf__ float*)__cce_get_tile_ptr(sum_out);
    auto* lo = (__ubuf__ float*)__cce_get_tile_ptr(low_out);
    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> bf;
        RegTensor<float> x, product, a, s, v, e, tmp, l, w;
        MaskReg bfMask = PSetWithType<half>(PAT_ALL);
        MaskReg fp32Mask = PSetWithType<float>(PAT_ALL);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            for (uint16_t row = 0; row < rows; ++row) {
                const uint32_t outputOffset = row * 128 + half * 64;
                vlds(a, sp, outputOffset, NORM);
                vlds(l, lp, outputOffset, NORM);
                for (uint16_t tap = 0; tap < 4; ++tap) {
                    const uint32_t inputOffset = (row + tap) * 128 + half * 64;
                    vlds(bf, bp, inputOffset, UNPK_B16);
                    vcvt(x, bf, bfMask, PART_EVEN);
                    vlds(w, wp, tap * 128 + half * 64, NORM);
                    vmul(product, x, w, fp32Mask, MODE_ZEROING);

                    // Full TwoSum, including the product residual, exactly as
                    // in the existing per-tap helper.
                    vadd(s, a, product, fp32Mask, MODE_ZEROING);
                    vsub(v, s, a, fp32Mask, MODE_ZEROING);
                    vsub(tmp, s, v, fp32Mask, MODE_ZEROING);
                    vsub(e, a, tmp, fp32Mask, MODE_ZEROING);
                    vsub(tmp, product, v, fp32Mask, MODE_ZEROING);
                    vadd(e, e, tmp, fp32Mask, MODE_ZEROING);
                    vadd(l, l, e, fp32Mask, MODE_ZEROING);
                    a = s;
                }
                vsts(a, so, outputOffset, dist, fp32Mask);
                vsts(l, lo, outputOffset, dist, fp32Mask);
            }
        }
    }
}

// Batch independent Conv/QK rows; keep the gate scan in its original order.
// Only used after the three history-dependent tokens of each sequence.
AICORE inline void ConvRowsA5(const Args& a, const TilingData& t, __gm__ uint8_t* ws,
                              const Workspace& w, int64_t head, int64_t token, int rows)
{
    constexpr int R = kFrontendRowsA5;
    ND<float, R, 128, DYNAMIC> acc(rows), x(rows), tmp(rows), norm(rows);
    ND<bfloat16_t, R, 128, DYNAMIC> bf(rows);
    ND<float, R, 64, DYNAMIC> reduce(rows);
    Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, DYNAMIC, 1> column(rows);
    ND<float, 1, R, 1, DYNAMIC> scalar(rows);
    ND<float, 1, 128> parameter;
    constexpr int Base = 24576, F = R * 128 * sizeof(float), B = F / 2;
    TASSIGN(acc, Base); TASSIGN(x, Base + F); TASSIGN(tmp, Base + 2 * F);
    TASSIGN(norm, Base + 3 * F); TASSIGN(bf, Base + 4 * F);
    TASSIGN(reduce, Base + 5 * F);
    TASSIGN(column, Base + 5 * F + B); TASSIGN(scalar, Base + 5 * F + B);
    const int64_t channels = t.heads * 384;
    for (int part = 0; part < 3; ++part) {
        TASSIGN(parameter, 14336 + part * 512);
        TCOLEXPAND(acc, parameter); PipeBarrierVec();
        TEXPANDS(norm, 0.0f); PipeBarrierVec();
        const int64_t channel = (part * t.heads + head) * 128;
        for (int tap = 0; tap < 4; ++tap) {
            Load(bf, a.qkv + (token + tap - 3) * channels + channel, rows, 128, channels);
            TASSIGN(parameter, 8192 + (part * 4 + tap) * 512);
#if defined(__DAV_C310_VEC__)
            ConvertMulAddCompensatedRowsA5<decltype(acc), decltype(bf), decltype(parameter)>(
                acc.data(), norm.data(), acc.data(), norm.data(),
                bf.data(), parameter.data(), rows);
#else
            TCVT(x, bf, RoundMode::CAST_NONE); PipeBarrierVec();
            TCOLEXPANDMUL(tmp, x, parameter); PipeBarrierVec();
            TADD(acc, acc, tmp); PipeBarrierVec();
#endif
        }
#if defined(__DAV_C310_VEC__)
        SiluRowsA5<decltype(acc)>(acc.data(), acc.data(), norm.data(), rows);
#else
        TMULS(tmp, acc, -1.0f); PipeBarrierVec();
        TEXP(tmp, tmp); PipeBarrierVec();
        TADDS(tmp, tmp, 1.0f); PipeBarrierVec();
        TDIV(acc, acc, tmp); PipeBarrierVec();
#endif
        TCVT(bf, acc, RoundMode::CAST_RINT); PipeBarrierVec();
        TCVT(acc, bf, RoundMode::CAST_NONE); PipeBarrierVec();
        if (part < 2) {
#if defined(__DAV_C310_VEC__)
            decltype(column) low(rows), epsilon(rows);
            TASSIGN(low,Base+5*F+B+256); TASSIGN(epsilon,Base+5*F+B+512);
            TABS(tmp,acc); PipeBarrierVec();
            TROWMAX(column,tmp,reduce); PipeBarrierVec();
            SplitSquaresRangeA5<decltype(acc),decltype(column)>(tmp.data(),norm.data(),acc.data(),epsilon.data(),acc.data(),column.data(),rows);
            TROWSUM(column,tmp,reduce); PipeBarrierVec();
            TROWSUM(low,norm,reduce); PipeBarrierVec();
            NormDenominatorWithEpsilonA5<decltype(column)>(column.data(),low.data(),column.data(),low.data(),epsilon.data(),rows);
            NormalizeSplitA5<decltype(acc),decltype(column)>(acc.data(),acc.data(),column.data(),low.data(),rows);
#else
            TMUL(tmp, acc, acc); PipeBarrierVec();
            TROWSUM(column, tmp, reduce); PipeBarrierVec();
            TADDS(scalar, scalar, 1.0e-6f); PipeBarrierVec();
            TSQRT(scalar, scalar); PipeBarrierVec();
            TROWEXPAND(norm, column); PipeBarrierVec();
            TDIV(acc, acc, norm); PipeBarrierVec();
#endif
            TCVT(bf, acc, RoundMode::CAST_RINT); PipeBarrierVec();
            Store(reinterpret_cast<__gm__ bfloat16_t*>(ws + (part == 0 ? w.q : w.k)) +
                (head * t.tokens + token) * 128, bf, rows, 128, 128);
        } else {
            Store(reinterpret_cast<__gm__ float*>(ws + w.value) + (token * t.heads + head) * 128,
                  acc, rows, 128, t.heads * 128);
        }
    }
}

// A5-only deeper regbase variant.  It intentionally covers only the
// history-independent rows; the first three tokens continue through Conv(),
// which handles conv_state_in and sequence boundaries exactly as before.
#if defined(__DAV_C310_VEC__)
AICORE inline void WaitConvPrefetchA5(uint32_t id)
{
    wait_flag(PIPE_MTE2, PIPE_V, static_cast<event_t>(id));
}

template <typename TileT>
AICORE inline void LoadConvPrefetchA5(TileT& tile, __gm__ bfloat16_t* ptr,
                                      int rows, int cols, int stride, uint32_t id)
{
    // Unlike Load(), this deliberately only posts the two pipe dependencies.
    // The MTE2->V wait is deferred until the tile is consumed by VF code.
    const event_t event = static_cast<event_t>(id);
    set_flag(PIPE_V, PIPE_MTE2, event);
    wait_flag(PIPE_V, PIPE_MTE2, event);
    GM<bfloat16_t> gm(ptr, GS(1, 1, 1, rows, cols), ST(stride));
    TLOAD(tile, gm);
    set_flag(PIPE_MTE2, PIPE_V, event);
}

AICORE inline void ConvRowsRegbaseA5(const Args& a, const TilingData& t, __gm__ uint8_t* ws,
                                     const Workspace& w, int64_t head, int64_t token, int rows)
{
    constexpr int R = kFrontendRowsA5;
    constexpr int F = R * 128 * sizeof(float);
    constexpr int B = F / 2;
    constexpr int Base = 24576;
    constexpr int InputRows = R + 3;
    constexpr int InputBytes = InputRows * 128 * sizeof(bfloat16_t);
    constexpr int Prefetch0 = 155648;
    constexpr int Prefetch1 = Prefetch0 + InputBytes;
    constexpr int UBBytes = 256 * 1024;
    static_assert(InputBytes <= F, "ConvRowsRegbaseA5 input tile exceeds its UB slot");
    static_assert(Prefetch0 % 256 == 0 && Prefetch1 % 256 == 0,
                  "ConvRowsRegbaseA5 prefetch tiles must be 256B aligned");
    static_assert(Prefetch1 + InputBytes <= UBBytes,
                  "ConvRowsRegbaseA5 prefetch tiles exceed the A5 UB");

    ND<float, R, 128, DYNAMIC> acc(rows), low(rows), tmp(rows);
    ND<bfloat16_t, InputRows, 128, DYNAMIC> bf_prefetch0(rows + 3);
    ND<bfloat16_t, InputRows, 128, DYNAMIC> bf_prefetch1(rows + 3);
    ND<bfloat16_t, R, 128, DYNAMIC> bf_output(rows);
    ND<float, R, 64, DYNAMIC> reduce(rows);
    Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, DYNAMIC, 1> column(rows);
    ND<float, 1, 128> parameter;
    TASSIGN(acc, Base);
    TASSIGN(low, Base + F);
    TASSIGN(bf_prefetch0, Prefetch0);
    TASSIGN(bf_prefetch1, Prefetch1);
    TASSIGN(bf_output, Base + 2 * F);
    // bf_output and tmp deliberately share this lifetime-disjoint slot.  The
    // BF16 tile is produced/consumed before tmp is used for normalization.
    TASSIGN(tmp, Base + 2 * F);
    TASSIGN(reduce, Base + 3 * F);
    TASSIGN(column, Base + 3 * F + B);

    const int64_t channels = t.heads * 384;
    // The first part has no predecessor to overlap.  Its completion is
    // awaited only immediately before ConvertConvRowsRegbaseA5.
    LoadConvPrefetchA5(bf_prefetch0,
                       a.qkv + (token - 3) * channels + (head) * 128,
                       rows + 3, 128, channels, 1);
    for (int part = 0; part < 3; ++part) {
        TASSIGN(parameter, 14336 + part * 512);
        // Bias remains the first addend, matching ConvRowsA5 and Conv().
        TCOLEXPAND(acc, parameter);
        PipeBarrierVec();
        TEXPANDS(low, 0.0f);
        PipeBarrierVec();

        if (part == 0) {
            LoadConvPrefetchA5(
                bf_prefetch1,
                a.qkv + (token - 3) * channels + (t.heads + head) * 128,
                rows + 3, 128, channels, 2);
        } else if (part == 1) {
            LoadConvPrefetchA5(
                bf_prefetch0,
                a.qkv + (token - 3) * channels + (2 * t.heads + head) * 128,
                rows + 3, 128, channels, 1);
        }

        // batched_start >= sequence_start + 3, hence this GM span never
        // crosses a sequence or consults conv_state_in.
        if (part == 0) {
            WaitConvPrefetchA5(1);
        } else if (part == 1) {
            WaitConvPrefetchA5(2);
        } else {
            WaitConvPrefetchA5(1);
        }
        TASSIGN(parameter, 8192 + part * 4 * 512);
        if (part == 0) {
            ConvertConvRowsRegbaseA5<decltype(acc), decltype(low), decltype(bf_prefetch0),
                                     decltype(parameter)>(
                acc.data(), low.data(), acc.data(), low.data(), bf_prefetch0.data(),
                parameter.data(), rows);
        } else if (part == 1) {
            ConvertConvRowsRegbaseA5<decltype(acc), decltype(low), decltype(bf_prefetch1),
                                     decltype(parameter)>(
                acc.data(), low.data(), acc.data(), low.data(), bf_prefetch1.data(),
                parameter.data(), rows);
        } else {
            ConvertConvRowsRegbaseA5<decltype(acc), decltype(low), decltype(bf_prefetch0),
                                     decltype(parameter)>(
                acc.data(), low.data(), acc.data(), low.data(), bf_prefetch0.data(),
                parameter.data(), rows);
        }

        // SiLU retains its full TwoSum renormalization and compensated small-
        // argument path; only the convolution's intermediate UB traffic is
        // removed here.
        SiluRowsA5<decltype(acc)>(acc.data(), acc.data(), low.data(), rows);
        TCVT(bf_output, acc, RoundMode::CAST_RINT);
        PipeBarrierVec();
        // Preserve the existing BF16 publication point: normalization and the
        // value stream consume the FP32 reconstruction of the rounded BF16,
        // not the pre-rounding SiLU result.
        TCVT(acc, bf_output, RoundMode::CAST_NONE);
        PipeBarrierVec();

        if (part < 2) {
            decltype(column) normLow(rows), epsilon(rows);
            TASSIGN(normLow, Base + 3 * F + B + 256);
            TASSIGN(epsilon, Base + 3 * F + B + 512);
            TABS(tmp, acc);
            PipeBarrierVec();
            TROWMAX(column, tmp, reduce);
            PipeBarrierVec();
            SplitSquaresRangeA5<decltype(acc), decltype(column)>(
                tmp.data(), low.data(), acc.data(), epsilon.data(), acc.data(), column.data(), rows);
            TROWSUM(column, tmp, reduce);
            PipeBarrierVec();
            TROWSUM(normLow, low, reduce);
            PipeBarrierVec();
            NormDenominatorWithEpsilonA5<decltype(column)>(
                column.data(), normLow.data(), column.data(), normLow.data(), epsilon.data(), rows);
            NormalizeSplitA5<decltype(acc), decltype(column)>(
                acc.data(), acc.data(), column.data(), normLow.data(), rows);
            TCVT(bf_output, acc, RoundMode::CAST_RINT);
            PipeBarrierVec();
            Store(reinterpret_cast<__gm__ bfloat16_t*>(ws + (part == 0 ? w.q : w.k)) +
                      (head * t.tokens + token) * 128,
                  bf_output, rows, 128, 128);
        } else {
            Store(reinterpret_cast<__gm__ float*>(ws + w.value) +
                      (token * t.heads + head) * 128,
                  acc, rows, 128, t.heads * 128);
        }
    }
}
#endif
#endif

#if defined(__DAV_C310_VEC__)
template <typename Matrix, typename Row>
__tf__ PTO_INTERNAL void GateScanA5(typename Matrix::TileDType __out__ output,
                                   typename Row::TileDType __out__ carry,
                                   typename Matrix::TileDType __in__ values,
                                   typename Row::TileDType __in__ cumulative,
                                   uint16_t rows)
{
    auto* p = (__ubuf__ float*)__cce_get_tile_ptr(values);
    auto* c = (__ubuf__ float*)__cce_get_tile_ptr(cumulative);
    auto* out = (__ubuf__ float*)__cce_get_tile_ptr(output);
    auto* next = (__ubuf__ float*)__cce_get_tile_ptr(carry);
    __VEC_SCOPE__ {
        RegTensor<float> sum, value;
        MaskReg mask = PSetWithType<float>(PAT_ALL);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            vlds(sum, c, half * 64, NORM);
            for (uint16_t row = 0; row < rows; ++row) {
                const uint32_t offset = row * 128 + half * 64;
                vlds(value, p, offset, NORM);
                vadd(sum, sum, value, mask, MODE_ZEROING);
                vsts(sum, out, offset, dist, mask);
            }
            vsts(sum, next, half * 64, dist, mask);
        }
    }
}

// Evaluate the complete gate row chain in VF registers.  This is deliberately
// kept separate from the legacy GateScanA5 helper above: the latter remains
// available to the non-fused paths, while this helper only publishes the
// final gate-prefix rows and the post-chunk carry to UB.
//
// The source gate is BF16.  UNPK_B16 followed by the PART_EVEN expansion is
// the same exact BF16 -> FP32 conversion used by TCVT on A5; no intermediate
// FP16 value or approximation is introduced.  Bias, cached exp(a_log), and
// the incoming cumulative row are loaded once for each 64-lane half and are
// then reused for every row.  The scalar operations intentionally remain
// independent vector instructions and retain the original order:
//   add, mul, muls(-1), exp, adds(1), div(-5 / ...), add(cumulative, ...).
template <typename Matrix, typename BfMatrix, typename Row>
__tf__ PTO_INTERNAL void GateRowsVFA5(
    typename Matrix::TileDType __out__ output,
    typename Row::TileDType __out__ carry,
    typename BfMatrix::TileDType __in__ bf_input,
    typename Row::TileDType __in__ bias,
    typename Row::TileDType __in__ scale,
    typename Row::TileDType __in__ cumulative,
    uint16_t rows)
{
    using namespace pto;
    auto* gp = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(bf_input);
    auto* bp = (__ubuf__ float*)__cce_get_tile_ptr(bias);
    auto* xp = (__ubuf__ float*)__cce_get_tile_ptr(scale);
    auto* cp = (__ubuf__ float*)__cce_get_tile_ptr(cumulative);
    auto* op = (__ubuf__ float*)__cce_get_tile_ptr(output);
    auto* np = (__ubuf__ float*)__cce_get_tile_ptr(carry);
    __VEC_SCOPE__ {
        RegTensor<bfloat16_t> bf;
        RegTensor<float> gate, biasReg, scaleReg, sum, value, negFive;
        MaskReg bfMask = PSetWithType<half>(PAT_ALL);
        MaskReg fp32Mask = PSetWithType<float>(PAT_ALL);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            const uint32_t halfOffset = half * 64;
            vlds(biasReg, bp, halfOffset, NORM);
            vlds(scaleReg, xp, halfOffset, NORM);
            vlds(sum, cp, halfOffset, NORM);
            vbr(negFive, -5.0f);
            for (uint16_t row = 0; row < rows; ++row) {
                const uint32_t offset = row * 128 + halfOffset;
                vlds(bf, gp, offset, UNPK_B16);
                vcvt(gate, bf, bfMask, PART_EVEN);
                vadd(gate, gate, biasReg, fp32Mask, MODE_ZEROING);
                vmul(gate, gate, scaleReg, fp32Mask, MODE_ZEROING);
                vmuls(gate, gate, -1.0f, fp32Mask, MODE_ZEROING);
                vexp(gate, gate, fp32Mask, MODE_ZEROING);
                vadds(gate, gate, 1.0f, fp32Mask, MODE_ZEROING);
                vdiv(value, negFive, gate, fp32Mask, MODE_ZEROING);
                vadd(sum, sum, value, fp32Mask, MODE_ZEROING);
                vsts(sum, op, offset, dist, fp32Mask);
            }
            vsts(sum, np, halfOffset, dist, fp32Mask);
        }
    }
}

AICORE inline void GateRowsA5(FrontScratch& s, const Args& a, const TilingData& t,
                              __gm__ uint8_t* ws, const Workspace& w,
                              int64_t head, int64_t token, int rows)
{
    constexpr int R = kFrontendRowsA5;
    ND<float, R, 128, DYNAMIC> gate(rows), tmp(rows);
    ND<bfloat16_t, R, 128, DYNAMIC> bf(rows);
    ND<float, R, 8, DYNAMIC, 1> betas(rows);
    ND<float, 1, 128> bias, scale;
    constexpr int Base = 24576, F = R * 128 * sizeof(float);
    TASSIGN(gate, Base); TASSIGN(tmp, Base + F); TASSIGN(bf, Base + 2 * F);
    TASSIGN(betas, Base + 2 * F + F / 2);
    TASSIGN(bias, 16384); TASSIGN(scale, 16896);
    Load(bf, a.gate + (token * t.heads + head) * 128, rows, 128, t.heads * 128);
    GateRowsVFA5<decltype(gate), decltype(bf), decltype(bias)>(
        gate.data(), s.cumulative.data(), bf.data(), bias.data(), scale.data(), s.cumulative.data(), rows);
    Store(reinterpret_cast<__gm__ float*>(ws + w.g) + (head * t.tokens + token) * 128,
          gate, rows, 128, 128);
    Load(betas, a.beta + token * t.heads + head, rows, 1, t.heads);
    Store(reinterpret_cast<__gm__ float*>(ws + w.beta) + head * t.tokens + token,
          betas, rows, 1, 1);
    TEXPANDS(tmp, 0.0f); PipeBarrierVec();
    Store(reinterpret_cast<__gm__ float*>(ws + w.lower) + (token * t.heads + head) * 128,
          tmp, rows, 128, t.heads * 128);
    Store(reinterpret_cast<__gm__ float*>(ws + w.aqk) + (token * t.heads + head) * 128,
          tmp, rows, 128, t.heads * 128);
}
#endif

AICORE inline void Prepare(const Args& a, const TilingData& t, __gm__ uint8_t* ws, const Workspace& w)
{
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    const int64_t core = get_block_idx() * 2 + get_subblockid(), cores = get_block_num() * 2;
    int64_t chunks = 0;
    for (int64_t b = 0; b < t.batch; ++b) chunks += (a.cu[b + 1] - a.cu[b] + 127) / 128;
    FrontScratch s; s.Init();
#if __CCE_AICORE__ == 310
    int64_t cached_head = -1;
#endif
    for (int64_t task = core; task < chunks * t.heads; task += cores) {
#if __CCE_AICORE__ == 310
        // Reuse parameters only when each AIV can visit multiple chunks/head.
        // For short inputs retain token-major scheduling and its load balance.
        const bool reuse_head = chunks >= 2 * cores;
        const int64_t h = reuse_head ? task / chunks : task % t.heads;
        int64_t chunk = reuse_head ? task % chunks : task / t.heads, b = 0;
#else
        const int64_t h = task % t.heads;
        int64_t chunk = task / t.heads, b = 0;
#endif
        for (; b < t.batch; ++b) {
            const int64_t n = (a.cu[b + 1] - a.cu[b] + 127) / 128;
            if (chunk < n) break;
            chunk -= n;
        }
        const int64_t start = a.cu[b] + chunk * 128;
        const int64_t end = start + 128 < a.cu[b + 1] ? start + 128 : a.cu[b + 1];
#if __CCE_AICORE__ == 310
        // Parameters are invariant over this head/chunk. Keep FP32 copies in
        // disjoint UB storage instead of reloading and converting every token.
        ND<float, 1, 128> cached, gate_bias_cache, gate_scale_cache;
        TASSIGN(gate_bias_cache, 16384);
        TASSIGN(gate_scale_cache, 16896);
        if (cached_head != h) {
            for (int part = 0; part < 3; ++part) {
                const int64_t channel = (part * t.heads + h) * 128;
                for (int tap = 0; tap < 4; ++tap) {
                    TASSIGN(cached, 8192 + (part * 4 + tap) * 512);
                    s.BfLoad(cached, a.weight + tap * t.heads * 384 + channel);
                }
                TASSIGN(cached, 14336 + part * 512);
                if (a.bias) s.BfLoad(cached, a.bias + channel);
                else { TEXPANDS(cached, 0.0f); PipeBarrierVec(); }
            }
            Load(gate_bias_cache, a.gate_bias + h * 128, 1, 128, 128);
            TEXPANDS(gate_scale_cache, a.a_log[h]); PipeBarrierVec();
            TEXP(gate_scale_cache, gate_scale_cache); PipeBarrierVec();
            cached_head = h;
        }
        const int64_t batched_start = start > a.cu[b] + 3 ? start : a.cu[b] + 3;
        for (int64_t token = batched_start; token < end; token += kFrontendRowsA5) {
            const int rows = end - token < kFrontendRowsA5 ? end - token : kFrontendRowsA5;
#if defined(__DAV_C310_VEC__)
            ConvRowsRegbaseA5(a, t, ws, w, h, token, rows);
#else
            ConvRowsA5(a, t, ws, w, h, token, rows);
#endif
        }
#endif
        TEXPANDS(s.cumulative, 0.0f); PipeBarrierVec();
#if __CCE_AICORE__ == 310
        for (int64_t token = start; token < end; token += kFrontendRowsA5) {
            const int rows = end - token < kFrontendRowsA5 ? end - token : kFrontendRowsA5;
            GateRowsA5(s, a, t, ws, w, h, token, rows);
        }
        const int64_t scalar_end = batched_start < end ? batched_start : end;
        for (int64_t token = start; token < scalar_end; ++token) {
#else
        for (int64_t token = start; token < end; ++token) {
#endif
#if __CCE_AICORE__ == 310
            if (token < batched_start) {
#endif
            for (int part = 0; part < 3; ++part) {
                Conv(s, a, t, b, token, (part * t.heads + h) * 128);
                if (part < 2) {
                    Normalize(s);
                    Store(reinterpret_cast<__gm__ bfloat16_t*>(ws + (part == 0 ? w.q : w.k)) +
                        (h * t.tokens + token) * 128, s.bf, 1, 128, 128);
                } else {
                    Store(reinterpret_cast<__gm__ float*>(ws + w.value) + (token * t.heads + h) * 128,
                          s.acc, 1, 128, 128);
                }
            }
#if __CCE_AICORE__ == 310
            }
#endif
#if __CCE_AICORE__ != 310
            s.BfLoad(s.gate, a.gate + (token * t.heads + h) * 128);
            Load(s.tmp, a.gate_bias + h * 128, 1, 128, 128);
            TADD(s.gate, s.gate, s.tmp); PipeBarrierVec();
            TEXPANDS(s.tmp, a.a_log[h]); PipeBarrierVec();
            TEXP(s.tmp, s.tmp); PipeBarrierVec();
            TMUL(s.gate, s.gate, s.tmp); PipeBarrierVec();
            TMULS(s.gate, s.gate, -1.0f); PipeBarrierVec();
            TEXP(s.gate, s.gate); PipeBarrierVec();
            TADDS(s.gate, s.gate, 1.0f); PipeBarrierVec();
            TEXPANDS(s.tmp, -5.0f); PipeBarrierVec();
            TDIV(s.gate, s.tmp, s.gate); PipeBarrierVec();
            TADD(s.cumulative, s.cumulative, s.gate); PipeBarrierVec();
            Store(reinterpret_cast<__gm__ float*>(ws + w.g) + (h * t.tokens + token) * 128,
                  s.cumulative, 1, 128, 128);
            Load(s.scalar, a.beta + token * t.heads + h, 1, 1, 1);
            Store(reinterpret_cast<__gm__ float*>(ws + w.beta) + h * t.tokens + token, s.scalar, 1, 1, 1);
            TEXPANDS(s.tmp, 0.0f); PipeBarrierVec();
            Store(reinterpret_cast<__gm__ float*>(ws + w.lower) + (token * t.heads + h) * 128, s.tmp, 1, 128, 128);
            Store(reinterpret_cast<__gm__ float*>(ws + w.aqk) + (token * t.heads + h) * 128, s.tmp, 1, 128, 128);
#endif
        }
        if (end == a.cu[b + 1] && a.conv_write[b] >= 0) {
            for (int part = 0; part < 3; ++part) {
                const int64_t channel = (part * t.heads + h) * 128, c = t.heads * 384;
                for (int tap = 0; tap < 3; ++tap) {
                    const int64_t pos = end - 3 + tap;
                    if (pos < a.cu[b] && a.conv_read[b] < 0) {
                        TEXPANDS(s.tmp, 0.0f); PipeBarrierVec();
                        TCVT(s.bf, s.tmp, RoundMode::CAST_RINT); PipeBarrierVec();
                    } else {
                        auto* p = pos < a.cu[b] ? a.conv_in +
                            (int64_t(a.conv_read[b]) * 3 + pos - a.cu[b] + 3) * c + channel
                            : a.qkv + pos * c + channel;
                        Load(s.bf, p, 1, 128, 128);
                    }
                    Store(a.conv_out + (int64_t(a.conv_write[b]) * 3 + tap) * c + channel, s.bf, 1, 128, 128);
                }
            }
        }
    }
    // Constants are produced in UB and published through MTE3, never scalar GM stores.
    for (int row = core; row < 128; row += cores) {
        SetWaitFlag<PIPE_V, PIPE_S>(0);
        for (int col = 0; col < 128; ++col) s.tmp.data()[col] = col < row ? 1.0f : 0.0f;
        SetWaitFlag<PIPE_S, PIPE_V>(0);
        Store(reinterpret_cast<__gm__ float*>(ws + w.masks) + row * 128, s.tmp, 1, 128, 128);
        SetWaitFlag<PIPE_V, PIPE_S>(0);
        for (int col = 0; col < 128; ++col) s.tmp.data()[col] = col <= row ? 1.0f : 0.0f;
        SetWaitFlag<PIPE_S, PIPE_V>(0);
        Store(reinterpret_cast<__gm__ float*>(ws + w.masks) + 16384 + row * 128, s.tmp, 1, 128, 128);
        SetWaitFlag<PIPE_V, PIPE_S>(0);
        for (int col = 0; col < 128; ++col) s.tmp.data()[col] = col == row ? 1.0f : 0.0f;
        SetWaitFlag<PIPE_S, PIPE_V>(0);
        Store(reinterpret_cast<__gm__ float*>(ws + w.eye) + row * 128, s.tmp, 1, 128, 128);
    }
#endif
}

template <bool Publish>
AICORE inline void StateIO(const Args& a, const TilingData& t, __gm__ uint8_t* ws, const Workspace& w)
{
#if defined(__DAV_VEC__)
    const int64_t core = get_block_idx() * 2 + get_subblockid(), cores = get_block_num() * 2;
    ND<float, 16, 128> src;
    ND<float, 128, 16> dst;
    ND<float, 16, 128> tmp;
    TASSIGN(src, 0); TASSIGN(dst, 8192); TASSIGN(tmp, 16384);
    for (int64_t task = core; task < t.batch * t.heads * 8; task += cores) {
        const int64_t b = task / (t.heads * 8), h = task / 8 % t.heads, row = task % 8 * 16;
        if (a.cu[b] == a.cu[b + 1]) continue;
        if constexpr (Publish) {
            const int64_t slot = a.ssm_write[b];
            if (slot < 0) continue;
            Load(src, reinterpret_cast<__gm__ float*>(ws + w.final) + (b * t.heads + h) * 16384 + row * 128, 16, 128, 128);
            TTRANS(dst, src, tmp); PipeBarrierVec();
            Store(a.ssm_out + (slot * t.heads + h) * 16384 + row, dst, 128, 16, 128);
        } else {
            const int64_t slot = a.ssm_read[b];
            if (slot < 0) { TEXPANDS(src, 0.0f); PipeBarrierVec(); }
            else Load(src, a.ssm_in + (slot * t.heads + h) * 16384 + row * 128, 16, 128, 128);
            TTRANS(dst, src, tmp); PipeBarrierVec();
            Store(reinterpret_cast<__gm__ float*>(ws + w.initial) + (b * t.heads + h) * 16384 + row, dst, 128, 16, 128);
        }
    }
#endif
}

AICORE inline void Output(const Args& a, const TilingData& t, __gm__ uint8_t* ws, const Workspace& w)
{
#if defined(__DAV_VEC__)
#if __CCE_AICORE__ == 310
    constexpr int Rows = 32;
    const int64_t core = get_block_idx() * 2 + get_subblockid();
    const int64_t cores = get_block_num() * 2, total = t.tokens * t.heads;
    for (int64_t task = core * Rows; task < total; task += cores * Rows) {
        const int rows = total - task < Rows ? total - task : Rows;
        ND<float, Rows, 128, DYNAMIC> x(rows);
        ND<bfloat16_t, Rows, 128, DYNAMIC> y(rows);
        TASSIGN(x, 0); TASSIGN(y, Rows * 128 * sizeof(float));
        Load(x, reinterpret_cast<__gm__ float*>(ws + w.output) + task * 128, rows, 128, 128);
        TMULS(x, x, 0.08838834764831845f); PipeBarrierVec();
        TCVT(y, x, RoundMode::CAST_RINT); PipeBarrierVec();
        Store(a.output + task * 128, y, rows, 128, 128);
    }
#else
    ND<float, 1, 128> x;
    ND<bfloat16_t, 1, 128> y;
    TASSIGN(x, 0); TASSIGN(y, 512);
    const int64_t cores = get_block_num() * 2;
    for (int64_t task = get_block_idx() * 2 + get_subblockid(); task < t.tokens * t.heads; task += cores) {
        Load(x, reinterpret_cast<__gm__ float*>(ws + w.output) + task * 128, 1, 128, 128);
        TMULS(x, x, 0.08838834764831845f); PipeBarrierVec();
        TCVT(y, x, RoundMode::CAST_RINT); PipeBarrierVec();
        Store(a.output + task * 128, y, 1, 128, 128);
    }
#endif
#endif
}
} // namespace kda_prefill
