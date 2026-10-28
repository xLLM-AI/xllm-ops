/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include "kernel_operator.h"
#include <pto/pto-inst.hpp>
#include "mega_kda_prefill_tiling_data.h"
#include "kda_kernel_utils.h"

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

// Dedicated flags, disjoint from the arithmetic stages' local 0-3 and 10-13 flags.
AICORE inline void Rendezvous()
{
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
}

struct FrontScratch {
    ND<float, 1, 128> acc, x, weight, tmp, norm_broadcast, gate, cumulative;
    ND<bfloat16_t, 1, 128> bf;
    ND<half, 1, 128> fp16;
    ND<float, 1, 64> reduce;
    ND<float, 1, 8, 1, 1> scalar;
    Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 1, 1> column;
    AICORE inline void Init()
    {
        TASSIGN(acc, 0); TASSIGN(x, 512); TASSIGN(weight, 1024); TASSIGN(tmp, 1536);
        TASSIGN(norm_broadcast, 2048); TASSIGN(gate, 2560); TASSIGN(cumulative, 3072);
        TASSIGN(bf, 3584); TASSIGN(fp16, 3840); TASSIGN(reduce, 4096);
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
    if (a.bias) s.BfLoad(s.acc, a.bias + channel);
    else { TEXPANDS(s.acc, 0.0f); PipeBarrierVec(); }
    for (int tap = 0; tap < 4; ++tap) {
        const int64_t pos = token + tap - 3;
        if (pos < begin && read < 0) { TEXPANDS(s.x, 0.0f); PipeBarrierVec(); }
        else {
            auto* p = pos < begin ? a.conv_in + (read * 3 + pos - begin + 3) * c + channel
                                 : a.qkv + pos * c + channel;
            s.BfLoad(s.x, p);
        }
        TCVT(s.fp16, s.x, RoundMode::CAST_RINT); PipeBarrierVec();
        TCVT(s.x, s.fp16, RoundMode::CAST_NONE); PipeBarrierVec();
        s.BfLoad(s.weight, a.weight + tap * c + channel);
        TMUL(s.tmp, s.x, s.weight); PipeBarrierVec();
        TADD(s.acc, s.acc, s.tmp); PipeBarrierVec();
    }
    TMULS(s.tmp, s.acc, -1.0f); PipeBarrierVec();
    TEXP(s.tmp, s.tmp); PipeBarrierVec();
    TADDS(s.tmp, s.tmp, 1.0f); PipeBarrierVec();
    TDIV(s.acc, s.acc, s.tmp); PipeBarrierVec();
    TCVT(s.bf, s.acc, RoundMode::CAST_RINT); PipeBarrierVec();
    TCVT(s.acc, s.bf, RoundMode::CAST_NONE); PipeBarrierVec();
}

AICORE inline void Normalize(FrontScratch& s)
{
    TMUL(s.tmp, s.acc, s.acc); PipeBarrierVec();
    TROWSUM(s.column, s.tmp, s.reduce); PipeBarrierVec();
    TADDS(s.scalar, s.scalar, 1.0e-6f); PipeBarrierVec();
    TSQRT(s.scalar, s.scalar); PipeBarrierVec();
    TROWEXPAND(s.norm_broadcast, s.column); PipeBarrierVec();
    TDIV(s.acc, s.acc, s.norm_broadcast); PipeBarrierVec();
    // The chunk stages consume BF16 normalized Q/K, not unrounded FP32 vectors.
    TCVT(s.bf, s.acc, RoundMode::CAST_RINT); PipeBarrierVec();
}

AICORE inline void Prepare(const Args& a, const TilingData& t, __gm__ uint8_t* ws, const Workspace& w)
{
#if defined(__DAV_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    const int64_t core = get_block_idx() * 2 + get_subblockid(), cores = get_block_num() * 2;
    int64_t chunks = 0;
    for (int64_t b = 0; b < t.batch; ++b) chunks += (a.cu[b + 1] - a.cu[b] + 127) / 128;
    FrontScratch s; s.Init();
    for (int64_t task = core; task < chunks * t.heads; task += cores) {
        const int64_t h = task % t.heads;
        int64_t chunk = task / t.heads, b = 0;
        for (; b < t.batch; ++b) {
            const int64_t n = (a.cu[b + 1] - a.cu[b] + 127) / 128;
            if (chunk < n) break;
            chunk -= n;
        }
        const int64_t start = a.cu[b] + chunk * 128;
        const int64_t end = start + 128 < a.cu[b + 1] ? start + 128 : a.cu[b + 1];
        TEXPANDS(s.cumulative, 0.0f); PipeBarrierVec();
        for (int64_t token = start; token < end; ++token) {
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
}
} // namespace kda_prefill
