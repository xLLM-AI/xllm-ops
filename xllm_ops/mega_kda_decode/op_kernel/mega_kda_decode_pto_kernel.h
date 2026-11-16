/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

#include "kernel_operator.h"
#include <pto/pto-inst.hpp>
#include "mega_kda_decode_tiling_data.h"

namespace mega_kda {
using namespace pto;

template <typename T, int R, int C, int VR = R, int VC = C>
using TileND = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, VR, VC>;
template <int R, int VR = R>
using Column = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, VR, 1>;

AICORE inline void VBarrier() { pipe_barrier(PIPE_V); }

template <typename T, int R, int C, typename TileT>
AICORE inline void Load(TileT& tile, __gm__ T* ptr)
{
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    GlobalTensor<T, Shape<1, 1, 1, R, C>, Stride<R * C, R * C, R * C, C, 1>> gm(ptr);
    TLOAD(tile, gm);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}

template <typename T, int R, int C, typename TileT>
AICORE inline void Store(__gm__ T* ptr, TileT& tile)
{
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    GlobalTensor<T, Shape<1, 1, 1, R, C>, Stride<R * C, R * C, R * C, C, 1>> gm(ptr);
    TSTORE(gm, tile);
    // A subsequent token/task may reuse every UB address below.
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

struct Args {
    __gm__ bfloat16_t *qkv, *gate, *beta;
    __gm__ float *a_log, *gate_bias;
    __gm__ bfloat16_t *conv_weight, *conv_bias, *conv_in;
    __gm__ float* ssm_in;
    __gm__ int32_t *offsets, *conv_read, *conv_write, *ssm_read, *ssm_write, *accepted;
    __gm__ bfloat16_t *output, *conv_out;
    __gm__ float* ssm_out;
};

constexpr int kVRows = 32;
constexpr int kVShards = 128 / kVRows;

// One task owns kVRows contiguous V rows. Q/K/Conv are private to each task;
// only V-shard zero publishes Conv state, so no cross-core rendezvous is needed.
struct Scratch {
    TileND<float, kVRows, 128> state, expanded, product;
    TileND<float, kVRows, 64> reduce_work;
    TileND<float, 1, 128> q, k, v, g, tmp, x, weight, acc;
    TileND<bfloat16_t, 1, 128> bf16;
    TileND<half, 1, 128> fp16;
    TileND<float, 1, kVRows> reduced, delta, value;
    Column<kVRows> reduced_col, delta_col;
    Column<8, 1> norm_col;
    TileND<float, 1, 8, 1, 1> norm, beta_scalar;
    TileND<bfloat16_t, 1, 16, 1, 1> beta_bf16;
    TileND<bfloat16_t, 1, kVRows> out;

    AICORE inline void Init(int64_t vStart)
    {
        constexpr int stateBytes = kVRows * 128 * sizeof(float);
        constexpr int vectors = 3 * stateBytes + kVRows * 64 * sizeof(float);
        constexpr int scalars = vectors + 4608 + 2 * kVRows * sizeof(float);
        TASSIGN(state, 0);
        TASSIGN(expanded, stateBytes);
        TASSIGN(product, 2 * stateBytes);
        TASSIGN(reduce_work, 3 * stateBytes);
        TASSIGN(q, vectors);
        TASSIGN(k, vectors + 512);
        TASSIGN(v, vectors + 1024);
        TASSIGN(value, vectors + 1024 + vStart * 4);
        TASSIGN(g, vectors + 1536);
        TASSIGN(tmp, vectors + 2048);
        TASSIGN(x, vectors + 2560);
        TASSIGN(weight, vectors + 3072);
        TASSIGN(acc, vectors + 3584);
        TASSIGN(bf16, vectors + 4096);
        TASSIGN(fp16, vectors + 4352);
        TASSIGN(reduced, vectors + 4608);
        TASSIGN(reduced_col, vectors + 4608);
        TASSIGN(delta, vectors + 4608 + kVRows * 4);
        TASSIGN(delta_col, vectors + 4608 + kVRows * 4);
        TASSIGN(norm, scalars);
        TASSIGN(norm_col, scalars);
        TASSIGN(beta_scalar, scalars + 32);
        TASSIGN(beta_bf16, scalars + 64);
        TASSIGN(out, scalars + 96);
    }
};

AICORE inline void LoadBf16(Scratch& s, TileND<float, 1, 128>& dst, __gm__ bfloat16_t* ptr)
{
    Load<bfloat16_t, 1, 128>(s.bf16, ptr);
    TCVT(dst, s.bf16, RoundMode::CAST_NONE);
    VBarrier();
}

AICORE inline void Conv(Scratch& s, const Args& a, const TilingData& t,
                       int64_t row, int64_t token, int64_t begin, int64_t historyOffset,
                       int64_t channel, TileND<float, 1, 128>& dst)
{
    const int64_t channels = 384 * t.heads;
    if (a.conv_bias) {
        LoadBf16(s, s.acc, a.conv_bias + channel);
    } else {
        TEXPANDS(s.acc, 0.0f);
        VBarrier();
    }
    for (int64_t tap = 0; tap < 4; ++tap) {
        const int64_t relative = token - begin + tap - 3;
        __gm__ bfloat16_t* source = relative < 0
            ? a.conv_in + (row * t.conv_history + historyOffset + relative + 3) * channels + channel
            : a.qkv + (begin + relative) * channels + channel;
        LoadBf16(s, s.x, source);
        // vLLM-Ascend Conv loads activation/history through FP16, weights in FP32.
        TCVT(s.fp16, s.x, RoundMode::CAST_RINT);
        VBarrier();
        TCVT(s.x, s.fp16, RoundMode::CAST_NONE);
        LoadBf16(s, s.weight, a.conv_weight + tap * channels + channel);
        TMUL(s.tmp, s.x, s.weight);
        VBarrier();
        TADD(s.acc, s.acc, s.tmp);
        VBarrier();
    }
    TMULS(s.tmp, s.acc, -1.0f);
    VBarrier();
    TEXP(s.tmp, s.tmp);
    VBarrier();
    TADDS(s.tmp, s.tmp, 1.0f);
    VBarrier();
    TDIV(s.acc, s.acc, s.tmp);
    VBarrier();
    // Preserve the public Conv -> recurrence BF16 boundary inside the fusion.
    TCVT(s.bf16, s.acc, RoundMode::CAST_RINT);
    VBarrier();
    TCVT(dst, s.bf16, RoundMode::CAST_NONE);
    VBarrier();
}

AICORE inline void Normalize(Scratch& s, TileND<float, 1, 128>& x, float scale)
{
    TMUL(s.tmp, x, x);
    VBarrier();
    TROWSUM(s.norm_col, s.tmp, s.reduce_work);
    VBarrier();
    TADDS(s.norm, s.norm, 1.0e-6f);
    VBarrier();
    TSQRT(s.norm, s.norm);
    VBarrier();
    TROWEXPAND(s.tmp, s.norm_col);
    VBarrier();
    TDIV(x, x, s.tmp);
    VBarrier();
    TMULS(x, x, scale);
    VBarrier();
}

AICORE inline void Recurrent(Scratch& s, const Args& a, const TilingData& t,
                            int64_t token, int64_t head)
{
    Normalize(s, s.q, 0.08838834764831845f);
    Normalize(s, s.k, 1.0f);
    LoadBf16(s, s.g, a.gate + (token * t.heads + head) * 128);
    Load<float, 1, 128>(s.tmp, a.gate_bias + head * 128);
    TADD(s.g, s.g, s.tmp);
    TEXPANDS(s.tmp, a.a_log[head]);
    VBarrier();
    TEXP(s.tmp, s.tmp);
    VBarrier();
    TMUL(s.g, s.g, s.tmp);
    VBarrier();
    TMULS(s.g, s.g, -1.0f);
    VBarrier();
    TEXP(s.g, s.g);
    VBarrier();
    TADDS(s.g, s.g, 1.0f);
    TEXPANDS(s.tmp, -5.0f);
    VBarrier();
    TDIV(s.g, s.tmp, s.g);
    VBarrier();
    TEXP(s.g, s.g);
    Load<bfloat16_t, 1, 1>(s.beta_bf16, a.beta + token * t.heads + head);
    TCVT(s.beta_scalar, s.beta_bf16, RoundMode::CAST_NONE);
    VBarrier();
    TMULS(s.beta_scalar, s.beta_scalar, -1.0f);
    VBarrier();
    TEXP(s.beta_scalar, s.beta_scalar);
    VBarrier();
    TADDS(s.beta_scalar, s.beta_scalar, 1.0f);
    TEXPANDS(s.norm, 1.0f);
    VBarrier();
    TDIV(s.beta_scalar, s.norm, s.beta_scalar);
    TCOLEXPAND(s.expanded, s.g);
    VBarrier();
    TMUL(s.state, s.state, s.expanded);
    VBarrier();
    TCOLEXPAND(s.expanded, s.k);
    VBarrier();
    TMUL(s.product, s.state, s.expanded);
    VBarrier();
    TROWSUM(s.reduced_col, s.product, s.reduce_work);
    VBarrier();
    TSUB(s.delta, s.value, s.reduced);
    VBarrier();
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const float beta = s.beta_scalar.GetValue(0);
    TMULS(s.delta, s.delta, beta);
    VBarrier();
    TROWEXPAND(s.product, s.delta_col);
    VBarrier();
    TMUL(s.product, s.product, s.expanded);
    VBarrier();
    TADD(s.state, s.state, s.product);
    TCOLEXPAND(s.expanded, s.q);
    VBarrier();
    TMUL(s.product, s.state, s.expanded);
    VBarrier();
    TROWSUM(s.reduced_col, s.product, s.reduce_work);
    VBarrier();
    TCVT(s.out, s.reduced, RoundMode::CAST_RINT);
    VBarrier();
}

template <bool Mtp>
AICORE inline void Run(const Args& a, const TilingData& t)
{
#if defined(__DAV_C220_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    const int64_t channels = 384 * t.heads;
    for (int64_t task = get_block_idx(); task < t.batch * t.heads * kVShards; task += get_block_num()) {
        const int64_t batch = task / (t.heads * kVShards);
        const int64_t head = task / kVShards % t.heads;
        const int64_t vStart = task % kVShards * kVRows;
        const int64_t begin = a.offsets[batch], end = a.offsets[batch + 1];
        const int64_t count = end - begin;
        const int64_t accepted = Mtp ? a.accepted[batch] : 1;
        if (begin < 0 || end > t.tokens || count <= 0 || count > t.max_query_tokens ||
            (!Mtp && (begin != batch || count != 1)) ||
            accepted < 1 || accepted > t.max_query_tokens) {
            continue;
        }
        const int64_t convRead = a.conv_read[batch], convWrite = a.conv_write[batch];
        const int64_t ssmRead = a.ssm_read[batch * t.max_query_tokens + accepted - 1];
        if (convRead < 0 || convRead >= t.conv_input_slots ||
            ssmRead < 0 || ssmRead >= t.ssm_input_slots ||
            convWrite < -1 || convWrite >= t.conv_output_slots) {
            continue;
        }
        bool valid = true;
        for (int64_t i = 0; i < count; ++i) {
            const int64_t slot = a.ssm_write[batch * t.max_query_tokens + i];
            valid = valid && slot >= -1 && slot < t.ssm_output_slots;
        }
        if (!valid) {
            continue;
        }
        Scratch s;
        s.Init(vStart);
        Load<float, kVRows, 128>(s.state, a.ssm_in + (ssmRead * t.heads + head) * 16384 + vStart * 128);
        for (int64_t token = begin; token < end; ++token) {
            Conv(s, a, t, convRead, token, begin, accepted - 1, head * 128, s.q);
            Conv(s, a, t, convRead, token, begin, accepted - 1, (t.heads + head) * 128, s.k);
            Conv(s, a, t, convRead, token, begin, accepted - 1, (2 * t.heads + head) * 128, s.v);
            Recurrent(s, a, t, token, head);
            const int64_t slot = a.ssm_write[batch * t.max_query_tokens + token - begin];
            if (slot >= 0) {
                Store<float, kVRows, 128>(a.ssm_out + (slot * t.heads + head) * 16384 + vStart * 128, s.state);
            }
            Store<bfloat16_t, 1, kVRows>(a.output + (token * t.heads + head) * 128 + vStart, s.out);
        }
        if (vStart == 0 && convWrite >= 0) {
            // Width four: keep two accepted history tokens, then append raw QKV.
            // Ragged MTP writes count+2 positions and preserves the capacity tail.
            for (int64_t part = 0; part < 3; ++part) {
                const int64_t channel = (part * t.heads + head) * 128;
                for (int64_t i = 0; i < count + 2; ++i) {
                    __gm__ bfloat16_t* src = i < 2
                        ? a.conv_in + (convRead * t.conv_history + accepted + i) * channels + channel
                        : a.qkv + (begin + i - 2) * channels + channel;
                    Load<bfloat16_t, 1, 128>(s.bf16, src);
                    Store<bfloat16_t, 1, 128>(a.conv_out + (convWrite * t.conv_history + i) * channels + channel, s.bf16);
                }
            }
        }
    }
    pipe_barrier(PIPE_ALL);
#endif
}
} // namespace mega_kda
