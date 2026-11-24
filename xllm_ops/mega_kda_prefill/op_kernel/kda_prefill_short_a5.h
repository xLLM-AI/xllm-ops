/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

namespace kda_prefill {
#if defined(__DAV_C310_VEC__)
constexpr int kShortRows = 32;
constexpr int kShortShards = 128 / kShortRows;
using ShortColumn = Tile<TileType::Vec, float, kShortRows, 1, BLayout::ColMajor, kShortRows, 1>;

struct ShortScratch {
    FrontScratch f;
    ND<float, kShortRows, 128> state, expanded, product;
    ND<float, kShortRows, 64> reduction;
    ND<float, 1, 128> q, k, v, g;
    ND<float, 1, kShortRows> reduced, delta;
    ShortColumn reduced_column, delta_column;
    ND<float, 1, 8, 1, 1> epsilon;
    ND<bfloat16_t, 1, kShortRows> output;

    AICORE KDA_ARCH_INLINE void Init()
    {
        f.Init();
        TASSIGN(q, 20480); TASSIGN(k, 20992); TASSIGN(v, 21504); TASSIGN(g, 22016);
        TASSIGN(epsilon, 22528);
        TASSIGN(reduced, 22784); TASSIGN(reduced_column, 22784);
        TASSIGN(delta, 23040); TASSIGN(delta_column, 23040); TASSIGN(output, 23296);
        TASSIGN(state, 32768); TASSIGN(expanded, 49152); TASSIGN(product, 65536);
        TASSIGN(reduction, 81920);
        static_assert(81920 + kShortRows * 64 * sizeof(float) <= 256 * 1024,
                      "Short prefill exceeds the A5 UB");
    }
};

template <typename Matrix, typename Scalar>
__tf__ PTO_INTERNAL void ShortScaleNormVf(
    typename Matrix::TileDType __out__ output, typename Scalar::TileDType __out__ epsilon,
    typename Matrix::TileDType __in__ input, typename Scalar::TileDType __in__ maximum)
{
    auto* in = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto* out = (__ubuf__ float*)__cce_get_tile_ptr(output);
    auto* ep = (__ubuf__ float*)__cce_get_tile_ptr(epsilon);
    auto* mp = (__ubuf__ float*)__cce_get_tile_ptr(maximum);
    __VEC_SCOPE__ {
        RegTensor<float> x, scaled, m, e, zero;
        MaskReg mask = PSetWithType<float>(PAT_ALL), large;
        uint32_t one = 1;
        MaskReg scalar_mask = CreatePredicate<float>(one);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        vlds(m, mp, 0, BRC_B32); vcmps_gt(large, m, 0x1p60f, mask);
        vbr(e, 1.0e-6f); vbr(zero, 0.0f); vsel(e, zero, e, large);
        vsts(e, ep, 0, dist, scalar_mask);
        for (uint16_t half = 0; half < 2; ++half) {
            vlds(x, in, half * 64, NORM);
            vmuls(scaled, x, 0x1p-68f, mask, MODE_ZEROING);
            vsel(x, scaled, x, large); vsts(x, out, half * 64, dist, mask);
        }
    }
}

AICORE KDA_ARCH_INLINE void ShortParameters(ShortScratch& s, const Args& a,
                                           const TilingData& t, int64_t head)
{
    ND<float, 1, 128> parameter;
    for (int part = 0; part < 3; ++part) {
        const int64_t channel = (part * t.heads + head) * 128;
        for (int tap = 0; tap < 4; ++tap) {
            TASSIGN(parameter, 8192 + (part * 4 + tap) * 512);
            s.f.BfLoad(parameter, a.weight + tap * t.heads * 384 + channel);
        }
        TASSIGN(parameter, 14336 + part * 512);
        if (a.bias) s.f.BfLoad(parameter, a.bias + channel);
        else { TEXPANDS(parameter, 0.0f); PipeBarrierVec(); }
    }
    TASSIGN(parameter, 16384); Load(parameter, a.gate_bias + head * 128, 1, 128, 128);
    TASSIGN(parameter, 16896); TEXPANDS(parameter, a.a_log[head]); PipeBarrierVec();
    TEXP(parameter, parameter); PipeBarrierVec();
}

AICORE KDA_ARCH_INLINE void ShortNormalize(ShortScratch& s, ND<float, 1, 128>& destination)
{
    TABS(s.f.tmp, s.f.acc); PipeBarrierVec();
    TROWMAX(s.f.column, s.f.tmp, s.f.reduce); PipeBarrierVec();
    ShortScaleNormVf<decltype(s.f.acc), decltype(s.f.scalar)>(
        s.f.acc.data(), s.epsilon.data(), s.f.acc.data(), s.f.scalar.data());
    TMUL(s.f.tmp, s.f.acc, s.f.acc); PipeBarrierVec();
    TROWSUM(s.f.column, s.f.tmp, s.f.reduce); PipeBarrierVec();
    TADD(s.f.scalar, s.f.scalar, s.epsilon); PipeBarrierVec();
    TRSQRT<RsqrtAlgorithm::DEFAULT>(s.f.scalar, s.f.scalar); PipeBarrierVec();
    TROWEXPAND(s.f.norm_broadcast, s.f.column); PipeBarrierVec();
    TMUL(s.f.acc, s.f.acc, s.f.norm_broadcast); PipeBarrierVec();
    TCVT(s.f.bf, s.f.acc, RoundMode::CAST_RINT); PipeBarrierVec();
    TCVT(destination, s.f.bf, RoundMode::CAST_NONE); PipeBarrierVec();
}

AICORE KDA_ARCH_INLINE void ShortCompute(const Args& a, const TilingData& t)
{
    set_mask_norm(); set_vector_mask(-1, -1);
    const int64_t core = get_block_idx() * 2 + get_subblockid(), cores = get_block_num() * 2;
    for (int64_t task = core; task < t.heads * kShortShards; task += cores) {
        const int64_t head = task / kShortShards, first_v = task % kShortShards * kShortRows;
        ShortScratch s; s.Init(); ShortParameters(s, a, t, head);
        if (a.ssm_read[0] < 0) { TEXPANDS(s.state, 0.0f); PipeBarrierVec(); }
        else Load(s.state, a.ssm_in + (a.ssm_read[0] * t.heads + head) * 16384 + first_v * 128,
                  kShortRows, 128, 128);
        for (int64_t token = 0; token < t.tokens; ++token) {
            Conv(s.f, a, t, 0, token, head * kHeadDim); ShortNormalize(s, s.q);
            Conv(s.f, a, t, 0, token, (t.heads + head) * kHeadDim); ShortNormalize(s, s.k);
            Conv(s.f, a, t, 0, token, (2 * t.heads + head) * kHeadDim);
            TMOV(s.v, s.f.acc); PipeBarrierVec();
            s.f.BfLoad(s.g, a.gate + (token * t.heads + head) * 128);
            ND<float, 1, 128> parameter;
            TASSIGN(parameter, 16384); TADD(s.g, s.g, parameter); PipeBarrierVec();
            TASSIGN(parameter, 16896); TMUL(s.g, s.g, parameter); PipeBarrierVec();
            TMULS(s.g, s.g, -1.0f); PipeBarrierVec(); TEXP(s.g, s.g); PipeBarrierVec();
            TADDS(s.g, s.g, 1.0f); TEXPANDS(s.f.tmp, -5.0f); PipeBarrierVec();
            TDIV(s.g, s.f.tmp, s.g); PipeBarrierVec(); TEXP(s.g, s.g); PipeBarrierVec();
            TCOLEXPAND(s.expanded, s.g); PipeBarrierVec();
            TMUL(s.state, s.state, s.expanded); PipeBarrierVec();
            TCOLEXPAND(s.expanded, s.k); PipeBarrierVec();
            TMUL(s.product, s.state, s.expanded); PipeBarrierVec();
            TROWSUM(s.reduced_column, s.product, s.reduction); PipeBarrierVec();
            ND<float, 1, kShortRows> value;
            TASSIGN(value, 21504 + first_v * sizeof(float));
            TSUB(s.delta, value, s.reduced); PipeBarrierVec();
            TMULS(s.delta, s.delta, a.beta[token * t.heads + head]); PipeBarrierVec();
            TROWEXPAND(s.product, s.delta_column); PipeBarrierVec();
            TMUL(s.product, s.product, s.expanded); PipeBarrierVec();
            TADD(s.state, s.state, s.product); PipeBarrierVec();
            TCOLEXPAND(s.expanded, s.q); PipeBarrierVec();
            TMUL(s.product, s.state, s.expanded); PipeBarrierVec();
            TROWSUM(s.reduced_column, s.product, s.reduction); PipeBarrierVec();
            TMULS(s.reduced, s.reduced, 0.08838834764831845f); PipeBarrierVec();
            TCVT(s.output, s.reduced, RoundMode::CAST_RINT); PipeBarrierVec();
            Store(a.output + (token * t.heads + head) * 128 + first_v, s.output, 1, kShortRows, kShortRows);
        }
        if (a.ssm_write[0] >= 0) {
            Store(a.ssm_out + (a.ssm_write[0] * t.heads + head) * 16384 + first_v * 128,
                  s.state, kShortRows, 128, 128);
        }
    }
}

AICORE KDA_ARCH_INLINE void ShortPublishConv(const Args& a, const TilingData& t)
{
    const int64_t core = get_block_idx() * 2 + get_subblockid(), cores = get_block_num() * 2;
    if (a.conv_write[0] < 0) return;
    const int64_t channels = t.heads * 384;
    FrontScratch f; f.Init();
    for (int64_t head = core; head < t.heads; head += cores) {
        for (int part = 0; part < 3; ++part) {
            const int64_t channel = (part * t.heads + head) * 128;
            for (int index = 0; index < 3; ++index) {
                const int64_t position = t.tokens - 3 + index;
                if (position < 0 && a.conv_read[0] < 0) {
                    TEXPANDS(f.tmp, 0.0f); PipeBarrierVec();
                    TCVT(f.bf, f.tmp, RoundMode::CAST_RINT); PipeBarrierVec();
                }
                else {
                    auto* input = position < 0 ? a.conv_in + (a.conv_read[0] * 3 + position + 3) * channels + channel
                                               : a.qkv + position * channels + channel;
                    Load(f.bf, input, 1, 128, 128);
                }
                Store(a.conv_out + (a.conv_write[0] * 3 + index) * channels + channel, f.bf, 1, 128, 128);
            }
        }
    }
}
#endif

AICORE KDA_ARCH_INLINE void RunShortA5(const Args& a, const TilingData& t)
{
#if defined(__DAV_C310_VEC__)
    ShortCompute(a, t);
#endif
    // All shards finish reading history before any aliased Conv publication.
    Rendezvous();
#if defined(__DAV_C310_VEC__)
    ShortPublishConv(a, t);
#endif
    Rendezvous();
}
} // namespace kda_prefill
