/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

// A5-only implementation.  Keep the legacy implementation in the including
// header's non-A5 branch so A2/A3 preprocessing and objects stay unchanged.
template <int Rows>
struct Scratch {
    static_assert(Rows == 32 || Rows == 64, "A5 decode supports only 32/64 V rows");
    static constexpr int kStateBytes = Rows * 128 * sizeof(float);
    static constexpr int kVectors = 3 * kStateBytes + Rows * 64 * sizeof(float);
    static constexpr int kScalars = kVectors + 4608 + 2 * Rows * sizeof(float);
    static constexpr int kParameterBase = (kScalars + 96 + Rows * sizeof(bfloat16_t) + 511) / 512 * 512;
    static constexpr int kUbEnd = kParameterBase + 17 * 512;
    static_assert(kUbEnd <= 262144, "A5 decode Scratch exceeds 256 KiB UB");

    TileND<float, Rows, 128> state, expanded, product;
    TileND<float, Rows, 64> reduce_work;
    TileND<float, 1, 128> q, k, v, g, tmp, x, weight, acc;
    TileND<float, 1, 128> parameter, gate_bias_cached, a_exp_cached;
    TileND<bfloat16_t, 1, 128> bf16;
    TileND<half, 1, 128> fp16;
    TileND<float, 1, Rows> reduced, delta, value;
    Column<Rows> reduced_col, delta_col;
    Column<8, 1> norm_col;
    TileND<float, 1, 8, 1, 1> norm, beta_scalar;
    TileND<bfloat16_t, 1, 16, 1, 1> beta_bf16;
    TileND<bfloat16_t, 1, Rows> out;

    AICORE inline void Init(int64_t vStart)
    {
        constexpr int stateBytes = kStateBytes;
        constexpr int vectors = kVectors;
        constexpr int scalars = kScalars;
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
        TASSIGN(delta, vectors + 4608 + Rows * 4);
        TASSIGN(delta_col, vectors + 4608 + Rows * 4);
        TASSIGN(norm, scalars);
        TASSIGN(norm_col, scalars);
        TASSIGN(beta_scalar, scalars + 32);
        TASSIGN(beta_bf16, scalars + 64);
        TASSIGN(out, scalars + 96);
        TASSIGN(gate_bias_cached, kParameterBase + 15 * 512);
        TASSIGN(a_exp_cached, kParameterBase + 16 * 512);
    }
};

template <int Rows>
AICORE inline void LoadBf16(Scratch<Rows>& s, TileND<float, 1, 128>& dst,
                            __gm__ bfloat16_t* ptr)
{
    load<bfloat16_t, 1, 128>(s.bf16, ptr);
    TCVT(dst, s.bf16, RoundMode::CAST_NONE);
    v_barrier();
}

template <typename TileT>
__tf__ PTO_INTERNAL void SiluStableVf(typename TileT::TileDType __out__ output,
                                     typename TileT::TileDType __in__ input)
{
    auto* source = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto* destination = (__ubuf__ float*)__cce_get_tile_ptr(output);
    __VEC_SCOPE__ {
        RegTensor<float> x, exponential, denominator, numerator, result, positive_result;
        MaskReg mask = PSetWithType<float>(PAT_ALL), positive, large;
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            vlds(x, source, half * 64, NORM);
            vmuls(exponential, x, -1.0f, mask, MODE_ZEROING);
            vexp(exponential, exponential, mask, MODE_ZEROING);
            vadds(denominator, exponential, 1.0f, mask, MODE_ZEROING);
            vmul(numerator, x, exponential, mask, MODE_ZEROING);
            vcmps_gt(positive, x, 8.0f, mask);
            vsel(numerator, numerator, x, positive);
            vdiv(result, numerator, denominator, mask, MODE_ZEROING);
            vsub(positive_result, x, result, mask, MODE_ZEROING);
            vsel(result, positive_result, result, positive);
            // SiLU rounds to x in FP32 here; also preserve positive infinity.
            vcmps_gt(large, x, 80.0f, mask);
            vsel(result, x, result, large);
            vsts(result, destination, half * 64, dist, mask);
        }
    }
}

template <int Rows>
AICORE inline void SiluStable(Scratch<Rows>& s)
{
    // For positive x use x - x*exp(-x)/(1+exp(-x)); its small quotient
    // avoids losing the activation tail when the denominator rounds near 1.
    // All arithmetic is ordinary FP32, with one exp and one division.
    SiluStableVf<decltype(s.acc)>(s.acc.data(), s.acc.data());
    v_barrier();
}

template <int Rows>
AICORE inline void LoadParameters(Scratch<Rows>& s, const Args& a, const TilingData& t,
                                  int64_t head)
{
    const int64_t channels = 384 * t.heads;
    for (int64_t part = 0; part < 3; ++part) {
        const int64_t channel = (part * t.heads + head) * 128;
        for (int64_t tap = 0; tap < 4; ++tap) {
            TASSIGN(s.parameter, Scratch<Rows>::kParameterBase + (part * 4 + tap) * 512);
            LoadBf16(s, s.parameter, a.conv_weight + tap * channels + channel);
        }
        TASSIGN(s.parameter, Scratch<Rows>::kParameterBase + (12 + part) * 512);
        if (a.conv_bias) {
            LoadBf16(s, s.parameter, a.conv_bias + channel);
        } else {
            TEXPANDS(s.parameter, 0.0f);
            v_barrier();
        }
    }
    load<float, 1, 128>(s.gate_bias_cached, a.gate_bias + head * 128);
    TEXPANDS(s.a_exp_cached, a.a_log[head]);
    v_barrier();
    TEXP(s.a_exp_cached, s.a_exp_cached);
    v_barrier();
}

template <bool Mtp, int Rows>
AICORE inline void Conv(Scratch<Rows>& s, const Args& a, const TilingData& t,
                        int64_t row, int64_t token, int64_t begin, int64_t historyOffset,
                        int64_t channel, TileND<float, 1, 128>& dst)
{
    const int64_t channels = 384 * t.heads;
    const int64_t part = channel / (t.heads * 128);
    if constexpr (Mtp) {
        TASSIGN(s.parameter, Scratch<Rows>::kParameterBase + (12 + part) * 512);
        TMOV(s.acc, s.parameter);
        v_barrier();
    } else if (a.conv_bias) {
        LoadBf16(s, s.acc, a.conv_bias + channel);
    } else {
        TEXPANDS(s.acc, 0.0f);
        v_barrier();
    }
    for (int64_t tap = 0; tap < 4; ++tap) {
        const int64_t relative = token - begin + tap - 3;
        __gm__ bfloat16_t* source = relative < 0
            ? a.conv_in + (row * t.conv_history + historyOffset + relative + 3) * channels + channel
            : a.qkv + (begin + relative) * channels + channel;
        LoadBf16(s, s.x, source);
        if constexpr (Mtp) {
            TCVT(s.fp16, s.x, RoundMode::CAST_RINT);
            v_barrier();
            TCVT(s.x, s.fp16, RoundMode::CAST_NONE);
            TASSIGN(s.parameter, Scratch<Rows>::kParameterBase + (part * 4 + tap) * 512);
            TMUL(s.tmp, s.x, s.parameter);
        } else {
            LoadBf16(s, s.weight, a.conv_weight + tap * channels + channel);
            TMUL(s.tmp, s.x, s.weight);
        }
        v_barrier();
        TADD(s.acc, s.acc, s.tmp);
        v_barrier();
    }
    if constexpr (Mtp) {
        // xLLM's speculative Conv1d V2 writes BF16 before applying SiLU.
        TCVT(s.bf16, s.acc, RoundMode::CAST_RINT);
        v_barrier();
        TCVT(s.acc, s.bf16, RoundMode::CAST_NONE);
        v_barrier();
    }
    SiluStable(s);
    TCVT(s.bf16, s.acc, RoundMode::CAST_RINT);
    v_barrier();
    TCVT(dst, s.bf16, RoundMode::CAST_NONE);
    v_barrier();
}

template <int Rows>
AICORE inline void Normalize(Scratch<Rows>& s, TileND<float, 1, 128>& x, float scale)
{
    TMUL(s.tmp, x, x);
    v_barrier();
    TROWSUM(s.norm_col, s.tmp, s.reduce_work);
    v_barrier();
    TADDS(s.norm, s.norm, 1.0e-6f);
    v_barrier();
    TSQRT(s.norm, s.norm);
    v_barrier();
    TROWEXPAND(s.tmp, s.norm_col);
    v_barrier();
    TDIV(x, x, s.tmp);
    v_barrier();
    TMULS(x, x, scale);
    v_barrier();
}

// A5-specific VF path for a column broadcast followed by FP32 multiply.
// load each 64-lane column chunk once, then reuse it for every V row.  The
// generic TCOLEXPANDMUL implementation reloads the broadcast source inside
// its row loop, which can cost more than the UB round trip it removes here.
template <typename DstTile, typename SrcTile, typename ColumnTile>
__tf__ PTO_INTERNAL void ColExpandMulReuseVf(
    typename DstTile::TileDType __out__ dst, typename SrcTile::TileDType __in__ src,
    typename ColumnTile::TileDType __in__ column, uint16_t rows)
{
    auto* dstPtr = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    auto* srcPtr = (__ubuf__ float*)__cce_get_tile_ptr(src);
    auto* columnPtr = (__ubuf__ float*)__cce_get_tile_ptr(column);
    __VEC_SCOPE__ {
        RegTensor<float> columnReg, srcReg, dstReg;
        MaskReg mask = PSetWithType<float>(PAT_ALL);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            const uint32_t columnOffset = half * 64;
            vlds(columnReg, columnPtr, columnOffset, NORM);
            for (uint16_t row = 0; row < rows; ++row) {
                const uint32_t offset = row * 128 + columnOffset;
                vlds(srcReg, srcPtr, offset, NORM);
                vmul(dstReg, srcReg, columnReg, mask, MODE_ZEROING);
                vsts(dstReg, dstPtr, offset, dist, mask);
            }
        }
    }
}

// Direct row-by-column product for the recurrent update.  load the K vector
// once per 64-lane chunk, then form the outer product with one delta scalar
// per row; no expanded K or expanded delta tile is materialized in UB.
template <typename DstTile, typename KeyTile, typename RowScaleTile>
__tf__ PTO_INTERNAL void RowMulKReuseVf(
    typename DstTile::TileDType __out__ dst, typename KeyTile::TileDType __in__ key,
    typename RowScaleTile::TileDType __in__ rowScale, uint16_t rows)
{
    auto* dstPtr = (__ubuf__ float*)__cce_get_tile_ptr(dst);
    auto* keyPtr = (__ubuf__ float*)__cce_get_tile_ptr(key);
    auto* rowScalePtr = (__ubuf__ float*)__cce_get_tile_ptr(rowScale);
    __VEC_SCOPE__ {
        RegTensor<float> rowReg, keyReg, dstReg;
        MaskReg mask = PSetWithType<float>(PAT_ALL);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            const uint32_t columnOffset = half * 64;
            vlds(keyReg, keyPtr, columnOffset, NORM);
            for (uint16_t row = 0; row < rows; ++row) {
                const uint32_t offset = row * 128 + columnOffset;
                vlds(rowReg, rowScalePtr, row, BRC_B32);
                vmul(dstReg, keyReg, rowReg, mask, MODE_ZEROING);
                vsts(dstReg, dstPtr, offset, dist, mask);
            }
        }
    }
}

template <int Rows>
AICORE inline void ColExpandMulReuse(
    TileND<float, Rows, 128>& dst, TileND<float, Rows, 128>& src, TileND<float, 1, 128>& column)
{
    ColExpandMulReuseVf<TileND<float, Rows, 128>, TileND<float, Rows, 128>, TileND<float, 1, 128>>(
        dst.data(), src.data(), column.data(), Rows);
}

template <int Rows>
AICORE inline void RowMulKReuse(
    TileND<float, Rows, 128>& dst, TileND<float, 1, 128>& key, Column<Rows>& rowScale)
{
    RowMulKReuseVf<TileND<float, Rows, 128>, TileND<float, 1, 128>, Column<Rows>>(
        dst.data(), key.data(), rowScale.data(), Rows);
}

template <typename StateTile, typename KeyTile, typename RowScaleTile>
__tf__ PTO_INTERNAL void StateFmaVf(
    typename StateTile::TileDType __out__ output, typename StateTile::TileDType __in__ input,
    typename KeyTile::TileDType __in__ key, typename RowScaleTile::TileDType __in__ row_scale,
    uint16_t rows)
{
    auto* out = (__ubuf__ float*)__cce_get_tile_ptr(output);
    auto* state = (__ubuf__ float*)__cce_get_tile_ptr(input);
    auto* keys = (__ubuf__ float*)__cce_get_tile_ptr(key);
    auto* scales = (__ubuf__ float*)__cce_get_tile_ptr(row_scale);
    __VEC_SCOPE__ {
        RegTensor<float> value, key_reg, scale_reg;
        MaskReg mask = PSetWithType<float>(PAT_ALL);
        constexpr auto dist = std::integral_constant<::DistVST,
            static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
        for (uint16_t half = 0; half < 2; ++half) {
            vlds(key_reg, keys, half * 64, NORM);
            for (uint16_t row = 0; row < rows; ++row) {
                const uint32_t offset = row * 128 + half * 64;
                vlds(value, state, offset, NORM);
                vlds(scale_reg, scales, row, BRC_B32);
                vmula(value, key_reg, scale_reg, mask, MODE_ZEROING);
                vsts(value, out, offset, dist, mask);
            }
        }
    }
}

template <int Rows>
AICORE inline void StateFma(Scratch<Rows>& s)
{
    StateFmaVf<decltype(s.state), decltype(s.k), decltype(s.delta_col)>(
        s.state.data(), s.state.data(), s.k.data(), s.delta_col.data(), Rows);
}

template <bool Cached, int Rows>
AICORE inline void Recurrent(Scratch<Rows>& s, const Args& a, const TilingData& t,
                             int64_t token, int64_t head)
{
    Normalize(s, s.q, 0.08838834764831845f);
    Normalize(s, s.k, 1.0f);
    LoadBf16(s, s.g, a.gate + (token * t.heads + head) * 128);
    if constexpr (Cached) {
        TADD(s.g, s.g, s.gate_bias_cached);
        v_barrier();
        TMUL(s.g, s.g, s.a_exp_cached);
    } else {
        load<float, 1, 128>(s.tmp, a.gate_bias + head * 128);
        TADD(s.g, s.g, s.tmp);
        TEXPANDS(s.tmp, a.a_log[head]);
        v_barrier();
        TEXP(s.tmp, s.tmp);
        v_barrier();
        TMUL(s.g, s.g, s.tmp);
    }
    v_barrier();
    TMULS(s.g, s.g, -1.0f);
    v_barrier();
    TEXP(s.g, s.g);
    v_barrier();
    TADDS(s.g, s.g, 1.0f);
    TEXPANDS(s.tmp, -5.0f);
    v_barrier();
    TDIV(s.g, s.tmp, s.g);
    v_barrier();
    TEXP(s.g, s.g);
    load<bfloat16_t, 1, 1>(s.beta_bf16, a.beta + token * t.heads + head);
    TCVT(s.beta_scalar, s.beta_bf16, RoundMode::CAST_NONE);
    v_barrier();
    TMULS(s.beta_scalar, s.beta_scalar, -1.0f);
    v_barrier();
    TEXP(s.beta_scalar, s.beta_scalar);
    v_barrier();
    TADDS(s.beta_scalar, s.beta_scalar, 1.0f);
    TEXPANDS(s.norm, 1.0f);
    v_barrier();
    TDIV(s.beta_scalar, s.norm, s.beta_scalar);
    // Keep each column chunk in a VF register and use a standalone vmul
    // (rather than a fused multiply-add) to preserve the original FP32
    // operation and order without the expanded UB tile.
    ColExpandMulReuse(s.state, s.state, s.g);
    v_barrier();
    ColExpandMulReuse(s.product, s.state, s.k);
    v_barrier();
    TROWSUM(s.reduced_col, s.product, s.reduce_work);
    v_barrier();
    TSUB(s.delta, s.value, s.reduced);
    v_barrier();
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
    const float beta = s.beta_scalar.GetValue(0);
    TMULS(s.delta, s.delta, beta);
    v_barrier();
    // One FP32 FMA avoids the rounded outer-product temporary.
    StateFma(s);
    v_barrier();
    ColExpandMulReuse(s.product, s.state, s.q);
    v_barrier();
    TROWSUM(s.reduced_col, s.product, s.reduce_work);
    v_barrier();
    TCVT(s.out, s.reduced, RoundMode::CAST_RINT);
    v_barrier();
}

template <bool Mtp, int Rows>
AICORE inline void RunImpl(const Args& a, const TilingData& t)
{
#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    constexpr int kVShards = 128 / Rows;
    set_mask_norm();
    set_vector_mask(-1, -1);
    const int64_t channels = 384 * t.heads;
    for (int64_t task = get_block_idx(); task < t.batch * t.heads * kVShards; task += get_block_num()) {
        const int64_t batch = task / (t.heads * kVShards);
        const int64_t head = task / kVShards % t.heads;
        const int64_t vStart = task % kVShards * Rows;
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
        Scratch<Rows> s;
        s.Init(vStart);
        if constexpr (Mtp) {
            // These parameters are immutable across this sequence's tokens.
            LoadParameters(s, a, t, head);
        }
        load<float, Rows, 128>(s.state, a.ssm_in + (ssmRead * t.heads + head) * 16384 + vStart * 128);
        for (int64_t token = begin; token < end; ++token) {
            Conv<Mtp>(s, a, t, convRead, token, begin, accepted - 1, head * 128, s.q);
            Conv<Mtp>(s, a, t, convRead, token, begin, accepted - 1, (t.heads + head) * 128, s.k);
            Conv<Mtp>(s, a, t, convRead, token, begin, accepted - 1, (2 * t.heads + head) * 128, s.v);
            Recurrent<Mtp>(s, a, t, token, head);
            const int64_t slot = a.ssm_write[batch * t.max_query_tokens + token - begin];
            if (slot >= 0) {
                store<float, Rows, 128>(a.ssm_out + (slot * t.heads + head) * 16384 + vStart * 128, s.state);
            }
            store<bfloat16_t, 1, Rows>(a.output + (token * t.heads + head) * 128 + vStart, s.out);
        }
        if (vStart == 0 && convWrite >= 0) {
            for (int64_t part = 0; part < 3; ++part) {
                const int64_t channel = (part * t.heads + head) * 128;
                for (int64_t i = 0; i < count + 2; ++i) {
                    __gm__ bfloat16_t* src = i < 2
                        ? a.conv_in + (convRead * t.conv_history + accepted + i) * channels + channel
                        : a.qkv + (begin + i - 2) * channels + channel;
                    load<bfloat16_t, 1, 128>(s.bf16, src);
                    store<bfloat16_t, 1, 128>(a.conv_out + (convWrite * t.conv_history + i) * channels + channel, s.bf16);
                }
            }
        }
    }
    pipe_barrier(PIPE_ALL);
#endif
}

template <bool Mtp>
AICORE inline void Run(const Args& a, const TilingData& t)
{
#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    // The host still provisions min(AIV, BH*8) blocks.  Rows64 owns two
    // tasks per BH, so select it only when those tasks fill the provisioned
    // blocks; otherwise retain Rows32's better small-BH parallelism.
    if (t.batch * t.heads * 2 >= get_block_num()) {
        RunImpl<Mtp, 64>(a, t);
    } else {
        RunImpl<Mtp, 32>(a, t);
    }
#endif
}
