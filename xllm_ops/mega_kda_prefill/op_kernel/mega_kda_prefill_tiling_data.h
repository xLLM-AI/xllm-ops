/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include <cstdint>
#ifdef __CCE_AICORE__
#define KDA_HOST_DEVICE __aicore__
#else
#define KDA_HOST_DEVICE
#endif

namespace kda_prefill {
constexpr int64_t kHeadDim = 128;
enum Input : unsigned {
    kQkv, kGate, kBeta, kALog, kGateBias, kConvWeight, kConvBias,
    kConvStateIn, kSsmStateIn, kCuSeqlens, kConvReadIndices,
    kConvWriteIndices, kSsmReadIndices, kSsmWriteIndices, kInputCount
};
enum Attr : unsigned { kConvOutputSlots, kSsmOutputSlots, kFftsAddr };
struct TilingData {
    int64_t batch, tokens, heads;
    int64_t conv_input_slots, conv_output_slots, ssm_input_slots, ssm_output_slots;
    uint64_t blocks, ffts_addr;
};

// Shared host/device layout. Every buffer starts on a DMA-safe boundary.
struct Workspace {
    uint64_t q, k, beta, g, value, lower, inverse, aqk, u, w, corrected, output;
    uint64_t snapshots, initial, final, wy_a, wy_k, h_ws, o_ws, masks, eye, bytes;
    KDA_HOST_DEVICE static inline uint64_t Align(uint64_t n) { return (n + 511) / 512 * 512; }
    KDA_HOST_DEVICE explicit inline Workspace(const TilingData& t) : bytes(0)
    {
        const uint64_t n = static_cast<uint64_t>(t.tokens) * t.heads;
        const uint64_t state = static_cast<uint64_t>(t.batch) * t.heads * 16384 * 4;
        const uint64_t chunks = (t.tokens + 127) / 128 + t.batch - 1;
        q = Take(n * 128 * 2); k = Take(n * 128 * 2);
        beta = Take(n * 4); g = Take(n * 128 * 4); value = Take(n * 128 * 4);
        lower = Take(n * 128 * 4); inverse = Take(n * 128 * 4); aqk = Take(n * 128 * 4);
        u = Take(n * 128 * 4); w = Take(n * 128 * 4);
        corrected = Take(n * 128 * 4); output = Take(n * 128 * 4);
        snapshots = Take(chunks * t.heads * 16384 * 4);
        initial = Take(state); final = Take(state);
        wy_a = Take(t.blocks * 16384 * 4); wy_k = Take(t.blocks * 16384 * 4);
        h_ws = Take(t.blocks * 5 * 16384 * 4); o_ws = Take(t.blocks * 7 * 16384 * 4);
        masks = Take(2 * 16384 * 4); eye = Take(16384 * 4);
    }
private:
    KDA_HOST_DEVICE inline uint64_t Take(uint64_t n) { const auto p = bytes; bytes += Align(n); return p; }
};
} // namespace kda_prefill
#undef KDA_HOST_DEVICE
