/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

#include <cstdint>

namespace mega_kda {
constexpr uint64_t kPlainKey = 1;
constexpr uint64_t kMtpKey = 2;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kValueTile = 16;

// Runtime offsets, accepted counts and slot IDs must remain device inputs.
struct TilingData {
    int64_t batch;
    int64_t tokens;
    int64_t heads;
    int64_t max_query_tokens;
    int64_t conv_history;
    int64_t conv_input_slots;
    int64_t conv_output_slots;
    int64_t ssm_input_slots;
    int64_t ssm_output_slots;
};
static_assert(sizeof(TilingData) == 72, "MegaKdaDecode tiling ABI changed");

enum Input : uint32_t {
    kQkv, kGate, kBeta, kALog, kGateBias, kConvWeight, kConvBias,
    kConvStateIn, kSsmStateIn, kCuSeqlens, kConvReadIndices,
    kConvWriteIndices, kSsmReadIndices, kSsmWriteIndices, kAcceptedTokens,
    kInputCount
};
enum Attr : uint32_t { kMode, kMaxQueryTokens, kConvOutputSlots, kSsmOutputSlots };
} // namespace mega_kda
