/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

#include <cstdint>

namespace mega_kda {
constexpr uint64_t kPlainKey = 1;
constexpr uint64_t kMtpKey = 2;
constexpr uint64_t kPublicCompatPlainKey = 3;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kValueTile = 16;
constexpr int64_t kQkvParts = 3;
constexpr int64_t kConvTaps = 4;
constexpr int64_t kConvHistory = kConvTaps - 1;
constexpr int64_t kBf16Bytes = sizeof(uint16_t);
constexpr int64_t kFloatBytes = sizeof(float);

// PUBLIC_COMPAT preserves Conv conversion, zero-norm, gate operation and
// 32-row reduction semantics. The external attribute values stay unchanged.
enum class DecodeMode : int64_t { NATIVE = 0, MTP = 1, PUBLIC_COMPAT = 3 };

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

enum class Input : uint32_t {
  QKV,
  GATE,
  BETA,
  A_LOG,
  GATE_BIAS,
  CONV_WEIGHT,
  CONV_BIAS,
  CONV_STATE_IN,
  SSM_STATE_IN,
  CU_SEQLENS,
  CONV_READ_INDICES,
  CONV_WRITE_INDICES,
  SSM_READ_INDICES,
  SSM_WRITE_INDICES,
  ACCEPTED_TOKENS,
  INPUT_COUNT
};
enum class Attr : uint32_t {
  MODE,
  MAX_QUERY_TOKENS,
  CONV_OUTPUT_SLOTS,
  SSM_OUTPUT_SLOTS
};
}  // namespace mega_kda
