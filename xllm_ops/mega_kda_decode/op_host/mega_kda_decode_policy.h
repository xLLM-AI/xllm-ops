/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

#include <initializer_list>
#include <limits>

#include "../op_kernel/mega_kda_decode_tiling_data.h"

namespace mega_kda {
inline bool fits_bytes(std::initializer_list<int64_t> dims) {
  int64_t bytes = 1;
  for (int64_t dim : dims) {
    if (dim < 0 || (dim && bytes > std::numeric_limits<int64_t>::max() / dim)) {
      return false;
    }
    bytes *= dim;
  }
  return true;
}

inline uint64_t select_key(int64_t mode,
                           bool has_accepted,
                           const TilingData& t) {
  for (int64_t dim : {t.batch,
                      t.tokens,
                      t.heads,
                      t.max_query_tokens,
                      t.conv_history,
                      t.conv_input_slots,
                      t.conv_output_slots,
                      t.ssm_input_slots,
                      t.ssm_output_slots}) {
    if (dim < 1 || dim > std::numeric_limits<int32_t>::max()) {
      return 0;
    }
  }
  // MTP16 verifies the current token plus sixteen speculative tokens.
  if (t.max_query_tokens > 17 || t.heads > 128 ||
      t.tokens > t.batch * t.max_query_tokens ||
      !fits_bytes({t.tokens, t.heads, kQkvParts * kHeadDim, kBf16Bytes}) ||
      !fits_bytes({t.conv_input_slots,
                   t.conv_history,
                   t.heads,
                   kQkvParts * kHeadDim,
                   kBf16Bytes}) ||
      !fits_bytes({t.conv_output_slots,
                   t.conv_history,
                   t.heads,
                   kQkvParts * kHeadDim,
                   kBf16Bytes}) ||
      !fits_bytes(
          {t.ssm_input_slots, t.heads, kHeadDim, kHeadDim, kFloatBytes}) ||
      !fits_bytes(
          {t.ssm_output_slots, t.heads, kHeadDim, kHeadDim, kFloatBytes})) {
    return 0;
  }
  if ((mode == static_cast<int64_t>(DecodeMode::NATIVE) ||
       mode == static_cast<int64_t>(DecodeMode::PUBLIC_COMPAT)) &&
      !has_accepted && t.max_query_tokens == 1 && t.tokens == t.batch &&
      t.conv_history == kConvHistory) {
    return mode == static_cast<int64_t>(DecodeMode::PUBLIC_COMPAT)
               ? kPublicCompatPlainKey
               : kPlainKey;
  }
  if (mode == static_cast<int64_t>(DecodeMode::MTP) && has_accepted &&
      t.conv_history == t.max_query_tokens + 2) {
    return kMtpKey;
  }
  return 0;
}
}  // namespace mega_kda
