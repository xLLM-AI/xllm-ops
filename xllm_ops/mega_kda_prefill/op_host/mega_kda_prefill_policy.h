/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include <initializer_list>

#include "../op_kernel/mega_kda_prefill_tiling_data.h"

namespace kda_prefill {
constexpr int64_t chunk_size_for_soc(bool is_a5) {
  return is_a5 ? kChunkSize : kAscend910bSmallChunkSize;
}

inline bool valid_tiling(const TilingData& t, bool is_a5 = false) {
  // Bound all products before constructing workspace offsets or int32 metadata.
  if (t.batch < 1 || t.batch > 1024 || t.tokens < 1 || t.tokens > 131072 ||
      t.heads < 1 || t.heads > 128 || t.blocks < 1 || t.blocks > 32 ||
      (!is_a5 && !t.ffts_addr)) {
    return false;
  }
  for (int64_t n : {t.conv_input_slots,
                    t.conv_output_slots,
                    t.ssm_input_slots,
                    t.ssm_output_slots}) {
    if (n < 1 || n > 1048576) {
      return false;
    }
  }
  return Workspace(t, chunk_size_for_soc(is_a5)).bytes <= (uint64_t{32} << 30);
}
}  // namespace kda_prefill
