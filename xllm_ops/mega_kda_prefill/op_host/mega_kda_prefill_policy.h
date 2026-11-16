/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include <initializer_list>
#include "../op_kernel/mega_kda_prefill_tiling_data.h"

namespace kda_prefill {
inline bool ValidTiling(const TilingData& t)
{
    // Bound all products before constructing workspace offsets or int32 metadata.
    if (t.batch < 1 || t.batch > 1024 || t.tokens < 1 || t.tokens > 131072 ||
        t.heads < 1 || t.heads > 128 || t.blocks < 1 || t.blocks > 32 || !t.ffts_addr) {
        return false;
    }
    for (auto n : {t.conv_input_slots, t.conv_output_slots, t.ssm_input_slots, t.ssm_output_slots}) {
        if (n < 1 || n > 1048576) return false;
    }
    return Workspace(t).bytes <= (uint64_t{32} << 30);
}
} // namespace kda_prefill
