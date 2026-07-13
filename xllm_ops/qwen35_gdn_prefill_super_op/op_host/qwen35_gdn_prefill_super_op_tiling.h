/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Qwen35GdnPrefillSuperOpTilingData)
TILING_DATA_FIELD_DEF(uint32_t, block_dim);
TILING_DATA_FIELD_DEF(uint32_t, num_matrices);
TILING_DATA_FIELD_DEF(uint32_t, num_heads);
TILING_DATA_FIELD_DEF(uint32_t, num_key_heads);
TILING_DATA_FIELD_DEF(uint32_t, token_block_size);
TILING_DATA_FIELD_DEF(uint32_t, token_block_count);
TILING_DATA_FIELD_DEF(int64_t, conv_state_index);
TILING_DATA_FIELD_DEF(int64_t, ssm_state_index);
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, seq_len);
TILING_DATA_FIELD_DEF(int64_t, total_tokens);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Qwen35GdnPrefillSuperOp, Qwen35GdnPrefillSuperOpTilingData)
}  // namespace optiling
