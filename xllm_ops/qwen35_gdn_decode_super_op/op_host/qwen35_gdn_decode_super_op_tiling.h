/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Qwen35GdnDecodeSuperOpTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_size);
TILING_DATA_FIELD_DEF(int64_t, num_k_heads);
TILING_DATA_FIELD_DEF(int64_t, num_v_heads);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Qwen35GdnDecodeSuperOp,
                           Qwen35GdnDecodeSuperOpTilingData)
}  // namespace optiling
