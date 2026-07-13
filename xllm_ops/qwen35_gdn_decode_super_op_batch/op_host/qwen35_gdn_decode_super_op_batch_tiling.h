/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Qwen35GdnDecodeSuperOpBatchTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch_size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Qwen35GdnDecodeSuperOpBatch,
                           Qwen35GdnDecodeSuperOpBatchTilingData)
}  // namespace optiling
