/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EcrTokenRoutingStatisticsTilingData)
TILING_DATA_FIELD_DEF(uint32_t, s);
TILING_DATA_FIELD_DEF(uint32_t, e);
TILING_DATA_FIELD_DEF(uint32_t, b);
TILING_DATA_FIELD_DEF(uint32_t, total_tasks);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, seq_base);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, seq_len);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 32, task_base);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EcrTokenRoutingStatistics,
                           EcrTokenRoutingStatisticsTilingData)
}  // namespace optiling
