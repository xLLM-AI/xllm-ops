/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MtpPrepareNextDraftTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, speculativeWidth);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, numBlocksPerSequence);
TILING_DATA_FIELD_DEF(int32_t, blockSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MtpPrepareNextDraft,
                           MtpPrepareNextDraftTilingData)

}  // namespace optiling
