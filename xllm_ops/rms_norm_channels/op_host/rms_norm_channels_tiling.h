/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(RmsNormChannelsTilingData)
// Problem size: value is [b, c, n] with n = T * H * W.
TILING_DATA_FIELD_DEF(uint64_t, b);
TILING_DATA_FIELD_DEF(uint64_t, c);
TILING_DATA_FIELD_DEF(uint64_t, n);
// Tile plan for the grid-stride families (SMALL_C / STREAM / SINGLE_CHANNEL).
TILING_DATA_FIELD_DEF(uint64_t, totalTiles);
TILING_DATA_FIELD_DEF(uint64_t, tilesPerBatch);
// Launch geometry.
TILING_DATA_FIELD_DEF(uint64_t, numBlocks);
TILING_DATA_FIELD_DEF(uint64_t, cPerBlock);
TILING_DATA_FIELD_DEF(uint64_t, workspaceStride);
TILING_DATA_FIELD_DEF(uint32_t, nTile);
TILING_DATA_FIELD_DEF(uint32_t, cChunk);
// One of RMS_NORM_CHANNELS_TK_* from op_kernel/rms_norm_channels_common.h.
TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(RmsNormChannels, RmsNormChannelsTilingData)
} // namespace optiling
