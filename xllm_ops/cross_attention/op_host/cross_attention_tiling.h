/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossAttentionTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, qHeads);
  TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, groupSize);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
  TILING_DATA_FIELD_DEF(uint32_t, qSeqlen);
  TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(uint32_t, qNBlockTile);
  TILING_DATA_FIELD_DEF(uint32_t, qSBlockTile);
  TILING_DATA_FIELD_DEF(uint32_t, qNBlockNumPerGroup);
  TILING_DATA_FIELD_DEF(uint32_t, qNBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, qSBlockNum);
  TILING_DATA_FIELD_DEF(uint32_t, perBatchTaskNum);
  TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
  TILING_DATA_FIELD_DEF(uint64_t, pOutSize);
  TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
  TILING_DATA_FIELD_DEF(uint64_t, wsBlockSizeDb);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossAttention, CrossAttentionTilingData)

// Native CV direct-connect tiling; key 3510 keeps the A2 layout independent.
BEGIN_TILING_DATA_DEF(CrossAttentionArch35TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, qHeads);
  TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
  TILING_DATA_FIELD_DEF(uint32_t, groupSize);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
  TILING_DATA_FIELD_DEF(uint32_t, qSeqlen);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
  TILING_DATA_FIELD_DEF(float, scaleValue);
  TILING_DATA_FIELD_DEF(uint32_t, qRowsPerTile);
  TILING_DATA_FIELD_DEF(uint32_t, qRowTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerTask);
  TILING_DATA_FIELD_DEF(uint32_t, gPerTile);
  TILING_DATA_FIELD_DEF(uint32_t, gTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, kvTile);
  TILING_DATA_FIELD_DEF(uint32_t, kSeg);
  TILING_DATA_FIELD_DEF(uint32_t, mergeSegBlocks);
  TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, dtype);
  TILING_DATA_FIELD_DEF(uint64_t, mergeSlotBytes);
  TILING_DATA_FIELD_DEF(uint64_t, sStageOff);
  TILING_DATA_FIELD_DEF(uint64_t, pStageOff);
  TILING_DATA_FIELD_DEF(uint64_t, oStageOff);
  TILING_DATA_FIELD_DEF(uint64_t, userWsBytes);
  TILING_DATA_FIELD_DEF(uint32_t, tilingVersion);
  TILING_DATA_FIELD_DEF(uint32_t, reserved0);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossAttention_3510, CrossAttentionArch35TilingData)
}
