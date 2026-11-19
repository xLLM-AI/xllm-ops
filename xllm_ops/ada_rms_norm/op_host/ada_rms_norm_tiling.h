/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdaRmsNormTilingData)
TILING_DATA_FIELD_DEF(uint64_t, rows);
TILING_DATA_FIELD_DEF(uint64_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint64_t, seqLength);
TILING_DATA_FIELD_DEF(uint64_t, scaleBatchStride);
TILING_DATA_FIELD_DEF(uint64_t, scaleSeqStride);
TILING_DATA_FIELD_DEF(uint64_t, shiftBatchStride);
TILING_DATA_FIELD_DEF(uint64_t, shiftSeqStride);
TILING_DATA_FIELD_DEF(uint32_t, hiddenDim);
TILING_DATA_FIELD_DEF(uint32_t, rawRowBytes);
TILING_DATA_FIELD_DEF(uint32_t, floatRowBytes);
TILING_DATA_FIELD_DEF(uint32_t, hasWeight);
TILING_DATA_FIELD_DEF(float, epsilon);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(AdaRmsNorm, AdaRmsNormTilingData)
} // namespace optiling
