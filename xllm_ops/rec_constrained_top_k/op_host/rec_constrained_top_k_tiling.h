/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(RecConstrainedTopKTilingData)
  TILING_DATA_FIELD_DEF(int32_t, num_rows);
  TILING_DATA_FIELD_DEF(int32_t, vocab_size);
  TILING_DATA_FIELD_DEF(int32_t, top_k);
  TILING_DATA_FIELD_DEF(int32_t, current_step);
  TILING_DATA_FIELD_DEF(int32_t, sequence_stride);
  TILING_DATA_FIELD_DEF(int32_t, first_token_count);
  TILING_DATA_FIELD_DEF(int32_t, prefix1_pair_count);
  TILING_DATA_FIELD_DEF(int32_t, temperature_count);
  TILING_DATA_FIELD_DEF(int32_t, max_candidate_count);
  TILING_DATA_FIELD_DEF(int32_t, used_core_num);
  TILING_DATA_FIELD_DEF(int32_t, debug_mode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(RecConstrainedTopK, RecConstrainedTopKTilingData)

}  // namespace optiling