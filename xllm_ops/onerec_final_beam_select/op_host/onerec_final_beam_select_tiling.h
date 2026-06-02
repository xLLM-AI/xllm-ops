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

#pragma once

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(OnerecFinalBeamSelectTilingData)
TILING_DATA_FIELD_DEF(int32_t, num_sequences);
TILING_DATA_FIELD_DEF(int32_t, active_beam_width);
TILING_DATA_FIELD_DEF(int32_t, candidate_top_k);
TILING_DATA_FIELD_DEF(int32_t, result_width);
TILING_DATA_FIELD_DEF(int32_t, request_num);
TILING_DATA_FIELD_DEF(int32_t, current_step);
TILING_DATA_FIELD_DEF(int32_t, max_decode_step);
TILING_DATA_FIELD_DEF(int32_t, core_num);
TILING_DATA_FIELD_DEF(int32_t, step_size);
TILING_DATA_FIELD_DEF(int32_t, min_size);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, firstTopkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, mergeTopkTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(OnerecFinalBeamSelect, OnerecFinalBeamSelectTilingData)

}  // namespace optiling