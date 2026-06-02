/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    
BEGIN_TILING_DATA_DEF(BeamSearchTilingData)
TILING_DATA_FIELD_DEF(int32_t, num_sequences);
TILING_DATA_FIELD_DEF(int32_t, sequence_length);
TILING_DATA_FIELD_DEF(int32_t, beam_width);
TILING_DATA_FIELD_DEF(int32_t, top_k);
TILING_DATA_FIELD_DEF(int32_t, request_num);
TILING_DATA_FIELD_DEF(int32_t, core_num);
TILING_DATA_FIELD_DEF(int32_t, min_size);
TILING_DATA_FIELD_DEF(int32_t, step_size);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topkTilingData);
TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topKTilingData1);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BeamSearch, BeamSearchTilingData);

class TilingBeamSearchFunc {
 public:
  explicit TilingBeamSearchFunc(gert::TilingContext* tiling_context)
      : tiling_context_(tiling_context) {}

  ge::graphStatus Init();
  ge::graphStatus RunKernelTiling();

 private:
  BeamSearchTilingData tiling_data_;
  gert::TilingContext* tiling_context_ = nullptr;

  void SetTilingKey();
  void FillTilingData();
// TODO: uint32_t to int32_t
  uint32_t num_sequences_ = 0;
  uint32_t block_num_ = 0;
  uint32_t sequence_length_ = 0;
  uint32_t beam_width_ = 0;
  uint32_t top_k_ = 0;
  uint32_t core_num_ = 0;
  uint32_t request_num_ = 0;
  uint32_t min_size_ = 0;
  uint32_t step_size_ = 0;
  size_t sync_workspace_size_ = 0;
};

}  // namespace optiling