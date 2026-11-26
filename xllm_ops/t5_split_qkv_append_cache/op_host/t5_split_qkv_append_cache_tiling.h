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

#ifndef T5_SPLIT_QKV_APPEND_CACHE_TILING_H
#define T5_SPLIT_QKV_APPEND_CACHE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(T5SplitQkvAppendCacheTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, headNum);
  TILING_DATA_FIELD_DEF(uint32_t, historyLen);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(T5SplitQkvAppendCache, T5SplitQkvAppendCacheTilingData)
} // namespace optiling

#endif // T5_SPLIT_QKV_APPEND_CACHE_TILING_H
