/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <limits>

#include "../op_kernel/kda_replayssm_verify_record_tiling_data.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

using namespace kda_replayssm;

bool has_shape(const gert::StorageShape* storage,
               std::initializer_list<int64_t> dimensions) {
  if (storage == nullptr) {
    return false;
  }
  const auto& shape = storage->GetOriginShape();
  if (shape.GetDimNum() != dimensions.size()) {
    return false;
  }
  size_t axis = 0;
  for (int64_t dimension : dimensions) {
    if (shape.GetDim(axis++) != dimension) {
      return false;
    }
  }
  return true;
}

bool matches_qkv_stride(const gert::TilingContext* context,
                        size_t index,
                        int64_t token_stride) {
  const auto* stride = context->GetInputStride(index);
  return stride == nullptr ||
         (stride->GetDimNum() == 3 && stride->GetStride(0) == token_stride &&
          stride->GetStride(1) == kKeyDimension && stride->GetStride(2) == 1);
}

ge::graphStatus tile(gert::TilingContext* context) {
  const auto* query = context->GetRequiredInputShape(0);
  const auto* state = context->GetRequiredInputShape(5);
  if (query == nullptr || query->GetOriginShape().GetDimNum() != 3 ||
      state == nullptr || state->GetOriginShape().GetDimNum() != 4) {
    return ge::GRAPH_FAILED;
  }
  const int64_t tokens = query->GetOriginShape().GetDim(0);
  const int64_t slots = state->GetOriginShape().GetDim(0);
  const int64_t sequences = tokens / kWindow;
  const auto* attrs = context->GetAttrs();
  if (attrs == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const float* scale = attrs->GetAttrPointer<float>(0);
  // ACLNN can omit view strides in tiling. Include the token stride in the
  // attributes so the cache distinguishes dense and packed QKV.
  const int64_t* stride = attrs->GetAttrPointer<int64_t>(1);
  if (scale == nullptr || !std::isfinite(*scale) || *scale <= 0.0f ||
      stride == nullptr || (*stride != kFeatures && *stride != 3 * kFeatures)) {
    return ge::GRAPH_FAILED;
  }
  if (tokens <= 0 || tokens % kWindow != 0 ||
      sequences > std::numeric_limits<int32_t>::max() / kHeads / 4 ||
      slots <= 0 || slots > std::numeric_limits<int32_t>::max()) {
    return ge::GRAPH_FAILED;
  }
  for (size_t index = 0; index < 3; ++index) {
    if (!matches_qkv_stride(context, index, *stride) ||
        !has_shape(context->GetRequiredInputShape(index),
                   {tokens, kHeads, kKeyDimension})) {
      return ge::GRAPH_FAILED;
    }
  }
  if (!has_shape(context->GetRequiredInputShape(3),
                 {tokens, kHeads, kKeyDimension}) ||
      !has_shape(context->GetRequiredInputShape(4), {tokens, kHeads}) ||
      !has_shape(state, {slots, kHeads, kValueDimension, kKeyDimension}) ||
      !has_shape(context->GetRequiredInputShape(6), {kHeads}) ||
      !has_shape(context->GetRequiredInputShape(7), {kFeatures}) ||
      !has_shape(context->GetRequiredInputShape(8), {sequences}) ||
      !has_shape(context->GetOutputShape(3),
                 {tokens, kHeads, kValueDimension})) {
    return ge::GRAPH_FAILED;
  }
  for (size_t index = 0; index < 3; ++index) {
    if (!has_shape(context->GetOutputShape(index),
                   {slots, kWindow, kHeads, kKeyDimension})) {
      return ge::GRAPH_FAILED;
    }
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ub_bytes = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_bytes);
  const uint32_t workers = std::min<uint32_t>(platform.GetCoreNumAiv(), 48);
  auto* raw = context->GetRawTilingData();
  auto* workspace = context->GetWorkspaceSizes(1);
  if (workers == 0 || ub_bytes < kUbBytes || raw == nullptr ||
      raw->GetCapacity() < sizeof(KdaReplayssmVerifyRecordTilingData) ||
      workspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const KdaReplayssmVerifyRecordTilingData data{static_cast<int32_t>(sequences),
                                                static_cast<int32_t>(slots),
                                                static_cast<int32_t>(workers),
                                                static_cast<int32_t>(*stride),
                                                *scale};
  std::memcpy(raw->GetData(), &data, sizeof(data));
  raw->SetDataSize(sizeof(data));
  workspace[0] = 0;
  context->SetBlockDim(workers);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}

}  // namespace
IMPL_OP_OPTILING(KdaReplayssmVerifyRecord).Tiling(tile);

}  // namespace optiling
