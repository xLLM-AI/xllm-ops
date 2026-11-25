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
#include <cstring>
#include <initializer_list>
#include <limits>

#include "../op_kernel/kda_replayssm_correction_fold_layout.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

bool has_shape(const gert::StorageShape* storage,
               std::initializer_list<int64_t> dimensions) {
  if (storage == nullptr) {
    return false;
  }
  const auto& shape = storage->GetStorageShape();
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

ge::graphStatus tile(gert::TilingContext* context) {
  using namespace kda_replayssm;
  const auto* indices = context->GetRequiredInputShape(4);
  if (indices == nullptr || indices->GetStorageShape().GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }
  const int64_t sequences = indices->GetStorageShape().GetDim(0);
  if (sequences <= 0 ||
      sequences > std::numeric_limits<int32_t>::max() / kHeads ||
      !has_shape(context->GetRequiredInputShape(5), {sequences})) {
    return ge::GRAPH_FAILED;
  }
  int64_t layers = 0;
  for (;; ++layers) {
    const auto* state = context->GetDynamicInputShape(0, layers);
    if (state == nullptr) {
      break;
    }
    if (state->GetStorageShape().GetDimNum() != 4) {
      return ge::GRAPH_FAILED;
    }
    const int64_t slots = state->GetStorageShape().GetDim(0);
    if (slots <= 0 ||
        slots > std::numeric_limits<int32_t>::max() / kHeads / kStateElements ||
        !has_shape(state, {slots, kHeads, kValueDimension, kKeyDimension}) ||
        !has_shape(context->GetOutputShape(layers),
                   {slots, kHeads, kValueDimension, kKeyDimension})) {
      return ge::GRAPH_FAILED;
    }
    for (size_t field = 1; field < 4; ++field) {
      if (!has_shape(context->GetDynamicInputShape(field, layers),
                     {slots, kWindow, kHeads, kKeyDimension})) {
        return ge::GRAPH_FAILED;
      }
    }
  }
  if (layers == 0 ||
      layers > std::numeric_limits<int32_t>::max() / sequences / kHeads ||
      context->GetOutputShape(layers) != nullptr) {
    return ge::GRAPH_FAILED;
  }
  for (size_t field = 1; field < 4; ++field) {
    if (context->GetDynamicInputShape(field, layers) != nullptr) {
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
      raw->GetCapacity() < sizeof(KdaReplayssmFoldTilingData) ||
      workspace == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const KdaReplayssmFoldTilingData data{static_cast<int32_t>(sequences),
                                        static_cast<int32_t>(layers),
                                        static_cast<int32_t>(workers)};
  std::memcpy(raw->GetData(), &data, sizeof(data));
  raw->SetDataSize(sizeof(data));
  workspace[0] = 0;
  context->SetBlockDim(workers);
  context->SetTilingKey(0);
  return ge::GRAPH_SUCCESS;
}

}  // namespace
IMPL_OP_OPTILING(KdaReplayssmCorrectionFold).Tiling(tile);

}  // namespace optiling
