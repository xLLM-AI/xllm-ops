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
#include "dsa_kv_offloading_plan_nope_tiling.h"

#include <algorithm>
#include <limits>

#include "register/op_def_registry.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto* shape = context->GetInputShape(1);
  if (shape == nullptr || shape->GetStorageShape().GetDimNum() != 3)
    return ge::GRAPH_FAILED;
  const auto& dims = shape->GetStorageShape();
  const int64_t rows = dims.GetDim(0), heads = dims.GetDim(1),
                width = dims.GetDim(2);
  if (rows <= 0 || heads != 1 || width != 2176 ||
      rows > std::numeric_limits<int32_t>::max() / (heads * width))
    return ge::GRAPH_FAILED;
  DsaKvOffloadingPlanNopeTilingData tiling;
  tiling.set_rows(static_cast<uint32_t>(rows));
  tiling.set_heads(static_cast<uint32_t>(heads));
  tiling.set_topk(static_cast<uint32_t>(width));
  tiling.set_mapStride(0);
  tiling.set_tableStride(0);
  const uint32_t workers =
      static_cast<uint32_t>(std::min<int64_t>(48, rows * heads));
  tiling.set_workerCount(workers);
  context->SetBlockDim(workers);
  context->SetTilingKey(0);
  context->GetWorkspaceSizes(1)[0] = 0;
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
// Fixed-layout kernels need no JSON compile-info pattern. Register a
// native parse entry rather than using CANN's auto-tiling parser.
struct CompileInfo {};
static ge::graphStatus TilingParse(gert::TilingParseContext* context) {
  return context != nullptr ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
IMPL_OP_OPTILING(DsaKvOffloadingPlanNope)
    .Tiling(TilingFunc)
    .TilingParse<CompileInfo>(TilingParse);
}  // namespace optiling
