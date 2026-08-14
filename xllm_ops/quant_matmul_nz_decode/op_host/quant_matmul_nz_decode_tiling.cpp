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

#include "quant_matmul_nz_decode_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kCoreCount = 20;
constexpr uint32_t kGateUpCoreCount = 20;
constexpr uint32_t kQkvCoreCount = 10;
constexpr uint32_t kTileM = 16;
constexpr uint32_t kMaxOptimizedM = 16;
constexpr uint32_t kTileN = 320;
constexpr uint32_t kWorkspaceStages = 2;

ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const auto x_shape = context->GetInputShape(0)->GetOriginShape();
  const auto weight_shape = context->GetInputShape(1)->GetOriginShape();
  const auto scale_shape = context->GetInputShape(2)->GetOriginShape();
  const auto bias_shape = context->GetInputShape(3)->GetOriginShape();
  const size_t x_dim = x_shape.GetDimNum();
  const size_t weight_dim = weight_shape.GetDimNum();
  if (x_dim != 2 || weight_dim != 2 || scale_shape.GetDimNum() != 1 ||
      bias_shape.GetDimNum() != 1) {
    return ge::GRAPH_FAILED;
  }

  const int64_t m_dim = x_shape.GetDim(0);
  const int64_t k_dim = x_shape.GetDim(1);
  const int64_t n_dim = weight_shape.GetDim(1);

  const bool is_gate_up_shape = k_dim == 5120 && n_dim == 6400;
  const bool is_down_shape = k_dim == 3200 && n_dim == 5120;
  const bool is_qkv_shape = k_dim == 5120 && n_dim == 1280;
  if (m_dim <= 0 || m_dim > kMaxOptimizedM ||
      (!is_gate_up_shape && !is_down_shape && !is_qkv_shape) ||
      weight_shape.GetDim(0) != k_dim || scale_shape.GetDim(0) != n_dim ||
      bias_shape.GetDim(0) != n_dim) {
    return ge::GRAPH_FAILED;
  }

  const uint32_t m = static_cast<uint32_t>(m_dim);
  const uint32_t k = static_cast<uint32_t>(k_dim);
  const uint32_t n = static_cast<uint32_t>(n_dim);
  QuantMatmulNzDecodeTilingData tiling;
  tiling.set_m(m);
  tiling.set_k(k);
  tiling.set_n(n);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetBlockDim(
      is_qkv_shape ? kQkvCoreCount
                   : (is_gate_up_shape ? kGateUpCoreCount : kCoreCount));
  if (is_down_shape) {
    context->SetTilingKey(m == 1 ? 1 : (m <= 4 ? 3 : 2));
  } else if (is_qkv_shape) {
    context->SetTilingKey(4);
  } else if (m > 2) {
    context->SetTilingKey(5);
  } else {
    context->SetTilingKey(0);
  }

  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  size_t* workspace_size = context->GetWorkspaceSizes(1);
  workspace_size[0] =
      platform.GetLibApiWorkSpaceSize() +
      kTileM * kTileN * kGateUpCoreCount * kWorkspaceStages * sizeof(int32_t);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_OPTILING(QuantMatmulNzDecode).Tiling(TilingFunc);
}  // namespace optiling
