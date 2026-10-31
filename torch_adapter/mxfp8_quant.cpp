/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <aclnn_grouped_matmul_swiglu_quant_v2.h>
#include <aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/platform/mx_quantization.h"
#include "xllm_ops/mx_quantization.h"
namespace xllm::kernel::npu {
namespace {
constexpr int64_t kFloat8E4M3FN = 292;
}  // namespace

bool mxfp8_runtime_available() {
  const char* const apis[] = {"aclnnDynamicMxQuant",
                              "aclnnSwigluMxQuant",
                              "aclnnRmsNormDynamicMxQuant",
                              "aclnnAddRmsNormDynamicMxQuant",
                              "aclnnQuantMatmulV5",
                              "aclnnGroupedMatmulSwigluQuantV2",
                              "aclnnGroupedMatmulSwigluQuantWeightNzV2"};
  for (const char* api : apis) {
    const std::string workspace_api = std::string(api) + "GetWorkspaceSize";
    if (aclnn::detail::get_op_api_func_addr(api) == nullptr ||
        aclnn::detail::get_op_api_func_addr(workspace_api.c_str()) == nullptr) {
      return false;
    }
  }
  return true;
}

std::tuple<at::Tensor, at::Tensor> mxfp8_dynamic_quant(
    const at::Tensor& input) {
  check_mx_quantization_supported();
  TORCH_CHECK(input.dim() >= 2,
              "mxfp8 dynamic quant input must be at least 2D");
  // MXFP8 uses the dedicated NPU op. npu_dynamic_quant only accepts the
  // per-tensor/per-token/per-channel modes and rejects "mx" at runtime.
  // 292 is torch_npu.float8_e4m3fn (the GE dtype value); the returned scale
  // is packed FLOAT8_E8M0 as required by the MX format.
  auto result = at_npu::native::custom_ops::npu_dynamic_mx_quant(
      input, -1, "rint", kFloat8E4M3FN, 32, 0, 0.0);
  // CANN exposes E8M0 scale storage as uint8. Keep the storage dtype intact;
  // the dtype arguments on QuantBatchMatmulV5 describe its interpretation.
  return result;
}

}  // namespace xllm::kernel::npu

namespace xllm::kernel::npu {
std::tuple<at::Tensor, at::Tensor> mxfp8_grouped_matmul_swiglu_quant(
    const at::Tensor& x,
    const torch::TensorList& weight,
    const torch::TensorList& weight_scale,
    const at::Tensor& x_scale,
    const at::Tensor& group_list) {
  check_mx_quantization_supported();
  TORCH_CHECK(x.dim() == 2, "MXFP8 MoE input must be 2D");
  TORCH_CHECK(!weight.empty() && !weight_scale.empty(),
              "MXFP8 MoE weights must not be empty");
  TORCH_CHECK(weight.size() == weight_scale.size(),
              "MXFP8 weight and weight_scale list sizes must match");
  TORCH_CHECK(weight[0].dim() == 3,
              "MXFP8 packed weight must be 3D [expert, K, N]");
  TORCH_CHECK(weight_scale[0].dim() == 4,
              "MXFP8 packed weight_scale must be 4D [expert, K/64, N, 2]");
  TORCH_CHECK(weight[0].scalar_type() == at::kFloat8_e4m3fn ||
                  weight[0].scalar_type() == at::kFloat8_e5m2,
              "MXFP8 weight must use FLOAT8_E4M3FN or FLOAT8_E5M2");
  TORCH_CHECK(weight_scale[0].scalar_type() == at::kFloat8_e8m0fnu,
              "MXFP8 weight_scale must use FLOAT8_E8M0");
  TORCH_CHECK(x_scale.scalar_type() == at::kFloat8_e8m0fnu,
              "MXFP8 x_scale must use FLOAT8_E8M0");
  TORCH_CHECK(group_list.dim() == 1,
              "MXFP8 group_list must be a 1D token-count tensor");
  const int64_t m = x.size(0);
  const int64_t n = weight_scale[0].size(weight_scale[0].dim() - 2);
  TORCH_CHECK(n % 2 == 0, "MXFP8 SwiGLU output dimension must be even");
  auto output = at::empty({m, n / 2}, x.options().dtype(at::kFloat8_e4m3fn));
  auto output_scale = at::empty({m, (n / 2 + 63) / 64, 2},
                                x.options().dtype(at::kFloat8_e8m0fnu));
  at::Tensor empty;
  torch::TensorList empty_list;
  int64_t dequant_mode = 2;
  int64_t dequant_dtype = 0;
  int64_t quant_mode = 2;
  int64_t group_list_type = 1;
  c10::IntArrayRef tuning_config;
  double swiglu_limit = 0.0;
  // The generic entry rejects FRACTAL_NZ storage during workspace validation;
  // prepacked MoE weights must use the WeightNz V2 entry.
#define XLLM_GROUPED_MX_ARGS                                                   \
  x, weight, weight_scale, empty_list, empty, x_scale, empty, group_list,      \
      dequant_mode, dequant_dtype, quant_mode, group_list_type, tuning_config, \
      swiglu_limit, output, output_scale
  if (at_npu::native::get_npu_format(weight[0]) == ACL_FORMAT_FRACTAL_NZ) {
    TORCH_CHECK(aclnn::detail::get_op_api_func_addr(
                    "aclnnGroupedMatmulSwigluQuantWeightNzV2") != nullptr,
                "prepacked MX weights require "
                "aclnnGroupedMatmulSwigluQuantWeightNzV2");
    EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantWeightNzV2, XLLM_GROUPED_MX_ARGS);
  } else {
    EXEC_NPU_CMD(aclnnGroupedMatmulSwigluQuantV2, XLLM_GROUPED_MX_ARGS);
  }
#undef XLLM_GROUPED_MX_ARGS
  return std::make_tuple(output, output_scale);
}

namespace {
// W8A8 MXFP8 and W4A8 MXFP4 both consume FP8 E4M3 activations; the FP4 in the
// W4A8 name refers to the down-projection weight, so the fused SiLU-gate quant
// is shared by both paths.
std::tuple<at::Tensor, at::Tensor> swiglu_mx_quant(
    const at::Tensor& gate_up_out,
    bool activate_left,
    double clamp_limit) {
  check_mx_quantization_supported();
  TORCH_CHECK(gate_up_out.dim() == 2,
              "MX swiglu quant input must be 2D [M, 2I]");
  TORCH_CHECK(gate_up_out.size(1) % 128 == 0,
              "MX swiglu quant requires 2I divisible by 128, got ",
              gate_up_out.size(1));
  // The op requires a positive finite clamp_limit. Models without a swish
  // clamp (swiglu_limit <= 0, e.g. Qwen3.5) pass a sentinel that never
  // binds; 1e6 mirrors xllm's own has_effective_swiglu_limit threshold.
  const double safe_clamp =
      (clamp_limit > 0.0 && std::isfinite(clamp_limit)) ? clamp_limit : 1e6;
  return at_npu::native::custom_ops::npu_swiglu_mx_quant(
      gate_up_out,
      /*group_index=*/std::nullopt,
      /*activate_dim=*/-1,
      /*activate_left=*/activate_left,
      /*swiglu_mode=*/0,  // 0: halves (left/right) activation
      /*clamp_limit=*/safe_clamp,
      /*glu_alpha=*/1.702,
      /*glu_bias=*/1.0,
      /*group_mode=*/0,
      /*axis=*/-1,
      /*dst_type=*/kFloat8E4M3FN,
      /*round_mode=*/"rint",
      /*scale_alg=*/0,
      /*max_dtype_value=*/0.0);
}
}  // namespace

std::tuple<at::Tensor, at::Tensor> mxfp8_swiglu_quant(
    const at::Tensor& gate_up_out,
    bool activate_left,
    double clamp_limit) {
  return swiglu_mx_quant(gate_up_out, activate_left, clamp_limit);
}

std::tuple<at::Tensor, at::Tensor> mxfp4_swiglu_quant(
    const at::Tensor& gate_up_out,
    bool activate_left,
    double clamp_limit) {
  return swiglu_mx_quant(gate_up_out, activate_left, clamp_limit);
}

std::tuple<at::Tensor, at::Tensor> mxfp8_rms_norm_quant(const at::Tensor& x,
                                                        const at::Tensor& gamma,
                                                        double epsilon) {
  check_mx_quantization_supported();
  // Fused RMSNorm + dynamic MX quant with a CEIL scale policy.
  // gamma must already include the Gemma-style +1 if needed.
  constexpr int64_t kFloat8E4M3FN = 292;
  auto result = at_npu::native::custom_ops::npu_rms_norm_dynamic_mx_quant(
      x, gamma, std::nullopt, epsilon, /*scale_alg=*/0, "rint", kFloat8E4M3FN);
  return std::make_tuple(std::get<0>(result), std::get<1>(result));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mxfp8_add_rms_norm_quant(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& gamma,
    double epsilon) {
  check_mx_quantization_supported();
  // Fused (x1 + x2) + RMSNorm + dynamic MX quant with a CEIL scale policy
  // (npu_dynamic_mx_quant rounds scales to nearest and clips block maxima).
  // Returns (e4m3 [M,K], uint8 [M,K/64,2] scale, residual = x1 + x2).
  // gamma must already include the Gemma-style +1 if the caller's norm is
  // Gemma-RMSNorm.
  constexpr int64_t kFloat8E4M3FN = 292;
  auto result = at_npu::native::custom_ops::npu_add_rms_norm_dynamic_mx_quant(
      x1,
      x2,
      gamma,
      std::nullopt,
      epsilon,
      /*scale_alg=*/0,
      "rint",
      kFloat8E4M3FN);
  return std::make_tuple(
      std::get<0>(result), std::get<2>(result), std::get<1>(result));
}

namespace {

// Dequantize a dynamic-MX pair to float32 for verification. Handles both
// observed scale layouts: [M, K/64, 2] interleaved and [M, K/32] per-block.
at::Tensor dequant_mx_activation(const at::Tensor& q, const at::Tensor& s) {
  const int64_t m = q.size(0);
  const int64_t k = q.size(1);
  at::Tensor exponent;
  if (s.dim() == 3 && s.size(0) == m && s.size(1) == k / 64 && s.size(2) == 2) {
    exponent = s.view({m, k / 64, 2, 1})
                   .expand({m, k / 64, 2, 32})
                   .reshape({m, k})
                   .to(at::kInt) -
               127;
  } else if (s.dim() == 2 && s.size(0) == m && s.size(1) == k / 32) {
    exponent = s.view({m, k / 32, 1})
                   .expand({m, k / 32, 32})
                   .reshape({m, k})
                   .to(at::kInt) -
               127;
  } else {
    TORCH_CHECK(false, "unsupported mx activation scale layout for verify");
  }
  return q.to(at::kFloat).mul(at::pow(2.0, exponent.to(at::kFloat)));
}

double mean_rel_err(const at::Tensor& got, const at::Tensor& ref) {
  auto diff = (got.to(at::kDouble) - ref.to(at::kDouble)).abs();
  auto denom = ref.to(at::kDouble).abs() + 1e-3;
  return diff.div(denom).mean().item<double>();
}

}  // namespace

int64_t verify_mxfp8_swiglu_quant(const at::Tensor& x,
                                  const at::Tensor& ref_act,
                                  double clamp_limit) {
  try {
    auto reference = mxfp8_dynamic_quant(ref_act);
    const double baseline = mean_rel_err(
        dequant_mx_activation(std::get<0>(reference), std::get<1>(reference)),
        ref_act);
    const double threshold = std::max(baseline * 1.5, 0.03);
    double best_err = std::numeric_limits<double>::max();
    int64_t best_side = -1;
    for (const int64_t side : {int64_t(1), int64_t(0)}) {
      auto fused = mxfp8_swiglu_quant(x, side != 0, clamp_limit);
      const double err = mean_rel_err(
          dequant_mx_activation(std::get<0>(fused), std::get<1>(fused)),
          ref_act);
      if (err < best_err) {
        best_err = err;
        best_side = side;
      }
    }
    if (best_side >= 0 && best_err <= threshold) {
      return best_side;
    }
    return -1;
  } catch (const std::exception&) {
    return -1;
  }
}
}  // namespace xllm::kernel::npu
