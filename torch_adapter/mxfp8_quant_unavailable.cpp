/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>

#include <cmath>
#include <cstdlib>

#if defined(XLLM_HAS_NPU_MX_SWIGLU) || defined(XLLM_HAS_NPU_MX_DYNAMIC_QUANT)
#include <torch_npu/csrc/aten/CustomFunctions.h>
#endif

#include "core/platform/mx_quantization.h"
#include "xllm_ops/mx_quantization.h"

namespace xllm::kernel::npu {
namespace {
[[noreturn]] void unavailable() {
  LOG(FATAL) << "This build does not support Torch-NPU/CANN MX quantization.";
  std::abort();
}
}  // namespace

bool mxfp8_runtime_available() { return false; }

std::tuple<at::Tensor, at::Tensor> mxfp8_dynamic_quant(
    const at::Tensor& input) {
  check_mx_quantization_supported();
#if defined(XLLM_HAS_NPU_MX_DYNAMIC_QUANT)
  CHECK_GE(input.dim(), 2);
  return at_npu::native::custom_ops::npu_dynamic_mx_quant(
      input, -1, "rint", 292, 32, 0, 0.0);
#else
  unavailable();
#endif
}

std::tuple<at::Tensor, at::Tensor> mxfp8_swiglu_quant(const at::Tensor&,
                                                      bool,
                                                      double) {
  unavailable();
}
std::tuple<at::Tensor, at::Tensor> mxfp4_swiglu_quant(
    const at::Tensor& gate_up_out,
    bool activate_left,
    double clamp_limit) {
  check_mx_quantization_supported();
#if defined(XLLM_HAS_NPU_MX_SWIGLU)
  CHECK_EQ(gate_up_out.dim(), 2);
  CHECK_EQ(gate_up_out.size(1) % 128, 0);
  constexpr int64_t kFloat8E4M3FN = 292;
  const double safe_clamp =
      (clamp_limit > 0.0 && std::isfinite(clamp_limit)) ? clamp_limit : 1e6;
  return at_npu::native::custom_ops::npu_swiglu_mx_quant(gate_up_out,
                                                         std::nullopt,
                                                         -1,
                                                         activate_left,
                                                         0,
                                                         safe_clamp,
                                                         1.702,
                                                         1.0,
                                                         0,
                                                         -1,
                                                         kFloat8E4M3FN,
                                                         "rint",
                                                         0,
                                                         0.0);
#else
  unavailable();
#endif
}
std::tuple<at::Tensor, at::Tensor> mxfp8_rms_norm_quant(const at::Tensor&,
                                                        const at::Tensor&,
                                                        double) {
  unavailable();
}
std::tuple<at::Tensor, at::Tensor, at::Tensor> mxfp8_add_rms_norm_quant(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double) {
  unavailable();
}
std::tuple<at::Tensor, at::Tensor> mxfp8_grouped_matmul_swiglu_quant(
    const at::Tensor&,
    const torch::TensorList&,
    const torch::TensorList&,
    const at::Tensor&,
    const at::Tensor&) {
  unavailable();
}
int64_t verify_mxfp8_swiglu_quant(const at::Tensor&,
                                  const at::Tensor&,
                                  double) {
  return -1;
}
}  // namespace xllm::kernel::npu
