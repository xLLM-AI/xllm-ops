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

#include <acl/acl_base.h>
#include <glog/logging.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>

#include <algorithm>
#include <cstdlib>

#if defined(XLLM_HAS_NPU_MX_MATMUL)
#include <torch_npu/csrc/aten/CustomFunctions.h>
#endif

#include "core/platform/mx_quantization.h"
#include "xllm_ops/mx_quantization.h"

namespace xllm::kernel::npu {
namespace {
#if defined(XLLM_HAS_NPU_MX_MATMUL)
constexpr int64_t kFloat8E4M3FN = 292;
constexpr int64_t kFloat8E5M2 = 291;
constexpr int64_t kFloat4E2M1X2 = 296;
constexpr int64_t kE8M0Scale = 293;
constexpr int64_t kMxBlockSize = 32;
#else
[[noreturn]] void unavailable() {
  LOG(FATAL) << "This build does not provide the QuantMatmulV5 MXFP4 API.";
  std::abort();
}
#endif
}  // namespace

// Also used by the MXFP4 prefill cache after its lossless FP8 expansion.
at::Tensor mxfp8_quant_matmul(const at::Tensor& quantized_input,
                              const at::Tensor& input_scale,
                              const at::Tensor& weight,
                              const at::Tensor& weight_scale,
                              const std::optional<at::Tensor>& bias,
                              at::ScalarType output_dtype) {
  check_mx_quantization_supported();
#if defined(XLLM_HAS_NPU_MX_MATMUL)
  const int64_t weight_dtype =
      weight.scalar_type() == at::kFloat8_e5m2 ? kFloat8E5M2 : kFloat8E4M3FN;
  std::optional<at::Tensor> bias_fp32;
  if (bias.has_value() && bias->defined()) {
    bias_fp32 = bias->scalar_type() == at::kFloat
                    ? bias
                    : std::optional<at::Tensor>(bias->to(at::kFloat));
  }
  return at_npu::native::custom_ops::npu_quant_matmul(
      quantized_input,
      weight,
      weight_scale,
      std::nullopt,
      input_scale,
      bias_fp32,
      static_cast<int64_t>(output_dtype),
      kFloat8E4M3FN,
      weight_dtype,
      kE8M0Scale,
      kE8M0Scale,
      c10::IntArrayRef({1, 1, kMxBlockSize}),
      std::nullopt);
#else
  unavailable();
#endif
}

// W4A8 GEMM only needs QuantMatmulV5.  Gate it on its own probe instead of
// the aggregate MXFP8 flag so a partial FP8 installation keeps MXFP4 usable.
at::Tensor mxfp4_quant_matmul(const at::Tensor& quantized_input,
                              const at::Tensor& input_scale,
                              const at::Tensor& weight_packed,
                              const at::Tensor& weight_scale,
                              at::ScalarType output_dtype,
                              int64_t input_dtype,
                              const std::optional<at::Tensor>& bias) {
  check_mx_quantization_supported();
#if defined(XLLM_HAS_NPU_MX_MATMUL)
  std::optional<at::Tensor> bias_bf16;
  if (bias.has_value() && bias->defined()) {
    bias_bf16 = bias->reshape({1, -1}).to(at::kBFloat16);
  }
  return at_npu::native::custom_ops::npu_quant_matmul(
      quantized_input,
      weight_packed,
      weight_scale,
      std::nullopt,
      input_scale,
      bias_bf16,
      static_cast<int64_t>(output_dtype),
      input_dtype,
      kFloat4E2M1X2,
      kE8M0Scale,
      kE8M0Scale,
      c10::IntArrayRef({1, 1, kMxBlockSize}),
      std::nullopt);
#else
  unavailable();
#endif
}

// The two layout helpers below only use generic Torch-NPU format casts, so
// they stay available even when no MX ACLNN op was detected.
void mxfp4_prepack_nz_weight(at::Tensor& weight_packed) {
  check_mx_quantization_supported();
  if (at_npu::native::get_npu_format(weight_packed) == ACL_FORMAT_FRACTAL_NZ) {
    return;
  }
  const int64_t n = weight_packed.size(0);
  const int64_t k_half = weight_packed.size(1);
  if (n % 16 != 0 || k_half % 32 != 0) {
    return;
  }
  // uint8 format_cast uses byte tiles; rearrange packed FP4's half-byte tiles
  // once so the native NZ kernel reads exactly the checkpoint values.
  auto packed = weight_packed.to(at::kCPU)
                    .reshape({n, k_half / 16, 16})
                    .permute({1, 0, 2})
                    .contiguous()
                    .reshape({k_half / 32, n / 16, 16, 32})
                    .permute({1, 2, 0, 3})
                    .contiguous()
                    .reshape({n, k_half})
                    .to(weight_packed.device());
  weight_packed.set_data(
      at_npu::native::npu_format_cast(packed, ACL_FORMAT_FRACTAL_NZ));
}

at::Tensor mxfp4_prefill_fp8_weight(const at::Tensor& weight_packed) {
  check_mx_quantization_supported();
  TORCH_CHECK(weight_packed.dim() == 2, "MXFP4 packed weight must be [N, K/2]");
  TORCH_CHECK(at_npu::native::get_npu_format(weight_packed) == ACL_FORMAT_ND,
              "Prepare the FP8 prefill cache before FP4 NZ conversion");
  const int64_t n = weight_packed.size(0);
  const int64_t k = weight_packed.size(1) * 2;
  const auto packed = weight_packed.to(at::kCPU).contiguous();
  // E2M1 code -> exact FP8 value. Nibble order is low half first.
  const auto lut = at::tensor({0.0f,
                               0.5f,
                               1.0f,
                               1.5f,
                               2.0f,
                               3.0f,
                               4.0f,
                               6.0f,
                               -0.0f,
                               -0.5f,
                               -1.0f,
                               -1.5f,
                               -2.0f,
                               -3.0f,
                               -4.0f,
                               -6.0f},
                              at::kFloat);
  auto decoded = at::empty({n, k}, packed.options().dtype(at::kFloat8_e4m3fn));
  constexpr int64_t kRowsPerDecode = 256;
  for (int64_t start = 0; start < n; start += kRowsPerDecode) {
    const int64_t count = std::min(kRowsPerDecode, n - start);
    const auto bytes = packed.narrow(/*dim=*/0, start, count);
    const auto low = at::bitwise_and(bytes, 15).to(at::kLong);
    const auto high = at::bitwise_right_shift(bytes, 4).to(at::kLong);
    const auto codes = at::stack({low, high}, /*dim=*/-1).reshape({-1});
    decoded.narrow(/*dim=*/0, start, count)
        .copy_(lut.index_select(/*dim=*/0, codes)
                   .reshape({count, k})
                   .to(at::kFloat8_e4m3fn));
  }
  return at_npu::native::npu_format_cast(
      decoded.transpose(0, 1).contiguous().to(weight_packed.device()),
      ACL_FORMAT_FRACTAL_NZ);
}
}  // namespace xllm::kernel::npu
