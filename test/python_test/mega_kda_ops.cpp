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

#include <acl/acl_rt.h>
#include <algorithm>
#include <cctype>
#include <string>

#include "mega_kda_ops.h"

#include <dlfcn.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor> mega_kda_prefill(
    const at::Tensor& qkv,
    const at::Tensor& gate,
    const at::Tensor& beta,
    const at::Tensor& a_log,
    const at::Tensor& gate_bias,
    const at::Tensor& conv_weight,
    const c10::optional<at::Tensor>& conv_bias,
    const at::Tensor& conv_state_in,
    const at::Tensor& ssm_state_in,
    const at::Tensor& cu_seqlens,
    const at::Tensor& conv_read_indices,
    const at::Tensor& conv_write_indices,
    const at::Tensor& ssm_read_indices,
    const at::Tensor& ssm_write_indices,
    at::Tensor& output,
    at::Tensor& conv_state_out,
    at::Tensor& ssm_state_out) {
  TORCH_CHECK(qkv.device().type() == c10::DeviceType::PrivateUse1,
              "MegaKdaPrefill requires NPU tensors");
  std::vector<at::Tensor> inputs = {qkv,
                                    gate,
                                    beta,
                                    a_log,
                                    gate_bias,
                                    conv_weight,
                                    conv_state_in,
                                    ssm_state_in,
                                    cu_seqlens,
                                    conv_read_indices,
                                    conv_write_indices,
                                    ssm_read_indices,
                                    ssm_write_indices};
  if (conv_bias) {
    inputs.push_back(*conv_bias);
  }
  std::vector<at::Tensor> outputs = {output, conv_state_out, ssm_state_out};
  for (const auto& tensor : inputs) {
    TORCH_CHECK(tensor.defined() && tensor.device() == qkv.device() &&
                    tensor.is_contiguous(),
                "MegaKdaPrefill inputs must be contiguous on the same NPU");
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i];
    TORCH_CHECK(tensor.defined() && tensor.device() == qkv.device() &&
                    tensor.is_contiguous(),
                "MegaKdaPrefill outputs must be contiguous on the same NPU");
    for (const auto& input : inputs) {
      TORCH_CHECK(!tensor.is_alias_of(input),
                  "MegaKdaPrefill outputs must not alias inputs");
    }
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(!tensor.is_alias_of(outputs[j]),
                  "MegaKdaPrefill outputs must not alias each other");
    }
  }
  TORCH_CHECK(conv_state_out.dim() == 3 && ssm_state_out.dim() == 4,
              "MegaKdaPrefill requires separate Conv [slots,L,C] and SSM "
              "[slots,H,V,K] outputs");
  const int64_t conv_output_slots = conv_state_out.size(0);
  const int64_t ssm_output_slots = ssm_state_out.size(0);
  const char* soc = aclrtGetSocName();
  std::string soc_name = soc ? soc : "";
  std::transform(soc_name.begin(), soc_name.end(), soc_name.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  const bool is_a5 = soc_name.rfind("ascend950", 0) == 0 ||
                     soc_name.rfind("ascend910_95", 0) == 0;
  uint64_t ffts_addr = 0;
  // A5 uses GM-based synchronization and can run with a zero FFTS address.
  if (!is_a5) {
    using GetC2c = int32_t (*)(uint64_t*, uint32_t*);
    static const GetC2c get_c2c = [] {
      void* handle = dlopen("libruntime.so", RTLD_LAZY | RTLD_LOCAL);
      TORCH_CHECK(handle, "failed to load libruntime.so: ", dlerror());
      void* symbol = dlsym(handle, "rtGetC2cCtrlAddr");
      TORCH_CHECK(symbol, "failed to resolve rtGetC2cCtrlAddr: ", dlerror());
      return reinterpret_cast<GetC2c>(symbol);
    }();
    uint32_t ffts_len = 0;
    TORCH_CHECK(get_c2c(&ffts_addr, &ffts_len) == 0 && ffts_addr,
                "MegaKdaPrefill requires an FFTS address on Ascend 910B and 910C");
  }
  const int64_t ffts_attr = static_cast<int64_t>(ffts_addr);
  // No metadata D2H: replay reads current device tensors and caller-owned
  // outputs.
  EXEC_NPU_CMD(aclnnMegaKdaPrefill,
               qkv,
               gate,
               beta,
               a_log,
               gate_bias,
               conv_weight,
               conv_bias,
               conv_state_in,
               ssm_state_in,
               cu_seqlens,
               conv_read_indices,
               conv_write_indices,
               ssm_read_indices,
               ssm_write_indices,
               conv_output_slots,
               ssm_output_slots,
               ffts_attr,
               output,
               conv_state_out,
               ssm_state_out);
  return std::make_tuple(output, conv_state_out, ssm_state_out);
}

enum class KdaDecodeMode : int64_t { NATIVE = 0, MTP = 1, PUBLIC_COMPAT = 3 };

std::tuple<at::Tensor, at::Tensor, at::Tensor> mega_kda_decode(
    const at::Tensor& qkv,
    const at::Tensor& gate,
    const at::Tensor& beta,
    const at::Tensor& a_log,
    const at::Tensor& gate_bias,
    const at::Tensor& conv_weight,
    const c10::optional<at::Tensor>& conv_bias,
    const at::Tensor& conv_state_in,
    const at::Tensor& ssm_state_in,
    const at::Tensor& cu_seqlens,
    const at::Tensor& conv_read_indices,
    const at::Tensor& conv_write_indices,
    const at::Tensor& ssm_read_indices,
    const at::Tensor& ssm_write_indices,
    const c10::optional<at::Tensor>& num_accepted_tokens,
    int64_t mode,
    int64_t max_query_tokens,
    at::Tensor& output,
    at::Tensor& conv_state_out,
    at::Tensor& ssm_state_out) {
  TORCH_CHECK(qkv.device().type() == c10::DeviceType::PrivateUse1,
              "MegaKdaDecode requires NPU tensors");
  const KdaDecodeMode decode_mode = static_cast<KdaDecodeMode>(mode);
  const bool ordinary = decode_mode == KdaDecodeMode::NATIVE ||
                        decode_mode == KdaDecodeMode::PUBLIC_COMPAT;
  TORCH_CHECK((ordinary && max_query_tokens == 1 && !num_accepted_tokens) ||
                  (decode_mode == KdaDecodeMode::MTP && max_query_tokens >= 1 &&
                   max_query_tokens <= 17 && num_accepted_tokens),
              "invalid MegaKdaDecode mode/capacity/accepted-count combination");
  std::vector<at::Tensor> inputs = {qkv,
                                    gate,
                                    beta,
                                    a_log,
                                    gate_bias,
                                    conv_weight,
                                    conv_state_in,
                                    ssm_state_in,
                                    cu_seqlens,
                                    conv_read_indices,
                                    conv_write_indices,
                                    ssm_read_indices,
                                    ssm_write_indices};
  if (conv_bias) {
    inputs.push_back(*conv_bias);
  }
  if (num_accepted_tokens) {
    inputs.push_back(*num_accepted_tokens);
  }
  std::vector<at::Tensor> outputs = {output, conv_state_out, ssm_state_out};
  for (const auto& tensor : inputs) {
    TORCH_CHECK(tensor.defined() && tensor.device() == qkv.device() &&
                    tensor.is_contiguous(),
                "MegaKdaDecode inputs must be contiguous on the same NPU");
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& tensor = outputs[i];
    TORCH_CHECK(tensor.defined() && tensor.device() == qkv.device() &&
                    tensor.is_contiguous(),
                "MegaKdaDecode outputs must be contiguous on the same NPU");
    for (const auto& input : inputs) {
      TORCH_CHECK(!tensor.is_alias_of(input),
                  "MegaKdaDecode outputs must not alias inputs");
    }
    for (size_t j = 0; j < i; ++j) {
      TORCH_CHECK(!tensor.is_alias_of(outputs[j]),
                  "MegaKdaDecode outputs must not alias each other");
    }
  }
  TORCH_CHECK(conv_state_out.dim() == 3 && ssm_state_out.dim() == 4,
              "MegaKdaDecode requires separate Conv [slots,L,C] and SSM "
              "[slots,H,V,K] outputs");
  const int64_t conv_output_slots = conv_state_out.size(0);
  const int64_t ssm_output_slots = ssm_state_out.size(0);
  // No metadata D2H: replay reads current device tensors and caller-owned
  // outputs.
  EXEC_NPU_CMD(aclnnMegaKdaDecode,
               qkv,
               gate,
               beta,
               a_log,
               gate_bias,
               conv_weight,
               conv_bias,
               conv_state_in,
               ssm_state_in,
               cu_seqlens,
               conv_read_indices,
               conv_write_indices,
               ssm_read_indices,
               ssm_write_indices,
               num_accepted_tokens,
               mode,
               max_query_tokens,
               conv_output_slots,
               ssm_output_slots,
               output,
               conv_state_out,
               ssm_state_out);
  return std::make_tuple(output, conv_state_out, ssm_state_out);
}

}  // namespace

void register_mega_kda_ops(pybind11::module_& m) {
  m.def("mega_kda_prefill",
        &mega_kda_prefill,
        "mega_kda_prefill",
        pybind11::arg("qkv"),
        pybind11::arg("gate"),
        pybind11::arg("beta"),
        pybind11::arg("a_log"),
        pybind11::arg("gate_bias"),
        pybind11::arg("conv_weight"),
        pybind11::arg("conv_bias"),
        pybind11::arg("conv_state_in"),
        pybind11::arg("ssm_state_in"),
        pybind11::arg("cu_seqlens"),
        pybind11::arg("conv_read_indices"),
        pybind11::arg("conv_write_indices"),
        pybind11::arg("ssm_read_indices"),
        pybind11::arg("ssm_write_indices"),
        pybind11::arg("output"),
        pybind11::arg("conv_state_out"),
        pybind11::arg("ssm_state_out"));
  m.def("mega_kda_decode",
        &mega_kda_decode,
        "mega_kda_decode",
        pybind11::arg("qkv"),
        pybind11::arg("gate"),
        pybind11::arg("beta"),
        pybind11::arg("a_log"),
        pybind11::arg("gate_bias"),
        pybind11::arg("conv_weight"),
        pybind11::arg("conv_bias"),
        pybind11::arg("conv_state_in"),
        pybind11::arg("ssm_state_in"),
        pybind11::arg("cu_seqlens"),
        pybind11::arg("conv_read_indices"),
        pybind11::arg("conv_write_indices"),
        pybind11::arg("ssm_read_indices"),
        pybind11::arg("ssm_write_indices"),
        pybind11::arg("num_accepted_tokens"),
        pybind11::arg("mode"),
        pybind11::arg("max_query_tokens"),
        pybind11::arg("output"),
        pybind11::arg("conv_state_out"),
        pybind11::arg("ssm_state_out"));
}
