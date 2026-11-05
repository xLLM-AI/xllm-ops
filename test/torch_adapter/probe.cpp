// Copyright 2026 The xLLM Authors. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License").
// http://www.apache.org/licenses/LICENSE-2.0

#include <aclnn/acl_meta.h>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops/mx_quantization.h"

namespace mx = xllm::kernel::npu;
namespace bridge = xllm::kernel::npu::aclnn::detail;

int64_t test_bridge_descriptor_dtype(const at::Tensor& tensor);

namespace {
int64_t core_bridge_descriptor_dtype(const at::Tensor& tensor) {
  aclTensor* descriptor = bridge::convert_type(tensor);
  TORCH_CHECK(descriptor != nullptr, "core bridge returned no descriptor");
  aclDataType dtype = ACL_DT_UNDEFINED;
  const int32_t status = aclGetDataType(descriptor, &dtype);
  bridge::release(descriptor);
  TORCH_CHECK(status == 0, "aclGetDataType failed: ", status);
  TORCH_CHECK(bridge::convert_type(tensor.scalar_type()) == dtype,
              "core bridge scalar/tensor mappings disagree");
  return static_cast<int64_t>(dtype);
}
}  // namespace

PYBIND11_MODULE(mx_adapter_probe, module) {
  module.attr("e8m0_enum") =
      static_cast<int64_t>(at::ScalarType::Float8_e8m0fnu);
  module.def("test_descriptor_dtype", &test_bridge_descriptor_dtype);
  module.def("core_descriptor_dtype", &core_bridge_descriptor_dtype);
  module.def("core_scalar_dtype", [](int64_t dtype) {
    return static_cast<int64_t>(
        bridge::convert_type(static_cast<at::ScalarType>(dtype)));
  });
  module.def("available", &mx::mxfp8_runtime_available);
  module.def("dynamic", &mx::mxfp8_dynamic_quant);
  module.def("matmul4", &mx::mxfp4_quant_matmul);
  module.def("matmul8", &mx::mxfp8_quant_matmul);
  module.def("prepack", &mx::mxfp4_prepack_nz_weight);
  module.def("prefill", &mx::mxfp4_prefill_fp8_weight);
  module.def("swiglu4", &mx::mxfp4_swiglu_quant);
  module.def("grouped",
             [](const at::Tensor& x,
                const std::vector<at::Tensor>& weights,
                const std::vector<at::Tensor>& scales,
                const at::Tensor& x_scale,
                const at::Tensor& groups) {
               return mx::mxfp8_grouped_matmul_swiglu_quant(
                   x, weights, scales, x_scale, groups);
             });
}
