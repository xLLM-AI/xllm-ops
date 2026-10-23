# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)

option(XLLM_ENABLE_NPU_MXFP8 "Enable MXFP8 when Torch-NPU/CANN support it" ON)
cmake_push_check_state(RESET)
set(_mxfp8_try_compile_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_REQUIRED_INCLUDES
    ${TORCH_INCLUDE_DIRS}
    "$ENV{PYTORCH_INSTALL_PATH}/include"
    "$ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include")
check_cxx_source_compiles([[
  #include <torch/types.h>
  int main() { (void)torch::kFloat8_e8m0fnu; }
]] XLLM_TORCH_HAS_E8M0)
if(XLLM_TORCH_HAS_E8M0)
  add_compile_definitions(XLLM_HAS_FLOAT8_E8M0)
endif()

set(XLLM_HAS_NPU_MXFP8 OFF)
if(USE_NPU)
  list(APPEND CMAKE_REQUIRED_INCLUDES
      "$ENV{PYTORCH_NPU_INSTALL_PATH}/include"
      "$ENV{NPU_HOME_PATH}/include"
      "$ENV{NPU_HOME_PATH}/include/aclnnop")
  check_cxx_source_compiles([[
    #include <torch_npu/csrc/aten/CustomFunctions.h>
    #include <aclnn_dynamic_mx_quant.h>
    int main() {
      at::Tensor x;
      (void)at_npu::native::custom_ops::npu_dynamic_mx_quant(
          x, -1, "rint", 292, 32, 0, 0.0);
    }
  ]] XLLM_TORCH_NPU_HAS_MX_DYNAMIC_QUANT)
  if(XLLM_TORCH_NPU_HAS_MX_DYNAMIC_QUANT)
    add_compile_definitions(XLLM_HAS_NPU_MX_DYNAMIC_QUANT)
  endif()
  check_cxx_source_compiles([[
    #include <torch_npu/csrc/aten/CustomFunctions.h>
    #include <aclnn_quant_matmul_v5.h>
    int main() {
      at::Tensor x;
      (void)at_npu::native::custom_ops::npu_quant_matmul(
          x, x, x, std::nullopt, x, std::nullopt, 27,
          292, 292, 293, 293, c10::IntArrayRef({1, 1, 32}), std::nullopt);
    }
  ]] XLLM_TORCH_NPU_HAS_MX_MATMUL)
  if(XLLM_TORCH_NPU_HAS_MX_MATMUL)
    add_compile_definitions(XLLM_HAS_NPU_MX_MATMUL)
  endif()
  check_cxx_source_compiles([[
    #include <torch_npu/csrc/aten/CustomFunctions.h>
    #include <aclnn_swiglu_mx_quant.h>
    int main() {
      at::Tensor x;
      (void)at_npu::native::custom_ops::npu_swiglu_mx_quant(
          x, std::nullopt, -1, true, 0, 1e6, 1.702, 1.0,
          0, -1, 292, "rint", 0, 0.0);
    }
  ]] XLLM_TORCH_NPU_HAS_MX_SWIGLU)
  if(XLLM_TORCH_NPU_HAS_MX_SWIGLU)
    add_compile_definitions(XLLM_HAS_NPU_MX_SWIGLU)
  endif()
  check_cxx_source_compiles([[
    #include <aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h>
    int main() {
      (void)&aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize;
    }
  ]] XLLM_TORCH_NPU_HAS_MX_GROUPED_SWIGLU_NZ)
  if(XLLM_TORCH_NPU_HAS_MX_GROUPED_SWIGLU_NZ)
    add_compile_definitions(XLLM_HAS_NPU_MX_GROUPED_SWIGLU_NZ)
  endif()
endif()
if(USE_NPU AND XLLM_ENABLE_NPU_MXFP8 AND XLLM_TORCH_HAS_E8M0)
  check_cxx_source_compiles([[
    #include <torch_npu/csrc/aten/CustomFunctions.h>
    #include <aclnn_dynamic_mx_quant.h>
    #include <aclnn_swiglu_mx_quant.h>
    #include <aclnn_rms_norm_dynamic_mx_quant.h>
    #include <aclnn_add_rms_norm_dynamic_mx_quant.h>
    #include <aclnn_quant_matmul_v5.h>
    #include <aclnn_grouped_matmul_swiglu_quant_v2.h>
    #include <aclnn_grouped_matmul_swiglu_quant_weight_nz_v2.h>
    int main() {
      at::Tensor x;
      (void)at_npu::native::custom_ops::npu_dynamic_mx_quant(
          x, -1, "rint", 292, 32, 0, 0.0);
      (void)at_npu::native::custom_ops::npu_swiglu_mx_quant(
          x, std::nullopt, -1, true, 0, 1e6, 1.702, 1.0,
          0, -1, 292, "rint", 0, 0.0);
      (void)at_npu::native::custom_ops::npu_rms_norm_dynamic_mx_quant(
          x, x, std::nullopt, 1e-6, 0, "rint", 292);
      (void)at_npu::native::custom_ops::npu_add_rms_norm_dynamic_mx_quant(
          x, x, x, std::nullopt, 1e-6, 0, "rint", 292);
      (void)at_npu::native::custom_ops::npu_quant_matmul(
          x, x, x, std::nullopt, x, std::nullopt, 27,
          292, 292, 293, 293, c10::IntArrayRef({1, 1, 32}), std::nullopt);
      (void)&aclnnQuantMatmulV5GetWorkspaceSize;
      (void)&aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize;
      (void)&aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize;
    }
  ]] XLLM_TORCH_NPU_CANN_HAS_MXFP8)
  if(XLLM_TORCH_NPU_CANN_HAS_MXFP8)
    set(XLLM_HAS_NPU_MXFP8 ON)
    add_compile_definitions(XLLM_HAS_NPU_MXFP8)
  endif()
endif()
message(STATUS "NPU MX capabilities: dynamic=${XLLM_TORCH_NPU_HAS_MX_DYNAMIC_QUANT}"
               " matmul=${XLLM_TORCH_NPU_HAS_MX_MATMUL}"
               " swiglu=${XLLM_TORCH_NPU_HAS_MX_SWIGLU}"
               " grouped_nz=${XLLM_TORCH_NPU_HAS_MX_GROUPED_SWIGLU_NZ}"
               " full=${XLLM_HAS_NPU_MXFP8}")
if(_mxfp8_try_compile_type STREQUAL "")
  unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
else()
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_mxfp8_try_compile_type}")
endif()
cmake_pop_check_state()
