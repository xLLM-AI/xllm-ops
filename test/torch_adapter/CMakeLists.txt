# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# http://www.apache.org/licenses/LICENSE-2.0

cmake_minimum_required(VERSION 3.18)
project(mx_adapter_regression LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(USE_NPU ON)
get_filename_component(OPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
if(NOT XLLM_SOURCE_DIR OR NOT DEPENDENCY_INCLUDE_DIR)
  message(FATAL_ERROR "Set XLLM_SOURCE_DIR and DEPENDENCY_INCLUDE_DIR")
endif()
if(NOT XLLM_BRIDGE_INCLUDE_DIR)
  set(XLLM_BRIDGE_INCLUDE_DIR "${XLLM_SOURCE_DIR}/xllm")
endif()
set(TORCH_INCLUDE_DIRS
    "$ENV{PYTORCH_INSTALL_PATH}/include"
    "$ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include")
include_directories(
    "${XLLM_BRIDGE_INCLUDE_DIR}"
    "${OPS_ROOT}"
    ${TORCH_INCLUDE_DIRS}
    "$ENV{PYTORCH_NPU_INSTALL_PATH}/include"
    "$ENV{PYTORCH_NPU_INSTALL_PATH}/include/third_party/acl/inc"
    "$ENV{PYTHON_INCLUDE_PATH}"
    "$ENV{NPU_HOME_PATH}/include"
    "$ENV{NPU_HOME_PATH}/include/aclnnop"
    "${XLLM_SOURCE_DIR}/xllm"
    "${XLLM_SOURCE_DIR}/xllm/core"
    "${DEPENDENCY_INCLUDE_DIR}")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
include("${OPS_ROOT}/cmake/mxfp8.cmake")
add_library(xllm_ops STATIC platform_fixture.cpp)
add_subdirectory("${OPS_ROOT}/torch_adapter" mx_adapter)

add_library(mx_adapter_probe MODULE probe.cpp test_bridge_probe.cpp)
target_link_directories(mx_adapter_probe PRIVATE
    "$ENV{PYTORCH_INSTALL_PATH}/lib"
    "$ENV{PYTORCH_NPU_INSTALL_PATH}/lib"
    "$ENV{NPU_HOME_PATH}/lib64"
    "$ENV{PYTHON_LIB_PATH}"
    "${DEPENDENCY_INCLUDE_DIR}/../lib")
target_link_libraries(mx_adapter_probe PRIVATE
    xllm_ops torch torch_cpu c10 torch_npu torch_python python3.11
    opapi nnopbase ascendcl glog gflags dl pthread)
target_link_options(mx_adapter_probe PRIVATE -Wl,--no-undefined)
set_target_properties(mx_adapter_probe PROPERTIES PREFIX "")
