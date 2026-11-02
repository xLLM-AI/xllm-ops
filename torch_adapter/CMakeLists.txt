# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Attach the adapters to xLLM's existing Torch/CANN host library.
# The consumer supplies its ACLNN invocation helper and platform guard.
if(NOT TARGET xllm_ops)
  message(FATAL_ERROR "MX Torch adapters require the xllm_ops host target")
endif()

# The MX GEMMs and W4 weight helpers have their own capability probe:
# QuantMatmulV5 is enough for both GEMMs, and the layout helpers only need
# generic Torch-NPU format casts.  Keep them out of the aggregate MXFP8
# selection so a partial FP8 install does not disable a usable MXFP4 path.
set(MX_QUANT_SOURCES mx_linear.cpp)
if(XLLM_HAS_NPU_MXFP8)
  list(APPEND MX_QUANT_SOURCES mxfp8_quant.cpp)
else()
  list(APPEND MX_QUANT_SOURCES mxfp8_quant_unavailable.cpp)
endif()

target_sources(xllm_ops PRIVATE ${MX_QUANT_SOURCES})
target_include_directories(xllm_ops PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
)
