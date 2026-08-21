# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        x_attention_def.cpp
    )
endif()

# Dynamically set CATLASS_ARCH based on the SOC being built.
# NOTE: In the CMake scope SOC_VERSION may be empty; the reliable variable is
# ASCEND_COMPUTE_UNIT (see CMakeCache, e.g. "ascend950"). We accept both and also
# any *950 / *310p5 spelling. On A5 we inject -DCATLASS_ARCH=3510 so the arch35
# (A5) branch is compiled in for both kernel and host; on A3 it is empty and the
# A3 (AtlasA2) branch is used. The host tiling object (ophost_xllm_tiling_obj)
# is built by the framework in another directory, so per-source compile defs from
# here cannot reach it; instead we generate xa_arch_config.h (included by the
# tiling sources) to bake the arch selection in at configure time.
string(TOLOWER "${SOC_VERSION}" _XA_SOC_LOWER)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _XA_UNIT_LOWER)
if(_XA_SOC_LOWER MATCHES "ascend950" OR _XA_SOC_LOWER MATCHES "ascend310p5"
   OR _XA_UNIT_LOWER MATCHES "ascend950" OR _XA_UNIT_LOWER MATCHES "ascend310p5")
    set(CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")
    set(_XA_IS_A5 TRUE)
else()
    set(CATLASS_ARCH_DEF "")
    set(_XA_IS_A5 FALSE)
endif()

add_ops_compile_options(
    OP_NAME XAttention
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            ${CATLASS_ARCH_DEF}
            -I${CANN_3RD_LIB_PATH}/catlass/include
            -I${CMAKE_CURRENT_LIST_DIR}/../../../
)

# Generate per-build arch config header consumed by the host tiling sources
# (x_attention_tiling.h includes xa_arch_config.h). A5 builds get
# `#define CATLASS_ARCH 3510`; A3 builds get an empty header.
if(_XA_IS_A5)
    set(XA_ARCH_CONFIG_BODY "#define CATLASS_ARCH 3510")
else()
    set(XA_ARCH_CONFIG_BODY "")
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/xa_arch_config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/xa_arch_config.h
    @ONLY
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE x_attention ACLNNTYPE aclnn)
endif()