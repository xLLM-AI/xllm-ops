# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
         x_flash_attention_infer_def.cpp
    )
endif()

# Dynamically set CATLASS_ARCH based on the SOC being built.
# NOTE: In the CMake scope SOC_VERSION may be empty; the reliable variable is
# ASCEND_COMPUTE_UNIT (see CMakeCache, e.g. "ascend950"). We accept both and also
# any *950 / *310p5 spelling. The previous `SOC_VERSION STREQUAL "Ascend950"`
# check never matched, so -DCATLASS_ARCH=3510 was never injected and the A5
# (arch35) branch was silently compiled out on the HOST side, causing the host
# tiling to fall back to XFAInferTilingData(SaveToBuffer) while the kernel read
# the 896B FATilingData -> field mis-alignment -> garbage tiling -> out all-zero.
string(TOLOWER "${SOC_VERSION}" _XFA_SOC_LOWER)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _XFA_UNIT_LOWER)
if(_XFA_SOC_LOWER MATCHES "ascend950" OR _XFA_SOC_LOWER MATCHES "ascend310p5"
   OR _XFA_UNIT_LOWER MATCHES "ascend950" OR _XFA_UNIT_LOWER MATCHES "ascend310p5")
    set(CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")
    set(_XFA_IS_A5 TRUE)
else()
    set(CATLASS_ARCH_DEF "-DCATLASS_ARCH=2201")
    set(_XFA_IS_A5 FALSE)
endif()

add_ops_compile_options(
    OP_NAME XFlashAttentionInfer
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            ${CATLASS_ARCH_DEF}
            -I${CANN_3RD_LIB_PATH}/catlass/include
)

# CRITICAL: add_ops_compile_options only affects the op_impl(kernel) build, NOT
# the host tiling object (ophost_xllm_tiling_obj). That object library is defined
# by the CANN framework in another directory (beam_search/op_host), so neither
# set_source_files_properties nor add_compile_definitions from this subdir can
# reach it. Instead we generate a per-build config header next to the tiling
# sources; the header content depends on the SOC selected at CONFIGURE time, so
# the A3 build gets an empty header (falls back to XFAInferTilingData/SaveToBuffer)
# while the A5 build gets `#define CATLASS_ARCH 3510` (compiles the arch35 branch).
# The tiling sources include this header via a same-dir relative path, so it works
# regardless of which aggregated target actually compiles the .cpp.
if(_XFA_IS_A5)
    set(XFA_ARCH_CONFIG_BODY "#define CATLASS_ARCH 3510")
else()
    set(XFA_ARCH_CONFIG_BODY "")
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/xfa_arch_config.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/xfa_arch_config.h
    @ONLY
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE  x_flash_attention_infer ACLNNTYPE aclnn)
endif()
