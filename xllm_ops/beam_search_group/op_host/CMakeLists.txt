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
        beam_search_group_def.cpp
    )
endif()

# Dynamically set CATLASS_ARCH based on the SOC being built.
# beam_search_group is a pure-vector op, but common/common.h unconditionally
# #includes "catlass/gemm/tile/tile_copy.hpp", whose dispatch headers
# (copy_gm_to_l1.hpp etc.) only bring in an implementation when CATLASS_ARCH is
# defined (2201=AtlasA2 / 3510=Ascend950). On the A5 (ascend950) build the macro
# was never injected, so the ascend950 cube templates were compiled out and the
# TileCopy aliases referred to a non-existent CopyGmToL1 -> "no template named".
# We follow the same SOC-aware injection as x_flash_attention_infer, but here we
# use the macro to explicitly distinguish A3 from A5 (per requirement): emit
#   -DCATLASS_ARCH=3510  for ascend950/310p5  (A5, ascend950 cube path)
#   -DCATLASS_ARCH=2201  otherwise            (A3 ascend910_93 / A2 ascend910b,
#                                              AtlasA2 cube path)
# Both branches MUST define CATLASS_ARCH, because tile_copy.hpp's dispatch header
# only brings in a CopyGmToL1 implementation when the macro is set. Leaving it
# empty would also break A3/A2 (CopyGmToL1 would not exist there either).
string(TOLOWER "${SOC_VERSION}" _BSG_SOC_LOWER)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _BSG_UNIT_LOWER)
if(_BSG_SOC_LOWER MATCHES "ascend950" OR _BSG_SOC_LOWER MATCHES "ascend310p5"
   OR _BSG_UNIT_LOWER MATCHES "ascend950" OR _BSG_UNIT_LOWER MATCHES "ascend310p5")
    set(BSG_CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")   # A5 (ascend950)
else()
    set(BSG_CATLASS_ARCH_DEF "-DCATLASS_ARCH=2201")   # A3 (ascend910_93) / A2 (ascend910b)
endif()

add_ops_compile_options(
    OP_NAME BeamSearchGroup
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            ${BSG_CATLASS_ARCH_DEF}
            -I${CANN_3RD_LIB_PATH}/catlass/include
            -I${CMAKE_CURRENT_LIST_DIR}/
            -I${CMAKE_CURRENT_LIST_DIR}/../../../
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE beam_search_group ACLNNTYPE aclnn)
endif()