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
        cross_attention_def.cpp
    )
endif()

# Ascend950 (arch35 / DAV_3510) builds need the 3510 catlass constants
# (BYTE_PER_BLK_FP=64, MX_*); the AtlasA2 path keeps 2201.
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _CA_UNIT_LOWER)
if(_CA_UNIT_LOWER MATCHES "ascend950")
    set(CA_CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")
    set(CA_ARCH35_COMPILE_OPTIONS "-DENABLE_CV_COMM_VIA_SSBUF=true")
else()
    set(CA_CATLASS_ARCH_DEF "-DCATLASS_ARCH=2201")
    set(CA_ARCH35_COMPILE_OPTIONS "")
endif()

add_ops_compile_options(
    OP_NAME CrossAttention
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            -Wno-error
            -g
            ${CA_CATLASS_ARCH_DEF}
            ${CA_ARCH35_COMPILE_OPTIONS}
            -I${CANN_3RD_LIB_PATH}/catlass/include
            -I${CMAKE_CURRENT_LIST_DIR}/../third_party/catlass/include
            -I${CMAKE_CURRENT_LIST_DIR}/../../../
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE cross_attention ACLNNTYPE aclnn)
    # Native CV scheme: the Ascend950 (arch35) host tiling source lives in
    # op_host/arch35 and is not covered by the add_modules_sources GLOB.
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/arch35/cross_attention_tiling_arch35.cpp
        ${CMAKE_CURRENT_LIST_DIR}/arch35/cross_attention_tiling.cpp)
endif()
