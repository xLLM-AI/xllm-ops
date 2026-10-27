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

# CrossAttention kernel uses the AtlasA2 catlass FA building blocks (Arch::AtlasA2,
# MmadAtlasA2FAIQK/PV ...). Those blocks live behind CATLASS_ARCH == 2201, so the
# define is required for every target, including ascend950.
set(CA_CATLASS_ARCH_DEF "-DCATLASS_ARCH=2201")

add_ops_compile_options(
    OP_NAME CrossAttention
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            ${CA_CATLASS_ARCH_DEF}
            -I${CANN_3RD_LIB_PATH}/catlass/include
            -I${CMAKE_CURRENT_LIST_DIR}/../../../
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE cross_attention ACLNNTYPE aclnn)
endif()
