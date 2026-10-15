# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()
add_modules_sources(OPTYPE quant_lightning_indexer_v2_metadata_va ACLNNTYPE aclnn)

# AICPU kernel 位于 op_kernel_aicpu/ (与 op_host 同级)。add_modules_sources 只 glob
# op_host/*_aicpu*.cpp，不会编译它；此处显式把 AICPU kernel 编进 libmath_aicpu_kernels.so
# (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON 时走 add_aicpu_cust_kernel_modules 交叉编译)。
file(GLOB QLI_META_AICPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel_aicpu/*_aicpu*.cpp)
if(QLI_META_AICPU_SRCS)
    file(GLOB QLI_META_AICPU_JSON ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel_aicpu/*.json)
    add_aicpu_cust_kernel_modules(quant_lightning_indexer_v2_metadata_va ${QLI_META_AICPU_SRCS} ${QLI_META_AICPU_JSON})
endif()
