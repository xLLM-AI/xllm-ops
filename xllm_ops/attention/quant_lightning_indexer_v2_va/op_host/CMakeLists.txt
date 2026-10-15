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

if (BUILD_OPEN_PROJECT)
    # M2: 原 _depends 指向 attention/lightning_indexer_v2, 该算子目录在本仓 (xllm_ops) 不存在;
    # custom_build.cmake 对缺失目录的 GLOB 结果为空且 install ... OPTIONAL 不报错, 依赖恒为空,
    # 属死依赖, 故删除。
    target_sources(op_host_aclnn PRIVATE
        quant_lightning_indexer_v2_va_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME QuantLightningIndexerV2Va
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE quant_lightning_indexer_v2_va ACLNNTYPE aclnn)
endif()
