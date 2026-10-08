# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

set(DEQUANT_SITU_QUANT_SUPPORTED_SOCS ascend910_93)
if(NOT "ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    return()
endif()

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE dequant_situ_quant_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME DequantSituQuant
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dequant_situ_quant ACLNNTYPE aclnn)
endif()
