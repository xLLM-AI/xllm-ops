# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        qwen35_gdn_prefill_super_op_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME Qwen35GdnPrefillSuperOp
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            -DGDN_KERNEL_NAME=qwen35_gdn_prefill_super_op
            -DGDN_D=128
            -DGDN_C=128
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../causal_conv1d/op_kernel
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/pto-isa/include
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE qwen35_gdn_prefill_super_op ACLNNTYPE aclnn)
endif()
