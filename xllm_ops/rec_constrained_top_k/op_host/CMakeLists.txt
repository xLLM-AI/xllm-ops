# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        rec_constrained_top_k_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME RecConstrainedTopK
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE rec_constrained_top_k ACLNNTYPE aclnn)
endif()