# Copyright 2026 The xLLM Authors. All Rights Reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        ecr_expert_selection_threshold_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME EcrExpertSelectionThreshold
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Wno-error
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE ecr_expert_selection_threshold ACLNNTYPE aclnn)
endif()
