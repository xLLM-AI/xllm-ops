# Copyright 2026 The xLLM Authors. All Rights Reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        layer_norm_fwd_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME LayerNormFwd
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE layer_norm_fwd ACLNNTYPE aclnn)
endif()
