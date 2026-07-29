# Copyright 2026 The xLLM Authors. All Rights Reserved.

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        reshape_and_cache_a5_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ReshapeAndCacheA5
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Werror
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE reshape_and_cache_a5 ACLNNTYPE aclnn)
endif()
