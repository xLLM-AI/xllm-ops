# Copyright 2026 The xLLM Authors. All Rights Reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE rms_norm_channels_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME RmsNormChannels
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE rms_norm_channels ACLNNTYPE aclnn)
endif()
