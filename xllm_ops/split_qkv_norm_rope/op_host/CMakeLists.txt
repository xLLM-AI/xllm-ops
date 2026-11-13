# Copyright 2026 The xLLM Authors. All Rights Reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        split_qkv_norm_rope_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SplitQkvNormRope
    OPTIONS
        --cce-auto-sync=off
        -Wno-deprecated-declarations
        -Wno-error
)

if (NOT BUILD_OPS_RTY_KERNEL)
    # aclnn 接口由 opbuild 依据 OpDef 自动生成（本算子不手写 op_api）
    add_modules_sources(OPTYPE split_qkv_norm_rope ACLNNTYPE aclnn)
endif()
