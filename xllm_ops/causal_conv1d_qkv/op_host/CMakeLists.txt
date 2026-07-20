add_op_to_compiled_list()
set(causal_conv1d_qkv_depends causal_conv1d PARENT_SCOPE)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        causal_conv1d_qkv_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME CausalConv1dQkv
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE causal_conv1d_qkv ACLNNTYPE aclnn)
endif()
