add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        index_group_matmul_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME IndexGroupMatmul
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE index_group_matmul ACLNNTYPE aclnn)
endif()

