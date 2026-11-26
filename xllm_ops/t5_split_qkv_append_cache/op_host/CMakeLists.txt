add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        t5_split_qkv_append_cache_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME T5SplitQkvAppendCache
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE t5_split_qkv_append_cache ACLNNTYPE aclnn)
endif()
