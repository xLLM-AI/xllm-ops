add_op_to_compiled_list()
add_ops_compile_options(
    OP_NAME SparseFlashAttention
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -fpermissive
)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        sparse_flash_attention_def.cpp
    )
endif()



if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sparse_flash_attention ACLNNTYPE aclnn)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
endif()
