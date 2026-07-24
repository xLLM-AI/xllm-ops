
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        mla_preprocess_v2_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MlaPreprocessV2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -I${CMAKE_CURRENT_LIST_DIR}/../../mla_preprocess/op_kernel
)


if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE mla_preprocess_v2 ACLNNTYPE aclnn)
endif()
