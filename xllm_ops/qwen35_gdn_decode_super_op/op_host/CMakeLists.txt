if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        qwen35_gdn_decode_super_op_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME Qwen35GdnDecodeSuperOp
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/pto-isa/include
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE qwen35_gdn_decode_super_op ACLNNTYPE aclnn)
endif()
