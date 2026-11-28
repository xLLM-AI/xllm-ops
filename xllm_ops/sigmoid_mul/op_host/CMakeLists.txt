# Copyright 2026 The xLLM Authors. All Rights Reserved.
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        sigmoid_mul_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SigmoidMul
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Wno-error
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sigmoid_mul ACLNNTYPE aclnn_exclude)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/pkg_inc
    )
endif()
