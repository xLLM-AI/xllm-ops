# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        onerec_final_beam_select_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME OnerecFinalBeamSelect
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            -I${CMAKE_SOURCE_DIR}/common
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE onerec_final_beam_select ACLNNTYPE aclnn)
endif()