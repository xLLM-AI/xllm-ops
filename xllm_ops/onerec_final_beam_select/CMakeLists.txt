# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

add_ops_compile_options(
    OP_NAME OnerecFinalBeamSelect
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
            -I${CMAKE_SOURCE_DIR}/common
)

target_sources(op_host_aclnn PRIVATE
    onerec_final_beam_select_cpu.cpp
)

target_sources(optiling PRIVATE
    onerec_final_beam_select_cpu.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
    onerec_final_beam_select_cpu.cpp
)

install(FILES onerec_final_beam_select.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)

install(FILES onerec_final_beam_select.h
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic)
