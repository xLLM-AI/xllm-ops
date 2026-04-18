# Copyright (c) 2026 Huawei Technologies Co., Ltd.

add_ops_compile_options(
        OP_NAME RecurrentGatedDeltaRule
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Wno-error
)

target_sources(op_host_aclnnExc PRIVATE
    recurrent_gated_delta_rule_def.cpp
)

target_sources(opapi PRIVATE
    recurrent_gated_delta_rule_l0.cpp
    aclnn_recurrent_gated_delta_rule.cpp
)

# optiling
target_sources(optiling PRIVATE
    recurrent_gated_delta_rule_tiling.cpp
)

target_include_directories(optiling PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# opproto
target_sources(opsproto PRIVATE
    recurrent_gated_delta_rule_infershape.cpp
)

target_include_directories(opsproto PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

install(FILES recurrent_gated_delta_rule.cpp
    DESTINATION ${ASCEND_IMPL_OUT_DIR}/dynamic
)
