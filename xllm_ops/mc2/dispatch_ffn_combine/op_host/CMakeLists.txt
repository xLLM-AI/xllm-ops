set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()
if (EXISTS ${CMAKE_SOURCE_DIR}/third_party/catlass/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/third_party/catlass/include)
endif()
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel)
endif()

add_op_to_compiled_list()

string(TOLOWER "${SOC_VERSION}" _DFFN_SOC_LOWER)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _DFFN_UNIT_LOWER)
if(_DFFN_SOC_LOWER MATCHES "ascend950" OR _DFFN_SOC_LOWER MATCHES "ascend310p5"
   OR _DFFN_UNIT_LOWER MATCHES "ascend950" OR _DFFN_UNIT_LOWER MATCHES "ascend310p5")
    set(DFFN_CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")
else()
    set(DFFN_CATLASS_ARCH_DEF "-DCATLASS_ARCH=2201")
endif()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        dispatch_ffn_combine_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME DispatchFFNCombine
    OPTIONS
        --cce-auto-sync=on
        -Wno-deprecated-declarations
        -Werror
        -DHCCL_COMM
        ${DFFN_CATLASS_ARCH_DEF}
        ${_DISPATCH_FFN_INC_OPTS}
        -I${CANN_3RD_LIB_PATH}/catlass/include
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dispatch_ffn_combine ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
