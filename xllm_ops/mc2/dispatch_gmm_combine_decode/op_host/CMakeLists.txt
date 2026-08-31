set(_DISPATCH_GMM_INC_OPTS)
if (EXISTS ${CMAKE_SOURCE_DIR}/../third_party/catlass/include)
    list(APPEND _DISPATCH_GMM_INC_OPTS -I${CMAKE_SOURCE_DIR}/../third_party/catlass/include)
else()
    message(FATAL_ERROR "dependency catlass is missing, you can fetch it by running 'git submodule update --init --recursive'")
endif()

add_op_to_compiled_list()

string(TOLOWER "${SOC_VERSION}" _DGMM_SOC_LOWER)
string(TOLOWER "${ASCEND_COMPUTE_UNIT}" _DGMM_UNIT_LOWER)
if(_DGMM_SOC_LOWER MATCHES "ascend950" OR _DGMM_SOC_LOWER MATCHES "ascend310p5"
   OR _DGMM_UNIT_LOWER MATCHES "ascend950" OR _DGMM_UNIT_LOWER MATCHES "ascend310p5")
    set(DGMM_CATLASS_ARCH_DEF "-DCATLASS_ARCH=3510")
else()
    set(DGMM_CATLASS_ARCH_DEF "-DCATLASS_ARCH=2201")
endif()

add_ops_compile_options(
        OP_NAME DispatchGmmCombineDecode
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -DASCENDC_DUMP=0
                ${DGMM_CATLASS_ARCH_DEF}
                ${_DISPATCH_GMM_INC_OPTS}
)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        dispatch_gmm_combine_decode_def.cpp
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dispatch_gmm_combine_decode ACLNNTYPE aclnn_inner)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_SOURCE_DIR}/../utils/inc/kernel
    )
endif()
