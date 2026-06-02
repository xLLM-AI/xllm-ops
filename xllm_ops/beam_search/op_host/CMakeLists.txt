add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        beam_search_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME BeamSearch
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Werror
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/catlas/include
            -I${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE beam_search ACLNNTYPE aclnn)
endif()
