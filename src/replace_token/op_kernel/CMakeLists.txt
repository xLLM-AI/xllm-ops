# set custom compile options
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0)
endif()

add_ops_compile_options(ALL OPTIONS -sanitizer)

if (${ENABLE_SOURCE_PACKAGE})
    set(DYNAMIC_PATH ${CMAKE_CURRENT_BINARY_DIR}/binary/dynamic)
    execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_source_compile.py
                            --dynamic-dir=${DYNAMIC_PATH}
                            --src-dir=${CMAKE_SOURCE_DIR}/op_kernel
                    RESULT_VARIABLE EXEC_RESULT
                    ERROR_VARIABLE EXEC_ERROR
    )
    if (${EXEC_RESULT})
        message(FATAL_ERROR, "get op name failed, gen error:${EXEC_ERROR}" )
    endif()
else()
    set(DYNAMIC_PATH "")
endif()

foreach(compute_unit ${ASCEND_COMPUTE_UNIT})

    # generate aic-${compute_unit}-ops-info.json
    add_ops_info_target(TARGET ops_info_gen_${compute_unit}
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json
        OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
        INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/config/${compute_unit}
    )

    # define a target:binary to prevent kernel file from being rebuilt during the preinstall process
    if (NOT TARGET binary)
        add_custom_target(binary)
    endif()

    if (${ENABLE_BINARY_PACKAGE} OR ${ENABLE_SOURCE_PACKAGE})
        if (${ENABLE_BINARY_PACKAGE})
            # gen binary_info_config.json and <file_name>.json
            add_custom_target(ascendc_bin_${compute_unit}_gen_ops_config
                            COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/insert_simplified_keys.py
                                    -p ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
                            COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_ops_config.py
                                    -p ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}
                                    -s ${compute_unit}
                            COMMAND ${CMAKE_COMMAND} -E make_directory
                                    ${CMAKE_CURRENT_BINARY_DIR}/binary/config/${compute_unit}
                            COMMAND mv ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}/*.json
                                    ${CMAKE_CURRENT_BINARY_DIR}/binary/config/${compute_unit}
            )
        else()
            if (NOT TARGET ascendc_bin_${compute_unit}_gen_ops_config)
                add_custom_target(ascendc_bin_${compute_unit}_gen_ops_config)
            endif()
        endif()
        add_dependencies(binary ascendc_bin_${compute_unit}_gen_ops_config)

        # get op_type-op_name from aic-${compute_unit}-ops-info.ini
        execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_get_op_name.py
                                --ini-file=${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini
                        OUTPUT_VARIABLE OP_TYPA_NAME
                        RESULT_VARIABLE EXEC_RESULT
                        ERROR_VARIABLE EXEC_ERROR
        )
        if (${EXEC_RESULT})
            message(FATAL_ERROR, "get op name failed, gen error: ${EXEC_ERROR}")
        endif()

        # compile op one by one with ascendc_compile_kernel.py
        string(REPLACE "\n" ";" TYPE_NAME_LIST "${OP_TYPA_NAME}")
        foreach(TYPE_NAME IN LISTS TYPE_NAME_LIST)
            if (NOT "${TYPE_NAME}" STREQUAL "")
                string(REPLACE "-" ";" bin_sep ${TYPE_NAME})
                list(GET bin_sep 0 op_type)
                list(GET bin_sep 1 op_file)
                add_kernel_compile(${op_type} ${CMAKE_SOURCE_DIR}/op_kernel/${op_file}.cpp ${compute_unit} ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core/${compute_unit}/aic-${compute_unit}-ops-info.json)
            endif()
        endforeach()
    endif()
endforeach()

# generate npu_supported_ops.json
add_npu_support_target(TARGET npu_supported_ops
    OPS_INFO_DIR ${ASCEND_AUTOGEN_PATH}
    OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/tbe/op_info_cfg/ai_core
    INSTALL_DIR packages/vendors/${vendor_name}/framework/${ASCEND_FRAMEWORK_TYPE}
)

if(ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()

# install kernel file
if (${ENABLE_SOURCE_PACKAGE})
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/dynamic/
            DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/${vendor_name}_impl/dynamic/
    )
endif()

# install *.o files and *.json files 
if (${ENABLE_BINARY_PACKAGE})
    set(INSTALL_DIR packages/vendors/${vendor_name}/op_impl/ai_core/tbe/)
    foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/${compute_unit}/
            DESTINATION ${INSTALL_DIR}/kernel/${compute_unit}/
        )
    endforeach()
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/binary/config/
        DESTINATION ${INSTALL_DIR}/kernel/config/
    )
endif()
