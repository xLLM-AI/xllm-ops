
function(get_system_info SYSTEM_INFO)
  if (UNIX)
    execute_process(COMMAND grep -i ^id= /etc/os-release OUTPUT_VARIABLE TEMP)
    string(REGEX REPLACE "\n|id=|ID=|\"" "" SYSTEM_NAME ${TEMP})
    set(${SYSTEM_INFO} ${SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR} PARENT_SCOPE)
  elseif (WIN32)
    message(STATUS "System is Windows. Only for pre-build.")
  else ()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} not support.")
  endif ()
endfunction()

function(opbuild)
  message(STATUS "Opbuild generating sources")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;ENABLE_SOURCE" "OPS_SRC" ${ARGN})
  execute_process(COMMAND ${CMAKE_COMPILE} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
                  -I ${ASCEND_CANN_PACKAGE_PATH}/include -I ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
                  -L ${ASCEND_CANN_PACKAGE_PATH}/lib64 -lexe_graph -lregister -ltiling_api
                  -o ${OPBUILD_OUT_DIR}/libascend_all_ops.so
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  if (${EXEC_RESULT})
    message("build ops lib info: ${EXEC_INFO}")
    message("build ops lib error: ${EXEC_ERROR}")
    message(FATAL_ERROR "opbuild run failed!")
  endif()
  set(proj_env "")
  set(prefix_env "")
  if (NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    set(proj_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if (NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    set(prefix_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()

  set(ENV{ENABLE_SOURCE_PACAKGE} ${OPBUILD_ENABLE_SOURCE})
  execute_process(COMMAND ${proj_env} ${prefix_env} ${ASCEND_CANN_PACKAGE_PATH}/toolkit/tools/opbuild/op_build
                          ${OPBUILD_OUT_DIR}/libascend_all_ops.so ${OPBUILD_OUT_DIR}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR
  )
  unset(ENV{ENABLE_SOURCE_PACAKGE})
  if (${EXEC_RESULT})
    message("opbuild ops info: ${EXEC_INFO}")
    message("opbuild ops error: ${EXEC_ERROR}")
  endif()
  message(STATUS "Opbuild generating sources - done")
endfunction()

function(add_ops_info_target)
  cmake_parse_arguments(OPINFO "" "TARGET;OPS_INFO;OUTPUT;INSTALL_DIR" "" ${ARGN})
  get_filename_component(opinfo_file_path "${OPINFO_OUTPUT}" DIRECTORY)
  add_custom_command(OUTPUT ${OPINFO_OUTPUT}
      COMMAND mkdir -p ${opinfo_file_path}
      COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/parse_ini_to_json.py
              ${OPINFO_OPS_INFO} ${OPINFO_OUTPUT}
  )
  add_custom_target(${OPINFO_TARGET} ALL
      DEPENDS ${OPINFO_OUTPUT}
  )
  install(FILES ${OPINFO_OUTPUT}
          DESTINATION ${OPINFO_INSTALL_DIR}
  )
endfunction()

function(add_ops_compile_options OP_TYPE)
  cmake_parse_arguments(OP_COMPILE "" "OP_TYPE" "COMPUTE_UNIT;OPTIONS" ${ARGN})
  execute_process(COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_gen_options.py
                          ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS} ${OP_TYPE} ${OP_COMPILE_COMPUTE_UNIT}
                          ${OP_COMPILE_OPTIONS}
                  RESULT_VARIABLE EXEC_RESULT
                  OUTPUT_VARIABLE EXEC_INFO
                  ERROR_VARIABLE  EXEC_ERROR)
  if (${EXEC_RESULT})
      message("add ops compile options info: ${EXEC_INFO}")
      message("add ops compile options error: ${EXEC_ERROR}")
      message(FATAL_ERROR "add ops compile options failed!")
  endif()
endfunction()

function(add_npu_support_target)
  cmake_parse_arguments(NPUSUP "" "TARGET;OPS_INFO_DIR;OUT_DIR;INSTALL_DIR" "" ${ARGN})
  get_filename_component(npu_sup_file_path "${NPUSUP_OUT_DIR}" DIRECTORY)
  add_custom_command(OUTPUT ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    COMMAND mkdir -p ${NPUSUP_OUT_DIR}
    COMMAND ${CMAKE_SOURCE_DIR}/cmake/util/gen_ops_filter.sh
            ${NPUSUP_OPS_INFO_DIR}
            ${NPUSUP_OUT_DIR}
  )
  add_custom_target(npu_supported_ops ALL
    DEPENDS ${NPUSUP_OUT_DIR}/npu_supported_ops.json
  )
  install(FILES ${NPUSUP_OUT_DIR}/npu_supported_ops.json
    DESTINATION ${NPUSUP_INSTALL_DIR}
  )
endfunction()

function(add_kernel_compile op_type src compute_unit json_file_path)
  cmake_parse_arguments(BINCMP "" "OPS_INFO;OUT_DIR;TILING_LIB" "COMPUTE_UNIT;OPTIONS;CONFIGS" ${ARGN})
  if (NOT DEFINED BINCMP_OUT_DIR)
    set(BINCMP_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/binary)
  endif()
  if (NOT DEFINED BINCMP_TILING_LIB)
    set(BINCMP_TILING_LIB $<TARGET_FILE:cust_optiling>)
  endif()

  # add Environment Variable Configurations of ccache
  set(_ASCENDC_ENV_VAR)
  if(${CMAKE_CXX_COMPILER_LAUNCHER} MATCHES "ccache$")
    list(APPEND _ASCENDC_ENV_VAR export ASCENDC_CCACHE_EXECUTABLE=${CMAKE_CXX_COMPILER_LAUNCHER} &&)
  endif()

  if (NOT DEFINED BINCMP_OPS_INFO)
    set(BINCMP_OPS_INFO ${ASCEND_AUTOGEN_PATH}/aic-${compute_unit}-ops-info.ini)
  endif()
  if (NOT ${ENABLE_CROSS_COMPILE})
    add_custom_target(${op_type}_${compute_unit}
                      COMMAND ${_ASCENDC_ENV_VAR} ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_compile_kernel.py
                      --op-name=${op_type}
                      --src-file=${src}
                      --compute-unit=${compute_unit}
                      --compile-options=\"${BINCMP_OPTIONS}\"
                      --debug-config=\"${BINCMP_CONFIGS}\"
                      --config-ini=${BINCMP_OPS_INFO}
                      --tiling-lib=${BINCMP_TILING_LIB}
                      --output-path=${BINCMP_OUT_DIR}
                      --dynamic-dir=${DYNAMIC_PATH}
                      --enable-binary=\"${ENABLE_BINARY_PACKAGE}\"
                      --json-file=${json_file_path}
                      --build-tool=$(MAKE))
    add_dependencies(${op_type}_${compute_unit} cust_optiling)
  else()
    if (NOT DEFINED HOST_NATIVE_TILING_LIB)
      message("Native host libs was not set for cross compile!")
    endif()
    add_custom_target(${op_type}_${compute_unit}
                      COMMAND ${_ASCENDC_ENV_VAR} ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/cmake/util/ascendc_compile_kernel.py
                      --op-name=${op_type}
                      --src-file=${src}
                      --compute-unit=${compute_unit}
                      --compile-options=\"${BINCMP_OPTIONS}\"
                      --debug-config=\"${BINCMP_CONFIGS}\"
                      --config-ini=${BINCMP_OPS_INFO}
                      --tiling-lib=${HOST_NATIVE_TILING_LIB}
                      --output-path=${BINCMP_OUT_DIR}
                      --dynamic-dir=${DYNAMIC_PATH}
                      --enable-binary=\"${ENABLE_BINARY_PACKAGE}\"
                      --json-file=${json_file_path}
                      --build-tool=$(MAKE))
  endif()
  add_dependencies(ascendc_bin_${compute_unit}_gen_ops_config ${op_type}_${compute_unit})
  add_dependencies(${op_type}_${compute_unit} ops_info_gen_${compute_unit})
endfunction()