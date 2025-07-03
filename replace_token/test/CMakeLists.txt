# CMake lowest version requirement
cmake_minimum_required(VERSION 3.14)

# 设置工程名
project(ACLNN_EXAMPLE)

# Compile options
add_compile_options(-std=c++11)

# 设置编译选项
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")    
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 设置可执行文件名（如opapi_test），并指定待运行算子文件*.cpp所在目录
add_executable(test_replace_token
               test_replace_token.cpp) 

# 设置ASCEND_PATH（CANN软件包目录，请根据实际路径修改）和INCLUDE_BASE_DIR（头文件目录）
if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")      
    set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
set(OP_API_PATH "/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
    ${OP_API_PATH}/include
)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/xllm/op_api/lib/")
# 设置链接的库文件路径
target_link_libraries(test_replace_token PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libnnopbase.so
                      ${ASCEND_PATH}/lib64/libacl_op_compiler.so
                      ${OP_API_PATH}/lib/libcust_opapi.so
                      stdc++)

# 可执行文件在CMakeLists文件所在目录的bin目录下
install(TARGETS test_replace_token DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})