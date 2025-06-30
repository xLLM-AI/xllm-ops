cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test_ops_custom)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置ASCEND相关环境变量
set(ASCEND_DIR $ENV{ASCEND_DIR})
if(NOT DEFINED ASCEND_DIR)
    set(ASCEND_DIR /usr/local/Ascend)
endif()

# 添加ASCEND相关包含目录
include_directories(
    ${ASCEND_DIR}/include
    ${ASCEND_DIR}/include/ascend_nn
)

# 添加ASCEND库目录
link_directories(
    ${ASCEND_DIR}/lib64
)

# 查找LibTorch包
find_package(Torch REQUIRED)

# 源文件
set(SOURCES
    test_ops_custom.cpp
    replace_token_op.cpp
)

# 添加可执行文件
add_executable(test_ops_custom ${SOURCES})

# 链接LibTorch和ASCEND库
target_link_libraries(test_ops_custom 
    ${TORCH_LIBRARIES}
    ascendcl
    acl_op_compiler
)

# 设置编译选项
set_property(TARGET test_ops_custom PROPERTY CXX_STANDARD 14) 