#!/bin/bash

# 创建并进入构建目录
mkdir -p build
cd build

# 运行CMake配置
cmake ..

# 编译项目
make -j$(nproc)

# 返回上级目录
cd .. 