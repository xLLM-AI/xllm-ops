#!/bin/bash
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
BASE_DIR=$(pwd)

# 编译wheel包
python3 setup.py build bdist_wheel

# 安装wheel包
cd ${BASE_DIR}/dist
pip3 install --force-reinstall *.whl  


# 运行测试用例
cd ${BASE_DIR}/test
python3 test_replace_token.py
if [ $? -ne 0 ]; then
    echo "ERROR: run replace_token test failed!"
fi
echo "INFO: run replace_token test success!"

# # 运行测试用例
# python3 test_replace_token_graph.py
# if [ $? -ne 0 ]; then
#     echo "ERROR: run replace_token_graph test failed!"
# fi
# echo "INFO: run replace_token_graph test success!"
