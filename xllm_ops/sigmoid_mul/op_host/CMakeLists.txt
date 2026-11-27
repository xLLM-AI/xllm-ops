# -----------------------------------------------------------------------------------------------------------
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        sigmoid_mul_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SigmoidMul
    OPTIONS --cce-auto-sync=on
            -Wno-deprecated-declarations
            -Wno-error
)

# Two kernels share the op entry: 1000 flat elementwise, 1001 [M, 1] broadcast.
# Without this the build emits one unkeyed binary and the runtime rejects the launch.
add_ops_tiling_keys(
    OP_NAME SigmoidMul
    COMPUTE_UNIT ascend910_93 ascend910b
    TILING_KEYS 1000 1001
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sigmoid_mul ACLNNTYPE aclnn)
endif()
