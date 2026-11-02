# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
  target_sources(op_host_aclnnExc PRIVATE dsa_kv_offloading_plan_nope_def.cpp)
endif()
add_ops_compile_options(OP_NAME DsaKvOffloadingPlanNope OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror)
if(NOT BUILD_OPS_RTY_KERNEL)
  add_modules_sources(OPTYPE dsa_kv_offloading_plan_nope ACLNNTYPE aclnn_exclude)
endif()
