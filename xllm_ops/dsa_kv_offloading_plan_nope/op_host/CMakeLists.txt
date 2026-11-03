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
set(dsa_kv_offloading_plan_nope_depends dsa_kv_offloading_common PARENT_SCOPE)
get_filename_component(DSA_KV_OFFLOADING_KERNEL_INCLUDE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../dsa_kv_offloading_common/op_kernel" ABSOLUTE)
if(BUILD_OPEN_PROJECT)
  target_sources(op_host_aclnnExc PRIVATE dsa_kv_offloading_plan_nope_def.cpp)
endif()
add_ops_compile_options(OP_NAME DsaKvOffloadingPlanNope OPTIONS
  --cce-auto-sync=off -Wno-deprecated-declarations -Werror
  "-I${DSA_KV_OFFLOADING_KERNEL_INCLUDE}")
# Keep installed kernels self-contained when the source-tree include is absent.
install(FILES "${DSA_KV_OFFLOADING_KERNEL_INCLUDE}/dsa_kv_offloading_plan_serve.h"
  DESTINATION "${IMPL_INSTALL_DIR}/ascendc/dsa_kv_offloading_plan_nope")
if(NOT BUILD_OPS_RTY_KERNEL)
  add_modules_sources(OPTYPE dsa_kv_offloading_plan_nope ACLNNTYPE aclnn_exclude)
endif()
