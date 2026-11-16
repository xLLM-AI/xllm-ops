/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "dsa_kv_offloading_launch.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
#include "opdev/op_log.h"
using namespace op;
namespace xllm_ops::dsa_kv_offloading {
OP_TYPE_REGISTER(DsaKvOffloadingPlanNope);
OP_TYPE_REGISTER(DsaKvOffloadingPlanRope);
OP_TYPE_REGISTER(DsaKvOffloadingServeNope);
OP_TYPE_REGISTER(DsaKvOffloadingServeRope);
aclnnStatus LaunchNopePlan(const aclTensor* descriptor,
                           const aclTensor* topk,
                           aclTensor* residentTokens,
                           aclTensor* residentSlots,
                           aclTensor* residentCount,
                           aclTensor* indices,
                           aclTensor* lengths,
                           aclTensor* table,
                           aclTensor* compactIndices,
                           aclOpExecutor* executor) {
  return ADD_TO_LAUNCHER_LIST_AICORE(
      DsaKvOffloadingPlanNope,
      OP_INPUT(descriptor, topk, residentTokens, residentSlots, residentCount),
      OP_OUTPUT(residentTokens,
                residentSlots,
                residentCount,
                indices,
                lengths,
                table,
                compactIndices),
      OP_ATTR());
}
aclnnStatus LaunchRopePlan(const aclTensor* descriptor,
                           const aclTensor* topk,
                           aclTensor* residentTokens,
                           aclTensor* residentSlots,
                           aclTensor* residentCount,
                           aclTensor* indices,
                           aclTensor* lengths,
                           aclTensor* table,
                           aclTensor* planTokens,
                           aclTensor* planSlots,
                           aclOpExecutor* executor) {
  return ADD_TO_LAUNCHER_LIST_AICORE(
      DsaKvOffloadingPlanRope,
      OP_INPUT(descriptor, topk, residentTokens, residentSlots, residentCount),
      OP_OUTPUT(residentTokens,
                residentSlots,
                residentCount,
                indices,
                lengths,
                table,
                planTokens,
                planSlots),
      OP_ATTR());
}
aclnnStatus LaunchServe(aclTensor* hostLatent,
                        const aclTensor* blockTable,
                        const aclTensor* descriptor,
                        const aclTensor* currentMap,
                        const aclTensor* tokens,
                        const aclTensor* appendSlots,
                        const aclTensor* currentLatent,
                        const aclTensor* encodedSlots,
                        aclTensor* selection,
                        aclTensor* mutableIndices,
                        aclTensor* hostRope,
                        const aclTensor* currentRope,
                        aclTensor* selectionRope,
                        aclOpExecutor* executor) {
  if (hostRope == nullptr) {
    return ADD_TO_LAUNCHER_LIST_AICORE(
        DsaKvOffloadingServeNope,
        OP_INPUT(hostLatent,
                 blockTable,
                 descriptor,
                 currentMap,
                 tokens,
                 appendSlots,
                 currentLatent,
                 encodedSlots,
                 selection),
        OP_OUTPUT(hostLatent, selection, mutableIndices),
        OP_ATTR());
  }
  return ADD_TO_LAUNCHER_LIST_AICORE(
      DsaKvOffloadingServeRope,
      OP_INPUT(hostLatent,
               blockTable,
               descriptor,
               currentMap,
               tokens,
               appendSlots,
               currentLatent,
               encodedSlots,
               selection,
               hostRope,
               currentRope,
               selectionRope),
      OP_OUTPUT(hostLatent, selection, hostRope, selectionRope),
      OP_ATTR());
}
}  // namespace xllm_ops::dsa_kv_offloading
