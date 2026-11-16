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
#pragma once
#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"
namespace xllm_ops::dsa_kv_offloading {
aclnnStatus LaunchNopePlan(const aclTensor* descriptor,
                           const aclTensor* topk,
                           aclTensor* residentTokens,
                           aclTensor* residentSlots,
                           aclTensor* residentCount,
                           aclTensor* indices,
                           aclTensor* lengths,
                           aclTensor* table,
                           aclTensor* compactIndices,
                           aclOpExecutor* executor);
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
                           aclOpExecutor* executor);
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
                        aclOpExecutor* executor);
}  // namespace xllm_ops::dsa_kv_offloading
