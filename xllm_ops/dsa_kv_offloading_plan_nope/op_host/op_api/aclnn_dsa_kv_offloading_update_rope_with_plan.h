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
#ifdef __cplusplus
extern "C" {
#endif
// Stable storage capacity is independent of query rows.
// descriptor[:,3] addresses the persistent selection/resident slot.
// All buffers are caller-owned contiguous ND device views. Host KV views
// must wrap an already registered/mapped Host allocation. No implicit copy.
// planTokens/planSlots are immutable for all same-step Serve-only consumers.
// Each layer owns separate selection payload; the plan does not share KV data.
__attribute__((visibility("default"))) aclnnStatus
aclnnDsaKvOffloadingUpdateRopeWithPlanGetWorkspaceSize(
    aclTensor* hostLatent,
    const aclTensor* blockTable,
    const aclTensor* descriptor,
    const aclTensor* currentMap,
    const aclTensor* topk,
    const aclTensor* appendSlots,
    const aclTensor* currentLatent,
    aclTensor* selection,
    aclTensor* indices,
    aclTensor* lengths,
    aclTensor* selectionTable,
    aclTensor* residentTokens,
    aclTensor* residentSlots,
    aclTensor* residentCount,
    aclTensor* hostRope,
    const aclTensor* currentRope,
    aclTensor* selectionRope,
    aclTensor* planTokens,
    aclTensor* planSlots,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
__attribute__((visibility("default"))) aclnnStatus
aclnnDsaKvOffloadingUpdateRopeWithPlan(void* workspace,
                                       uint64_t workspaceSize,
                                       aclOpExecutor* executor,
                                       aclrtStream stream);
#ifdef __cplusplus
}
#endif
