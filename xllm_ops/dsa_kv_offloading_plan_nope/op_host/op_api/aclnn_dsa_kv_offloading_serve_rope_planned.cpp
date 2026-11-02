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
#include "aclnn_dsa_kv_offloading_serve_rope_planned.h"

#include "dsa_kv_offloading_launch.h"
#include "dsa_kv_offloading_stable_validate.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
using namespace op;
using namespace xllm_ops::dsa_kv_offloading;
extern "C" aclnnStatus aclnnDsaKvOffloadingServeRopePlannedGetWorkspaceSize(
    aclTensor* hostLatent,
    aclTensor* hostRope,
    const aclTensor* blockTable,
    const aclTensor* descriptor,
    const aclTensor* currentMap,
    const aclTensor* planTokens,
    const aclTensor* planSlots,
    const aclTensor* appendSlots,
    const aclTensor* currentLatent,
    const aclTensor* currentRope,
    aclTensor* selection,
    aclTensor* selectionRope,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
  if (workspaceSize == nullptr || executor == nullptr)
    return ACLNN_ERR_PARAM_NULLPTR;
  *workspaceSize = 0;
  *executor = nullptr;
  if (!(StableCommon(hostLatent,
                     blockTable,
                     descriptor,
                     currentMap,
                     planTokens,
                     appendSlots,
                     currentLatent,
                     planSlots,
                     selection) &&
        StableRope(hostRope,
                   currentRope,
                   selectionRope,
                   hostLatent,
                   planTokens,
                   selection) &&
        Distinct({hostLatent,
                  hostRope,
                  blockTable,
                  descriptor,
                  currentMap,
                  planTokens,
                  planSlots,
                  appendSlots,
                  currentLatent,
                  currentRope,
                  selection,
                  selectionRope})))
    return ACLNN_ERR_PARAM_INVALID;
  L2_DFX_PHASE_1(aclnnDsaKvOffloadingServeRopePlanned,
                 DFX_IN(hostLatent,
                        hostRope,
                        blockTable,
                        descriptor,
                        currentMap,
                        planTokens,
                        planSlots,
                        appendSlots,
                        currentLatent,
                        currentRope,
                        selection,
                        selectionRope),
                 DFX_OUT());
  auto uniqueExecutor = CREATE_EXECUTOR();
  if (uniqueExecutor.get() == nullptr) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
  auto status = LaunchServe(hostLatent,
                            blockTable,
                            descriptor,
                            currentMap,
                            planTokens,
                            appendSlots,
                            currentLatent,
                            planSlots,
                            selection,
                            nullptr,
                            hostRope,
                            currentRope,
                            selectionRope,
                            uniqueExecutor.get());
  if (status != ACLNN_SUCCESS) return status;
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnDsaKvOffloadingServeRopePlanned(void* workspace,
                                               uint64_t workspaceSize,
                                               aclOpExecutor* executor,
                                               aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDsaKvOffloadingServeRopePlanned);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
