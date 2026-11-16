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
#include "aclnn_dsa_kv_offloading_update_rope_with_plan.h"

#include "dsa_kv_offloading_launch.h"
#include "dsa_kv_offloading_stable_validate.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
using namespace op;
using namespace xllm_ops::dsa_kv_offloading;
extern "C" aclnnStatus aclnnDsaKvOffloadingUpdateRopeWithPlanGetWorkspaceSize(
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
    aclOpExecutor** executor) {
  if (workspaceSize == nullptr || executor == nullptr)
    return ACLNN_ERR_PARAM_NULLPTR;
  *workspaceSize = 0;
  *executor = nullptr;
  if (!(StableCommon(hostLatent,
                     blockTable,
                     descriptor,
                     currentMap,
                     topk,
                     appendSlots,
                     currentLatent,
                     indices,
                     selection) &&
        StableState(topk,
                    selection,
                    residentTokens,
                    residentSlots,
                    residentCount,
                    indices,
                    lengths,
                    selectionTable) &&
        StableRope(hostRope,
                   currentRope,
                   selectionRope,
                   hostLatent,
                   topk,
                   selection) &&
        xllm_ops::dsa_kv_offloading::Tensor(planTokens,
                                            op::DataType::DT_INT32) &&
        xllm_ops::dsa_kv_offloading::Tensor(planSlots,
                                            op::DataType::DT_INT32) &&
        xllm_ops::dsa_kv_offloading::Shape(
            planTokens, {topk->GetViewShape().GetDim(0), 1, 2176}) &&
        xllm_ops::dsa_kv_offloading::Shape(
            planSlots, {topk->GetViewShape().GetDim(0), 1, 2176}) &&
        Distinct({hostLatent,
                  blockTable,
                  descriptor,
                  currentMap,
                  topk,
                  appendSlots,
                  currentLatent,
                  selection,
                  indices,
                  lengths,
                  selectionTable,
                  residentTokens,
                  residentSlots,
                  residentCount,
                  hostRope,
                  currentRope,
                  selectionRope,
                  planTokens,
                  planSlots})))
    return ACLNN_ERR_PARAM_INVALID;
  L2_DFX_PHASE_1(aclnnDsaKvOffloadingUpdateRopeWithPlan,
                 DFX_IN(hostLatent,
                        blockTable,
                        descriptor,
                        currentMap,
                        topk,
                        appendSlots,
                        currentLatent,
                        hostRope,
                        currentRope,
                        residentTokens,
                        residentSlots,
                        residentCount),
                 DFX_OUT(hostLatent,
                         selection,
                         indices,
                         lengths,
                         selectionTable,
                         residentTokens,
                         residentSlots,
                         residentCount,
                         hostRope,
                         selectionRope,
                         planTokens,
                         planSlots));
  auto uniqueExecutor = CREATE_EXECUTOR();
  if (uniqueExecutor.get() == nullptr) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
  auto status = LaunchRopePlan(descriptor,
                               topk,
                               residentTokens,
                               residentSlots,
                               residentCount,
                               indices,
                               lengths,
                               selectionTable,
                               planTokens,
                               planSlots,
                               uniqueExecutor.get());
  if (status != ACLNN_SUCCESS) return status;
  status = LaunchServe(hostLatent,
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
extern "C" aclnnStatus aclnnDsaKvOffloadingUpdateRopeWithPlan(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDsaKvOffloadingUpdateRopeWithPlan);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
