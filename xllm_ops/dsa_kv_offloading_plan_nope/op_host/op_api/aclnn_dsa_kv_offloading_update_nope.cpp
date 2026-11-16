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
#include "aclnn_dsa_kv_offloading_update_nope.h"

#include "dsa_kv_offloading_launch.h"
#include "dsa_kv_offloading_stable_validate.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_errno.h"
using namespace op;
using namespace xllm_ops::dsa_kv_offloading;
extern "C" aclnnStatus aclnnDsaKvOffloadingUpdateNopeGetWorkspaceSize(
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
    aclTensor* compactIndices,
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
        xllm_ops::dsa_kv_offloading::Tensor(compactIndices,
                                            op::DataType::DT_INT32) &&
        xllm_ops::dsa_kv_offloading::Shape(
            compactIndices, {topk->GetViewShape().GetDim(0), 1, 2176}) &&
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
                  compactIndices})))
    return ACLNN_ERR_PARAM_INVALID;
  L2_DFX_PHASE_1(aclnnDsaKvOffloadingUpdateNope,
                 DFX_IN(hostLatent,
                        blockTable,
                        descriptor,
                        currentMap,
                        topk,
                        appendSlots,
                        currentLatent,
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
                         compactIndices));
  auto uniqueExecutor = CREATE_EXECUTOR();
  if (uniqueExecutor.get() == nullptr) return ACLNN_ERR_INNER_CREATE_EXECUTOR;
  auto status = LaunchNopePlan(descriptor,
                               topk,
                               residentTokens,
                               residentSlots,
                               residentCount,
                               indices,
                               lengths,
                               selectionTable,
                               compactIndices,
                               uniqueExecutor.get());
  if (status != ACLNN_SUCCESS) return status;
  status = LaunchServe(hostLatent,
                       blockTable,
                       descriptor,
                       currentMap,
                       topk,
                       appendSlots,
                       currentLatent,
                       indices,
                       selection,
                       indices,
                       nullptr,
                       nullptr,
                       nullptr,
                       uniqueExecutor.get());
  if (status != ACLNN_SUCCESS) return status;
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnDsaKvOffloadingUpdateNope(void* workspace,
                                                      uint64_t workspaceSize,
                                                      aclOpExecutor* executor,
                                                      aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDsaKvOffloadingUpdateNope);
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

extern "C" int64_t aclnnDsaKvOffloadingSelectionCapacity() { return 2176; }
