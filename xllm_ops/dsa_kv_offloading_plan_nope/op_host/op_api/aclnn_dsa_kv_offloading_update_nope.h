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
#include <stdint.h>

#include "aclnn/aclnn_base.h"
#ifdef __cplusplus
extern "C" {
#endif
// All buffers are caller-owned contiguous ND device views. Host KV views
// must wrap an already registered/mapped Host allocation. No implicit copy.
// Resident/selection capacity is independent of query rows. descriptor[:,3]
// must be in [0,capacity), with distinct slots for all visible query rows.
// Zero-visible rows publish empty outputs without modifying resident state.
// indices retains positional slots (Serve decodes miss markers in place).
// compactIndices is a separate stable valid prefix for attention: duplicate
// tokens retain multiplicity and original top-k order; suffix is -1. Both
// outputs have [rows, 1, 2176] shape and must not alias any other argument.
__attribute__((visibility("default"))) aclnnStatus
aclnnDsaKvOffloadingUpdateNopeGetWorkspaceSize(aclTensor* hostLatent,
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
                                  aclOpExecutor** executor);
__attribute__((visibility("default"))) aclnnStatus
aclnnDsaKvOffloadingUpdateNope(void* workspace,
                  uint64_t workspaceSize,
                  aclOpExecutor* executor,
                  aclrtStream stream);
__attribute__((visibility("default"))) int64_t aclnnDsaKvOffloadingSelectionCapacity();
#ifdef __cplusplus
}
#endif
