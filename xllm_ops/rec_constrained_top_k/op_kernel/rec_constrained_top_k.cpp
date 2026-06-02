/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "rec_constrained_top_k.h"

using namespace AscendC;

#define INVOKE_REC_CONSTRAINED_TOPK_IMPL(LogitType)                         \
  do {                                                                       \
    kernels::RecConstrainedTopKKernel<LogitType> op;                         \
    op.Init(logits,                                                           \
            sequence_group,                                                   \
            first_token_ids,                                                  \
            prefix1_offsets,                                                  \
            prefix1_values,                                                   \
            prefix1_pair_keys,                                                \
            prefix2_value_offsets,                                            \
            prefix2_values,                                                   \
            temperatures,                                                     \
            out_tokens,                                                       \
            out_logprobs,                                                     \
            &tiling_data);                                                    \
    op.Process();                                                             \
  } while (0)

extern "C" __global__ __aicore__ void rec_constrained_top_k(
    GM_ADDR logits,
    GM_ADDR sequence_group,
    GM_ADDR first_token_ids,
    GM_ADDR prefix1_offsets,
    GM_ADDR prefix1_values,
    GM_ADDR prefix1_pair_keys,
    GM_ADDR prefix2_value_offsets,
    GM_ADDR prefix2_values,
    GM_ADDR temperatures,
    GM_ADDR out_tokens,
    GM_ADDR out_logprobs,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  (void)workspace;
  GET_TILING_DATA(tiling_data, tiling);
#if (ORIG_DTYPE_LOGITS == DT_FLOAT)
  INVOKE_REC_CONSTRAINED_TOPK_IMPL(float);
  return;
#endif
#if (ORIG_DTYPE_LOGITS == DT_FLOAT16)
  INVOKE_REC_CONSTRAINED_TOPK_IMPL(half);
  return;
#endif
#if (ORIG_DTYPE_LOGITS == DT_BF16)
  INVOKE_REC_CONSTRAINED_TOPK_IMPL(bfloat16_t);
  return;
#endif
}
