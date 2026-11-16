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
#include "dsa_kv_offloading_plan_serve.h"
using namespace xllm_ops::dsa_kv_offloading;
extern "C" __global__ __aicore__ void dsa_kv_offloading_plan_rope(
    GM_ADDR in_descriptor,
    GM_ADDR in_topk,
    GM_ADDR in_resident_tokens,
    GM_ADDR in_resident_slots,
    GM_ADDR in_resident_count,
    GM_ADDR out_resident_tokens,
    GM_ADDR out_resident_slots,
    GM_ADDR out_resident_count,
    GM_ADDR out_indices,
    GM_ADDR out_lengths,
    GM_ADDR out_table,
    GM_ADDR out_plan_tokens,
    GM_ADDR out_plan_slots,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (g_coreType == AIC) return;
  GET_TILING_DATA(data, tiling);
  AscendC::TPipe pipe;
  SparseKvPlan<true, false, true> op(
      &pipe, data.workerCount, data.rows, data.heads, data.topk);
  op.init(in_descriptor,
          in_topk,
          out_indices,
          out_lengths,
          out_table,
          out_resident_tokens,
          out_resident_slots,
          out_resident_count,
          out_plan_tokens,
          out_plan_slots);
  op.process();
}
