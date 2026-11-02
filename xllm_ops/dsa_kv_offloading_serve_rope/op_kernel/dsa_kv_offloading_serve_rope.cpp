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
extern "C" __global__ __aicore__ void dsa_kv_offloading_serve_rope(GM_ADDR in_host_latent,
                                                    GM_ADDR in_block_table,
                                                    GM_ADDR in_descriptor,
                                                    GM_ADDR in_current_map,
                                                    GM_ADDR in_tokens,
                                                    GM_ADDR in_append_slots,
                                                    GM_ADDR in_current_latent,
                                                    GM_ADDR in_encoded_slots,
                                                    GM_ADDR in_selection,
                                                    GM_ADDR in_host_rope,
                                                    GM_ADDR in_current_rope,
                                                    GM_ADDR in_selection_rope,
                                                    GM_ADDR out_host_latent,
                                                    GM_ADDR out_selection,
                                                    GM_ADDR out_host_rope,
                                                    GM_ADDR out_selection_rope,
                                                    GM_ADDR workspace,
                                                    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (g_coreType == AIC) return;
  GET_TILING_DATA(data, tiling);
  AscendC::TPipe pipe;
  SparseKvServe<bfloat16_t, 64, true> op(&pipe,
                                         data.workerCount,
                                         data.rows,
                                         data.heads,
                                         data.topk,
                                         data.mapStride,
                                         data.tableStride);
  op.init(out_host_latent,
          in_block_table,
          in_descriptor,
          in_current_map,
          in_tokens,
          in_append_slots,
          in_current_latent,
          out_selection,
          in_encoded_slots,
          out_host_rope,
          in_current_rope,
          out_selection_rope);
  op.process();
}
