/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Ascend950 native CV entry: all workspace offsets are relative to the user region.
#ifndef CROSS_ATTN_ARCH35_FUSION_HELPER_H
#define CROSS_ATTN_ARCH35_FUSION_HELPER_H

#include "cross_attention_fusion_kernel.h"

template <class INPUT_T, class TABLE_T, class KVLEN_T>
__aicore__ inline void CallCrossAttentionArch35(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockTable,
    GM_ADDR actualKvLens, GM_ADDR attnOut, GM_ADDR workspace,
    const __gm__ Xllm::CrossAttentionTilingData* tilingDataPtr)
{
    static_assert(AscendC::IsSameType<TABLE_T, int32_t>::value, "block_table must be int32");
    static_assert(AscendC::IsSameType<KVLEN_T, int64_t>::value, "actual_kv_lens must be int64");
    Xllm::CrossAttentionAiv<INPUT_T> op;
    op.Init(tilingDataPtr, attnOut, actualKvLens, workspace);
    op.Process();
}

template <class INPUT_T, class TABLE_T, class KVLEN_T>
__aicore__ inline void CallCrossAttentionArch35Aic(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockTable,
    GM_ADDR actualKvLens, GM_ADDR workspace, const __gm__ Xllm::CrossAttentionTilingData* tilingDataPtr)
{
    static_assert(AscendC::IsSameType<TABLE_T, int32_t>::value, "block_table must be int32");
    static_assert(AscendC::IsSameType<KVLEN_T, int64_t>::value, "actual_kv_lens must be int64");
    Xllm::CrossAttentionAic<INPUT_T> op;
    op.Init(tilingDataPtr, query, key, value, blockTable, actualKvLens, workspace);
    op.Process();
}

#endif // CROSS_ATTN_ARCH35_FUSION_HELPER_H
