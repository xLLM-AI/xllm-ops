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

/*!
 * \file split_qkv_norm_rope_tiling.h
 * \brief SplitQkvNormRope tiling 数据结构与常量（host / kernel 共享字段顺序）。
 */

#ifndef SPLIT_QKV_NORM_ROPE_TILING_H_
#define SPLIT_QKV_NORM_ROPE_TILING_H_

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t SQNR_HEAD_DIM = 128;
constexpr uint32_t SQNR_TOKEN_TILE = 8;
constexpr uint32_t SQNR_MAX_HEADS = 16;
constexpr uint32_t SQNR_FLAG_GQA_EXPAND = 1u << 0;
constexpr uint32_t SQNR_FLAG_LAYOUT_BNSD = 1u << 1;

// 入口 tilingKey 固定 0：kernel 只有一个入口（无 TILING_KEY_IS 分支），
// dtype 由框架按 bin 展开（kernel 侧用 DTYPE_QKV），因此 SetTilingKey(0)。
constexpr uint32_t SQNR_KEY_SINGLE = 0;

BEGIN_TILING_DATA_DEF(SplitQkvNormRopeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, seqLen);
TILING_DATA_FIELD_DEF(uint32_t, numHeadQ);
TILING_DATA_FIELD_DEF(uint32_t, numHeadK);
TILING_DATA_FIELD_DEF(uint32_t, numHeadV);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, qkvStride);
TILING_DATA_FIELD_DEF(uint32_t, padStride);
TILING_DATA_FIELD_DEF(uint32_t, ropeStride);
TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
TILING_DATA_FIELD_DEF(uint32_t, tokenPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tokenTile);
TILING_DATA_FIELD_DEF(uint32_t, lastBlockTokenNum);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, reciprocal);
TILING_DATA_FIELD_DEF(uint32_t, flags);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SplitQkvNormRope, SplitQkvNormRopeTilingData)

struct SplitQkvNormRopeCompileInfo {
    uint32_t totalCoreNum = 0;
    uint64_t totalUbSize = 0;
};
}  // namespace optiling

#endif  // SPLIT_QKV_NORM_ROPE_TILING_H_
