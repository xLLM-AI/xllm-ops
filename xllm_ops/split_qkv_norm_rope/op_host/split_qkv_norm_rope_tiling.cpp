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
 * \file split_qkv_norm_rope_tiling.cpp
 * \brief SplitQkvNormRope host 侧 tiling。
 */

#include "split_qkv_norm_rope_tiling.h"

#include <cstdio>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr size_t IDX_QKV = 0;
constexpr size_t IDX_COS = 3;
constexpr size_t ATTR_NUM_Q_HEADS = 0;
constexpr size_t ATTR_NUM_KV_HEADS = 1;
constexpr size_t ATTR_EPSILON = 2;
constexpr size_t ATTR_GQA_EXPAND = 3;
constexpr size_t ATTR_LAYOUT_BNSD = 4;
constexpr size_t ATTR_KV_INTERLEAVED = 5;

inline uint32_t RoundUp(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

// UB 需求（与 kernel 侧 InitBuffer 一致）
uint64_t CalcUbNeed(uint32_t tile, uint32_t padStride, uint32_t ropeStride, uint32_t headTotal, uint32_t elemBytes)
{
    uint64_t need = static_cast<uint64_t>(tile) * 2 *
                    (static_cast<uint64_t>(padStride) * elemBytes +
                     static_cast<uint64_t>(ropeStride) * elemBytes * 2);   // inQkv/inCos/inSin ×2
    need += 2ull * tile * headTotal * SQNR_HEAD_DIM * elemBytes;            // out ×2
    need += 2ull * SQNR_HEAD_DIM * elemBytes + 2ull * SQNR_HEAD_DIM * 4;    // inGamma ×2 + gamma fp32 ×2
    need += static_cast<uint64_t>(tile) * SQNR_MAX_HEADS * 4;               // sumsq
    return need;
}
}  // namespace

static ge::graphStatus TilingPrepare4SplitQkvNormRope(gert::TilingParseContext *context)
{
    auto compileInfo = context->GetCompiledInfo<SplitQkvNormRopeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4SplitQkvNormRope(gert::TilingContext *context)
{
    const gert::Shape qkvShape = context->GetInputShape(IDX_QKV)->GetStorageShape();
    const gert::CompileTimeTensorDesc *qkvDesc = context->GetInputDesc(IDX_QKV);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvDesc);
    OP_CHECK_IF(qkvShape.GetDimNum() != 3, OP_LOGE(context, "qkv must be [B,S,stride]."), return ge::GRAPH_FAILED);

    const gert::Shape cosShape = context->GetInputShape(IDX_COS)->GetStorageShape();
    OP_CHECK_IF(cosShape.GetDimNum() != 3, OP_LOGE(context, "cos must be [B,S,ropeStride]."),
                return ge::GRAPH_FAILED);

    const uint32_t batch = static_cast<uint32_t>(qkvShape.GetDim(0));
    const uint32_t seqLen = static_cast<uint32_t>(qkvShape.GetDim(1));
    const uint32_t qkvStride = static_cast<uint32_t>(qkvShape.GetDim(2));
    const uint32_t ropeStride = static_cast<uint32_t>(cosShape.GetDim(2));

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *nqPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_Q_HEADS);
    const int64_t *nkPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS);
    OP_CHECK_NULL_WITH_CONTEXT(context, nqPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, nkPtr);
    const uint32_t numHeadQ = static_cast<uint32_t>(*nqPtr);
    const uint32_t numHeadK = static_cast<uint32_t>(*nkPtr);
    const float *epsPtr = attrs->GetAttrPointer<float>(ATTR_EPSILON);
    const bool *gqaPtr = attrs->GetAttrPointer<bool>(ATTR_GQA_EXPAND);
    const bool *bnsdPtr = attrs->GetAttrPointer<bool>(ATTR_LAYOUT_BNSD);
    const bool *kvIntPtr = attrs->GetAttrPointer<bool>(ATTR_KV_INTERLEAVED);
    const float eps = (epsPtr != nullptr) ? *epsPtr : 1e-6f;
    const bool gqaExpand = (gqaPtr != nullptr) ? *gqaPtr : false;
    const bool layoutBNSD = (bnsdPtr != nullptr) ? *bnsdPtr : false;
    const bool kvInterleaved = (kvIntPtr != nullptr) ? *kvIntPtr : false;

    const uint32_t numHeadV = numHeadK;  // v1：Nv == Nk
    const uint32_t headTotal = numHeadQ + numHeadK + numHeadV;

    OP_CHECK_IF(numHeadQ == 0 || numHeadK == 0, OP_LOGE(context, "head num must be positive."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeadQ % numHeadK != 0, OP_LOGE(context, "num_q_heads %% num_kv_heads != 0."),
                return ge::GRAPH_FAILED);
    const uint32_t qkHeads = numHeadQ + numHeadK;
    OP_CHECK_IF(qkHeads > SQNR_MAX_HEADS, OP_LOGE(context, "Nq+Nk=%u > %u.", qkHeads, SQNR_MAX_HEADS),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qkvStride < headTotal * SQNR_HEAD_DIM,
                OP_LOGE(context, "qkv stride %u < heads*D = %u.", qkvStride, headTotal * SQNR_HEAD_DIM),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ropeStride < SQNR_HEAD_DIM / 2 || ropeStride > SQNR_HEAD_DIM,
                OP_LOGE(context, "rope stride must be in [64,128], got %u.", ropeStride), return ge::GRAPH_FAILED);

    SplitQkvNormRopeCompileInfo compileInfo;
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    (void)compileInfo;

    const uint32_t totalTokens = batch * seqLen;
    const ge::DataType dtype = qkvDesc->GetDataType();
    // kernel 用 sizeof(T) 计算 UB 布局，host 必须与之对齐：当前仅支持 2 字节 dtype
    uint32_t elemBytes = 0;
    if (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16) {
        elemBytes = 2;
    } else {
        OP_LOGE(context, "unsupported dtype %d, expect BF16/FLOAT16.", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }
    const uint32_t padStride = RoundUp(qkvStride, 16);  // 32B 对齐的 UB 行距

    // 按 UB 预算从大到小挑 tile 行数：行数越大越能摊薄每 tile 的固定开销
    uint32_t tokenTile = 0;
    for (uint32_t cand = SQNR_TOKEN_TILE; cand >= 1; cand >>= 1) {
        if (CalcUbNeed(cand, padStride, ropeStride, headTotal, elemBytes) <= ubSize) {
            tokenTile = cand;
            break;
        }
    }
    OP_CHECK_IF(tokenTile == 0,
                OP_LOGE(context, "UB insufficient even at tile=1: stride=%u pad=%u heads=%u.",
                        qkvStride, padStride, headTotal),
                return ge::GRAPH_FAILED);

    uint32_t blockDim = (aivNum < (totalTokens + tokenTile - 1) / tokenTile)
                            ? aivNum
                            : (totalTokens + tokenTile - 1) / tokenTile;
    if (blockDim == 0) {
        blockDim = 1;
    }
    uint32_t tokenPerCore = RoundUp((totalTokens + blockDim - 1) / blockDim, tokenTile);
    blockDim = (totalTokens + tokenPerCore - 1) / tokenPerCore;
    if (blockDim == 0) {
        blockDim = 1;
    }

    SplitQkvNormRopeTilingData tilingData;
    tilingData.set_batch(batch);
    tilingData.set_seqLen(seqLen);
    tilingData.set_numHeadQ(numHeadQ);
    tilingData.set_numHeadK(numHeadK);
    tilingData.set_numHeadV(numHeadV);
    tilingData.set_headDim(SQNR_HEAD_DIM);
    tilingData.set_qkvStride(qkvStride);
    tilingData.set_padStride(padStride);
    tilingData.set_ropeStride(ropeStride);
    tilingData.set_totalTokens(totalTokens);
    tilingData.set_tokenPerCore(tokenPerCore);
    tilingData.set_tokenTile(tokenTile);
    tilingData.set_lastBlockTokenNum(totalTokens - tokenPerCore * (blockDim - 1));
    tilingData.set_eps(eps);
    tilingData.set_reciprocal(1.0f / static_cast<float>(SQNR_HEAD_DIM));
    uint32_t flags = 0;
    if (gqaExpand) {
        flags |= SQNR_FLAG_GQA_EXPAND;
    }
    if (layoutBNSD) {
        flags |= SQNR_FLAG_LAYOUT_BNSD;
    }
    if (kvInterleaved) {
        flags |= SQNR_FLAG_KV_INTERLEAVED;
    }
    tilingData.set_flags(flags);
    tilingData.set_blockDim(blockDim);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(SQNR_KEY_SINGLE);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t *workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = 0;

    OP_LOGI(context, "tokens=%u blockDim=%u tokenPerCore=%u tile=%u lastCore=%u stride=%u pad=%u rope=%u heads=%u/%u gqa=%d bnsd=%d",
            totalTokens, blockDim, tokenPerCore, tokenTile, tilingData.get_lastBlockTokenNum(), qkvStride, padStride,
            ropeStride, numHeadQ, numHeadK, static_cast<int32_t>(gqaExpand), static_cast<int32_t>(layoutBNSD));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SplitQkvNormRope)
    .Tiling(Tiling4SplitQkvNormRope)
    .TilingParse<SplitQkvNormRopeCompileInfo>(TilingPrepare4SplitQkvNormRope);

}  // namespace optiling
