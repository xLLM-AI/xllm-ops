/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_common_arch22.h
 * \brief LIV2公共定义：Layout枚举、类型模板LIV2Type、运行时信息RunInfo/ConstInfo/SplitCoreInfo、工具函数Align/Min/Max/CeilDiv
 */
#ifndef LIGHTNING_INDEXER_V2_COMMON_ARCH22_H
#define LIGHTNING_INDEXER_V2_COMMON_ARCH22_H

namespace LIV2Common {

// 与tiling的layout保持一致
enum class LI_V2_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

template <typename Q_T, typename K_T, typename OUT_T, typename QK_T, typename SCORE_T,
          const bool PAGE_ATTENTION = false, LI_V2_LAYOUT LAYOUT_T = LI_V2_LAYOUT::BSND,
          LI_V2_LAYOUT K_LAYOUT_T = LI_V2_LAYOUT::PA_BBND, bool DT_W_FLAG = false, typename... Args>
struct LIV2Type {
    static constexpr bool weightsTypeFlag = DT_W_FLAG;   // weight的dtype是否为FP32
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    using queryKeyType = QK_T;
    using scoreType = SCORE_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_V2_LAYOUT layout = LAYOUT_T;
    static constexpr LI_V2_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actS2SizeOrig = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // Cube分块大小(Mmad M维度对齐单位)
    static constexpr uint32_t BLOCK_CUBE = 16;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C1 = 0U;

    // 基本块大小
    uint32_t mBaseSize = 1U;
    uint32_t mBaseSizeAlign = 1U;
    uint32_t s1BaseSize = 1U;
    uint32_t s2BaseSize = 1U;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;             // topK选取大小
    uint64_t kSeqSize = 0ULL;         // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    LI_V2_LAYOUT outputLayout;           // 输出的格式
    bool attenMaskFlag = false;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    bool returnValue = false;
    int64_t cmpRatio = INT64_MAX;

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    uint32_t usedLenKDims = 0U;  // KV 的used_seq_len 的维度
    bool isAccumSeqS1 = false;    // 是否累加模式
    bool isAccumSeqS2 = false;    // 是否累加模式
    bool isSparseCountOver2K = false;     // sparseCount小于等于2048为false
    static constexpr uint32_t NEG_INF_FLOAT = 0xFF800000; // float负无穷
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;     // 当前核是否需要进行Decode归约任务
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
} // namespace LIV2Common

#endif // LIGHTNING_INDEXER_V2_COMMON_ARCH22_H