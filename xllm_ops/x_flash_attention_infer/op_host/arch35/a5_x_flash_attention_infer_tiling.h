/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLLM_OPS_XFAI_ARCH35_HOST_A5_X_FLASH_ATTENTION_INFER_TILING_H
#define XLLM_OPS_XFAI_ARCH35_HOST_A5_X_FLASH_ATTENTION_INFER_TILING_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../../op_kernel/arch35/a5_x_flash_attention_infer_tiling_data_def.h"

namespace FAInferTiling {
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr int32_t SPARSE_MODE_NO_MASK = 0;
constexpr int32_t SPARSE_MODE_LEFT_UP = 1;
constexpr int32_t SPARSE_MODE_RIGHT_DOWN = 2;

constexpr int32_t BLOCK_BASE_SIZE = 128;
constexpr uint32_t CV_RATIO = 2;
const int32_t WORKSPACE_BLOCK_SIZE_DB = 131072;

struct FAInfo {
    int64_t batchSize = 0;
    int64_t numOfHeads = 0;
    int64_t numOfKVHeads = 0;
    int64_t seqSize = 0;
    int64_t seqInnerSize = 0;
    int64_t headSize = 0;

    uint32_t numBlocks = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    uint32_t maskType = SPARSE_MODE_NO_MASK;
    float scaleValue = 1.0;
    int64_t* actualSeqLengths{nullptr};
    int64_t* actualSeqLengthsKV{nullptr};
};

template <typename T>
auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

template <typename T>
auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

inline void GetPreNextTokensLeftUp(
    FATilingData& tilingData, int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp,
    int64_t& nextTokensLeftUp)
{
    auto& baseParams = tilingData.inputParamsRegbase;
    int64_t preTokens = SPARSE_MODE_INT_MAX;
    int64_t nextTokens = SPARSE_MODE_INT_MAX;
    if (baseParams.attenMaskCompressMode == SPARSE_MODE_LEFT_UP) {
        preTokens = SPARSE_MODE_INT_MAX;
        nextTokens = 0;
    }
    if (baseParams.attenMaskCompressMode == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = preTokens;
        nextTokensLeftUp = nextTokens;
    }
}

inline void FixParamWithRowInvalid(
    int64_t& actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp)
{
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                                 (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) :
                                 0;
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;
    actualSeqLength -= preTokensError;
}

inline int64_t GetCutBlockNums(
    int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter, int64_t token)
{
    if (sInner == 0 || sOuter == 0) {
        return 0;
    }
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    int64_t tmpInnerCutBlockNums =
        (innerCutBlockNums > 0) ?
            (innerCutBlockNums % tolerance + innerCutBlockNums) * (innerCutBlockNums / tolerance + 1) / 2 :
            0;
    blockNums += tmpInnerCutBlockNums;
    int64_t tmpInnerCutBlockLeftNums =
        (innerCutBlockLeftNums > 0) ?
            (innerCutBlockLeftNums % tolerance + innerCutBlockLeftNums) * (innerCutBlockLeftNums / tolerance + 1) / 2 :
            0;
    blockNums -= tmpInnerCutBlockLeftNums;
    int64_t tmpInnerCutBlockDownNums =
        (innerCutBlockDownNums > 0) ?
            (innerCutBlockDownNums % tolerance + innerCutBlockDownNums) * (innerCutBlockDownNums / tolerance + 1) / 2 :
            0;
    blockNums -= tmpInnerCutBlockDownNums;
    return blockNums;
}

inline int64_t GetCalcBlockNumsOneHead(
    int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t sOuterSize, int64_t sInnerSize, int64_t preTokensLeftUp,
    int64_t nextTokensLeftUp, bool isAttenMaskUsed)
{
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums =
            (actualSeqLengthKV + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize);
        int64_t outerBlockNums =
            (actualSeqLength + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        toCalcBlockNums -= GetCutBlockNums(
            blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize), static_cast<int64_t>(sOuterSize),
            nextTokensLeftUp);
        toCalcBlockNums -= GetCutBlockNums(
            blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize), static_cast<int64_t>(sOuterSize),
            blockSeqLengthKV - blockSeqLength + preTokensLeftUp);
        return toCalcBlockNums;
    }
}

inline int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums)
{
    int64_t sInnerBlockNums = 0;
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        int64_t tmpSInnerBlockNums = sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0;
        sInnerBlockNums = (sInnerIndexStart < 0) ? innerBlockNums : tmpSInnerBlockNums;
    }
    return sInnerBlockNums;
}

// 对Batch/headNum/qSeqLen三根轴切多核策略,采用贪心切分,使得每个AI Core上的计算量尽可能均衡.
inline void ComputeSplitNBSeq(
    FATilingData& tilingData, uint32_t batchSize, const size_t tilingElementArrayLen,
    std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV, int64_t sOuterSize,
    int64_t sInnerSize, double coreWightTarget, uint32_t& curCore)
{
    auto& baseParams = tilingData.inputParamsRegbase;
    std::vector<uint32_t> bnAxisStartIdx(tilingElementArrayLen, 0U);
    std::vector<int64_t> qSeqAxisStartIdx(tilingElementArrayLen, 0L);
    int64_t curWeight = 0;
    uint32_t lastHeadIdx = 0; // actual seq为0时不分配核
    uint32_t lastBatchIdx = 0;
    uint32_t lastQSeqOuterIdx = 0;
    for (uint32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        for (uint32_t headNum = 0; headNum < baseParams.qHeads; headNum++) {
            int64_t preTokensLeftUp = 0;
            int64_t nextTokensLeftUp = 0;
            GetPreNextTokensLeftUp(
                tilingData, actualSeqLengths[batchIdx], actualSeqLengthsKV[batchIdx], preTokensLeftUp,
                nextTokensLeftUp);
            FixParamWithRowInvalid(
                actualSeqLengths[batchIdx], actualSeqLengthsKV[batchIdx], preTokensLeftUp, nextTokensLeftUp);
            int64_t outerBlockNums = (actualSeqLengths[batchIdx] + sOuterSize - 1) / sOuterSize;
            int64_t innerBlockNums = (actualSeqLengthsKV[batchIdx] + sInnerSize - 1) / sInnerSize;
            for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
                int64_t diff = static_cast<int64_t>(coreWightTarget * double(curCore + 1)) - curWeight;
                int64_t sInnerIndexStart =
                    -(preTokensLeftUp > 0 ? (preTokensLeftUp + sInnerSize - 1) / sInnerSize :
                                            preTokensLeftUp / sInnerSize);
                int64_t sInnerIndexEnd = nextTokensLeftUp > 0 ? (nextTokensLeftUp + sInnerSize - 1) / sInnerSize :
                                                                nextTokensLeftUp / sInnerSize;
                int64_t sInnerBlockNums = GetSInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
                if (sInnerBlockNums - diff > diff &&
                    !(lastHeadIdx == 0 && lastBatchIdx == 0 && lastQSeqOuterIdx == 0)) {
                    curCore += 1;
                    bnAxisStartIdx[curCore] = batchIdx * baseParams.qHeads + headNum;
                    qSeqAxisStartIdx[curCore] = sOuterIndex;
                }
                lastHeadIdx = headNum + 1;
                lastBatchIdx = batchIdx + 1;
                lastQSeqOuterIdx = sOuterIndex + 1;
                curWeight += sInnerBlockNums;
                preTokensLeftUp -= sOuterSize;
                nextTokensLeftUp += sOuterSize;
            }
        }
    }
    bnAxisStartIdx[curCore + 1] = batchSize * baseParams.qHeads;
    qSeqAxisStartIdx[curCore + 1] = static_cast<int64_t>(lastQSeqOuterIdx);

    std::copy(
        std::begin(bnAxisStartIdx), std::end(bnAxisStartIdx),
        std::begin(tilingData.multiCoreParamsRegbase.bnAxisStartIdx));
    std::copy(
        std::begin(qSeqAxisStartIdx), std::end(qSeqAxisStartIdx),
        std::begin(tilingData.multiCoreParamsRegbase.sparseStartIdx));
}

inline void FillInputParams(const FAInfo& faInfo, FATilingData& tilingData)
{
    auto& inputParams = tilingData.inputParamsRegbase;
    inputParams.batch = faInfo.batchSize;
    inputParams.qHeads = faInfo.numOfHeads;
    inputParams.kvHeads = faInfo.numOfKVHeads;
    inputParams.groupSize = faInfo.numOfHeads / faInfo.numOfKVHeads;
    inputParams.qSeqlen = faInfo.seqSize;
    inputParams.kvSeqlen = faInfo.seqInnerSize;
    inputParams.embed = faInfo.headSize;
    inputParams.scaleValue = faInfo.scaleValue;

    inputParams.attenMaskCompressMode = faInfo.maskType;
    inputParams.headNumRatio = static_cast<uint32_t>(faInfo.numOfHeads / faInfo.numOfKVHeads);
    inputParams.blockSize = faInfo.blockSize;
    inputParams.blockTableDim2 = faInfo.maxBlockNumPerBatch;
    inputParams.paBlockNumSum = faInfo.numBlocks;
    inputParams.attenMaskQSeqlen = static_cast<uint32_t>(faInfo.seqSize);
    inputParams.attenMaskKvSeqlen = static_cast<uint32_t>(faInfo.seqInnerSize);
}

inline void FillActualSeqLengths(
    const FAInfo& faInfo, FATilingData& tilingData, std::vector<int64_t>& actualSeqLengths,
    std::vector<int64_t>& actualSeqLengthsKV)
{
    auto& inputParams = tilingData.inputParamsRegbase;
    int64_t batchSize = inputParams.batch;
    bool isActualSeqLengthsNull = (faInfo.actualSeqLengths == nullptr) ? true : false;
    bool isActualSeqLengthsKVNull = (faInfo.actualSeqLengthsKV == nullptr) ? true : false;
    auto actualSeqLengthsSize = (faInfo.actualSeqLengths == nullptr) ? batchSize : 0;
    auto actualSeqLengthsKVSize = (faInfo.actualSeqLengthsKV == nullptr) ? batchSize : 0;
    inputParams.isActualSeqLengthsNull = isActualSeqLengthsNull;
    inputParams.isActualSeqLengthsKVNull = isActualSeqLengthsKVNull;
    inputParams.actualSeqLengthsSize = static_cast<uint32_t>(actualSeqLengthsSize);
    inputParams.actualSeqLengthsKVSize = static_cast<uint32_t>(actualSeqLengthsKVSize);
    for (int64_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        if (isActualSeqLengthsNull) {
            actualSeqLengths[batchIdx] = inputParams.qSeqlen;
        } else {
            actualSeqLengths[batchIdx] = faInfo.actualSeqLengths[batchIdx];
        }
        if (isActualSeqLengthsKVNull) {
            actualSeqLengthsKV[batchIdx] = inputParams.kvSeqlen;
        } else {
            actualSeqLengthsKV[batchIdx] = faInfo.actualSeqLengthsKV[batchIdx];
        }
    }
}

inline int32_t GetFATilingParam(const FAInfo& faInfo, uint32_t blockDim, FATilingData& faTilingData)
{
    FillInputParams(faInfo, faTilingData);
    auto& inputParams = faTilingData.inputParamsRegbase;
    int64_t batchSize = inputParams.batch;
    std::vector<int64_t> actualSeqLengths(batchSize);
    std::vector<int64_t> actualSeqLengthsKV(batchSize);
    FillActualSeqLengths(faInfo, faTilingData, actualSeqLengths, actualSeqLengthsKV);

    bool isAttenMaskUsed = faInfo.maskType != SPARSE_MODE_NO_MASK;
    int64_t totalBlockNumsOneHead = 0;
    constexpr static auto sInnerSize = BLOCK_BASE_SIZE;
    constexpr static auto sOuterSize = BLOCK_BASE_SIZE;
    for (int64_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[batchIdx];
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
  GetPreNextTokensLeftUp(
            faTilingData, actualSeqLengths[batchIdx], actualSeqLengthsKV[batchIdx], preTokensLeftUp, nextTokensLeftUp);
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[batchIdx], preTokensLeftUp, nextTokensLeftUp);
        totalBlockNumsOneHead += GetCalcBlockNumsOneHead(
            actualSeqLengthsTmp, actualSeqLengthsKV[batchIdx], sOuterSize, sInnerSize, preTokensLeftUp,
            nextTokensLeftUp, isAttenMaskUsed);
    }

    double coreWeightTarget = (double(totalBlockNumsOneHead * inputParams.qHeads) / double(blockDim));
    int64_t qSeqlenOuterSize = (inputParams.qSeqlen + sOuterSize - 1) / sOuterSize;
    const size_t tilingElementArrayLen = MAX_CORE_NUM;
    uint32_t curIndx = 0;
    ComputeSplitNBSeq(
        faTilingData, batchSize, tilingElementArrayLen, actualSeqLengths, actualSeqLengthsKV, sOuterSize, sInnerSize,
        coreWeightTarget, curIndx);

    int64_t sInnerBlockNum = (inputParams.kvSeqlen + sInnerSize - 1) / sInnerSize;
    int64_t totalSize = (totalBlockNumsOneHead / sInnerBlockNum) * inputParams.qHeads;

    faTilingData.multiCoreParamsRegbase.qSeqlenOuterSize = qSeqlenOuterSize;
    faTilingData.multiCoreParamsRegbase.coreNum = static_cast<int32_t>(curIndx + 1);
    faTilingData.multiCoreParamsRegbase.totalSize = totalSize;
    faTilingData.multiCoreParamsRegbase.splitFactorSize = CeilDivision(totalSize, static_cast<int64_t>(curIndx + 1));
    faTilingData.multiCoreParamsRegbase.splitFactorTailSize =
        CalcTailSize(totalSize, faTilingData.multiCoreParamsRegbase.splitFactorSize);

    return 0;
}

} // namespace FAInferTiling
#endif // XLLM_OPS_XFAI_ARCH35_HOST_A5_X_FLASH_ATTENTION_INFER_TILING_H