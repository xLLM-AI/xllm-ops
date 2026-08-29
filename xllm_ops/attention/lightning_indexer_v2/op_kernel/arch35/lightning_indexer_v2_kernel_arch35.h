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
 * \file lightning_indexer_v2_kernel_arch35.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_V2_KERNEL_ARCH35_H
#define LIGHTNING_INDEXER_V2_KERNEL_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch35.h"
#include "lightning_indexer_v2_service_vector_arch35.h"
#include "lightning_indexer_v2_service_cube_arch35.h"
#include "../lightning_indexer_v2_metadata.h"

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace matmul;
using namespace optiling;
using namespace optiling::detail;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
constexpr uint64_t Align_16_Bytes = 16UL;
// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;   // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;    // S2方向循环的结束Idx
    uint32_t actS1Size = 1U;  // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0U;
    uint32_t actS2SizeOrig = 0U; // 压缩前s2
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false;  // S1的实际长度小于shape的S1长度时，是否需要清理输出
    uint32_t actMBaseSize = 0U;            // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;          // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;         // S2方向循环的尾基本块大小
    bool isNeedLD = false;     // 该基本块是否需要LD
};

template <typename LIT>
class LightningIndexerV2Kernel {
public:
    __aicore__ inline LightningIndexerV2Kernel(){};
    __aicore__ inline void Init(__gm__ uint8_t *q, __gm__ uint8_t *k, __gm__ uint8_t *w,
                                __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK,
                                __gm__ uint8_t *blockTable, __gm__ uint8_t *outputIdxOffset,
                                __gm__ uint8_t *metadata, __gm__ uint8_t *sparseIndices, __gm__ uint8_t *sparseValues,
                                __gm__ uint8_t *workspace, const LIV2TilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = true;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using OUT_T = typename LIT::outputType;
    using SCORE_T = typename LIT::scoreType;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr LI_V2_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_V2_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    using W_T = float;

    LightningIndexerV2ServiceCube<LIT> matmulService;
    LightningIndexerV2ServiceVector<LIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 256;
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S1_BASE_SIZE_SMALL = 2;
    static constexpr uint32_t S2_BASE_SIZE = 128;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOBULE = 2;

protected:
    TPipe *pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    uint64_t valueOutCoreOffset = 0ULL;
    uint64_t outputIdxCoreOffset = 0ULL;
    bool isUsedCoreEqZero = false;
    bool isOutputIdxOffsetValid = false;
    bool hasCuSeqlensQ = false;
    bool hasCuSeqlensK = false;
    bool hasSequsedQ = false;
    bool hasSequsedK = false;
    bool hasCmpResidualK = false;
    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<uint32_t> metadataGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;

    GlobalTensor<uint32_t> cuSeqlensQGm;
    GlobalTensor<uint32_t> cuSeqlensKGm;
    GlobalTensor<uint32_t> sequsedQGm;
    GlobalTensor<uint32_t> sequsedKGm;
    GlobalTensor<uint32_t> cmpResidualKGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    LIV2Common::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    LIV2Common::SplitCoreInfo splitCoreInfo{};
    LIV2Common::LdSplitCoreInfo ldInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const LIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *cuSeqlensK,
                                            __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedK,
                                            __gm__ uint8_t *cmpResidualK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, LIV2Common::SplitCoreInfo &info);
    __aicore__ inline void SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                            GlobalTensor<uint32_t> &metadataGm);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig);
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            LIV2Common::RunInfo runInfo);
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t> &cuSeqlensQGm,
                                               GlobalTensor<uint32_t> &sequsedQGm,
                                               uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                  GlobalTensor<uint32_t> &cuSeqlensKGm,
                                                  GlobalTensor<uint32_t> &sequsedKGm,
                                                  uint32_t defaultSeqLen, uint32_t cmpRatio,
                                                  GlobalTensor<uint32_t> &cmpResidualKGm);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t &actS1Size, uint32_t &actS2Size,
                                               uint32_t &actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, LIV2Common::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitTilingData(const LIV2TilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->maskMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.topk = tilingData->topk;
    constInfo.cmpRatio = tilingData->cmpRatio;
    constInfo.batchSupperFlag = tilingData->batchSupperFlag;
    constInfo.keyStride0 = tilingData->keyStride0;
    constInfo.outputLayout = LAYOUT_T;  // 输出和输入形状一致
    if (LAYOUT_T == LI_V2_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if (K_LAYOUT_T == LI_V2_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    if (constInfo.gSize > 32 || constInfo.topk > 2048) {
        constInfo.mBaseSize = S1_BASE_SIZE_SMALL * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE_SMALL;
    } else {
        constInfo.mBaseSize = S1_BASE_SIZE * constInfo.gSize;
        constInfo.s1BaseSize = S1_BASE_SIZE;
    }
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.returnValueFlag = tilingData->returnValue;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::InitActualSeqLen(__gm__ uint8_t *cuSeqlensQ,
                                                          __gm__ uint8_t *cuSeqlensK,
                                                          __gm__ uint8_t *sequsedQ,
                                                          __gm__ uint8_t *sequsedK,
                                                          __gm__ uint8_t *cmpResidualK)
{
    if (cuSeqlensQ != nullptr) {
        cuSeqlensQGm.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensQ);
        hasCuSeqlensQ = true;
    }
    if (cuSeqlensK != nullptr) {
        cuSeqlensKGm.SetGlobalBuffer((__gm__ uint32_t *)cuSeqlensK);
        hasCuSeqlensK = true;
    }
    if (sequsedQ != nullptr) {
        sequsedQGm.SetGlobalBuffer((__gm__ uint32_t *)sequsedQ);
        hasSequsedQ = true;
    }
    if (sequsedK != nullptr) {
        sequsedKGm.SetGlobalBuffer((__gm__ uint32_t *)sequsedK);
        hasSequsedK = true;
    }
    if (cmpResidualK != nullptr) {
        cmpResidualKGm.SetGlobalBuffer((__gm__ uint32_t *)cmpResidualK);
        hasCmpResidualK = true;
    }
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims,
                                                             bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &cuSeqlensQGm,
                                                             GlobalTensor<uint32_t> &sequsedQGm,
                                                             uint32_t defaultSeqLen)
{
    if (hasSequsedQ) {
        return sequsedQGm.GetValue(bIdx);
    } else if (hasCuSeqlensQ) {
        return cuSeqlensQGm.GetValue(bIdx + 1) - cuSeqlensQGm.GetValue(bIdx);
    } else {
        return defaultSeqLen;
    }
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims,
                                                             bool isAccumSeq,
                                                             GlobalTensor<uint32_t> &cuSeqlensKGm,
                                                             GlobalTensor<uint32_t> &sequsedKGm,
                                                             uint32_t defaultSeqLen, uint32_t cmpRatio,
                                                             GlobalTensor<uint32_t> &cmpResidualKGm)
{
    uint32_t residual = hasCmpResidualK ? cmpResidualKGm.GetValue(bIdx) : 0;
    if (hasSequsedK) {
        return sequsedKGm.GetValue(bIdx) * cmpRatio + residual;
    } else if (hasCuSeqlensK) {
        return (cuSeqlensKGm.GetValue(bIdx + 1) - cuSeqlensKGm.GetValue(bIdx)) * cmpRatio + residual;
    } else {
        return defaultSeqLen * cmpRatio + residual;
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::GetS1S2ActualSeqLen(uint32_t bIdx,
                                                             uint32_t &actS1Size,
                                                             uint32_t &actS2Size,
                                                             uint32_t &actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, cuSeqlensQGm, sequsedQGm,
                                constInfo.qSeqSize);
    // 压缩前的actS2Size
    actS2SizeOrig =
        GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.isAccumSeqS2,
                           cuSeqlensKGm, sequsedKGm, constInfo.kSeqSize, constInfo.cmpRatio, cmpResidualKGm);
    // 真实使用的压缩后S2长度
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::SplitCoreByAICPU(uint32_t cubeCoreIdx, uint32_t vecCoreIdx,
                                                                       GlobalTensor<uint32_t> &metadataGm)
{
    uint32_t liCoreEnableIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_CORE_ENABLE_INDEX);
    uint32_t bN2StartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_BN2_START_INDEX);
    uint32_t mStartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_M_START_INDEX);
    uint32_t s2StartIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_S2_START_INDEX);
    uint32_t bN2EndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_BN2_END_INDEX);
    uint32_t mEndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_M_END_INDEX);
    uint32_t s2EndIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_S2_END_INDEX);

    uint32_t liZeroCoreEnableIndex = GetAttrAbsIndex(0, LI_V2_CORE_ENABLE_INDEX);
    if (metadataGm.GetValue(liZeroCoreEnableIndex) == 0) {
        isUsedCoreEqZero = true;
    }
    if (metadataGm.GetValue(liCoreEnableIndex) == 0) {
        splitCoreInfo.isCoreEnable = false;
        return;
    } else {
        splitCoreInfo.isCoreEnable = true;
    }

    splitCoreInfo.bN2Start = metadataGm.GetValue(bN2StartIndex);
    splitCoreInfo.gS1Start = metadataGm.GetValue(mStartIndex);
    splitCoreInfo.s2Start = metadataGm.GetValue(s2StartIndex);
    splitCoreInfo.bN2End = metadataGm.GetValue(bN2EndIndex);
    splitCoreInfo.gS1End = metadataGm.GetValue(mEndIndex);
    splitCoreInfo.s2End  = metadataGm.GetValue(s2EndIndex);

    if (splitCoreInfo.s2End != 0) {
        // 此时只需要s2End往前退一格，bN2End和gS1End都不变
        splitCoreInfo.s2End = splitCoreInfo.s2End - 1;
    } else {
        if (splitCoreInfo.gS1End != 0) {
            // splitCoreInfo.gS1End != 0 splitCoreInfo.s2End == 0 时，gS1End需要往前退一格, bN2End不变
            // 此时需要使用bIdx获取实际Actal S2来计算出 s2End
            splitCoreInfo.gS1End = splitCoreInfo.gS1End - 1;
            // 需要获取当前的Actaul S2
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kHeadNum;
            uint32_t actS1Size, actS2Size, actS2SizeOrig;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
            // s2的切块数量
            uint32_t s2BaseNum;
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2SizeOrig);
            } else {
                s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            }
            splitCoreInfo.s2End = s2BaseNum - 1;
        } else {
            // splitCoreInfo.gS1End == 0 splitCoreInfo.s2End == 0 时，bN2End需要往前退一格
            // 此时需要使用bIdx获取实际Actal S1和S2来计算出 gS1End 和 s2End
            splitCoreInfo.bN2End = splitCoreInfo.bN2End - 1;

            // 需要获取当前的Actaul S1 S2
            uint32_t bIdx = splitCoreInfo.bN2End / constInfo.kHeadNum;
            uint32_t actS1Size, actS2Size, actS2SizeOrig;
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);

            // s1的切块数量
            uint32_t s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            splitCoreInfo.gS1End = s1GBaseNum - 1;

            // s2的切块数量
            uint32_t s2BaseNum;
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(splitCoreInfo.gS1End, actS1Size, actS2SizeOrig);
            } else {
                s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            }
            splitCoreInfo.s2End = s2BaseNum - 1;
        }
    }

    uint32_t ldFirstWorkSpaceIndex = GetAttrAbsIndex(cubeCoreIdx, LI_V2_FIRST_LD_V2_DATA_WORKSPACE_IDX_INDEX, false);
    ldInfo.saveWorkSpaceIdx = metadataGm.GetValue(ldFirstWorkSpaceIndex);
    if ASCEND_IS_AIV {
        uint32_t ldCoreEnableIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_CORE_ENABLE_INDEX, true);
        ldInfo.isLdCoreEnable = metadataGm.GetValue(ldCoreEnableIndex);

        if (!ldInfo.isLdCoreEnable) {
            return;
        }

        uint32_t ldBn2IdxIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_BN2_IDX_INDEX, true);
        uint32_t ldMIdxIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_M_IDX_INDEX, true);
        uint32_t ldWorkspaceIdxIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_WORKSPACE_IDX_INDEX, true);
        uint32_t ldWorkspaceNumIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_WORKSPACE_NUM_INDEX, true);
        uint32_t ldMstartIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_M_START_INDEX, true);
        uint32_t ldMNumIndex = GetAttrAbsIndex(vecCoreIdx, LD_V2_M_NUM_INDEX, true);

        ldInfo.bn2Idx = metadataGm.GetValue(ldBn2IdxIndex);
        ldInfo.bIdx = ldInfo.bn2Idx / constInfo.kHeadNum;
        ldInfo.n2Idx = ldInfo.bn2Idx % constInfo.kHeadNum;
        ldInfo.mIdx = metadataGm.GetValue(ldMIdxIndex);
        ldInfo.workspaceIdx = metadataGm.GetValue(ldWorkspaceIdxIndex);
        ldInfo.workspaceNum = metadataGm.GetValue(ldWorkspaceNumIndex);
        ldInfo.mStart = metadataGm.GetValue(ldMstartIndex);
        ldInfo.mNum = metadataGm.GetValue(ldMNumIndex);
        uint64_t actualSeqQPrefixSum = 0;
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::TND) {
            actualSeqQPrefixSum = cuSeqlensQGm.GetValue(ldInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(ldInfo.bIdx) * constInfo.qSeqSize;
        }
        ldInfo.indiceOutCoreOffset = static_cast<uint64_t>(actualSeqQPrefixSum) * constInfo.kHeadNum * constInfo.topk +
                                    ldInfo.n2Idx * constInfo.topk +
                                    static_cast<uint64_t>(ldInfo.mIdx) * constInfo.s1BaseSize * constInfo.kHeadNum *
                                    constInfo.topk;
    }
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                     uint32_t actS2SizeOrig)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    // 压缩前的validS2LenBase
    int32_t validS2LenBase = static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size);
    int32_t validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase +
                          static_cast<int32_t>(constInfo.s1BaseSize)) / static_cast<int32_t>(constInfo.cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename LIT>
__aicore__ inline uint32_t LightningIndexerV2Kernel<LIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = constInfo.isLDOpen ? CeilDiv(actS2Size, constInfo.s2BaseSize) : (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = constInfo.isLDOpen ? GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig) :
                                             (actS2Size > 0 ? 1 : 0);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}
 
// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
template <typename LIT>
__aicore__ void inline LightningIndexerV2Kernel<LIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum,
                                                   LIV2Common::SplitCoreInfo &info)
{
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastGS1RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;
    if (curCoreIdx < coreNum) {
        splitCoreInfo.isCoreEnable = true;
    } else {
        splitCoreInfo.isCoreEnable = false;
        return;
    }

    bool findLastCoreEnd = true;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum, s2Loop;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
            s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
        }
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::BSND) {
            if (findLastCoreEnd && (s1GBaseNum == 0U || s2BaseNum == 0U)) {
                info.bN2Start = bN2Idx;
                info.gS1Start = 0;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
        }
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            if (constInfo.attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2SizeOrig);
            }
            if (findLastCoreEnd && s2BaseNum == 0U) {
                info.bN2Start = bN2Idx;
                info.gS1Start = gS1Idx;
                info.s2Start = 0;
                findLastCoreEnd = false;
            }
            s2Loop = constInfo.isLDOpen ? s2BaseNum : (actS2Size > 0 ? 1 : 0);
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx;
                    info.gS1Start = gS1Idx;
                    info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                uint32_t s2RemainBaseNum = s2Loop - s2Idx;
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx;
                    info.gS1End = gS1Idx;
                    info.s2End = constInfo.isLDOpen ? s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1
                                                    : s2BaseNum - 1;
                    if (coreIdx == curCoreIdx) {
                        // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                        if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                            info.isLD = true;
                        }
                        // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                        if (coreIdx == coreNum - 1 && info.bN2End != constInfo.batchSize - 1) {
                            info.bN2End = constInfo.batchSize - 1;
                            info.gS1End = 0;
                            info.s2End = 0;
                        }
                        return;
                    }
                    coreIdx++;
                    findLastCoreEnd = true;
                    s2Idx = info.s2End + 1;
                    lastGS1RemainBlockCnt = 0;
                    coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
                } else {
                    lastGS1RemainBlockCnt += s2RemainBaseNum;
                    break;
                }
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx,
                                                                          uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_V2_LAYOUT::TND) {
            uint32_t tBase = cuSeqlensQGm.GetValue(bIdx);
            uint32_t s1Count = cuSeqlensQGm.GetValue(bIdx + 1) - tBase;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (static_cast<uint64_t>(tBase) + s1Idx) * constInfo.kHeadNum * constInfo.topk +  // T轴、s1轴偏移
                    static_cast<uint64_t>(n2Idx) * constInfo.topk;                                  // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_V2_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset = static_cast<uint64_t>(bIdx) * constInfo.qSeqSize *
                                           constInfo.kHeadNum * constInfo.topk +
                                           static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum * constInfo.topk +
                                           static_cast<uint64_t>(n2Idx) * constInfo.topk;
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                                           __gm__ uint8_t *weights, __gm__ uint8_t *cuSeqlensQ,
                                                           __gm__ uint8_t *cuSeqlensK, __gm__ uint8_t *sequsedQ,
                                                           __gm__ uint8_t *sequsedK, __gm__ uint8_t *cmpResidualK,
                                                           __gm__ uint8_t *blockTable, __gm__ uint8_t *outputIdxOffset,
                                                           __gm__ uint8_t *metadata, __gm__ uint8_t *sparseIndices,
                                                           __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace,
                                                           const LIV2TilingData *__restrict tiling,
                                                           TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();  // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx();  // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK);

    // 获取分核信息
    if (metadata != nullptr) {
        metadataGm.SetGlobalBuffer((__gm__ uint32_t *)metadata);
        SplitCoreByAICPU(aiCoreIdx, tmpBlockIdx, metadataGm);
    } else {
        SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);
    }

    pipe = tPipe;

    uint64_t offset = 0;
    uint32_t topkCountAlign16_ = LIV2Common::Align(constInfo.topk, Align_16_Bytes); // topkCount对齐到16
    // vec 把整个s2的score存储在GM，大小为s1BaseSize * 16K * 4
    GlobalTensor<SCORE_T> scoreGm; // 存放vec核写出的score
    uint64_t singleCoreScoreSize = constInfo.s1BaseSize * LIV2Common::Align((uint64_t)constInfo.kSeqSize,
                                                                            (uint64_t)constInfo.s2BaseSize)
                                                                            * sizeof(SCORE_T);
    scoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + aiCoreIdx * singleCoreScoreSize));
    offset += GetBlockNum() * singleCoreScoreSize;
    // vec 存储需要LD的s1对应的s2的score与index，
    // 大小为s1BaseSize * sparseCount * 2，一个核内最多有两个s1BaseSize需要LD
    GlobalTensor<SCORE_T> ldScoreGm; // 存放进行LD的s2 score
    ldScoreGm.SetGlobalBuffer((__gm__ SCORE_T *)(workspace + offset));
    offset += static_cast<uint64_t>(GetBlockNum()) * constInfo.s1BaseSize * topkCountAlign16_ * 2 * sizeof(SCORE_T);
    GlobalTensor<int32_t> ldIndexGm; // 存放进行LD的s2 Index
    ldIndexGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + offset));
    offset += static_cast<uint64_t>(GetBlockNum()) * constInfo.s1BaseSize * topkCountAlign16_ * 2 * sizeof(int32_t);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, ldInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        valueOutGm.SetGlobalBuffer((__gm__ float *)sparseValues);
        weightsGm.SetGlobalBuffer((__gm__ W_T *)weights);
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        if (outputIdxOffset != nullptr) {
            isOutputIdxOffsetValid = true;
            outputIdxOffsetGm.SetGlobalBuffer((__gm__ int32_t *)outputIdxOffset);
        }
        vectorService.InitVecInputTensor(weightsGm, indiceOutGm, valueOutGm, blockTableGm, outputIdxOffsetGm);
        vectorService.InitVecWorkspaceTensor(scoreGm, ldScoreGm, ldIndexGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm);
    }
    InitBuffers();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx == splitCoreInfo.bN2End) && (gS1LoopIdx == splitCoreInfo.gS1End);
    uint32_t s2BlockNum;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    }
    tempLoopInfo.s2LoopEnd = isEnd ? splitCoreInfo.s2End : s2BlockNum - 1;
    if (splitCoreInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < s2BlockNum - 1) {
        tempLoopInfo.isNeedLD = true;
    } else {
        tempLoopInfo.isNeedLD = false;
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.actS2Size % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End) ? splitCoreInfo.gS1End : gS1SplitNum - 1;
    if constexpr (LAYOUT_T == LI_V2_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum - 1 && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx,
                                                                  LIV2Common::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx <= tempLoopInfo.s2LoopEnd;
    runInfo.isNeedLD = tempLoopInfo.isNeedLD;
    if (runInfo.isNeedLD && s2LoopIdx == tempLoopInfo.s2LoopEnd) {
        runInfo.saveWorkSpaceIdx = ldInfo.saveWorkSpaceIdx;
        ldInfo.saveWorkSpaceIdx++;
    }
    
    if (!runInfo.isValid) {
        return;
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        LIV2Common::Align((uint32_t)runInfo.actualSingleProcessSInnerSize,
        LIV2Common::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.gS1Idx == splitCoreInfo.gS1End) &&
                           (runInfo.s2Idx == splitCoreInfo.s2End);
    runInfo.isOutputIdxOffsetValid = isOutputIdxOffsetValid;
    runInfo.needTndPadding = false;
    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (LAYOUT_T == LI_V2_LAYOUT::TND) {
            actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.bIdx);
            if (hasSequsedQ) {
                uint32_t curSequsedQ = sequsedQGm.GetValue(runInfo.bIdx);
                uint32_t nextPrefixSum = cuSeqlensQGm.GetValue(runInfo.bIdx + 1);
                uint32_t curCuLensQ = nextPrefixSum - actualSeqQPrefixSum;
                if (curSequsedQ < curCuLensQ) {
                    runInfo.needTndPadding = true;
                    runInfo.curCuSeqlensQ = curCuLensQ;
                    runInfo.curSequsedQ = curSequsedQ;
                }
            }
        } else {  // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : static_cast<uint64_t>(runInfo.bIdx) * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.topk + runInfo.n2Idx * constInfo.topk;
        // B,S1,N2,k/T,N2,k
        valueOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.topk + runInfo.n2Idx * constInfo.topk;
        outputIdxCoreOffset =
            (actualSeqQPrefixSum + runInfo.gS1Idx * constInfo.s1BaseSize) * constInfo.kHeadNum + runInfo.n2Idx;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_V2_LAYOUT::TND) { // T N2 D
        actualSeqKPrefixSum = cuSeqlensKGm.GetValue(runInfo.bIdx);
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.valueOutOffset = valueOutCoreOffset;
    runInfo.outputIdxCoreOffset = outputIdxCoreOffset;
    if ASCEND_IS_AIV {
        if (runInfo.needTndPadding) {
            vectorService.DoTndPadding(runInfo);
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::Process()
{
    if (usedCoreNum == 0 || isUsedCoreEqZero) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }

    ProcessMain();
    ProcessDecode();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2;  // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            static_cast<uint64_t>(constInfo.batchSize) * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.topk;
        uint64_t singleCoreSize =
            LIV2Common::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
            if (constInfo.returnValueFlag) {
                event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
                
                GlobalTensor<uint32_t> valueOutGmTmp;
                valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t *)valueOutGm.GetPhyAddr());
                GlobalTensor<uint32_t> valueOut = valueOutGmTmp[baseSize];

                AscendC::InitGlobalMemory(valueOut, dealSize, constInfo.NEG_INF_FLOAT);
            }
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessMain()
{
    if (!splitCoreInfo.isCoreEnable) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 1);
    } else {
        matmulService.AllocEventID();
    }

    LIV2Common::RunInfo runInfo;
    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            runInfo.s2Start = splitCoreInfo.s2Start;
            runInfo.s2LoopEnd = tempLoopInfo.s2LoopEnd;
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 0);
        CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_FIX>(LIV2Common::ConstInfo::CROSS_VC_EVENT + 1);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                                          LIV2Common::RunInfo runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
    } else {
        vectorService.ProcessVec1(runInfo);
        if (runInfo.isLastS2InnerLoop) {   // 本核s2last
            vectorService.ProcessTopK(runInfo);
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2Kernel<LIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe, ldInfo);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (ldInfo.isLdCoreEnable) {
            vectorService.ProcessLD();
        }
    }
}
}  // namespace LIKernel
#endif  // LIGHTNING_INDEXER_V2_KERNEL_ARCH35_H
