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
 * \file lightning_indexer_v2_service_vector_arch35.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_ARCH35_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_arch35.h"
#include "../arch35/vf/lightning_indexer_v2_vector1.h"
#include "../arch35/vf/lightning_indexer_v2_topk.h"

namespace LIV2Kernel {
using namespace LIV2Common;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t TRUNK_LEN_4K = 4096;
constexpr uint32_t TRUNK_LEN_2K = 2048;
constexpr uint32_t TRUNK_LEN_1K = 1024;
constexpr uint32_t TOPK_LEN_7K = 7168;
constexpr uint32_t TOPK_LEN_5K = 5120;

template <typename LIT>
class LightningIndexerV2ServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_V2_LAYOUT LAYOUT_T = LIT::layout;
    static constexpr LI_V2_LAYOUT K_LAYOUT_T = LIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
    static constexpr bool DT_W_FLAG = true;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    using QK_T = typename LIT::queryKeyType;
    using SCORE_T = typename LIT::scoreType;
    using W_T = float;

    __aicore__ inline LightningIndexerV2ServiceVector(){};
    __aicore__ inline void ProcessVec1(const LIV2Common::RunInfo &info);
    __aicore__ inline void ProcessTopK(const LIV2Common::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                      const struct LIV2Common::LdSplitCoreInfo &ldInfo,
                                      const LIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitLDBuffers(TPipe *pipe, const struct LIV2Common::LdSplitCoreInfo &ldInfo);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                  GlobalTensor<SCORE_T> ldScoreGm,
                                                  GlobalTensor<int32_t> ldIndexGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<float> valueOutGm, GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<int32_t> outputIdxOffsetGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void DoTndPadding(const LIV2Common::RunInfo &runInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<SCORE_T> ldScoreGm;
    GlobalTensor<int32_t> ldIndexGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<float> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;
    static constexpr uint32_t MTE3_V_EVENT = EVENT_ID6;
    static constexpr uint32_t M_BASIC_BLOCK = 128;

private:
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<float> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;

    // tmp buff for returnValue K_T
    TBuf<TPosition::VECCALC> valueOutBuf_;
    LocalTensor<float> valueOutLocal_;
    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;
    LocalTensor<uint32_t> ldIndexLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t topkCountAlign16_ = 0;
    uint32_t trunkLen_ = 0;
    bool returnValueFlag = false;

    struct LIV2Common::ConstInfo constInfo_;
    struct LIV2Common::LdSplitCoreInfo ldInfo_;
    liV2Topk::LIV2Topk<SCORE_T> topkOp_;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(M_BASIC_BLOCK, 2) * s2BaseSize_ * sizeof(float));
    resMm1UB_ = resMm1Buf_.Get<float>();
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<W_T>();
    // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    // out
    vec1OutUB_ = outBuf_.Get<SCORE_T>();
    // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    // returnvalue
    // 大小：topK * sizeof(float) 64:duplicate刷-1需要额外空间
    if (topkCount_ <= 2048) {
        pipe->InitBuffer(valueOutBuf_, (topkCountAlign256_ + 64) * sizeof(float));
        valueOutLocal_ = valueOutBuf_.Get<float>();
    } else {
        valueOutLocal_ = mrgValueBuf_.Get<float>();
    }
    // Topk
    // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();
    // 大小：topkCountAlign256_ * sizeof(SCORE_T)
    pipe->InitBuffer(scoreOutBuf_, (topkCountAlign256_ + 64)* sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                                    const struct LIV2Common::LdSplitCoreInfo &ldInfo,
                                                    const LIV2TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    returnValueFlag = constInfo.returnValueFlag;
    blockId_ = GetBlockIdx();
    trunkLen_ =
        constInfo.topk > TOPK_LEN_5K ? (constInfo.topk > TOPK_LEN_7K ? TRUNK_LEN_1K : TRUNK_LEN_4K) : TRUNK_LEN_8K;
    topkCount_ = constInfo.topk;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = LIV2Common::Align(constInfo.topk, (uint64_t)256); // topkCount对齐到256
    topkCountAlign16_ = LIV2Common::Align(constInfo.topk, (uint64_t)16);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitLDBuffers(TPipe *pipe,
                                                    const struct LIV2Common::LdSplitCoreInfo &ldInfo)
{
    ldIndexLocal_ = resMm1UB_.template ReinterpretCast<uint32_t>();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm,
                                                                                GlobalTensor<int32_t> indiceOutGm,
                                                                                GlobalTensor<float> valueOutGm,
                                                                                GlobalTensor<int32_t> blockTableGm,
                                                                                GlobalTensor<int32_t> outputIdxOffsetGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
    this->outputIdxOffsetGm = outputIdxOffsetGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                                                    GlobalTensor<SCORE_T> ldScoreGm,
                                                                                    GlobalTensor<int32_t> ldIndexGm)
{
    this->scoreGm = scoreGm; // resucesum*k
    this->ldScoreGm = ldScoreGm;
    this->ldIndexGm = ldIndexGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    uint64_t dealSize = constInfo_.topk;
    GlobalTensor<int32_t> indexOutput = indiceOutGm[invalidS1Offset];
    AscendC::InitGlobalMemory(indexOutput, dealSize, constInfo_.INVALID_IDX);

    if (returnValueFlag) {
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>(), constInfo_.NEG_INF_FLOAT, constInfo_.topk);

        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        AscendC::DataCopyParams copyOutValueParams;
        copyOutValueParams.blockCount = 1;
        copyOutValueParams.blockLen = constInfo_.topk * sizeof(float);
        copyOutValueParams.srcStride = 0;
        copyOutValueParams.dstStride = 0;
        AscendC::DataCopyPad(valueOutGm[invalidS1Offset], valueOutLocal_, copyOutValueParams);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::DoTndPadding(const LIV2Common::RunInfo &runInfo)
{
    uint32_t paddingLen = runInfo.curCuSeqlensQ - runInfo.curSequsedQ;
    uint64_t paddingOffset = runInfo.indiceOutOffset + runInfo.curSequsedQ * constInfo_.kHeadNum * constInfo_.topk;
    uint64_t dealSize = paddingLen * constInfo_.kHeadNum * constInfo_.topk;
    GlobalTensor<int32_t> indiceOutPaddingStart = indiceOutGm[paddingOffset];
    AscendC::InitGlobalMemory(indiceOutPaddingStart, dealSize, constInfo_.INVALID_IDX);

    if (constInfo_.returnValueFlag) {
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);

        GlobalTensor<uint32_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint32_t *)valueOutGm.GetPhyAddr());
        GlobalTensor<uint32_t> valueOut = valueOutGmTmp[paddingOffset];

        AscendC::InitGlobalMemory(valueOut, dealSize, constInfo_.NEG_INF_FLOAT);
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessVec1(const LIV2Common::RunInfo &info)
{
    auto pingpong = info.loop % 2;
    // CV同步
    CrossCoreWaitFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(
        LIV2Common::ConstInfo::CROSS_CV_EVENT + pingpong); // V核等C核计算完mm1，mm1Res已搬运到UB

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    if (curAivS1ProcNum == 0) {
        // V核处理完，通知C核可以把mm1Res搬运到UB
        CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                       pingpong);
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // weightsGm --> weightUB_
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<W_T> padWeightsParams{true, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = curAivS1ProcNum;
    qwDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qwDataCopyExtParams.blockLen) / 32;
    DataCopyPad(weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(W_T))], weightsGm[weightGmOffset],
                qwDataCopyExtParams, padWeightsParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    auto qkVLStride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * M_BASIC_BLOCK;
    auto outBase = vec1OutUB_[(pingpong) * CeilDiv(s1BaseSize_, 2) * s2BaseSize_];
    auto qkBase = resMm1UB_[(pingpong) * (UB_BANK_STRIDE / sizeof(QK_T))];
    auto weightBase = weightUB_[(pingpong) * (UB_BANK_STRIDE / sizeof(float))];
    liV2Vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                        qkBase, qkVLStride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)),
                                        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T),
                                        gSize_, curAivS1ProcNum);

    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    // outUB_ --->  scoreGm
    int64_t vec1OutGmOffset =
        blockId_ % 2 == 0 ?
            curS2Idx :
            CeilDiv(s1BaseSize_, 2) * LIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride =
        (LIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);

    DataCopyPad(scoreGm[vec1OutGmOffset], vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_],
                copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    // V核处理完，通知C核可以把mm1Res搬运到UB
    CrossCoreSetFlag<LIV2Common::ConstInfo::LI_SYNC_MODE4, PIPE_V>(LIV2Common::ConstInfo::CROSS_VC_EVENT +
                                                                   pingpong);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessTopK(const LIV2Common::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2StartIdx = info.s2Start * s2BaseSize_;
    int64_t curS2EndIdx = (info.s2LoopEnd + 1) * s2BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    // LD 需要搬运到的起始位置 32字节
    int64_t ldDstOffset = (info.isNeedLD) ? curS2StartIdx : 0;
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyIndicesOutParams;
    ldCopyIndicesOutParams.blockCount = 1;
    ldCopyIndicesOutParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
    ldCopyIndicesOutParams.srcStride = 0;
    ldCopyIndicesOutParams.dstStride = 0;

    AscendC::DataCopyParams ldCopyScoreOutParams;
    ldCopyScoreOutParams.blockCount = 1;
    ldCopyScoreOutParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T); // bytes
    ldCopyScoreOutParams.srcStride = 0;
    ldCopyScoreOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    int32_t validAllS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;
        int64_t outputIdxOffset = 0;
        if (info.isOutputIdxOffsetValid) {
            outputIdxOffset = outputIdxOffsetGm.GetValue(info.outputIdxCoreOffset + rowIdx * kHeadNum_);
        }

        SCORE_T zero = 0;
        int32_t neg = -1;
        uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
        if (constInfo_.attenMaskFlag) {
            validAllS2Len = ((int32_t)i + cuRealAcSeq) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t validS2Len = validAllS2Len;
        if (info.isNeedLD) {
            // 当前核处理的s2长度validS2Len
            validS2Len = Min((info.s2LoopEnd + 1) * s2BaseSize_, validAllS2Len) - curS2StartIdx;
        }
        uint64_t ldOffset = info.saveWorkSpaceIdx * s1BaseSize_ * topkCountAlign16_ + rowIdx * topkCountAlign16_;
        if (validS2Len <= 0 && !info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            if (returnValueFlag) {
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

                Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>(), constInfo_.NEG_INF_FLOAT, topkCount_);

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(float);
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_], valueOutLocal_,
                                     copyOutValueParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        } else if (validS2Len <= 0 && info.isNeedLD) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCountAlign16_);
            Duplicate(scoreOutLocal_, zero, topkCountAlign16_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[ldOffset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[ldOffset], indicesOutLocal_.ReinterpretCast<int32_t>(),
                ldCopyIndicesOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            bool useSingleLoop =
                (s2LoopNum == 1) || ((topkCount_ > trunkLen_) && (validS2Len <= (uint32_t)topkCountAlign256_));
            if (useSingleLoop) {
                uint32_t validS2LenAlign = LIV2Common::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mrgValueLocal_,
                    scoreGm[vecOffset * LIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)
                     + ldDstOffset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign,
                        0, 1, info.isNeedLD, returnValueFlag, outputIdxOffset);
            } else {
                uint32_t outputIdxOffsetTmp = 0;
                uint32_t actS2LoopNum = 0;
                if (topkCount_ > trunkLen_) {
                    actS2LoopNum = 1 + (validS2Len - topkCountAlign256_ + trunkLen_ - 1) / trunkLen_;
                } else {
                    actS2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
                }
                for (uint32_t loopIdx = 0; loopIdx < actS2LoopNum; loopIdx++) {
                    if (loopIdx == actS2LoopNum - 1) {
                        outputIdxOffsetTmp = outputIdxOffset;
                    }
                    if (loopIdx == 0) {
                        if (topkCount_ > trunkLen_) {
                            copyInParams.blockLen = topkCountAlign256_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(scoreOutLocal_,
                                                scoreGm[vecOffset * LIV2Common::Align(
                                                    (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                                    ldDstOffset],
                                                copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                            AscendC::CreateVecIndex(topkSharedTmpLocal_.ReinterpretCast<int32_t>(), (int32_t)zero,
                                                    topkCountAlign256_);
                        } else {
                            copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                            AscendC::DataCopyPad(mrgValueLocal_,
                                                scoreGm[vecOffset * LIV2Common::Align(
                                                    (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                                    ldDstOffset],
                                                copyInParams, padParams);
                            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_,
                                    loopIdx, actS2LoopNum, info.isNeedLD, returnValueFlag, outputIdxOffsetTmp);
                        }

                        continue;
                    }
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                    uint32_t validTrunkLen = 0;
                    uint32_t offset = 0;
                    if (topkCount_ > trunkLen_) {
                        validTrunkLen = (topkCountAlign256_ + (loopIdx - 1) * trunkLen_ + trunkLen_) > validS2Len ?
                                            (validS2Len - topkCountAlign256_) % trunkLen_ :
                                            trunkLen_;
                        offset = vecOffset * LIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) +
                                topkCountAlign256_ + (loopIdx - 1) * trunkLen_ + ldDstOffset;
                    } else {
                        validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len ?
                                              validS2Len % trunkLen_ : trunkLen_;
                        offset = vecOffset * LIV2Common::Align(
                                (uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + loopIdx * trunkLen_ +
                                ldDstOffset;
                    }
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // topk如果没有对齐到256，则把topkCountAlign256_ - topkCount_部分刷0
                    // 如果topk > trunkLen，第一轮调用topk是直接拷贝的，不需要刷0
                    bool isZeroPadding = (topkCount_ > trunkLen_) ? (loopIdx > 1) : true;
                    if (topkCountAlign256_ != topkCount_ && isZeroPadding) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                                  topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if ((topkCountAlign256_ + validTrunkLen) % 256 != 0) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero,
                                  LIV2Common::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                    }
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, LIV2Common::Align(
                            topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx,
                            actS2LoopNum, info.isNeedLD, returnValueFlag, outputIdxOffsetTmp);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                    (int32_t)(zero + outputIdxOffset), validS2Len);

            // 如果需要LD 则将需要保存Value值
            if (info.isNeedLD || returnValueFlag) {
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
                copyInParams.blockLen = LIV2Common::Align(validS2Len, (int32_t)32) * sizeof(SCORE_T);
                AscendC::DataCopyPad(scoreOutLocal_,
                    scoreGm[vecOffset * LIV2Common::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)
                    + ldDstOffset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            }
        }
        if (!info.isNeedLD) {
            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
            }
            
            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64],
                        neg, topkCount_ - (validS2Len / 8 * 8 + 64));
            }

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            
            // 是否返回Value值
            if (returnValueFlag) {
                WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                liV2Vector1::UIntToFloatReturnValue(valueOutLocal_, scoreOutLocal_, topkCountAlign256_,
                    constInfo_.NEG_INF_FLOAT);
                // 无效值刷0
                if (validS2Len < topkCount_) {
                    uint64_t mask[1];
                    mask[0] = ~0;
                    mask[0] = mask[0] << (validS2Len % 8);
                    PipeBarrier<PIPE_V>();
                    Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 8 * 8],
                            constInfo_.NEG_INF_FLOAT, mask, 1, 1, 0);
                }
            
                if (validS2Len / 8 * 8 + 64 < topkCount_) {
                    PipeBarrier<PIPE_V>();
                    Duplicate(valueOutLocal_.template ReinterpretCast<uint32_t>()[validS2Len / 8 * 8 + 64],
                            constInfo_.NEG_INF_FLOAT,
                            topkCount_ - (validS2Len / 8 * 8 + 64));
                }
                SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                AscendC::DataCopyParams copyOutValueParams;
                copyOutValueParams.blockCount = 1;
                copyOutValueParams.blockLen = topkCount_ * sizeof(float); // bytes
                copyOutValueParams.srcStride = 0;
                copyOutValueParams.dstStride = 0;
                // 搬运到GM
                AscendC::DataCopyPad(valueOutGm[info.valueOutOffset + (curS1Idx + rowIdx) * topkCount_],
                    valueOutLocal_, copyOutValueParams);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        } else {
            PipeBarrier<PIPE_V>();
            AscendC::Adds(indicesOutLocal_.ReinterpretCast<int32_t>(), indicesOutLocal_.ReinterpretCast<int32_t>(),
                static_cast<int32_t>(curS2StartIdx), topkCountAlign16_);
            if (validS2Len < topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (validS2Len % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskI[1];
                maskI[0] = ~0;
                maskI[0] = maskI[0] << (validS2Len % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, maskI, 1, 1, 0);
            }
            if (validS2Len / scoreAlign * scoreAlign + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[validS2Len / scoreAlign * scoreAlign + 64], zero,
                    topkCount_ - (validS2Len / scoreAlign * scoreAlign + 64));
            }
            if (validS2Len / 8 * 8 + 64 < topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg,
                    topkCount_ - (validS2Len / 8 * 8 + 64));
            }
            if (topkCountAlign16_ != topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % scoreAlign);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / scoreAlign * scoreAlign], zero, mask, 1, 1, 0);
                uint64_t maskIndices[1];
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 8 * 8], neg, maskIndices, 1, 1, 0);
            }

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            // 将每一行S1的Topk结果存放在ldScoreGm和ldIndexGm中
            AscendC::DataCopyPad(ldScoreGm[ldOffset], scoreOutLocal_, ldCopyScoreOutParams);
            AscendC::DataCopyPad(ldIndexGm[ldOffset], indicesOutLocal_.ReinterpretCast<int32_t>(),
                ldCopyIndicesOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessLD()
{
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyParams copyOutValueParams;
    copyOutValueParams.blockCount = 1;
    copyOutValueParams.blockLen = topkCount_ * sizeof(float); // bytes
    copyOutValueParams.srcStride = 0;
    copyOutValueParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<SCORE_T> scorePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams ldScoreParams;
    ldScoreParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T); // bytes
    // s1Basesize-1 两个基本块之间的距离
    ldScoreParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(SCORE_T);
    ldScoreParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<uint32_t> indexPadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams ldIndexParams;
    ldIndexParams.blockLen = topkCountAlign16_ * sizeof(uint32_t); // bytes
    ldIndexParams.srcStride = (s1BaseSize_ - 1) * topkCountAlign16_ * sizeof(uint32_t);
    ldIndexParams.dstStride = 0;

    uint64_t ldProcessLen = ldInfo_.workspaceNum * topkCountAlign16_;
    if (ldProcessLen <= 0) {
        return;
    }

    SCORE_T zero = 0;
    int32_t neg = -1;
    uint32_t scoreAlign = sizeof(SCORE_T) == 4 ? 8 : 16; // int32对应UB对齐数为8，int16需要16
    if (topkCountAlign16_ > trunkLen_) {
        AscendC::DataCopyExtParams ldScoreSliceParams;
        ldScoreSliceParams.blockCount = 1;
        ldScoreSliceParams.srcStride = 0;
        ldScoreSliceParams.dstStride = 0;

        AscendC::DataCopyExtParams ldIndexSliceParams;
        ldIndexSliceParams.blockCount = 1;
        ldIndexSliceParams.srcStride = 0;
        ldIndexSliceParams.dstStride = 0;

        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            Duplicate(mrgValueLocal_, zero, topkCountAlign256_ + trunkLen_);
            Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>(), neg, topkCountAlign256_ + trunkLen_);

            uint64_t baseLdGmOffset = ldInfo_.workspaceIdx * s1BaseSize_ * topkCountAlign16_ +
                                      topkCountAlign16_ * (ldInfo_.mStart + j);
            
            ldScoreSliceParams.blockLen = topkCountAlign16_ * sizeof(SCORE_T);
            ldIndexSliceParams.blockLen = topkCountAlign16_ * sizeof(uint32_t);

            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            AscendC::DataCopyPad(mrgValueLocal_, ldScoreGm[baseLdGmOffset], ldScoreSliceParams, scorePadParams);
            AscendC::DataCopyPad(ldIndexLocal_, ldIndexGm[baseLdGmOffset].ReinterpretCast<uint32_t>(),
                                 ldIndexSliceParams, indexPadParams);

            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            if (topkCountAlign16_ < topkCountAlign256_) {
                Duplicate(mrgValueLocal_[topkCountAlign16_], zero, topkCountAlign256_ - topkCountAlign16_);
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[topkCountAlign16_], neg,
                    topkCountAlign256_ - topkCountAlign16_);
                PipeBarrier<PIPE_V>();
            }
            
            AscendC::DataCopy(indicesOutLocal_, ldIndexLocal_, topkCountAlign256_);
            AscendC::DataCopy(scoreOutLocal_, mrgValueLocal_, topkCountAlign256_);

            for (uint32_t workspaceOffset =1; workspaceOffset < ldInfo_.workspaceNum; workspaceOffset++) {
                for (uint32_t chunkOffset = 0; chunkOffset < topkCountAlign16_; chunkOffset += trunkLen_) {
                    uint32_t remainLen = topkCountAlign16_ - chunkOffset;
                    uint32_t chunkLen = remainLen > trunkLen_ ? trunkLen_ : remainLen;
                    uint64_t LDGmOffset = (ldInfo_.workspaceIdx + workspaceOffset) * s1BaseSize_ * topkCountAlign16_ +
                                        topkCountAlign16_ * (ldInfo_.mStart + j) + chunkOffset;

                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[topkCountAlign256_], zero, trunkLen_);
                    Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[topkCountAlign256_], neg, trunkLen_);

                    ldScoreSliceParams.blockLen = chunkLen * sizeof(SCORE_T);
                    ldIndexSliceParams.blockLen = chunkLen * sizeof(uint32_t);
                    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], ldScoreGm[LDGmOffset],
                                         ldScoreSliceParams, scorePadParams);
                    AscendC::DataCopyPad(ldIndexLocal_[topkCountAlign256_],
                                         ldIndexGm[LDGmOffset].ReinterpretCast<uint32_t>(),
                                         ldIndexSliceParams, indexPadParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

                    uint32_t s2Len = topkCountAlign256_ + chunkLen;
                    uint32_t s2LenAlign = LIV2Common::Align(s2Len, (uint32_t)256);
                    
                    topkOp_.LdTopK(mrgValueLocal_, ldIndexLocal_, indicesOutLocal_, scoreOutLocal_,
                                    s2LenAlign, j, ldInfo_.workspaceNum);
                    if (topkCountAlign256_ != topkCount_) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        Duplicate(scoreOutLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        uint64_t maskIndices[1];
                        maskIndices[0] = ~0;
                        maskIndices[0] = maskIndices[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64], neg,
                                maskIndices, 1, 1, 0);
                    }
                    if (topkCount_ / 64 * 64 + 64 < topkCountAlign256_) {
                        PipeBarrier<PIPE_V>();
                        Duplicate(scoreOutLocal_[topkCount_ / 64 * 64 + 64], zero,
                            topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                        PipeBarrier<PIPE_V>();
                        Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64 + 64], neg,
                            topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                    }
                    PipeBarrier<PIPE_V>();
                    AscendC::DataCopy(ldIndexLocal_, indicesOutLocal_, topkCountAlign256_);
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                }
            }

            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            uint64_t indiceOutGmOffset = ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) *
                                constInfo_.kHeadNum * topkCount_;
            AscendC::DataCopyPad(indiceOutGm[indiceOutGmOffset],
                            indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            
            if (returnValueFlag) {
                PipeBarrier<PIPE_V>();
                liV2Vector1::UIntToFloatReturnValue(valueOutLocal_, scoreOutLocal_, topkCountAlign256_,
                    constInfo_.NEG_INF_FLOAT);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                AscendC::DataCopyPad(valueOutGm[indiceOutGmOffset], valueOutLocal_, copyOutValueParams);
            }
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        return;
    }
    uint64_t mrgValueLen = trunkLen_ + topkCountAlign256_;
    uint32_t ldWorkspaceNum = (ldProcessLen > mrgValueLen) ? (trunkLen_ / topkCountAlign16_) : ldInfo_.workspaceNum;
    // 搬运次数，一次搬运ldworkspaceNum块
    uint32_t ldProcessNum = CeilDiv(ldInfo_.workspaceNum, ldWorkspaceNum);
    uint32_t ldProcessOffset = 0;
    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // 读取数据段过长
            if (ldProcessNum > 1) {
                ldProWorkspaceNum = (i == ldProcessNum - 1)
                    ? ldInfo_.workspaceNum - i * ldWorkspaceNum : ldWorkspaceNum; // 当前搬运块数
            }
            ldProcessOffset = (i != 0) ? topkCountAlign256_ : 0;
            PipeBarrier<PIPE_V>();
            // 索引全部刷-1 value全部刷0
            Duplicate(mrgValueLocal_[ldProcessOffset], zero, topkCountAlign256_ + trunkLen_ - ldProcessOffset);
            Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[ldProcessOffset], neg,
                topkCountAlign256_ + trunkLen_ - ldProcessOffset);

            ldScoreParams.blockCount = ldProWorkspaceNum;
            ldIndexParams.blockCount = ldProWorkspaceNum;

            int32_t s2Len = topkCountAlign16_ * ldProWorkspaceNum + ldProcessOffset;
            uint32_t s2LenAlign = LIV2Common::Align(s2Len, (int32_t)256); // 寄存器需要256对齐
            uint64_t LDGmOffset = ldInfo_.workspaceIdx * s1BaseSize_ * topkCountAlign16_ +
                                  topkCountAlign16_ * (ldInfo_.mStart + j) + i * ldWorkspaceNum * s1BaseSize_
                                      * topkCountAlign16_; // 加入LD处理偏移
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            AscendC::DataCopyPad(mrgValueLocal_[ldProcessOffset], ldScoreGm[LDGmOffset], ldScoreParams,
                                 scorePadParams);
            AscendC::DataCopyPad(ldIndexLocal_[ldProcessOffset], ldIndexGm[LDGmOffset].ReinterpretCast<uint32_t>(),
                                 ldIndexParams, indexPadParams);

            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            // 对非对齐索引和值刷0 -1
            if (s2LenAlign != s2Len) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (s2Len % 64);
                Duplicate(mrgValueLocal_[s2Len / 64 * 64], zero, mask, 1, 1, 0);
                // 把s2Len对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64], neg, mask, 1, 1, 0);
                PipeBarrier<PIPE_V>();
            }
            if (s2Len / 64 * 64 + 64 < s2LenAlign) {
                PipeBarrier<PIPE_V>();
                Duplicate(mrgValueLocal_[s2Len / 64 * 64 + 64], zero, s2LenAlign - (s2Len / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(ldIndexLocal_.ReinterpretCast<int32_t>()[s2Len / 64 * 64 + 64], neg,
                    s2LenAlign - (s2Len / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
            }

            topkOp_.LdTopK(mrgValueLocal_, ldIndexLocal_, indicesOutLocal_, scoreOutLocal_,
                            s2LenAlign, j, ldProcessNum);
            if (topkCountAlign256_ != topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                uint64_t maskIndices[1];
                maskIndices[0] = ~0;
                maskIndices[0] = maskIndices[0] << (topkCount_ % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64], neg,
                          maskIndices, 1, 1, 0);
            }
            if (topkCount_ / 64 * 64 + 64 < topkCountAlign256_) {
                PipeBarrier<PIPE_V>();
                Duplicate(scoreOutLocal_[topkCount_ / 64 * 64 + 64], zero,
                    topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[topkCount_ / 64 * 64 + 64], neg,
                    topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
            }

            PipeBarrier<PIPE_V>();
            uint32_t copyLen = topkCountAlign256_; // 32B对齐 topkCountAlign256_
            AscendC::DataCopy(ldIndexLocal_, indicesOutLocal_, copyLen);
            AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, copyLen);
        }
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        uint64_t indiceOutGmOffset = ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) *
                            constInfo_.kHeadNum * topkCount_;
        AscendC::DataCopyPad(indiceOutGm[indiceOutGmOffset],
                         indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        
        if (returnValueFlag) {
            PipeBarrier<PIPE_V>();
            liV2Vector1::UIntToFloatReturnValue(valueOutLocal_, scoreOutLocal_, topkCountAlign256_,
                constInfo_.NEG_INF_FLOAT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(valueOutGm[indiceOutGmOffset], valueOutLocal_, copyOutValueParams);
        }
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    }
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}
} // namespace LIV2Kernel
#endif
