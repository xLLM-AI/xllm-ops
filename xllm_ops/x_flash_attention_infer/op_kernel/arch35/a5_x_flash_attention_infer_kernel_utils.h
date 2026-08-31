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

/*!
 * \file a5_x_flash_attention_infer_kernel_utils.h
 * \brief
 */

#ifndef XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_KERNEL_UTILS_H
#define XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_KERNEL_UTILS_H

#include "catlass/catlass.hpp"
using namespace Catlass;
using namespace AscendC;

constexpr uint32_t CV_RATIO = 2;
constexpr uint32_t NUM2 = 2;
constexpr uint32_t KERNEL_TASK_NUM = 3;

template <typename T>
CATLASS_DEVICE T Min(T a, T b)
{
    return (a > b) ? b : a;
}

struct FAIKernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQSeqlen;
    GM_ADDR actualKvSeqlen;
    GM_ADDR o;
    GM_ADDR tiling;
    // Methods
    CATLASS_DEVICE
    FAIKernelParams()
    {}
    CATLASS_DEVICE
    FAIKernelParams(
        GM_ADDR q_, GM_ADDR k_, GM_ADDR v_, GM_ADDR mask_, GM_ADDR blockTables_, GM_ADDR actualQSeqlen_,
        GM_ADDR actualKvSeqlen_, GM_ADDR o_, GM_ADDR tiling_)
        : q(q_),
          k(k_),
          v(v_),
          mask(mask_),
          blockTables(blockTables_),
          actualQSeqlen(actualQSeqlen_),
          actualKvSeqlen(actualKvSeqlen_),
 o(o_),
          tiling(tiling_)
    {}
};

constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};

constexpr uint64_t MM2_RES_INTRA_EVENT[2] = {7, 8};  // mm2ResIntraEvent
constexpr uint64_t MM1_RES_INTRA_EVENT[2] = {9, 10}; // mm1ResIntraEvent

struct CubeCoordInfo {
    uint32_t curBIdx;
    uint32_t qSeqCoord;
    uint32_t kvSeqCoord;
};

struct RunParamStr { // 分核与切块需要使用到参数
    int64_t batchOuterIdx;
    int64_t qSeqOuterAxisIdx;
    int64_t kvHeadsOuterIdx;
    int64_t groupIdx;
    int32_t kvSeqLoopStartIdx;         /* kvSeq方向的循环控制信息 souter层确定 */
    int32_t kvSeqLoopEndIdx;           /* kvSeq方向的循环控制信息 souter层确定 */
    int64_t kvSeqAxisLineStartIdx = 0; /* kvSeq方向按行的起始位置 */
    int64_t kvSeqAxisLineEndIdx;       /* kvSeq方向按行的结束位置 */
    uint32_t qSeqRealSize;
    uint32_t halfQSeqRealSize;
    uint32_t firstHalfQSeqRealSize;
    int64_t actualQSeqSize;  /* Q的actualSeqLength */
    int64_t actualKvSeqSize; /* KV的actualSeqLength */
    int64_t qSeqLoopTimes;
};

struct RunInfo {
    int64_t kvSeqAxisStartIdx; /* kvSeq的起始位置*/
    int64_t kvSeqAxisEndIdx;
    int64_t kvSeqLoopCount; /* kvSeq循环当前的循环index */
    int64_t kvSeqLoopStartIdx;
    int64_t kvSeqLoopLimit;
    int64_t qSeqOuterAxisIdx = 0; /* qSeq轴的index */
    int64_t batchOuterIdx = 0;    /* b轴的index */
    int64_t kvHeadsOuterIdx = 0;  /* n2轴的index */
    int64_t groupIdx = 0;         /* g轴的index */
    int32_t qSeqRealSize;
    int32_t halfQSeqRealSize; /* vector侧实际的qSeq基本块大小，如果Cube基本块=128，那么halfQSeqRealSize=64 */
    int32_t
        firstHalfQSeqRealSize; /* 当qSeqRealSize不是2的整数倍时，v0比v1少计算一行，计算subblock偏移的时候需要使用v0的qSeq
                                  size */
    int32_t kvSeqRealSize; /* kvSeq方向基本块的真实长度 */
    int64_t taskId;
    int64_t multiCoreInnerIdx = 0;
    int64_t actualQSeqSize;  /* 非TND场景=总qSeqSize, Tnd场景下当前batch对应的qSeq */
    int64_t actualKvSeqSize; /* 非TND场景=总kvSeqSize, Tnd场景下当前batch对应的kvSeq */
    uint8_t taskIdMod2;
    uint8_t taskIdMod3;
    uint8_t multiCoreIdxMod2 = 0;
    uint8_t multiCoreIdxMod3 = 0;
    int64_t blockTableOffset;
};

struct ConstInfo {
    /* 全局的基本块信息 */
    uint32_t qSeqlenBase;
    uint32_t kvSeqlenBase;
    int64_t embed;
    int64_t groupSize; /* g轴的大小 */
    int64_t qHeads;
    int64_t kvHeads;
    int64_t qSeqlen;  /* qSeq总大小 */
    int64_t kvSeqlen; /* kvSeq总大小 */
    /* 轴的乘积 */
    int64_t qSeqlenOuterSize;
    uint8_t subBlockIdx;
    float scaleValue;
    /* 推理新增 */
    bool isActualLenDimsNull;    /* 判断是否有actualseq */
    bool isActualLenDimsKVNull;  /* 判断是否有actualseq_kv */
    uint32_t actualSeqLenSize;   /* 用户输入的actualseq的长度 */
    uint32_t actualSeqLenKVSize; /* 用户输入的actualseq_kv的长度 */
    /* service mm1 mm2 pageAttention */
    uint32_t blockTableDim2;
  uint32_t blockSize;
    uint32_t paBlockNumSum;
    /* G S不合轴场景，外层循环是B、N2、G，内层循环S，headNumRatio = groupSize */
    uint32_t headNumRatio;
    uint32_t bnAxisStartIdx;
    uint32_t bnAxisEndIdx;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    bool isActualSeqLengthsNull;
    bool isActualSeqLengthsKVNull;
    /* base params */
    uint32_t batch;
    /* special params */
    uint32_t attenMaskQSeqlen;
    uint32_t attenMaskKvSeqlen;
    /* core params */
    volatile int64_t multiCoreInnerOffset; /* 二次赋值的变量需要volatile修饰 */
    volatile int64_t multiCoreInnerLimit;  /* 二次赋值的变量需要volatile修饰 */
    uint32_t coreNum;
};

struct AttenMaskInfo {
    int64_t attenMaskShapeType;
    int64_t attenMaskQSeqlen;
    int64_t attenMaskKvSeqlen;
    int64_t attenMaskOffsetPre;
};

constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;
CATLASS_DEVICE constexpr uint16_t Align64Func(uint16_t data)
{
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}
CATLASS_DEVICE constexpr uint16_t Align(uint16_t data, uint16_t baseSize)
{
    return (data - 1) / baseSize * baseSize + baseSize;
}

CATLASS_DEVICE void ComputeParamBatch(
    RunParamStr& runParam, const ConstInfo& constInfo, const AttenMaskInfo& attenMaskInfo)
{
    runParam.actualQSeqSize = constInfo.qSeqlen;
    ;
    runParam.actualKvSeqSize = constInfo.kvSeqlen;
    ;
}

template <uint32_t qSeqlenTemplateType>
CATLASS_DEVICE void ComputeQseqLoopInfo(
    RunParamStr& runParam, const ConstInfo& constInfo, bool lastBN, int64_t nextQSeqAxisIdx)
{
    constexpr int32_t qSeqlenBase = static_cast<int32_t>(qSeqlenTemplateType);
    int32_t qSeqLoopTimes = CeilDiv(runParam.actualQSeqSize, qSeqlenBase);
    // 不是最后一个bn, 赋值souterBlockNum
    if (!lastBN) {
        runParam.qSeqLoopTimes = qSeqLoopTimes;
    } else { // 最后一个bn, 从数组下一个元素取值
        runParam.qSeqLoopTimes = nextQSeqAxisIdx == 0 ? qSeqLoopTimes : nextQSeqAxisIdx;
    }
}

template <uint32_t qSeqlenTemplateType>
CATLASS_DEVICE void ComputeParamQSeq(RunParamStr& runParam, const ConstInfo& constInfo, uint32_t sOuterLoopIdx)
{
    int64_t cubeSOuterOffset = sOuterLoopIdx * (uint32_t)qSeqlenTemplateType;
    if (runParam.actualQSeqSize == 0) {
        runParam.qSeqRealSize = 0;
    } else {
        runParam.qSeqRealSize =
            Min((uint32_t)qSeqlenTemplateType, (uint32_t)(runParam.actualQSeqSize - cubeSOuterOffset));
    }

    runParam.halfQSeqRealSize = (runParam.qSeqRealSize + 1) >> 1;
    runParam.firstHalfQSeqRealSize = runParam.halfQSeqRealSize;
    if (constInfo.subBlockIdx == 1) {
        runParam.halfQSeqRealSize = runParam.qSeqRealSize - runParam.halfQSeqRealSize;
    }
}

template <uint32_t kvSeqlenTemplateType>
CATLASS_DEVICE void ComputeKvSeqLoopInfo(RunParamStr& runParam, const ConstInfo& constInfo)
{
    constexpr int32_t kvSeqlenBase = static_cast<int32_t>(kvSeqlenTemplateType);
    runParam.kvSeqAxisLineStartIdx = 0;
    runParam.kvSeqAxisLineEndIdx = runParam.actualKvSeqSize;
    runParam.kvSeqLoopStartIdx = 0;
    runParam.kvSeqLoopEndIdx = (runParam.kvSeqAxisLineEndIdx + kvSeqlenBase - 1) / kvSeqlenBase;
}

#endif // XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_KERNEL_UTILS_H