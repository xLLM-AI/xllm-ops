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

#ifndef XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_TILING_DATA_DEF_H
#define XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_TILING_DATA_DEF_H

constexpr uint32_t MAX_CORE_NUM = 64;

class InputParamsRegbase {
public:
    int64_t batch;
    int64_t qHeads;
    int64_t kvHeads;
    int64_t groupSize;
    int64_t qSeqlen;
    int64_t kvSeqlen;
    int64_t embed;
    float scaleValue;
    uint8_t attenMaskCompressMode; // SPARSE_MODE_NO_MASK: 0, SPARSE_MODE_LEFT_UP: 1, SPARSE_MODE_RIGHT_DOWN : 2

    // PFA
    uint8_t isActualSeqLengthsNull;
    uint8_t isActualSeqLengthsKVNull;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;

    uint32_t headNumRatio;
    uint32_t blockSize;
    uint32_t blockTableDim2;
    uint32_t paBlockNumSum;
    uint32_t attenMaskQSeqlen;
    uint32_t attenMaskKvSeqlen;
};

class MultiCoreParamsRegbase {
public:
    int32_t coreNum;
    int64_t totalSize;
    int64_t qSeqlenOuterSize;
    int64_t splitFactorSize;
    int64_t splitFactorTailSize;
    uint32_t bnAxisStartIdx[MAX_CORE_NUM];
    int64_t sparseStartIdx[MAX_CORE_NUM];
};

class FATilingData {
public:
    InputParamsRegbase inputParamsRegbase;
    MultiCoreParamsRegbase multiCoreParamsRegbase;
};
#endif // XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_TILING_DATA_DEF_H