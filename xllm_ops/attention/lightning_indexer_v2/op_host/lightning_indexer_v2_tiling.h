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
 * \file lightning_indexer_v2_tiling.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_V2_TILING_H_
#define LIGHTNING_INDEXER_V2_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "../../../../common/tiling/tiling_util.h"
#include "lightning_indexer_v2_error_log.h"

namespace optiling {
// ------------------公共定义--------------------------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

enum class DataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t WEIGTHS_INDEX = 2;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 3;
constexpr uint32_t CU_SEQLENS_K_INDEX = 4;
constexpr uint32_t SEQUSED_Q_INDEX = 5;
constexpr uint32_t SEQUSED_K_INDEX = 6;
constexpr uint32_t CMP_RESIDUAL_K_INDEX = 7;
constexpr uint32_t BLOCK_TABLE_INDEX = 8;
constexpr uint32_t OUTPUT_IDX_OFFSET_INDEX = 9;
constexpr uint32_t METADATA_INDEX = 10;
// Outputs Index
constexpr uint32_t LIGHTNING_INDEXER = 0;
constexpr uint32_t LIGHTNING_VALUES = 1;
// Attributes Index
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 4;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 5;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;
// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
// 入参限制常量
constexpr uint32_t HEAD_DIM_LIMIT = 128;
constexpr uint32_t SPARSE_2K = 2048;
constexpr uint32_t SPARSE_MODE_LOWER = 3;
constexpr uint32_t TOPK_MAX = 8192;
constexpr uint32_t TOPK_MULTIPLE = 1024;
constexpr uint32_t G_SIZE_LIMIT = 64;
constexpr uint32_t METADATA_LIMIT = 1024;

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(LIV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, topk)
TILING_DATA_FIELD_DEF(uint32_t, maxSeqlenQ)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, maskMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(int64_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint32_t, batchSupperFlag)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningIndexerV2, LIV2TilingData)

// -----------算子CompileInfo定义-------------------
struct LIV2CompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct LiV2ParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo weights = {nullptr, nullptr};
    TilingOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    TilingOptionalParaInfo cuSeqlensK = {nullptr, nullptr};
    TilingOptionalParaInfo sequsedQ = {nullptr, nullptr};
    TilingOptionalParaInfo sequsedK = {nullptr, nullptr};
    TilingOptionalParaInfo cmpResidualK = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingOptionalParaInfo outputIdxOffset = {nullptr, nullptr};
    TilingOptionalParaInfo metadata = {nullptr, nullptr};
    TilingRequiredParaInfo attenOut = {nullptr, nullptr};
    TilingRequiredParaInfo valuesOut = {nullptr, nullptr};

    const char *layOut = nullptr;
    const char *layOutKey = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *maskMode = nullptr;
    const int32_t *topk = nullptr;
    const int32_t *maxSeqlenQ = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
    const int64_t *cmpRatio = nullptr;
    const int32_t *returnValue = nullptr;
};

// -----------算子Tiling入参信息类---------------
class LIV2TilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    LiV2ParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t maskMode = 0;
    // Others Flag
    uint32_t topk = 0;
    int32_t maxSeqlenQ = -1;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    int64_t cmpRatio = 1;
    uint32_t batchSupperFlag = 0;
    uint32_t returnValue = 0;
    uint32_t keyStride0 = 0;
    std::vector<uint32_t> keyStridesVec;

    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::PA_BBND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class LIV2InfoParser {
public:
    explicit LIV2InfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~LIV2InfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName) const;
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatchQbsnd();
    ge::graphStatus ValidateInputShapesMatchQtnd();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetActualSeqInfo();
    ge::graphStatus CheckKeyContiguous() const;
    void GenerateInfo(LIV2TilingInfo &liInfo);
    ge::graphStatus ParseAndCheck(LIV2TilingInfo &liInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_ = nullptr;
    fe::PlatFormInfos *platformInfo_ = nullptr;
    LiV2ParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t batchSupperFlag_ = 0;
    // Layout
    DataLayout qLayout_ = DataLayout::BSND;
    DataLayout kLayout_ = DataLayout::PA_BBND;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType weightsType_ = ge::DT_FLOAT16;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
    ge::DataType valuesOutType_ = ge::DT_FLOAT16;
    std::vector<uint32_t> keyStridesVec_;
    std::vector<uint32_t> keyDequantScaleStridesVec_;
};

// ---------------算子Tiling类---------------
class LightningIndexerV2Tiling {
public:
    explicit LightningIndexerV2Tiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus DoTiling(LIV2TilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    LIV2TilingData tilingData_;
};
} // namespace optiling
#endif // LIGHTNING_INDEXER_V2_TILING_H_
