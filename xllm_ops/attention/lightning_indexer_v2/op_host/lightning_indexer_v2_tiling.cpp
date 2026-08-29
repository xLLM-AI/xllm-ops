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
 * \file lightning_indexer_v2_tiling.cpp
 * \brief
 */

#include "lightning_indexer_v2_tiling.h"
#include "../op_kernel/lightning_indexer_v2_template_tiling_key.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
namespace optiling {
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

static const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},           // Used to indicate a DataType field has not been set.
    {ge::DT_FLOAT, "DT_FLOAT"},                   // float type
    {ge::DT_FLOAT16, "DT_FLOAT16"},               // fp16 type
    {ge::DT_INT8, "DT_INT8"},                     // int8 type
    {ge::DT_INT16, "DT_INT16"},                   // int16 type
    {ge::DT_UINT16, "DT_UINT16"},                 // uint16 type
    {ge::DT_UINT8, "DT_UINT8"},                   // uint8 type
    {ge::DT_INT32, "DT_INT32"},                   // uint32 type
    {ge::DT_INT64, "DT_INT64"},                   // int64 type
    {ge::DT_UINT32, "DT_UINT32"},                 // unsigned int32
    {ge::DT_UINT64, "DT_UINT64"},                 // unsigned int64
    {ge::DT_BOOL, "DT_BOOL"},                     // bool type
    {ge::DT_DOUBLE, "DT_DOUBLE"},                 // double type
    {ge::DT_DUAL, "DT_DUAL"},                     // dual output type
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},   // dual output int8 type
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"}, // dual output uint8 type
    {ge::DT_COMPLEX32, "DT_COMPLEX32"},           // complex32 type
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},           // complex64 type
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},         // complex128 type
    {ge::DT_QINT8, "DT_QINT8"},                   // qint8 type
    {ge::DT_QINT16, "DT_QINT16"},                 // qint16 type
    {ge::DT_QINT32, "DT_QINT32"},                 // qint32 type
    {ge::DT_QUINT8, "DT_QUINT8"},                 // quint8 type
    {ge::DT_QUINT16, "DT_QUINT16"},               // quint16 type
    {ge::DT_RESOURCE, "DT_RESOURCE"},             // resource type
    {ge::DT_STRING_REF, "DT_STRING_REF"},         // string ref type
    {ge::DT_STRING, "DT_STRING"},                 // string type
    {ge::DT_VARIANT, "DT_VARIANT"},               // dt_variant type
    {ge::DT_BF16, "DT_BFLOAT16"},                 // dt_bfloat16 type
    {ge::DT_INT4, "DT_INT4"},                     // dt_variant type
    {ge::DT_UINT1, "DT_UINT1"},                   // dt_variant type
    {ge::DT_INT2, "DT_INT2"},                     // dt_variant type
    {ge::DT_UINT2, "DT_UINT2"}                    // dt_variant type
};

std::string LIV2DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("LIV2DataTypeToSerialString ", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

static std::vector<int64_t> ToVector(const gert::Shape &shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

static std::string ToStringRaw(const gert::Shape &shape)
{
    std::ostringstream oss;
    auto v = ToVector(shape);
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    return oss.str();
}

// --------------------------LIV2InfoParser类成员函数定义-------------------------------------
ge::graphStatus LIV2InfoParser::CheckRequiredInOutExistence() const
{
    OP_CHECK_IF(opParamInfo_.query.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "q", "The shape of q is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.query.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "q", "The desc of q is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "k", "The shape of k is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.key.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "k", "The desc of k is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "w", "The shape of w is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(opParamInfo_.weights.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "w", "The desc of w is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.shape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The shape of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        opParamInfo_.attenOut.desc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "sparse_indices", "The desc of sparse_indices is nullptr"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(opParamInfo_.layOut == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_q", "Layout_q is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.layOutKey == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "layout_k", "Layout_k is nullptr"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.topk == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "topk", "Topk is nullptr"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.maskMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "mask_mode", "Mask_mode is nullptr"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("LightningIndexerV2", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo_ == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(aicNum == 0 || aivNum == 0, OP_LOGE(opName_, "num of core obtained is 0."), return GRAPH_FAILED);

    socVersion_ = ascendcPlatform.GetSocVersion();
    npuArch_ = ascendcPlatform.GetCurNpuArch();
    if ((npuArch_ != NpuArch::DAV_2201) && (npuArch_ != NpuArch::DAV_3510)) {
        OP_LOGE(opName_, "NpuArch[%d] is not support.", static_cast<int32_t>(npuArch_));
        return GRAPH_FAILED;
    }
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr, OP_LOGE(opName_, "workSpaceSize got from GE is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void LIV2InfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.cuSeqlensQ.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensQ.desc = context_->GetOptionalInputDesc(CU_SEQLENS_Q_INDEX);
    opParamInfo_.cuSeqlensK.tensor = context_->GetOptionalInputTensor(CU_SEQLENS_K_INDEX);
    opParamInfo_.cuSeqlensK.desc = context_->GetOptionalInputDesc(CU_SEQLENS_K_INDEX);
    opParamInfo_.sequsedQ.tensor = context_->GetOptionalInputTensor(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedQ.desc = context_->GetOptionalInputDesc(SEQUSED_Q_INDEX);
    opParamInfo_.sequsedK.tensor = context_->GetOptionalInputTensor(SEQUSED_K_INDEX);
    opParamInfo_.sequsedK.desc = context_->GetOptionalInputDesc(SEQUSED_K_INDEX);
    opParamInfo_.cmpResidualK.tensor = context_->GetOptionalInputTensor(CMP_RESIDUAL_K_INDEX);
    opParamInfo_.cmpResidualK.desc = context_->GetOptionalInputDesc(CMP_RESIDUAL_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
    opParamInfo_.outputIdxOffset.tensor = context_->GetOptionalInputTensor(OUTPUT_IDX_OFFSET_INDEX);
    opParamInfo_.outputIdxOffset.desc = context_->GetOptionalInputDesc(OUTPUT_IDX_OFFSET_INDEX);
    opParamInfo_.metadata.tensor = context_->GetOptionalInputTensor(METADATA_INDEX);
    opParamInfo_.metadata.desc = context_->GetOptionalInputDesc(METADATA_INDEX);
}

void LIV2InfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.weights.desc = context_->GetInputDesc(WEIGTHS_INDEX);
    opParamInfo_.weights.shape = context_->GetInputShape(WEIGTHS_INDEX);
    GetOptionalInputParaInfo();
}

void LIV2InfoParser::GetOutputParaInfo()
{
    opParamInfo_.attenOut.desc = context_->GetOutputDesc(LIGHTNING_INDEXER);
    opParamInfo_.attenOut.shape = context_->GetOutputShape(LIGHTNING_INDEXER);
    opParamInfo_.valuesOut.desc = context_->GetOutputDesc(LIGHTNING_VALUES);
    opParamInfo_.valuesOut.shape = context_->GetOutputShape(LIGHTNING_VALUES);
}

ge::graphStatus LIV2InfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.maxSeqlenQ = attrs->GetAttrPointer<int32_t>(ATTR_MAX_SEQLEN_Q_INDEX);
    opParamInfo_.layOut = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    opParamInfo_.topk = attrs->GetAttrPointer<int32_t>(ATTR_TOPK_INDEX);
    opParamInfo_.maskMode = attrs->GetAttrPointer<int32_t>(ATTR_MASK_MODE_INDEX);
    opParamInfo_.cmpRatio = attrs->GetAttrPointer<int64_t>(ATTR_CMP_RATIO_INDEX);
    opParamInfo_.returnValue = attrs->GetAttrPointer<int32_t>(ATTR_RETURN_VALUE_INDEX);

    auto keyStrides = context_->GetDynamicInputStride(KEY_INDEX, 0);
    if (keyStrides != nullptr && keyStrides->GetDimNum() > 0) {
        for (size_t i = 0; i < keyStrides->GetDimNum(); i++) {
            keyStridesVec_.push_back(keyStrides->GetStride(i));
        }
    }

    if (opParamInfo_.layOut != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_q is:%s", opParamInfo_.layOut);
    }
    if (opParamInfo_.layOutKey != nullptr) {
        OP_LOGI(context_->GetNodeName(), "layout_k is:%s", opParamInfo_.layOutKey);
    }
    if (opParamInfo_.topk != nullptr) {
        OP_LOGI(context_->GetNodeName(), "topk is:%d", *opParamInfo_.topk);
    }
    if (opParamInfo_.maxSeqlenQ != nullptr) {
        OP_LOGI(context_->GetNodeName(), "maxSeqlenQ is:%d", *opParamInfo_.maxSeqlenQ);
    }
    if (opParamInfo_.maskMode != nullptr) {
        OP_LOGI(context_->GetNodeName(), "mask mode is:%d", *opParamInfo_.maskMode);
    }
    if (opParamInfo_.cmpRatio != nullptr) {
        OP_LOGI(context_->GetNodeName(), "cmpRatio is:%lld", *opParamInfo_.cmpRatio);
    }
    if (opParamInfo_.returnValue != nullptr) {
        OP_LOGI(context_->GetNodeName(), "return value is:%d", *opParamInfo_.returnValue);
    }
    OP_LOGI(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");
    OP_CHECK_IF(((std::string(opParamInfo_.layOutKey) != "PA_BBND") &&
                 (std::string(opParamInfo_.layOut) != std::string(opParamInfo_.layOutKey))),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName_, "layout_q and layout_k",
                    std::string(opParamInfo_.layOut) + " and " + std::string(opParamInfo_.layOutKey),
                    "When layout_k is non-PA, layout_q and layout_k must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((std::string(opParamInfo_.layOutKey) != "PA_BBND") &&
                 (std::string(opParamInfo_.layOutKey) != "BSND") && (std::string(opParamInfo_.layOutKey) != "TND")),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_k", std::string(opParamInfo_.layOutKey).c_str(),
                                                      "Layout_k only supports PA_BBND, BSND or TND"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((std::string(opParamInfo_.layOut) != "BSND") && (std::string(opParamInfo_.layOut) != "TND")),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "layout_q", std::string(opParamInfo_.layOut).c_str(),
                                                      "Layout_q only supports BSND or TND"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(((std::string(opParamInfo_.layOutKey) != "PA_BBND") &&
                     (std::string(opParamInfo_.layOut)) != (std::string(opParamInfo_.layOutKey))),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName_, "layout_q and layout_k",
                        std::string(opParamInfo_.layOut) + " and " + std::string(opParamInfo_.layOutKey),
                        "Outside of PA, layout_q and layout_k must be the same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (*opParamInfo_.maxSeqlenQ < -1),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, "max_seqlen_q", std::to_string(*opParamInfo_.maxSeqlenQ).c_str(), "Max_seqlen_q must >= -1"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((*opParamInfo_.returnValue != 0) && (*opParamInfo_.returnValue != 1),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "return_value",
                                                          std::to_string(*opParamInfo_.returnValue).c_str(),
                                                          "Return_value only supports 0 or 1"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((!((*opParamInfo_.topk > 0) && (*opParamInfo_.topk <= TOPK_MAX))),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "topk", std::to_string(*opParamInfo_.topk),
                                                      "Topk must > 0 and <= 8192"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_2201) {
        OP_CHECK_IF(*opParamInfo_.topk > SPARSE_2K && *opParamInfo_.topk % TOPK_MULTIPLE != 0,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        opName_, "topk", std::to_string(*opParamInfo_.topk),
                        "input attr topk > 2048 must be an integer multiple of 1024 on 910B/C"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(!((*opParamInfo_.maskMode == 0) || (*opParamInfo_.maskMode == SPARSE_MODE_LOWER)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "mask_mode", std::to_string(*opParamInfo_.maskMode),
                                                      "Mask_mode only supported 0 or 3"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((*opParamInfo_.cmpRatio <= 0) || (*opParamInfo_.cmpRatio > 128),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "cmp_ratio", std::to_string(*opParamInfo_.cmpRatio),
                                                      "Cmp_ratio must > 0 and <= 128"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    weightsType_ = opParamInfo_.weights.desc->GetDataType();
    outputType_ = opParamInfo_.attenOut.desc->GetDataType();
    valuesOutType_ = opParamInfo_.valuesOut.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_);
    OP_CHECK_IF(!inDTypeAllEqual,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "q and k",
                    LIV2DataTypeToSerialString(inputQType_) + " and " + LIV2DataTypeToSerialString(inputKType_),
                    "The dtype of q and k must be same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(((inputQType_ != ge::DT_FLOAT16) && (inputQType_ != ge::DT_BF16)),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    opName_, "q and k",
                    LIV2DataTypeToSerialString(inputQType_) + " and " + LIV2DataTypeToSerialString(inputKType_),
                    "The dtype of q and k must be float16 or bfloat16"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((weightsType_ != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "w", LIV2DataTypeToSerialString(weightsType_).c_str(),
                                                      "The dtype of w must be float32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(outputType_ != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_indices",
                                                      LIV2DataTypeToSerialString(outputType_).c_str(),
                                                      "The dtype of sparse_indices must be int32"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(valuesOutType_ != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "sparse_values",
                                                      LIV2DataTypeToSerialString(valuesOutType_).c_str(),
                                                      "The dtype of sparse_values must be float32"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND}, {"TND", DataLayout::TND}, {"PA_BBND", DataLayout::PA_BBND}};

    std::string layout(opParamInfo_.layOut);
    auto it = layoutMap.find(layout);
    if (it != layoutMap.end()) {
        qLayout_ = it->second;
    }

    std::string layoutKey(opParamInfo_.layOutKey);
    auto itKey = layoutMap.find(layoutKey);
    if (itKey != layoutMap.end()) {
        kLayout_ = itKey->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::PA_BBND) {
        OP_CHECK_IF(opParamInfo_.blockTable.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "block_table",
                                                             "When layout_k is PA_BSND, block_table must not be null"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(opParamInfo_.sequsedK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "seqused_k",
                                                             "When layout_k is PA_BSND, seqused_k must not be null"),
                    return ge::GRAPH_FAILED);
        if (npuArch_ == NpuArch::DAV_3510) {
            OP_CHECK_IF(opParamInfo_.cuSeqlensK.tensor != nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            opName_, "cu_seqlens_k", "When layout_k is PA_BBND, cu_seqlens_k must not be provided"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "block_table",
                            LIV2DataTypeToSerialString(opParamInfo_.blockTable.desc->GetDataType()).c_str(),
                            "The dtype of block_table only supports int32"),
                        return ge::GRAPH_FAILED);
        }
        OP_CHECK_IF(
            opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName_, "block_table", LIV2DataTypeToSerialString(opParamInfo_.blockTable.desc->GetDataType()).c_str(),
                "The dtype of block_table only supports int32"),
            return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.cuSeqlensK.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_k",
                                                             "When layout_k is TND, cu_seqlens_k must not be null"),
                    return ge::GRAPH_FAILED);
        if (npuArch_ == NpuArch::DAV_3510) {
            OP_CHECK_IF(opParamInfo_.cuSeqlensK.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "cu_seqlens_k",
                            LIV2DataTypeToSerialString(opParamInfo_.cuSeqlensK.desc->GetDataType()).c_str(),
                            "The dtype of cu_seqlens_k only supports int32"),
                        return ge::GRAPH_FAILED);
            // seqused_k 可选 - 仅校验数据类型
            if (opParamInfo_.sequsedK.tensor != nullptr) {
                OP_CHECK_IF(opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                opName_, "seqused_k",
                                LIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                                "The dtype of seqused_k only supports int32"),
                            return ge::GRAPH_FAILED);
            }
        }
    } else {
        // BSND: cu_seqlens_k 不传, seqused_k 可选
        OP_CHECK_IF(opParamInfo_.cuSeqlensK.tensor != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_k",
                                                             "When layout_k is BSND, cu_seqlens_k must not be null"),
                    return ge::GRAPH_FAILED);
        if (opParamInfo_.sequsedK.tensor != nullptr) {
            OP_CHECK_IF(
                opParamInfo_.sequsedK.desc->GetDataType() != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName_, "seqused_k", LIV2DataTypeToSerialString(opParamInfo_.sequsedK.desc->GetDataType()).c_str(),
                    "The dtype of seqused_k only supports int32"),
                return ge::GRAPH_FAILED);
        }
    }
    // =============== cmpResidualK 校验 ===============
    // cmpRatio 不等于 1 且 maskMode 不等于 0 时 cmpResidualK 必传
    if (npuArch_ == NpuArch::DAV_3510) {
        if (opParamInfo_.cmpRatio != nullptr && *opParamInfo_.cmpRatio != 1 && opParamInfo_.maskMode != nullptr &&
            *opParamInfo_.maskMode != 0) {
            OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor == nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            opName_, "cmp_residual_k",
                            "Cmp_ratio is not 1 and mask_mode is not 0, cmp_residual_k must not be null"),
                        return ge::GRAPH_FAILED);
            // cmpResidualK 传入时校验维度 & 数据类型
            if (qLayout_ == DataLayout::BSND) {
                OP_CHECK_IF(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO) !=
                                opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize(),
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                opName_, "cmp_residual_k",
                                ToStringRaw(opParamInfo_.cmpResidualK.tensor->GetStorageShape()).c_str(),
                                "When layout_q is BSND, the shape of cmp_residual_k must be (B,)"),
                            return ge::GRAPH_FAILED);
            } else if (qLayout_ == DataLayout::TND) {
                OP_CHECK_IF(opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize() !=
                                opParamInfo_.cuSeqlensQ.tensor->GetStorageShape().GetShapeSize() - 1,
                            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                                opName_, "cmp_residual_k",
                                std::to_string(opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize()),
                                "When layout_q is TND, the shape size of cmp_residual_k "
                                "must equal the shape size - 1 of cu_seqlens_q"),
                            return ge::GRAPH_FAILED);
            }
            if (opParamInfo_.cmpResidualK.tensor != nullptr) {
                OP_CHECK_IF(opParamInfo_.cmpResidualK.desc->GetDataType() != ge::DT_INT32,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                opName_, "cmp_residual_k",
                                LIV2DataTypeToSerialString(opParamInfo_.cmpResidualK.desc->GetDataType()).c_str(),
                                "The dtype of cmp_residual_k supports int32"),
                            return ge::GRAPH_FAILED);
            }
        } else {
            OP_CHECK_IF(
                opParamInfo_.cmpResidualK.tensor != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName_, "cmp_residual_k", "Cmp_ratio is 1 or sparse_mode is 0, cmp_residual_k must be null"),
                return ge::GRAPH_FAILED);
        }
    }
    if (qLayout_ == DataLayout::TND) {
        OP_CHECK_IF(opParamInfo_.cuSeqlensQ.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "cu_seqlens_q",
                                                             "When layout_q is TND, cu_seqlens_q must not be null"),
                    return ge::GRAPH_FAILED);
        // cuSeqlensQ 维度 & 类型校验
        if (npuArch_ == NpuArch::DAV_3510) {
            if (kLayout_ == DataLayout::PA_BBND) {
                // k为PA_BBND必传sequsedK, 用sequsedK的维度校验
                OP_CHECK_IF(opParamInfo_.cuSeqlensQ.tensor->GetStorageShape().GetShapeSize() !=
                                opParamInfo_.sequsedK.tensor->GetStorageShape().GetShapeSize() + 1,
                            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                                opName_, "cmp_residual_k",
                                std::to_string(opParamInfo_.cmpResidualK.tensor->GetStorageShape().GetShapeSize()),
                                "When layout_q is TND and layout_k is PA_BBND, "
                                "the shape size of cu_seqlens_q must equal the shape size + 1 of seqused_k"),
                            return ge::GRAPH_FAILED);
            } else if (kLayout_ == DataLayout::TND) {
                // q、k都为TND, cuSeqlensQ与cuSeqlensK维度一致校验
                OP_CHECK_IF(opParamInfo_.cuSeqlensQ.tensor->GetStorageShape().GetShapeSize() !=
                                opParamInfo_.cuSeqlensK.tensor->GetStorageShape().GetShapeSize(),
                            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                opName_, "cu_seqlens_q and cu_seqlens_k",
                                Ops::Base::ToString(opParamInfo_.cuSeqlensQ.tensor->GetStorageShape()) + " and " +
                                    Ops::Base::ToString(opParamInfo_.cuSeqlensK.tensor->GetStorageShape()),
                                "When layout_q is TND and layout_k is TND, "
                                "the shape of cu_seqlens_q must equal the shape of cu_seqlens_k"),
                            return ge::GRAPH_FAILED);
            }
            OP_CHECK_IF(opParamInfo_.cuSeqlensQ.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "cu_seqlens_q",
                            LIV2DataTypeToSerialString(opParamInfo_.cuSeqlensK.desc->GetDataType()).c_str(),
                            "The dtype of cu_seqlens_q only supports int32"),
                        return ge::GRAPH_FAILED);
            // seqused_q 可选 - 仅校验数据类型
            if (opParamInfo_.sequsedQ.tensor != nullptr) {
                OP_CHECK_IF(opParamInfo_.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                opName_, "seqused_q",
                                LIV2DataTypeToSerialString(opParamInfo_.sequsedQ.desc->GetDataType()).c_str(),
                                "The dtype of seqused_q only supports int32"),
                            return ge::GRAPH_FAILED);
            }
        }
    } else {
        if (npuArch_ == NpuArch::DAV_3510) {
            // BSND: cu_seqlens_q 不传, seqused_q 可选
            OP_CHECK_IF(opParamInfo_.cuSeqlensQ.tensor != nullptr,
                        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                            opName_, "cu_seqlens_q", "When layout_q is BSND, cu_seqlens_q must not be provided"),
                        return ge::GRAPH_FAILED);
            if (opParamInfo_.sequsedQ.tensor != nullptr) {
                OP_CHECK_IF(opParamInfo_.sequsedQ.desc->GetDataType() != ge::DT_INT32,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                opName_, "seqused_q",
                                LIV2DataTypeToSerialString(opParamInfo_.sequsedQ.desc->GetDataType()).c_str(),
                                "The dtype of seqused_q only supports int32"),
                            return ge::GRAPH_FAILED);
            }
        }
    }

    if (npuArch_ == NpuArch::DAV_3510) {
        // outputIdxOffset数据类型校验
        if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
            OP_CHECK_IF(opParamInfo_.outputIdxOffset.desc->GetDataType() != ge::DT_INT32,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            opName_, "output_idx_offset",
                            LIV2DataTypeToSerialString(opParamInfo_.outputIdxOffset.desc->GetDataType()).c_str(),
                            "The dtype of output_idx_offset only supports int32"),
                        return ge::GRAPH_FAILED);
        }
        // metadata 必传
        OP_CHECK_IF(opParamInfo_.metadata.tensor == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "metadata", "Metadata must not be null"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(kLayout_ != DataLayout::PA_BBND && opParamInfo_.blockTable.tensor != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "block_table",
                                                         "When layout_k is not PA_BBND, block_table must be null"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::CheckShapeDim()
{
    OP_CHECK_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                    (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
                OP_LOGE_FOR_INVALID_SHAPEDIM(
                    opName_, "block_table",
                    std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum()).c_str(), "2"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(
            ((kLayout_ == DataLayout::PA_BBND) || (kLayout_ == DataLayout::BSND)) &&
                (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_FOUR),
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                opName_, "k", std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDimNum()).c_str(), "4"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (kLayout_ == DataLayout::TND) && (opParamInfo_.key.shape->GetStorageShape().GetDimNum() != DIM_NUM_THREE),
            OP_LOGE_FOR_INVALID_SHAPEDIM(
                opName_, "k", std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDimNum()).c_str(), "3"),
            return ge::GRAPH_FAILED);
    }
    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t weightsShapeDim = opParamInfo_.weights.shape->GetStorageShape().GetDimNum();
    uint32_t outShapeDim = opParamInfo_.attenOut.shape->GetStorageShape().GetDimNum();
    uint32_t qExpectShapeDim = DIM_NUM_FOUR;
    uint32_t kExpectShapeDim = DIM_NUM_FOUR;
    if (qLayout_ == DataLayout::TND) {
        qExpectShapeDim = DIM_NUM_THREE;
    }
    if (kLayout_ == DataLayout::TND) {
        kExpectShapeDim = DIM_NUM_THREE;
    }
    OP_CHECK_IF(kShapeDim != kExpectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "k", std::to_string(kShapeDim).c_str(),
                                             std::to_string(kExpectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qShapeDim != qExpectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "q", std::to_string(qShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
            uint32_t outputIdxOffsetShapeDim = opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDimNum();
            OP_CHECK_IF((outputIdxOffsetShapeDim != qExpectShapeDim - 1),
                        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "output_idx_offset",
                                                     std::to_string(outputIdxOffsetShapeDim).c_str(),
                                                     std::to_string(qExpectShapeDim - 1).c_str()),
                        return ge::GRAPH_FAILED);
        }
        if (*opParamInfo_.returnValue == 1) {
            uint32_t sparseValuesShapeDim = opParamInfo_.valuesOut.shape->GetStorageShape().GetDimNum();
            OP_CHECK_IF(
                sparseValuesShapeDim != qExpectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "sparse_values", std::to_string(sparseValuesShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
        }
    }
    OP_CHECK_IF(outShapeDim != qExpectShapeDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "sparse_indices", std::to_string(outShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim).c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!(weightsShapeDim == qExpectShapeDim - 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "w", std::to_string(weightsShapeDim).c_str(),
                                             std::to_string(qExpectShapeDim - 1).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// key非连续校验：通过shape计算expected stride进行校验
// PA_BBND时，只允许0轴非连续，其余轴必须连续
// 非PA_BBND时，所有轴都必须连续
ge::graphStatus LIV2InfoParser::CheckKeyContiguous() const
{
    bool keyNonContiguous = false;
    // PA_BBND: 0轴允许非连续，从1轴开始检查；非PA_BBND: 从0轴开始检查
    // PA_BBND: axis 0 allows non-contiguous, check starts from axis 1
    // Non-PA_BBND: check starts from axis 0
    size_t checkStartIdx = (kLayout_ == DataLayout::PA_BBND) ? 1 : 0;
    if (!keyStridesVec_.empty() && opParamInfo_.key.shape != nullptr) {
        auto &shape = opParamInfo_.key.shape->GetStorageShape();
        std::vector<uint32_t> expectedStrides;
        if (kLayout_ == DataLayout::BSND || kLayout_ == DataLayout::PA_BBND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2) * shape.GetDim(3), shape.GetDim(2) * shape.GetDim(3),
                               shape.GetDim(3), 1};
        } else if (kLayout_ == DataLayout::TND) {
            expectedStrides = {shape.GetDim(1) * shape.GetDim(2), shape.GetDim(2), 1};
        }
        for (size_t i = checkStartIdx; i < expectedStrides.size(); ++i) {
            if (i < keyStridesVec_.size() && keyStridesVec_[i] != expectedStrides[i]) {
                keyNonContiguous = true;
                break;
            }
        }
    }

    OP_CHECK_IF(
        keyNonContiguous,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName_, "k", "k only supports non-continuous keying on the 0-axis"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetN1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO));
    } else {
        // TND
        n1Size_ = static_cast<uint32_t>(opParamInfo_.query.shape->GetStorageShape().GetDim(1));
    }
    OP_LOGI(context_->GetNodeName(), "n1Size is %d", n1Size_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                    const std::string &actualSeqLenName) const
{
    size = static_cast<uint32_t>(tensor->GetShapeSize() - 1);
    if (size <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName_, actualSeqLenName.c_str(), std::to_string(size).c_str(),
            "The shape size of " + actualSeqLenName + " should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = (kLayout_ == DataLayout::TND) ? DIM_IDX_ONE : DIM_IDX_TWO;
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OP_LOGI(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    OP_CHECK_IF(n2Size_ != 1,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "k",
                                                      ToStringRaw(opParamInfo_.key.shape->GetStorageShape()).c_str(),
                                                      "The head num of k must be 1"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetGSize()
{
    if (n1Size_ % n2Size_ != 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "q and k",
                                               Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) +
                                                   " and " +
                                                   Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                                               "The head num of q must be a multiple of the head num of k");
    }
    gSize_ = n1Size_ / n2Size_;
    if (npuArch_ == NpuArch::DAV_3510) {
        OP_CHECK_IF(gSize_ > G_SIZE_LIMIT,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q and k",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                        "The value of (the head num of q divided by the head num of k) must <= 64"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetBatchSize()
{
    // 获取B基准值
    // 1、非TND时, 以query的batch_size维度为基准;
    // 2、Q和K都为TND时, cu_seqlens_q必须传入, 以cu_seqlens_q数组的长度为B轴大小
    // 3、Q为TND，K为PA_BBND时，以cu_seqlens_q数组的长度为B轴大小
    if (qLayout_ == DataLayout::BSND) {
        bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_ZERO);
        return ge::GRAPH_SUCCESS;
    } else {
        // TND
        uint32_t bSizeQuery;
        GetActualSeqLenSize(bSizeQuery, opParamInfo_.cuSeqlensQ.tensor, "input cu_seqlens_q");
        if (kLayout_ == DataLayout::TND) {
            uint32_t bSizeKey;
            GetActualSeqLenSize(bSizeKey, opParamInfo_.cuSeqlensK.tensor, "input cu_seqlens_k");
            OP_CHECK_IF(bSizeQuery != bSizeKey,
                        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                            opName_, "cu_seqlens_q and cu_seqlens_k",
                            std::to_string(bSizeQuery) + " and " + std::to_string(bSizeKey),
                            "The lengths of cu_seqlens_q and cu_seqlens_k must be same"),
                        return ge::GRAPH_FAILED);
        }
        bSize_ = bSizeQuery;
        return ge::GRAPH_SUCCESS;
    }
}

ge::graphStatus LIV2InfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    uint32_t dIndex = DIM_IDX_TWO;
    // 根据layout确定D维度在shape中的位置
    switch (qLayout_) {
        case DataLayout::TND:
            // TND格式: [Total, N, D] -> D是第2维(索引2)
            dIndex = DIM_IDX_TWO;
            break;
        case DataLayout::BSND:
            // BSND格式: [Batch, SeqLen, N, D] -> D是第3维(索引3)
            dIndex = DIM_IDX_THREE;
            break;
        default:
            OP_LOGE(opName_, "unsupported layout for getting head dim.");
            return ge::GRAPH_FAILED;
    }
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(dIndex);
    OP_CHECK_IF(
        headDim_ != HEAD_DIM_LIMIT,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "q", ToStringRaw(opParamInfo_.query.shape->GetStorageShape()),
                                              "The head num of q only supports 128"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetS1Size()
{
    if (qLayout_ == DataLayout::BSND) {
        s1Size_ = opParamInfo_.query.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetAndCheckBlockSize()
{
    blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    OP_LOGI(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OP_CHECK_IF(((blockSize_ % 16 != 0) || (blockSize_ == 0) || (blockSize_ > 1024)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "k", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()),
                    "The block_size of k must be a multiple of 16 and be within the range (0, 1024]"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OP_CHECK_IF(
        (blockCount_ == 0),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "k", ToStringRaw(opParamInfo_.key.shape->GetStorageShape()),
                                              "The block_count of k cannot be 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OP_LOGI(context_->GetNodeName(), "maxBlockNumPerBatch_ is %u, blockSize_ is %d, s2Size_ is %lld",
            maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::PA_BBND) {
        return GetS2SizeForPageAttention();
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(0);
    } else if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::ValidateInputShapesMatchQtnd()
{
    // -----------------------check T-------------------
    OP_CHECK_IF(
        (kLayout_ == DataLayout::PA_BBND) && ((opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_) ||
                                              (opParamInfo_.blockTable.tensor != nullptr &&
                                               opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "cu_seqlens_q, seqused_k and block_table",
            Ops::Base::ToString(opParamInfo_.cuSeqlensQ.tensor->GetStorageShape()) + ", " +
                Ops::Base::ToString(opParamInfo_.sequsedK.tensor->GetStorageShape()) + " and " +
                Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()),
            "TND case cu_seqlens_q, seqused_k, block_table dim 0 are " + std::to_string(bSize_) + ", " +
                std::to_string(opParamInfo_.sequsedK.tensor->GetShapeSize()) + ", " +
                std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) +
                " respectively, they must be same"),
        return ge::GRAPH_FAILED);

    uint32_t qTsize = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != qTsize) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != qTsize),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "q, w and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "TND case q, w and sparse_indices dim 0 are " + std::to_string(qTsize) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    if (npuArch_ == NpuArch::DAV_3510) {
        if (*opParamInfo_.returnValue == 1) {
            if (opParamInfo_.valuesOut.shape != nullptr) {
                OP_CHECK_IF((opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0) != qTsize),
                            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                opName_, "q and sparse_values",
                                Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                                    Ops::Base::ToString(opParamInfo_.valuesOut.shape->GetStorageShape()),
                                "TND case q and sparse_values dim 0 are " + std::to_string(qTsize) + ", " +
                                    std::to_string(opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(0)) +
                                    " respectively, they must be same"),
                            return ge::GRAPH_FAILED);
            }
        }
        if (opParamInfo_.outputIdxOffset.tensor != nullptr) {
            OP_CHECK_IF((opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDim(0) != qTsize),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            opName_, "q and output_idx_offset",
                            Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                                Ops::Base::ToString(opParamInfo_.outputIdxOffset.tensor->GetStorageShape()),
                            "TND case q and output_idx_offset dim 0 are " + std::to_string(qTsize) + " and " +
                                std::to_string(opParamInfo_.outputIdxOffset.tensor->GetStorageShape().GetDim(0)) +
                                " respectively, they must be same"),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::ValidateInputShapesMatchQbsnd()
{
    // -----------------------check BatchSize-------------------
    // bSize_ 来源于query
    if (kLayout_ == DataLayout::PA_BBND) {
        OP_CHECK_IF((opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_) ||
                        (opParamInfo_.sequsedK.tensor->GetShapeSize() != bSize_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q, seqused_k, block_table",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                            Ops::Base::ToString(opParamInfo_.sequsedK.tensor->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.blockTable.tensor->GetStorageShape()),
                        "BSND case q, seqused_k, block_table dim 0 are " + std::to_string(bSize_) + ", " +
                            std::to_string(opParamInfo_.sequsedK.tensor->GetShapeSize()) + ", " +
                            std::to_string(opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
    } else if (kLayout_ == DataLayout::BSND) {
        OP_CHECK_IF(opParamInfo_.key.shape->GetStorageShape().GetDim(0) != bSize_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "q and k",
                        Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                        "BSND case q, k dim 0 are " + std::to_string(bSize_) + ", " +
                            std::to_string(opParamInfo_.key.shape->GetStorageShape().GetDim(0)) +
                            " respectively, they must be same"),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(0) != bSize_) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0) != bSize_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "q, w and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "BSND case q, w and sparse_indices dim 0 are " + std::to_string(bSize_) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(0)) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(0)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check S1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(1) != s1Size_) ||
                    (opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1) != s1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "q, w and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + ", " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "BSND case q, w and sparse_indices dim 1 are " + std::to_string(s1Size_) + ", " +
                        std::to_string(opParamInfo_.weights.shape->GetStorageShape().GetDim(1)) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(1)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LIV2InfoParser::ValidateInputShapesMatch()
{
    /*
    TND:
    query [T,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [T,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize],
    out [T,N2,topk]
    ----------------------
    BSND:
    query [BatchSize,S1,N1,D],
    key [BlockNum,BlockSize,N2,D],
    weight [BatchSize,S1,N1],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize] 可选
    out [BatchSize,S1,N2,topk]
    */
    uint32_t queryWeightsN1Dim = 1;
    uint32_t outN2Dim = 1;
    if (qLayout_ == DataLayout::TND) {
        if (ValidateInputShapesMatchQtnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else { // qLayout_ BSND
        if (ValidateInputShapesMatchQbsnd() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        queryWeightsN1Dim = DIM_IDX_TWO;
        outN2Dim = DIM_IDX_TWO;
    }
    // -----------------------check N1-------------------
    OP_CHECK_IF((opParamInfo_.weights.shape->GetStorageShape().GetDim(queryWeightsN1Dim) != n1Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "q and w",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.weights.shape->GetStorageShape()),
                    "The head num of q and w must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check D-------------------
    uint32_t keyDDim = kLayout_ == DataLayout::TND ? DIM_IDX_TWO : DIM_IDX_THREE;
    OP_CHECK_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) != headDim_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "q and k",
                    Ops::Base::ToString(opParamInfo_.query.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()),
                    "The last dim of q and k shape must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check N2-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName_, "k and sparse_indices",
                    Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                        Ops::Base::ToString(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "The head num of k and sparse_indices are " + std::to_string(n2Size_) + ", " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim)) +
                        " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check cmp_residual_k-------------------
    if (npuArch_ == NpuArch::DAV_3510) {
        if (opParamInfo_.cmpResidualK.tensor != nullptr) {
            OP_CHECK_IF(
                (opParamInfo_.cmpResidualK.tensor->GetShapeSize() != bSize_),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName_, "cmp_residual_k", std::to_string(opParamInfo_.cmpResidualK.tensor->GetShapeSize()),
                    "The shape size of cmp_residual_k must be equal to batch_size (" + std::to_string(bSize_) + ")"),
                return ge::GRAPH_FAILED);
        }
    }
    // -----------------------check sparse_count-------------------
    OP_CHECK_IF((opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.topk),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName_, "sparse_indices", ToStringRaw(opParamInfo_.attenOut.shape->GetStorageShape()),
                    "The last dim of sparse_indices and sparse_count are " +
                        std::to_string(opParamInfo_.attenOut.shape->GetStorageShape().GetDim(outN2Dim + 1)) + ", " +
                        std::to_string(*opParamInfo_.topk) + " respectively, they must be same"),
                return ge::GRAPH_FAILED);
    // -----------------------check sparse_values------------------
    if (npuArch_ == NpuArch::DAV_3510 && *opParamInfo_.returnValue == 1) {
        OP_CHECK_IF((opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "k and sparse_values",
                        Ops::Base::ToString(opParamInfo_.key.shape->GetStorageShape()) + " and " +
                            Ops::Base::ToString(opParamInfo_.valuesOut.shape->GetStorageShape()),
                        "The head num of k and sparse_values must be same"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((opParamInfo_.valuesOut.shape->GetStorageShape().GetDim(outN2Dim + 1) != *opParamInfo_.topk),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName_, "topk and sparse_values",
                        std::to_string(*opParamInfo_.topk) + " and " +
                            Ops::Base::ToString(opParamInfo_.valuesOut.shape->GetStorageShape()),
                        "The last dim of sparse_values must be same as topk"),
                    return ge::GRAPH_FAILED);
    }
    // -----------------------check metadata-------------------
    OP_CHECK_IF(
        ((opParamInfo_.metadata.tensor != nullptr) && (opParamInfo_.metadata.tensor->GetShapeSize() != METADATA_LIMIT)),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "metadata",
                                                  std::to_string(opParamInfo_.metadata.tensor->GetShapeSize()).c_str(),
                                                  "The dim 0 of metadata must be " + std::to_string(METADATA_LIMIT)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void LIV2InfoParser::GenerateInfo(LIV2TilingInfo &liInfo)
{
    liInfo.opName = opName_;
    liInfo.platformInfo = platformInfo_;
    liInfo.opParamInfo = opParamInfo_;
    liInfo.socVersion = socVersion_;

    liInfo.bSize = bSize_;
    liInfo.n1Size = n1Size_;
    liInfo.n2Size = n2Size_;
    liInfo.s1Size = s1Size_;
    liInfo.s2Size = s2Size_;
    liInfo.gSize = gSize_;

    liInfo.inputQType = inputQType_;
    liInfo.inputKType = inputKType_;
    liInfo.outputType = outputType_;

    liInfo.blockSize = blockSize_;
    liInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    std::string layOutKeyStr(opParamInfo_.layOutKey);
    liInfo.pageAttentionFlag = layOutKeyStr == "PA_BBND" ? true : false;
    liInfo.batchSupperFlag = batchSupperFlag_;
    liInfo.maskMode = *opParamInfo_.maskMode;
    liInfo.topk = *opParamInfo_.topk;
    liInfo.maxSeqlenQ = *opParamInfo_.maxSeqlenQ;
    liInfo.cmpRatio = *opParamInfo_.cmpRatio;
    liInfo.returnValue = *opParamInfo_.returnValue;

    if (!keyStridesVec_.empty()) {
        liInfo.keyStride0 = static_cast<uint32_t>(keyStridesVec_[0]);
    } else {
        liInfo.keyStride0 = 0; // 非PA无需使用stride
    }

    liInfo.inputQLayout = qLayout_;
    liInfo.inputKLayout = kLayout_;
}

ge::graphStatus LIV2InfoParser::ParseAndCheck(LIV2TilingInfo &liInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() || ge::GRAPH_SUCCESS != GetN1Size() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size() || ge::GRAPH_SUCCESS != GetGSize()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetS1Size() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckKeyContiguous()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(liInfo);

    return ge::GRAPH_SUCCESS;
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForLightningIndexerV2(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------LightningIndexerV2Tiling类成员函数定义-----------------------
ge::graphStatus LightningIndexerV2Tiling::DoTiling(LIV2TilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;            // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 8192;          // TopK选取个数
    uint64_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_3510) {
        constexpr uint32_t li3510S1Base = 4;
        constexpr uint32_t li3510S2Base = 128;
        workspaceSize += li3510S1Base * ((tilingInfo->s2Size + li3510S2Base - 1) / li3510S2Base) * li3510S2Base *
                         sizeof(uint32_t) * aicNum;
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    } else {
        // 主流程需Workspace大小
        uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
        workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
        // Decode流程(LD)需要Workspace大小
        // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
        workspaceSize +=
            V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * TOPK_MAX_SIZE * V1_RES_ELEM_SIZE * aicNum;
        // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
        workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    }

    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_topk(tilingInfo->topk);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_maskMode(tilingInfo->maskMode);
    tilingData_.set_preTokens(tilingInfo->preTokens);
    tilingData_.set_nextTokens(tilingInfo->nextTokens);
    tilingData_.set_cmpRatio(tilingInfo->cmpRatio);
    tilingData_.set_keyStride0(tilingInfo->keyStride0);
    tilingData_.set_returnValue(tilingInfo->returnValue);
    tilingData_.set_usedCoreNum(blockDim);
    tilingData_.set_batchSupperFlag(tilingInfo->batchSupperFlag);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputQLayout = static_cast<uint32_t>(tilingInfo->inputQLayout);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t weightTypeFlag = 0;
    uint64_t tilingKey = GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputQLayout,
                                            inputKLayout, weightTypeFlag);
    context_->SetTilingKey(tilingKey);

    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForLightningIndexerV2(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("LightningIndexerV2", "Tiling context is null."),
                return ge::GRAPH_FAILED);
    LIV2TilingInfo liV2Info;
    LIV2InfoParser LIV2InfoParser(context);
    if (LIV2InfoParser.ParseAndCheck(liV2Info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    LightningIndexerV2Tiling liTiling(context);
    return liTiling.DoTiling(&liV2Info);
}
// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(LightningIndexerV2)
    .Tiling(TilingForLightningIndexerV2)
    .TilingParse<LIV2CompileInfo>(TilingPrepareForLightningIndexerV2);
} // namespace optiling
