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
 * \file lightning_indexer_v2_metadata_check.h
 * \brief
 */

#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"
#include "opdev/data_type_utils.h"
#include "opdev/tensor_view_utils.h"
#include "../../lightning_indexer_v2/op_host/lightning_indexer_v2_error_log.h"
#include "../../lightning_indexer_v2/op_kernel/lightning_indexer_v2_metadata.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr const char *LI_V2_ACLNN_OP_NAME = "LightningIndexerV2Metadata";

// NOTE: CANN's opdev CHECK_COND expands to OP_LOGE(ret, fmt, ...) where the first argument is the
// int error code. In this build environment OP_LOGE(int, ...) wrongly resolves to the context-pointer
// overload (GetOpInfo<T=int>), causing a compile error. Override CHECK_COND here to pass the opname
// string as the log target, keeping the (cond, ret, fmt, ...) callsite signature unchanged.
#undef CHECK_COND
#define CHECK_COND(cond, ret, fmt, ...)                      \
    do {                                                     \
        if (!(cond)) {                                       \
            OP_LOGE(LI_V2_ACLNN_OP_NAME, fmt, ##__VA_ARGS__); \
            return ret;                                      \
        }                                                    \
    } while (false)

// NOTE: CANN's opdev CHECK_RET expands to OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_INNER, ...) whose first
// argument is an int error code, which wrongly resolves to the context-pointer OP_LOGE overload in
// this build environment. Override CHECK_RET here to log with the opname string instead.
#undef CHECK_RET
#define CHECK_RET(cond, ret_value)                                          \
    do {                                                                    \
        if (!(cond)) {                                                      \
            OP_LOGE(LI_V2_ACLNN_OP_NAME, "check %s failed.", #cond);        \
            return ret_value;                                              \
        }                                                                   \
    } while (false)

inline constexpr int64_t LI_V2_NO_MASK_MODE = 0;
inline constexpr int64_t LI_V2_CAUSAL_MASK_MODE = 3;
inline constexpr int64_t LI_V2_CMP_RATIO_LOWER_BOUND = 1;
inline constexpr int64_t LI_V2_CMP_RATIO_UPPER_BOUND = 128;
inline constexpr int64_t LI_V2_NUM_HEADS_Q_LOWER_BOUND = 1;
inline constexpr int64_t LI_V2_NUM_HEADS_Q_UPPER_BOUND = 64;
inline constexpr int64_t LI_V2_TOPK_LOWER_BOUND = 1;
inline constexpr int64_t LI_V2_TOPK_UPPER_BOUND = 8192;

inline bool IsTensorExistLiV2(const aclTensor *tensor)
{
    return (tensor != nullptr) && (tensor->GetViewShape().GetDimNum() > 0) && (tensor->GetViewShape().GetDim(0) > 0);
}

int64_t GetDimNumLiV2(const aclTensor *tensor)
{
    if (tensor == nullptr) {
        return -1;
    }
    return tensor->GetViewShape().GetDimNum();
}

aclDataType GetDataTypeLiV2(const aclTensor *tensor)
{
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    if (tensor == nullptr) {
        return dataType;
    }
    aclGetDataType(tensor, &dataType);
    return dataType;
}

inline bool IsTensorSourceLiV2(const std::string &source) { return source != "batch_size"; }

inline int64_t GetRawShapeSizeLiV2(const std::string &source, int64_t batchValue)
{
    if (source.find("cu_seqlens") != std::string::npos) {
        return batchValue + 1;
    }
    return batchValue;
}

int64_t GetQueryBatchSizeLiV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional, const aclTensor *sequsedQOptional,
                              const char *layoutQOptional, std::string &source)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    if (IsTensorExistLiV2(sequsedQOptional)) {
        source = "seqused_q";
        return sequsedQOptional->GetViewShape().GetDim(0);
    }
    // 2. 如果sequsedQOptional 没传，使用cuSeqlensQOptional获取BatchSize
    if (strcmp(layoutQOptional, "TND") == 0) {
        if (IsTensorExistLiV2(cuSeqlensQOptional)) { // 前序校验已保证layout_q = TND时，cu_seqlens_q必须传入，此通路必达
            source = "cu_seqlens_q";
            return cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    source = "batch_size";
    // 3. 使用batchSize
    return batchSize;
}

int64_t GetKeyBatchSizeLiV2(int64_t batchSize, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedKOptional,
                            const char *layoutKOptional, std::string &source)
{
    // 1. 如果sequsedKOptional 传了，使用sequsedKOptional获取BatchSize
    if (IsTensorExistLiV2(sequsedKOptional)) {
        source = "seqused_k";
        return sequsedKOptional->GetViewShape().GetDim(0);
    }
    // 如果是 TND，必须使用 cuSeqlensKOptional获取BatchSize
    if (strcmp(layoutKOptional, "TND") == 0) {
        if (IsTensorExistLiV2(cuSeqlensKOptional)) { // 前序校验已保证layout_k = TND时，cu_seqlens_k必须传入，此通路必达
            source = "cu_seqlens_k";
            return cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        }
    }
    source = "batch_size";
    // 3. 使用batchSize
    return batchSize;
}

aclnnStatus CheckSingleParamLiV2(int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t batchSize,
                                 int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
                                 const char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, uint32_t aicCoreNum,
                                 uint32_t aivCoreNum, const std::string &socVersion)
{
    // num_heads_q 校验
    if (numHeadsQ < LI_V2_NUM_HEADS_Q_LOWER_BOUND || numHeadsQ > LI_V2_NUM_HEADS_Q_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "num_heads_q", std::to_string(numHeadsQ),
                                              "The value of num_heads_q should be in range [" +
                                                  std::to_string(LI_V2_NUM_HEADS_Q_LOWER_BOUND) + ", " +
                                                  std::to_string(LI_V2_NUM_HEADS_Q_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_k 校验
    if (numHeadsK != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "num_heads_kv", std::to_string(numHeadsK),
                                              "The value of num_heads_kv should be 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // head_dim 校验
    if (headDim != 128) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "head_dim", std::to_string(numHeadsK),
                                              "The value of head_dim should be 128");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // topk 校验
    if (topk < LI_V2_TOPK_LOWER_BOUND || topk > LI_V2_TOPK_UPPER_BOUND) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "topk", std::to_string(topk),
                                              "The value of topk should be in range [" +
                                                  std::to_string(LI_V2_TOPK_LOWER_BOUND) + ", " +
                                                  std::to_string(LI_V2_TOPK_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // batch_size 非负校验
    if (batchSize < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "batch_size", std::to_string(batchSize),
                                              "The value of batch_size should not be negative");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 校验
    if (maxSeqlenQ < -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                              "The value of max_seqlen_q should be >= -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_k 校验
    if (maxSeqlenK < -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
                                              "The value of max_seqlen_k should be >= -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode 校验
    if ((maskMode != LI_V2_NO_MASK_MODE) && (maskMode != LI_V2_CAUSAL_MASK_MODE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "mask_mode", std::to_string(maskMode),
                                              "The value of mask_mode should be " + std::to_string(LI_V2_NO_MASK_MODE) +
                                                  " or " + std::to_string(LI_V2_CAUSAL_MASK_MODE));
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_ratio 校验
    if ((cmpRatio < LI_V2_CMP_RATIO_LOWER_BOUND) || (cmpRatio > LI_V2_CMP_RATIO_UPPER_BOUND)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "cmp_ratio", std::to_string(cmpRatio),
                                              "The value of cmp_ratio should be in range [" +
                                                  std::to_string(LI_V2_CMP_RATIO_LOWER_BOUND) + ", " +
                                                  std::to_string(LI_V2_CMP_RATIO_UPPER_BOUND) + "]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if (layoutQOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(LI_V2_ACLNN_OP_NAME, "layout_q", "Layout_q is null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((strcmp(layoutQOptional, "TND") != 0) && (strcmp(layoutQOptional, "BSND") != 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "layout_q", layoutQOptional,
                                              "The value of layout_q must be TND or BSND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k 校验
    if (layoutKOptional == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(LI_V2_ACLNN_OP_NAME, "layout_k", "Layout_k is null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((strcmp(layoutKOptional, "TND") != 0) && (strcmp(layoutKOptional, "BSND") != 0) &&
        (strcmp(layoutKOptional, "PA_BBND") != 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "layout_k", layoutKOptional,
                                              "The value of layout_k must be in [TND, BSND, PA_BBND]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((strcmp(layoutKOptional, "PA_BBND") != 0) && (strcmp(layoutQOptional, layoutKOptional) != 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(LI_V2_ACLNN_OP_NAME, "layout_q and layout_k",
                                               std::string(layoutQOptional) + " and " + std::string(layoutKOptional),
                                               "For layout_k != PA_BBND, layout_q and layout_k must be the same");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 layout_q 为 BSND 时，max_seqlen_q 必须大于 0
    if ((strcmp(layoutQOptional, "BSND") == 0) && (maxSeqlenQ <= 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(LI_V2_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
                                               "When layout_q is BSND, the value of max_seqlen_q "
                                               "must be equal to the size of the second axis of q");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 layout_k 为 BSND 时，max_seqlen_k 必须大于 0
    if ((strcmp(layoutKOptional, "BSND") == 0) && (maxSeqlenK <= 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(LI_V2_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
                                               "When layout_k is BSND, the value of max_seqlen_k "
                                               "must be equal to the size of the second axis of k");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 核心数校验
    CHECK_COND(aicCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIC num should be larger than 0, but got %u", aicCoreNum);
    CHECK_COND(aicCoreNum <= optiling::AIC_CORE_MAX_NUM, ACLNN_ERR_PARAM_INVALID,
               "The maximum supported AIC num is %u, but got %u", optiling::AIC_CORE_MAX_NUM, aicCoreNum);
    CHECK_COND(aivCoreNum > 0, ACLNN_ERR_PARAM_INVALID, "AIV num should be larger than 0, but got %u", aivCoreNum);
    CHECK_COND(aivCoreNum <= optiling::AIV_CORE_MAX_NUM, ACLNN_ERR_PARAM_INVALID,
               "The maximum supported AIV num is %u, but got %u", optiling::AIV_CORE_MAX_NUM, aivCoreNum);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceLiV2(int64_t maskMode, int64_t cmpRatio, const aclTensor *cuSeqlensQOptional,
                               const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
                               const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
                               int64_t maxSeqlenQ, int64_t maxSeqlenK, const char *layoutQOptional,
                               const char *layoutKOptional, const aclTensor *metadata)
{
    // cu_seqlens_q 存在性校验
    if ((strcmp(layoutQOptional, "TND") == 0) && !IsTensorExistLiV2(cuSeqlensQOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(LI_V2_ACLNN_OP_NAME, "cu_seqlens_q",
                                                 "When layout_q is TND, cu_seqlens_q must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q BSND, seqused_q 不存在时，max_seqlen_q 不能为-1
    if (strcmp(layoutQOptional, "BSND") == 0 && !IsTensorExistLiV2(sequsedQOptional) && (maxSeqlenQ <= -1)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            LI_V2_ACLNN_OP_NAME, "max_seqlen_q", std::to_string(maxSeqlenQ),
            "When layout_q is BSND and seqused_q is not provided, max_seqlen_q can not be -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cu_seqlens_k 存在性校验
    if ((strcmp(layoutKOptional, "TND") == 0) && !IsTensorExistLiV2(cuSeqlensKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(LI_V2_ACLNN_OP_NAME, "cu_seqlens_k",
                                                 "When layout_k is TND, cu_seqlens_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // seqused_k 存在性校验
    if ((strcmp(layoutKOptional, "PA_BBND") == 0) && !IsTensorExistLiV2(sequsedKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(LI_V2_ACLNN_OP_NAME, "seqused_k",
                                                 "When layout_k is PA_BBND, seqused_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k BSND, seqused_k 不存在时，max_seqlen_k 不能为-1
    if (strcmp(layoutKOptional, "BSND") == 0 && !IsTensorExistLiV2(sequsedKOptional) && maxSeqlenK <= -1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            LI_V2_ACLNN_OP_NAME, "max_seqlen_k", std::to_string(maxSeqlenK),
            "When layout_k is BSND and seqused_k is not provided, max_seqlen_k can not be -1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // cmp_residual_k 存在性校验
    if ((cmpRatio != LI_V2_CMP_RATIO_LOWER_BOUND) && (maskMode == LI_V2_CAUSAL_MASK_MODE) &&
        !IsTensorExistLiV2(cmpResidualKOptional)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            LI_V2_ACLNN_OP_NAME, "cmp_residual_k",
            "When cmp_ratio is not 1 and mask_mode is CAUSAL, cmp_residual_k must be provided");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // metadata 存在性校验
    if (!IsTensorExistLiV2(metadata)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(LI_V2_ACLNN_OP_NAME, "metadata", "Metadata is nullptr");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyLiV2(int64_t batchSize, const aclTensor *cuSeqlensQOptional,
                                 const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
                                 const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
                                 const char *layoutQOptional, const char *layoutKOptional, const aclTensor *metadata)
{
    int64_t dimNum = -1;
    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;

    // 校验 cu_seqlens_q
    if (IsTensorExistLiV2(cuSeqlensQOptional)) {
        dimNum = GetDimNumLiV2(cuSeqlensQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(LI_V2_ACLNN_OP_NAME, "cu_seqlens_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeLiV2(cuSeqlensQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "cu_seqlens_q", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cu_seqlens_k
    if (IsTensorExistLiV2(cuSeqlensKOptional)) {
        dimNum = GetDimNumLiV2(cuSeqlensKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(LI_V2_ACLNN_OP_NAME, "cu_seqlens_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeLiV2(cuSeqlensKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "cu_seqlens_k", ToString(dataType).GetString(),
                                                  "The dtype of cu_seqlens_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_q
    if (IsTensorExistLiV2(sequsedQOptional)) {
        dimNum = GetDimNumLiV2(sequsedQOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(LI_V2_ACLNN_OP_NAME, "seqused_q", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeLiV2(sequsedQOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "seqused_q", ToString(dataType).GetString(),
                                                  "The dtype of seqused_q must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 seqused_k
    if (IsTensorExistLiV2(sequsedKOptional)) {
        dimNum = GetDimNumLiV2(sequsedKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(LI_V2_ACLNN_OP_NAME, "seqused_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeLiV2(sequsedKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "seqused_k", ToString(dataType).GetString(),
                                                  "The dtype of seqused_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k
    if (IsTensorExistLiV2(cmpResidualKOptional)) {
        dimNum = GetDimNumLiV2(cmpResidualKOptional);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(LI_V2_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeLiV2(cmpResidualKOptional);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "cmp_residual_k", ToString(dataType).GetString(),
                                                  "The dtype of cmp_residual_k must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 metadata
    if (IsTensorExistLiV2(metadata)) {
        dimNum = GetDimNumLiV2(metadata);
        if (dimNum != 1) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(LI_V2_ACLNN_OP_NAME, "metadata", std::to_string(dimNum), "1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        dataType = GetDataTypeLiV2(metadata);
        if (dataType != aclDataType::ACL_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(LI_V2_ACLNN_OP_NAME, "metadata", ToString(dataType).GetString(),
                                                  "The dtype of metadata must be int32");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 校验 metadata 元素数
        if (metadata->GetViewShape().GetDim(0) != optiling::LI_V2_METADATA_TOTAL_SIZE) {
            OP_LOGE(LI_V2_ACLNN_OP_NAME, "The element num of metadata must be %u, but got %lld",
                    optiling::LI_V2_METADATA_TOTAL_SIZE, metadata->GetViewShape().GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验batch
    std::string querySource;
    std::string keySource;
    int64_t queryBatchSize =
        GetQueryBatchSizeLiV2(batchSize, cuSeqlensQOptional, sequsedQOptional, layoutQOptional, querySource);
    int64_t keyBatchSize =
        GetKeyBatchSizeLiV2(batchSize, cuSeqlensKOptional, sequsedKOptional, layoutKOptional, keySource);
    if (queryBatchSize != keyBatchSize) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            LI_V2_ACLNN_OP_NAME, querySource + " and " + keySource,
            std::to_string(GetRawShapeSizeLiV2(querySource, queryBatchSize)) + " and " +
                std::to_string(GetRawShapeSizeLiV2(keySource, keyBatchSize)),
            "The batch_size obtained from query should be the same as that obtained from key");
        if (IsTensorSourceLiV2(querySource) && IsTensorSourceLiV2(keySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                LI_V2_ACLNN_OP_NAME, querySource + " and " + keySource,
                std::to_string(GetRawShapeSizeLiV2(querySource, queryBatchSize)) + " and " +
                    std::to_string(GetRawShapeSizeLiV2(keySource, keyBatchSize)),
                "The batch_size obtained from query should be the same as that obtained from key");
        } else if (IsTensorSourceLiV2(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                LI_V2_ACLNN_OP_NAME, querySource, std::to_string(GetRawShapeSizeLiV2(querySource, queryBatchSize)),
                "The batch_size obtained from query should be the same as that obtained from key");
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                LI_V2_ACLNN_OP_NAME, keySource, std::to_string(GetRawShapeSizeLiV2(keySource, keyBatchSize)),
                "The batch_size obtained from query should be the same as that obtained from key");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验TND场景q维度一致性
    if (strcmp(layoutQOptional, "TND") == 0 && IsTensorExistLiV2(sequsedQOptional)) {
        int64_t cuSeqlensQBatchSize = cuSeqlensQOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensQBatchSize != queryBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                LI_V2_ACLNN_OP_NAME, "cu_seqlens_q and seqused_q",
                std::to_string(cuSeqlensQBatchSize) + " and " + std::to_string(queryBatchSize),
                "When layout_q is TND and seqused_q is passed, "
                "the shape size of cu_seqlens_q minus 1 must be equal to "
                "the shape size of seqused_q");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验TND场景k维度一致性
    if (strcmp(layoutKOptional, "TND") == 0 && IsTensorExistLiV2(sequsedKOptional)) {
        int64_t cuSeqlensKBatchSize = cuSeqlensKOptional->GetViewShape().GetDim(0) - 1;
        if (cuSeqlensKBatchSize != keyBatchSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                LI_V2_ACLNN_OP_NAME, "cu_seqlens_k and seqused_k",
                std::to_string(cuSeqlensKBatchSize) + " and " + std::to_string(keyBatchSize),
                "When layout_k is TND and seqused_k is passed, "
                "the shape size of cu_seqlens_k minus 1 must be equal to "
                "the shape size of seqused_k");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // 校验 cmp_residual_k 元素数
    auto cmpResidualKBatch = cmpResidualKOptional->GetViewShape().GetDim(0);
    if (IsTensorExistLiV2(cmpResidualKOptional) && (cmpResidualKBatch != queryBatchSize)) {
        if (IsTensorSourceLiV2(querySource)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                LI_V2_ACLNN_OP_NAME, "cmp_residual_k and " + querySource,
                std::to_string(cmpResidualKBatch) + " and " +
                    std::to_string(GetRawShapeSizeLiV2(querySource, queryBatchSize)),
                "The batch_size of cmp_residual_k should match the valid batch size");
        } else {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                LI_V2_ACLNN_OP_NAME, "cmp_residual_k", std::to_string(cmpResidualKBatch),
                "The batch_size of cmp_residual_k should match the valid batch size");
        }
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus ParamsCheckLiV2(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional,
                            const aclTensor *sequsedQOptional, const aclTensor *sequsedKOptional,
                            const aclTensor *cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK,
                            int64_t headDim, int64_t topk, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenK,
                            char *layoutQOptional, char *layoutKOptional, int64_t maskMode, int64_t cmpRatio,
                            const aclTensor *metadata, uint32_t aicCoreNum, uint32_t aivCoreNum,
                            const std::string &socVersion)
{
    auto ret =
        CheckSingleParamLiV2(numHeadsQ, numHeadsK, headDim, topk, batchSize, maxSeqlenQ, maxSeqlenK, layoutQOptional,
                             layoutKOptional, maskMode, cmpRatio, aicCoreNum, aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckExistenceLiV2(maskMode, cmpRatio, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional,
                             sequsedKOptional, cmpResidualKOptional, maxSeqlenQ, maxSeqlenK, layoutQOptional,
                             layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    ret = CheckConsistencyLiV2(batchSize, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                               cmpResidualKOptional, layoutQOptional, layoutKOptional, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
}
#endif
