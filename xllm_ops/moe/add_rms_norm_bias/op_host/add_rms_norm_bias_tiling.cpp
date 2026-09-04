/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_bias_tiling.cpp
 * \brief
 */
#include "add_rms_norm_bias_tiling.h"
#include "log/ops_log.h"

#include <algorithm>
#include <stdexcept>

namespace optiling {
constexpr uint32_t DTYPE_KEY_FP16 = 1;
constexpr uint32_t DTYPE_KEY_FP32 = 2;
constexpr uint32_t DTYPE_KEY_BF16 = 3;
constexpr uint32_t UB_USED = 1024;
constexpr uint32_t SMALL_REDUCE_NUM_WITH_BETA = 1600;
constexpr uint32_t FP32_WEIGHT_WITH_BETA = 28;
constexpr uint32_t OTHER_WEIGHT_WITH_BETA = 20;
constexpr size_t NUM_WITH_BETA = 4;

constexpr uint32_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t FLOAT_BLOCK_ALIGN_NUM = 8;
constexpr uint32_t SMALL_REDUCE_NUM = 2000;
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t MODE_MERGE_N = 2;
constexpr uint32_t MODE_SINGLE_N = 3;
constexpr uint32_t MODE_MULTI_N = 4;
constexpr uint32_t MODE_VF_R_FULL_LOAD = 5;   // arch35 VF：整行全载
constexpr uint32_t MODE_VF_SINGLE_N = 6;      // arch35 VF：每核一行
constexpr int32_t INPUT_X1_INDEX = 0;
constexpr int32_t INPUT_X2_INDEX = 1;
constexpr int32_t INPUT_GAMMA_INDEX = 2;
constexpr int32_t INPUT_BETA_INDEX = 3;
constexpr int32_t OUTPUT_Y_INDEX = 0;
constexpr int32_t OUTPUT_RSTD_INDEX = 1;
constexpr int32_t OUTPUT_X_INDEX = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t MIN_DIM_X = 1;
constexpr size_t MIN_DIM_GAMMA = 1;
constexpr size_t FP32_WEIGHT = 24;
constexpr size_t OTHER_WEIGHT = 18;
constexpr size_t DIV_FACTOR = 260;
constexpr size_t FLOAT_PER_REPEAT = 64;
constexpr size_t LEGACY_MULTI_FIXED_BYTES = 256;
constexpr size_t NUM = 2;
constexpr int32_t TEN = 10;

constexpr uint64_t VREG_BYTES = 256;
constexpr uint64_t UB_BLOCK_BYTES = 32;
constexpr uint64_t FULLLOAD_FIXED_RESERVE = 1024;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

uint64_t HighestPowerOfTwoBelow(uint64_t value)
{
    if (value <= 1) {
        return 1;
    }
    uint64_t power = 1ULL << (63U - static_cast<uint32_t>(__builtin_clzll(value)));
    return power == value ? power / 2 : power;
}

uint64_t BinaryReduceBytes(uint64_t aligned_r, uint64_t rows)
{
    const uint64_t quotient = HighestPowerOfTwoBelow(aligned_r);
    const uint64_t quotient_loops = (quotient + VREG_BYTES / sizeof(float) - 1) /
                                    (VREG_BYTES / sizeof(float));
    return AlignUp(quotient_loops, UB_BLOCK_BYTES / sizeof(float)) * sizeof(float) * rows;
}

uint64_t FullLoadPeakBytes(uint64_t aligned_r, uint64_t rows, uint32_t elem_bytes, bool has_beta)
{
    const uint64_t resident_params = (has_beta ? 2ULL : 1ULL) * aligned_r * elem_bytes;
    const uint64_t row_tensors = 6ULL * aligned_r * elem_bytes * rows; // x1/x2 DB + x/y
    const uint64_t fp32_spill = elem_bytes == sizeof(float) ? 0ULL : aligned_r * sizeof(float) * rows;
    const uint64_t row_scalars = 3ULL * AlignUp(rows, VREG_BYTES / sizeof(float)) * sizeof(float);
    return FULLLOAD_FIXED_RESERVE + resident_params + row_tensors + fp32_spill + row_scalars +
           BinaryReduceBytes(aligned_r, rows);
}

uint64_t SingleNStaticPeakBytes(uint64_t aligned_r, uint32_t elem_bytes, bool has_beta)
{
    const uint64_t t_slots = (has_beta ? 4ULL : 3ULL) * aligned_r * elem_bytes;
    const uint64_t fp32_slots = aligned_r * sizeof(float);
    const uint64_t scalar_slots = 2ULL * VREG_BYTES;
    return AlignUp(t_slots, UB_BLOCK_BYTES) + fp32_slots + scalar_slots + BinaryReduceBytes(aligned_r, 1);
}

uint64_t SingleNStaticSlotCapacity(uint32_t elem_bytes, bool has_beta)
{
    constexpr uint64_t static_ub_bytes = 191ULL * 1024ULL;
    constexpr uint64_t static_fixed_bytes = 2048ULL;
    const uint64_t t_slots = has_beta ? 4ULL : 3ULL;
    const uint64_t align_elems = UB_BLOCK_BYTES / elem_bytes;
    return ((static_ub_bytes - static_fixed_bytes) / (t_slots * elem_bytes + sizeof(float)) /
            align_elems) * align_elems;
}

uint64_t SplitDVfPeakBytes(uint64_t aligned_r, uint32_t elem_bytes, bool has_beta)
{
    // x1/x2 pair + gamma[/beta] + x output + y output, plus reduce/rstd scalar VRegs.
    const uint64_t t_slots = (has_beta ? 6ULL : 5ULL) * aligned_r * elem_bytes;
    return t_slots + 2ULL * VREG_BYTES;
}

// 非 950 平台编译的是 legacy split_d kernel（add_rms_norm_bias_split_d.h），其 UB 占用与
// arch35 VF kernel 不同（x1/x2 双缓冲 + gamma[/beta] + y + fp32 spill + sqx），
// 这里按该 kernel 的 InitBuffer 逐项记账：fp16 无 beta 为 16B/elem（含 fp32 spill 与 sqx），
// 每多 beta 加 elem_bytes；fp32 时无 spill、仅 sqx。
// 预留余量与旧硬编码表（UB_FACTOR_*_CUTD 按 200KB UB - ~11KB 标定）保持一致的保守程度。
// 该模型只用于非 950 的 MODE_SPLIT_D 路由，避免 A2/A3 上误用 VF 峰值模型导致 InitBuffer 超 UB。
constexpr uint64_t LEGACY_SPLIT_RESERVED_BYTES = 11 * 1024;
uint64_t SplitDPeakBytesLegacy(uint64_t aligned_r, uint32_t elem_bytes, bool has_beta)
{
    const uint64_t pairIn = 2ULL * aligned_r * elem_bytes;                 // inQueueX (x1+x2)
    const uint64_t params = (has_beta ? 2ULL : 1ULL) * aligned_r * elem_bytes; // gamma[/beta]
    const uint64_t yOut = aligned_r * elem_bytes;                          // outQueueY
    const uint64_t fp32Spill = elem_bytes == sizeof(float) ? 0ULL : aligned_r * sizeof(float); // xFp32Buf
    const uint64_t sqx = aligned_r * sizeof(float);                        // sqxBuf
    return pairIn + params + yOut + fp32Spill + sqx + LEGACY_SPLIT_RESERVED_BYTES;
}

template <typename PeakFn>
uint32_t MaxAlignedElements(uint64_t usable_ub, uint32_t align_elems, PeakFn peak)
{
    uint64_t low = align_elems;
    uint64_t high = usable_ub / 2;
    uint64_t best = 0;
    while (low <= high) {
        uint64_t mid = ((low + high) / 2 / align_elems) * align_elems;
        if (mid == 0) {
            break;
        }
        if (peak(mid) <= usable_ub) {
            best = mid;
            low = mid + align_elems;
        } else {
            if (mid < align_elems) {
                break;
            }
            high = mid - align_elems;
        }
    }
    return static_cast<uint32_t>(best);
}

constexpr int32_t PERFORMANC_DIM_ZERO = 0;
constexpr int32_t PERFORMANC_DIM_ONE = 1;
constexpr int32_t PERFORMANC_DIM_TWO = 2;
constexpr int32_t PERFORMANC_DIM_THREE = 3;
constexpr int32_t PERFORMANC_DIM_ONE_MAX = 512;
constexpr int32_t PERFORMANC_DIM_TWO_MAX = 8;
constexpr int32_t PERFORMANC_DIM_THREE_MAX = 5120;

platform_ascendc::SocVersion addRmsNormBiasSocVersion;

uint8_t getPerformanceFlag(uint32_t num_col, gert::Shape x_shape, gert::Shape gamma_shape, uint32_t xDtypeKey)
{
    uint8_t isPerformance = 0;
    if(addRmsNormBiasSocVersion != platform_ascendc::SocVersion::ASCEND910B) {
        return isPerformance;
    }
    size_t xDimNum = x_shape.GetDimNum();
    size_t gammaDimNum = gamma_shape.GetDimNum();
    bool dimOK = ((xDimNum == PERFORMANC_DIM_TWO || xDimNum == PERFORMANC_DIM_THREE) && gammaDimNum == PERFORMANC_DIM_ONE);
    bool sizeOk = num_col <= PERFORMANC_DIM_THREE_MAX && 
        ((xDimNum == PERFORMANC_DIM_TWO && x_shape.GetDim(PERFORMANC_DIM_ZERO) <= PERFORMANC_DIM_ONE_MAX) || 
         (xDimNum == PERFORMANC_DIM_THREE && x_shape.GetDim(PERFORMANC_DIM_ZERO) <= PERFORMANC_DIM_ONE_MAX && x_shape.GetDim(PERFORMANC_DIM_ONE) <= PERFORMANC_DIM_TWO_MAX));
    bool dtypeOk = (xDtypeKey == DTYPE_KEY_FP16 || xDtypeKey == DTYPE_KEY_BF16);
    if(dimOK && sizeOk && dtypeOk) {
        isPerformance = 1;
    }
    return isPerformance;
}

static void SetByDtype(ge::DataType dataType, uint32_t& dtypeKey, uint32_t& dataPerBlock)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            dtypeKey = DTYPE_KEY_FP16;
            dataPerBlock = BLOCK_ALIGN_NUM;
            break;
        case ge::DT_BF16:
            dtypeKey = DTYPE_KEY_BF16;
            dataPerBlock = BLOCK_ALIGN_NUM;
            break;
        default:
            dtypeKey = DTYPE_KEY_FP32;
            dataPerBlock = FLOAT_BLOCK_ALIGN_NUM;
            break;
    }
}

static bool CheckInputOutputDim(const gert::TilingContext* context)
{
    const gert::StorageShape* x1_shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape* x2_shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape* y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);
    const gert::StorageShape* rstd_shape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape* x_shape = context->GetOutputShape(OUTPUT_X_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
    size_t x2DimNum = x2_shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();
    size_t yDimNum = y_shape->GetStorageShape().GetDimNum();
    size_t rstdDimNum = rstd_shape->GetStorageShape().GetDimNum();
    size_t xDimNum = x_shape->GetStorageShape().GetDimNum();

    OP_CHECK_IF(
        x1DimNum > MAX_DIM_NUM || x1DimNum < MIN_DIM_X,
        OP_LOGE(context, "Input x1's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_CHECK_IF(
        gammaDimNum > MAX_DIM_NUM || gammaDimNum < MIN_DIM_GAMMA,
        OP_LOGE(context, "Input gamma's dim num should not greater than 8 or smaller than 1."),
        return false);
    OP_CHECK_IF(
        x1DimNum != yDimNum, OP_LOGE(context, "Input x's dim num must equal to output y's dim num."),
        return false);

    OP_CHECK_IF(
        x1DimNum != x2DimNum,
        OP_LOGE(context, "Input x2/x1 shape invalid, dim num is not equal x1 dim."), return false);
    OP_CHECK_IF(
        (yDimNum != xDimNum) || (xDimNum != x1DimNum) || (rstdDimNum != x1DimNum),
        OP_LOGE(context, "Output y/x/rstd shape invalid, dim num is not equal x1 dim."), return false);
    OP_CHECK_IF(
        x1DimNum < gammaDimNum, OP_LOGE(context, "X1 dim num should not be smaller than gamma dim num."),
        return false);
    return true;
}

static bool CheckInputOutputShape(const gert::TilingContext* context)
{
    OP_CHECK_IF(!CheckInputOutputDim(context), OP_LOGE(context, "Input Dim invalid."), return false);
    const gert::StorageShape* x1_shape = context->GetInputShape(INPUT_X1_INDEX);
    const gert::StorageShape* x2_shape = context->GetInputShape(INPUT_X2_INDEX);
    const gert::StorageShape* gamma_shape = context->GetInputShape(INPUT_GAMMA_INDEX);
    const gert::StorageShape* y_shape = context->GetOutputShape(OUTPUT_Y_INDEX);
    const gert::StorageShape* rstd_shape = context->GetOutputShape(OUTPUT_RSTD_INDEX);
    const gert::StorageShape* x_shape = context->GetOutputShape(OUTPUT_X_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gamma_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, rstd_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

    size_t x1DimNum = x1_shape->GetStorageShape().GetDimNum();
    size_t gammaDimNum = gamma_shape->GetStorageShape().GetDimNum();

    for (uint32_t i = 0; i < x1DimNum; i++) {
        OP_CHECK_IF(
            x1_shape->GetStorageShape().GetDim(i) == 0, OP_LOGE(context, "Input x1 shape can not be 0."),
            return false);
        OP_CHECK_IF(
            x2_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Input x2/x1 shape invalid, shape is not equal x1 shape."), return false);
        OP_CHECK_IF(
            (y_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i)) ||
                (x_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(i)),
            OP_LOGE(context, "Input y/x shape invalid, shape is not equal x1 shape."), return false);
    }
    for (uint32_t i = 0; i < x1DimNum - gammaDimNum; i++) {
        OP_CHECK_IF(
            rstd_shape->GetStorageShape().GetDim(i) != x2_shape->GetStorageShape().GetDim(i),
            OP_LOGE(context, "Output rstd shape invalid, shape is not equal x1 first few dim."),
            return false);
    }
    for (uint32_t i = 0; i < gammaDimNum; i++) {
        OP_CHECK_IF(
            gamma_shape->GetStorageShape().GetDim(i) != x1_shape->GetStorageShape().GetDim(x1DimNum - gammaDimNum + i),
            OP_LOGE(context, "Input gamma shape invalid, gamma shape is not equal x1 last few dim."),
            return false);
        OP_CHECK_IF(
            rstd_shape->GetStorageShape().GetDim(x1DimNum - 1 - i) != 1,
            OP_LOGE(context, "Output rstd shape invalid, last few dim is not equal to 1."),
            return false);
    }
    return true;
}

static void GetCompileParameters(
    gert::TilingContext* context, uint32_t& numCore, uint64_t& ubSize)
{
    auto ptrCompileInfo = reinterpret_cast<const AddRmsNormBiasCompileInfo*>(context->GetCompileInfo());
    if (ptrCompileInfo == nullptr) {
        auto ascendc_platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        addRmsNormBiasSocVersion = ascendc_platform.GetSocVersion();
        numCore = ascendc_platform.GetCoreNumAiv();
        ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    } else {
        numCore = ptrCompileInfo->totalCoreNum;
        ubSize = ptrCompileInfo->totalUbSize;
        addRmsNormBiasSocVersion = ptrCompileInfo->socVersion;
    }
    ubSize -= UB_USED;
}

static void CalculateRowAndColParameters(gert::TilingContext* context, uint32_t& numRow, uint32_t& numCol)
{
    const gert::Shape x1_shape = context->GetInputShape(0)->GetStorageShape();
    const size_t gammaIndex = 2;
    const gert::Shape gamma_shape = context->GetInputShape(gammaIndex)->GetStorageShape();
    numCol = gamma_shape.GetShapeSize();

    const size_t x1DimNum = x1_shape.GetDimNum();
    const size_t gammaDimNum = gamma_shape.GetDimNum();
    numRow = 1U;
    for (size_t i = 0; i < x1DimNum - gammaDimNum; ++i) {
        numRow *= x1_shape.GetDim(i);
    }
}

static ge::graphStatus GetEpsilonParameter(gert::TilingContext* context, float& epsilon)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    epsilon = *attrs->GetFloat(0);
    OP_CHECK_IF(
        epsilon < 0, OP_LOGE(context, "Epsilon less than zero, please check."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void CalculateBlockParameters(
    uint32_t numRow, uint32_t numCore, uint32_t& blockFactor, uint32_t& latsBlockFactor, uint32_t& useCoreNum)
{
    blockFactor = 1U;
    uint32_t tileNum = CeilDiv(numRow, numCore * blockFactor);
    blockFactor *= tileNum;
    useCoreNum = CeilDiv(numRow, blockFactor);
    latsBlockFactor = numRow - blockFactor * (useCoreNum - 1);
}

static ge::DataType SetDataTypeParameters(gert::TilingContext* context, uint32_t& dtype_key, uint32_t& data_per_block)
{
    auto data_type = context->GetInputDesc(0)->GetDataType();
    dtype_key = DTYPE_KEY_FP16;
    SetByDtype(data_type, dtype_key, data_per_block);
    return data_type;
}

static void DetermineModeParameters(
    AddRMSNormBiasTilingData* tiling, 
    uint32_t numCol, uint32_t& ubFactor, uint32_t& rowFactor, uint32_t blockFactor, 
    uint32_t latsBlockFactor, ge::DataType dataType, uint32_t dtypKey, uint64_t ubSize,
    uint32_t dataPerBlock, uint32_t numColAlign, uint32_t& modeKey, uint32_t isPerformance)
{
    const uint32_t elemBytes = dataType == ge::DT_FLOAT ? 4U : 2U;
    const bool hasBeta = tiling->get_nullptr_beta() == 0;
    // 非 950 平台编译 legacy split_d kernel，UB 峰值必须按该 kernel 记账；
    // 950 上 MODE_SPLIT_D 由 arch35 VF kernel 承接，用 SplitDVfPeakBytes。
    const bool isAscend950 = addRmsNormBiasSocVersion == platform_ascendc::SocVersion::ASCEND950;
    if (numCol > ubFactor) {
        modeKey = MODE_SPLIT_D;
        const auto splitPeak = isAscend950
            ? [](uint64_t r, uint32_t eb, bool b) { return SplitDVfPeakBytes(r, eb, b); }
            : [](uint64_t r, uint32_t eb, bool b) { return SplitDPeakBytesLegacy(r, eb, b); };
        const uint32_t maxSplit = MaxAlignedElements(
            ubSize, dataPerBlock,
            [=](uint64_t r) { return splitPeak(r, elemBytes, hasBeta); });
        if (maxSplit == 0U) {
            throw std::runtime_error("runtime UB cannot hold the minimum split-D tile");
        }
        const uint32_t colTileNum = CeilDiv(numCol, maxSplit);
        ubFactor = CeilDiv(numCol, colTileNum * dataPerBlock) * dataPerBlock;
        if (splitPeak(ubFactor, elemBytes, hasBeta) > ubSize) {
            ubFactor = maxSplit;
        }
        rowFactor = 1U;
    } else if (blockFactor == 1 && addRmsNormBiasSocVersion != platform_ascendc::SocVersion::ASCEND310P) {
        modeKey = MODE_SINGLE_N;
    } else if (((tiling->get_nullptr_beta() == 1 && numColAlign <= SMALL_REDUCE_NUM) || (tiling->get_nullptr_beta() == 0 && numColAlign <= SMALL_REDUCE_NUM_WITH_BETA)) && addRmsNormBiasSocVersion != platform_ascendc::SocVersion::ASCEND310P) {
        modeKey = MODE_MERGE_N;
        uint64_t numColAlignWeight = tiling->get_nullptr_beta() == 1 ? ((dtypKey == DTYPE_KEY_FP32) ? FP32_WEIGHT : OTHER_WEIGHT) : ((dtypKey == DTYPE_KEY_FP32) ? FP32_WEIGHT_WITH_BETA : OTHER_WEIGHT_WITH_BETA);
        rowFactor = static_cast<uint32_t>(ubSize) /
                    (numColAlign * static_cast<uint32_t>(numColAlignWeight) + static_cast<uint32_t>(DIV_FACTOR));
        ubFactor = rowFactor * numColAlign;

        uint32_t mulLoopFp32 = numColAlign / 64;
        uint32_t mulTailFp32 = numColAlign - mulLoopFp32 * 64;
        uint8_t dstRepStrideFp32 = numColAlign / 8; 

        uint32_t mulLoopFp16 = numColAlign / 128;
        uint32_t mulTailFp16 = numColAlign - mulLoopFp16 * 128;
        uint8_t dstRepStrideFp16 = numColAlign / 16; 

        tiling->set_is_performance(isPerformance);
        tiling->set_mul_loop_fp32(mulLoopFp32);
        tiling->set_mul_tail_fp32(mulTailFp32);
        tiling->set_dst_rep_stride_fp32(dstRepStrideFp32);
        tiling->set_mul_loop_fp16(mulLoopFp16);
        tiling->set_mul_tail_fp16(mulTailFp16);
        tiling->set_dst_rep_stride_fp16(dstRepStrideFp16);
    } else if ((dataType == ge::DT_FLOAT16 || isPerformance == 1) && numCol == numColAlign) {
        modeKey = MODE_MULTI_N;
        rowFactor = (static_cast<uint32_t>(ubSize) - static_cast<uint32_t>(LEGACY_MULTI_FIXED_BYTES) -
                     numColAlign * static_cast<uint32_t>(tiling->get_nullptr_beta() == 1 ? NUM : NUM_WITH_BETA)) /
                    (numColAlign * BLOCK_ALIGN_NUM + static_cast<uint32_t>(FLOAT_PER_REPEAT));
        ubFactor = rowFactor * numColAlign;
        if (rowFactor == 0U) {
            modeKey = MODE_NORMAL;
            rowFactor = 1U;
            ubFactor = std::max(dataPerBlock, std::min(numColAlign, ubFactor));
        }
    }
    uint32_t rowLoop = CeilDiv(blockFactor, rowFactor);
    uint32_t lastBlockRowLoop = CeilDiv(latsBlockFactor, rowFactor);
    uint32_t rowTail = blockFactor - (rowLoop - 1) * rowFactor;
    uint32_t lastBlockRowTail = latsBlockFactor - (lastBlockRowLoop - 1) * rowFactor;
    tiling->set_row_loop(rowLoop);
    tiling->set_last_block_row_loop(lastBlockRowLoop);
    tiling->set_row_tail(rowTail);
    tiling->set_last_block_row_tail(lastBlockRowTail);
}

static void SetTilingParameters(
    AddRMSNormBiasTilingData* tiling, uint32_t num_row, uint32_t num_col, uint32_t numColAlign, 
    uint32_t block_factor, uint32_t latsBlockFactor, uint32_t row_factor,
    uint32_t ub_factor, float epsilon)
{
    const float avg_factor = (num_col == 0) ? 0 : 1.0f / num_col;
    tiling->set_num_row(num_row);
    tiling->set_num_col(num_col);
    tiling->set_num_col_align(numColAlign);
    tiling->set_block_factor(block_factor);
    tiling->set_last_block_factor(latsBlockFactor);
    tiling->set_row_factor(row_factor);
    tiling->set_ub_factor(ub_factor);
    tiling->set_epsilon(epsilon);
    tiling->set_avg_factor(avg_factor);
}


// ---------------------------------------------------------------------------
// arch35（Ascend950 / dav-3510）RegBase-VF 路由
//   block_factor == 1（每核一行）                  -> 单行 VF 特化   mode 6
//   block_factor > 1 且整行能全载 UB               -> R-full-load VF mode 5
//   整行放不下（原 SPLIT_D 场景）                  -> 切 D VF        mode 7
// 谓词全部取自真实 tiling 字段与 UB 容量；不满足则保持原有 mode（老实现兜底）。
// 实测（Ascend950PR，20 例 cases_v3，msprof --reps 20 取中位数、5 轮）见 arch35/README.md
// ---------------------------------------------------------------------------
static bool TryArch35VfRoute(AddRMSNormBiasTilingData* tiling, uint32_t dtypeKey, uint32_t& modeKey,
                             uint32_t numRow, uint32_t numCol, uint32_t numColAlign, uint32_t numCore,
                             uint64_t ubSize, uint32_t& useCoreNum)
{
    if (addRmsNormBiasSocVersion != platform_ascendc::SocVersion::ASCEND950) {
        return false;
    }
    constexpr uint32_t ULONG_BIT_LEN = 64;
    const uint64_t vlfp32 = 256 / sizeof(float);
    const uint64_t binaryAddElemtMaxLen = vlfp32 * vlfp32 * 2 * 2;
    const uint32_t elemByte = (dtypeKey == DTYPE_KEY_FP32) ? 4 : 2;
    const bool hasBeta = tiling->get_nullptr_beta() == 0;

    uint64_t binAddQuotient = (numColAlign == 0) ? 1
        : (1UL << (ULONG_BIT_LEN - 1 - __builtin_clzl(static_cast<uint64_t>(numColAlign))));
    if (binAddQuotient == numColAlign) {
        binAddQuotient /= 2;
    }
    tiling->set_bin_add_quotient(static_cast<uint32_t>(binAddQuotient));

    if (modeKey == MODE_SPLIT_D) {
        // The split tile and row_factor were derived from SplitDVfPeakBytes in DetermineModeParameters.
        tiling->set_row_factor(1U);
        return true;
    }

    const uint64_t oneRowPeak = FullLoadPeakBytes(numColAlign, 1, elemByte, hasBeta);
    if (oneRowPeak > ubSize || numColAlign > binaryAddElemtMaxLen) {
        const uint32_t dataPerBlock = UB_BLOCK_BYTES / elemByte;
        const uint32_t maxSplit = MaxAlignedElements(
            ubSize, dataPerBlock,
            [=](uint64_t r) { return SplitDVfPeakBytes(r, elemByte, hasBeta); });
        if (maxSplit == 0U) {
            throw std::runtime_error("runtime UB model found no valid dav-3510 implementation tile");
        }
        const uint32_t chunks = CeilDiv(numCol, maxSplit);
        tiling->set_ub_factor(CeilDiv(numCol, chunks * dataPerBlock) * dataPerBlock);
        tiling->set_row_factor(1U);
        modeKey = MODE_SPLIT_D;
        return true;
    }

    const uint64_t vfBlockFactor = (numRow + numCore - 1) / numCore;
    if (vfBlockFactor == 1U) {
        if (SingleNStaticPeakBytes(numColAlign, elemByte, hasBeta) <= ubSize &&
            numColAlign <= SingleNStaticSlotCapacity(elemByte, hasBeta)) {
            modeKey = MODE_VF_SINGLE_N;
            tiling->set_row_factor(1U);
            return true;
        }
        // A static single-row layout cannot be substituted with the legacy SINGLE_N path.
        throw std::runtime_error("VF_SINGLE_N static layout exceeds the runtime UB capacity");
    }

    uint64_t vfRowFactor = 1;
    while (vfRowFactor < vfBlockFactor &&
           FullLoadPeakBytes(numColAlign, vfRowFactor + 1, elemByte, hasBeta) <= ubSize) {
        ++vfRowFactor;
    }
    tiling->set_block_factor(static_cast<uint32_t>(vfBlockFactor));
    tiling->set_row_factor(static_cast<uint32_t>(vfRowFactor));
    useCoreNum = static_cast<uint32_t>((numRow + vfBlockFactor - 1) / vfBlockFactor);
    modeKey = MODE_VF_R_FULL_LOAD;
    return true;
}

static void SaveTilingData(
    gert::TilingContext* context, AddRMSNormBiasTilingData* tiling, uint32_t dtype_key, uint32_t mode_key)
{
    const uint32_t tiling_key = dtype_key * 10 + mode_key;
    context->SetTilingKey(tiling_key);
    tiling->SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
}

static void SetWorkspaceSize(gert::TilingContext* context)
{
    constexpr size_t sysWorkspaceSize = 16 * 1024 * 1024;
    constexpr size_t usrSize = 256;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
}

static void LogTilingResults(
    gert::TilingContext* context, AddRMSNormBiasTilingData* tiling, uint32_t mode_key, uint32_t dtype_key,
    uint32_t use_core_num, float epsilon)
{
    OPS_LOG_I(context, "Tiling Key: %u", dtype_key * TEN + mode_key);
    OPS_LOG_I(context, "Block Dim: %u", use_core_num);
    OPS_LOG_I(context, "usr Workspace: 256");
    OPS_LOG_I(
        context,
        "num_row: %d, num_col: %d, block_factor: %d, row_factor: %d, ub_factor: %d, epsilon: %f, avg_factor: %f",
        tiling->get_num_row(), tiling->get_num_col(), tiling->get_block_factor(), tiling->get_row_factor(),
        tiling->get_ub_factor(), epsilon, tiling->get_avg_factor());
}

static ge::graphStatus Tiling4AddRmsNormBias(gert::TilingContext* context)
{
    OP_LOGI("Tiling4AddRmsNormBias", "Enter Tiling4AddRmsNormBias");
    OPS_LOG_D(context, "Tiling4AddRmsNormBias1 running. \n");
    OP_CHECK_IF(
        !CheckInputOutputShape(context), OP_LOGE(context, "Input shape invalid."),
        return ge::GRAPH_FAILED);

    AddRMSNormBiasTilingData tiling;

    auto betaDesc = context->GetOptionalInputDesc(INPUT_BETA_INDEX);
    tiling.set_nullptr_beta(betaDesc == nullptr ? 1 : 0);

    uint32_t num_core;
    uint64_t ub_size;
    GetCompileParameters(context, num_core, ub_size);
    uint32_t num_row;
    uint32_t num_col;
    CalculateRowAndColParameters(context, num_row, num_col);
    float epsilon = 0;
    GetEpsilonParameter(context, epsilon);
    if (epsilon < 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t block_factor;
    uint32_t latsBlockFactor;
    uint32_t use_core_num;
    CalculateBlockParameters(num_row, num_core, block_factor, latsBlockFactor, use_core_num);
    context->SetBlockDim(use_core_num);
    uint32_t dtype_key;
    uint32_t data_per_block;
    ge::DataType data_type = SetDataTypeParameters(context, dtype_key, data_per_block);
    uint32_t mode_key = MODE_NORMAL;
    uint32_t row_factor = 64;
    uint32_t numColAlign = CeilDiv(num_col, data_per_block) * data_per_block;
    const bool hasBeta = betaDesc != nullptr;
    const uint32_t elemBytes = dtype_key == DTYPE_KEY_FP32 ? 4U : 2U;
    uint32_t ub_factor = MaxAlignedElements(
        ub_size, data_per_block,
        [=](uint64_t r) { return FullLoadPeakBytes(r, 1, elemBytes, hasBeta); });
    if (ub_factor == 0U) {
        throw std::runtime_error("runtime UB cannot hold one aligned AddRmsNormBias tile");
    }
    const gert::Shape x1_shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape gamma_shape = context->GetInputShape(2)->GetStorageShape();
    uint8_t isPerformance = getPerformanceFlag(num_col, x1_shape, gamma_shape, dtype_key);
    DetermineModeParameters(
        &tiling, 
        num_col, ub_factor, row_factor, block_factor, latsBlockFactor, 
        data_type, dtype_key, ub_size, data_per_block, 
        numColAlign, mode_key, isPerformance);
    SetTilingParameters(&tiling, num_row, num_col, numColAlign, block_factor, latsBlockFactor, row_factor, ub_factor, epsilon);
    if (TryArch35VfRoute(&tiling, dtype_key, mode_key, num_row, num_col, numColAlign, num_core, ub_size, use_core_num)) {
        context->SetBlockDim(use_core_num);
    }
    SaveTilingData(context, &tiling, dtype_key, mode_key);
    SetWorkspaceSize(context);
    LogTilingResults(context, &tiling, mode_key, dtype_key, use_core_num, epsilon);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4AddRmsNormBias(gert::TilingParseContext* context)
{
    OPS_LOG_D(context, "TilingPrepare4AddRmsNormBias running. \n");
    OP_LOGI(context, "TilingPrepare4AddRmsNormBias running.");
    auto compileInfo = context->GetCompiledInfo<AddRmsNormBiasCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->totalUbSize);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddRmsNormBias).Tiling(Tiling4AddRmsNormBias).TilingParse<AddRmsNormBiasCompileInfo>(TilingPrepare4AddRmsNormBias);

} // namespace optiling
