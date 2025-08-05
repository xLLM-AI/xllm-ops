/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file apply_top_k_top_p_with_sorted_tiling.cc
 * \brief
 */

#include <iostream>
#include <map>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "apply_top_k_top_p_with_sorted_tiling.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")

namespace {
    constexpr uint32_t SYS_RESERVED_UB = uint32_t(16 * 1024);
    constexpr uint32_t SELECT_RESERVED_UB = uint32_t(8 * 1024);
    constexpr uint32_t DIM_ONE = 1;
    constexpr uint32_t DIM_TWO = 2;
    constexpr int32_t SORTED_VALUE_INPUT_INDEX = 0;
    constexpr int32_t SORTED_INDICES_INPUT_INDEX = 1;
    constexpr int32_t P_INPUT_INDEX = 2;
    constexpr int32_t K_INPUT_INDEX = 3;
    constexpr uint32_t DIM_INDEX0 = 0;
    static std::map<ge::DataType, uint32_t> DTYPE_MAP = {{ge::DT_BF16, 2}, {ge::DT_FLOAT16, 1}, {ge::DT_FLOAT, 0}};
    static std::map<ge::DataType, uint32_t> DATATYPE_LEN_MAP = {{ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}, {ge::DT_FLOAT, 4}};
    const static uint32_t SYS_WORKSPACESIZE = uint32_t(16 * 1024 * 1024);

    constexpr uint32_t DATA_PER_BLOCK_B32 = 8;
    constexpr uint32_t BYTES_B32 = 4;
    constexpr uint32_t BLOCK_BYTES = 32;
    constexpr uint32_t K_VALUE_MAX = 1024;
} // namespace

namespace ops {
#define OPS_CHECK_NULL_WITH_CONTEXT(context, ptr)                                                \
  if ((ptr) == nullptr) {                                                                        \
    std::printf("nullptr error!");                                                               \
    return ge::GRAPH_FAILED;                                                                     \
  }
}  // namespace ops

namespace optiling {
#define VECTOR_INNER_ERR_REPORT_TILIING(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace optiling {
class ApplyTopKTopPWithSortedTiling {
public:
    explicit ApplyTopKTopPWithSortedTiling(gert::TilingContext* context) : tilingcontext(context){};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
private:
    ApplyTopKTopPWithSortedTilingData tilingData;
    gert::TilingContext* tilingcontext = nullptr;
    void SetTilingKey();
    void GetUsedCore();
    void CalDataPerCore();
    void FillTilingData();
    void PrintTilingData();
    template <typename T1>
    inline T1 CeilAlign(T1 a, T1 b) const
    {
        return b == 0 ? a : (a + b - 1) / b * b;
    }
    template <typename T1>
    inline T1 FloorAlign(T1 a, T1 b) const
    {
        return b == 0 ? a : a / b * b;
    }

    const char *opName_ = nullptr;
    uint32_t coreNum_ = 0;
    uint32_t calUbSize_ = 0;
    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t tilingKey_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t batchPerCore_ = 1;
    uint32_t tailBatch_ = 0;
    uint32_t dataNumInit_ = 0;
    uint32_t dataNumInitAligned_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t ubFactorElementAligned_ = 0;
    uint32_t tailUbFactorElement_ = 0;
    uint32_t tailUbFactorElementAligned_ = 0;
};

ge::graphStatus ApplyTopKTopPWithSortedTiling::Init() {
    opName_ = tilingcontext->GetNodeName();
    // OP_LOGD(opName_, "TilingForApplyTopKTopPWithSorted init.");
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingcontext->GetPlatformInfo());
    coreNum_ = platformInfo.GetCoreNumAiv();
    uint64_t platformUbSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformUbSize);
    // OP_LOGD(opName_, "platformUbSize: %lu.", platformUbSize);
    uint32_t avaliableUb = static_cast<uint32_t>(platformUbSize) - SYS_RESERVED_UB - SELECT_RESERVED_UB;
    calUbSize_ = FloorAlign(avaliableUb, BLOCK_BYTES);

    auto sortedValueShape = tilingcontext->GetInputShape(SORTED_VALUE_INPUT_INDEX)->GetStorageShape();
    if (sortedValueShape.GetDimNum() != DIM_TWO) {
        OP_LOGE(opName_, "the dimNum of sorted_value should be 2, but got %ld.", sortedValueShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto sortedIndicesShape = tilingcontext->GetInputShape(SORTED_INDICES_INPUT_INDEX)->GetStorageShape();
    if (sortedIndicesShape.GetDimNum() != DIM_TWO) {
        OP_LOGE(opName_, "the dimNum of sorted_indices should be 2, but got %ld.", sortedIndicesShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto pShape = tilingcontext->GetInputShape(P_INPUT_INDEX)->GetStorageShape();
    if (pShape.GetDimNum() != DIM_ONE) {
        OP_LOGE(opName_, "the dimNum of p should be 1, but got %ld.", pShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    auto kShape = tilingcontext->GetInputShape(K_INPUT_INDEX)->GetStorageShape();
    if (kShape.GetDimNum() != DIM_ONE) {
        OP_LOGE(opName_, "the dimNum of k should be 1, but got %ld.", kShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    batchSize_ = sortedValueShape.GetDim(DIM_INDEX0);
    vocabSize_ = sortedValueShape.GetDim(DIM_ONE);
    if (sortedIndicesShape.GetDim(DIM_INDEX0) != batchSize_ || sortedIndicesShape.GetDim(DIM_ONE) != vocabSize_) {
        OP_LOGE(opName_, "the shape of sorted_indices should be equal to sorted_value.");
        return ge::GRAPH_FAILED;
    }

    if (batchSize_ != pShape.GetDim(DIM_INDEX0)) {
        OP_LOGE(opName_, "p.shape[0] should be equal to logits.shape[0].");
        return ge::GRAPH_FAILED;
    }
    if (batchSize_ != kShape.GetDim(DIM_INDEX0)) {
        OP_LOGE(opName_, "k.shape[0] should be equal to logits.shape[0].");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void ApplyTopKTopPWithSortedTiling::SetTilingKey() {
    tilingKey_ = DTYPE_MAP[tilingcontext->GetInputDesc(SORTED_VALUE_INPUT_INDEX)->GetDataType()];
    tilingcontext->SetTilingKey(tilingKey_);
}

void ApplyTopKTopPWithSortedTiling::GetUsedCore() {
    if (batchSize_ <= coreNum_) {
        batchPerCore_ = uint32_t(1);
        usedCoreNum_ = batchSize_;
        tailBatch_ = uint32_t(0);
        return;
    }
    batchPerCore_ = coreNum_ == uint32_t(0) ? batchSize_ : batchSize_ / coreNum_;
    tailBatch_ = batchSize_ % coreNum_;
    usedCoreNum_ = coreNum_;
}
void ApplyTopKTopPWithSortedTiling::CalDataPerCore() {
    uint32_t inputDataTypeByte = DATATYPE_LEN_MAP[tilingcontext->GetInputDesc(SORTED_VALUE_INPUT_INDEX)->GetDataType()];
    uint32_t dataPerBlock = BLOCK_BYTES / inputDataTypeByte;
    dataNumInit_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    dataNumInitAligned_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    ubFactorElement_ = vocabSize_ < K_VALUE_MAX ? vocabSize_ : K_VALUE_MAX;
    ubFactorElementAligned_ = CeilAlign(ubFactorElement_, dataPerBlock);
    tailUbFactorElement_ = vocabSize_ % ubFactorElement_;
    tailUbFactorElementAligned_ = CeilAlign(tailUbFactorElement_, dataPerBlock);

    uint32_t sortedValueBytes = ubFactorElementAligned_ * inputDataTypeByte + K_VALUE_MAX  * inputDataTypeByte;
    uint32_t sortedIndicesBytes = ubFactorElementAligned_ * BYTES_B32 + K_VALUE_MAX  * BYTES_B32;
    uint32_t pBytes = dataPerBlock * inputDataTypeByte;
    uint32_t kBytes = DATA_PER_BLOCK_B32 * BYTES_B32;
    uint32_t outTensorBytes = ubFactorElementAligned_ * inputDataTypeByte;

    calUbSize_ = calUbSize_ - sortedValueBytes - sortedIndicesBytes - pBytes - kBytes - outTensorBytes;
}

void ApplyTopKTopPWithSortedTiling::FillTilingData() {
    tilingData.set_batchSize(batchSize_);
    tilingData.set_vocabSize(vocabSize_);
    tilingData.set_batchPerCore(batchPerCore_);
    tilingData.set_tailBatch(tailBatch_);
    tilingData.set_blockNum(usedCoreNum_);
    tilingData.set_dataNumInit(dataNumInit_);
    tilingData.set_dataNumInitAligned(dataNumInitAligned_);
    tilingData.set_ubFactorElement(ubFactorElement_);
    tilingData.set_ubFactorElementAligned(ubFactorElementAligned_);
    tilingData.set_tailUbFactorElement(tailUbFactorElement_);
    tilingData.set_tailUbFactorElementAligned(tailUbFactorElementAligned_);
    tilingData.set_calUbSize(calUbSize_);
}

void ApplyTopKTopPWithSortedTiling::PrintTilingData() {
    OP_LOGD(opName_, "batchSize: %u.", tilingData.get_batchSize());
    OP_LOGD(opName_, "vocabSize: %u.", tilingData.get_vocabSize());
    OP_LOGD(opName_, "batchPerCore: %u.", tilingData.get_batchPerCore());
    OP_LOGD(opName_, "tailBatch: %u.", tilingData.get_tailBatch());
    OP_LOGD(opName_, "usedCoreNum: %u.", tilingData.get_blockNum());
    OP_LOGD(opName_, "dataNumInit_: %u.", tilingData.get_dataNumInit());
    OP_LOGD(opName_, "dataNumInitAligned_: %u.", tilingData.get_dataNumInitAligned());
    OP_LOGD(opName_, "ubFactorElement: %u.", tilingData.get_ubFactorElement());
    OP_LOGD(opName_, "ubFactorElementAligned: %u.", tilingData.get_ubFactorElementAligned());
    OP_LOGD(opName_, "tailUbFactorElement: %u.", tilingData.get_tailUbFactorElement());
    OP_LOGD(opName_, "tailUbFactorElementAligned: %u.", tilingData.get_tailUbFactorElementAligned());
    OP_LOGD(opName_, "calUbSize: %u.", tilingData.get_calUbSize());
}

ge::graphStatus ApplyTopKTopPWithSortedTiling::RunKernelTiling() {
    // OP_LOGD(opName_, "TilingForApplyTopKTopPWithSorted start.");

    SetTilingKey();
    GetUsedCore();
    CalDataPerCore();
    FillTilingData();
    // PrintTilingData();

    // OP_LOGD(opName_, "tilingKey: %u.", tilingKey_);
    uint32_t syncWorkspaceSize = SYS_WORKSPACESIZE;
    size_t* currentWorkspace = tilingcontext->GetWorkspaceSizes(1);
    currentWorkspace[0] = syncWorkspaceSize;
    tilingData.SaveToBuffer(tilingcontext->GetRawTilingData()->GetData(),
                            tilingcontext->GetRawTilingData()->GetCapacity());
    tilingcontext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingcontext->SetBlockDim(usedCoreNum_);
    
    // OP_LOGD(opName_, "TilingForApplyTopKTopPWithSorted end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForApplyTopKTopPWithSorted(gert::TilingContext* context) {
    ApplyTopKTopPWithSortedTiling tilingObject(context);
    auto ret = tilingObject.Init();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "tiling Init failed.");
        return ge::GRAPH_FAILED;
    }
    ret = tilingObject.RunKernelTiling();
    // OP_LOGD(context->GetNodeName(), "TilingForApplyTopKTopPWithSorted end.");
    return ret;
}

static ge::graphStatus TilingPrepareForApplyTopKTopPWithSorted(gert::TilingParseContext* context) {
    // OP_LOGD(context->GetNodeName(), "TilingPrepareForApplyTopKTopPWithSorted start");
    // auto compileInfo = GetCompileInfoPtr<TilingForApplyTopKTopPWithSortedCompileInfo>(context);
    // OPS_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    // auto platformInfo = context->GetPlatformInfo();
    // OPS_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    // compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    // uint64_t ubSizePlatForm;
    // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    // compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    // OP_TILING_CHECK((compileInfo->ubSizePlatForm <= 0),
    //                     VECTOR_INNER_ERR_REPORT_TILIING(context->GetNodeName(), "Failed to get ub size"),
    //                     return ge::GRAPH_FAILED);
    // OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ubSizePlatForm);
    // uint64_t totalUbSize = 0;
    // platformInfo->GetLocalMemSize(fe::LocalMemType::UB, totalUbSize);
    // OP_LOGD(context->GetNodeName(), "total ub size is %lu", totalUbSize);
    // OP_LOGD(context->GetNodeName(), "TilingPrepareForApplyTopKTopPWithSorted end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ApplyTopKTopPWithSorted)
    .Tiling(TilingForApplyTopKTopPWithSorted)
    .TilingParse<TilingForApplyTopKTopPWithSortedCompileInfo>(TilingPrepareForApplyTopKTopPWithSorted);
} // namespace optiling