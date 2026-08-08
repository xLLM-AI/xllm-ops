/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_example_tiling.cpp
 * \brief
 */

#include "log/ops_log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "../op_kernel/add_example_tiling_data.h"
#include "../op_kernel/add_example_tiling_key.h"

namespace optiling {

using namespace Ops::Xllm::OpTiling;

const uint32_t BLOCK_DIM = 8;
const int64_t TILE_NUM = 8;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int32_t DIMS_LIMIT = 4;
constexpr int32_t ATTRPOS0 = 0;
constexpr uint32_t INDEXZERO = 0;
constexpr uint32_t INDEXONE = 1;
constexpr uint32_t INDEXTWO = 2;
constexpr uint32_t INDEXTHREE = 3;

struct AddExampleCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platformInfoPtr, return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OPS_CHECK(coreNum == 0, OPS_LOG_E(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OPS_CHECK(ubSize == 0, OPS_LOG_E(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto inputX = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL(context, inputX, return ge::GRAPH_FAILED);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    auto inputY = context->GetInputShape(1);
    OPS_LOG_E_IF_NULL(context, inputY, return ge::GRAPH_FAILED);
    auto inputShapeY = EnsureNotScalar(inputY->GetStorageShape());
    auto outZ = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL(context, outZ, return ge::GRAPH_FAILED);
    auto outShapeZ = EnsureNotScalar(outZ->GetStorageShape());

    // shape校验
    OPS_CHECK(
        inputShapeX.GetDimNum() != DIMS_LIMIT || inputShapeY.GetDimNum() != DIMS_LIMIT ||
            outShapeZ.GetDimNum() != DIMS_LIMIT,
        OPS_LOG_E(
            context, "AddExample: inputx,inputy,outputz shape dim = %zu, %zu, %zu, should be equal 4",
            inputShapeX.GetDimNum(), inputShapeY.GetDimNum(), outShapeZ.GetDimNum()),
        return ge::GRAPH_FAILED);

    // 获取shape dim值
    auto nDim = inputShapeX.GetDim(INDEXZERO);
    auto cDim = inputShapeX.GetDim(INDEXONE);
    auto hDim = inputShapeX.GetDim(INDEXTWO);
    auto wDim = inputShapeX.GetDim(INDEXTHREE);
    totalIdx = nDim * cDim * hDim * wDim;
    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_INT32};
    auto inputDesc = context->GetInputDesc(0);
    OPS_LOG_E_IF_NULL(context, inputDesc, return ge::GRAPH_FAILED);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OPS_LOG_E(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL(context, currentWorkspace, return ge::GRAPH_FAILED);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus AddExampleTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    OPS_CHECK(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OPS_LOG_E(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    int64_t totalIdx;
    ge::DataType dataType;

    OPS_CHECK(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OPS_LOG_E(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 3、获取WorkspaceSize信息
    OPS_CHECK(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OPS_LOG_E(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    AddExampleTilingData* tiling = context->GetTilingData<AddExampleTilingData>();
    OPS_LOG_E_IF_NULL(context, tiling, return ge::GRAPH_FAILED);
    OPS_CHECK(
        memset_s(tiling, sizeof(AddExampleTilingData), 0, sizeof(AddExampleTilingData)) != EOK,
        OPS_LOG_E(context, "set tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalLength = totalIdx;
    tiling->tileNum = TILE_NUM;

    context->SetBlockDim(BLOCK_DIM);
    uint64_t tilingKey = 0;
    // 区分dtype走不同得tiling key分支.
    if (dataType == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    } else if (dataType == ge::DT_INT32) {
        tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else {
        OPS_LOG_E(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAddExample([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(AddExample).Tiling(AddExampleTilingFunc).TilingParse<AddExampleCompileInfo>(TilingParseForAddExample);
} // namespace optiling
