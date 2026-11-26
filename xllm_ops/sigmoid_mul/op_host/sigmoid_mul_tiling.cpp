/*!
 * \file sigmoid_mul_tiling.cpp
 * \brief SigmoidMul tiling (arch22).
 */
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sigmoid_mul_tiling_data.h"
#include "../op_kernel/sigmoid_mul_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr size_t WORKSPACE_NUM = 1;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    // Platform is constant per process; cache after first query to avoid per-call cost.
    static uint64_t cachedUb = 0;
    static int64_t cachedCores = 0;
    if (cachedUb != 0 && cachedCores != 0) {
        *ubSize = cachedUb;
        *coreNum = cachedCores;
        return ge::GRAPH_SUCCESS;
    }
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    OP_CHECK_IF(*ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    cachedUb = *ubSize;
    cachedCores = *coreNum;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t* totalIdx, int64_t* rowWidth,
                                         int32_t* isBroadcast, ge::DataType* dataType)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    auto inputG = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputG);
    auto inputShapeG = EnsureNotScalar(inputG->GetStorageShape());

    *totalIdx = inputShapeX.GetShapeSize();
    *rowWidth = inputShapeX.GetDim(inputShapeX.GetDimNum() - 1);
    *isBroadcast = (inputShapeG.GetShapeSize() != *totalIdx) ? 1 : 0;

    const std::set<ge::DataType> supportedDtype = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(supportedDtype.count(*dataType) == 0, OP_LOGE(context, "invalid dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SigmoidMulTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    int64_t totalIdx;
    int64_t rowWidth;
    int32_t isBroadcast;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, &totalIdx, &rowWidth, &isBroadcast, &dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    SigmoidMulTilingData* tiling = context->GetTilingData<SigmoidMulTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SigmoidMulTilingData), 0, sizeof(SigmoidMulTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalIdx;
    tiling->rowWidth = rowWidth;
    tiling->isBroadcast = isBroadcast;

    if (totalIdx == 0) {
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), 0);
        return ge::GRAPH_SUCCESS;
    }

    int64_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    int64_t usedCoreNum;
    uint32_t useDoubleBuffer = (totalIdx > MIN_SPLIT_THRESHOLD) ? 1 : 0;
    int64_t bufNum = useDoubleBuffer ? 2 : 1;
    // UB bytes per element for one buffer set: x(T)+gate(T)+x32+g32+z32+out(T).
    // 2-byte dtype -> 18; 4-byte dtype (fp32) -> 24.
    int64_t bytesPerElem = (dataType == ge::DT_FLOAT) ? 24 : 18;
    int64_t ubCapacity = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / bytesPerElem, bufNum), ubBlockSize);
    if (ubCapacity < ubBlockSize) {
        ubCapacity = ubBlockSize;
    }

    if (isBroadcast) {
        // rows split across cores; within a row, tile by a UB budget that also
        // reserves space for this core's gate slice (inputQueueG + sigGateBuf).
        int64_t rows = CeilDiv(totalIdx, rowWidth);
        int64_t rowsPerCore = CeilDiv(rows, coreNum);
        usedCoreNum = (rows > coreNum) ? coreNum : rows;
        tiling->blockFactor = rowsPerCore;
        int64_t typeBytes = (dataType == ge::DT_FLOAT) ? 4 : 2;
        int64_t gateVecLen = ((rowsPerCore + 127) / 128) * 128;  // matches kernel gateVecLen_
        int64_t gateBytes = gateVecLen * (typeBytes + 4);         // gate(T) + sigGate(fp32)
        int64_t remain = static_cast<int64_t>(ubSize) - gateBytes;
        if (remain < 0) {
            remain = 0;
        }
        int64_t bcastCap = FloorAlign(FloorDiv(remain / bytesPerElem, bufNum), ubBlockSize);
        if (bcastCap < ubBlockSize) {
            bcastCap = ubBlockSize;
        }
        tiling->ubFactor = (bcastCap < rowWidth) ? bcastCap : rowWidth;
    } else {
        // flat element split across cores
        tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
        usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);
        tiling->ubFactor = ubCapacity;
    }

    context->SetBlockDim(usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), useDoubleBuffer);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoidMul([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SigmoidMulCompileInfo {};

IMPL_OP_OPTILING(SigmoidMul).Tiling(SigmoidMulTilingFunc).TilingParse<SigmoidMulCompileInfo>(TilingParseForSigmoidMul);

} // namespace optiling
