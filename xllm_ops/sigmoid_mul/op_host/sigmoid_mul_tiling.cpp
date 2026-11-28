/*!
 * \file sigmoid_mul_tiling.cpp
 * \brief SigmoidMul tiling (arch22): 2-D bf16, two kernels selected by MODE
 *        (0 flat element-wise, 1 [M,1] broadcast).
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
constexpr int64_t PER_CORE_FLOOR = 4096;    // flat: each core owns at least this many elements
constexpr int64_t UB_BYTES_PER_ELEM = 18;   // bf16 x/gate/out (6 B) + fp32 xf/gf/zf (12 B)

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
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
    auto inputG = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputG);
    const gert::Shape& xShape = inputX->GetStorageShape();
    const gert::Shape& gShape = inputG->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 2, OP_LOGE(context, "SigmoidMul: x must be 2-D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gShape.GetDimNum() != 2, OP_LOGE(context, "SigmoidMul: gate must be 2-D"), return ge::GRAPH_FAILED);

    *totalIdx = xShape.GetShapeSize();
    *rowWidth = xShape.GetDim(1);
    // Classify by dimensions (robust for empty tensors where both sizes are 0).
    *isBroadcast = (gShape.GetDim(1) == 1) ? 1 : 0;
    if (*isBroadcast) {
        OP_CHECK_IF(gShape.GetDim(0) != xShape.GetDim(0),
                    OP_LOGE(context, "SigmoidMul: broadcast gate must be [M, 1]"), return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(!(gShape.GetDim(0) == xShape.GetDim(0) && gShape.GetDim(1) == xShape.GetDim(1)),
                    OP_LOGE(context, "SigmoidMul: element-wise gate must match x shape"), return ge::GRAPH_FAILED);
    }
    // Trailing dim: 32B aligned (16 bf16 elements) and within [128, 16384].
    OP_CHECK_IF(*rowWidth < 128 || *rowWidth > 16384 || (*rowWidth % 16) != 0,
                OP_LOGE(context, "SigmoidMul: tail dim N must be 32B aligned and in [128, 16384]"),
                return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    *dataType = inputDesc->GetDataType();
    OP_CHECK_IF(*dataType != ge::DT_BF16, OP_LOGE(context, "SigmoidMul: only bf16 supported"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* ws = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = 0;
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
        tiling->blockFactor = 1;
        tiling->ubFactor = 2048;  // keep InitBuffer sizes valid for the empty launch
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), static_cast<uint32_t>(isBroadcast));
        return ge::GRAPH_SUCCESS;
    }

    const int64_t ubBlock = Ops::Base::GetUbBlockSize(context);
    // The kernel pads the vector compute and buffer sizes to 128 elements, so
    // budget the UB capacity at 128-element alignment to avoid over-committing UB.
    const int64_t VEC_ALIGN = 128;
    const int64_t bufNum = 2;
    int64_t ubCapacity = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / UB_BYTES_PER_ELEM, bufNum), VEC_ALIGN);
    if (ubCapacity < VEC_ALIGN) {
        ubCapacity = VEC_ALIGN;
    }

    int64_t usedCoreNum;
    if (isBroadcast) {
        const int64_t rows = totalIdx / rowWidth;
        int64_t blockDim = (rows < coreNum) ? rows : coreNum;
        if (blockDim < 1) {
            blockDim = 1;
        }
        const int64_t rowsPerCore = CeilDiv(rows, blockDim);
        tiling->blockFactor = rowsPerCore;
        usedCoreNum = CeilDiv(rows, rowsPerCore);
        int64_t tile = (ubCapacity < rowWidth) ? ubCapacity : rowWidth;
        tiling->ubFactor = (tile < ubBlock) ? ubBlock : tile;
    } else {
        int64_t blockDim = CeilDiv(totalIdx, PER_CORE_FLOOR);
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
        if (blockDim < 1) {
            blockDim = 1;
        }
        tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, blockDim), VEC_ALIGN);
        usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);
        int64_t tile = (ubCapacity < tiling->blockFactor) ? ubCapacity : tiling->blockFactor;
        tiling->ubFactor = (tile < VEC_ALIGN) ? VEC_ALIGN : tile;
    }

    context->SetBlockDim(usedCoreNum);
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType), static_cast<uint32_t>(isBroadcast));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSigmoidMul([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SigmoidMulCompileInfo {};

IMPL_OP_OPTILING(SigmoidMul).Tiling(SigmoidMulTilingFunc).TilingParse<SigmoidMulCompileInfo>(TilingParseForSigmoidMul);

} // namespace optiling
