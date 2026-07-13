#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "qwen35_gdn_decode_super_op_batch_tiling.h"

namespace optiling {

static ge::graphStatus Qwen35GdnDecodeSuperOpBatchTiling(gert::TilingContext* context)
{
    constexpr uint32_t kBlockDim = 20;
    constexpr int64_t kMaxBatchSize = 4;
    const auto* qkvShape = context->GetInputShape(0);
    if (qkvShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t batchSize = qkvShape->GetStorageShape().GetDim(0);
    if (batchSize < 1 || batchSize > kMaxBatchSize) {
        return ge::GRAPH_FAILED;
    }

    Qwen35GdnDecodeSuperOpBatchTilingData tiling;
    tiling.set_batch_size(batchSize);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(kBlockDim);
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Qwen35GdnDecodeSuperOpBatch).Tiling(Qwen35GdnDecodeSuperOpBatchTiling);

} // namespace optiling
