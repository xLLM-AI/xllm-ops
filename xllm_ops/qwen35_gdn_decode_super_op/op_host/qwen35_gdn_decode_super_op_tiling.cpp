#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus Qwen35GdnDecodeSuperOpTiling(gert::TilingContext* context)
{
    constexpr uint32_t kBlockDim = 20;
    context->SetBlockDim(kBlockDim);
    if (context->SetScheduleMode(1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();
    context->GetRawTilingData()->SetDataSize(0);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Qwen35GdnDecodeSuperOp).Tiling(Qwen35GdnDecodeSuperOpTiling);

} // namespace optiling
