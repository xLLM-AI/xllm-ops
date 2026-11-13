/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#include <algorithm>
#include <cmath>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ada_rms_norm_tiling.h"
#include "../op_kernel/ada_rms_norm.h"

namespace optiling {
namespace {
bool BroadcastStrides(const gert::Shape& x, const gert::Shape& tensor,
    uint64_t& batch, uint64_t& seq)
{
    batch = seq = 0;
    const auto rank = tensor.GetDimNum();
    const int64_t d = x.GetDim(x.GetDimNum() - 1);
    if (rank < 1 || tensor.GetDim(rank - 1) != d) {
        return false;
    }
    if (rank == 1) {
        return true;
    }
    if (x.GetDimNum() == 2) {
        if (rank != 2 || (tensor.GetDim(0) != 1 && tensor.GetDim(0) != x.GetDim(0))) {
            return false;
        }
        seq = tensor.GetDim(0) == 1 ? 0 : static_cast<uint64_t>(d);
        return true;
    }
    if (rank == 2) {
        if (tensor.GetDim(0) != 1 && tensor.GetDim(0) != x.GetDim(0)) {
            return false;
        }
        batch = tensor.GetDim(0) == 1 ? 0 : static_cast<uint64_t>(d);
        return true;
    }
    if (rank != 3 || (tensor.GetDim(0) != 1 && tensor.GetDim(0) != x.GetDim(0)) ||
        (tensor.GetDim(1) != 1 && tensor.GetDim(1) != x.GetDim(1))) {
        return false;
    }
    batch = tensor.GetDim(0) == 1 ? 0 : static_cast<uint64_t>(tensor.GetDim(1)) * d;
    seq = tensor.GetDim(1) == 1 ? 0 : static_cast<uint64_t>(d);
    return true;
}
uint32_t AlignBytes(uint64_t bytes)
{
    return static_cast<uint32_t>((bytes + ADA_RMS_NORM_ALIGNMENT - 1) /
        ADA_RMS_NORM_ALIGNMENT * ADA_RMS_NORM_ALIGNMENT);
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr ||
        context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputShape(2) == nullptr || context->GetOutputShape(0) == nullptr ||
        context->GetInputDesc(0) == nullptr || context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& x = context->GetInputShape(0)->GetOriginShape();
    const auto rank = x.GetDimNum();
    if (rank != 2 && rank != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t d = x.GetDim(rank - 1);
    if (d < 1 || d > ADA_RMS_NORM_MAX_D) {
        return ge::GRAPH_FAILED;
    }
    uint64_t rows = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        const int64_t size = x.GetDim(i);
        if (size < 0 || (size > 0 && rows > std::numeric_limits<uint64_t>::max() /
                                                   static_cast<uint64_t>(size))) {
            return ge::GRAPH_FAILED;
        }
        rows *= static_cast<uint64_t>(size);
    }
    if (rows > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(d)) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = context->GetInputDesc(0)->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 1; i < 3; ++i) {
        if (context->GetInputDesc(i) == nullptr || context->GetInputDesc(i)->GetDataType() != dtype) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto* weight = context->GetOptionalInputDesc(3);
    if (weight != nullptr) {
        const auto* shape = context->GetOptionalInputShape(3);
        if (weight->GetDataType() != dtype || shape == nullptr ||
            shape->GetOriginShape().GetDimNum() != 1 || shape->GetOriginShape().GetDim(0) != d) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto* epsilon = context->GetAttrs()->GetAttrPointer<float>(0);
    if (epsilon == nullptr || !std::isfinite(*epsilon) || *epsilon <= 0.0f) {
        return ge::GRAPH_FAILED;
    }
    const auto& output = context->GetOutputShape(0)->GetOriginShape();
    if (output.GetDimNum() != rank || context->GetOutputDesc(0) == nullptr ||
        context->GetOutputDesc(0)->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (output.GetDim(i) != x.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    uint64_t scaleBatch, scaleSeq, shiftBatch, shiftSeq;
    if (!BroadcastStrides(x, context->GetInputShape(1)->GetOriginShape(), scaleBatch, scaleSeq) ||
        !BroadcastStrides(x, context->GetInputShape(2)->GetOriginShape(), shiftBatch, shiftSeq)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t cores = platform.GetCoreNumAiv();
    uint64_t ub = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    const uint32_t rawBytes = AlignBytes(d * (dtype == ge::DT_FLOAT ? 4 : 2));
    const uint32_t floatBytes = AlignBytes(d * sizeof(float));
    if (cores == 0 || 2ULL * rawBytes + 3ULL * floatBytes + ADA_RMS_NORM_SCALAR_BYTES +
                                         ADA_RMS_NORM_UB_RESERVE > ub) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t wanted = std::min<uint64_t>(std::max<uint64_t>(1, rows), cores);
    const uint64_t rowsPerCore = rows == 0 ? 1 : rows / wanted + (rows % wanted != 0);
    AdaRmsNormTilingData tiling;
    tiling.set_rows(rows);
    tiling.set_rowsPerCore(rowsPerCore);
    tiling.set_seqLength(static_cast<uint64_t>(x.GetDim(rank == 3 ? 1 : 0)));
    tiling.set_scaleBatchStride(scaleBatch);
    tiling.set_scaleSeqStride(scaleSeq);
    tiling.set_shiftBatchStride(shiftBatch);
    tiling.set_shiftSeqStride(shiftSeq);
    tiling.set_hiddenDim(static_cast<uint32_t>(d));
    tiling.set_rawRowBytes(rawBytes);
    tiling.set_floatRowBytes(floatBytes);
    tiling.set_hasWeight(weight != nullptr ? 1 : 0);
    tiling.set_epsilon(*epsilon);
    context->SetBlockDim(rows == 0 ? 1 : static_cast<uint32_t>(rows / rowsPerCore + (rows % rowsPerCore != 0)));
    // The single unspecialized kernel is exported as function entry zero.
    context->SetTilingKey(0);
    auto* buffer = context->GetRawTilingData();
    if (buffer == nullptr || buffer->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(buffer->GetData(), buffer->GetCapacity());
    buffer->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(AdaRmsNorm).Tiling(TilingFunc);
} // namespace optiling
