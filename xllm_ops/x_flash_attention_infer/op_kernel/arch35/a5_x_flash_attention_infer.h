/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_H
#define XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_H

#include "a5_x_flash_attention_infer_kernel.h"

namespace XllmOps {
namespace XfaArch35 {

// A5(Ascend950/DAV_3510) dispatch helper.
// This is a device-callable (non-global) helper that inlines the example49 FAInferTla
// assembly logic, so the extern "C" global entry can dispatch into it directly.
// Parameter order follows the xllm_ops op_kernel entry signature.
template <class Dtype, bool enableMaskFlag = false, bool enablePaFlag = false>
CATLASS_DEVICE void FAInferA5Dispatch(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR mask, GM_ADDR block_table,
    GM_ADDR actual_q_lens, GM_ADDR actual_kv_lens, GM_ADDR attn_out, GM_ADDR tiling)
{
    using namespace Catlass;
    using ArchTag = Arch::Ascend950;
    using ElementQ = Dtype;
    using LayoutTagQ = layout::RowMajor;
    using ElementK = Dtype;
    using LayoutTagK = layout::ColumnMajor;
    using ElementV = Dtype;
    using LayoutTagV = layout::RowMajor;
    using ElementS = float;
    using LayoutTagS = layout::RowMajor;
    using ElementP = Dtype;
    using LayoutTagP = layout::zN;
    using ElementO = Dtype;
    using LayoutTagO = layout::RowMajor;
    using ElementMask = uint8_t;
    using LayoutTagMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutTagOTmp = layout::RowMajor;
    // L1TileShape::K must be embedding
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
    // GEMM Block: Flash Attention Infer Q * K^T
    using DispatchPolicyQK = Gemm::MmadFAIQK<ArchTag, enablePaFlag>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementS, LayoutTagS, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS, void, TileCopyQK, TileMmadQK>;

    // Epilogue Block: online softmax on current S base block
    using DispatchPolicySoftmax = Epilogue::EpilogueAscend950FASoftmax<enableMaskFlag>;
    using PType = Gemm::GemmType<ElementP, LayoutTagP>;
    using SType = Gemm::GemmType<ElementS, LayoutTagS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutTagMask>;
    using EpilogueOnlineSoftmax =
  Epilogue::Block::BlockEpilogue<DispatchPolicySoftmax, L1TileShape, PType, SType, maskType>;

    // GEMM Block: Flash Attention Infer P * V
    using DispatchPolicyPV = Gemm::MmadFAIPV<ArchTag, enablePaFlag>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV, ElementOTmp, LayoutTagV, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShape, L0TileShape, ElementP, ElementV, ElementOTmp, void, TileCopyPV, TileMmadPV>;

    // Epilogue Block: O base block rescale/update
    using DispatchPolicyRescaleO = Epilogue::EpilogueAscend950FARescaleO;
    using OType = Gemm::GemmType<ElementO, LayoutTagO>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutTagOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, L1TileShape, OType, OTmpType>;

    using FAInferKernelType =
        FAInferKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, enablePaFlag>;
    FAIKernelParams params{
        query, key_cache, value_cache, mask, block_table, actual_q_lens, actual_kv_lens, attn_out, tiling};
    // call kernel
    FAInferKernelType flashAttnInfer;
    flashAttnInfer(params);
}

}  // namespace XfaArch35
}  // namespace XllmOps

#endif  // XLLM_OPS_XFAI_ARCH35_A5_X_FLASH_ATTENTION_INFER_H