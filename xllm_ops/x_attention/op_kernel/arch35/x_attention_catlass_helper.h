/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef X_ATTN_CATLASS_HELPER_H
#define X_ATTN_CATLASS_HELPER_H
#include "shared_infer_catlass_kernel.h"
#include "unshared_infer_catlass_kernel.h"
#include "combine_kernel.h"

template <typename INPUT_T, typename KVLEN_T>
CATLASS_DEVICE void CallSharedInferKernel(const XAttnKernelCommonParams& params, XAttentionTilingData* tilingData) {
    using ArchTag = Arch::Ascend950;
    using ElementQ = INPUT_T;
    using LayoutQ = layout::RowMajor;
    using ElementK = INPUT_T;
    using LayoutK = layout::ColumnMajor;
    using ElementV = INPUT_T;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = INPUT_T;
    using LayoutP = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    // L1TileShape::K must be embdding
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
    // GEMM Block, implement Q @ K^T of Flash Attention Infer
    using DispatchPolicyQK = Gemm::MmadXASharedQK<ArchTag>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
            ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS, void, TileCopyQK, TileMmadQK>;

    // Shared Epilogue Block, update rowsum rowmax and copyOut on lastStackTile 
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAscend950XASharedSoftmax;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, L1TileShape, PType, SType>;

    // GEMM Block, implement P @ V of Flash Attention Infer
    using DispatchPolicyPV = Gemm::MmadXASharedPV<ArchTag>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
            ArchTag, ElementP, LayoutP, ElementV, LayoutV, ElementOTmp, LayoutOTmp, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShape, L0TileShape, ElementP, ElementV, ElementOTmp, void, TileCopyPV, TileMmadPV>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAscend950XASharedRescaleO;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, L1TileShape, OTmpType>;
    
    using SharedFAInferKernel = SharedFaInferKernel<
            BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, KVLEN_T>;

    SharedFAInferKernel sharedInferKernel(tilingData);
    sharedInferKernel(params);
}

template <typename INPUT_T, typename KVLEN_T, typename TABLE_T>
CATLASS_DEVICE void CallUnsharedInferKernel(const XAttnKernelCommonParams& params, XAttentionTilingData* tilingData) {
    using ArchTag = Arch::Ascend950;
    using ElementQ = INPUT_T;
    using LayoutQ = layout::RowMajor;
    using ElementK = INPUT_T;
    using LayoutK = layout::ColumnMajor;
    using ElementV = INPUT_T;
    using LayoutV = layout::RowMajor;
    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = INPUT_T;
    using LayoutP = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    // L1TileShape::K must be embdding
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
    // GEMM Block, implement Q @ K^T of Flash Attention Infer
    using DispatchPolicyQK = Gemm::MmadXAUnsharedQK<ArchTag>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
            ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS, void, TileCopyQK, TileMmadQK>;

    // Shared Epilogue Block, update rowsum rowmax and copyOut on lastStackTile 
    using DispatchPolicySoftmax = Epilogue::EpilogueAscend950XAUnsharedSoftmax;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicySoftmax, L1TileShape, PType, SType>;

    // GEMM Block, implement P @ V of Flash Attention Infer
    using DispatchPolicyPV = Gemm::MmadXAUnsharedPV<ArchTag>;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTla<
            ArchTag, ElementP, LayoutP, ElementV, LayoutV, ElementOTmp, LayoutOTmp>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShape, L0TileShape, ElementP, ElementV, ElementOTmp, void, TileCopyPV, TileMmadPV>;

    using UnSharedInferKernel = UnSharedInferKernel<
            BlockMmadQK, BlockMmadPV, EpilogueSoftmax, KVLEN_T, TABLE_T>;
    
    UnSharedInferKernel unsharedInferKernel(tilingData);
    unsharedInferKernel(params);
}


template <typename INPUT_T>
CATLASS_DEVICE void CallCombineScale(const XAttnKernelCommonParams& params, XAttentionTilingData* tilingData) {
    using DispatchPolicyCombine = Epilogue::EpilogueAscend950XACombineScale;
    using ElementInput = float;
    using LayoutInput = layout::RowMajor;
    using ElementOutput = INPUT_T;
    using LayoutOutput = layout::RowMajor;
    using InputType = Gemm::GemmType<ElementInput, LayoutInput>;
    using OutputType = Gemm::GemmType<ElementOutput, LayoutOutput>;
    using EpilogueCombineScale = Epilogue::Block::BlockEpilogue<DispatchPolicyCombine, OutputType, InputType>;

    using CombineKernel = CombineScaleKernel<EpilogueCombineScale>;
    CombineKernel combineKernel(tilingData);
    combineKernel(params);
}
#endif
