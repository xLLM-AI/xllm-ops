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

#ifndef CROSS_ATTN_CATLASS_HELPER_H
#define CROSS_ATTN_CATLASS_HELPER_H

// A2 (AtlasA2) path. Never compiled into the Ascend950 (arch35 / DAV_3510)
// translation unit: cross_attention.cpp routes on __NPU_ARCH__ == 3510 and
// includes the arch35 fusion helper instead.
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
#include "cross_attention_catlass_kernel.h"

template <typename INPUT_T, typename TABLE_T, typename KVLEN_T>
CATLASS_DEVICE void CallCrossAttention(const CrossAttnKernelParams& params, CrossAttentionTilingData* tilingData) {
    using ArchTag = Arch::AtlasA2;
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
    using ElementO = INPUT_T;
    using LayoutO = layout::RowMajor;
    using ElementMask = INPUT_T;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;
    // L1TileShape::K must be embdding
    using L1TileShape = GemmShape<128, 128, 128>;
    using L0TileShape = L1TileShape;
    // GEMM Block, implement Q @ K^T of Flash Attention Infer
    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQK<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShape, L0TileShape, QType, KType, SType>;

    using DispatchPolicyQKTail = Gemm::MmadAtlasA2FAITailQK<true, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQKTail = Gemm::Block::BlockMmad<DispatchPolicyQKTail, L1TileShape, L0TileShape, QType, KType, SType>;

    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    // GEMM Block, implement P @ V of Flash Attention Infer
    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPV<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    using DispatchPolicyPVTail = Gemm::MmadAtlasA2FAITailPV<true, false>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPVTail =
        Gemm::Block::BlockMmad<DispatchPolicyPVTail, L1TileShape, L0TileShape, PType, VType, OTmpType>;

    // Epilogue RescaleO
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType>;

    using CrossAttnInferKernel = CrossAttnInferKernel<
        BlockMmadQK, BlockMmadPV, BlockMmadQKTail, BlockMmadPVTail, EpilogueOnlineSoftmax, EpilogueRescaleO, TABLE_T, KVLEN_T>;
    CrossAttnInferKernel crossAttnInferKernel(tilingData);
    crossAttnInferKernel(params);
}

#endif // A2 (AtlasA2) compilation unit only
#endif
