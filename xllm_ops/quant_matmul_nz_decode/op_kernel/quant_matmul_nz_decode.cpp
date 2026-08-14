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

#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_operator.h"
#include "quant_matmul_nz_workspace.hpp"

using namespace Catlass;

template <uint32_t kL1TileN,
          uint32_t kL1TileK,
          uint32_t kL0TileN = 128,
          uint32_t kL0TileK = 256,
          bool kEnableShuffleK = false>
__aicore__ inline void RunQuantMatmulNzDecode(GM_ADDR x,
                                              GM_ADDR weight,
                                              GM_ADDR scale,
                                              GM_ADDR bias,
                                              GM_ADDR y,
                                              GM_ADDR workspace,
                                              uint32_t m,
                                              uint32_t k,
                                              uint32_t n) {
  using ArchTag = Arch::AtlasA2;
  using L1TileShape = GemmShape<16, kL1TileN, kL1TileK>;
  using L0TileShape = GemmShape<16, kL0TileN, kL0TileK>;
  using AType = Gemm::GemmType<int8_t, layout::RowMajor>;
  using BType = Gemm::GemmType<int8_t, layout::zN>;
  using CType = Gemm::GemmType<int32_t, layout::RowMajor>;
  using DispatchPolicy =
      Gemm::MmadAtlasA2PreloadAsyncWithCallback<1,
                                                2,
                                                2,
                                                2,
                                                1,
                                                false,
                                                kEnableShuffleK>;
  using BlockMmad = Gemm::Block::
      BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

  using EpiloguePolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<2>;
  using ScaleType = Gemm::GemmType<float, layout::VectorLayout>;
  using PerTokenScaleType = Gemm::GemmType<float, layout::VectorLayout>;
  using DType = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
  using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
  using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
  using ColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
  using EpilogueTileShape = MatrixShape<16, 256>;
  using TileRowBroadcastMul = Epilogue::Tile::
      TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
  using TileBroadcastOneBlk = Epilogue::Tile::
      TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
  using TileColumnBroadcastMul =
      Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag,
                                                   ColumnBroadcastMulType,
                                                   EpilogueTileShape>;
  using TileCopy = Epilogue::Tile::
      TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
  using EpilogueScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
  using BlockEpilogue = Epilogue::Block::BlockEpilogue<EpiloguePolicy,
                                                       CType,
                                                       ScaleType,
                                                       PerTokenScaleType,
                                                       DType,
                                                       TileRowBroadcastMul,
                                                       TileBroadcastOneBlk,
                                                       TileColumnBroadcastMul,
                                                       TileCopy,
                                                       EpilogueScheduler>;
  using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
  using MatmulKernel = Gemm::Kernel::
      QuantMatmulNzWorkspace<BlockMmad, BlockEpilogue, BlockScheduler, 2>;

  GemmCoord problem_shape{m, n, k};
  layout::RowMajor layout_x{m, k};
  auto layout_weight = layout::zN::MakeLayout<int8_t>(k, n);
  layout::VectorLayout layout_scale{n};
  layout::VectorLayout unused_layout{m};
  layout::RowMajor layout_y{m, n};
  typename MatmulKernel::Params params{problem_shape,
                                       x,
                                       layout_x,
                                       weight,
                                       layout_weight,
                                       scale,
                                       layout_scale,
                                       scale,
                                       unused_layout,
                                       bias,
                                       y,
                                       layout_y,
                                       AscendC::GetUserWorkspace(workspace)};
  MatmulKernel matmul;
  matmul(params);
}

extern "C" __global__ __aicore__ void quant_matmul_nz_decode(GM_ADDR x,
                                                             GM_ADDR weight,
                                                             GM_ADDR scale,
                                                             GM_ADDR bias,
                                                             GM_ADDR y,
                                                             GM_ADDR workspace,
                                                             GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  GET_TILING_DATA(tiling_data, tiling);

  const uint32_t m = tiling_data.m;
  const uint32_t k = tiling_data.k;
  const uint32_t n = tiling_data.n;
  if (TILING_KEY_IS(0)) {
    RunQuantMatmulNzDecode<320, 512>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(5)) {
    RunQuantMatmulNzDecode<160, 512>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(1)) {
    RunQuantMatmulNzDecode<64, 3200, 64, 512, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(3)) {
    RunQuantMatmulNzDecode<128, 1792, 128, 256, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(2)) {
    RunQuantMatmulNzDecode<256, 896, 128, 256, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  } else if (TILING_KEY_IS(4)) {
    RunQuantMatmulNzDecode<128, 1536, 128, 256, true>(
        x, weight, scale, bias, y, workspace, m, k, n);
  }
}

// The generated mixed-core wrapper clears workspace through matmul helpers.
#include "lib/matmul_intf.h"

namespace AscendC {

__aicore__ inline void PrepareQuantMatmulNzMixedCoreWorkspace(__gm__ uint8_t*) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
  SetAtomicNone();
  SetMaskNorm();
  SetLoadDataBoundary(static_cast<uint64_t>(0));
  SetLoadDataPaddingValue(static_cast<uint64_t>(0));
  NotifyEvent<PIPE_MTE3>(WORKSPACE_SYNC_ID);
#endif
}

}  // namespace AscendC

// CATLASS uses FFTS cross-core flags directly and does not use the KFC message
// queues. Preserve the wrapper's completion notification, but skip clearing
// the unused 30 KiB KFC region on every AIC.
#define clearWorkspace(workspace) \
  PrepareQuantMatmulNzMixedCoreWorkspace(workspace)
