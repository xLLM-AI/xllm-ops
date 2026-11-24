/* Copyright 2025 JD.com
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// KDA implementation introduced in xllm_ops commit be99d50 (2026-09-15).
#pragma once
#include <pto/pto-inst.hpp>
#include <type_traits>

#include "kda_kernel_utils.h"

namespace mega_kda_prefill_wy_kda {
// ============================================================================
// WY representation for KDA chunk recurrence (per-dim gate).
//
// A2, K_eff and both WY matmul inputs stay FP32. Cube loads A2 once into L1
// and reuses it for both GEMMs. L1 storage at D=C=128: 192 KiB.
// Computes the two auxiliary tensors U and W per chunk:
//   U[r, d_v] = sum_c INV[r, c] * beta[c] * v[c, d_v]
//   W[r, d_k] = sum_c INV[r, c] * beta[c] * exp(g_cs[c, d_k]) * k[c, d_k]
//
// where INV = (I + L)^{-1} is full lower-triangular, beta is post-sigmoid
// scalar per token, and g_cs is per-DIMENSION cumulative gate sum (the KDA
// difference from GDN).
//
// Key insight: both U and W reuse a single beta-scaled A2 matrix:
//   A2[r, c]    = INV[r, c] * beta[c]            (column-scale by beta)
//   K_eff[c, d] = k[c, d] * exp(g_cs[c, d])      (element-wise per-dim)
//   U = A2 @ V                                    (V from BSND, FP32)
//   W = A2 @ K_eff                                (K_eff from workspace, FP32)
//
// Cube loads A2 once into L1 and reuses it across both GEMMs — saves one L1
// load and one workspace pass compared to GDN wy_fast.cpp (which needs two
// distinct reweighted-A matrices because GDN's scalar g could be folded into
// a column scale of A). In KDA g is per-dim so we must precompute K_eff
// instead.
//
//  Vec core (both sub-blocks active; each owns one HalfChunk-row stripe):
//    For each (seq, chunk, head):
//      Phase 1: Load beta [1,C], INV [HC,C] BSND -> build A2 = INV * beta_2d
//               -> store FP32 stripe to ws_a2 [block_dim, C, C]
//      Phase 2: Load k [HC,K], g_cs [HC,K] head-major -> K_eff = k*exp(g_cs)
//               -> store FP32 stripe to ws_keff [block_dim, C, K]
//      Signal Cube via cross-core flags 10 and 11.
//
//  Cube core:
//    For each (seq, chunk, head):
//      Load V from BSND -> v_l1
//      Wait flag 10 (ws_a2 ready) -> TLOAD a2_l1
//      GEMM U = A2 @ V -> u_l0 -> store BSND -> signal flag 12 (ws_a2 free)
//      Wait flag 11 (ws_keff ready) -> TLOAD keff_l1
//      GEMM W = A2 @ K_eff (a2_l1 still in L1!) -> w_l0 -> store BSND
//      Signal flag 13 (ws_keff free)
//
// FFTS flags (single-buffered with explicit free signals):
//   10 : V→C reduce (both vids must signal) "ws_a2 ready"
//   11 : V→C reduce "ws_keff ready"
//   12 : C→V broadcast "ws_a2 free"
//   13 : C→V broadcast "ws_keff free"
// Layout (packed tokens when cu_seqlens is present):
//   k       head-major [HV, T, K]    BF16
//   v       BSND       [B, T, HV, V] FP32   (V == K in current setup)
//   beta    head-major [HV, T]       fp32
//   g_cs    head-major [HV, T, K]    fp32
//   A (INV) BSND       [B, T, HV, C] fp32
//   ws_a2              [bd, C, C]    FP32
//   ws_keff            [bd, C, K]    FP32
//   u_out   BSND       [B, T, HV, V] fp32
//   w_out   BSND       [B, T, HV, K] fp32
//
// Compile-time template params: HiddenSize = K = V, ChunkSize = C.
// Runtime argument: num_heads = HV.
// ============================================================================

using namespace pto;
constexpr int32_t kA2Ready = 10;
constexpr int32_t kKeffReady = 11;
constexpr int32_t kA2Free = 12;
constexpr int32_t kKeffFree = 13;
using namespace mega_kda_prefill_utils;

#ifdef __CCE_AICORE__

namespace {

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1 = pto::Tile<pto::TileType::Mat,
                            T,
                            Rows,
                            Cols,
                            pto::BLayout::ColMajor,
                            RowValid,
                            ColValid,
                            pto::SLayout::RowMajor,
                            512,
                            pto::PadValue::Zero>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL1ZN = pto::Tile<pto::TileType::Mat,
                              T,
                              Rows,
                              Cols,
                              pto::BLayout::RowMajor,
                              RowValid,
                              ColValid,
                              pto::SLayout::ColMajor,
                              512,
                              pto::PadValue::Zero>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0A =
    pto::Tile<pto::TileType::Left,
              T,
              Rows,
              Cols,
              mega_kda_prefill_utils::GetOuterLayout(/*is_left=*/true),
              RowValid,
              ColValid,
              pto::SLayout::RowMajor,
              512,
              pto::PadValue::Zero>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols>
using TileMatL0B =
    pto::Tile<pto::TileType::Right,
              T,
              Rows,
              Cols,
              mega_kda_prefill_utils::GetOuterLayout(/*is_left=*/false),
              RowValid,
              ColValid,
              pto::SLayout::ColMajor,
              512,
              pto::PadValue::Zero>;

template <typename T,
          int Rows,
          int Cols,
          int RowValid = Rows,
          int ColValid = Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using TileUbDataND = pto::Tile<pto::TileType::Vec,
                               T,
                               Rows,
                               Cols,
                               pto::BLayout::RowMajor,
                               RowValid,
                               ColValid,
                               pto::SLayout::NoneBox,
                               512,
                               PadVal>;

using GmShape2D = pto::Shape<1, 1, 1, pto::DYNAMIC, pto::DYNAMIC>;
using GmStride2D = pto::Stride<1, 1, 1, pto::DYNAMIC, 1>;

template <typename T>
using GmTensor2D = pto::GlobalTensor<T, GmShape2D, GmStride2D>;

template <typename T, int32_t Rows, int32_t Cols>
using DynMatL1 = pto::Tile<pto::TileType::Mat,
                           T,
                           Rows,
                           Cols,
                           pto::BLayout::ColMajor,
                           pto::DYNAMIC,
                           pto::DYNAMIC,
                           pto::SLayout::RowMajor,
                           512,
                           pto::PadValue::Zero>;

template <typename T,
          int32_t Rows,
          int32_t Cols,
          pto::PadValue PadVal = pto::PadValue::Null>
using DynVecTile = pto::Tile<pto::TileType::Vec,
                             T,
                             Rows,
                             Cols,
                             pto::BLayout::RowMajor,
                             pto::DYNAMIC,
                             pto::DYNAMIC,
                             pto::SLayout::NoneBox,
                             512,
                             PadVal>;

template <typename T, int32_t Rows, int32_t Cols>
using DynAccTile = pto::TileAcc<T, Rows, Cols, pto::DYNAMIC, pto::DYNAMIC>;

// Local K-sliced matmul helper:  C = A @ B (clear=true) on the Cube engine.
// Copied verbatim from wy_fast.cpp gemm_v0 — handles the L1 -> L0 movement
// and accumulator init/tail logic.  See wy_fast.cpp lines 149-260 for the
// commented version.
template <typename T1,
          typename T2,
          uint32_t M,
          uint32_t N,
          uint32_t K,
          uint32_t validM = M,
          uint32_t validN = N,
          uint32_t validK = K,
          uint32_t K_tail,
          bool transpose_A = false,
          bool transpose_B = false>
AICORE PTO_INLINE void gemm_v0(
    std::conditional_t<transpose_A,
                       TileMatL1<T1, K, M, validK, validM>,
                       TileMatL1<T1, M, K, validM, validK>>& A,
    std::conditional_t<transpose_B,
                       TileMatL1<T1, N, K, validN, validK>,
                       TileMatL1<T1, K, N, validK, validN>>& B,
    pto::TileAcc<T2, M, N, validM, validN>& C,
    bool clear) {
  constexpr uint32_t kL0Size = 128;
  const uint32_t kL0split = (K + kL0Size - 1) / kL0Size;

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);

  for (uint32_t kL0Idx = 0; kL0Idx < kL0split; ++kL0Idx) {
    const bool initflag = clear && (kL0Idx == 0);
    const bool is_tail_block = (kL0Idx == kL0split - 1);

    if (is_tail_block) {
      TileMatL0A<T1, M, K_tail, M, K_tail> l0a;
      TileMatL0B<T1, K_tail, N, K_tail, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * K_tail);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * K_tail);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * K_tail, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * K_tail, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }
    } else {
      TileMatL0A<T1, M, kL0Size, M, kL0Size> l0a;
      TileMatL0B<T1, kL0Size, N, kL0Size, N> l0b;
      pto::TASSIGN(l0a, 0x0);
      pto::TASSIGN(l0b, 0x0);

      set_flag(PIPE_M, PIPE_MTE1, war_event_id);
      wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

      set_flag(PIPE_FIX, PIPE_M, war_event_id);
      wait_flag(PIPE_FIX, PIPE_M, war_event_id);

      if constexpr (!transpose_A) {
        pto::TEXTRACT(l0a, A, 0, kL0Idx * kL0Size);
      } else {
        TileMatL1ZN<T1, M, K, validM, validK> A_t;
        pto::TRESHAPE(A_t, A);
        pto::TEXTRACT(l0a, A_t, 0, kL0Idx * kL0Size);
      }

      if constexpr (!transpose_B) {
        pto::TEXTRACT(l0b, B, kL0Idx * kL0Size, 0);
      } else {
        TileMatL1ZN<T1, K, N, validK, validN> B_t;
        pto::TRESHAPE(B_t, B);
        pto::TEXTRACT(l0b, B_t, kL0Idx * kL0Size, 0);
      }

      set_flag(PIPE_MTE1, PIPE_M, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_M, war_event_id);

      if (initflag) {
        pto::TMATMUL(C, l0a, l0b);
      } else {
        pto::TMATMUL_ACC(C, C, l0a, l0b);
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
      wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
    }
  }

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);

  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

}  // namespace

#endif

AICORE inline void round_wu(__gm__ float* w, __gm__ float* u,
                            int64_t tokens, int32_t heads) {
#if defined(__DAV_VEC__)
  constexpr int32_t kTile = 1024;
  const int32_t core = get_block_idx() * 2 + get_subblockid();
  const int32_t cores = get_block_num() * 2;
  const int64_t count = tokens * heads * 128;
  for (int64_t start = int64_t(core) * kTile; start < count;
       start += int64_t(cores) * kTile) {
    const int32_t n = count - start < kTile ? count - start : kTile;
    GmTensor2D<float> w_gm(w + start, GmShape2D(1, n), GmStride2D(kTile));
    GmTensor2D<float> u_gm(u + start, GmShape2D(1, n), GmStride2D(kTile));
    DynVecTile<float, 1, kTile> w_tile(1, n), u_tile(1, n);
    DynVecTile<bfloat16_t, 1, kTile> typed(1, n);
    TASSIGN(w_tile, 0);
    TASSIGN(u_tile, kTile * sizeof(float));
    TASSIGN(typed, 2 * kTile * sizeof(float));
    TLOAD(w_tile, w_gm);
    TLOAD(u_tile, u_gm);
    SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    round_bf16(w_tile, w_tile, typed);
    round_bf16(u_tile, u_tile, typed);
    SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
    TSTORE(w_gm, w_tile);
    TSTORE(u_gm, u_tile);
    SetWaitFlag<PIPE_MTE3, PIPE_MTE2>(0);
  }
#endif
}

template <int32_t HiddenSize, int32_t ChunkSize, bool UseLog2Gate>
AICORE void wy_kda_kernel(__gm__ bfloat16_t* K_handle,
                          __gm__ float* V_handle,
                          __gm__ float* Beta_handle,
                          __gm__ float* G_handle,
                          __gm__ float* A_handle,
                          __gm__ float* workspace_a2_handle,
                          __gm__ float* workspace_keff_handle,
                          __gm__ float* U_handle,
                          __gm__ float* W_handle,
                          __gm__ const int32_t* cu_seqlens,
                          int64_t batch_size,
                          int64_t total_tokens,
                          int32_t num_heads,
                          uint64_t ffts_addr) {
  // Per-chunk math (matches ref_wy_kda):
  //   A2[r, c]    = INV[r, c] * beta[c]
  //   K_eff[c, d] = k[c, d] * exp(g_cs[c, d])
  //   U = A2 @ V,   W = A2 @ K_eff
  constexpr int32_t HalfChunk = ChunkSize / 2;
  // The matmul inner-dimension here is ChunkSize (not HiddenSize), so KTail
  // must be computed from ChunkSize.  This differs from wy_fast.cpp where
  // HiddenSize and ChunkSize happen to coincide (both 128) so the wrong
  // expression silently worked.
  constexpr uint32_t KTail = (ChunkSize % 128 == 0) ? 128 : (ChunkSize % 128);

  // Head count (HV) is a runtime argument — it only drives loop bounds and GM
  // strides, never a UB buffer size or tile shape.
  const int32_t NumHeads = num_heads;
  const int32_t H = NumHeads;
  // In KDA the keys arrive pre-expanded to HV heads (no GQA at this stage),
  // so K, V, U, W all use stride H * HiddenSize.
  const int32_t BSND_STRIDE = H * HiddenSize;
  const int32_t BSND_C_STRIDE = H * ChunkSize;

  // ── UB address plan ───────────────────────────────────────────────────────
  // Phase 1 (A2 = INV * beta_2d) and Phase 2 (K_eff = k * exp(g_cs)) run
  // sequentially, so their tiles overlap in UB.  Peak usage ≈ 116 KB
  // per Vec sub-block (well under the 192 KB budget).
  //
  // Phase 1:
  constexpr int32_t BetaUbAddr = 0;
  constexpr int32_t BetaRUbAddr = 512;
  constexpr int32_t Beta2dUbAddr = 1024;
  constexpr int32_t InvUbAddr = Beta2dUbAddr + HalfChunk * ChunkSize * 4;
  constexpr int32_t A2UbAddr = InvUbAddr + HalfChunk * ChunkSize * 4;
  // Phase 2 (overlaps with Phase 1 — addresses reused):
  constexpr int32_t KUbAddr = 0;
  constexpr int32_t GcsUbAddr = KUbAddr + HalfChunk * HiddenSize * 4;
  constexpr int32_t KeffUbAddr = GcsUbAddr + HalfChunk * HiddenSize * 4;
  constexpr int32_t KeffHalfUbAddr = KeffUbAddr + HalfChunk * HiddenSize * 4;
  constexpr int32_t VRoundUbAddr = KeffHalfUbAddr + HalfChunk * HiddenSize * sizeof(bfloat16_t);
  constexpr int32_t SeedBetaUbAddr = VRoundUbAddr + HalfChunk * HiddenSize * sizeof(bfloat16_t);

  // Workspace element counts (FP32).
  using Operand = std::conditional_t<UseLog2Gate, bfloat16_t, float>;
  constexpr int32_t WsA2Size = ChunkSize * ChunkSize;
  constexpr int32_t WsKeffSize = ChunkSize * HiddenSize;

  set_ffts_base_addr(ffts_addr);
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  auto vid = get_subblockid();

  int64_t num_seqs = batch_size;

  // ── UB tile declarations (TASSIGNed to fixed addresses) ──────────────────
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize, pto::PadValue::Zero> beta_ub;
  TASSIGN(beta_ub, BetaUbAddr);
  TileUbDataND<float, 1, ChunkSize, 1, ChunkSize> beta_r_ub;
  TASSIGN(beta_r_ub, BetaRUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> beta_2d_ub;
  TASSIGN(beta_2d_ub, Beta2dUbAddr);
  TileUbDataND<float,
               HalfChunk,
               ChunkSize,
               HalfChunk,
               ChunkSize,
               pto::PadValue::Zero>
      inv_ub;
  TASSIGN(inv_ub, InvUbAddr);
  TileUbDataND<float, HalfChunk, ChunkSize, HalfChunk, ChunkSize> a2_ub;
  TASSIGN(a2_ub, A2UbAddr);

  TileUbDataND<float,
               HalfChunk,
               HiddenSize,
               HalfChunk,
               HiddenSize,
               pto::PadValue::Zero>
      k_ub;
  TASSIGN(k_ub, KUbAddr);
  TileUbDataND<float,
               HalfChunk,
               HiddenSize,
               HalfChunk,
               HiddenSize,
               pto::PadValue::Zero>
      gcs_ub;
  TASSIGN(gcs_ub, GcsUbAddr);
  TileUbDataND<float, HalfChunk, HiddenSize, HalfChunk, HiddenSize> keff_ub;
  TASSIGN(keff_ub, KeffUbAddr);

  // ── L1 tiles (Cube only) ──────────────────────────────────────────────────
  // FP32: v_l1 at 0, keff_l1 at 64 KiB, a2_w_l1 at 128 KiB.
  TileMatL1<Operand, ChunkSize, HiddenSize, ChunkSize, HiddenSize> v_l1;
  TASSIGN(v_l1, 0);
  TileMatL1<Operand, ChunkSize, HiddenSize, ChunkSize, HiddenSize> keff_l1;
  TASSIGN(keff_l1, 65536);
  TileMatL1<Operand, ChunkSize, ChunkSize, ChunkSize, ChunkSize> a2_w_l1;
  TASSIGN(a2_w_l1, 131072);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> u_l0;
  TASSIGN(u_l0, 0);
  TileAcc<float, ChunkSize, HiddenSize, ChunkSize, HiddenSize> w_l0;
  TASSIGN(w_l0, 65536);

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  // Vec prepares the two workspaces (ws_a2 holding A2 = INV*beta_2d, and
  // ws_keff holding K_eff = k*exp(g_cs)) that the Cube phase then consumes.
  // `first_iter` is shared across all sequences and the trailing
  // drain below — set to false once this block has processed any work item.
  bool first_iter = true;
  {
    // Packed sequence boundaries are required by the operator contract.
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t local_rows =
                valid_rows - static_cast<int32_t>(vid) * HalfChunk;
            if (local_rows < 0) {
              local_rows = 0;
            }
            if (local_rows > HalfChunk) {
              local_rows = HalfChunk;
            }
            int32_t head_idx = h;

            // ─── Phase 1: build A2 = INV * beta_2d ─────────────────────────
            {
              GmShape2D beta_shape(1, valid_rows);
              GmStride2D beta_stride(1);
              GmTensor2D<float> beta_global(
                  Beta_handle + static_cast<int64_t>(head_idx) * total_tokens +
                      chunk_token_start,
                  beta_shape,
                  beta_stride);
              DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero> beta_load(
                  1, valid_rows);
              TASSIGN(beta_load, BetaUbAddr);
              TLOAD(beta_load, beta_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD_INPLACE(beta_ub, beta_load);
              }
            }

            if (local_rows > 0) {
              int64_t a_gm_offset =
                  ((chunk_token_start + static_cast<int64_t>(vid) * HalfChunk) *
                       NumHeads +
                   head_idx) *
                  static_cast<int64_t>(ChunkSize);
              GmShape2D a_shape(local_rows, ChunkSize);
              GmStride2D a_stride(NumHeads * ChunkSize);
              GmTensor2D<float> a_global(
                  A_handle + a_gm_offset, a_shape, a_stride);
              TileUbDataND<float,
                           HalfChunk,
                           ChunkSize,
                           HalfChunk,
                           ChunkSize,
                           pto::PadValue::Zero>
                  a_stg_full;
              TASSIGN(a_stg_full, InvUbAddr);
              DynVecTile<float, HalfChunk, ChunkSize, pto::PadValue::Zero>
                  a_load(local_rows, ChunkSize);
              TASSIGN(a_load, InvUbAddr);
              TLOAD(a_load, a_global);
              if (local_rows != HalfChunk) {
                TFILLPAD_INPLACE(a_stg_full, a_load);
              }
            } else {
              TEXPANDS(a2_ub, 0.0f);
              PipeBarrierVec();
            }

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if (local_rows > 0) {
              if constexpr (!UseLog2Gate) {
                TMOV(beta_r_ub, beta_ub);
                PipeBarrierVec();
                TCOLEXPAND(beta_2d_ub, beta_r_ub);
                TMUL(a2_ub, inv_ub, beta_2d_ub);
              }
            }
            TileUbDataND<bfloat16_t, HalfChunk, ChunkSize, HalfChunk, ChunkSize>
                inverse_bf16;
            if constexpr (UseLog2Gate) {
              TASSIGN(inverse_bf16, A2UbAddr);
              if (local_rows > 0) {
                TCVT(inverse_bf16, inv_ub, pto::RoundMode::CAST_RINT);
              } else {
                TEXPANDS(inverse_bf16, bfloat16_t(0.0f));
              }
            }

            // Keep A2's FP32 product for both GEMMs.
            PipeBarrierVec();
            // Wait Cube: ws_a2 slot free (flag 12).
            // A2: Cube is a separate core → FFTS cross-core flag.
            // A5: Cube shares the core → intra-block flag.
            if (!first_iter) {
#if __CCE_AICORE__ == 220
              wait_flag_dev(kA2Free);
#else
              wait_intra_block(PIPE_MTE3, kA2Free);
              pipe_barrier(PIPE_ALL);
#endif
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D a2_shape(HalfChunk, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<Operand> workspace_a2_w_global(
                  reinterpret_cast<__gm__ Operand*>(workspace_a2_handle) +
                      static_cast<int64_t>(cid) * WsA2Size +
                      static_cast<int64_t>(vid) * HalfChunk * ChunkSize,
                  a2_shape, a2_stride);
              if constexpr (UseLog2Gate) {
                TSTORE(workspace_a2_w_global, inverse_bf16);
              } else {
                TSTORE(workspace_a2_w_global, a2_ub);
              }
            }
            pipe_barrier(PIPE_ALL);
            // Signal Cube: ws_a2 ready (flag 10)
#if __CCE_AICORE__ == 220
            SetCrossFlag<PIPE_MTE3>(kA2Ready);
#else
            set_intra_block(PIPE_MTE3, kA2Ready);
#endif

            // ─── Phase 2: build K_eff = k * exp(g_cs) ──────────────────────
            if (local_rows > 0) {
              int64_t hk_base =
                  static_cast<int64_t>(head_idx) * total_tokens *
                      static_cast<int64_t>(HiddenSize) +
                  (chunk_token_start + static_cast<int64_t>(vid) * HalfChunk) *
                      static_cast<int64_t>(HiddenSize);
              {
                GmShape2D k_shape(local_rows, HiddenSize);
                GmStride2D k_stride(HiddenSize);
                GmTensor2D<bfloat16_t> k_global(
                    K_handle + hk_base, k_shape, k_stride);
                TileUbDataND<bfloat16_t,
                             HalfChunk,
                             HiddenSize,
                             HalfChunk,
                             HiddenSize,
                             pto::PadValue::Zero>
                    k_stg_full;
                TASSIGN(k_stg_full, KeffHalfUbAddr);
                DynVecTile<bfloat16_t,
                           HalfChunk,
                           HiddenSize,
                           pto::PadValue::Zero>
                    k_load(local_rows, HiddenSize);
                TASSIGN(k_load, KeffHalfUbAddr);
                TLOAD(k_load, k_global);
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(k_stg_full, k_load);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              {
                TileUbDataND<bfloat16_t,
                             HalfChunk,
                             HiddenSize,
                             HalfChunk,
                             HiddenSize>
                    k_stg_cvt;
                TASSIGN(k_stg_cvt, KeffHalfUbAddr);
                TCVT(k_ub, k_stg_cvt, pto::RoundMode::CAST_NONE);
                PipeBarrierVec();
              }
              {
                GmShape2D g_shape(local_rows, HiddenSize);
                GmStride2D g_stride(HiddenSize);
                GmTensor2D<float> g_global(
                    G_handle + hk_base, g_shape, g_stride);
                TileUbDataND<float,
                             HalfChunk,
                             HiddenSize,
                             HalfChunk,
                             HiddenSize,
                             pto::PadValue::Zero>
                    g_stg_full;
                TASSIGN(g_stg_full, GcsUbAddr);
                DynVecTile<float, HalfChunk, HiddenSize, pto::PadValue::Zero>
                    g_load(local_rows, HiddenSize);
                TASSIGN(g_load, GcsUbAddr);
                TLOAD(g_load, g_global);  // g_cs fp32 → gcs_ub directly
                if (local_rows != HalfChunk) {
                  TFILLPAD_INPLACE(g_stg_full, g_load);
                }
              }
              set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
              if constexpr (UseLog2Gate) {
                main_gate_exp(gcs_ub, gcs_ub);
              } else {
                exp_gate<false>(gcs_ub, gcs_ub);
              }
              PipeBarrierVec();
              TMUL(keff_ub, k_ub, gcs_ub);
              PipeBarrierVec();
            } else {
              TEXPANDS(keff_ub, 0.0f);
              PipeBarrierVec();
            }

            TileUbDataND<bfloat16_t, HalfChunk, HiddenSize, HalfChunk, HiddenSize> w_seed, v_seed;
            if constexpr (UseLog2Gate) {
              static_assert(ChunkSize == 64 && HiddenSize == 128);
              TASSIGN(w_seed, KeffHalfUbAddr);
              TASSIGN(v_seed, VRoundUbAddr);
              if (local_rows > 0) {
                // The main path rounds gated K before beta. V is already
                // BF16 when beta is applied. Both seeds are BF16 MMAD inputs.
                TCVT(w_seed, keff_ub, pto::RoundMode::CAST_RINT);
                PipeBarrierVec();
                TCVT(keff_ub, w_seed, pto::RoundMode::CAST_NONE);
                PipeBarrierVec();
                DynVecTile<float, HalfChunk, HiddenSize, pto::PadValue::Zero>
                    v_load(local_rows, HiddenSize);
                TASSIGN(v_load, KUbAddr);
                const int64_t start = chunk_token_start + static_cast<int64_t>(vid) * HalfChunk;
                GmTensor2D<float> value(
                    V_handle + (start * H + head_idx) * HiddenSize,
                    GmShape2D(local_rows, HiddenSize), GmStride2D(BSND_STRIDE));
                TileUbDataND<float, 1, ChunkSize, 1, ChunkSize, pto::PadValue::Zero>
                    seed_beta;
                TASSIGN(seed_beta, SeedBetaUbAddr);
                DynVecTile<float, 1, ChunkSize, pto::PadValue::Zero>
                    seed_beta_load(1, valid_rows);
                TASSIGN(seed_beta_load, SeedBetaUbAddr);
                GmTensor2D<float> beta_src(
                    Beta_handle + int64_t(head_idx) * total_tokens + chunk_token_start,
                    GmShape2D(1, valid_rows), GmStride2D(1));
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                TLOAD(v_load, value);
                TLOAD(seed_beta_load, beta_src);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (local_rows < HalfChunk) {
                  TFILLPAD_INPLACE(k_ub, v_load);
                }
                set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
                for (int row = 0; row < local_rows; ++row) {
                  TileUbDataND<float, 1, HiddenSize, 1, HiddenSize> w_row, v_row;
                  TASSIGN(w_row, KeffUbAddr + row * HiddenSize * sizeof(float));
                  TASSIGN(v_row, KUbAddr + row * HiddenSize * sizeof(float));
                  const float b = seed_beta.data()[static_cast<int>(vid) * HalfChunk + row];
                  TMULS(w_row, w_row, b);
                  TMULS(v_row, v_row, b);
                }
                PipeBarrierVec();
                TCVT(w_seed, keff_ub, pto::RoundMode::CAST_RINT);
                TCVT(v_seed, k_ub, pto::RoundMode::CAST_RINT);
              } else {
                TEXPANDS(w_seed, bfloat16_t(0.0f));
                TEXPANDS(v_seed, bfloat16_t(0.0f));
              }
              PipeBarrierVec();
            }

            // Wait Cube: ws_keff slot free (flag 13).
            if (!first_iter) {
#if __CCE_AICORE__ == 220
              wait_flag_dev(kKeffFree);
#else
              wait_intra_block(PIPE_MTE3, kKeffFree);
              pipe_barrier(PIPE_ALL);
#endif
            }
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            {
              GmShape2D keff_shape(HalfChunk, HiddenSize);
              GmStride2D keff_stride(HiddenSize);
              if constexpr (UseLog2Gate) {
                auto* seeds = reinterpret_cast<__gm__ bfloat16_t*>(workspace_keff_handle) +
                              (2 * static_cast<int64_t>(cid) * WsKeffSize) +
                              static_cast<int64_t>(vid) * HalfChunk * HiddenSize;
                GmTensor2D<bfloat16_t> w_global(seeds, keff_shape, keff_stride);
                GmTensor2D<bfloat16_t> v_global(seeds + WsKeffSize, keff_shape, keff_stride);
                TSTORE(w_global, w_seed);
                TSTORE(v_global, v_seed);
              } else {
                GmTensor2D<float> workspace_keff_global(
                    workspace_keff_handle +
                        static_cast<int64_t>(cid) * WsKeffSize +
                        static_cast<int64_t>(vid) * HalfChunk * HiddenSize,
                    keff_shape, keff_stride);
                TSTORE(workspace_keff_global, keff_ub);
              }
            }
            pipe_barrier(PIPE_ALL);
            // Signal Cube: ws_keff ready (flag 11)
#if __CCE_AICORE__ == 220
            SetCrossFlag<PIPE_MTE3>(kKeffReady);
#else
            set_intra_block(PIPE_MTE3, kKeffReady);
#endif
            first_iter = false;
          }
          gi++;
        }
      }
    }
  }

  // Drain the trailing "workspace free" signals emitted by Cube on the last
  // iteration of each active block.  Without this, every block leaves +1 on
  // flags 12 and 13 (because the in-loop wait is gated by `!first_iter`), and
  // that stale count corrupts the next launch's first non-first-iter wait.
  if (!first_iter) {
#if __CCE_AICORE__ == 220
    wait_flag_dev(kA2Free);
    wait_flag_dev(kKeffFree);
#else
    wait_intra_block(PIPE_MTE3, kA2Free);
    wait_intra_block(PIPE_MTE3, kKeffFree);
    pipe_barrier(PIPE_ALL);
#endif
  }
#endif

#if defined(__DAV_CUBE__)
  // Cube reads V from BSND and the two Vec-produced workspaces, then issues
  // U = A2 @ V and W = A2 @ K_eff.  a2_l1 stays resident in L1 across both
  // GEMMs so the second matmul is just a different B-side load away.
  {
    int64_t gi = 0;
    for (int64_t si = 0; si < num_seqs; ++si) {
      int64_t bos = static_cast<int64_t>(cu_seqlens[si]);
      int64_t eos = static_cast<int64_t>(cu_seqlens[si + 1]);
      int64_t slen = eos - bos;
      int64_t nc = (slen + ChunkSize - 1) / ChunkSize;

      for (int64_t ci = 0; ci < nc; ++ci) {
        for (int32_t h = 0; h < NumHeads; ++h) {
          if (gi % static_cast<int64_t>(block_num) ==
              static_cast<int64_t>(cid)) {
            int64_t chunk_start = ci * ChunkSize;
            int64_t remaining = slen - chunk_start;
            int32_t valid_rows = static_cast<int32_t>(
                remaining < ChunkSize ? remaining : ChunkSize);
            int64_t chunk_token_start = bos + chunk_start;
            int32_t head_idx = h;
            int64_t v_off = (chunk_token_start * static_cast<int64_t>(H) +
                             static_cast<int64_t>(head_idx)) *
                            static_cast<int64_t>(HiddenSize);

            if constexpr (!UseLog2Gate) {
              GmShape2D v_shape(valid_rows, HiddenSize);
              GmStride2D v_stride(BSND_STRIDE);
              GmTensor2D<float> v_global(V_handle + v_off, v_shape, v_stride);
              DynMatL1<float, ChunkSize, HiddenSize> v_l1_load(valid_rows,
                                                               HiddenSize);
              TASSIGN(v_l1_load, 0);
              TLOAD(v_l1_load, v_global);
              if (valid_rows != ChunkSize) {
                TFILLPAD(v_l1_load, v_l1_load);
              }
            }

            // Cube waits Vec: ws_a2 ready (flag 10).
            // The wait must block MTE2: the consuming op is the GM->L1 TLOAD.
#if __CCE_AICORE__ == 220
            wait_flag_dev(kA2Ready);
#else
            WaitBothVecOnA5<PIPE_MTE2>(kA2Ready);
            pipe_barrier(PIPE_ALL);
#endif
            {
              GmShape2D a2_shape(ChunkSize, ChunkSize);
              GmStride2D a2_stride(ChunkSize);
              GmTensor2D<Operand> workspace_a2_w_global(
                  reinterpret_cast<__gm__ Operand*>(workspace_a2_handle) +
                      static_cast<int64_t>(cid) * WsA2Size,
                  a2_shape, a2_stride);
              TLOAD(a2_w_l1, workspace_a2_w_global);
            }
            if constexpr (UseLog2Gate) {
              wait_flag_dev(kKeffReady);
              auto* seeds = reinterpret_cast<__gm__ bfloat16_t*>(workspace_keff_handle) +
                            2 * static_cast<int64_t>(cid) * WsKeffSize;
              const GmShape2D shape(ChunkSize, HiddenSize);
              const GmStride2D stride(HiddenSize);
              GmTensor2D<bfloat16_t> w_global(seeds, shape, stride);
              GmTensor2D<bfloat16_t> v_global(seeds + WsKeffSize, shape, stride);
              TLOAD(v_l1, v_global);
              TLOAD(keff_l1, w_global);
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            gemm_v0<Operand,
                    float,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    KTail,
                    false,
                    false>(a2_w_l1, v_l1, u_l0, true);

            {
              GmShape2D u_shape(valid_rows, HiddenSize);
              GmStride2D u_stride(BSND_STRIDE);
              GmTensor2D<float> u_global(U_handle + v_off, u_shape, u_stride);
              DynAccTile<float, ChunkSize, HiddenSize> u_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(u_store, 0);
              TSTORE(u_global, u_store);
            }
            // Cube signals Vec: ws_a2 slot free (flag 12)
#if __CCE_AICORE__ == 220
            SetCrossFlag<PIPE_FIX>(kA2Free);
#else
            pipe_barrier(PIPE_ALL);
            SignalBothVecOnA5<PIPE_FIX>(kA2Free);
#endif

            if constexpr (!UseLog2Gate) {
            // Cube waits Vec: ws_keff ready (flag 11).
            // The wait must block MTE2: the consuming op is the GM->L1 TLOAD.
#if __CCE_AICORE__ == 220
            wait_flag_dev(kKeffReady);
#else
            WaitBothVecOnA5<PIPE_MTE2>(kKeffReady);
            pipe_barrier(PIPE_ALL);
#endif
            {
              GmShape2D keff_shape(ChunkSize, HiddenSize);
              GmStride2D keff_stride(HiddenSize);
              GmTensor2D<float> workspace_keff_global(
                  workspace_keff_handle +
                      static_cast<int64_t>(cid) * WsKeffSize,
                  keff_shape,
                  keff_stride);
              TLOAD(keff_l1, workspace_keff_global);
            }
            }

            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            gemm_v0<Operand,
                    float,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    ChunkSize,
                    HiddenSize,
                    ChunkSize,
                    KTail,
                    false,
                    false>(a2_w_l1, keff_l1, w_l0, true);

            {
              GmShape2D w_shape(valid_rows, HiddenSize);
              GmStride2D w_stride(BSND_STRIDE);
              GmTensor2D<float> w_global(W_handle + v_off, w_shape, w_stride);
              DynAccTile<float, ChunkSize, HiddenSize> w_store(valid_rows,
                                                               HiddenSize);
              TASSIGN(w_store, 65536);
              TSTORE(w_global, w_store);
            }
            // Cube signals Vec: ws_keff slot free (flag 13)
#if __CCE_AICORE__ == 220
            SetCrossFlag<PIPE_FIX>(kKeffFree);
#else
            pipe_barrier(PIPE_ALL);
            SignalBothVecOnA5<PIPE_FIX>(kKeffFree);
#endif
          }
          gi++;
        }
      }
    }
  }
#endif
}

}  // namespace mega_kda_prefill_wy_kda
