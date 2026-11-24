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

namespace mega_kda_prefill_chunk_o_exact {
// FP32 output stage: q*exp(g) @ incoming state + precomputed causal QK @ Vcorr.
// Mask_handle carries [T,H,C] FP32 causal Gram values from kkt_rows.
// Cube/Vec handshake and FP32 state/value arithmetic remain unchanged.

using namespace pto;
using namespace mega_kda_prefill_utils;

#ifdef __CCE_AICORE__

namespace {

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

template <typename T,
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
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
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
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
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
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
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols>
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
          int32_t Rows,
          int32_t Cols,
          int32_t RowValid = Rows,
          int32_t ColValid = Cols,
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

// Single-shot dense GEMM via L0A/L0B — used when the K-dim is one L0 tile.
// All three of our GEMMs have inner-dim == 128 == L0 tile size, so a one-shot
// matmul is sufficient (no K-slicing needed, unlike chunk_h_kda's gemm_v0).
template <typename T1,
          typename T2,
          int32_t M,
          int32_t N,
          int32_t K,
          bool transpose_B = false>
AICORE PTO_INLINE void gemm_oneshot(
    TileMatL1<T1, M, K, M, K>& A,
    std::conditional_t<transpose_B,
                       TileMatL1<T1, N, K, N, K>,
                       TileMatL1<T1, K, N, K, N>>& B,
    pto::TileAcc<T2, M, N, M, N>& C) {
  TileMatL0A<T1, M, K, M, K> l0a;
  TileMatL0B<T1, K, N, K, N> l0b;
  pto::TASSIGN(l0a, 0x0);
  pto::TASSIGN(l0b, 0x0);

  auto war_event_id = (event_t)(((int)EVENT_ID0 + 1) % 8);
  set_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_MTE2, PIPE_MTE1, war_event_id);
  set_flag(PIPE_M, PIPE_MTE1, war_event_id);
  wait_flag(PIPE_M, PIPE_MTE1, war_event_id);

  pto::TEXTRACT(l0a, A, 0, 0);
  if constexpr (!transpose_B) {
    pto::TEXTRACT(l0b, B, 0, 0);
  } else {
    TileMatL1ZN<T1, K, N, K, N> B_t;
    pto::TRESHAPE(B_t, B);
    pto::TEXTRACT(l0b, B_t, 0, 0);
  }

  set_flag(PIPE_MTE1, PIPE_M, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_M, war_event_id);
  pto::TMATMUL(C, l0a, l0b);

  set_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  wait_flag(PIPE_MTE1, PIPE_MTE2, war_event_id);
  set_flag(PIPE_M, PIPE_FIX, war_event_id);
  wait_flag(PIPE_M, PIPE_FIX, war_event_id);
}

}  // namespace

#endif

template <int32_t HiddenSize, int32_t ChunkSize, bool UseLog2Gate>
AICORE void chunk_o_kda_kernel(__gm__ bfloat16_t* Q_handle,
                               __gm__ bfloat16_t* K_handle,
                               __gm__ float* V_handle,
                               __gm__ float* S_handle,
                               __gm__ float* G_handle,
                               __gm__ float* Mask_handle,
                               __gm__ float* workspace_handle,
                               __gm__ bfloat16_t* O_handle,
                               __gm__ const int32_t* cu_seqlens,
                               int64_t batch_size,
                               int64_t seq_len,
                               int64_t total_tokens,
                               int32_t num_heads,
                               uint64_t ffts_addr) {
  auto cid = get_block_idx();
  auto block_num = get_block_num();
  set_ffts_base_addr(ffts_addr);

  constexpr int32_t K_DIM = HiddenSize;
  constexpr int32_t V_DIM = HiddenSize;
  constexpr int32_t C = ChunkSize;
  // Head count (HV) is a runtime argument; it only drives the work-item decode
  // and the BSND GM stride, never a UB buffer size or tile shape.
  const int32_t H = num_heads;  // HV in KDA terminology
  constexpr int32_t HalfC = C / 2;
  const int32_t BSND_STRIDE = H * HiddenSize;
  constexpr int32_t HM_STRIDE = HiddenSize;  // head-major Q, K, G stride
  constexpr int32_t KV = K_DIM * V_DIM;

  // ── Workspace slots (fp32 elements, per AI core) ─────────────────────────
  constexpr int32_t WS_Q = 0;
  constexpr int32_t WS_K = WS_Q + C * K_DIM;
  constexpr int32_t WS_V = WS_K + C * K_DIM;
  constexpr int32_t WS_S =
      WS_V + C * V_DIM;  // Keep the existing workspace stride.
  constexpr int32_t WS_QK = WS_S + KV;
  constexpr int32_t WS_QS = WS_QK + C * C;
  constexpr int32_t WS_QKV = WS_QS + C * V_DIM;
  constexpr int32_t WS_PER_CORE = WS_QKV + C * V_DIM;

#if defined(__DAV_CUBE__)
  // ── Cube L1 tiles ────────────────────────────────────────────────────────
  // FP32 L1 offsets at C=K=V=128: Q=0, S=128 KiB, Aqk=192 KiB,
  // V_corr=256 KiB. The 64 KiB K slot is reserved; Aqk is precomputed.
  TileMatL1<float, C, K_DIM, C, K_DIM> q_l1;
  TASSIGN(q_l1, 0);
  TileMatL1<float, K_DIM, V_DIM, K_DIM, V_DIM> s_l1;
  TASSIGN(s_l1, (C * K_DIM + C * K_DIM) * sizeof(float));
  TileMatL1<float, C, C, C, C> qkm_l1;
  TASSIGN(qkm_l1, (C * K_DIM + C * K_DIM + KV) * sizeof(float));
  TileMatL1<float, C, V_DIM, C, V_DIM> v_l1;
  TASSIGN(v_l1, (C * K_DIM + C * K_DIM + KV + C * C) * sizeof(float));

  // Separate L0C accumulators: QS at C*C*4 bytes, Aqk @ V_corr at zero.
  TileAcc<float, C, V_DIM, C, V_DIM> qs_l0;
  TASSIGN(qs_l0, C * C * sizeof(float));
  TileAcc<float, C, V_DIM, C, V_DIM> qkv_l0;
  TASSIGN(qkv_l0, 0);
#endif

#if defined(__DAV_VEC__)
  // Vec UB: FP32 arithmetic, with BF16 staging for raw Q/K. Buffers are
  // reused after publication and again when combining the two Cube results.
  constexpr int32_t MASK_UB_ADDR = 0;
  constexpr int32_t SLOT_A_ADDR = MASK_UB_ADDR + HalfC * C * sizeof(float);
  constexpr int32_t SLOT_B_ADDR = SLOT_A_ADDR + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_C_ADDR = SLOT_B_ADDR + HalfC * K_DIM * sizeof(float);
  constexpr int32_t SLOT_D_ADDR = SLOT_C_ADDR + HalfC * K_DIM * sizeof(float);
  constexpr float kScale = 0.08838834764831845f;
#endif

  int64_t num_seqs = batch_size;

  // Cube and both Vec subcores visit the same flattened chunk/head tasks.
  int64_t total_chunks = 0;
  bool all_full_chunks = true;
  for (int64_t seq = 0; seq < num_seqs; ++seq) {
    int64_t length =
        cu_seqlens ? cu_seqlens[seq + 1] - cu_seqlens[seq] : seq_len;
    total_chunks += (length + C - 1) / C;
    all_full_chunks &= length % C == 0;
  }
  int64_t total_work = total_chunks * H;

#if defined(__DAV_CUBE__)
  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) {
      break;
    }

    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;
    // With no tail chunks, the flattened chunk index is also a token offset.
    const int64_t token = (cu_seqlens ? cu_seqlens[0] : 0) + (pid / H) * C;
    const int64_t token_head = token * H + pid % H;
    {
      // ── Wait Vec phase A: q_eff and any padded operands are ready ─
      // A2: Cube and Vec are separate cores → FFTS cross-core flag.
      // A5: Cube and both Vec sub-blocks share ONE core → intra-block flags,
      //     with each Vec sub-block signalling its own flag (base, base+16).
#if __CCE_AICORE__ == 220
      wait_flag_dev(0);
#else
      WaitBothVecOnA5<PIPE_MTE2>(0);
      pipe_barrier(PIPE_ALL);
#endif

      // Load q_eff [C, K] from WS_Q.
      {
        GmShape2D q_shape(C, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<float> q_global(
            workspace_handle + ws_base + WS_Q, q_shape, q_stride);
        DynMatL1<float, C, K_DIM> q_l1_load(C, K_DIM);
        TASSIGN(q_l1_load, 0);
        TLOAD(q_l1_load, q_global);
      }
      // ChunkH publishes snapshots before the stage barrier. The flattened
      // (chunk, head) index also indexes this dense FP32 state matrix.
      {
        GmShape2D s_shape(K_DIM, V_DIM);
        GmStride2D s_stride(V_DIM);
        GmTensor2D<float> s_global(S_handle + pid * KV, s_shape, s_stride);
        DynMatL1<float, K_DIM, V_DIM> s_l1_load(K_DIM, V_DIM);
        TASSIGN(s_l1_load, (C * K_DIM + C * K_DIM) * sizeof(float));
        TLOAD(s_l1_load, s_global);
      }
      // ChunkH already published V_corr. Only tail-containing requests need
      // the Vec-side staging that gives this fixed-size GEMM zero padding.
      {
        GmShape2D v_shape(C, V_DIM);
        GmStride2D v_stride(all_full_chunks ? H * V_DIM : V_DIM);
        auto* values = all_full_chunks ? V_handle + token_head * V_DIM
                                       : workspace_handle + ws_base + WS_V;
        GmTensor2D<float> v_global(values, v_shape, v_stride);
        DynMatL1<float, C, V_DIM> v_l1_load(C, V_DIM);
        TASSIGN(v_l1_load,
                (C * K_DIM + C * K_DIM + KV + C * C) * sizeof(float));
        TLOAD(v_l1_load, v_global);
      }
      // On full chunks KKT writes the whole Gram, including its zero region.
      {
        GmShape2D qkm_shape(C, C);
        const bool direct_gram = all_full_chunks && !UseLog2Gate;
        GmStride2D qkm_stride(direct_gram ? H * C : C);
        auto* gram = direct_gram ? Mask_handle + token_head * C
                                 : workspace_handle + ws_base + WS_QK;
        GmTensor2D<float> qkm_global(gram, qkm_shape, qkm_stride);
        DynMatL1<float, C, C> qkm_l1_load(C, C);
        TASSIGN(qkm_l1_load, (C * K_DIM + C * K_DIM + KV) * sizeof(float));
        TLOAD(qkm_l1_load, qkm_global);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM2: QS = q_eff @ S — [C, K] @ [K, V] → [C, V]  (inter-chunk term).
      gemm_oneshot<float, float, C, V_DIM, K_DIM, /*transpose_B=*/false>(
          q_l1, s_l1, qs_l0);

      // Store QS fp32 → WS_QS.
      {
        GmShape2D qs_shape(C, V_DIM);
        GmStride2D qs_stride(V_DIM);
        GmTensor2D<float> qs_global(
            workspace_handle + ws_base + WS_QS, qs_shape, qs_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qs_store;
        TASSIGN(qs_store, C * C * sizeof(float));
        TSTORE(qs_global, qs_store);
      }

      set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

      // GEMM3: QKV = Aqk_masked @ V_corr — [C, C] @ [C, V] → [C, V].
      gemm_oneshot<float, float, C, V_DIM, C, /*transpose_B=*/false>(
          qkm_l1, v_l1, qkv_l0);

      // Store QKV fp32 → WS_QKV.
      {
        GmShape2D qkv_shape(C, V_DIM);
        GmStride2D qkv_stride(V_DIM);
        GmTensor2D<float> qkv_global(
            workspace_handle + ws_base + WS_QKV, qkv_shape, qkv_stride);
        TileAcc<float, C, V_DIM, C, V_DIM> qkv_store;
        TASSIGN(qkv_store, 0);
        TSTORE(qkv_global, qkv_store);
      }
      // Signal Vec: QS + QKV ready (flag 1)
#if __CCE_AICORE__ == 220
      SetCrossFlag<PIPE_FIX>(1);
#else
      pipe_barrier(PIPE_ALL);
      SignalBothVecOnA5<PIPE_FIX>(1);
#endif
    }
  }
#endif

#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);

  auto vid = get_subblockid();
  int32_t my_row_offset = static_cast<int32_t>(vid) * HalfC;

  // The former Mask argument now carries the precomputed inclusive QK Gram.

  for (int64_t wi = 0; wi < (total_work + block_num - 1) / block_num; ++wi) {
    int64_t pid = wi * block_num + cid;
    if (pid >= total_work) {
      break;
    }

    int64_t head = pid % H;
    int64_t ci = pid / H;
    int64_t bos = 0;
    int64_t slen = 0;
    for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
      bos = cu_seqlens ? cu_seqlens[seq_idx] : seq_idx * seq_len;
      slen = cu_seqlens ? cu_seqlens[seq_idx + 1] - bos : seq_len;
      int64_t count = (slen + C - 1) / C;
      if (ci < count) {
        break;
      }
      ci -= count;
    }
    int64_t ws_base = static_cast<int64_t>(cid) * WS_PER_CORE;
    {
      int64_t chunk_start = bos + static_cast<int64_t>(ci) * C;
      int64_t valid = slen - static_cast<int64_t>(ci) * C;
      if (valid > C) {
        valid = C;
      }
      int32_t valid_rows =
          static_cast<int32_t>(valid - static_cast<int64_t>(vid) * HalfC);
      if (valid_rows < 0) {
        valid_rows = 0;
      }
      if (valid_rows > HalfC) {
        valid_rows = HalfC;
      }

      // ====================================================================
      // PHASE A — load Q, K, G_cs; pre-scale q_eff/k_eff; cast V_corr, S.
      // ====================================================================
      int64_t hk_base =
          static_cast<int64_t>(head) * total_tokens * K_DIM +
          (chunk_start + static_cast<int64_t>(vid) * HalfC) * K_DIM;

      // Tile views into the UB slots (declared inside the loop so we can
      // re-bind them by phase without touching constexpr globals).
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> g_ub;
      TASSIGN(g_ub, SLOT_A_ADDR);
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero> q_ub;
      TASSIGN(q_ub, SLOT_B_ADDR);
      TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM> exp_ub;
      TASSIGN(exp_ub, SLOT_C_ADDR);

      // Load head-major Q (BF16) and G_cs (FP32).
      if (valid_rows > 0) {
        {
          GmShape2D q_shape(valid_rows, K_DIM);
          GmStride2D q_stride(HM_STRIDE);
          GmTensor2D<bfloat16_t> q_global(
              Q_handle + hk_base, q_shape, q_stride);
          TileUbDataND<bfloat16_t,
                       HalfC,
                       K_DIM,
                       HalfC,
                       K_DIM,
                       pto::PadValue::Zero>
              q_stg_full;
          TASSIGN(q_stg_full, SLOT_D_ADDR);
          DynVecTile<bfloat16_t, HalfC, K_DIM, pto::PadValue::Zero> q_load(
              valid_rows, K_DIM);
          TASSIGN(q_load, SLOT_D_ADDR);
          TLOAD(q_load, q_global);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(q_stg_full, q_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        {
          TileUbDataND<bfloat16_t, HalfC, K_DIM, HalfC, K_DIM> q_stg_cvt;
          TASSIGN(q_stg_cvt, SLOT_D_ADDR);
          TCVT(q_ub, q_stg_cvt, pto::RoundMode::CAST_NONE);
          PipeBarrierVec();
        }
        {
          GmShape2D g_shape(valid_rows, K_DIM);
          GmStride2D g_stride(HM_STRIDE);
          GmTensor2D<float> g_global(G_handle + hk_base, g_shape, g_stride);
          TileUbDataND<float, HalfC, K_DIM, HalfC, K_DIM, pto::PadValue::Zero>
              g_stg_full;
          TASSIGN(g_stg_full, SLOT_A_ADDR);
          DynVecTile<float, HalfC, K_DIM, pto::PadValue::Zero> g_load(
              valid_rows, K_DIM);
          TASSIGN(g_load, SLOT_A_ADDR);
          TLOAD(g_load, g_global);  // g_cs fp32 → g_ub directly
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(g_stg_full, g_load);
          }
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      } else {
        TEXPANDS(q_ub, 0.0f);
        TEXPANDS(g_ub, 0.0f);
        PipeBarrierVec();
      }

      // ── (A.2) q_eff = Q * exp(g_cs) ──────────────────────────────────
      // exp(g_cs) ≤ 1 (g_cs ≤ 0) so q_eff is bounded; kept fp32 to match the
      // fp32 GEMM (k_eff below overflows fp16).
      if constexpr (UseLog2Gate) {
        main_gate_exp(exp_ub, g_ub);
      } else {
        exp_gate<false>(exp_ub, g_ub);
      }
      PipeBarrierVec();
      // q_eff into exp_ub (SLOT_C) so q_ub (SLOT_B) keeps the raw scaled Q,
      // which the Aqk element-wise pass below needs as its row factor.
      TMUL(exp_ub, q_ub, exp_ub);
      PipeBarrierVec();
      if constexpr (UseLog2Gate) {
        TileUbDataND<bfloat16_t, HalfC, K_DIM, HalfC, K_DIM> typed;
        TASSIGN(typed, SLOT_D_ADDR);
        round_bf16(exp_ub, exp_ub, typed);
        TMULS(exp_ub, exp_ub, kScale);
        PipeBarrierVec();
        round_bf16(exp_ub, exp_ub, typed);
      }

      // Store q_eff fp32 → WS_Q (full HalfC rows; padded zeros for invalid).
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      {
        GmShape2D q_shape(HalfC, K_DIM);
        GmStride2D q_stride(K_DIM);
        GmTensor2D<float> q_global(
            workspace_handle + ws_base + WS_Q +
                static_cast<int64_t>(vid) * HalfC * K_DIM,
            q_shape,
            q_stride);
        DynVecTile<float, HalfC, K_DIM> q_store(HalfC, K_DIM);
        TASSIGN(q_store, SLOT_C_ADDR);
        TSTORE(q_global, q_store);
      }

      // KKT publishes only the rectangular blocks touching the causal
      // triangle. Pad the remaining rows and columns here instead of
      // clearing the whole GM tile in the front end.
      if (!all_full_chunks || UseLog2Gate) {
        pipe_barrier(PIPE_ALL);
        TileUbDataND<float, HalfC, C, HalfC, C, PadValue::Zero> gram;
        TASSIGN(gram, SLOT_C_ADDR);
        if (valid_rows > 0) {
          const int32_t columns =
              valid < (vid + 1) * HalfC ? valid : (vid + 1) * HalfC;
          DynVecTile<float, HalfC, C, PadValue::Zero> load(valid_rows, columns);
          TASSIGN(load, SLOT_C_ADDR);
          GmTensor2D<float> source(
              Mask_handle + ((chunk_start + vid * HalfC) * H + head) * C,
              GmShape2D(valid_rows, columns),
              GmStride2D(H * C));
          TLOAD(load, source);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          if (valid_rows != HalfC || columns != C) {
            TFILLPAD_INPLACE(gram, load);
          }
        } else {
          TEXPANDS(gram, 0.0f);
        }
        PipeBarrierVec();
        if constexpr (UseLog2Gate) {
          TileUbDataND<bfloat16_t, HalfC, C, HalfC, C> typed;
          TASSIGN(typed, SLOT_D_ADDR);
          TMULS(gram, gram, kScale);
          PipeBarrierVec();
          round_bf16(gram, gram, typed);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmTensor2D<float> destination(
            workspace_handle + ws_base + WS_QK + vid * HalfC * C,
            GmShape2D(HalfC, C),
            GmStride2D(C));
        TSTORE(destination, gram);
        pipe_barrier(PIPE_ALL);
      }
      if (!all_full_chunks) {
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM, pto::PadValue::Zero>
            v_f_ub;
        TASSIGN(v_f_ub, SLOT_A_ADDR);

        int64_t v_offset = (chunk_start * H + head) * V_DIM +
                           static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
        if (valid_rows > 0) {
          GmShape2D v_shape(valid_rows, V_DIM);
          GmStride2D v_stride(BSND_STRIDE);
          GmTensor2D<float> v_global(V_handle + v_offset, v_shape, v_stride);
          DynVecTile<float, HalfC, V_DIM, pto::PadValue::Zero> v_load(
              valid_rows, V_DIM);
          TASSIGN(v_load, SLOT_A_ADDR);
          TLOAD(v_load, v_global);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          if (valid_rows != HalfC) {
            TFILLPAD_INPLACE(v_f_ub, v_load);
          }
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        } else {
          TEXPANDS(v_f_ub, 0.0f);
          PipeBarrierVec();
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        GmShape2D vw_shape(HalfC, V_DIM);
        GmStride2D vw_stride(V_DIM);
        GmTensor2D<float> vw_global(
            workspace_handle + ws_base + WS_V +
                static_cast<int64_t>(vid) * HalfC * V_DIM,
            vw_shape,
            vw_stride);
        DynVecTile<float, HalfC, V_DIM> v_store(HalfC, V_DIM);
        TASSIGN(v_store, SLOT_A_ADDR);
        TSTORE(vw_global, v_store);
      }

      // ── (A.6) Signal Cube: phase A workspace ready ───────────────────
      pipe_barrier(PIPE_ALL);
#if __CCE_AICORE__ == 220
      SetCrossFlag<PIPE_MTE3>(0);
#else
      set_intra_block(PIPE_MTE3, 0);
#endif

      // ====================================================================
      // PHASE C — wait QS + QKV from Cube; combine O = QS + QKV; write to GM.
      // (No separate mask phase: Aqk was masked element-wise in phase A.)
      // ====================================================================
#if __CCE_AICORE__ == 220
      wait_flag_dev(1);
#else
      wait_intra_block(PIPE_MTE3, 1);
#endif
      pipe_barrier(PIPE_ALL);

      if (valid_rows > 0) {
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qs_ub;
        TASSIGN(qs_ub, SLOT_A_ADDR);
        TileUbDataND<float, HalfC, V_DIM, HalfC, V_DIM> qkv_ub;
        TASSIGN(qkv_ub, SLOT_B_ADDR);

        // Load QS fp32 → SLOT_A.
        {
          GmShape2D qs_shape(HalfC, V_DIM);
          GmStride2D qs_stride(V_DIM);
          GmTensor2D<float> qs_global(
              workspace_handle + ws_base + WS_QS +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qs_shape,
              qs_stride);
          DynVecTile<float, HalfC, V_DIM> qs_load(HalfC, V_DIM);
          TASSIGN(qs_load, SLOT_A_ADDR);
          TLOAD(qs_load, qs_global);
        }
        // Load QKV fp32 → SLOT_B.
        {
          GmShape2D qkv_shape(HalfC, V_DIM);
          GmStride2D qkv_stride(V_DIM);
          GmTensor2D<float> qkv_global(
              workspace_handle + ws_base + WS_QKV +
                  static_cast<int64_t>(vid) * HalfC * V_DIM,
              qkv_shape,
              qkv_stride);
          DynVecTile<float, HalfC, V_DIM> qkv_load(HalfC, V_DIM);
          TASSIGN(qkv_load, SLOT_B_ADDR);
          TLOAD(qkv_load, qkv_global);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Match the main-path FP32 Cube outputs through the final sum.
        // Only the final attention output is rounded to BF16.
        TADD(qs_ub, qs_ub, qkv_ub);
        PipeBarrierVec();

        // The QKV contribution is dead after the sum. Round directly at the
        // existing model boundary, reusing its UB slot for BF16 output.
        DynVecTile<float, HalfC, V_DIM> scaled(valid_rows, V_DIM);
        TASSIGN(scaled, SLOT_A_ADDR);
        DynVecTile<bfloat16_t, HalfC, V_DIM> o_store(valid_rows, V_DIM);
        TASSIGN(o_store, SLOT_B_ADDR);
        if constexpr (!UseLog2Gate) {
          TMULS(scaled, scaled, kScale);
          PipeBarrierVec();
        }
        TCVT(o_store, scaled, pto::RoundMode::CAST_RINT);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        const int64_t o_offset =
            (chunk_start * H + head) * V_DIM +
            static_cast<int64_t>(vid) * HalfC * BSND_STRIDE;
        GmShape2D o_shape(valid_rows, V_DIM);
        GmStride2D o_stride(BSND_STRIDE);
        GmTensor2D<bfloat16_t> o_global(O_handle + o_offset, o_shape, o_stride);
        TSTORE(o_global, o_store);
      }
      // Drain all pipes before next chunk iteration.  Without this, the next
      // iteration's Phase A.1 TLOAD (PIPE_MTE2 → SLOT_A/B) can race with the
      // in-flight Phase C TSTORE (PIPE_MTE3 reading SLOT_A) or TADD writes
      // (PIPE_V on SLOT_A/B) from this iteration.
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

}  // namespace mega_kda_prefill_chunk_o_exact
