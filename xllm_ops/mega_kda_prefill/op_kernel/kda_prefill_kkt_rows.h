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

#include "kda_kernel_utils.h"
#include "kda_prefill_chunk_o.h"

namespace mega_kda_prefill_kkt_rows {
using namespace pto;
using namespace mega_kda_prefill_utils;

#ifdef __CCE_AICORE__
template <typename T, int R, int C, int VR = R, int VC = C>
using ND = Tile<TileType::Vec,
                T,
                R,
                C,
                BLayout::RowMajor,
                VR,
                VC,
                SLayout::NoneBox,
                512>;
using GS = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using ST = pto::Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T>
using GM = GlobalTensor<T, GS, ST>;
#endif

#if defined(__DAV_VEC__)
// This is the same FP32 row sum as PTO's dynamic 64x128 path:
// add the two 64-element halves, then issue vcadd for each row.
// n is always in [1, 64], so normal-mode repeats are sufficient.
AICORE inline void sum_pair64(__ubuf__ float* dst,
                              __ubuf__ float* product,
                              __ubuf__ float* scratch,
                              int n) {
  vadd(scratch, product, product + 64, n, 1, 1, 1, 8, 16, 16);
  PipeBarrierVec();
  vcadd(dst, scratch, n, 1, 1, 8, false);
  PipeBarrierVec();
}
#endif

// Match the main-path score operands on 910B. The two BF16 casts are
// performed independently on q/k and on the transposed key before the dot.
template <typename Tile>
AICORE inline void score_exp2(Tile& value) {
  constexpr float kLn2 = 0.6931471805599453f;
  TMULS(value, value, kLn2);
  PipeBarrierVec();
  TMINS(value, value, 120.0f * kLn2);
  PipeBarrierVec();
  TMAXS(value, value, -126.0f * kLn2);
  PipeBarrierVec();
  TEXP(value, value);
  PipeBarrierVec();
}

template <int C, bool UseLog2Gate>
AICORE inline void launch_kkt_rows(__gm__ bfloat16_t* keys,
                                   __gm__ float* gates,
                                   __gm__ float* beta,
                                   __gm__ float* out,
                                   __gm__ bfloat16_t* queries,
                                   __gm__ float* aqk,
                                   __gm__ const int32_t* cu,
                                   int64_t batches,
                                   int64_t length,
                                   int64_t tokens,
                                   int32_t heads,
                                   __gm__ bfloat16_t* akk_left,
                                   __gm__ bfloat16_t* akk_right) {
#if defined(__DAV_VEC__)
  constexpr int D = 128;
  constexpr int N = 64;
  constexpr int ROWS = 32;
  constexpr int kRowGate = 147456;
  constexpr int kRowKeyBf16 = 155648;
  constexpr int kRowQueryBf16 = 159744;
  constexpr int kRowKey = 163840;
  constexpr int kRowQuery = 172032;
  constexpr int kReferenceGate = 180224;
  constexpr int kNegReferenceGate = kReferenceGate + D * sizeof(float);
  const int core = get_block_idx();
  const int cores = get_block_num();
  const int vid = get_subblockid();
  int64_t total_chunks = 0;
  bool all_full_chunks = true;
  for (int seq = 0; seq < batches; ++seq) {
    int64_t len = cu ? cu[seq + 1] - cu[seq] : length;
    total_chunks += (len + C - 1) / C;
    all_full_chunks &= len % C == 0;
  }
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t chunk_heads = total_chunks * heads;
  for (int64_t task = core; task < chunk_heads * (C / ROWS); task += cores) {
    int row_group = task / chunk_heads;
    int64_t chunk_head = task % chunk_heads;
    int head = chunk_head % heads;
    int64_t chunk = chunk_head / heads;
    int64_t bos = 0;
    int64_t len = 0;
    for (int seq = 0; seq < batches; ++seq) {
      bos = cu ? cu[seq] : seq * length;
      len = cu ? cu[seq + 1] - bos : length;
      int64_t count = (len + C - 1) / C;
      if (chunk < count) {
        break;
      }
      chunk -= count;
    }
    int valid = len - chunk * C < C ? len - chunk * C : C;
    if (row_group * ROWS >= valid) {
      continue;
    }
    int64_t first = bos + chunk * C;
    int64_t hbase = int64_t(head) * tokens * D + first * D;
    const bool cube_akk = UseLog2Gate && valid == C;
    const int64_t right_base = 2 * hbase + row_group * C * D;
    ND<float, 1, C, 1, DYNAMIC> beta_f(valid);
    TASSIGN(beta_f, 189440);
    GM<float> beta_gm(
        beta + int64_t(head) * tokens + first, GS(1, 1, 1, 1, valid), ST(C));
    TLOAD(beta_f, beta_gm);
    SetWaitFlag<PIPE_MTE2, PIPE_S>(0);
    ND<float, 1, D> ref, neg_ref;
    if constexpr (UseLog2Gate) {
      static_assert(C == 64 && ROWS == 32);
      const int band_start = row_group * ROWS;
      const int band_rows = valid - band_start < ROWS ? valid - band_start : ROWS;
      TASSIGN(ref, kReferenceGate);
      TASSIGN(neg_ref, kNegReferenceGate);
      GM<float> reference(gates + hbase + (band_start + band_rows / 2) * D,
                          GS(1, 1, 1, 1, D), ST(D));
      TLOAD(ref, reference);
      SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
      TMULS(neg_ref, ref, -1.0f);
      PipeBarrierVec();
    }
    for (int col0 = 0; col0 < valid && col0 < (row_group + 1) * ROWS;
         col0 += N) {
      int block_n = valid - col0 < N ? valid - col0 : N;
      const int causal_columns = (row_group + 1) * ROWS - col0;
      const int load_n = causal_columns < block_n ? causal_columns : block_n;
      ND<float, N, D, DYNAMIC> columns_g(load_n), columns_k(load_n);
      ND<bfloat16_t, N, D, DYNAMIC> columns_kh(load_n);
      TASSIGN(columns_g, 0);
      TASSIGN(columns_k, 32768);
      TASSIGN(columns_kh, 98304);
      GM<float> gcols(gates + hbase + col0 * D, GS(1, 1, 1, load_n, D), ST(D));
      GM<bfloat16_t> kcols(
          keys + hbase + col0 * D, GS(1, 1, 1, load_n, D), ST(D));
      TLOAD(columns_g, gcols);
      TLOAD(columns_kh, kcols);
      SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
      TCVT(columns_k, columns_kh, RoundMode::CAST_NONE);
      TMULS(columns_g, columns_g, -1.0f);
      PipeBarrierVec();
      if constexpr (UseLog2Gate) {
        TCOLEXPANDADD(columns_g, columns_g, ref);
        PipeBarrierVec();
        score_exp2(columns_g);
        TMUL(columns_k, columns_k, columns_g);
        PipeBarrierVec();
        TCVT(columns_kh, columns_k, RoundMode::CAST_RINT);
        PipeBarrierVec();
        if (cube_akk && vid == 0) {
          GM<bfloat16_t> right(akk_right + right_base,
                               GS(1, 1, 1, load_n, D), ST(D));
          SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
          TSTORE(right, columns_kh);
          SetWaitFlag<PIPE_MTE3, PIPE_V>(0);
        }
        TCVT(columns_k, columns_kh, RoundMode::CAST_NONE);
        PipeBarrierVec();
      }
      for (int row0 = row_group * ROWS;
           row0 < valid && row0 < (row_group + 1) * ROWS;
           row0 += ROWS) {
        int rows = valid - row0 < ROWS ? valid - row0 : ROWS;
        // Each AIV owns every other row. Load and convert only its own
        // operands, with dense UB rows and a doubled GM row stride.
        const int32_t owned_rows = (rows + 1 - vid) / 2;
        if (owned_rows == 0) {
          continue;
        }
        ND<float, ROWS / 2, D, DYNAMIC> row_g(owned_rows), row_k(owned_rows),
            row_q(owned_rows);
        ND<bfloat16_t, ROWS / 2, D, DYNAMIC> row_kh(owned_rows),
            row_qh(owned_rows);
        TASSIGN(row_g, kRowGate);
        TASSIGN(row_kh, kRowKeyBf16);
        TASSIGN(row_qh, kRowQueryBf16);
        TASSIGN(row_k, kRowKey);
        TASSIGN(row_q, kRowQuery);
        const int64_t row_base = hbase + (row0 + vid) * D;
        const GS row_shape(1, 1, 1, owned_rows, D);
        const ST row_stride(2 * D);
        GM<float> grow(gates + row_base, row_shape, row_stride);
        GM<bfloat16_t> krow(keys + row_base, row_shape, row_stride);
        GM<bfloat16_t> qrow(queries + row_base, row_shape, row_stride);
        TLOAD(row_g, grow);
        TLOAD(row_kh, krow);
        TLOAD(row_qh, qrow);
        SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
        TCVT(row_k, row_kh, RoundMode::CAST_NONE);
        TCVT(row_q, row_qh, RoundMode::CAST_NONE);
        PipeBarrierVec();
        if constexpr (UseLog2Gate) {
          TCOLEXPANDADD(row_g, row_g, neg_ref);
          PipeBarrierVec();
          score_exp2(row_g);
          TMUL(row_q, row_q, row_g);
          TMUL(row_k, row_k, row_g);
          PipeBarrierVec();
          TCVT(row_qh, row_q, RoundMode::CAST_RINT);
          TCVT(row_kh, row_k, RoundMode::CAST_RINT);
          PipeBarrierVec();
          if (cube_akk) {
            GM<bfloat16_t> left(akk_left + row_base, row_shape, row_stride);
            SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
            TSTORE(left, row_kh);
            SetWaitFlag<PIPE_MTE3, PIPE_V>(0);
          } else {
            TCVT(row_k, row_kh, RoundMode::CAST_NONE);
          }
          TCVT(row_q, row_qh, RoundMode::CAST_NONE);
          PipeBarrierVec();
        }
        for (int local = 0; local < owned_rows; ++local) {
          int r = row0 + vid + 2 * local;
          int n = r - col0 + 1 < block_n ? r - col0 + 1 : block_n;
          ND<float, N, D, DYNAMIC> gc(n), kc(n), diff(n), tmp(n);
          ND<float, N, 64, DYNAMIC> scratch(n);
          TASSIGN(gc, 0);
          TASSIGN(kc, 32768);
          TASSIGN(diff, 65536);
          TASSIGN(tmp, 98304);
          TASSIGN(scratch, 131072);
          ND<float, 1, D> gr;
          ND<float, 1, D> kr;
          ND<float, 1, D> qr;
          TASSIGN(gr, kRowGate + local * D * sizeof(float));
          TASSIGN(kr, kRowKey + local * D * sizeof(float));
          TASSIGN(qr, kRowQuery + local * D * sizeof(float));
          ND<float, 1, N, 1, DYNAMIC> cached_aqk(block_n), cached_kk(block_n);
          TASSIGN(cached_aqk, kRowGate + local * D * sizeof(float));
          TASSIGN(cached_kk,
                  kRowGate + local * D * sizeof(float) + N * sizeof(float));
          if constexpr (!UseLog2Gate) {
            TCOLEXPANDADD(diff, gc, gr);
            PipeBarrierVec();
            TMINS(diff, diff, 0.0f);
            PipeBarrierVec();
            exp_gate<false>(diff, diff);
            PipeBarrierVec();
          }
          if (n < block_n) {
            // A row's gate is dead after the subtraction. Use its two
            // halves as the reductions' final destinations.
            TEXPANDS(cached_aqk, 0.0f);
            if (!cube_akk) {
              TEXPANDS(cached_kk, 0.0f);
            }
          }
          if constexpr (UseLog2Gate) {
            TCOLEXPANDMUL(tmp, kc, qr);
          } else {
            TCOLEXPANDMUL(tmp, diff, qr);
            PipeBarrierVec();
            TMUL(tmp, tmp, kc);
          }
          PipeBarrierVec();
          sum_pair64(cached_aqk.data(), tmp.data(), scratch.data(), n);
          if (!cube_akk) {
            if constexpr (UseLog2Gate) {
              TCOLEXPANDMUL(tmp, kc, kr);
            } else {
              TCOLEXPANDMUL(tmp, diff, kr);
              PipeBarrierVec();
              TMUL(tmp, tmp, kc);
            }
            PipeBarrierVec();
            sum_pair64(cached_kk.data(), tmp.data(), scratch.data(), n);
            TMULS(cached_kk, cached_kk, beta_f.data()[r]);
            PipeBarrierVec();
          }
        }
        {
          // Gate inputs are dead. Each dense row now holds AQK followed by KK.
          ND<float, ROWS / 2, D, DYNAMIC, DYNAMIC> cached_aqk(owned_rows,
                                                              block_n);
          ND<float, ROWS / 2, D, DYNAMIC, DYNAMIC> cached_kk(owned_rows,
                                                             block_n);
          TASSIGN(cached_aqk, kRowGate);
          TASSIGN(cached_kk, kRowGate + N * sizeof(float));
          GM<float> qdst(aqk + ((first + row0 + vid) * heads + head) * C + col0,
                         GS(1, 1, 1, owned_rows, block_n),
                         ST(2 * heads * C));
          GM<float> dst(out + ((first + row0 + vid) * heads + head) * C + col0,
                        GS(1, 1, 1, owned_rows, block_n),
                        ST(2 * heads * C));
          SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
          TSTORE(qdst, cached_aqk);
          if (!cube_akk) {
            TSTORE(dst, cached_kk);
          }
          if (C > N && all_full_chunks && col0 == 0 && row0 < N) {
            // Complete the zero-only upper right quadrant for the direct
            // Cube load. Wait for the preceding store before reusing its UB.
            SetWaitFlag<PIPE_MTE3, PIPE_V>(0);
            TEXPANDS(cached_kk, 0.0f);
            SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
            GM<float> upper(aqk + ((first + row0 + vid) * heads + head) * C + N,
                            GS(1, 1, 1, owned_rows, N),
                            ST(2 * heads * C));
            TSTORE(upper, cached_kk);
          }
        }
      }
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}

#if defined(__DAV_CUBE__)
template <int N>
AICORE inline void akk_cube_band(__gm__ bfloat16_t* left,
                                 __gm__ bfloat16_t* right,
                                 __gm__ float* dst,
                                 int row_stride) {
  constexpr int R = 32;
  constexpr int D = 128;
  using namespace mega_kda_prefill_chunk_o_exact;
  TileMatL1<bfloat16_t, R, D> lhs;
  TileMatL1<bfloat16_t, N, D> rhs;
  pto::TileAcc<float, R, N> accum;
  TASSIGN(lhs, 0);
  TASSIGN(rhs, R * D * sizeof(bfloat16_t));
  TASSIGN(accum, 0);
  GM<bfloat16_t> a(left, GS(1, 1, 1, R, D), ST(D));
  GM<bfloat16_t> b(right, GS(1, 1, 1, N, D), ST(D));
  TLOAD(lhs, a);
  TLOAD(rhs, b);
  gemm_oneshot<bfloat16_t, float, R, N, D, true>(lhs, rhs, accum);
  GM<float> out(dst, GS(1, 1, 1, R, N), ST(row_stride));
  TSTORE(out, accum);
  pipe_barrier(PIPE_ALL);
}
#endif

// Each full chunk owns two 32-row bands. The right operand starts at
// the chunk boundary for each band and uses that band's gate reference.
AICORE inline void launch_full_akk(__gm__ bfloat16_t* left,
                                   __gm__ bfloat16_t* right,
                                   __gm__ float* lower,
                                   __gm__ const int32_t* cu,
                                   int64_t batches,
                                   int64_t tokens,
                                   int32_t heads) {
#if defined(__DAV_CUBE__)
  constexpr int C = 64;
  constexpr int Band = 32;
  constexpr int D = 128;
  int64_t work = get_block_idx();
  int64_t end = 0;
  for (int64_t seq = 0; seq < batches; ++seq) {
    const int64_t begin = end;
    end += ((cu[seq + 1] - cu[seq]) / C) * heads;
    for (; work < end; work += get_block_num()) {
      const int64_t task = work - begin;
      const int head = task % heads;
      const int64_t first = cu[seq] + (task / heads) * C;
      const int64_t token_base = int64_t(head) * tokens + first;
      auto* out = lower + (first * heads + head) * C;
      auto* lhs = left + token_base * D;
      auto* rhs = right + 2 * token_base * D;
      akk_cube_band<Band>(lhs, rhs, out, heads * C);
      akk_cube_band<C>(lhs + Band * D, rhs + C * D,
                       out + Band * heads * C, heads * C);
    }
  }
#endif
}

}  // namespace mega_kda_prefill_kkt_rows
