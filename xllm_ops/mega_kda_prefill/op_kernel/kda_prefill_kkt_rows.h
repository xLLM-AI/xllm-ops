#pragma once
#include <pto/pto-inst.hpp>
#include <type_traits>
#include "kda_kernel_utils.h"

namespace mega_kda_prefill_kkt_rows {
using namespace pto;
using namespace mega_kda_prefill_utils;

#ifdef __CCE_AICORE__
template <typename T, int R, int C, int VR = R, int VC = C>
using ND = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, VR, VC,
                SLayout::NoneBox, 512>;
using GS = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using ST = pto::Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T> using GM = GlobalTensor<T, GS, ST>;
#endif

AICORE inline void launch_kkt_rows(
    __gm__ bfloat16_t *keys, __gm__ float *gates, __gm__ float *beta,
    __gm__ float *mask, __gm__ float *out, __gm__ bfloat16_t *queries,
    __gm__ float *aqk, __gm__ int32_t *cu,
    int64_t batches, int64_t length, int64_t tokens, int32_t heads) {
#if defined(__DAV_VEC__)
  constexpr int C = 128, D = 128, N = 64, ROWS = 32;
  const int core = get_block_idx(), cores = get_block_num();
  const int vid = get_subblockid();
  int64_t total_chunks = 0;
  for (int seq = 0; seq < batches; ++seq) {
    int64_t len = cu ? cu[seq + 1] - cu[seq] : length;
    total_chunks += (len + C - 1) / C;
  }
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t chunk_heads = total_chunks * heads;
  for (int64_t task = core; task < chunk_heads * (C / ROWS); task += cores) {
    int row_group = task / chunk_heads;
    int64_t chunk_head = task % chunk_heads;
    int head = chunk_head % heads;
    int64_t chunk = chunk_head / heads, bos = 0, len = 0;
    for (int seq = 0; seq < batches; ++seq) {
      bos = cu ? cu[seq] : seq * length;
      len = cu ? cu[seq + 1] - bos : length;
      int64_t count = (len + C - 1) / C;
      if (chunk < count) break;
      chunk -= count;
    }
    int valid = len - chunk * C < C ? len - chunk * C : C;
    if (row_group * ROWS >= valid) continue;
    int64_t first = bos + chunk * C;
    int64_t hbase = int64_t(head) * tokens * D + first * D;
    ND<float, 1, C, 1, DYNAMIC> beta_f(valid);
    TASSIGN(beta_f, 189440);
    GM<float> beta_gm(beta + int64_t(head) * tokens + first, GS(1,1,1,1,valid), ST(C));
    TLOAD(beta_f, beta_gm);
    SetWaitFlag<PIPE_MTE2, PIPE_S>(0);
    for (int col0 = 0; col0 < valid && col0 < (row_group + 1) * ROWS; col0 += N) {
      int block_n = valid - col0 < N ? valid - col0 : N;
      ND<float, N, D, DYNAMIC> columns_g(block_n), columns_k(block_n);
      ND<bfloat16_t, N, D, DYNAMIC> columns_kh(block_n);
      TASSIGN(columns_g, 0); TASSIGN(columns_k, 32768); TASSIGN(columns_kh, 98304);
      GM<float> gcols(gates + hbase + col0 * D, GS(1,1,1,block_n,D), ST(D));
      GM<bfloat16_t> kcols(keys + hbase + col0 * D, GS(1,1,1,block_n,D), ST(D));
      TLOAD(columns_g, gcols); TLOAD(columns_kh, kcols);
      SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
      TCVT(columns_k, columns_kh, RoundMode::CAST_NONE); PipeBarrierVec();
      TMULS(columns_g, columns_g, -1.0f); PipeBarrierVec();
      for (int row0 = row_group * ROWS; row0 < valid && row0 < (row_group + 1) * ROWS; row0 += ROWS) {
        int rows = valid - row0 < ROWS ? valid - row0 : ROWS;
        // Load 32 row operands at once; each AIV consumes alternating rows.
        ND<float, ROWS, D, DYNAMIC> row_g(rows);
        ND<bfloat16_t, ROWS, D, DYNAMIC> row_k(rows), row_q(rows);
        ND<float, ROWS, N, DYNAMIC, DYNAMIC> row_mask(rows, block_n);
        TASSIGN(row_g, 147456); TASSIGN(row_k, 163840);
        TASSIGN(row_q, 172032); TASSIGN(row_mask, 180224);
        GM<float> grow(gates + hbase + row0 * D, GS(1,1,1,rows,D), ST(D));
        GM<bfloat16_t> krow(keys + hbase + row0 * D, GS(1,1,1,rows,D), ST(D));
        GM<bfloat16_t> qrow(queries + hbase + row0 * D, GS(1,1,1,rows,D), ST(D));
        GM<float> mrow(mask + row0 * C + col0, GS(1,1,1,rows,block_n), ST(C));
        TLOAD(row_g, grow); TLOAD(row_k, krow); TLOAD(row_q, qrow); TLOAD(row_mask, mrow);
        SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
        for (int local = vid; local < rows; local += 2) {
          int r = row0 + local;
          int n = r - col0 + 1 < block_n ? r - col0 + 1 : block_n;
          ND<float, N, D, DYNAMIC> gc(n), kc(n), diff(n), tmp(n);
          ND<float, N, 64, DYNAMIC> scratch(n);
          TASSIGN(gc, 0); TASSIGN(kc, 32768); TASSIGN(diff, 65536);
          TASSIGN(tmp, 98304); TASSIGN(scratch, 131072);
          ND<float, 1, D> gr, kr, qr;
          ND<bfloat16_t, 1, D> krh, qrh;
          TASSIGN(gr, 147456 + local * D * 4);
          TASSIGN(krh, 163840 + local * D * 2); TASSIGN(qrh, 172032 + local * D * 2);
          TASSIGN(kr, 188416); TASSIGN(qr, 188928);
          ND<float, 1, N, 1, DYNAMIC> values(block_n), masks(block_n);
          Tile<TileType::Vec, float, N, 1, BLayout::ColMajor, DYNAMIC, 1> sums(n);
          TASSIGN(values, 189952); TASSIGN(sums, 189952);
          TASSIGN(masks, 180224 + local * N * 4);
          TCVT(kr, krh, RoundMode::CAST_NONE); PipeBarrierVec();
          TCVT(qr, qrh, RoundMode::CAST_NONE); PipeBarrierVec();
          TCOLEXPANDADD(diff, gc, gr); PipeBarrierVec();
          TMINS(diff, diff, 0.0f); PipeBarrierVec();
          TEXP(diff, diff); PipeBarrierVec();
          TCOLEXPANDMUL(tmp, diff, qr); PipeBarrierVec();
          TMUL(tmp, tmp, kc); PipeBarrierVec();
          // Reuse the result only after the preceding row's KK store completes.
          SetWaitFlag<PIPE_MTE3, PIPE_V>(0);
          TEXPANDS(values, 0.0f); PipeBarrierVec();
          TROWSUM(sums, tmp, scratch); PipeBarrierVec();
          SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
          GM<float> qdst(aqk + ((first + r) * heads + head) * C + col0,
                        GS(1,1,1,1,block_n), ST(N));
          TSTORE(qdst, values);
          TCOLEXPANDMUL(tmp, diff, kr); PipeBarrierVec();
          TMUL(tmp, tmp, kc); PipeBarrierVec();
          SetWaitFlag<PIPE_MTE3, PIPE_V>(0);
          TROWSUM(sums, tmp, scratch); PipeBarrierVec();
          TMULS(values, values, beta_f.data()[r]); PipeBarrierVec();
          TMUL(values, values, masks); PipeBarrierVec();
          SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
          GM<float> dst(out + ((first + r) * heads + head) * C + col0,
                       GS(1,1,1,1,block_n), ST(N));
          TSTORE(dst, values);
        }
        pipe_barrier(PIPE_ALL);
      }
      pipe_barrier(PIPE_ALL);
    }
  }
#endif
}


} // namespace mega_kda_prefill_kkt_rows
