#pragma once
#include <pto/pto-inst.hpp>
#include <type_traits>
#include "kda_kernel_utils.h"

namespace mega_kda_prefill_inverse_fp32 {
using namespace pto;
using namespace mega_kda_prefill_utils;

#ifdef __CCE_AICORE__
using GS = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using ST = pto::Stride<1, 1, 1, DYNAMIC, 1>;
using GM = GlobalTensor<float, GS, ST>;
template <int R, int C, int VR = R>
using Vec = Tile<TileType::Vec, float, R, C, BLayout::RowMajor,
                 VR, C, SLayout::NoneBox, 512>;
#endif

AICORE inline void inverse_fp32(
    __gm__ float *lower, __gm__ float *out, __gm__ float *identity,
    __gm__ float *workspace, __gm__ int32_t *cu, int64_t batches,
    int64_t tokens, int32_t heads) {
#if defined(__DAV_VEC__)
  constexpr int C = 128, N = 64;
  const int core = get_block_idx(), cores = get_block_num();
  const int col0 = get_subblockid() * N;
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t total_chunks = 0;
  for (int seq = 0; seq < batches; ++seq)
    total_chunks += (cu[seq + 1] - cu[seq] + C - 1) / C;
  Vec<C, N> eye;
  TASSIGN(eye, 147456);
  GM geye(identity + col0, GS(1,1,1,C,N), ST(C));
  TLOAD(eye, geye);
  for (int64_t task = core; task < total_chunks * heads; task += cores) {
    const int head = task % heads;
    int64_t chunk = task / heads, bos = 0, len = 0;
    for (int seq = 0; seq < batches; ++seq) {
      bos = cu[seq];
      len = cu[seq + 1] - bos;
      const int64_t count = (len + C - 1) / C;
      if (chunk < count) break;
      chunk -= count;
    }
    const int valid = len - chunk * C < C ? len - chunk * C : C;
    const int64_t start = bos + chunk * C;
    Vec<C, C, DYNAMIC> low(valid);
    Vec<C, N> x;
    Vec<64, N> reduction;
    Vec<C, 8> broadcast;
    TASSIGN(low, 0);
    TASSIGN(x, 65536);
    TASSIGN(reduction, 131072);
    TASSIGN(broadcast, 180224);
    GM src(lower + (start * heads + head) * C,
           GS(1,1,1,valid,C), ST(heads * C));
    TLOAD(low, src);
    SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    TMOV(x, eye);
    PipeBarrierVec();
    // X[i,:] = I[i,:] - sum_{j<i} L[i,j] X[j,:]. Each AIV owns its RHS.
    for (int row = 1; row < valid; ++row) {
      Vec<C, N, DYNAMIC> solved(row), products(row);
      Tile<TileType::Vec, float, C, 1, BLayout::ColMajor, DYNAMIC, 1> coeff(row);
      Vec<1, N> result, rhs;
      TASSIGN(solved, 65536);
      TASSIGN(products, 98304);
      TASSIGN(coeff, row * C * 4);
      TASSIGN(result, 65536 + row * N * 4);
      TASSIGN(rhs, 147456 + row * N * 4);
      TROWEXPANDMUL(products, solved, coeff, broadcast);
      PipeBarrierVec();
      TCOLSUM(result, products, reduction, true);
      PipeBarrierVec();
      TSUB(result, rhs, result);
      PipeBarrierVec();
    }
    Vec<C, N, DYNAMIC> store(valid);
    TASSIGN(store, 65536);
    GM dst(out + (start * heads + head) * C + col0,
           GS(1,1,1,valid,N), ST(heads * C));
    SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
    TSTORE(dst, store);
    // Return both the solution and lower buffers before the next task reuses UB.
    pipe_barrier(PIPE_ALL);
  }
#endif
}


} // namespace mega_kda_prefill_inverse_fp32
