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

#if defined(__DAV_C310_VEC__)
template <typename Lower, typename Solution, typename Scratch>
__tf__ PTO_INTERNAL void InverseRowsA5(
    typename Solution::TileDType __out__ output,
    typename Solution::TileDType __in__ initial,
    typename Lower::TileDType __in__ lower,
    typename Solution::TileDType __in__ eye,
    typename Scratch::TileDType __out__ temporary, uint16_t row, uint16_t first)
{
  auto* x = (__ubuf__ float*)__cce_get_tile_ptr(output);
  auto* solved = (__ubuf__ float*)__cce_get_tile_ptr(initial);
  auto* lo = (__ubuf__ float*)__cce_get_tile_ptr(lower);
  auto* rhs = (__ubuf__ float*)__cce_get_tile_ptr(eye);
  auto* tmp = (__ubuf__ float*)__cce_get_tile_ptr(temporary);
  // initial aliases output: explicit reads retain the TMOV dependency.
  __VEC_SCOPE__ {
    RegTensor<float> a, b, ca, cb, pa, pb, sum, right;
    MaskReg mask = PSetWithType<float>(PAT_ALL);
    constexpr auto dist = std::integral_constant<::DistVST,
        static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
    {
      mem_bar(VST_VLD);
      const uint16_t count = row - first;
      const uint16_t pairs = count / 2;
      vbr(sum, 0);
      for (uint16_t j = 0; j < pairs; ++j) {
        vlds(a, solved, (first + 2 * j) * 64, NORM);
        vlds(b, solved, (first + 2 * j + 1) * 64, NORM);
        vlds(ca, lo, row * 128 + first + 2 * j, BRC_B32);
        vlds(cb, lo, row * 128 + first + 2 * j + 1, BRC_B32);
        vmul(pa, a, ca, mask, MODE_ZEROING);
        vmul(pb, b, cb, mask, MODE_ZEROING);
        vadd(sum, pa, pb, mask, MODE_ZEROING);
        vsts(sum, tmp, j * 64, dist, mask);
      }
      if (count % 2) {
        vlds(a, solved, (row - 1) * 64, NORM);
        vlds(ca, lo, row * 128 + row - 1, BRC_B32);
        vmul(pa, a, ca, mask, MODE_ZEROING);
        vadd(sum, pa, sum, mask, MODE_ZEROING);
        const uint16_t last = pairs > 0 ? pairs - 1 : 0;
        vsts(sum, tmp, last * 64, dist, mask);
      }
      const uint16_t depth = pairs > 0 ? 31 - __builtin_clz(static_cast<uint32_t>(pairs)) : 0;
      // Match TCOLSUM(..., true), including its odd-row accumulation order.
      TColSum_Binary_TmpProc<float, 64>(a, b, sum, mask, tmp, pairs, depth);
      vlds(right, rhs, row * 64, NORM);
      vsub(sum, right, sum, mask, MODE_ZEROING);
      vsts(sum, x, row * 64, dist, mask);
    }
  }
}
#endif

AICORE inline void inverse_fp32(
    __gm__ float *lower, __gm__ float *out, __gm__ float *identity,
    __gm__ float *workspace, __gm__ int32_t *cu, int64_t batches,
    int64_t tokens, int32_t heads) {
#if defined(__DAV_VEC__)
  constexpr int C = 128, N = 64;
#if defined(__DAV_C310_VEC__)
  const int core = get_block_idx() * 2 + get_subblockid(), cores = get_block_num() * 2;
  int previous_col = -1;
#else
  const int core = get_block_idx(), cores = get_block_num();
  const int col0 = get_subblockid() * N;
#endif
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t total_chunks = 0;
  for (int seq = 0; seq < batches; ++seq)
    total_chunks += (cu[seq + 1] - cu[seq] + C - 1) / C;
  Vec<C, N> eye;
  TASSIGN(eye, 147456);
#if !defined(__DAV_C310_VEC__)
  GM geye(identity + col0, GS(1,1,1,C,N), ST(C));
  TLOAD(eye, geye);
#endif
#if defined(__DAV_C310_VEC__)
  const int64_t chunk_heads = total_chunks * heads;
  for (int64_t task = core; task < chunk_heads * 2; task += cores) {
    const int col0 = (task / chunk_heads) * N;
    const int64_t chunk_head = task % chunk_heads;
    if (col0 != previous_col) {
      GM geye(identity + col0, GS(1,1,1,C,N), ST(C));
      TLOAD(eye, geye);
      previous_col = col0;
    }
    const int head = chunk_head % heads;
    int64_t chunk = chunk_head / heads, bos = 0, len = 0;
#else
  for (int64_t task = core; task < total_chunks * heads; task += cores) {
    const int head = task % heads;
    int64_t chunk = task / heads, bos = 0, len = 0;
#endif
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
#if defined(__DAV_C310_VEC__)
    for (uint16_t row = col0 + 1; row < valid; ++row) {
      InverseRowsA5<decltype(low), decltype(x), decltype(reduction)>(x.data(), x.data(),
          low.data(), eye.data(), reduction.data(), row, col0);
    }
#else
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
#endif
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
