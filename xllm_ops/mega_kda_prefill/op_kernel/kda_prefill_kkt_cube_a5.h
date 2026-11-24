#pragma once
#include "kda_prefill_chunk_h_a5.h"

// Experimental A5-only off-diagonal KKT; diagonal blocks keep the exact VF.
namespace mega_kda_prefill_kkt_cube_a5 {
#if __CCE_AICORE__ == 310
using namespace pto;
using namespace mega_kda_prefill_utils;
namespace H = mega_kda_prefill_chunk_h_kda;
using GS = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using ST = pto::Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T> using GM = GlobalTensor<T, GS, ST>;
template <typename T, int R, int C, int VR = R>
using V = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, VR, C, SLayout::NoneBox, 512>;

// Keep the AIV-only scale and BF16->FP32 multiply in VF registers.  The
// reference path expresses the same sequence as TCOLEXPANDSUB/TMULS/TMINS/
// TEXP followed by TCVT/TMUL, which materializes scale and the converted
// tile in UB between each operator.  This helper deliberately keeps the
// element order, FP32 operations, and BF16 expansion unchanged; it only
// removes those temporary UB tiles.  The caller still publishes the same
// FP32 scratch matrices through GM for the Cube path.
template <bool NegateScale, typename G, typename Pivot, typename Bf, typename Out>
__tf__ PTO_INTERNAL void ScaleMulRowsA5(
    typename G::TileDType __in__ g_in,
    typename Pivot::TileDType __in__ pivot_in,
    typename Bf::TileDType __in__ bf_in,
    typename Out::TileDType __out__ out,
    uint16_t rows)
{
  auto* gp = (__ubuf__ float*)__cce_get_tile_ptr(g_in);
  auto* pp = (__ubuf__ float*)__cce_get_tile_ptr(pivot_in);
  auto* bp = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(bf_in);
  auto* op = (__ubuf__ float*)__cce_get_tile_ptr(out);
  __VEC_SCOPE__ {
    RegTensor<float> g, pivot, delta, x, product;
    RegTensor<bfloat16_t> bf;
    MaskReg mask = PSetWithType<float>(PAT_ALL);
    MaskReg bfMask = PSetWithType<half>(PAT_ALL);
    constexpr auto dist = std::integral_constant<::DistVST,
        static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
    for (uint16_t half = 0; half < 2; ++half) {
      vlds(pivot, pp, half * 64, NORM);
      for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t offset = row * 128 + half * 64;
        vlds(g, gp, offset, NORM);
        vsub(delta, g, pivot, mask, MODE_ZEROING);
        if constexpr (NegateScale) {
          vmuls(delta, delta, -1.0f, mask, MODE_ZEROING);
        }
        vmins(delta, delta, 0.0f, mask, MODE_ZEROING);
        vexp(delta, delta, mask, MODE_ZEROING);
        vlds(bf, bp, offset, UNPK_B16);
        vcvt(x, bf, bfMask, PART_EVEN);
        vmul(product, x, delta, mask, MODE_ZEROING);
        vsts(product, op, offset, dist, mask);
      }
    }
  }
}

// vid=0 uses the same scale for target K and Q.  Load/convert both BF16
// inputs while that scale is live, then publish independent UB outputs.  This
// avoids the reference K-store -> barrier -> Q-load chain without changing
// either GM matrix or the per-element arithmetic.
template <bool NegateScale, typename G, typename Pivot, typename Bf, typename Out>
__tf__ PTO_INTERNAL void ScaleMulPairRowsA5(
    typename G::TileDType __in__ g_in,
    typename Pivot::TileDType __in__ pivot_in,
    typename Bf::TileDType __in__ bf_k,
    typename Bf::TileDType __in__ bf_q,
    typename Out::TileDType __out__ k_out,
    typename Out::TileDType __out__ q_out,
    uint16_t rows)
{
  auto* gp = (__ubuf__ float*)__cce_get_tile_ptr(g_in);
  auto* pp = (__ubuf__ float*)__cce_get_tile_ptr(pivot_in);
  auto* kp = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(bf_k);
  auto* qp = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(bf_q);
  auto* ko = (__ubuf__ float*)__cce_get_tile_ptr(k_out);
  auto* qo = (__ubuf__ float*)__cce_get_tile_ptr(q_out);
  __VEC_SCOPE__ {
    RegTensor<float> g, pivot, delta, k, q, k_product, q_product;
    RegTensor<bfloat16_t> bk, bq;
    MaskReg mask = PSetWithType<float>(PAT_ALL);
    MaskReg bfMask = PSetWithType<half>(PAT_ALL);
    constexpr auto dist = std::integral_constant<::DistVST,
        static_cast<::DistVST>(GetDistVst<float, pto::DistVST::DIST_NORM>())>();
    for (uint16_t half = 0; half < 2; ++half) {
      vlds(pivot, pp, half * 64, NORM);
      for (uint16_t row = 0; row < rows; ++row) {
        const uint32_t offset = row * 128 + half * 64;
        vlds(g, gp, offset, NORM);
        vsub(delta, g, pivot, mask, MODE_ZEROING);
        if constexpr (NegateScale) {
          vmuls(delta, delta, -1.0f, mask, MODE_ZEROING);
        }
        vmins(delta, delta, 0.0f, mask, MODE_ZEROING);
        vexp(delta, delta, mask, MODE_ZEROING);
        vlds(bk, kp, offset, UNPK_B16);
        vlds(bq, qp, offset, UNPK_B16);
        vcvt(k, bk, bfMask, PART_EVEN);
        vcvt(q, bq, bfMask, PART_EVEN);
        vmul(k_product, k, delta, mask, MODE_ZEROING);
        vmul(q_product, q, delta, mask, MODE_ZEROING);
        vsts(k_product, ko, offset, dist, mask);
        vsts(q_product, qo, offset, dist, mask);
      }
    }
  }
}

#if defined(__DAV_CUBE__)
// The AIC off-diagonal GEMM keeps the original R=32,D=128 arithmetic and
// synchronization, but specializes the column width to the actual rb block.
// The AIV producer and its three task/flag streams remain in Run unchanged.
// The second GEMM's GM scratch is deliberately still row-strided by 96: the
// AIV consumer uses that fixed stride when it reads the lower intermediate.
template <int N>
AICORE PTO_INLINE void RunOffdiagCubeN(
    __gm__ float* base, __gm__ float* aqk, int heads, int64_t start,
    int rb, int cb, int head, int nr, int nc) {
  constexpr int R = 32, D = 128, Slot = R * D, ScratchN = 96;
  wait_intra_block(PIPE_MTE2, 10);
  wait_intra_block(PIPE_MTE2, 10 + SYNC_FLAG_ID_MAX);
  H::TileMatL1<float, R, D> a;
  H::TileMatL1<float, N, D> b;
  H::DynMatL1<float, R, D> aload(nr,D);
  H::DynMatL1<float, N, D> bload(nc,D);
  TileAcc<float, R, N> acc;
  TASSIGN(a,0); TASSIGN(aload,0); TASSIGN(b,16384); TASSIGN(bload,16384); TASSIGN(acc,0);
  GM<float> bsrc(base + 2 * Slot, GS(1,1,1,nc,D), ST(D));
  TLOAD(bload, bsrc); if (nc < N) TFILLPAD(bload,bload);
  for (int which = 0; which < 2; ++which) {
    GM<float> asrc(base + which * Slot, GS(1,1,1,nr,D), ST(D));
    TLOAD(aload, asrc); if (nr < R) TFILLPAD(aload, aload);
    H::gemm_v0<float,float,R,N,D,R,N,D,D,false,true>(a,b,acc,true);
    H::DynAccTile<float,R,N> result(nr,nc); TASSIGN(result,0);
    GM<float> dst(which == 0 ? aqk + ((start + rb * R) * heads + head) * 128 + cb * R
                            : base + 32768,
                  GS(1,1,1,nr,nc), ST(which == 0 ? heads * 128 : ScratchN));
    TSTORE(dst,result); pipe_barrier(PIPE_ALL);
  }
  set_intra_block(PIPE_FIX,12);
  set_intra_block(PIPE_FIX,12 + SYNC_FLAG_ID_MAX);
}
#endif

AICORE PTO_INLINE void Run(__gm__ bfloat16_t* q, __gm__ bfloat16_t* k,
    __gm__ float* gates, __gm__ float* beta, __gm__ float* lower,
    __gm__ float* aqk, __gm__ float* scratch, __gm__ int32_t* cu,
    int64_t batches, int64_t tokens, int heads) {
  constexpr int R = 32, N = 96, D = 128, Slot = R * D;
  const int core = get_block_idx(), cores = get_block_num();
  int64_t chunks = 0;
  for (int b = 0; b < batches; ++b) chunks += (cu[b+1] - cu[b] + 127) / 128;
  // Reuse the existing H-stage per-core workspace before H starts.
  auto* base = scratch + int64_t(core) * 5 * 16384;
  for (int64_t task = core; task < chunks * heads * 3; task += cores) {
    const int rb = task % 3 + 1, cb = 0;
    const int nc = rb * R;
    const int head = (task / 3) % heads;
    int64_t chunk = task / (3 * heads), bos = 0, len = 0;
    for (int b = 0; b < batches; ++b) {
      bos = cu[b]; len = cu[b+1] - bos;
      const int64_t count = (len + 127) / 128;
      if (chunk < count) break;
      chunk -= count;
    }
    const int valid = len - chunk * 128 < 128 ? len - chunk * 128 : 128;
    if (rb * R >= valid) continue;
    const int nr = valid - rb * R < R ? valid - rb * R : R;
    const int64_t start = bos + chunk * 128;
    const int64_t hbase = (int64_t(head) * tokens + start) * D;
#if defined(__DAV_C310_VEC__)
    set_mask_norm(); set_vector_mask(-1, -1);
    const int vid = get_subblockid();
    const int row = (vid == 0 ? rb : cb) * R;
    const int n = vid == 0 ? nr : nc;
    V<float, N, D, DYNAMIC> g(n), x(n), q_out(n), k_out(n);
    V<bfloat16_t, N, D, DYNAMIC> bf(n), bf_q(n);
    V<float, 1, D> pivot;
    // g/bf/pivot retain their original lifetime.  q_out/k_out are separate
    // UB destinations so vid=0 can compute both matrices before either GM
    // store.  Actual byte ranges (not the N=96 type capacity) are:
    //   vid0 (n=nr<=32): bf_q [172544,180736), q_out [184320,200704),
    //                    k_out [200704,217088);
    //   vid1 (n=nc<=96): x [184320,233472).
    // The shared g/bf slots end at 49152/172032, pivot ends at 172544, so
    // these disjoint lifetimes remain below the 256 KiB UB limit.
    TASSIGN(g, 0); TASSIGN(bf, 147456); TASSIGN(pivot, 172032);
    TASSIGN(bf_q, 172544); TASSIGN(x, 184320);
    TASSIGN(q_out, 184320); TASSIGN(k_out, 200704);
    GM<float> gp(gates + hbase + row * D, GS(1,1,1,n,D), ST(D));
    GM<float> pp(gates + hbase + rb * R * D, GS(1,1,1,1,D), ST(D));
    TLOAD(g, gp); TLOAD(pivot, pp);
    SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    GM<bfloat16_t> kp(k + hbase + row * D, GS(1,1,1,n,D), ST(D));
    if (vid == 0) {
      GM<bfloat16_t> qp(q + hbase + row * D, GS(1,1,1,n,D), ST(D));
      TLOAD(bf, kp); TLOAD(bf_q, qp);
    } else {
      TLOAD(bf, kp);
    }
    SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    GM<float> ks(base + (vid == 0 ? Slot : 2 * Slot), GS(1,1,1,n,D), ST(D));
    if (vid == 0) {
      GM<float> qs(base, GS(1,1,1,n,D), ST(D));
      ScaleMulPairRowsA5<false, decltype(g), decltype(pivot), decltype(bf), decltype(q_out)>(
          g.data(), pivot.data(), bf.data(), bf_q.data(), k_out.data(), q_out.data(), n);
      SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
      TSTORE(ks, k_out); TSTORE(qs, q_out);
    } else {
      ScaleMulRowsA5<true, decltype(g), decltype(pivot), decltype(bf), decltype(x)>(
          g.data(), pivot.data(), bf.data(), x.data(), n);
      SetWaitFlag<PIPE_V, PIPE_MTE3>(0); TSTORE(ks, x);
    }
    // Preserve the reference's MTE3-to-next-task ordering before signalling
    // the Cube.  In particular, the AIV BF16 input slots are not reused by a
    // later task while their stores are still in flight.
    pipe_barrier(PIPE_ALL);
    set_intra_block(PIPE_MTE3, 10);
    wait_intra_block(PIPE_MTE2, 12);
    if (vid == 0) {
      Tile<TileType::Vec, float, R, N, BLayout::RowMajor, DYNAMIC, DYNAMIC,
           SLayout::NoneBox, 512> kk(nr,nc), expanded(nr,nc);
      Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, DYNAMIC, 1> bet(nr);
      TASSIGN(kk, 0); TASSIGN(expanded, 12288); TASSIGN(bet, 24576);
      GM<float> bg(beta + int64_t(head) * tokens + start + rb * R, GS(1,1,1,1,nr), ST(R));
      // Load only nr beta elements into the contiguous DN column storage.
      Tile<TileType::Vec, float, 1, R, BLayout::RowMajor, 1, DYNAMIC> bt(nr);
      TASSIGN(bt, 24576); TLOAD(bt, bg);
      GM<float> kg(base + 32768, GS(1,1,1,nr,nc), ST(N));
      TLOAD(kk, kg); SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
      TROWEXPAND(expanded, bet); TMUL(kk, kk, expanded);
      GM<float> dst(lower + ((start + rb * R) * heads + head) * 128 + cb * R,
                    GS(1,1,1,nr,nc), ST(heads * 128));
      SetWaitFlag<PIPE_V, PIPE_MTE3>(0); TSTORE(dst, kk);
    }
    pipe_barrier(PIPE_ALL);
#elif defined(__DAV_CUBE__)
    if (nc == 32) {
      RunOffdiagCubeN<32>(base, aqk, heads, start, rb, cb, head, nr, nc);
    } else if (nc == 64) {
      RunOffdiagCubeN<64>(base, aqk, heads, start, rb, cb, head, nr, nc);
    } else {
      RunOffdiagCubeN<96>(base, aqk, heads, start, rb, cb, head, nr, nc);
    }
#endif
  }
}
#endif
} // namespace mega_kda_prefill_kkt_cube_a5
