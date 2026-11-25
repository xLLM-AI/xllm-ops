/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_token_routing_statistics_vf.h"

namespace {

using ecr_token_routing_statistics_impl::AccumVf;
using ecr_token_routing_statistics_impl::FillZeroVf;
using ecr_token_routing_statistics_impl::kMaxBatch;
using ecr_token_routing_statistics_impl::kMaxExperts;
using ecr_token_routing_statistics_impl::kRampLength;
using ecr_token_routing_statistics_impl::kTokenTile;
using ecr_token_routing_statistics_impl::kVlF32;

// Each dependency is issued and consumed locally; no event remains outstanding.
template <AscendC::HardEvent Event>
__aicore__ inline void SyncEvent() {
  AscendC::SetFlag<Event>(EVENT_ID0);
  AscendC::WaitFlag<Event>(EVENT_ID0);
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b) {
  return b == 0 ? 0 : (a + b - 1) / b;
}

template <typename T>
__aicore__ inline void CopyIntoUb(const AscendC::LocalTensor<T>& dst,
                                  const AscendC::GlobalTensor<T>& src,
                                  uint32_t off,
                                  uint32_t count) {
  AscendC::DataCopyExtParams params;
  params.blockCount = 1;
  params.blockLen = count * sizeof(T);
  params.srcStride = 0;
  params.dstStride = 0;
  params.rsv = 0;
  AscendC::DataCopyPadExtParams<T> pad;
  pad.isPad = false;
  pad.leftPadding = 0;
  pad.rightPadding = 0;
  pad.paddingValue = 0;
  AscendC::DataCopyPad(dst, src[off], params, pad);
}

template <typename T>
__aicore__ inline void CopyOutOfUb(const AscendC::GlobalTensor<T>& dst,
                                   const AscendC::LocalTensor<T>& src,
                                   uint32_t count) {
  AscendC::DataCopyExtParams params;
  params.blockCount = 1;
  params.blockLen = count * sizeof(T);
  params.srcStride = 0;
  params.dstStride = 0;
  params.rsv = 0;
  AscendC::DataCopyPad(dst, src, params);
}

template <typename Tiling>
__aicore__ inline void StatisticsImpl(GM_ADDR prob,
                                      GM_ADDR tau_val,
                                      GM_ADDR tau_idx,
                                      GM_ADDR sums,
                                      GM_ADDR cnts,
                                      const Tiling& t) {
  const uint32_t s = t.s;
  const uint32_t experts = t.e;
  const uint32_t batch = t.b;
  const uint32_t total_tasks = t.total_tasks;
  const uint32_t core_num = AscendC::GetBlockNum();
  const uint32_t block_idx = AscendC::GetBlockIdx();
  const uint32_t tau_count = experts * batch;

  AscendC::GlobalTensor<float> prob_gt;
  prob_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(prob));
  AscendC::GlobalTensor<float> tau_val_gt;
  tau_val_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tau_val));
  AscendC::GlobalTensor<int32_t> tau_idx_gt;
  tau_idx_gt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tau_idx));
  AscendC::GlobalTensor<float> sums_gt;
  sums_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sums));
  AscendC::GlobalTensor<float> cnts_gt;
  cnts_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cnts));

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> tv_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ti_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> ramp_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> p_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sum_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> cnt_buf;
  pipe.InitBuffer(tv_buf, kMaxExperts * kMaxBatch * sizeof(float));
  pipe.InitBuffer(ti_buf, kMaxExperts * kMaxBatch * sizeof(int32_t));
  pipe.InitBuffer(ramp_buf, kRampLength * sizeof(float));
  pipe.InitBuffer(p_buf, kTokenTile * sizeof(float));
  pipe.InitBuffer(sum_buf, kTokenTile * sizeof(float));
  pipe.InitBuffer(cnt_buf, kTokenTile * sizeof(float));
  AscendC::LocalTensor<float> ub_tau_val = tv_buf.Get<float>();
  AscendC::LocalTensor<int32_t> ub_tau_idx = ti_buf.Get<int32_t>();
  AscendC::LocalTensor<float> ub_ramp = ramp_buf.Get<float>();
  AscendC::LocalTensor<float> ub_p = p_buf.Get<float>();
  AscendC::LocalTensor<float> ub_sum = sum_buf.Get<float>();
  AscendC::LocalTensor<float> ub_cnt = cnt_buf.Get<float>();

  // Bring the whole threshold table in once; the lane ramp is cheap to build.
  CopyIntoUb(ub_tau_val, tau_val_gt, 0, tau_count);
  CopyIntoUb(ub_tau_idx, tau_idx_gt, 0, tau_count);
  SyncEvent<AscendC::HardEvent::MTE2_S>();
  SyncEvent<AscendC::HardEvent::MTE2_V>();
  for (uint32_t i = 0; i < kRampLength; ++i) {
    ub_ramp.SetValue(i, static_cast<float>(i));
  }
  SyncEvent<AscendC::HardEvent::S_V>();

  for (uint32_t task = block_idx; task < total_tasks; task += core_num) {
    uint32_t sample = 0;
    bool found = false;
    for (uint32_t b = 0; b < batch; ++b) {
      const uint32_t tiles = CeilDiv(t.seq_len[b], kTokenTile);
      if (task >= t.task_base[b] && task < t.task_base[b] + tiles) {
        sample = b;
        found = true;
        break;
      }
    }
    if (!found) {
      continue;
    }

    const uint32_t len = t.seq_len[sample];
    const uint32_t base = t.seq_base[sample];
    const uint32_t tile_local = (task - t.task_base[sample]) * kTokenTile;
    if (tile_local >= len) {
      continue;
    }
    const uint32_t count = MinU32(kTokenTile, len - tile_local);
    const uint32_t tok_start = base + tile_local;
    const uint16_t loops = static_cast<uint16_t>(CeilDiv(count, kVlF32));

    asc_vf_call<FillZeroVf>(
        reinterpret_cast<__ubuf__ float*>(ub_sum.GetPhyAddr()), count, loops);
    asc_vf_call<FillZeroVf>(
        reinterpret_cast<__ubuf__ float*>(ub_cnt.GetPhyAddr()), count, loops);
    AscendC::PipeBarrier<PIPE_V>();

    for (uint32_t e = 0; e < experts; ++e) {
      const uint32_t tau_pos = e * batch + sample;
      const float threshold = ub_tau_val.GetValue(tau_pos);
      const int32_t threshold_idx = ub_tau_idx.GetValue(tau_pos);
      const float idx_limit =
          static_cast<float>(threshold_idx - static_cast<int32_t>(tile_local));

      AscendC::GlobalTensor<float> row_gt =
          prob_gt[static_cast<uint64_t>(e) * s + tok_start];
      CopyIntoUb(ub_p, row_gt, 0, count);
      SyncEvent<AscendC::HardEvent::MTE2_V>();
      asc_vf_call<AccumVf>(
          reinterpret_cast<__ubuf__ float*>(ub_p.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float*>(ub_ramp.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float*>(ub_sum.GetPhyAddr()),
          reinterpret_cast<__ubuf__ float*>(ub_cnt.GetPhyAddr()),
          threshold, idx_limit, count, loops);
      // Protect ub_p from the next DMA overwrite, including the next task.
      SyncEvent<AscendC::HardEvent::V_MTE2>();
    }
    SyncEvent<AscendC::HardEvent::V_MTE3>();

    CopyOutOfUb(sums_gt[tok_start], ub_sum, count);
    CopyOutOfUb(cnts_gt[tok_start], ub_cnt, count);
    // Output DMA must finish before the next task zeroes sum/cnt.
    SyncEvent<AscendC::HardEvent::MTE3_V>();
  }
}

}  // namespace

extern "C" __global__ __aicore__ void ecr_token_routing_statistics(
    GM_ADDR prob,
    GM_ADDR tau_val,
    GM_ADDR tau_idx,
    GM_ADDR sums,
    GM_ADDR cnts,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(t, tiling);
  (void)workspace;
  StatisticsImpl(prob, tau_val, tau_idx, sums, cnts, t);
}
