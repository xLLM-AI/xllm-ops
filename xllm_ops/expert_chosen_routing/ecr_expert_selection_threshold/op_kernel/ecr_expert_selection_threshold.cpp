/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "ecr_expert_selection_threshold_vf.h"

namespace {

using ecr_expert_selection_threshold_impl::CandCompactVf;
using ecr_expert_selection_threshold_impl::CountCmpVf;
using ecr_expert_selection_threshold_impl::EqIdxCompactVf;
using ecr_expert_selection_threshold_impl::kBisectValueIters;
using ecr_expert_selection_threshold_impl::kCandidateCapacity;
using ecr_expert_selection_threshold_impl::kEqSegmentElems;
using ecr_expert_selection_threshold_impl::kProbHiBits;
using ecr_expert_selection_threshold_impl::kProbLoBits;
using ecr_expert_selection_threshold_impl::kProbUbElems;
using ecr_expert_selection_threshold_impl::kVlF32;

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

__aicore__ inline float BitsToFloat(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } cvt;
  cvt.u = bits;
  return cvt.f;
}

__aicore__ inline void CopyRowIntoUb(const AscendC::LocalTensor<float>& dst,
                                     const AscendC::GlobalTensor<float>& src,
                                     uint32_t off,
                                     uint32_t count) {
  AscendC::DataCopyExtParams params;
  params.blockCount = 1;
  params.blockLen = count * sizeof(float);
  params.srcStride = 0;
  params.dstStride = 0;
  params.rsv = 0;
  AscendC::DataCopyPadExtParams<float> pad;
  pad.isPad = false;
  pad.leftPadding = 0;
  pad.rightPadding = 0;
  pad.paddingValue = 0.0f;
  AscendC::DataCopyPad(dst, src[off], params, pad);
}

template <AscendC::CMPMODE Mode>
__aicore__ inline uint32_t CountCmp(
    const AscendC::GlobalTensor<float>& row_gt,
    const AscendC::LocalTensor<float>& ub_prob,
    const AscendC::LocalTensor<float>& ub_red,
    float thr,
    uint32_t eff_len,
    bool resident) {
  uint32_t total = 0;
  for (uint32_t off = 0; off < eff_len; off += kProbUbElems) {
    const uint32_t cnt = MinU32(kProbUbElems, eff_len - off);
    uint32_t ub_off = off;
    if (!resident) {
      CopyRowIntoUb(ub_prob, row_gt, off, cnt);
      SyncEvent<AscendC::HardEvent::MTE2_V>();
      ub_off = 0;
    }
    const uint16_t loops = static_cast<uint16_t>(CeilDiv(cnt, kVlF32));
    asc_vf_call<CountCmpVf<Mode>>(
        reinterpret_cast<__ubuf__ float*>(ub_prob[ub_off].GetPhyAddr()),
        reinterpret_cast<__ubuf__ float*>(ub_red.GetPhyAddr()), thr, cnt, loops);
    // 非驻留路径同时保护标量回读与下一段搬入对输入缓冲的覆写。
    if (resident) {
      AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
      AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    } else {
      SyncEvent<AscendC::HardEvent::V_S>();
      SyncEvent<AscendC::HardEvent::V_MTE2>();
    }
    total += static_cast<uint32_t>(ub_red.GetValue(0));
  }
  return total;
}

template <typename Tiling>
__aicore__ inline void ThresholdImpl(GM_ADDR prob,
                                     GM_ADDR tau_val,
                                     GM_ADDR tau_idx,
                                     GM_ADDR n_gt,
                                     const Tiling& t) {
  const uint32_t s = t.s;
  const uint32_t e_count = t.e;
  const uint32_t b = t.b;
  const uint32_t total_tasks = e_count * b;
  const uint32_t core_num = AscendC::GetBlockNum();
  const uint32_t block_idx = AscendC::GetBlockIdx();

  AscendC::GlobalTensor<float> prob_gt;
  prob_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(prob));
  AscendC::GlobalTensor<float> tau_val_gt;
  tau_val_gt.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tau_val));
  AscendC::GlobalTensor<int32_t> tau_idx_gt;
  tau_idx_gt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tau_idx));
  AscendC::GlobalTensor<int32_t> n_gt_gt;
  n_gt_gt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(n_gt));

  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> prob_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> red_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> out_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> eq_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> cand_buf;
  pipe.InitBuffer(prob_buf, kProbUbElems * sizeof(float));
  pipe.InitBuffer(red_buf, 32);
  pipe.InitBuffer(out_buf, 32);
  // 压缩输出预留一个向量长度，防止尾部整寄存器写出越界。
  pipe.InitBuffer(eq_buf, (kEqSegmentElems + kVlF32) * sizeof(int32_t));
  pipe.InitBuffer(cand_buf, (kCandidateCapacity + kVlF32) * sizeof(float));
  AscendC::LocalTensor<float> ub_prob = prob_buf.Get<float>();
  AscendC::LocalTensor<float> ub_red = red_buf.Get<float>();
  AscendC::LocalTensor<float> ub_out_f = out_buf.Get<float>();
  AscendC::LocalTensor<int32_t> ub_out_i = out_buf.Get<int32_t>();
  AscendC::LocalTensor<int32_t> ub_eq_idx = eq_buf.Get<int32_t>();
  AscendC::LocalTensor<float> ub_cand = cand_buf.Get<float>();

  for (uint32_t task = block_idx; task < total_tasks; task += core_num) {
    const uint32_t e = task / b;
    const uint32_t i = task % b;
    const uint32_t len = t.seq_len[i];
    const uint32_t base = t.seq_base[i];
    const uint32_t k = t.seq_k[i];
    float row_tau_val = 2.0f;
    int32_t row_tau_idx = -1;
    int32_t row_n_gt = 0;

    if (len > 0 && k > 0) {
      AscendC::GlobalTensor<float> row_gt =
          prob_gt[static_cast<uint64_t>(e) * s + base];
      const bool resident = (len <= kProbUbElems);
      if (resident) {
        CopyRowIntoUb(ub_prob, row_gt, 0, len);
        SyncEvent<AscendC::HardEvent::MTE2_V>();
      }

      const uint32_t positive_count = CountCmp<AscendC::CMPMODE::GT>(
          row_gt, ub_prob, ub_red, 0.0f, len, resident);
      if (positive_count < k) {
        row_tau_val = 0.0f;
        row_n_gt = static_cast<int32_t>(positive_count);
      } else {
        // Phase A：非负 fp32 的位序与数值序一致。
        uint32_t lo = kProbLoBits;
        uint32_t hi = kProbHiBits;
        uint32_t cnt_hi = 0;
        uint32_t cnt_lo = 0;
        bool has_lo = false;
        bool compacted = false;
        uint32_t cand_cnt = 0;
        uint32_t base_gt = 0;
        for (uint32_t it = 0; it < kBisectValueIters; ++it) {
          if (hi - lo <= 1) {
            break;
          }
          const uint32_t mid = lo + ((hi - lo) >> 1);
          uint32_t c;
          if (compacted) {
            c = base_gt + CountCmp<AscendC::CMPMODE::GT>(
                               row_gt, ub_cand, ub_red, BitsToFloat(mid),
                               cand_cnt, true);
          } else {
            c = CountCmp<AscendC::CMPMODE::GT>(
                row_gt, ub_prob, ub_red, BitsToFloat(mid), len, resident);
          }
          if (c >= k) {
            lo = mid;
            cnt_lo = c;
            has_lo = true;
          } else {
            hi = mid;
            cnt_hi = c;
          }

          if (!compacted && has_lo) {
            const uint32_t n_cand = cnt_lo - cnt_hi;
            if (n_cand <= kCandidateCapacity && n_cand * 2 <= len) {
              if (resident) {
                const uint16_t cmp_loops =
                    static_cast<uint16_t>(CeilDiv(len, kVlF32));
                asc_vf_call<CandCompactVf>(
                    reinterpret_cast<__ubuf__ float*>(ub_prob.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ float*>(ub_cand.GetPhyAddr()),
                    BitsToFloat(lo), BitsToFloat(hi), len, cmp_loops);
                // Publish compacted candidates before the next VF reads them.
                AscendC::PipeBarrier<PIPE_V>();
              } else {
                uint32_t written = 0;
                for (uint32_t off = 0; off < len; off += kProbUbElems) {
                  const uint32_t cnt = MinU32(kProbUbElems, len - off);
                  CopyRowIntoUb(ub_prob, row_gt, off, cnt);
                  SyncEvent<AscendC::HardEvent::MTE2_V>();
                  const uint16_t cmp_loops =
                      static_cast<uint16_t>(CeilDiv(cnt, kVlF32));
                  asc_vf_call<CandCompactVf>(
                      reinterpret_cast<__ubuf__ float*>(ub_prob.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ float*>(
                          ub_cand[written].GetPhyAddr()),
                      BitsToFloat(lo), BitsToFloat(hi), cnt, cmp_loops);
                  SyncEvent<AscendC::HardEvent::V_S>();
                  SyncEvent<AscendC::HardEvent::V_MTE2>();
                  written += static_cast<uint32_t>(
                      AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() /
                      static_cast<int64_t>(sizeof(float)));
                }
              }
              cand_cnt = n_cand;
              base_gt = cnt_hi;
              compacted = true;
            }
          }
        }
        row_tau_val = BitsToFloat(hi);
        row_n_gt = static_cast<int32_t>(cnt_hi);
      }

      // Phase B：按行内下标升序定位第 need 个等值元素。
      if (static_cast<uint32_t>(row_n_gt) < k) {
        const uint32_t need = k - static_cast<uint32_t>(row_n_gt);
        uint32_t acc = 0;
        row_tau_idx = static_cast<int32_t>(len - 1);
        for (uint32_t off = 0; off < len; off += kEqSegmentElems) {
          const uint32_t cnt = MinU32(kEqSegmentElems, len - off);
          uint32_t ub_off = off;
          if (!resident) {
            CopyRowIntoUb(ub_prob, row_gt, off, cnt);
            SyncEvent<AscendC::HardEvent::MTE2_V>();
            ub_off = 0;
          }
          const uint16_t seg_loops =
              static_cast<uint16_t>(CeilDiv(cnt, kVlF32));
          asc_vf_call<EqIdxCompactVf>(
              reinterpret_cast<__ubuf__ float*>(ub_prob[ub_off].GetPhyAddr()),
              reinterpret_cast<__ubuf__ int32_t*>(ub_eq_idx.GetPhyAddr()),
              row_tau_val, static_cast<int32_t>(off), cnt, seg_loops);
          SyncEvent<AscendC::HardEvent::V_S>();
          if (!resident) {
            SyncEvent<AscendC::HardEvent::V_MTE2>();
          }
          const uint32_t seg_cnt = static_cast<uint32_t>(
              AscendC::GetSpr<AscendC::SpecialPurposeReg::AR>() /
              static_cast<int64_t>(sizeof(int32_t)));
          if (acc + seg_cnt >= need) {
            row_tau_idx = ub_eq_idx.GetValue(need - acc - 1);
            break;
          }
          acc += seg_cnt;
        }
      }
    }

    const uint64_t out_off = static_cast<uint64_t>(e) * b + i;
    AscendC::DataCopyExtParams w1;
    w1.blockCount = 1;
    w1.blockLen = sizeof(float);
    w1.srcStride = 0;
    w1.dstStride = 0;
    w1.rsv = 0;
    // The next task may reload ub_prob after a resident VF has read it.
    SyncEvent<AscendC::HardEvent::V_MTE2>();
    // Float/int output tensors alias the same buffer: publish scalar stores
    // and wait for each DMA read before the next scalar store reuses it.
    ub_out_f.SetValue(0, row_tau_val);
    SyncEvent<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(tau_val_gt[out_off], ub_out_f, w1);
    SyncEvent<AscendC::HardEvent::MTE3_S>();
    ub_out_i.SetValue(0, row_tau_idx);
    SyncEvent<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(tau_idx_gt[out_off], ub_out_i, w1);
    SyncEvent<AscendC::HardEvent::MTE3_S>();
    ub_out_i.SetValue(0, row_n_gt);
    SyncEvent<AscendC::HardEvent::S_MTE3>();
    AscendC::DataCopyPad(n_gt_gt[out_off], ub_out_i, w1);
    SyncEvent<AscendC::HardEvent::MTE3_S>();
  }
}

}  // namespace

extern "C" __global__ __aicore__ void ecr_expert_selection_threshold(
    GM_ADDR prob,
    GM_ADDR tau_val,
    GM_ADDR tau_idx,
    GM_ADDR n_gt,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(t, tiling);
  (void)workspace;
  ThresholdImpl(prob, tau_val, tau_idx, n_gt, t);
}
