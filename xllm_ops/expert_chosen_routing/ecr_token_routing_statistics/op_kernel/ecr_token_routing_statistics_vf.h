/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "kernel_operator.h"

namespace ecr_token_routing_statistics_impl {

namespace reg = AscendC::MicroAPI;

constexpr uint32_t kVlF32 = 64;
constexpr uint32_t kMaxExperts = 64;
constexpr uint32_t kMaxBatch = 32;
constexpr uint32_t kRampLength = 512;
constexpr uint32_t kTokenTile = 512;

// Accumulate per-token selection over one [expert, tile] row: a token counts
// when its probability is above tau, or equal to tau with a sample-local index
// (ramp + tile offset) not past tau_idx.
__simd_vf__ inline void AccumVf(__ubuf__ float* p,
                                __ubuf__ float* ramp,
                                __ubuf__ float* acc_sum,
                                __ubuf__ float* acc_cnt,
                                float tau_val,
                                float idx_limit,
                                uint32_t count,
                                uint16_t loops) {
  reg::RegTensor<float> pv;
  reg::RegTensor<float> rv;
  reg::RegTensor<float> av;
  reg::RegTensor<float> cv;
  reg::RegTensor<float> one;
  reg::RegTensor<float> zero;
  reg::RegTensor<float> ga;
  reg::RegTensor<float> eb;
  reg::RegTensor<float> ic;
  reg::RegTensor<float> sv;
  reg::RegTensor<float> contrib;

  reg::Duplicate(one, 1.0f);
  reg::Duplicate(zero, 0.0f);

  uint32_t full_cnt = kVlF32;
  reg::MaskReg full = reg::UpdateMask<float>(full_cnt);

  uint32_t remain = count;
  for (uint16_t i = 0; i < loops; ++i) {
    reg::MaskReg mask = reg::UpdateMask<float>(remain);
    const uint32_t off = static_cast<uint32_t>(i) * kVlF32;

    reg::LoadAlign<float>(pv, p + off);
    reg::LoadAlign<float>(rv, ramp + off);
    reg::LoadAlign<float>(av, acc_sum + off);
    reg::LoadAlign<float>(cv, acc_cnt + off);

    reg::MaskReg m_gt;
    reg::MaskReg m_eq;
    reg::MaskReg m_le;
    reg::Compares<float, AscendC::CMPMODE::GT>(m_gt, pv, tau_val, mask);
    reg::Select<float>(ga, one, zero, m_gt);
    reg::Compares<float, AscendC::CMPMODE::EQ>(m_eq, pv, tau_val, mask);
    reg::Select<float>(eb, one, zero, m_eq);
    reg::Compares<float, AscendC::CMPMODE::LE>(m_le, rv, idx_limit, mask);
    reg::Select<float>(ic, one, zero, m_le);

    reg::Mul(eb, eb, ic, full);
    reg::Add(sv, ga, eb, full);

    reg::Mul(contrib, pv, sv, full);
    reg::Add(av, av, contrib, full);
    reg::Add(cv, cv, sv, full);

    reg::StoreAlign<float>(acc_sum + off, av, mask);
    reg::StoreAlign<float>(acc_cnt + off, cv, mask);
  }
  // The accumulators are read back by the next expert's VF call.
  reg::LocalMemBar<reg::MemType::VEC_STORE, reg::MemType::VEC_LOAD>();
}

__simd_vf__ inline void FillZeroVf(__ubuf__ float* dst,
                                   uint32_t count,
                                   uint16_t loops) {
  reg::RegTensor<float> zero;
  reg::Duplicate(zero, 0.0f);

  uint32_t remain = count;
  for (uint16_t i = 0; i < loops; ++i) {
    reg::MaskReg mask = reg::UpdateMask<float>(remain);
    const uint32_t off = static_cast<uint32_t>(i) * kVlF32;
    reg::StoreAlign<float>(dst + off, zero, mask);
  }
  reg::LocalMemBar<reg::MemType::VEC_STORE, reg::MemType::VEC_LOAD>();
}

}  // namespace ecr_token_routing_statistics_impl
