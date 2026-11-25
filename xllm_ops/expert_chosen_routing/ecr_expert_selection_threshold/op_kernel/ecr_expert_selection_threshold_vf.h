/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "kernel_operator.h"

namespace ecr_expert_selection_threshold_impl {

namespace reg = AscendC::MicroAPI;

constexpr uint32_t kVlF32 = 64;
constexpr uint32_t kProbUbElems = 32768;
constexpr uint32_t kEqSegmentElems = 4096;
constexpr uint32_t kCandidateCapacity = 8192;
constexpr uint32_t kBisectValueIters = 31;
constexpr uint32_t kProbLoBits = 0x00000000u;
constexpr uint32_t kProbHiBits = 0x3F800000u;

// Count the elements satisfying `CMPMODE::GT`/`EQ`-style comparison with a
// scalar and write the total to out[0].
template <AscendC::CMPMODE Mode>
__simd_vf__ inline void CountCmpVf(__ubuf__ float* src,
                                   __ubuf__ float* out,
                                   float thr,
                                   uint32_t count,
                                   uint16_t loops) {
  reg::RegTensor<float> pv;
  reg::RegTensor<float> sel;
  reg::RegTensor<float> acc;
  reg::RegTensor<float> one;
  reg::RegTensor<float> zero;
  reg::RegTensor<float> red;

  reg::Duplicate<float, float>(one, 1.0f);
  reg::Duplicate<float, float>(zero, 0.0f);
  reg::Duplicate<float, float>(acc, 0.0f);

  uint32_t full_cnt = kVlF32;
  reg::MaskReg full = reg::UpdateMask<float>(full_cnt);

  uint32_t remain = count;
  for (uint16_t i = 0; i < loops; ++i) {
    reg::MaskReg mask = reg::UpdateMask<float>(remain);
    const uint32_t off = static_cast<uint32_t>(i) * kVlF32;

    reg::LoadAlign<float>(pv, src + off);

    reg::MaskReg m_cmp;
    reg::Compares<float, Mode>(m_cmp, pv, thr, mask);
    reg::Select<float>(sel, one, zero, m_cmp);
    reg::Add(acc, acc, sel, full);
  }

  reg::Reduce<reg::ReduceType::SUM, float, float, reg::MaskMergeMode::ZEROING>(
      red, acc, full);
  reg::MaskReg single = reg::CreateMask<float, reg::MaskPattern::VL1>();
  reg::StoreAlign<float>(out, red, single);
  reg::LocalMemBar<reg::MemType::VEC_STORE, reg::MemType::SCALAR_LOAD>();
}

// Squeeze the indices equal to tau_val into idx_out in ascending index order.
__simd_vf__ inline void EqIdxCompactVf(__ubuf__ float* src,
                                       __ubuf__ int32_t* idx_out,
                                       float tau_val,
                                       int32_t idx_base,
                                       uint32_t count,
                                       uint16_t loops) {
  reg::RegTensor<float> pv;
  reg::RegTensor<int32_t> idv;
  reg::RegTensor<int32_t> i_res;
  reg::UnalignRegForStore ui;

  uint32_t full_cnt = kVlF32;
  reg::MaskReg full = reg::UpdateMask<float>(full_cnt);

  reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
  uint32_t remain = count;
  for (uint16_t i = 0; i < loops; ++i) {
    reg::MaskReg mask = reg::UpdateMask<float>(remain);
    const uint32_t off = static_cast<uint32_t>(i) * kVlF32;

    reg::LoadAlign<float>(pv, src + off);

    reg::MaskReg m_eq;
    reg::Compares<float, AscendC::CMPMODE::EQ>(m_eq, pv, tau_val, mask);
    reg::And(m_eq, m_eq, mask, full);

    reg::Arange(idv, idx_base + static_cast<int32_t>(off));
    reg::Squeeze<int32_t, reg::GatherMaskMode::STORE_REG>(i_res, idv, m_eq);
    reg::StoreUnAlign<int32_t, reg::PostLiteral::POST_MODE_UPDATE>(
        idx_out, i_res, ui);
  }
  reg::StoreUnAlignPost<int32_t>(idx_out, ui);
  reg::LocalMemBar<reg::MemType::VEC_STORE, reg::MemType::SCALAR_LOAD>();
}

// Squeeze the elements within (lo_val, hi_val] into dst for the next bisection
// rounds.
__simd_vf__ inline void CandCompactVf(__ubuf__ float* src,
                                      __ubuf__ float* dst,
                                      float lo_val,
                                      float hi_val,
                                      uint32_t count,
                                      uint16_t loops) {
  reg::RegTensor<float> pv;
  reg::RegTensor<float> res;
  reg::UnalignRegForStore ui;

  uint32_t full_cnt = kVlF32;
  reg::MaskReg full = reg::UpdateMask<float>(full_cnt);

  reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
  uint32_t remain = count;
  for (uint16_t i = 0; i < loops; ++i) {
    reg::MaskReg mask = reg::UpdateMask<float>(remain);
    const uint32_t off = static_cast<uint32_t>(i) * kVlF32;

    reg::LoadAlign<float>(pv, src + off);

    reg::MaskReg m_gt;
    reg::MaskReg m_le;
    reg::Compares<float, AscendC::CMPMODE::GT>(m_gt, pv, lo_val, mask);
    reg::Compares<float, AscendC::CMPMODE::LE>(m_le, pv, hi_val, mask);
    reg::And(m_gt, m_gt, m_le, full);
    reg::And(m_gt, m_gt, mask, full);

    reg::Squeeze<float, reg::GatherMaskMode::STORE_REG>(res, pv, m_gt);
    reg::StoreUnAlign<float, reg::PostLiteral::POST_MODE_UPDATE>(dst, res, ui);
  }
  reg::StoreUnAlignPost<float>(dst, ui);
  reg::LocalMemBar<reg::MemType::VEC_STORE, reg::MemType::SCALAR_LOAD>();
}

}  // namespace ecr_expert_selection_threshold_impl
