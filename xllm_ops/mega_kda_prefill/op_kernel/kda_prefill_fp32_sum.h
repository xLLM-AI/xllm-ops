/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

#include <pto/pto-inst.hpp>
#include <cstdint>

namespace kda_prefill {

// Shared byte layout for the bulk Conv frontend. R16/R32 retain their original
// offsets. R64 reuses the inactive window and narrow-conversion storage only
// after their consumers have drained at the existing vector barriers.
template <int R>
struct ConvRowsLayout {
  static_assert(R == 16 || R == 32 || R == 64);
  static constexpr int kD = 128;
  static constexpr int kFloatTile = R * kD * sizeof(float);
  static constexpr int kTemporary = R == 64 ? kFloatTile : 2 * kFloatTile;
  static constexpr int kBf = kTemporary + kFloatTile;
  static constexpr int kHalf = kBf + R * kD * 2;
  static constexpr int kWeight = kHalf + R * kD * 2;
  static constexpr int kWeightBf = kWeight + kD * sizeof(float);
  static constexpr int kScalar = kWeightBf + kD * 2;
  static constexpr int kReduce = kScalar + R * 8 * sizeof(float);
  static constexpr int kNorm = kReduce + R * 64 * sizeof(float);
  static constexpr int kCachedWeights = kNorm + kFloatTile;
  static constexpr int kWindow = kCachedWeights + 4 * kD * sizeof(float);
  static constexpr int kRawWindow = R == 64 ? kBf : kWindow + (R + 3) * kD * sizeof(float);
  static constexpr int kInput = R == 64 ? kWindow : kFloatTile;
  static constexpr int kFrontendLimit = R == 16 ? 65536 : R == 32 ? 131072 : 192 * 1024;
  static_assert(kWindow + (R + 3) * kD * sizeof(float) <= kFrontendLimit);
  static_assert(kRawWindow + (R + 3) * kD * 2 <= kFrontendLimit);
};

#if defined(__DAV_VEC__)
// Match the 910B mainline reduction for 128 features. Add the two
// 64-feature halves lane-wise before the hardware row sum.
template <typename Out, typename In, typename Scratch>
__tf__ PTO_INTERNAL void sum_fp32_features_impl(
    typename Out::TileDType __out__ out_data,
    typename In::TileDType __in__ in_data,
    typename Scratch::TileDType __out__ scratch_data) {
  constexpr int kHalfFeatures = 64;
  static_assert(In::Cols == 2 * kHalfFeatures && In::ValidCol == In::Cols);
  static_assert(Scratch::Cols == kHalfFeatures && Out::Cols == 1);
  auto* out = (__ubuf__ float*)__cce_get_tile_ptr(out_data);
  auto* input = (__ubuf__ float*)__cce_get_tile_ptr(in_data);
  auto* scratch = (__ubuf__ float*)__cce_get_tile_ptr(scratch_data);
  constexpr uint8_t rows = In::ValidRow;
  constexpr uint8_t src_stride = In::RowStride / 8;
  constexpr uint8_t tmp_stride = Scratch::RowStride / 8;

  set_mask_norm();
  set_vector_mask(-1, -1);
  vadd(scratch, input, input + kHalfFeatures,
       rows, 1, 1, 1, tmp_stride, src_stride, src_stride);
  pipe_barrier(PIPE_V);
  vcadd(out, scratch, rows, 1, 1, tmp_stride, false);
}

#endif

template <typename Out, typename In, typename Scratch>
AICORE inline void sum_fp32_features(Out& output, In& input, Scratch& scratch) {
#if defined(__DAV_VEC__)
  sum_fp32_features_impl<Out, In, Scratch>(output.data(), input.data(), scratch.data());
#endif
}

}  // namespace kda_prefill
