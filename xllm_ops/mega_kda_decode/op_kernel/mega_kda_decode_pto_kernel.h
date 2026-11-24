/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once

#include <pto/pto-inst.hpp>

#include "kernel_operator.h"
#include "mega_kda_decode_tiling_data.h"

namespace mega_kda {
using namespace pto;

template <typename T, int R, int C, int VR = R, int VC = C>
using TileND = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, VR, VC>;
template <int R, int VR = R>
using Column = Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, VR, 1>;

AICORE inline void v_barrier() {
#if __CCE_AICORE__ == 220
  pipe_barrier(PIPE_V);
#endif
}

template <typename T, int R, int C, typename TileT>
AICORE inline void load(TileT& tile, __gm__ T* ptr) {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  GlobalTensor<T, Shape<1, 1, 1, R, C>, pto::Stride<R * C, R * C, R * C, C, 1>> gm(
      ptr);
  TLOAD(tile, gm);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}

template <typename T, int R, int C, typename TileT>
AICORE inline void store(__gm__ T* ptr, TileT& tile) {
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  GlobalTensor<T, Shape<1, 1, 1, R, C>, pto::Stride<R * C, R * C, R * C, C, 1>> gm(
      ptr);
  TSTORE(gm, tile);
  // A subsequent token/task may reuse every UB address below.
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

struct Args {
  __gm__ bfloat16_t* qkv;
  __gm__ bfloat16_t* gate;
  __gm__ bfloat16_t* beta;
  __gm__ float* a_log;
  __gm__ float* gate_bias;
  __gm__ bfloat16_t* conv_weight;
  __gm__ bfloat16_t* conv_bias;
  __gm__ bfloat16_t* conv_in;
  __gm__ float* ssm_in;
  __gm__ const int32_t* offsets;
  __gm__ const int32_t* conv_read;
  __gm__ const int32_t* conv_write;
  __gm__ const int32_t* ssm_read;
  __gm__ const int32_t* ssm_write;
  __gm__ const int32_t* accepted;
  __gm__ bfloat16_t* output;
  __gm__ bfloat16_t* conv_out;
  __gm__ float* ssm_out;
};

#if __CCE_AICORE__ == 310
#include "mega_kda_decode_a5_dynamic.h"
#else
// FP32 rounding of 1 / sqrt(kHeadDim).
constexpr float kQueryScale = 0.08838834764831845f;
static_assert(kHeadDim == 128, "Update the query scale for the head dimension");
constexpr int64_t kSsmHeadStride = kHeadDim * kHeadDim;

// Byte offsets shared by scalar Conv, batched Conv and split reductions.
// value aliases a subrow of v; column tiles alias their row-vector storage.
template <int VRows>
struct ScratchLayout {
  static constexpr int32_t kVectorBytes = kHeadDim * sizeof(float);
  static constexpr int32_t kNarrowVectorBytes = kHeadDim * sizeof(uint16_t);
  static constexpr int32_t kStateBytes = VRows * kVectorBytes;
  static constexpr int32_t kReduceColumns = 64;
  static constexpr int32_t kState = 0;
  static constexpr int32_t kExpanded = kStateBytes;
  static constexpr int32_t kProduct = 2 * kStateBytes;
  static constexpr int32_t kReduceWork = 3 * kStateBytes;
  static constexpr int32_t kVectors =
      kReduceWork + VRows * kReduceColumns * sizeof(float);
  static constexpr int32_t kQ = kVectors;
  static constexpr int32_t kK = kQ + kVectorBytes;
  static constexpr int32_t kV = kK + kVectorBytes;
  static constexpr int32_t kG = kV + kVectorBytes;
  static constexpr int32_t kTmp = kG + kVectorBytes;
  static constexpr int32_t kX = kTmp + kVectorBytes;
  static constexpr int32_t kWeight = kX + kVectorBytes;
  static constexpr int32_t kAcc = kWeight + kVectorBytes;
  static constexpr int32_t kBf16 = kAcc + kVectorBytes;
  static constexpr int32_t kFp16 = kBf16 + kNarrowVectorBytes;
  static constexpr int32_t kReduced = kFp16 + kNarrowVectorBytes;
  static constexpr int32_t kDelta = kReduced + VRows * sizeof(float);
  static constexpr int32_t kScalarBytes = 8 * sizeof(float);
  static constexpr int32_t kNorm = kDelta + VRows * sizeof(float);
  static constexpr int32_t kBetaScalar = kNorm + kScalarBytes;
  static constexpr int32_t kBetaBf16 = kBetaScalar + kScalarBytes;
  static constexpr int32_t kOut = kBetaBf16 + kScalarBytes;
  static constexpr int32_t kEnd = kOut + VRows * sizeof(uint16_t);
};
static_assert(ScratchLayout<32>::kVectors == 57344 &&
                  ScratchLayout<32>::kEnd == 62368,
              "32-row decode scratch layout changed");
static_assert(ScratchLayout<64>::kVectors == 114688 &&
                  ScratchLayout<64>::kEnd == 120032,
              "64-row decode scratch layout changed");

// One task owns VRows contiguous V rows. Q/K/conv are private to each task;
// only V-shard zero publishes conv state, so no cross-core rendezvous is
// needed.
template <int VRows>
class Scratch final {
 public:
  static constexpr int kVRows = VRows;
  TileND<float, kVRows, 128> state;
  TileND<float, kVRows, 128> expanded;
  TileND<float, kVRows, 128> product;
  TileND<float, kVRows, 64> reduce_work;
  TileND<float, 1, 128> q;
  TileND<float, 1, 128> k;
  TileND<float, 1, 128> v;
  TileND<float, 1, 128> g;
  TileND<float, 1, 128> tmp;
  TileND<float, 1, 128> x;
  TileND<float, 1, 128> weight;
  TileND<float, 1, 128> acc;
  TileND<bfloat16_t, 1, 128> bf16;
  TileND<half, 1, 128> fp16;
  TileND<float, 1, kVRows> reduced;
  TileND<float, 1, kVRows> delta;
  TileND<float, 1, kVRows> value;
  Column<kVRows> reduced_col;
  Column<kVRows> delta_col;
  Column<8, 1> norm_col;
  TileND<float, 1, 8, 1, 1> norm;
  TileND<float, 1, 8, 1, 1> beta_scalar;
  TileND<bfloat16_t, 1, 16, 1, 1> beta_bf16;
  TileND<bfloat16_t, 1, kVRows> out;

  AICORE inline void init(int64_t v_start) {
    using Layout = ScratchLayout<VRows>;
    TASSIGN(state, Layout::kState);
    TASSIGN(expanded, Layout::kExpanded);
    TASSIGN(product, Layout::kProduct);
    TASSIGN(reduce_work, Layout::kReduceWork);
    TASSIGN(q, Layout::kQ);
    TASSIGN(k, Layout::kK);
    TASSIGN(v, Layout::kV);
    TASSIGN(value, Layout::kV + v_start * sizeof(float));
    TASSIGN(g, Layout::kG);
    TASSIGN(tmp, Layout::kTmp);
    TASSIGN(x, Layout::kX);
    TASSIGN(weight, Layout::kWeight);
    TASSIGN(acc, Layout::kAcc);
    TASSIGN(bf16, Layout::kBf16);
    TASSIGN(fp16, Layout::kFp16);
    TASSIGN(reduced, Layout::kReduced);
    TASSIGN(reduced_col, Layout::kReduced);
    TASSIGN(delta, Layout::kDelta);
    TASSIGN(delta_col, Layout::kDelta);
    TASSIGN(norm, Layout::kNorm);
    TASSIGN(norm_col, Layout::kNorm);
    TASSIGN(beta_scalar, Layout::kBetaScalar);
    TASSIGN(beta_bf16, Layout::kBetaBf16);
    TASSIGN(out, Layout::kOut);
  }
};

template <int VRows>
AICORE inline void load_bf16(Scratch<VRows>& s,
                             TileND<float, 1, 128>& dst,
                             __gm__ bfloat16_t* ptr) {
  load<bfloat16_t, 1, 128>(s.bf16, ptr);
  TCVT(dst, s.bf16, RoundMode::CAST_NONE);
  v_barrier();
}

template <int VRows, bool PublicCompat = false>
AICORE inline void conv(Scratch<VRows>& s,
                        const Args& a,
                        const TilingData& t,
                        int64_t row,
                        int64_t token,
                        int64_t begin,
                        int64_t history_offset,
                        int64_t channel,
                        TileND<float, 1, 128>& dst) {
  const int64_t channels = 384 * t.heads;
  if (a.conv_bias) {
    load_bf16(s, s.acc, a.conv_bias + channel);
  } else {
    TEXPANDS(s.acc, 0.0f);
    v_barrier();
  }
  for (int64_t tap = 0; tap < 4; ++tap) {
    const int64_t relative = token - begin + tap - 3;
    __gm__ bfloat16_t* source =
        relative < 0
            ? a.conv_in +
                  (row * t.conv_history + history_offset + relative + 3) *
                      channels +
                  channel
            : a.qkv + (begin + relative) * channels + channel;
    load_bf16(s, s.x, source);
    // vLLM-Ascend conv loads activation/history through FP16, weights in FP32.
    if constexpr (!PublicCompat) {
      TCVT(s.fp16, s.x, RoundMode::CAST_RINT);
      v_barrier();
      TCVT(s.x, s.fp16, RoundMode::CAST_NONE);
    }
    load_bf16(s, s.weight, a.conv_weight + tap * channels + channel);
    TMUL(s.tmp, s.x, s.weight);
    v_barrier();
    TADD(s.acc, s.acc, s.tmp);
    v_barrier();
  }
  // match the public conv BF16 rounding before SiLU.
  TCVT(s.bf16, s.acc, RoundMode::CAST_RINT);
  v_barrier();
  TCVT(s.acc, s.bf16, RoundMode::CAST_NONE);
  v_barrier();
  TMULS(s.tmp, s.acc, -1.0f);
  v_barrier();
  TEXP(s.tmp, s.tmp);
  v_barrier();
  TADDS(s.tmp, s.tmp, 1.0f);
  v_barrier();
  TDIV(s.acc, s.acc, s.tmp);
  v_barrier();
  // Preserve the public conv -> recurrence BF16 boundary inside the fusion.
  TCVT(s.bf16, s.acc, RoundMode::CAST_RINT);
  v_barrier();
  TCVT(dst, s.bf16, RoundMode::CAST_NONE);
  v_barrier();
}

// Batch Q/K/V along tile rows while keeping each V-state shard independent.
template <int VRows>
class QkvScratch final {
 public:
  TileND<float, 3, 128> output;
  TileND<float, 3, 128> x;
  TileND<float, 3, 128> weight;
  TileND<float, 3, 128> acc;
  TileND<float, 3, 128> tmp;
  TileND<bfloat16_t, 3, 128> bf16;
  TileND<half, 3, 128> fp16;

  AICORE inline void init() {
    using Layout = ScratchLayout<VRows>;
    constexpr int32_t kQkvVectorBytes = kQkvParts * Layout::kVectorBytes;
    constexpr int32_t kQkvNarrowBytes = kQkvParts * Layout::kNarrowVectorBytes;
    constexpr int32_t kBase = Layout::kEnd;
    // Q, K and V in Scratch are contiguous and alias the three output rows.
    TASSIGN(output, Layout::kVectors);
    TASSIGN(x, kBase);
    TASSIGN(weight, kBase + kQkvVectorBytes);
    TASSIGN(acc, kBase + 2 * kQkvVectorBytes);
    TASSIGN(tmp, kBase + 3 * kQkvVectorBytes);
    TASSIGN(bf16, kBase + 4 * kQkvVectorBytes);
    TASSIGN(fp16, kBase + 4 * kQkvVectorBytes + kQkvNarrowBytes);
  }
};

template <int VRows>
AICORE inline void load_qkv_bf16(QkvScratch<VRows>& s,
                                 TileND<float, 3, 128>& dst,
                                 __gm__ bfloat16_t* ptr,
                                 int64_t part_stride) {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  using QkvStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<bfloat16_t, Shape<1, 1, 1, 3, 128>, QkvStride> gm(
      ptr, {}, QkvStride(part_stride));
  TLOAD(s.bf16, gm);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(dst, s.bf16, RoundMode::CAST_NONE);
  v_barrier();
}

template <int VRows, bool PublicCompat = false>
AICORE inline void conv_qkv(const Args& a,
                            const TilingData& t,
                            int64_t row,
                            int64_t token,
                            int64_t begin,
                            int64_t history_offset,
                            int64_t head) {
  QkvScratch<VRows> s;
  s.init();
  const int64_t part_stride = t.heads * 128;
  const int64_t channels = 3 * part_stride;
  const int64_t channel = head * 128;
  if (a.conv_bias) {
    load_qkv_bf16(s, s.acc, a.conv_bias + channel, part_stride);
  } else {
    TEXPANDS(s.acc, 0.0f);
    v_barrier();
  }
  for (int64_t tap = 0; tap < 4; ++tap) {
    const int64_t relative = token - begin + tap - 3;
    __gm__ bfloat16_t* input =
        relative < 0
            ? a.conv_in +
                  (row * t.conv_history + history_offset + relative + 3) *
                      channels +
                  channel
            : a.qkv + (begin + relative) * channels + channel;
    load_qkv_bf16(s, s.x, input, part_stride);
    if constexpr (!PublicCompat) {
      TCVT(s.fp16, s.x, RoundMode::CAST_RINT);
      v_barrier();
      TCVT(s.x, s.fp16, RoundMode::CAST_NONE);
    }
    load_qkv_bf16(
        s, s.weight, a.conv_weight + tap * channels + channel, part_stride);
    TMUL(s.tmp, s.x, s.weight);
    v_barrier();
    TADD(s.acc, s.acc, s.tmp);
    v_barrier();
  }
  // match the public conv BF16 rounding before SiLU.
  TCVT(s.bf16, s.acc, RoundMode::CAST_RINT);
  v_barrier();
  TCVT(s.acc, s.bf16, RoundMode::CAST_NONE);
  v_barrier();
  TMULS(s.tmp, s.acc, -1.0f);
  v_barrier();
  TEXP(s.tmp, s.tmp);
  v_barrier();
  TADDS(s.tmp, s.tmp, 1.0f);
  v_barrier();
  TDIV(s.acc, s.acc, s.tmp);
  v_barrier();
  TCVT(s.bf16, s.acc, RoundMode::CAST_RINT);
  v_barrier();
  TCVT(s.output, s.bf16, RoundMode::CAST_NONE);
  v_barrier();
}

template <int Rows, typename TileT>
AICORE inline void load_conv_rows(TileT& tile,
                                  __gm__ bfloat16_t* ptr,
                                  int64_t part_stride) {
  set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
  using ConvStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<bfloat16_t, Shape<1, 1, 1, Rows, 128>, ConvStride> gm(
      ptr, {}, ConvStride(part_stride));
  TLOAD(tile, gm);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}

// A one-token decode window has three history taps and one current QKV tap.
// Batch their loads, then accumulate taps in the original FP32 order.
template <int VRows, bool PublicCompat = false>
AICORE inline void conv_qkv_plain(const Args& a,
                                  const TilingData& t,
                                  int64_t row,
                                  int64_t token,
                                  int64_t head) {
  using Layout = ScratchLayout<VRows>;
  constexpr int32_t kBase = Layout::kEnd;
  constexpr int32_t kTapVectorBytes = kQkvParts * Layout::kVectorBytes;
  constexpr int32_t kWindowVectorBytes = kConvTaps * kTapVectorBytes;
  constexpr int32_t kInputs = kBase + kWindowVectorBytes;
  constexpr int32_t kAcc = kInputs + kWindowVectorBytes;
  constexpr int32_t kTmp = kAcc + kTapVectorBytes;
  constexpr int32_t kBf16 = kTmp + kTapVectorBytes;
  constexpr int32_t kHistoryBytes =
      kConvHistory * kQkvParts * Layout::kNarrowVectorBytes;
  constexpr int32_t kFp16 =
      kBf16 + kConvTaps * kQkvParts * Layout::kNarrowVectorBytes;
  TileND<float, 12, 128> weights;
  TileND<float, 12, 128> inputs;
  TileND<float, 3, 128> acc;
  TileND<float, 3, 128> tmp;
  TileND<float, 3, 128> input_tap;
  TileND<float, 3, 128> weight_tap;
  TileND<float, 3, 128> output;
  TileND<bfloat16_t, 12, 128> bf16;
  TileND<bfloat16_t, 9, 128> history;
  TileND<bfloat16_t, 3, 128> current;
  TileND<bfloat16_t, 3, 128> bias;
  TileND<half, 12, 128> fp16;
  TASSIGN(weights, kBase);
  TASSIGN(inputs, kInputs);
  TASSIGN(acc, kAcc);
  TASSIGN(tmp, kTmp);
  TASSIGN(bf16, kBf16);
  TASSIGN(history, kBf16);
  TASSIGN(current, kBf16 + kHistoryBytes);
  TASSIGN(bias, kBf16);
  TASSIGN(fp16, kFp16);
  TASSIGN(output, Layout::kVectors);
  const int64_t part_stride = t.heads * 128;
  const int64_t channels = 3 * part_stride;
  const int64_t channel = head * 128;
  load_conv_rows<12>(bf16, a.conv_weight + channel, part_stride);
  TCVT(weights, bf16, RoundMode::CAST_NONE);
  v_barrier();
  load_conv_rows<9>(
      history, a.conv_in + row * 3 * channels + channel, part_stride);
  load_conv_rows<3>(current, a.qkv + token * channels + channel, part_stride);
  TCVT(inputs, bf16, RoundMode::CAST_NONE);
  v_barrier();
  if constexpr (!PublicCompat) {
    TCVT(fp16, inputs, RoundMode::CAST_RINT);
    v_barrier();
    TCVT(inputs, fp16, RoundMode::CAST_NONE);
    v_barrier();
  }
  if (a.conv_bias) {
    load_conv_rows<3>(bias, a.conv_bias + channel, part_stride);
    TCVT(acc, bias, RoundMode::CAST_NONE);
  } else {
    TEXPANDS(acc, 0.0f);
  }
  v_barrier();
  for (int32_t tap = 0; tap < 4; ++tap) {
    TASSIGN(input_tap, kInputs + tap * kTapVectorBytes);
    TASSIGN(weight_tap, kBase + tap * kTapVectorBytes);
    TMUL(tmp, input_tap, weight_tap);
    v_barrier();
    TADD(acc, acc, tmp);
    v_barrier();
  }
  // The raw window is already in inputs; reuse the 3-row BF16 tile.
  TCVT(current, acc, RoundMode::CAST_RINT);
  v_barrier();
  TCVT(acc, current, RoundMode::CAST_NONE);
  v_barrier();
  TMULS(tmp, acc, -1.0f);
  v_barrier();
  TEXP(tmp, tmp);
  v_barrier();
  TADDS(tmp, tmp, 1.0f);
  v_barrier();
  TDIV(acc, acc, tmp);
  v_barrier();
  TCVT(current, acc, RoundMode::CAST_RINT);
  v_barrier();
  TCVT(output, current, RoundMode::CAST_NONE);
  v_barrier();
}

// The designated conv owner publishes two history taps and the current QKV.
// SSM/output stores have drained, so their UB storage can hold the BF16 window.
AICORE inline void store_plain_conv(const Args& a,
                                    const TilingData& t,
                                    int64_t read_slot,
                                    int64_t write_slot,
                                    int64_t token,
                                    int64_t head) {
  TileND<bfloat16_t, 9, 128> window;
  TileND<bfloat16_t, 6, 128> history;
  TileND<bfloat16_t, 3, 128> current;
  TASSIGN(window, 0);
  TASSIGN(history, 0);
  TASSIGN(current, 2 * kQkvParts * kHeadDim * sizeof(bfloat16_t));
  const int64_t part_stride = t.heads * 128;
  const int64_t channels = 3 * part_stride;
  const int64_t channel = head * 128;
  load_conv_rows<6>(history,
                    a.conv_in + (read_slot * 3 + 1) * channels + channel,
                    part_stride);
  load_conv_rows<3>(current, a.qkv + token * channels + channel, part_stride);
  using ConvStride = Stride<1, 1, 1, DYNAMIC, 1>;
  GlobalTensor<bfloat16_t, Shape<1, 1, 1, 9, 128>, ConvStride> gm(
      a.conv_out + write_slot * 3 * channels + channel,
      {},
      ConvStride(part_stride));
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(gm, window);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}

template <int VRows>
AICORE inline void normalize(Scratch<VRows>& s,
                             TileND<float, 1, 128>& x,
                             float scale) {
  TMUL(s.tmp, x, x);
  v_barrier();
  TROWSUM(s.norm_col, s.tmp, s.reduce_work);
  v_barrier();
  TSQRT(s.norm, s.norm);
  v_barrier();
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  // Mainline recurrent KDA preserves zero Q/K rows.
  const float norm = s.norm.GetValue(0);
  if (norm > 0.0f) {
    TMULS(x, x, 1.0f / norm);
    v_barrier();
  }
  TMULS(x, x, scale);
  v_barrier();
}

// Ordinary decode uses the same FP32 reduction order for both tile sizes.
// Keep the 64-row state/conv tile when it fills the device.
template <int VRows, bool Mtp>
AICORE inline void recurrent_row_sum(Scratch<VRows>& s) {
  if constexpr (!Mtp && VRows == 64) {
    using Layout = ScratchLayout<VRows>;
    TileND<float, 32, 128> product_half;
    TileND<float, 32, 64> work_half;
    Column<32> reduced_half;
    TASSIGN(work_half, Layout::kReduceWork);
    TASSIGN(product_half, Layout::kProduct);
    TASSIGN(reduced_half, Layout::kReduced);
    TROWSUM(reduced_half, product_half, work_half);
    v_barrier();
    TASSIGN(product_half, Layout::kProduct + 32 * Layout::kVectorBytes);
    TASSIGN(reduced_half, Layout::kReduced + 32 * sizeof(float));
    TROWSUM(reduced_half, product_half, work_half);
    v_barrier();
  } else {
    TROWSUM(s.reduced_col, s.product, s.reduce_work);
  }
}

template <int VRows, bool Mtp, bool PublicCompat>
AICORE inline void recurrent(Scratch<VRows>& s,
                             const Args& a,
                             const TilingData& t,
                             int64_t token,
                             int64_t head) {
  normalize(s, s.q, kQueryScale);
  normalize(s, s.k, 1.0f);
  load_bf16(s, s.g, a.gate + (token * t.heads + head) * 128);
  load<float, 1, 128>(s.tmp, a.gate_bias + head * 128);
  TADD(s.g, s.g, s.tmp);
  TEXPANDS(s.tmp, a.a_log[head]);
  v_barrier();
  TEXP(s.tmp, s.tmp);
  v_barrier();
  TMUL(s.g, s.g, s.tmp);
  v_barrier();
  TMULS(s.g, s.g, -1.0f);
  v_barrier();
  TEXP(s.g, s.g);
  v_barrier();
  TADDS(s.g, s.g, 1.0f);
  TEXPANDS(s.tmp, PublicCompat ? 1.0f : -5.0f);
  v_barrier();
  TDIV(s.g, s.tmp, s.g);
  v_barrier();
  if constexpr (PublicCompat) {
    TMULS(s.g, s.g, -5.0f);
    v_barrier();
  }
  TEXP(s.g, s.g);
  load<bfloat16_t, 1, 1>(s.beta_bf16, a.beta + token * t.heads + head);
  TCVT(s.beta_scalar, s.beta_bf16, RoundMode::CAST_NONE);
  v_barrier();
  TMULS(s.beta_scalar, s.beta_scalar, -1.0f);
  v_barrier();
  TEXP(s.beta_scalar, s.beta_scalar);
  v_barrier();
  TADDS(s.beta_scalar, s.beta_scalar, 1.0f);
  TEXPANDS(s.norm, 1.0f);
  v_barrier();
  TDIV(s.beta_scalar, s.norm, s.beta_scalar);
  v_barrier();
  // Keep sigmoid(beta_raw.float()) in FP32 for the recurrence.
  TCOLEXPAND(s.expanded, s.g);
  v_barrier();
  TMUL(s.state, s.state, s.expanded);
  v_barrier();
  TCOLEXPAND(s.expanded, s.k);
  v_barrier();
  TMUL(s.product, s.state, s.expanded);
  v_barrier();
  recurrent_row_sum<VRows, Mtp>(s);
  v_barrier();
  TSUB(s.delta, s.value, s.reduced);
  v_barrier();
  set_flag(PIPE_V, PIPE_S, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
  const float beta = s.beta_scalar.GetValue(0);
  TMULS(s.delta, s.delta, beta);
  v_barrier();
  TROWEXPAND(s.product, s.delta_col);
  v_barrier();
  TMUL(s.product, s.product, s.expanded);
  v_barrier();
  TADD(s.state, s.state, s.product);
  TCOLEXPAND(s.expanded, s.q);
  v_barrier();
  TMUL(s.product, s.state, s.expanded);
  v_barrier();
  recurrent_row_sum<VRows, Mtp>(s);
  v_barrier();
  TCVT(s.out, s.reduced, RoundMode::CAST_RINT);
  v_barrier();
}

template <bool Mtp,
          int VRows,
          bool BatchConv = false,
          bool PublicCompat = false>
AICORE inline void run_tiled(const Args& a, const TilingData& t) {
#if defined(__DAV_C220_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  constexpr int kVRows = VRows;
  constexpr int kVShards = 128 / kVRows;
  const int64_t channels = 384 * t.heads;
  for (int64_t task = get_block_idx(); task < t.batch * t.heads * kVShards;
       task += get_block_num()) {
    const int64_t batch = task / (t.heads * kVShards);
    const int64_t head = task / kVShards % t.heads;
    const int64_t v_start = task % kVShards * kVRows;
    const int64_t begin = a.offsets[batch];
    const int64_t end = a.offsets[batch + 1];
    const int64_t count = end - begin;
    const int64_t accepted = Mtp ? a.accepted[batch] : 1;
    if (begin < 0 || end > t.tokens || count <= 0 ||
        count > t.max_query_tokens ||
        (!Mtp && (begin != batch || count != 1)) || accepted < 1 ||
        accepted > t.max_query_tokens) {
      continue;
    }
    const int64_t conv_read = a.conv_read[batch];
    const int64_t conv_write = a.conv_write[batch];
    const int64_t ssm_read =
        a.ssm_read[batch * t.max_query_tokens + accepted - 1];
    if (conv_read < 0 || conv_read >= t.conv_input_slots || ssm_read < 0 ||
        ssm_read >= t.ssm_input_slots || conv_write < -1 ||
        conv_write >= t.conv_output_slots) {
      continue;
    }
    bool valid = true;
    for (int64_t i = 0; i < count; ++i) {
      const int64_t slot = a.ssm_write[batch * t.max_query_tokens + i];
      valid = valid && slot >= -1 && slot < t.ssm_output_slots;
    }
    if (!valid) {
      continue;
    }
    Scratch<VRows> s;
    s.init(v_start);
    load<float, kVRows, kHeadDim>(
        s.state,
        a.ssm_in + (ssm_read * t.heads + head) * kSsmHeadStride +
            v_start * kHeadDim);
    for (int64_t token = begin; token < end; ++token) {
      if constexpr (BatchConv) {
        if constexpr (Mtp) {
          conv_qkv<VRows, PublicCompat>(
              a, t, conv_read, token, begin, accepted - 1, head);
        } else {
          conv_qkv_plain<VRows, PublicCompat>(a, t, conv_read, token, head);
        }
      } else {
        conv<VRows, PublicCompat>(
            s, a, t, conv_read, token, begin, accepted - 1, head * 128, s.q);
        conv<VRows, PublicCompat>(s,
                                  a,
                                  t,
                                  conv_read,
                                  token,
                                  begin,
                                  accepted - 1,
                                  (t.heads + head) * 128,
                                  s.k);
        conv<VRows, PublicCompat>(s,
                                  a,
                                  t,
                                  conv_read,
                                  token,
                                  begin,
                                  accepted - 1,
                                  (2 * t.heads + head) * 128,
                                  s.v);
      }
      recurrent<VRows, Mtp, PublicCompat>(s, a, t, token, head);
      const int64_t slot =
          a.ssm_write[batch * t.max_query_tokens + token - begin];
      if (slot >= 0) {
        store<float, kVRows, kHeadDim>(
            a.ssm_out + (slot * t.heads + head) * kSsmHeadStride +
                v_start * kHeadDim,
            s.state);
      }
      store<bfloat16_t, 1, kVRows>(
          a.output + (token * t.heads + head) * 128 + v_start, s.out);
    }
    if (v_start == 0 && conv_write >= 0) {
      if constexpr (BatchConv && !Mtp) {
        store_plain_conv(a, t, conv_read, conv_write, begin, head);
      } else {
        // Width four: keep two accepted history tokens, then append raw QKV.
        // Ragged MTP writes count+2 positions and preserves the capacity tail.
        for (int64_t part = 0; part < 3; ++part) {
          const int64_t channel = (part * t.heads + head) * 128;
          for (int64_t i = 0; i < count + 2; ++i) {
            __gm__ bfloat16_t* src =
                i < 2 ? a.conv_in +
                            (conv_read * t.conv_history + accepted + i) *
                                channels +
                            channel
                      : a.qkv + (begin + i - 2) * channels + channel;
            load<bfloat16_t, 1, 128>(s.bf16, src);
            store<bfloat16_t, 1, 128>(
                a.conv_out + (conv_write * t.conv_history + i) * channels +
                    channel,
                s.bf16);
          }
        }
      }
    }
  }
  pipe_barrier(PIPE_ALL);
#endif
}

template <bool Mtp, bool PublicCompat = false>
AICORE inline void run(const Args& a, const TilingData& t) {
#if defined(__DAV_C220_VEC__)
  // Keep all launched AIV cores busy before widening the V tile. The wider
  // tile halves duplicated Q/K/conv/gate work without changing state owners.
  if (t.heads == 8 && t.batch <= 4) {
    if (t.batch * t.heads * (128 / 64) >= get_block_num()) {
      run_tiled<Mtp, 64, true, PublicCompat>(a, t);
    } else {
      run_tiled<Mtp, 32, true, PublicCompat>(a, t);
    }
    return;
  }
  if (t.batch * t.heads * (128 / 64) >= get_block_num()) {
    run_tiled<Mtp, 64, false, PublicCompat>(a, t);
  } else {
    run_tiled<Mtp, 32, false, PublicCompat>(a, t);
  }
#endif
}
#endif
}  // namespace mega_kda
