/* Copyright 2026 The xLLM Authors. All Rights Reserved. */
#pragma once
#include <pto/pto-inst.hpp>

#include "kda_kernel_utils.h"
#include "kda_prefill_fp32_sum.h"
#include "kernel_operator.h"
#include "mega_kda_prefill_tiling_data.h"

namespace kda_prefill {
using namespace pto;
using namespace mega_kda_prefill_utils;
template <typename T, int R, int C, int VR = R, int VC = C>
using ND = Tile<TileType::Vec, T, R, C, BLayout::RowMajor, VR, VC>;
using GS = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using ST = pto::Stride<1, 1, 1, DYNAMIC, 1>;
template <typename T>
using GM = pto::GlobalTensor<T, GS, ST>;

template <typename T, typename TileT>
AICORE inline void load(TileT& tile,
                        __gm__ T* ptr,
                        int rows,
                        int cols,
                        int stride) {
  SetWaitFlag<PIPE_V, PIPE_MTE2>(0);
  GM<T> gm(ptr, GS(1, 1, 1, rows, cols), ST(stride));
  TLOAD(tile, gm);
  SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
}
template <typename T, typename TileT>
AICORE inline void store(__gm__ T* ptr,
                         TileT& tile,
                         int rows,
                         int cols,
                         int stride) {
  SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
  GM<T> gm(ptr, GS(1, 1, 1, rows, cols), ST(stride));
  TSTORE(gm, tile);
  SetWaitFlag<PIPE_MTE3, PIPE_V>(0);
}
struct Args {
  __gm__ bfloat16_t* qkv;
  __gm__ bfloat16_t* gate;
  __gm__ float* beta;
  __gm__ float* a_log;
  __gm__ float* gate_bias;
  __gm__ bfloat16_t* weight;
  __gm__ bfloat16_t* bias;
  __gm__ bfloat16_t* conv_in;
  __gm__ float* ssm_in;
  __gm__ const int32_t* cu;
  __gm__ const int32_t* conv_read;
  __gm__ const int32_t* conv_write;
  __gm__ const int32_t* ssm_read;
  __gm__ const int32_t* ssm_write;
  __gm__ bfloat16_t* output;
  __gm__ bfloat16_t* conv_out;
  __gm__ float* ssm_out;
};

AICORE inline bool valid_metadata(const Args& a, const TilingData& t) {
  if (a.cu[0] != 0 || a.cu[t.batch] != t.tokens) {
    return false;
  }
  for (int64_t b = 0; b < t.batch; ++b) {
    if (a.cu[b] < 0 || a.cu[b] > a.cu[b + 1] || a.cu[b + 1] > t.tokens ||
        a.conv_read[b] < -1 || a.conv_read[b] >= t.conv_input_slots ||
        a.ssm_read[b] < -1 || a.ssm_read[b] >= t.ssm_input_slots ||
        a.conv_write[b] < -1 || a.conv_write[b] >= t.conv_output_slots ||
        a.ssm_write[b] < -1 || a.ssm_write[b] >= t.ssm_output_slots) {
      return false;
    }
    if (a.cu[b] == a.cu[b + 1]) {
      continue;
    }
    for (int64_t j = 0; j < b; ++j) {
      if (a.cu[j] == a.cu[j + 1]) {
        continue;
      }
      if ((a.conv_write[b] >= 0 && a.conv_write[b] == a.conv_write[j]) ||
          (a.ssm_write[b] >= 0 && a.ssm_write[b] == a.ssm_write[j])) {
        return false;
      }
    }
  }
  return true;
}

// Dedicated flags, disjoint from the arithmetic stages' local 0-3 and 10-13
// flags.
constexpr uint16_t kAivReady = 5;
constexpr uint16_t kAllAicArrived = 6;
constexpr uint16_t kAivRelease = 7;

AICORE inline void rendezvous() {
  pipe_barrier(PIPE_ALL);
#if defined(__DAV_CUBE__)
  wait_flag_dev(kAivReady);
  ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0, kAllAicArrived));
  wait_flag_dev(kAllAicArrived);
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(2, kAivRelease));
#elif defined(__DAV_VEC__)
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(2, kAivReady));
  wait_flag_dev(kAivRelease);
#endif
}

class FrontScratch final {
 public:
  ND<float, 1, 128> acc;
  ND<float, 1, 128> x;
  ND<float, 1, 128> weight;
  ND<float, 1, 128> tmp;
  ND<float, 1, 128> norm_broadcast;
  ND<float, 1, 128> gate;
  ND<float, 1, 128> cumulative;
  ND<float, 1, 128> positive_exp_a;
  ND<float, 1, 128> cached_gate_bias;
  ND<bfloat16_t, 1, 128> bf;
  ND<half, 1, 128> fp16;
  ND<float, 1, 64> reduce;
  ND<float, 1, 8, 1, 1> scalar;
  Tile<TileType::Vec, float, 8, 1, BLayout::ColMajor, 1, 1> column;
  AICORE inline void init() {
    TASSIGN(acc, 0);
    TASSIGN(x, 512);
    TASSIGN(weight, 1024);
    TASSIGN(tmp, 1536);
    TASSIGN(norm_broadcast, 2048);
    TASSIGN(gate, 2560);
    TASSIGN(cumulative, 3072);
    TASSIGN(bf, 3584);
    TASSIGN(fp16, 3840);
    TASSIGN(reduce, 4096);
    TASSIGN(scalar, 4352);
    TASSIGN(column, 4352);
    TASSIGN(positive_exp_a, 4608);
    TASSIGN(cached_gate_bias, 5120);
  }
  AICORE inline void bf_load(ND<float, 1, 128>& dst, __gm__ bfloat16_t* src) {
    load(bf, src, 1, 128, 128);
    TCVT(dst, bf, RoundMode::CAST_NONE);
    PipeBarrierVec();
  }
};

template <bool UseMainConvRounding>
AICORE inline void conv(FrontScratch& s,
                        const Args& a,
                        const TilingData& t,
                        int64_t b,
                        int64_t token,
                        int64_t channel) {
  const int64_t c = t.heads * 384;
  const int64_t read = a.conv_read[b];
  const int64_t begin = a.cu[b];
  if (a.bias) {
    s.bf_load(s.acc, a.bias + channel);
  } else {
    TEXPANDS(s.acc, 0.0f);
    PipeBarrierVec();
  }
  for (int tap = 0; tap < 4; ++tap) {
    const int64_t pos = token + tap - 3;
    if (pos < begin && read < 0) {
      TEXPANDS(s.x, 0.0f);
      PipeBarrierVec();
    } else {
      auto* p = pos < begin
                    ? a.conv_in + (read * 3 + pos - begin + 3) * c + channel
                    : a.qkv + pos * c + channel;
      s.bf_load(s.x, p);
    }
    s.bf_load(s.weight, a.weight + tap * c + channel);
    if constexpr (UseMainConvRounding) {
      if (tap == 0 && !a.bias) {
        TMUL(s.acc, s.x, s.weight);
      } else {
        TMULADDDST(s.acc, s.x, s.weight);
      }
    } else {
      TCVT(s.fp16, s.x, RoundMode::CAST_RINT);
      PipeBarrierVec();
      TCVT(s.x, s.fp16, RoundMode::CAST_NONE);
      PipeBarrierVec();
      TMUL(s.tmp, s.x, s.weight);
      PipeBarrierVec();
      TADD(s.acc, s.acc, s.tmp);
    }
    PipeBarrierVec();
  }
  if constexpr (!UseMainConvRounding) {
    TCVT(s.bf, s.acc, RoundMode::CAST_RINT);
    PipeBarrierVec();
    TCVT(s.acc, s.bf, RoundMode::CAST_NONE);
    PipeBarrierVec();
  }
  TMULS(s.tmp, s.acc, -1.0f);
  PipeBarrierVec();
  TEXP(s.tmp, s.tmp);
  PipeBarrierVec();
  TADDS(s.tmp, s.tmp, 1.0f);
  PipeBarrierVec();
  TDIV(s.acc, s.acc, s.tmp);
  PipeBarrierVec();
  TCVT(s.bf, s.acc, RoundMode::CAST_RINT);
  PipeBarrierVec();
  TCVT(s.acc, s.bf, RoundMode::CAST_NONE);
  PipeBarrierVec();
}

constexpr float kQkNormEps = 1.0e-6f;

AICORE inline void normalize(FrontScratch& s) {
  TMUL(s.tmp, s.acc, s.acc);
  PipeBarrierVec();
  sum_fp32_features(s.column, s.tmp, s.reduce);
  PipeBarrierVec();
  TADDS(s.scalar, s.scalar, kQkNormEps);
  PipeBarrierVec();
  // Ascend910B lowers mainline rsqrt to sqrt followed by reciprocal division.
  TRSQRT(s.scalar, s.scalar, s.reduce);
  PipeBarrierVec();
  TROWEXPAND(s.norm_broadcast, s.column);
  PipeBarrierVec();
  TMUL(s.acc, s.acc, s.norm_broadcast);
  PipeBarrierVec();
  TCVT(s.bf, s.acc, RoundMode::CAST_RINT);
  PipeBarrierVec();
}

// Each row is independent in the conv and normalization front end. Keep the
// original tap order, FP16 input boundary and BF16 conv/normalized boundary.
// Only the first three sequence tokens require history-aware scalar rows.
template <int R, bool UseMainConvRounding>
static AICORE __attribute__((noinline)) void conv_rows(const Args& a,
                                                       const TilingData& t,
                                                       __gm__ uint8_t* ws,
                                                       const Workspace& w,
                                                       int64_t first,
                                                       int64_t end,
                                                       int64_t head,
                                                       int32_t selected_part) {
#if defined(__DAV_VEC__)
  if (first >= end) {
    pipe_barrier(PIPE_ALL);
    return;
  }
  constexpr int D = 128;
  using L = ConvRowsLayout<R>;
  ND<float, R, D> acc;
  ND<float, R, D> tmp;
  ND<float, R, D> norm;
  ND<bfloat16_t, R, D> bf;
  ND<half, R, D> fp16;
  ND<float, 1, D> weight;
  ND<bfloat16_t, 1, D> weight_bf;
  ND<float, R, 64> reduce;
  ND<float, 1, R> scalar;
  Tile<TileType::Vec, float, R, 1, BLayout::ColMajor, R, 1> column;
  TASSIGN(acc, 0);
  TASSIGN(tmp, L::kTemporary);
  TASSIGN(bf, L::kBf);
  TASSIGN(fp16, L::kHalf);
  TASSIGN(weight, L::kWeight);
  TASSIGN(weight_bf, L::kWeightBf);
  TASSIGN(scalar, L::kScalar);
  TASSIGN(column, L::kScalar);
  TASSIGN(reduce, L::kReduce);
  TASSIGN(norm, L::kNorm);
  const int64_t channels = t.heads * 384;
  constexpr int32_t kWindowRows = R + 3;
  ND<float, kWindowRows, D> window;
  ND<bfloat16_t, kWindowRows, D> raw_window;
  ND<half, kWindowRows, D> narrow_window;
  ND<float, 4, D> cached_weights;
  ND<bfloat16_t, 4, D> raw_weights;
  TASSIGN(cached_weights, L::kCachedWeights);
  TASSIGN(window, L::kWindow);
  TASSIGN(raw_window, L::kRawWindow);
  TASSIGN(narrow_window, L::kRawWindow);
  TASSIGN(raw_weights, L::kRawWindow);
  for (int32_t part = 0; part < 3; ++part) {
    if (selected_part >= 0 && part != selected_part) {
      continue;
    }
    const int64_t channel = (part * t.heads + head) * D;
    load(raw_weights, a.weight + channel, 4, D, channels);
    TCVT(cached_weights, raw_weights, RoundMode::CAST_NONE);
    PipeBarrierVec();
    if (a.bias) {
      load(weight_bf, a.bias + channel, 1, D, D);
      TCVT(weight, weight_bf, RoundMode::CAST_NONE);
      PipeBarrierVec();
    }
    for (int64_t token = first; token < end; token += R) {
      const int32_t rows =
          static_cast<int32_t>(end - token < R ? end - token : R);
      if (rows == R) {
        load(raw_window,
             a.qkv + (token - 3) * channels + channel,
             kWindowRows,
             D,
             channels);
        TCVT(window, raw_window, RoundMode::CAST_NONE);
        PipeBarrierVec();
        if constexpr (!UseMainConvRounding) {
          TCVT(narrow_window, window, RoundMode::CAST_RINT);
          PipeBarrierVec();
          TCVT(window, narrow_window, RoundMode::CAST_NONE);
          PipeBarrierVec();
        }
      }
      TEXPANDS(acc, 0.0f);
      PipeBarrierVec();
      if (a.bias) {
        TCOLEXPANDADD(acc, acc, weight);
        PipeBarrierVec();
      }
      for (int32_t tap = 0; tap < 4; ++tap) {
        ND<float, R, D> input;
        ND<float, 1, D> tap_weight;
        if (rows == R) {
          TASSIGN(input, L::kWindow + tap * D * sizeof(float));
        } else {
          // Keep the original partial-R load/cast path and its valid-row store.
          ND<bfloat16_t, R, D, DYNAMIC> raw(rows);
          TASSIGN(raw, L::kBf);
          TASSIGN(input, L::kInput);
          load(raw,
               a.qkv + (token + tap - 3) * channels + channel,
               rows,
               D,
               channels);
          TCVT(input, bf, RoundMode::CAST_NONE);
          PipeBarrierVec();
          if constexpr (!UseMainConvRounding) {
            TCVT(fp16, input, RoundMode::CAST_RINT);
            PipeBarrierVec();
            TCVT(input, fp16, RoundMode::CAST_NONE);
            PipeBarrierVec();
          }
        }
        TASSIGN(tap_weight, L::kCachedWeights + tap * D * sizeof(float));
        if constexpr (UseMainConvRounding) {
          TCOLEXPAND(tmp, tap_weight);
          PipeBarrierVec();
          if (tap == 0 && !a.bias) {
            TMUL(acc, input, tmp);
          } else {
            TMULADDDST(acc, input, tmp);
          }
        } else {
          TCOLEXPANDMUL(tmp, input, tap_weight);
          PipeBarrierVec();
          TADD(acc, acc, tmp);
        }
        PipeBarrierVec();
      }
      if constexpr (!UseMainConvRounding) {
        TCVT(bf, acc, RoundMode::CAST_RINT);
        PipeBarrierVec();
        TCVT(acc, bf, RoundMode::CAST_NONE);
        PipeBarrierVec();
      }
      TMULS(tmp, acc, -1.0f);
      PipeBarrierVec();
      TEXP(tmp, tmp);
      PipeBarrierVec();
      TADDS(tmp, tmp, 1.0f);
      PipeBarrierVec();
      TDIV(acc, acc, tmp);
      PipeBarrierVec();
      TCVT(bf, acc, RoundMode::CAST_RINT);
      PipeBarrierVec();
      TCVT(acc, bf, RoundMode::CAST_NONE);
      PipeBarrierVec();
      if (part < 2) {
        TMUL(tmp, acc, acc);
        PipeBarrierVec();
        sum_fp32_features(column, tmp, reduce);
        PipeBarrierVec();
        TADDS(scalar, scalar, kQkNormEps);
        PipeBarrierVec();
        TRSQRT(scalar, scalar, reduce);
        PipeBarrierVec();
        TROWEXPAND(norm, column);
        PipeBarrierVec();
        TMUL(acc, acc, norm);
        PipeBarrierVec();
        TCVT(bf, acc, RoundMode::CAST_RINT);
        PipeBarrierVec();
        ND<bfloat16_t, R, D, DYNAMIC> output(rows);
        TASSIGN(output, L::kBf);
        store(
            reinterpret_cast<__gm__ bfloat16_t*>(ws + (part == 0 ? w.q : w.k)) +
                (head * t.tokens + token) * D,
            output,
            rows,
            D,
            D);
      } else {
        ND<float, R, D, DYNAMIC> output(rows);
        TASSIGN(output, 0);
        store(reinterpret_cast<__gm__ float*>(ws + w.value) +
                  (token * t.heads + head) * D,
              output,
              rows,
              D,
              t.heads * D);
      }
    }
  }
  // Return the shared UB arena before the gate cumulative scan reuses it.
  pipe_barrier(PIPE_ALL);
#endif
}

// The per-token elementwise chain is independent; only its original FP32
// cumulative addition is serial. Reuse the fully drained ConvRows UB arena.
template <bool UseLog2Gate, typename Data, typename Scratch>
AICORE inline void finish_gate(Data& denominator, Scratch& numerator) {
  TEXPANDS(numerator, UseLog2Gate ? 1.0f : -5.0f);
  PipeBarrierVec();
  TDIV(denominator, numerator, denominator);
  PipeBarrierVec();
  if constexpr (UseLog2Gate) {
    TMULS(denominator, denominator, -5.0f);
    PipeBarrierVec();
    TMULS(denominator, denominator, kGateInvLn2);
    PipeBarrierVec();
  }
}

template <bool UseLog2Gate>
static AICORE __attribute__((noinline)) void gate_rows(const Args& a,
                                                       const TilingData& t,
                                                       __gm__ uint8_t* ws,
                                                       const Workspace& w,
                                                       int64_t start,
                                                       int64_t end,
                                                       int64_t head) {
#if defined(__DAV_VEC__)
  constexpr int32_t kRows = 64;
  constexpr int32_t kColumns = 128;
  constexpr int32_t kFloatBytes = static_cast<int32_t>(sizeof(float));
  constexpr int32_t kGateBytes = kRows * kColumns * kFloatBytes;
  constexpr int32_t kRawOffset = 2 * kGateBytes;
  constexpr int32_t kCumulativeOffset =
      kRawOffset + kRows * kColumns * sizeof(bfloat16_t);
  constexpr int32_t kExpOffset = kCumulativeOffset + kColumns * kFloatBytes;
  constexpr int32_t kBiasOffset = kExpOffset + kColumns * kFloatBytes;
  constexpr int32_t kBetaOffset = kBiasOffset + kColumns * kFloatBytes;
  static_assert(kBetaOffset + kRows * 8 * kFloatBytes <= 192 * 1024);
  ND<float, 1, kColumns> cumulative, positive_exp, bias;
  TASSIGN(cumulative, kCumulativeOffset);
  TASSIGN(positive_exp, kExpOffset);
  TASSIGN(bias, kBiasOffset);
  load(bias, a.gate_bias + head * kColumns, 1, kColumns, kColumns);
  TEXPANDS(positive_exp, a.a_log[head]);
  PipeBarrierVec();
  TEXP(positive_exp, positive_exp);
  PipeBarrierVec();
  TEXPANDS(cumulative, 0.0f);
  PipeBarrierVec();
  for (int64_t token = start; token < end; token += kRows) {
    const int32_t rows =
        static_cast<int32_t>(end - token < kRows ? end - token : kRows);
    ND<float, kRows, kColumns, DYNAMIC> gate(rows), temporary(rows);
    ND<bfloat16_t, kRows, kColumns, DYNAMIC> raw(rows);
    ND<float, kRows, 8, DYNAMIC, 1> beta(rows);
    TASSIGN(gate, 0);
    TASSIGN(temporary, kGateBytes);
    TASSIGN(raw, kRawOffset);
    TASSIGN(beta, kBetaOffset);
    load(raw,
         a.gate + (token * t.heads + head) * kColumns,
         rows,
         kColumns,
         t.heads * kColumns);
    TCVT(gate, raw, RoundMode::CAST_NONE);
    PipeBarrierVec();
    TCOLEXPANDADD(gate, gate, bias);
    PipeBarrierVec();
    TCOLEXPANDMUL(gate, gate, positive_exp);
    PipeBarrierVec();
    TMULS(gate, gate, -1.0f);
    PipeBarrierVec();
    TEXP(gate, gate);
    PipeBarrierVec();
    TADDS(gate, gate, 1.0f);
    PipeBarrierVec();
    finish_gate<UseLog2Gate>(gate, temporary);
    for (int32_t row = 0; row < rows; ++row) {
      ND<float, 1, kColumns> current;
      TASSIGN(current, row * kColumns * sizeof(float));
      if (row == 0) {
        TADD(current, cumulative, current);
      } else {
        ND<float, 1, kColumns> previous;
        TASSIGN(previous, (row - 1) * kColumns * sizeof(float));
        TADD(current, previous, current);
      }
      PipeBarrierVec();
    }
    ND<float, 1, kColumns> last;
    TASSIGN(last, (rows - 1) * kColumns * sizeof(float));
    TMOV(cumulative, last);
    PipeBarrierVec();
    store(reinterpret_cast<__gm__ float*>(ws + w.g) +
              (head * t.tokens + token) * kColumns,
          gate,
          rows,
          kColumns,
          kColumns);
    // beta is a head column in [T,H], not a contiguous token vector.
    load(beta, a.beta + token * t.heads + head, rows, 1, t.heads);
    store(
        reinterpret_cast<__gm__ float*>(ws + w.beta) + head * t.tokens + token,
        beta,
        rows,
        1,
        1);
  }
  pipe_barrier(PIPE_ALL);
#endif
}

template <int ChunkSize, bool UseMainConvRounding, bool UseLog2Gate>
AICORE inline void prepare(const Args& a,
                           const TilingData& t,
                           __gm__ uint8_t* ws,
                           const Workspace& w) {
#if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  const int64_t core = get_block_idx() * 2 + get_subblockid();
  const int64_t cores = get_block_num() * 2;
  int64_t chunks = 0;
  for (int64_t b = 0; b < t.batch; ++b) {
    chunks += (a.cu[b + 1] - a.cu[b] + ChunkSize - 1) / ChunkSize;
  }
  // Split independent Q/K/V work only when chunk/head tasks leave AIVs idle.
  // The V owner also produces gate/beta; every output region keeps one writer.
  const int64_t base_tasks = chunks * t.heads;
  const bool split_parts = base_tasks < cores;
  const int64_t task_count = base_tasks * (split_parts ? 3 : 1);
  FrontScratch s;
  s.init();
  for (int64_t task = core; task < task_count; task += cores) {
    const int64_t logical_task = split_parts ? task / 3 : task;
    const int32_t selected_part = split_parts ? static_cast<int32_t>(task % 3) : -1;
    const int64_t h = logical_task % t.heads;
    int64_t chunk = logical_task / t.heads;
    int64_t b = 0;
    for (; b < t.batch; ++b) {
      const int64_t n = (a.cu[b + 1] - a.cu[b] + ChunkSize - 1) / ChunkSize;
      if (chunk < n) {
        break;
      }
      chunk -= n;
    }
    const int64_t start = a.cu[b] + chunk * ChunkSize;
    const int64_t chunk_end = start + ChunkSize;
    const int64_t end = chunk_end < a.cu[b + 1] ? chunk_end : a.cu[b + 1];
    const int64_t history_end = a.cu[b] + 3 < end ? a.cu[b] + 3 : end;
    for (int64_t token = start; token < history_end; ++token) {
      for (int part = 0; part < 3; ++part) {
        if (selected_part >= 0 && part != selected_part) {
          continue;
        }
        conv<UseMainConvRounding>(s, a, t, b, token, (part * t.heads + h) * 128);
        if (part < 2) {
          normalize(s);
          store(reinterpret_cast<__gm__ bfloat16_t*>(ws +
                                                     (part == 0 ? w.q : w.k)) +
                    (h * t.tokens + token) * 128,
                s.bf,
                1,
                128,
                128);
        } else {
          store(reinterpret_cast<__gm__ float*>(ws + w.value) +
                    (token * t.heads + h) * 128,
                s.acc,
                1,
                128,
                128);
        }
      }
    }
    const int64_t bulk_start = start > history_end ? start : history_end;
    if (bulk_start < end) {
      if (end - bulk_start >= 64) {
        conv_rows<64, UseMainConvRounding>(a, t, ws, w, bulk_start, end, h, selected_part);
      } else if (end - bulk_start >= 32) {
        conv_rows<32, UseMainConvRounding>(a, t, ws, w, bulk_start, end, h, selected_part);
      } else {
        conv_rows<16, UseMainConvRounding>(a, t, ws, w, bulk_start, end, h, selected_part);
      }
    } else {
      pipe_barrier(PIPE_ALL);
    }
    if (selected_part < 0 || selected_part == 2) {
      if (end - start >= 16) {
        gate_rows<UseLog2Gate>(a, t, ws, w, start, end, h);
      } else {
        // ConvRows has drained the shared UB arena. Cache only this chunk/head's
        // immutable operands, retaining the device EXP and token arithmetic
        // order.
        load(s.cached_gate_bias, a.gate_bias + h * 128, 1, 128, 128);
        TEXPANDS(s.positive_exp_a, a.a_log[h]);
        PipeBarrierVec();
        TEXP(s.positive_exp_a, s.positive_exp_a);
        PipeBarrierVec();
        TEXPANDS(s.cumulative, 0.0f);
        PipeBarrierVec();
        for (int64_t token = start; token < end; ++token) {
          s.bf_load(s.gate, a.gate + (token * t.heads + h) * 128);
          TADD(s.gate, s.gate, s.cached_gate_bias);
          PipeBarrierVec();
          TMUL(s.gate, s.gate, s.positive_exp_a);
          PipeBarrierVec();
          TMULS(s.gate, s.gate, -1.0f);
          PipeBarrierVec();
          TEXP(s.gate, s.gate);
          PipeBarrierVec();
          TADDS(s.gate, s.gate, 1.0f);
          PipeBarrierVec();
          finish_gate<UseLog2Gate>(s.gate, s.tmp);
          TADD(s.cumulative, s.cumulative, s.gate);
          PipeBarrierVec();
          store(reinterpret_cast<__gm__ float*>(ws + w.g) +
                    (h * t.tokens + token) * 128,
                s.cumulative,
                1,
                128,
                128);
          load(s.scalar, a.beta + token * t.heads + h, 1, 1, 1);
          store(
              reinterpret_cast<__gm__ float*>(ws + w.beta) + h * t.tokens + token,
              s.scalar,
              1,
              1,
              1);
        }
      }
    }
    if (end == a.cu[b + 1] && a.conv_write[b] >= 0) {
      for (int part = 0; part < 3; ++part) {
        if (selected_part >= 0 && part != selected_part) {
          continue;
        }
        const int64_t channel = (part * t.heads + h) * 128;
        const int64_t c = t.heads * 384;
        for (int tap = 0; tap < 3; ++tap) {
          const int64_t pos = end - 3 + tap;
          if (pos < a.cu[b] && a.conv_read[b] < 0) {
            TEXPANDS(s.tmp, 0.0f);
            PipeBarrierVec();
            TCVT(s.bf, s.tmp, RoundMode::CAST_RINT);
            PipeBarrierVec();
          } else {
            auto* p = pos < a.cu[b]
                          ? a.conv_in +
                                (static_cast<int64_t>(a.conv_read[b]) * 3 +
                                 pos - a.cu[b] + 3) *
                                    c +
                                channel
                          : a.qkv + pos * c + channel;
            load(s.bf, p, 1, 128, 128);
          }
          store(a.conv_out +
                    (static_cast<int64_t>(a.conv_write[b]) * 3 + tap) * c +
                    channel,
                s.bf,
                1,
                128,
                128);
        }
      }
    }
  }
  // Constants are produced in UB and published through MTE3, never scalar GM
  // stores.
  for (int row = core; row < 128; row += cores) {
    SetWaitFlag<PIPE_V, PIPE_S>(0);
    for (int col = 0; col < 128; ++col) {
      s.tmp.data()[col] = col == row ? 1.0f : 0.0f;
    }
    SetWaitFlag<PIPE_S, PIPE_V>(0);
    store(reinterpret_cast<__gm__ float*>(ws + w.eye) + row * 128,
          s.tmp,
          1,
          128,
          128);
  }
#endif
}

template <bool Publish>
AICORE inline void state_io(const Args& a,
                            const TilingData& t,
                            __gm__ uint8_t* ws,
                            const Workspace& w) {
#if defined(__DAV_VEC__)
  const int64_t core = get_block_idx() * 2 + get_subblockid();
  const int64_t cores = get_block_num() * 2;
  ND<float, 16, 128> src;
  ND<float, 128, 16> dst;
  ND<float, 16, 128> tmp;
  TASSIGN(src, 0);
  TASSIGN(dst, 8192);
  TASSIGN(tmp, 16384);
  for (int64_t task = core; task < t.batch * t.heads * 8; task += cores) {
    const int64_t b = task / (t.heads * 8);
    const int64_t h = task / 8 % t.heads;
    const int64_t row = task % 8 * 16;
    if (a.cu[b] == a.cu[b + 1]) {
      continue;
    }
    if constexpr (Publish) {
      const int64_t slot = a.ssm_write[b];
      if (slot < 0) {
        continue;
      }
      load(src,
           reinterpret_cast<__gm__ float*>(ws + w.final) +
               (b * t.heads + h) * 16384 + row * 128,
           16,
           128,
           128);
      TTRANS(dst, src, tmp);
      PipeBarrierVec();
      store(a.ssm_out + (slot * t.heads + h) * 16384 + row, dst, 128, 16, 128);
    } else {
      const int64_t slot = a.ssm_read[b];
      if (slot < 0) {
        TEXPANDS(src, 0.0f);
        PipeBarrierVec();
      } else {
        load(src,
             a.ssm_in + (slot * t.heads + h) * 16384 + row * 128,
             16,
             128,
             128);
      }
      TTRANS(dst, src, tmp);
      PipeBarrierVec();
      store(reinterpret_cast<__gm__ float*>(ws + w.initial) +
                (b * t.heads + h) * 16384 + row,
            dst,
            128,
            16,
            128);
    }
  }
#endif
}

AICORE inline void output(const Args& a,
                          const TilingData& t,
                          __gm__ uint8_t* ws,
                          const Workspace& w) {
#if defined(__DAV_VEC__)
  constexpr int32_t kOutputRows = 16;
  const int64_t cores = get_block_num() * 2;
  const int64_t total_rows = t.tokens * t.heads;
  // Batch only long outputs; retain the original small-output parallelism.
  if (total_rows >= kOutputRows * cores) {
    for (int64_t first = (get_block_idx() * 2 + get_subblockid()) * kOutputRows;
         first < total_rows;
         first += cores * kOutputRows) {
      const int32_t rows = static_cast<int32_t>(
          total_rows - first < kOutputRows ? total_rows - first : kOutputRows);
      ND<float, kOutputRows, 128, DYNAMIC> x(rows);
      ND<bfloat16_t, kOutputRows, 128, DYNAMIC> y(rows);
      TASSIGN(x, 0);
      TASSIGN(y, kOutputRows * 128 * sizeof(float));
      load(x,
           reinterpret_cast<__gm__ float*>(ws + w.output) + first * 128,
           rows,
           128,
           128);
      TMULS(x, x, 0.08838834764831845f);
      PipeBarrierVec();
      TCVT(y, x, RoundMode::CAST_RINT);
      PipeBarrierVec();
      store(a.output + first * 128, y, rows, 128, 128);
    }
    return;
  }
  ND<float, 1, 128> x;
  ND<bfloat16_t, 1, 128> y;
  TASSIGN(x, 0);
  TASSIGN(y, 512);
  for (int64_t task = get_block_idx() * 2 + get_subblockid();
       task < t.tokens * t.heads;
       task += cores) {
    load(x,
         reinterpret_cast<__gm__ float*>(ws + w.output) + task * 128,
         1,
         128,
         128);
    TMULS(x, x, 0.08838834764831845f);
    PipeBarrierVec();
    TCVT(y, x, RoundMode::CAST_RINT);
    PipeBarrierVec();
    store(a.output + task * 128, y, 1, 128, 128);
  }
#endif
}
}  // namespace kda_prefill
