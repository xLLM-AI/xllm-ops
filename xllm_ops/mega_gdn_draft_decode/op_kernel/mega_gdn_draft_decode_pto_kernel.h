/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

// Reuse the precision-matched Conv, recurrent and Norm primitives. Draft has
// an independent schedule because it consumes packed 0/1/2-token sequences
// and stores only the final Conv/SSM state.
#include "mega_gdn_mtp_decode_pto_kernel.h"

namespace mega_gdn_draft_decode_pto {

using namespace pto;
using mega_gdn_decode_pto::TileUbDataND;
using namespace mega_gdn_mtp_decode_pto::ub_layout;

constexpr int32_t kHeadDim = 128;
constexpr int32_t kSsmHeadElements = kHeadDim * kHeadDim;
constexpr int32_t kConvStateLength = 3;

AICORE PTO_INLINE void ZeroConvHistory() {
  TileUbDataND<float, 1, kHeadDim> history0;
  TASSIGN(history0, kUbConvHistory0);
  TileUbDataND<float, 1, kHeadDim> history1;
  TASSIGN(history1, kUbConvHistory1);
  TileUbDataND<float, 1, kHeadDim> history2;
  TASSIGN(history2, kUbConvHistory2);
  TMULS(history0, history0, 0.0f);
  TMULS(history1, history1, 0.0f);
  TMULS(history2, history2, 0.0f);
  mega_gdn_decode_pto::VectorBarrier();
}

AICORE PTO_INLINE void StoreFinalConvHistory(
    __gm__ bfloat16_t* conv_state_out_handle,
    int32_t write_slot,
    int32_t conv_dim,
    int32_t channel_offset) {
  TileUbDataND<bfloat16_t, 1, kHeadDim> history_half0;
  TASSIGN(history_half0, kUbConvHistoryHalf0);
  TileUbDataND<bfloat16_t, 1, kHeadDim> history_half1;
  TASSIGN(history_half1, kUbConvHistoryHalf1);
  TileUbDataND<bfloat16_t, 1, kHeadDim> history_half2;
  TASSIGN(history_half2, kUbConvHistoryHalf2);
  TileUbDataND<float, 1, kHeadDim> history0;
  TASSIGN(history0, kUbConvHistory0);
  TileUbDataND<float, 1, kHeadDim> history1;
  TASSIGN(history1, kUbConvHistory1);
  TileUbDataND<float, 1, kHeadDim> history2;
  TASSIGN(history2, kUbConvHistory2);

  TCVT(history_half0, history0, RoundMode::CAST_RINT);
  TCVT(history_half1, history1, RoundMode::CAST_RINT);
  TCVT(history_half2, history2, RoundMode::CAST_RINT);
  mega_gdn_decode_pto::VectorBarrier();
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  mega_gdn_mtp_decode_pto::StoreBf16Row(
      conv_state_out_handle +
          write_slot * kConvStateLength * conv_dim + channel_offset,
      kUbConvHistoryHalf0);
  mega_gdn_mtp_decode_pto::StoreBf16Row(
      conv_state_out_handle +
          write_slot * kConvStateLength * conv_dim + conv_dim +
          channel_offset,
      kUbConvHistoryHalf1);
  mega_gdn_mtp_decode_pto::StoreBf16Row(
      conv_state_out_handle +
          write_slot * kConvStateLength * conv_dim + 2 * conv_dim +
          channel_offset,
      kUbConvHistoryHalf2);
  // The next batch reuses these BF16 history buffers from both MTE2 and V.
  // Match the proven MTP hand-off before either pipeline touches them again.
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
}

AICORE PTO_INLINE void RunConvPhase(
    __gm__ bfloat16_t* qkv_handle,
    __gm__ bfloat16_t* conv_weight_handle,
    __gm__ bfloat16_t* conv_state_handle,
    __gm__ int* read_state_indices_handle,
    __gm__ int* write_state_indices_handle,
    __gm__ int* q_cu_seq_lens_handle,
    __gm__ uint8_t* state_validity_mask_handle,
    __gm__ bfloat16_t* conv_out_handle,
    __gm__ bfloat16_t* conv_state_out_handle,
    int32_t batch_size,
    int32_t conv_dim,
    int32_t conv_tile_count,
    int32_t vector_core_idx,
    int32_t vector_core_count) {
  TileUbDataND<bfloat16_t, 1, kHeadDim> weight_half0;
  TASSIGN(weight_half0, kUbConvWeightHalf0);
  TileUbDataND<bfloat16_t, 1, kHeadDim> weight_half1;
  TASSIGN(weight_half1, kUbConvWeightHalf1);
  TileUbDataND<bfloat16_t, 1, kHeadDim> weight_half2;
  TASSIGN(weight_half2, kUbConvWeightHalf2);
  TileUbDataND<bfloat16_t, 1, kHeadDim> weight_half3;
  TASSIGN(weight_half3, kUbConvWeightHalf3);
  TileUbDataND<float, 1, kHeadDim> weight0;
  TASSIGN(weight0, kUbConvWeight0);
  TileUbDataND<float, 1, kHeadDim> weight1;
  TASSIGN(weight1, kUbConvWeight1);
  TileUbDataND<float, 1, kHeadDim> weight2;
  TASSIGN(weight2, kUbConvWeight2);
  TileUbDataND<float, 1, kHeadDim> weight3;
  TASSIGN(weight3, kUbConvWeight3);
  TileUbDataND<bfloat16_t, 1, kHeadDim> history_half0;
  TASSIGN(history_half0, kUbConvHistoryHalf0);
  TileUbDataND<bfloat16_t, 1, kHeadDim> history_half1;
  TASSIGN(history_half1, kUbConvHistoryHalf1);
  TileUbDataND<bfloat16_t, 1, kHeadDim> history_half2;
  TASSIGN(history_half2, kUbConvHistoryHalf2);
  TileUbDataND<bfloat16_t, 1, kHeadDim> input_half;
  TASSIGN(input_half, kUbConvInputHalf);
  TileUbDataND<float, 1, kHeadDim> history0;
  TASSIGN(history0, kUbConvHistory0);
  TileUbDataND<float, 1, kHeadDim> history1;
  TASSIGN(history1, kUbConvHistory1);
  TileUbDataND<float, 1, kHeadDim> history2;
  TASSIGN(history2, kUbConvHistory2);
  TileUbDataND<float, 1, kHeadDim> input;
  TASSIGN(input, kUbConvInput);
  TileUbDataND<float, 1, kHeadDim> conv_acc;
  TASSIGN(conv_acc, kUbConvAcc);
  TileUbDataND<float, 1, kHeadDim> conv_tmp;
  TASSIGN(conv_tmp, kUbConvTmp);
  TileUbDataND<float, 1, kHeadDim> conv_output;
  TASSIGN(conv_output, kUbConvOutput);
  TileUbDataND<bfloat16_t, 1, kHeadDim> conv_output_half;
  TASSIGN(conv_output_half, kUbConvOutputHalf);

  for (int32_t conv_tile = vector_core_idx; conv_tile < conv_tile_count;
       conv_tile += vector_core_count) {
    const int32_t channel_offset = conv_tile * kHeadDim;
    mega_gdn_decode_pto::CopyConvWeightsGmToUb(
        conv_weight_handle + channel_offset,
        kUbConvWeightHalf0,
        conv_dim);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TCVT(weight0, weight_half0, RoundMode::CAST_NONE);
    TCVT(weight1, weight_half1, RoundMode::CAST_NONE);
    TCVT(weight2, weight_half2, RoundMode::CAST_NONE);
    TCVT(weight3, weight_half3, RoundMode::CAST_NONE);
    mega_gdn_decode_pto::VectorBarrier();

    for (int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
      const bool has_initial_state =
          *(state_validity_mask_handle + batch_idx) != 0;
      const int32_t read_slot =
          has_initial_state ? *(read_state_indices_handle + batch_idx) : 0;
      const int32_t write_slot =
          *(write_state_indices_handle + batch_idx);
      const int32_t token_begin = *(q_cu_seq_lens_handle + batch_idx);
      const int32_t token_end = *(q_cu_seq_lens_handle + batch_idx + 1);
      const int32_t history_offset =
          read_slot * kConvStateLength * conv_dim + channel_offset;
      mega_gdn_decode_pto::CopyConvHistoryGmToUb(
          conv_state_handle + history_offset,
          kUbConvHistoryHalf0,
          conv_dim);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TCVT(history0, history_half0, RoundMode::CAST_NONE);
      TCVT(history1, history_half1, RoundMode::CAST_NONE);
      TCVT(history2, history_half2, RoundMode::CAST_NONE);
      mega_gdn_decode_pto::VectorBarrier();
      if (!has_initial_state) {
        ZeroConvHistory();
      }

      for (int32_t token_idx = token_begin; token_idx < token_end;
           ++token_idx) {
        const int32_t token_offset = token_idx * conv_dim + channel_offset;
        mega_gdn_mtp_decode_pto::LoadBf16Row(
            qkv_handle + token_offset, kUbConvInputHalf);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        TCVT(input, input_half, RoundMode::CAST_NONE);
        mega_gdn_decode_pto::VectorBarrier();

        TMUL(conv_acc, weight0, history0);
#if defined(PTO_NPU_ARCH_A5)
        mega_gdn_decode_pto::VectorBarrier();
        mega_gdn_decode_pto::MulAddDst<float, 1, kHeadDim>(
            conv_acc, history1, weight1, conv_tmp);
        mega_gdn_decode_pto::VectorBarrier();
        mega_gdn_decode_pto::MulAddDst<float, 1, kHeadDim>(
            conv_acc, history2, weight2, conv_tmp);
#else
        TMUL(conv_tmp, weight1, history1);
        mega_gdn_decode_pto::VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
        mega_gdn_decode_pto::VectorBarrier();
        TMUL(conv_tmp, weight2, history2);
        mega_gdn_decode_pto::VectorBarrier();
        TADD(conv_acc, conv_acc, conv_tmp);
#endif
        mega_gdn_decode_pto::VectorBarrier();
        TileUbDataND<float, 1, kHeadDim> vector_scratch;
        TASSIGN(vector_scratch, kUbVectorScratch);
        mega_gdn_decode_pto::MulAddDst<float, 1, kHeadDim>(
            conv_acc, input, weight3, vector_scratch);
        mega_gdn_decode_pto::VectorBarrier();
        mega_gdn_decode_pto::CausalConvSilu<float, 1, kHeadDim>(
            conv_output, conv_acc, vector_scratch);
        mega_gdn_decode_pto::VectorBarrier();
        // Contract: the unfused Conv hand-off rounds to BF16 before Q/K/V.
        TCVT(conv_output_half, conv_output, RoundMode::CAST_RINT);
        mega_gdn_decode_pto::VectorBarrier();
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID3);
        mega_gdn_mtp_decode_pto::StoreBf16Row(
            conv_out_handle + token_offset, kUbConvOutputHalf);
        // Complete the BF16 hand-off before the next packed token starts its
        // MTE2 load or the Vector pipeline updates recurrent Conv history.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);

        TMOV(history0, history1);
        TMOV(history1, history2);
        TMOV(history2, input);
        mega_gdn_decode_pto::VectorBarrier();
      }

      if (token_end > token_begin) {
        StoreFinalConvHistory(conv_state_out_handle,
                              write_slot,
                              conv_dim,
                              channel_offset);
      }
    }
  }
}

AICORE PTO_INLINE void LoadInitialState(
    __gm__ float* ssm_state_handle,
    __gm__ bfloat16_t* norm_weight_handle,
    __gm__ float* a_log_handle,
    __gm__ float* dt_bias_handle,
    int32_t read_slot,
    int32_t ssm_slot_stride,
    int32_t head_idx,
    bool has_initial_state) {
  TileUbDataND<bfloat16_t, 1, kHeadDim> norm_weight_half;
  TASSIGN(norm_weight_half, kUbNormWeightHalf);
  TileUbDataND<float, 1, kHeadDim> norm_weight;
  TASSIGN(norm_weight, kUbNormWeight);
  TileUbDataND<float, 1, 8, 1, 1> a_log_static;
  TASSIGN(a_log_static, kUbALogStatic);
  TileUbDataND<float, kHeadDim, kHeadDim> state;
  TASSIGN(state, kUbState);

  const int32_t safe_read_slot = has_initial_state ? read_slot : 0;
  const int64_t state_offset =
      static_cast<int64_t>(safe_read_slot) * ssm_slot_stride +
      head_idx * kSsmHeadElements;
  mega_gdn_mtp_decode_pto::LoadState(
      ssm_state_handle + state_offset, kUbState);
  mega_gdn_mtp_decode_pto::LoadBf16Row(
      norm_weight_handle, kUbNormWeightHalf);
  mega_gdn_mtp_decode_pto::LoadFloatScalar(
      a_log_handle + head_idx, kUbALogStatic);
  mega_gdn_mtp_decode_pto::LoadFloatScalar(
      dt_bias_handle + head_idx, kUbDtBiasStatic);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TCVT(norm_weight, norm_weight_half, RoundMode::CAST_NONE);
  TEXP(a_log_static, a_log_static);
  mega_gdn_decode_pto::VectorBarrier();
  if (!has_initial_state) {
    TMULS(state, state, 0.0f);
    mega_gdn_decode_pto::VectorBarrier();
  }
}

AICORE PTO_INLINE void StoreFinalState(
    __gm__ float* ssm_state_out_handle,
    int32_t write_slot,
    int32_t ssm_slot_stride,
    int32_t head_idx) {
  const int64_t state_offset =
      static_cast<int64_t>(write_slot) * ssm_slot_stride +
      head_idx * kSsmHeadElements;
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
  mega_gdn_mtp_decode_pto::StoreState(
      ssm_state_out_handle + state_offset, kUbState);
  // A grid-stride AIV may immediately load the next head into the same State
  // UB. Wait for MTE3 before either MTE2 or V reuses that buffer.
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID4);
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
}

template <bool FlaSsmStateLayout>
AICORE PTO_INLINE void Run(
    __gm__ bfloat16_t* qkv_handle,
    __gm__ bfloat16_t* z_handle,
    __gm__ bfloat16_t* b_handle,
    __gm__ bfloat16_t* a_handle,
    __gm__ bfloat16_t* conv_weight_handle,
    __gm__ bfloat16_t* conv_state_handle,
    __gm__ float* a_log_handle,
    __gm__ float* dt_bias_handle,
    __gm__ float* ssm_state_handle,
    __gm__ int* read_state_indices_handle,
    __gm__ int* write_state_indices_handle,
    __gm__ int* q_cu_seq_lens_handle,
    __gm__ uint8_t* state_validity_mask_handle,
    __gm__ bfloat16_t* norm_weight_handle,
    __gm__ bfloat16_t* conv_out_handle,
    __gm__ bfloat16_t* conv_state_out_handle,
    __gm__ float* ssm_state_out_handle,
    __gm__ bfloat16_t* out_handle,
    int32_t total_tokens,
    int32_t batch_size,
    int32_t num_k_heads,
    int32_t num_v_heads
#if defined(PTO_NPU_ARCH_A5)
    , GM_ADDR soft_sync_workspace
#endif
    ) {
  const int32_t conv_dim =
      (2 * num_k_heads + num_v_heads) * kHeadDim;
  const int32_t conv_tile_count = conv_dim / kHeadDim;
  const int32_t v_heads_per_k = num_v_heads / num_k_heads;
  const int32_t value_width = num_v_heads * kHeadDim;
  const int32_t ssm_slot_stride = num_v_heads * kSsmHeadElements;

  TileUbDataND<bfloat16_t, 1, kHeadDim> q_half;
  TASSIGN(q_half, kUbQHalf);
  TileUbDataND<bfloat16_t, 1, kHeadDim> k_half;
  TASSIGN(k_half, kUbKHalf);
  TileUbDataND<bfloat16_t, 1, kHeadDim> v_half;
  TASSIGN(v_half, kUbVHalf);
  TileUbDataND<bfloat16_t, 1, kHeadDim> z_half;
  TASSIGN(z_half, kUbZHalf);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> a_half;
  TASSIGN(a_half, kUbAHalf);
  TileUbDataND<bfloat16_t, 1, 16, 1, 1> b_half;
  TASSIGN(b_half, kUbBHalf);
  TileUbDataND<float, 1, kHeadDim> q;
  TASSIGN(q, kUbQ);
  TileUbDataND<float, 1, kHeadDim> k;
  TASSIGN(k, kUbK);
  TileUbDataND<float, 1, kHeadDim> v;
  TASSIGN(v, kUbV);
  TileUbDataND<float, 1, kHeadDim> z;
  TASSIGN(z, kUbZ);
  TileUbDataND<float, 1, 8, 1, 1> scalar;
  TASSIGN(scalar, kUbScalar);
  TileUbDataND<float, 1, 8, 1, 1> scalar2;
  TASSIGN(scalar2, kUbScalar2);
  TileUbDataND<float, 1, kHeadDim> norm_square;
  TASSIGN(norm_square, kUbNormSquare);
  TileUbDataND<float, 1, 8, 1, 1> norm_value;
  TASSIGN(norm_value, kUbNormValue);
  TileUbDataND<uint8_t, 128, 64> reduce_tmp;
  TASSIGN(reduce_tmp, kUbReduceTmp);
  TileUbDataND<float, 1, 8, 1, 1> scalar_tmp;
  TASSIGN(scalar_tmp, kUbScalarTmp);

#if defined(__DAV_VEC__) || defined(__DAV_C220_VEC__)
#if defined(PTO_NPU_ARCH_A2A3)
  const int32_t vector_core_idx =
      get_block_idx() * get_subblockdim() + get_subblockid();
  const int32_t vector_core_count =
      get_block_num() * get_subblockdim();
  set_mask_norm();
  set_vector_mask(-1, -1);
#else
  const int32_t vector_core_idx = get_block_idx();
  const int32_t vector_core_count = get_block_num();
#endif

  RunConvPhase(qkv_handle,
               conv_weight_handle,
               conv_state_handle,
               read_state_indices_handle,
               write_state_indices_handle,
               q_cu_seq_lens_handle,
               state_validity_mask_handle,
               conv_out_handle,
               conv_state_out_handle,
               batch_size,
               conv_dim,
               conv_tile_count,
               vector_core_idx,
               vector_core_count);

  // Conv output is a BF16 GM hand-off between channel and head owners.
#if defined(PTO_NPU_ARCH_A5)
  mega_gdn_mtp_decode_pto::SyncAllAivSoft(
      soft_sync_workspace, vector_core_count, vector_core_idx);
#else
  mega_gdn_decode_pto::SyncAllAiv();
#endif

  const int32_t total_heads = batch_size * num_v_heads;
  for (int32_t head_task = vector_core_idx; head_task < total_heads;
       head_task += vector_core_count) {
    const int32_t batch_idx = head_task / num_v_heads;
    const int32_t head_idx = head_task % num_v_heads;
    const int32_t qk_head_idx = head_idx / v_heads_per_k;
    const int32_t read_slot =
        *(read_state_indices_handle + batch_idx);
    const int32_t write_slot =
        *(write_state_indices_handle + batch_idx);
    const bool has_initial_state =
        *(state_validity_mask_handle + batch_idx) != 0;
    const int32_t token_begin = *(q_cu_seq_lens_handle + batch_idx);
    const int32_t token_end = *(q_cu_seq_lens_handle + batch_idx + 1);
    if (token_end <= token_begin) {
      continue;
    }

    LoadInitialState(ssm_state_handle,
                     norm_weight_handle,
                     a_log_handle,
                     dt_bias_handle,
                     read_slot,
                     ssm_slot_stride,
                     head_idx,
                     has_initial_state);

    for (int32_t token_idx = token_begin; token_idx < token_end;
         ++token_idx) {
      const int32_t conv_token_offset = token_idx * conv_dim;
      mega_gdn_mtp_decode_pto::LoadBf16Row(
          conv_out_handle + conv_token_offset + qk_head_idx * kHeadDim,
          kUbQHalf);
      mega_gdn_mtp_decode_pto::LoadBf16Row(
          conv_out_handle + conv_token_offset +
              num_k_heads * kHeadDim + qk_head_idx * kHeadDim,
          kUbKHalf);
      mega_gdn_mtp_decode_pto::LoadBf16Row(
          conv_out_handle + conv_token_offset +
              2 * num_k_heads * kHeadDim + head_idx * kHeadDim,
          kUbVHalf);
      const int32_t scalar_offset = token_idx * num_v_heads + head_idx;
      mega_gdn_mtp_decode_pto::LoadBf16Scalar(
          a_handle + scalar_offset, kUbAHalf);
      mega_gdn_mtp_decode_pto::LoadBf16Scalar(
          b_handle + scalar_offset, kUbBHalf);
      mega_gdn_mtp_decode_pto::LoadBf16Row(
          z_handle + token_idx * value_width + head_idx * kHeadDim,
          kUbZHalf);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
      TCVT(q, q_half, RoundMode::CAST_NONE);
      TCVT(k, k_half, RoundMode::CAST_NONE);
      TCVT(v, v_half, RoundMode::CAST_NONE);
      TCVT(z, z_half, RoundMode::CAST_NONE);
      TCVT(scalar, a_half, RoundMode::CAST_NONE);
      TCVT(scalar2, b_half, RoundMode::CAST_NONE);
      mega_gdn_decode_pto::VectorBarrier();

      mega_gdn_decode_pto::NormalizeQk128<true>(
          q, norm_square, norm_value, reduce_tmp, scalar_tmp);
      mega_gdn_decode_pto::NormalizeQk128<false>(
          k, norm_square, norm_value, reduce_tmp, scalar_tmp);
      mega_gdn_mtp_decode_pto::RunRecurrentStep<
          false,
          FlaSsmStateLayout>(kUbQ, kUbK, 0);
      // The shared verify path writes a checkpoint before Norm, and that
      // MTE3 dependency also waits for its final readout-to-Norm TMOV. Draft
      // has no per-token checkpoint, so provide the missing V-to-V hand-off.
      mega_gdn_decode_pto::VectorBarrier();
      mega_gdn_mtp_decode_pto::RunNormStep<0>(out_handle,
                                              0,
                                              token_idx,
                                              head_idx,
                                              total_tokens,
                                              value_width);
    }

    StoreFinalState(ssm_state_out_handle,
                    write_slot,
                    ssm_slot_stride,
                    head_idx);
  }
#endif
}

}  // namespace mega_gdn_draft_decode_pto
