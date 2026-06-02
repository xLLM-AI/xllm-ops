/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "onerec_final_beam_select.h"

#include "kernel_operator.h"

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void OnerecFinalBeamSelect<TokenIdType, LogProbType>::Init(
    GM_ADDR log_probs,
    GM_ADDR top_tokens,
    GM_ADDR top_probs,
    GM_ADDR sequence,
    GM_ADDR out_token_ids,
    GM_ADDR out_token_index,
    GM_ADDR out_log_probs,
    GM_ADDR out_sequence,
    int32_t num_sequences,
    int32_t active_beam_width,
    int32_t candidate_top_k,
    int32_t result_width,
    int32_t request_num,
    int32_t current_step,
    int32_t max_decode_step,
    int32_t core_num,
    int32_t step_size,
    int32_t min_size,
    TopkTiling& first_topk_tiling_data,
    TopkTiling& merge_topk_tiling_data) {
  log_probs_gm.SetGlobalBuffer((__gm__ LogProbType*)log_probs);
  top_tokens_gm.SetGlobalBuffer((__gm__ TokenIdType*)top_tokens);
  top_probs_gm.SetGlobalBuffer((__gm__ LogProbType*)top_probs);
  sequence_gm.SetGlobalBuffer((__gm__ TokenIdType*)sequence);
  out_token_ids_gm.SetGlobalBuffer((__gm__ TokenIdType*)out_token_ids);
  out_token_index_gm.SetGlobalBuffer((__gm__ TokenIdType*)out_token_index);
  out_log_probs_gm.SetGlobalBuffer((__gm__ LogProbType*)out_log_probs);
  out_sequence_gm.SetGlobalBuffer((__gm__ TokenIdType*)out_sequence);

  this->num_sequences = num_sequences;
  this->active_beam_width = active_beam_width;
  this->candidate_top_k = candidate_top_k;
  this->result_width = result_width;
  this->request_num = request_num;
  this->current_step = current_step;
  this->max_decode_step = max_decode_step;
  this->core_num = core_num;
  this->step_size = step_size;
  this->min_size = min_size;
  this->align_candidate_top_k =
      AlignUp(this->candidate_top_k, FLOAT_BLOCK_SIZE);
  this->align_result_width = AlignUp(this->result_width, 32);
  this->first_topk_tiling_data = first_topk_tiling_data;
  this->merge_topk_tiling_data = merge_topk_tiling_data;

  pipe.InitBuffer(log_probs_in_que,
                  1,
                  AlignUp(this->active_beam_width, 32) * sizeof(LogProbType));
  pipe.InitBuffer(top_probs_in_que,
                  1,
                  AlignUp(this->step_size * this->align_candidate_top_k, 32) *
                      sizeof(LogProbType));
  pipe.InitBuffer(out_token_ids_out_que,
                  1,
                  this->align_result_width * sizeof(TokenIdType));
  pipe.InitBuffer(out_token_index_out_que,
                  1,
                  this->align_result_width * sizeof(TokenIdType));
  pipe.InitBuffer(out_log_probs_out_que,
                  1,
                  this->align_result_width * sizeof(LogProbType));
  pipe.InitBuffer(top_k_result_prob_buf,
                  this->align_candidate_top_k * sizeof(LogProbType));
  pipe.InitBuffer(top_k_result_index_buf,
                  this->align_candidate_top_k * sizeof(int32_t));
  pipe.InitBuffer(prefix_probs_buf,
                  this->align_result_width * sizeof(LogProbType));
  pipe.InitBuffer(prefix_index_buf,
                  this->align_result_width * sizeof(int32_t));
  pipe.InitBuffer(merge_probs_buf,
                  this->align_result_width * 2 * sizeof(LogProbType));
  pipe.InitBuffer(merge_index_buf,
                  this->align_result_width * 2 * sizeof(int32_t));
  pipe.InitBuffer(top_k_second_res_buf,
                  this->align_result_width * sizeof(LogProbType));
  pipe.InitBuffer(top_k_second_res_index_buf,
                  this->align_result_width * sizeof(int32_t));
  pipe.InitBuffer(top_k_tmp_buf, this->min_size);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline int32_t
OnerecFinalBeamSelect<TokenIdType, LogProbType>::AlignUp(int32_t value,
                                                      int32_t alignment) {
  if (value % alignment != 0) {
    value = (value / alignment + 1) * alignment;
  }
  return value;
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
OnerecFinalBeamSelect<TokenIdType, LogProbType>::AlignUpDataCopyProb(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src,
    int32_t length) {
  int32_t block_size = AlignUp(length, 32 / sizeof(LogProbType));
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = static_cast<LogProbType>(0.0f);
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copy_params;
  copy_params.blockLen = length * sizeof(LogProbType);
  copy_params.blockCount = 1;
  copy_params.srcStride = 0;
  copy_params.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copy_params, params);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
OnerecFinalBeamSelect<TokenIdType, LogProbType>::AlignUpDataCopyProbSlice(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src,
    int32_t length,
    int32_t slice_length) {
  int32_t block_size = AlignUp(length, 32 / sizeof(LogProbType));
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = static_cast<LogProbType>(-1.0f / 0.0f);
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copy_params;
  copy_params.blockLen = length * sizeof(LogProbType);
  copy_params.blockCount = slice_length;
  copy_params.srcStride = 0;
  copy_params.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copy_params, params);
  AscendC::PipeBarrier<PIPE_MTE2>();
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
OnerecFinalBeamSelect<TokenIdType, LogProbType>::AlignUpDataCopyGm(
    AscendC::GlobalTensor<TokenIdType> dst,
    AscendC::LocalTensor<TokenIdType> src,
    int32_t length) {
  AscendC::DataCopyExtParams copy_params;
  copy_params.blockLen = length * sizeof(TokenIdType);
  copy_params.blockCount = 1;
  copy_params.srcStride = 0;
  copy_params.dstStride = 0;
  copy_params.rsv = 0;
  AscendC::DataCopyPad(dst, src, copy_params);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
OnerecFinalBeamSelect<TokenIdType, LogProbType>::AlignUpDataCopyGmLog(
    AscendC::GlobalTensor<LogProbType> dst,
    AscendC::LocalTensor<LogProbType> src,
    int32_t length) {
  AscendC::DataCopyExtParams copy_params;
  copy_params.blockLen = length * sizeof(LogProbType);
  copy_params.blockCount = 1;
  copy_params.srcStride = 0;
  copy_params.dstStride = 0;
  copy_params.rsv = 0;
  AscendC::DataCopyPad(dst, src, copy_params);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
OnerecFinalBeamSelect<TokenIdType, LogProbType>::Process() {
  for (int32_t request_idx = 0; request_idx < this->request_num;
       request_idx++) {
    if (request_idx % this->core_num == block_idx) {
      LocalTensor<LogProbType> prefix_top_probs =
          prefix_probs_buf.Get<LogProbType>();
      AscendC::Duplicate<LogProbType>(
          prefix_top_probs,
          static_cast<LogProbType>(-1.0f / 0.0f),
          this->align_result_width);
      LocalTensor<int32_t> prefix_top_index =
          prefix_index_buf.Get<int32_t>();
      AscendC::Duplicate<int32_t>(
          prefix_top_index, static_cast<int32_t>(0), this->align_result_width);
      LoadAndTopKRequest(request_idx, prefix_top_probs, prefix_top_index);
      WriteOutputs(request_idx, prefix_top_probs, prefix_top_index);
    }
  }
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
OnerecFinalBeamSelect<TokenIdType, LogProbType>::LoadAndTopKRequest(
    int32_t request_idx,
    AscendC::LocalTensor<LogProbType>& prefix_top_probs,
    AscendC::LocalTensor<int32_t>& prefix_top_index) {
  LocalTensor<LogProbType> log_probs_local =
      log_probs_in_que.AllocTensor<LogProbType>();
  AlignUpDataCopyProb(log_probs_local,
                      log_probs_gm[request_idx * this->active_beam_width],
                      this->active_beam_width);
  log_probs_in_que.EnQue(log_probs_local);
  LocalTensor<LogProbType> log_probs_tmp =
      log_probs_in_que.DeQue<LogProbType>();

  int32_t beam_width_round =
      (this->active_beam_width + this->step_size - 1) / this->step_size;
  int32_t tail_number = this->active_beam_width % this->step_size;
  int32_t full_round_top_k_block_size =
      AlignUp(this->step_size * this->align_candidate_top_k, 32);
  for (int32_t round_idx = 0; round_idx < beam_width_round; ++round_idx) {
    bool is_last_round_with_tail =
        (round_idx == beam_width_round - 1 && tail_number != 0);
    int32_t round_size = is_last_round_with_tail ? tail_number : this->step_size;
    LocalTensor<LogProbType> top_probs_local =
        top_probs_in_que.AllocTensor<LogProbType>();
    AscendC::Duplicate<LogProbType>(
        top_probs_local,
        static_cast<LogProbType>(-1.0f / 0.0f),
        full_round_top_k_block_size);
    int32_t event_id = static_cast<int32_t>(
        pipe.FetchEventID(AscendC::HardEvent::V_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);

    AlignUpDataCopyProbSlice(
        top_probs_local,
        top_probs_gm[request_idx * this->active_beam_width *
                         this->candidate_top_k +
                     round_idx * this->step_size * this->candidate_top_k],
        this->candidate_top_k,
        round_size);
    top_probs_in_que.EnQue(top_probs_local);
    LocalTensor<LogProbType> top_probs_tmp =
        top_probs_in_que.DeQue<LogProbType>();
    for (int32_t beam_offset = 0; beam_offset < round_size; ++beam_offset) {
      int32_t log_probs_beam_offset =
          round_idx * this->step_size + beam_offset;
      LogProbType log_probs_value =
          log_probs_tmp.GetValue(log_probs_beam_offset);
      AscendC::Adds<LogProbType>(
          top_probs_tmp[beam_offset * this->align_candidate_top_k],
          top_probs_tmp[beam_offset * this->align_candidate_top_k],
          log_probs_value,
          this->candidate_top_k);
    }
    MergeTopK(top_probs_tmp,
              prefix_top_probs,
              prefix_top_index,
              round_size,
              round_idx);
    top_probs_in_que.FreeTensor(top_probs_tmp);
  }
  log_probs_in_que.FreeTensor(log_probs_tmp);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void OnerecFinalBeamSelect<TokenIdType, LogProbType>::MergeTopK(
    AscendC::LocalTensor<LogProbType>& top_probs_local,
    AscendC::LocalTensor<LogProbType>& prefix_top_probs,
    AscendC::LocalTensor<int32_t>& prefix_top_index,
    int32_t round_length,
    int32_t round_idx) {
  LocalTensor<LogProbType> dst_local_value =
      top_k_result_prob_buf.Get<LogProbType>();
  LocalTensor<int32_t> dst_local_index = top_k_result_index_buf.Get<int32_t>();
  LocalTensor<int32_t> src_local_index;
  LocalTensor<bool> src_local_finish;
  int32_t block_size = AlignUp(round_length * this->align_candidate_top_k, 32);
  TopKInfo topk_info;
  topk_info.outter = 1;
  topk_info.inner = block_size;
  topk_info.n = round_length * this->align_candidate_top_k;
  bool is_largest = true;
  LocalTensor<uint8_t> tmp_local = top_k_tmp_buf.Get<uint8_t>();

  AscendC::TopK<LogProbType,
                false,
                false,
                false,
                AscendC::TopKMode::TOPK_NORMAL>(dst_local_value,
                                                dst_local_index,
                                                top_probs_local,
                                                src_local_index,
                                                src_local_finish,
                                                tmp_local,
                                                this->align_candidate_top_k,
                                                this->first_topk_tiling_data,
                                                topk_info,
                                                is_largest);
  AscendC::Adds<int32_t>(
      dst_local_index,
      dst_local_index,
      static_cast<int32_t>(round_idx * this->step_size *
                           this->align_candidate_top_k),
      this->align_candidate_top_k);

  LocalTensor<LogProbType> merge_top_probs =
      merge_probs_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(
      merge_top_probs,
      static_cast<LogProbType>(-1.0f / 0.0f),
      2 * this->align_result_width);
  AscendC::DataCopy(merge_top_probs,
                    dst_local_value,
                    this->align_candidate_top_k);
  AscendC::DataCopy(merge_top_probs[this->align_result_width],
                    prefix_top_probs,
                    this->align_result_width);
  LocalTensor<int32_t> merge_index = merge_index_buf.Get<int32_t>();
  AscendC::DataCopy(merge_index, dst_local_index, this->align_candidate_top_k);
  AscendC::DataCopy(merge_index[this->align_result_width],
                    prefix_top_index,
                    this->align_result_width);

  LocalTensor<LogProbType> dst_merge_probs =
      top_k_second_res_buf.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(
      dst_merge_probs,
      static_cast<LogProbType>(-1.0f / 0.0f),
      this->align_result_width);
  LocalTensor<int32_t> dst_merge_index =
      top_k_second_res_index_buf.Get<int32_t>();
  AscendC::Duplicate<int32_t>(
      dst_merge_index, static_cast<int32_t>(0), this->align_result_width);
  int32_t block_size_merge = AlignUp(this->result_width * 2, 32);
  TopKInfo topk_info2;
  topk_info2.outter = 1;
  topk_info2.inner = block_size_merge;
  topk_info2.n = this->result_width * 2;
  AscendC::TopK<LogProbType,
                true,
                false,
                false,
                AscendC::TopKMode::TOPK_NORMAL>(dst_merge_probs,
                                                dst_merge_index,
                                                merge_top_probs,
                                                merge_index,
                                                src_local_finish,
                                                tmp_local,
                                                this->align_result_width,
                                                this->merge_topk_tiling_data,
                                                topk_info2,
                                                is_largest);
  AscendC::DataCopy(
      prefix_top_probs, dst_merge_probs, this->align_result_width);
  AscendC::DataCopy(
      prefix_top_index, dst_merge_index, this->align_result_width);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
OnerecFinalBeamSelect<TokenIdType, LogProbType>::WriteOutputs(
    int32_t request_idx,
    AscendC::LocalTensor<LogProbType>& prefix_top_probs,
    AscendC::LocalTensor<int32_t>& prefix_top_index) {
  LocalTensor<TokenIdType> out_token_ids_local =
      out_token_ids_out_que.AllocTensor<TokenIdType>();
  LocalTensor<TokenIdType> out_token_index_local =
      out_token_index_out_que.AllocTensor<TokenIdType>();
  LocalTensor<LogProbType> out_log_probs_local =
      out_log_probs_out_que.AllocTensor<LogProbType>();

  for (int32_t i = 0; i < this->result_width; ++i) {
    int32_t flat_candidate = prefix_top_index.GetValue(i);
    int32_t parent_beam = flat_candidate / this->align_candidate_top_k;
    int32_t token_in_beam =
        flat_candidate - parent_beam * this->align_candidate_top_k;
    int32_t top_token_index =
        request_idx * this->active_beam_width * this->candidate_top_k +
        parent_beam * this->candidate_top_k + token_in_beam;
    out_token_ids_local.SetValue(i, top_tokens_gm.GetValue(top_token_index));
    out_token_index_local.SetValue(
        i,
        static_cast<TokenIdType>(request_idx * this->active_beam_width +
                                 parent_beam));
    out_log_probs_local.SetValue(i, prefix_top_probs.GetValue(i));

    for (int32_t step = 0; step < this->current_step; ++step) {
      int32_t src_seq_index =
          request_idx * this->active_beam_width * this->max_decode_step +
          parent_beam * this->max_decode_step + step;
      int32_t dst_seq_index =
          request_idx * this->result_width * this->max_decode_step +
          i * this->max_decode_step + step;
      out_sequence_gm.SetValue(dst_seq_index,
                               sequence_gm.GetValue(src_seq_index));
    }
    int32_t dst_token_index =
        request_idx * this->result_width * this->max_decode_step +
        i * this->max_decode_step + this->current_step;
    out_sequence_gm.SetValue(dst_token_index,
                             out_token_ids_local.GetValue(i));
  }

  out_token_ids_out_que.EnQue(out_token_ids_local);
  LocalTensor<TokenIdType> out_token_ids_tmp =
      out_token_ids_out_que.DeQue<TokenIdType>();
  AlignUpDataCopyGm(
      out_token_ids_gm[request_idx * this->result_width],
      out_token_ids_tmp,
      this->result_width);
  out_token_ids_out_que.FreeTensor(out_token_ids_tmp);

  out_token_index_out_que.EnQue(out_token_index_local);
  LocalTensor<TokenIdType> out_token_index_tmp =
      out_token_index_out_que.DeQue<TokenIdType>();
  AlignUpDataCopyGm(
      out_token_index_gm[request_idx * this->result_width],
      out_token_index_tmp,
      this->result_width);
  out_token_index_out_que.FreeTensor(out_token_index_tmp);

  out_log_probs_out_que.EnQue(out_log_probs_local);
  LocalTensor<LogProbType> out_log_probs_tmp =
      out_log_probs_out_que.DeQue<LogProbType>();
  AlignUpDataCopyGmLog(
      out_log_probs_gm[request_idx * this->result_width],
      out_log_probs_tmp,
      this->result_width);
  out_log_probs_out_que.FreeTensor(out_log_probs_tmp);
}

extern "C" __global__ __aicore__ void onerec_final_beam_select(
    GM_ADDR log_probs,
    GM_ADDR top_tokens,
    GM_ADDR top_probs,
    GM_ADDR sequence,
    GM_ADDR out_token_ids,
    GM_ADDR out_token_index,
    GM_ADDR out_log_probs,
    GM_ADDR out_sequence,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_VECTOR_CORE);
  GET_TILING_DATA(tiling_data, tiling);
  OnerecFinalBeamSelect<int32_t, float> op;
  op.Init(log_probs,
          top_tokens,
          top_probs,
          sequence,
          out_token_ids,
          out_token_index,
          out_log_probs,
          out_sequence,
          tiling_data.num_sequences,
          tiling_data.active_beam_width,
          tiling_data.candidate_top_k,
          tiling_data.result_width,
          tiling_data.request_num,
          tiling_data.current_step,
          tiling_data.max_decode_step,
          tiling_data.core_num,
          tiling_data.step_size,
          tiling_data.min_size,
          tiling_data.firstTopkTilingData,
          tiling_data.mergeTopkTilingData);
  op.Process();
  op.pipe.Destroy();
}