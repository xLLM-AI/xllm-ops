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

#pragma once

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

constexpr int32_t FLOAT_BLOCK_SIZE = 8;

using namespace AscendC;

template <typename TokenIdType, typename LogProbType>
class OnerecFinalBeamSelect {
 public:
  __aicore__ inline OnerecFinalBeamSelect() {}

  __aicore__ inline void Init(GM_ADDR log_probs,
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
                              TopkTiling& merge_topk_tiling_data);
  __aicore__ inline void Process();

  AscendC::TPipe pipe;

 private:
  __aicore__ inline int32_t AlignUp(int32_t value, int32_t alignment);
  __aicore__ inline void AlignUpDataCopyProb(
      AscendC::LocalTensor<LogProbType> dst,
      AscendC::GlobalTensor<LogProbType> src,
      int32_t length);
  __aicore__ inline void AlignUpDataCopyProbSlice(
      AscendC::LocalTensor<LogProbType> dst,
      AscendC::GlobalTensor<LogProbType> src,
      int32_t length,
      int32_t slice_length);
  __aicore__ inline void AlignUpDataCopyGm(
      AscendC::GlobalTensor<TokenIdType> dst,
      AscendC::LocalTensor<TokenIdType> src,
      int32_t length);
  __aicore__ inline void AlignUpDataCopyGmLog(
      AscendC::GlobalTensor<LogProbType> dst,
      AscendC::LocalTensor<LogProbType> src,
      int32_t length);
  __aicore__ inline void LoadAndTopKRequest(
      int32_t request_idx,
      AscendC::LocalTensor<LogProbType>& prefix_top_probs,
      AscendC::LocalTensor<int32_t>& prefix_top_index);
  __aicore__ inline void MergeTopK(
      AscendC::LocalTensor<LogProbType>& top_probs_local,
      AscendC::LocalTensor<LogProbType>& prefix_top_probs,
      AscendC::LocalTensor<int32_t>& prefix_top_index,
      int32_t round_length,
      int32_t round_idx);
  __aicore__ inline void WriteOutputs(
      int32_t request_idx,
      AscendC::LocalTensor<LogProbType>& prefix_top_probs,
      AscendC::LocalTensor<int32_t>& prefix_top_index);

  AscendC::GlobalTensor<LogProbType> log_probs_gm;
  AscendC::GlobalTensor<TokenIdType> top_tokens_gm;
  AscendC::GlobalTensor<LogProbType> top_probs_gm;
  AscendC::GlobalTensor<TokenIdType> sequence_gm;
  AscendC::GlobalTensor<TokenIdType> out_token_ids_gm;
  AscendC::GlobalTensor<TokenIdType> out_token_index_gm;
  AscendC::GlobalTensor<LogProbType> out_log_probs_gm;
  AscendC::GlobalTensor<TokenIdType> out_sequence_gm;

  int32_t num_sequences = 0;
  int32_t active_beam_width = 0;
  int32_t candidate_top_k = 0;
  int32_t result_width = 0;
  int32_t request_num = 0;
  int32_t current_step = 0;
  int32_t max_decode_step = 0;
  int32_t core_num = 0;
  int32_t step_size = 0;
  int32_t min_size = 0;
  int32_t align_candidate_top_k = 0;
  int32_t align_result_width = 0;
  TopkTiling first_topk_tiling_data;
  TopkTiling merge_topk_tiling_data;

  AscendC::TQue<AscendC::QuePosition::VECIN, 1> log_probs_in_que;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> top_probs_in_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_token_ids_out_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_token_index_out_que;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_log_probs_out_que;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_prob_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_result_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> prefix_probs_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> prefix_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> merge_probs_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> merge_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_second_res_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_second_res_index_buf;
  AscendC::TBuf<AscendC::TPosition::VECCALC> top_k_tmp_buf;
};