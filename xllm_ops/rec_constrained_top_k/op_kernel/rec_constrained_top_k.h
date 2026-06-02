/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "kernel_operator.h"

namespace kernels {

constexpr int32_t kRecConstrainedTopKMaxK = 256;
constexpr float kRecConstrainedTopKInvalidLogProb = -1.0e20f;
constexpr int32_t kRecConstrainedTopKMathBufferBytes = 32;
constexpr int32_t kRecConstrainedTopKExpBlockElements = 128;
constexpr int32_t kRecConstrainedTopKExpBufferBytes =
    kRecConstrainedTopKExpBlockElements *
    static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKLoadBf16Elements = 16;
constexpr int32_t kRecConstrainedTopKLoadBf16BufferBytes =
    kRecConstrainedTopKLoadBf16Elements * static_cast<int32_t>(sizeof(bfloat16_t));
constexpr int32_t kRecConstrainedTopKLoadFloatBufferBytes =
    kRecConstrainedTopKLoadBf16Elements * static_cast<int32_t>(sizeof(float));
constexpr int32_t kRecConstrainedTopKOutputTokenBufferBytes =
    kRecConstrainedTopKMaxK * static_cast<int32_t>(sizeof(int32_t));
constexpr int32_t kRecConstrainedTopKOutputLogprobBufferBytes =
    kRecConstrainedTopKMaxK * static_cast<int32_t>(sizeof(float));

template <typename LogitType>
class RecConstrainedTopKKernel {
 public:
  __aicore__ inline RecConstrainedTopKKernel() {}

  __aicore__ inline void Init(GM_ADDR logits,
                              GM_ADDR sequence_group,
                              GM_ADDR first_token_ids,
                              GM_ADDR prefix1_offsets,
                              GM_ADDR prefix1_values,
                              GM_ADDR prefix1_pair_keys,
                              GM_ADDR prefix2_value_offsets,
                              GM_ADDR prefix2_values,
                              GM_ADDR temperatures,
                              GM_ADDR out_tokens,
                              GM_ADDR out_logprobs,
                              const RecConstrainedTopKTilingData*
                                  tiling_data) {
    tiling_data_ = *tiling_data;
    pipe_.InitBuffer(math_buf_, kRecConstrainedTopKMathBufferBytes);
    pipe_.InitBuffer(exp_buf_, kRecConstrainedTopKExpBufferBytes);
    pipe_.InitBuffer(logit_bf16_buf_, kRecConstrainedTopKLoadBf16BufferBytes);
    pipe_.InitBuffer(logit_float_buf_, kRecConstrainedTopKLoadFloatBufferBytes);
    pipe_.InitBuffer(candidate_logit_buf_,
                     static_cast<uint32_t>(tiling_data_.max_candidate_count *
                                           static_cast<int32_t>(sizeof(float))));
    pipe_.InitBuffer(output_token_buf_,
                     kRecConstrainedTopKOutputTokenBufferBytes);
    pipe_.InitBuffer(output_logprob_buf_,
                     kRecConstrainedTopKOutputLogprobBufferBytes);
    logits_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ LogitType*>(logits));
    sequence_group_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(sequence_group));
    first_token_ids_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(first_token_ids));
    prefix1_offsets_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix1_offsets));
    prefix1_values_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix1_values));
    prefix1_pair_keys_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(prefix1_pair_keys));
    prefix2_value_offsets_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix2_value_offsets));
    prefix2_values_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(prefix2_values));
    temperatures_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(temperatures));
    out_tokens_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(out_tokens));
    out_logprobs_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(out_logprobs));
  }

  __aicore__ inline void Process() {
    const int32_t core_id = static_cast<int32_t>(AscendC::GetBlockIdx());
    const int32_t core_num = static_cast<int32_t>(AscendC::GetBlockNum());
    for (int32_t row = core_id; row < tiling_data_.num_rows; row += core_num) {
      if (tiling_data_.debug_mode == 1) {
        WriteDebugRow(row, core_id, core_num);
        continue;
      }
      ProcessRow(row);
    }
  }

 private:
  __aicore__ inline float ReadLogit(
      int32_t row,
      int32_t token,
      int32_t& cached_block_offset,
      AscendC::LocalTensor<bfloat16_t>& cached_logit_bf16,
      AscendC::LocalTensor<float>& cached_logit_float) {
    if constexpr (AscendC::IsSameType<LogitType, bfloat16_t>::value) {
      const int32_t block_offset =
          token / kRecConstrainedTopKLoadBf16Elements *
          kRecConstrainedTopKLoadBf16Elements;
      const int32_t max_block_offset =
          tiling_data_.vocab_size - kRecConstrainedTopKLoadBf16Elements;
      const int32_t safe_block_offset =
          max_block_offset > 0 && block_offset > max_block_offset
              ? max_block_offset
              : block_offset;
      const int32_t local_offset = token - safe_block_offset;
      const int32_t copy_len =
          tiling_data_.vocab_size - safe_block_offset <
                  kRecConstrainedTopKLoadBf16Elements
              ? tiling_data_.vocab_size - safe_block_offset
              : kRecConstrainedTopKLoadBf16Elements;
      if (cached_block_offset != safe_block_offset) {
        cached_block_offset = safe_block_offset;
        AscendC::DataCopy(
            cached_logit_bf16,
            logits_gm_[static_cast<int64_t>(row) * tiling_data_.vocab_size +
                       safe_block_offset],
            copy_len);
        event_t event_mte2_v = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_mte2_v);
        AscendC::Cast(cached_logit_float,
                      cached_logit_bf16,
                      AscendC::RoundMode::CAST_NONE,
                      copy_len);
        AscendC::PipeBarrier<PIPE_V>();
      }
      return cached_logit_float.GetValue(local_offset);
    }
    return static_cast<float>(
        logits_gm_.GetValue(static_cast<int64_t>(row) *
                                tiling_data_.vocab_size +
                            token));
  }

  __aicore__ inline float ReadTemperature(int32_t row) const {
    if (tiling_data_.temperature_count == tiling_data_.num_rows) {
      return temperatures_gm_.GetValue(row);
    }
    return temperatures_gm_.GetValue(0);
  }

  __aicore__ inline float ExpScalar(float value) {
    AscendC::LocalTensor<float> math_tensor = math_buf_.Get<float>();
    math_tensor.SetValue(0, value);
    AscendC::Exp(math_tensor, math_tensor, 1);
    AscendC::PipeBarrier<PIPE_V>();
    return math_tensor.GetValue(0);
  }

  __aicore__ inline float FlushExpBlock(
      AscendC::LocalTensor<float>& exp_values,
      int32_t exp_count) {
    if (exp_count <= 0) {
      return 0.0f;
    }
    AscendC::Exp(exp_values, exp_values, exp_count);
    AscendC::PipeBarrier<PIPE_V>();
    float sum_exp = 0.0f;
    for (int32_t idx = 0; idx < exp_count; ++idx) {
      sum_exp += exp_values.GetValue(idx);
    }
    return sum_exp;
  }

  __aicore__ inline float LnScalar(float value) {
    AscendC::LocalTensor<float> math_tensor = math_buf_.Get<float>();
    math_tensor.SetValue(0, value);
    AscendC::Ln(math_tensor, math_tensor, 1);
    AscendC::PipeBarrier<PIPE_V>();
    return math_tensor.GetValue(0);
  }

  __aicore__ inline void ResolveRange(int32_t row,
                                      int32_t& begin,
                                      int32_t& degree,
                                      int32_t& table_kind) const {
    begin = 0;
    degree = 0;
    table_kind = 0;
    if (tiling_data_.current_step == 0) {
      degree = tiling_data_.first_token_count;
      table_kind = 0;
      return;
    }

    table_kind = tiling_data_.current_step == 1 ? 1 : 2;
    const int32_t sequence_base = row * tiling_data_.sequence_stride;
    const int32_t t0 = sequence_group_gm_.GetValue(sequence_base);
    if (t0 < 0 || t0 + 1 >= tiling_data_.vocab_size) {
      return;
    }

    if (tiling_data_.current_step == 1) {
      begin = prefix1_offsets_gm_.GetValue(t0);
      const int32_t end = prefix1_offsets_gm_.GetValue(t0 + 1);
      degree = end - begin;
      return;
    }

    const int32_t t1 = sequence_group_gm_.GetValue(sequence_base + 1);
    if (t1 < 0 || t1 >= tiling_data_.vocab_size) {
      return;
    }
    const int64_t query_key =
        static_cast<int64_t>(t0) * tiling_data_.vocab_size +
        static_cast<int64_t>(t1);
    int32_t left = 0;
    int32_t right = tiling_data_.prefix1_pair_count;
    while (left < right) {
      const int32_t mid = left + (right - left) / 2;
      const int64_t key = prefix1_pair_keys_gm_.GetValue(mid);
      if (key < query_key) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    if (left >= tiling_data_.prefix1_pair_count ||
        prefix1_pair_keys_gm_.GetValue(left) != query_key) {
      return;
    }
    begin = prefix2_value_offsets_gm_.GetValue(left);
    const int32_t end = prefix2_value_offsets_gm_.GetValue(left + 1);
    degree = end - begin;
  }

  __aicore__ inline int32_t ReadCandidateToken(int32_t table_kind,
                                               int32_t index) const {
    if (table_kind == 0) {
      return first_token_ids_gm_.GetValue(index);
    }
    if (table_kind == 1) {
      return prefix1_values_gm_.GetValue(index);
    }
    return prefix2_values_gm_.GetValue(index);
  }

  __aicore__ inline int32_t DefaultToken(int32_t table_kind) const {
    if (table_kind == 0 && tiling_data_.first_token_count > 0) {
      return first_token_ids_gm_.GetValue(0);
    }
    if (table_kind == 1 && tiling_data_.prefix1_pair_count > 0) {
      return prefix1_values_gm_.GetValue(0);
    }
    if (table_kind == 2 && prefix2_value_offsets_gm_.GetValue(
                                 tiling_data_.prefix1_pair_count) > 0) {
      return prefix2_values_gm_.GetValue(0);
    }
    return 0;
  }

  __aicore__ inline void WriteInvalidRow(int32_t row, int32_t table_kind) {
    const int32_t default_token = DefaultToken(table_kind);
    AscendC::LocalTensor<int32_t> out_tokens =
        output_token_buf_.Get<int32_t>(kRecConstrainedTopKMaxK);
    AscendC::LocalTensor<float> out_logprobs =
        output_logprob_buf_.Get<float>(kRecConstrainedTopKMaxK);
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      out_tokens.SetValue(idx, default_token);
      out_logprobs.SetValue(idx, kRecConstrainedTopKInvalidLogProb);
    }
    FlushOutputRow(row, out_tokens, out_logprobs);
  }

  __aicore__ inline void WriteDebugRow(int32_t row,
                                       int32_t core_id,
                                       int32_t core_num) {
    AscendC::LocalTensor<int32_t> out_tokens =
        output_token_buf_.Get<int32_t>(kRecConstrainedTopKMaxK);
    AscendC::LocalTensor<float> out_logprobs =
        output_logprob_buf_.Get<float>(kRecConstrainedTopKMaxK);
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      out_tokens.SetValue(idx, row + 100);
      out_logprobs.SetValue(idx, static_cast<float>(core_id * 1000 + core_num));
    }
    FlushOutputRow(row, out_tokens, out_logprobs);
  }

  __aicore__ inline void FlushOutputRow(
      int32_t row,
      AscendC::LocalTensor<int32_t>& out_tokens,
      AscendC::LocalTensor<float>& out_logprobs) {
    event_t event_s_mte3 = static_cast<event_t>(
        GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(event_s_mte3);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(event_s_mte3);

    const int32_t output_base = row * tiling_data_.top_k;
    AscendC::DataCopyExtParams copy_params;
    copy_params.blockCount = 1;
    copy_params.blockLen = static_cast<uint32_t>(
        tiling_data_.top_k * static_cast<int32_t>(sizeof(int32_t)));
    copy_params.srcStride = 0;
    copy_params.dstStride = 0;
    copy_params.rsv = 0;
    AscendC::DataCopyPad(out_tokens_gm_[output_base], out_tokens, copy_params);
    AscendC::DataCopyPad(out_logprobs_gm_[output_base], out_logprobs,
                         copy_params);
  }

  __aicore__ inline void MinHeapSiftDown(float* top_values,
                                         int32_t* top_tokens,
                                         int32_t root,
                                         int32_t heap_size) const {
    while (true) {
      const int32_t left = root * 2 + 1;
      if (left >= heap_size) {
        return;
      }
      const int32_t right = left + 1;
      int32_t smallest = left;
      if (right < heap_size && top_values[right] < top_values[left]) {
        smallest = right;
      }
      if (top_values[root] <= top_values[smallest]) {
        return;
      }
      const float value = top_values[root];
      const int32_t token = top_tokens[root];
      top_values[root] = top_values[smallest];
      top_tokens[root] = top_tokens[smallest];
      top_values[smallest] = value;
      top_tokens[smallest] = token;
      root = smallest;
    }
  }

  __aicore__ inline void BuildMinHeap(float* top_values,
                                      int32_t* top_tokens,
                                      int32_t heap_size) const {
    for (int32_t idx = heap_size / 2 - 1; idx >= 0; --idx) {
      MinHeapSiftDown(top_values, top_tokens, idx, heap_size);
    }
  }

  __aicore__ inline void ReplaceMinHeapRoot(float value,
                                            int32_t token,
                                            float* top_values,
                                            int32_t* top_tokens,
                                            int32_t heap_size) const {
    if (value <= top_values[0]) {
      return;
    }
    top_values[0] = value;
    top_tokens[0] = token;
    MinHeapSiftDown(top_values, top_tokens, 0, heap_size);
  }

  __aicore__ inline void SortMinHeapDescending(float* top_values,
                                               int32_t* top_tokens,
                                               int32_t heap_size) const {
    int32_t remaining = heap_size;
    while (remaining > 1) {
      --remaining;
      const float value = top_values[0];
      const int32_t token = top_tokens[0];
      top_values[0] = top_values[remaining];
      top_tokens[0] = top_tokens[remaining];
      top_values[remaining] = value;
      top_tokens[remaining] = token;
      MinHeapSiftDown(top_values, top_tokens, 0, remaining);
    }
  }

  __aicore__ inline void ProcessRow(int32_t row) {
    int32_t begin = 0;
    int32_t degree = 0;
    int32_t table_kind = 0;
    ResolveRange(row, begin, degree, table_kind);
    if (degree <= 0 || tiling_data_.top_k <= 0 ||
        tiling_data_.top_k > kRecConstrainedTopKMaxK) {
      WriteInvalidRow(row, table_kind);
      return;
    }

    float temperature = ReadTemperature(row);
    if (temperature == 0.0f) {
      temperature = 1.0f;
    }

    int32_t cached_block_offset = -1;
    AscendC::LocalTensor<bfloat16_t> cached_logit_bf16 =
        logit_bf16_buf_.Get<bfloat16_t>(kRecConstrainedTopKLoadBf16Elements);
    AscendC::LocalTensor<float> cached_logit_float =
        logit_float_buf_.Get<float>(kRecConstrainedTopKLoadBf16Elements);
    AscendC::LocalTensor<float> candidate_logits =
        candidate_logit_buf_.Get<float>(tiling_data_.max_candidate_count);

    float max_value = -3.4028234663852886e38f;
    for (int32_t idx = 0; idx < degree; ++idx) {
      const int32_t token = ReadCandidateToken(table_kind, begin + idx);
      if (token < 0 || token >= tiling_data_.vocab_size) {
        candidate_logits.SetValue(idx, kRecConstrainedTopKInvalidLogProb);
        continue;
      }
      const float value = ReadLogit(row,
                                    token,
                                    cached_block_offset,
                                    cached_logit_bf16,
                                    cached_logit_float) /
                          temperature;
      candidate_logits.SetValue(idx, value);
      if (value > max_value) {
        max_value = value;
      }
    }

    if (max_value < -3.0e38f) {
      WriteInvalidRow(row, table_kind);
      return;
    }

    float top_values[kRecConstrainedTopKMaxK];
    int32_t top_tokens[kRecConstrainedTopKMaxK];
    const int32_t default_token = DefaultToken(table_kind);
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      top_values[idx] = -3.4028234663852886e38f;
      top_tokens[idx] = default_token;
    }
    int32_t filled_topk_count = 0;

    float sum_exp = 0.0f;
    int32_t exp_count = 0;
    AscendC::LocalTensor<float> exp_values =
        exp_buf_.Get<float>(kRecConstrainedTopKExpBlockElements);
    for (int32_t idx = 0; idx < degree; ++idx) {
      const int32_t token = ReadCandidateToken(table_kind, begin + idx);
      if (token < 0 || token >= tiling_data_.vocab_size) {
        continue;
      }
      const float value = candidate_logits.GetValue(idx);
      exp_values.SetValue(exp_count, value - max_value);
      ++exp_count;
      if (exp_count == kRecConstrainedTopKExpBlockElements) {
        sum_exp += FlushExpBlock(exp_values, exp_count);
        exp_count = 0;
      }
      if (filled_topk_count < tiling_data_.top_k) {
        top_values[filled_topk_count] = value;
        top_tokens[filled_topk_count] = token;
        ++filled_topk_count;
        if (filled_topk_count == tiling_data_.top_k) {
          BuildMinHeap(top_values, top_tokens, filled_topk_count);
        }
      } else {
        ReplaceMinHeapRoot(
            value, token, top_values, top_tokens, filled_topk_count);
      }
    }
    sum_exp += FlushExpBlock(exp_values, exp_count);

    if (sum_exp <= 0.0f) {
      WriteInvalidRow(row, table_kind);
      return;
    }

    if (filled_topk_count == tiling_data_.top_k) {
      SortMinHeapDescending(top_values, top_tokens, filled_topk_count);
    } else {
      BuildMinHeap(top_values, top_tokens, filled_topk_count);
      SortMinHeapDescending(top_values, top_tokens, filled_topk_count);
    }
    const float log_denom = LnScalar(sum_exp) + max_value;
    AscendC::LocalTensor<int32_t> out_tokens =
        output_token_buf_.Get<int32_t>(kRecConstrainedTopKMaxK);
    AscendC::LocalTensor<float> out_logprobs =
        output_logprob_buf_.Get<float>(kRecConstrainedTopKMaxK);
    for (int32_t idx = 0; idx < tiling_data_.top_k; ++idx) {
      if (top_values[idx] < -3.0e38f) {
        out_tokens.SetValue(idx, default_token);
        out_logprobs.SetValue(idx, kRecConstrainedTopKInvalidLogProb);
      } else {
        out_tokens.SetValue(idx, top_tokens[idx]);
        out_logprobs.SetValue(idx, top_values[idx] - log_denom);
      }
    }
    FlushOutputRow(row, out_tokens, out_logprobs);
  }

  RecConstrainedTopKTilingData tiling_data_;
  AscendC::TPipe pipe_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> math_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> exp_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> logit_bf16_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> logit_float_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> candidate_logit_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> output_token_buf_;
  AscendC::TBuf<AscendC::TPosition::VECCALC> output_logprob_buf_;
  AscendC::GlobalTensor<LogitType> logits_gm_;
  AscendC::GlobalTensor<int32_t> sequence_group_gm_;
  AscendC::GlobalTensor<int32_t> first_token_ids_gm_;
  AscendC::GlobalTensor<int32_t> prefix1_offsets_gm_;
  AscendC::GlobalTensor<int32_t> prefix1_values_gm_;
  AscendC::GlobalTensor<int64_t> prefix1_pair_keys_gm_;
  AscendC::GlobalTensor<int32_t> prefix2_value_offsets_gm_;
  AscendC::GlobalTensor<int32_t> prefix2_values_gm_;
  AscendC::GlobalTensor<float> temperatures_gm_;
  AscendC::GlobalTensor<int32_t> out_tokens_gm_;
  AscendC::GlobalTensor<float> out_logprobs_gm_;
};

}  // namespace kernels
