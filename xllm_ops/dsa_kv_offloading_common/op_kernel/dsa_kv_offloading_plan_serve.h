/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once
#include <algorithm>
#include <cstdint>

#include "kernel_operator.h"

namespace xllm_ops::dsa_kv_offloading {

using namespace AscendC;

constexpr int32_t kMaxTopK = 2176;
constexpr int32_t kPageSize = 128;
constexpr int32_t kKvDim = 512;
// NoPE and RoPE specialize the same DSA KV offloading Plan/Serve algorithm.
constexpr int32_t kSelectionBlockCount = kMaxTopK / kPageSize;
constexpr int32_t kVectorCoreCount = 48;
constexpr int32_t kCopyIndexTileSize = 48;
constexpr int32_t kSortRepeatElements = 32;
constexpr int32_t kInt32Bytes = sizeof(int32_t);
constexpr int32_t kQueryDescriptorFields = 4;
constexpr int32_t kDescriptorCacheRow = 0;
constexpr int32_t kDescriptorCurrentWindowLength = 1;
constexpr int32_t kDescriptorVisibleKvLength = 2;
constexpr int32_t kDescriptorStateTokenRow = 3;

static_assert(kMaxTopK % kPageSize == 0,
              "selection rows must contain complete KV pages");

struct QueryRow {
  int32_t token_row;
  int32_t head_index;
  int32_t cache_row;
  int32_t current_start;
  int32_t visible_kv_length;
  int32_t state_row;
};

template <bool ResolveCacheRow, bool ResolveLengths>
__aicore__ inline QueryRow resolve_query_row(
    int32_t active_row,
    int32_t topk_head_count,
    GlobalTensor<int32_t>& query_row_descriptor) {
  const int32_t token_row = active_row / topk_head_count;
  const int32_t head_index = active_row % topk_head_count;
  const int64_t descriptor_offset =
      static_cast<int64_t>(token_row) * kQueryDescriptorFields;
  int32_t cache_row = 0;
  int32_t current_start = 0;
  int32_t visible_kv_length = 0;
  if constexpr (ResolveCacheRow) {
    cache_row =
        query_row_descriptor.GetValue(descriptor_offset + kDescriptorCacheRow);
  }
  if constexpr (ResolveLengths) {
    const int32_t current_window_length = query_row_descriptor.GetValue(
        descriptor_offset + kDescriptorCurrentWindowLength);
    visible_kv_length = query_row_descriptor.GetValue(
        descriptor_offset + kDescriptorVisibleKvLength);
    current_start = visible_kv_length - current_window_length;
  }
  const int32_t state_token_row = query_row_descriptor.GetValue(
      descriptor_offset + kDescriptorStateTokenRow);
  return QueryRow{/*token_row=*/token_row,
                  /*head_index=*/head_index,
                  /*cache_row=*/cache_row,
                  /*current_start=*/current_start,
                  /*visible_kv_length=*/visible_kv_length,
                  /*state_row=*/state_token_row * topk_head_count + head_index};
}

template <bool ExportPlan,
          bool CompactIndices = false,
          bool PreserveInactiveResident = false>
class SparseKvPlan final {
 public:
  __aicore__ inline SparseKvPlan(TPipe* pipe,
                                 int32_t worker_count,
                                 int32_t query_token_count,
                                 int32_t topk_head_count,
                                 int32_t topk_count)
      : pipe_(pipe),
        worker_count_(worker_count),
        query_token_count_(query_token_count),
        topk_head_count_(topk_head_count),
        topk_count_(topk_count) {}

  __aicore__ inline void init(GM_ADDR query_row_descriptor,
                              GM_ADDR topk_indices,
                              GM_ADDR selection_sparse_indices,
                              GM_ADDR selection_actual_seq,
                              GM_ADDR selection_block_table,
                              GM_ADDR resident_token_ids,
                              GM_ADDR selection_slot_indices,
                              GM_ADDR selection_valid_count,
                              GM_ADDR plan_tokens = nullptr,
                              GM_ADDR plan_slots = nullptr,
                              GM_ADDR compact_indices = nullptr) {
    query_row_descriptor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(query_row_descriptor));
    topk_indices_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(topk_indices));
    selection_sparse_indices_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(selection_sparse_indices));
    selection_actual_seq_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(selection_actual_seq));
    selection_block_table_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(selection_block_table));
    resident_token_ids_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(resident_token_ids));
    selection_slot_indices_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(selection_slot_indices));
    selection_valid_count_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(selection_valid_count));

    if constexpr (ExportPlan) {
      plan_tokens_.SetGlobalBuffer(
          reinterpret_cast<__gm__ int32_t*>(plan_tokens));
      plan_slots_.SetGlobalBuffer(
          reinterpret_cast<__gm__ int32_t*>(plan_slots));
    }
    if constexpr (CompactIndices) {
      static_assert(!ExportPlan);
      compact_indices_.SetGlobalBuffer(
          reinterpret_cast<__gm__ int32_t*>(compact_indices));
    }
    pipe_->InitBuffer(work_buffer_, kWorkBufferBytes);
  }

  __aicore__ inline void process() { process_rows(); }

 private:
  __aicore__ inline void process_rows() {
    const int32_t task_id = static_cast<int32_t>(GetBlockIdx());
    const int32_t active_row_count = query_token_count_ * topk_head_count_;
    for (int32_t active_row = task_id; active_row < active_row_count;
         active_row += worker_count_) {
      process_row(active_row);
    }
  }

  static constexpr int32_t kBufferCount = 14;
  static constexpr int32_t kMetadataValidCountOffset = 0;
  static constexpr int32_t kMetadataBufferBytes = 3 * 32;
  static constexpr int32_t kWorkBufferBytes =
      kBufferCount * kMaxTopK * kInt32Bytes + kMetadataBufferBytes;

  __aicore__ inline void sort_topk(LocalTensor<int32_t> source_tokens,
                                   LocalTensor<uint32_t> source_indices,
                                   LocalTensor<float> sort_temp,
                                   LocalTensor<float> sorted_pairs,
                                   LocalTensor<int32_t> sorted_tokens,
                                   LocalTensor<uint32_t> sorted_indices,
                                   int32_t input_count) {
    const int32_t sort_count =
        ((input_count + kSortRepeatElements - 1) / kSortRepeatElements) *
        kSortRepeatElements;
    LocalTensor<float> source_float = source_tokens.ReinterpretCast<float>();
    Cast(source_float,
         source_tokens,
         RoundMode::CAST_ROUND,
         static_cast<uint32_t>(input_count));
    PipeBarrier<PIPE_V>();

    if (sort_count > input_count) {
      set_wait_flag<HardEvent::V_S>(HardEvent::V_S);
      for (int32_t index = input_count; index < sort_count; ++index) {
        source_float.SetValue(index, -1.0F);
      }
      set_wait_flag<HardEvent::S_V>(HardEvent::S_V);
    }

    LocalTensor<float> concat = source_float;
    Concat(concat,
           source_float,
           sort_temp,
           static_cast<uint32_t>(sort_count / kSortRepeatElements));
    PipeBarrier<PIPE_V>();
    Sort<float, true>(sorted_pairs,
                      concat,
                      source_indices,
                      sort_temp,
                      static_cast<uint32_t>(sort_count / kSortRepeatElements));
    PipeBarrier<PIPE_V>();

    LocalTensor<float> sorted_float = sorted_tokens.ReinterpretCast<float>();
    Extract(sorted_float,
            sorted_indices,
            sorted_pairs,
            static_cast<uint32_t>(sort_count / kSortRepeatElements));
    PipeBarrier<PIPE_V>();
    Cast(sorted_tokens,
         sorted_float,
         RoundMode::CAST_ROUND,
         static_cast<uint32_t>(sort_count));
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void write_selection_block_table(int32_t active_row,
                                                     int32_t state_row,
                                                     LocalTensor<int32_t> row) {
    ArithProgression(
        row, state_row * kSelectionBlockCount, 1, kSelectionBlockCount);
    PipeBarrier<PIPE_V>();
    set_wait_flag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    const DataCopyExtParams copy_params{
        1, static_cast<uint32_t>(kSelectionBlockCount * kInt32Bytes), 0, 0, 0};
    DataCopyPad(selection_block_table_[static_cast<int64_t>(active_row) *
                                       kSelectionBlockCount],
                row,
                copy_params);
  }

  __aicore__ inline void assign_miss_slot(int32_t sorted_index,
                                          int32_t selection_slot,
                                          LocalTensor<uint32_t> sorted_indices,
                                          LocalTensor<int32_t> new_slots,
                                          LocalTensor<int32_t> sparse_indices) {
    const int32_t topk_rank =
        static_cast<int32_t>(sorted_indices.GetValue(sorted_index));
    new_slots.SetValue(sorted_index, selection_slot);

    sparse_indices.SetValue(topk_rank, -selection_slot - 2);
  }

  __aicore__ inline void process_row(int32_t active_row) {
    const QueryRow row = resolve_query_row</*ResolveCacheRow=*/false,
                                           /*ResolveLengths=*/true>(
        active_row, topk_head_count_, query_row_descriptor_);

    // sorted_tokens/indices reuse sort_temp after Sort/Extract finish.
    LocalTensor<int32_t> source_tokens = work_buffer_.Get<int32_t>();
    LocalTensor<int32_t> source_index_storage = source_tokens[kMaxTopK];
    LocalTensor<uint32_t> source_indices =
        source_index_storage.ReinterpretCast<uint32_t>();
    LocalTensor<float> sort_temp =
        source_tokens[kMaxTopK * 2].ReinterpretCast<float>();
    LocalTensor<float> sorted_pairs =
        source_tokens[kMaxTopK * 4].ReinterpretCast<float>();
    LocalTensor<int32_t> sorted_tokens = source_tokens[kMaxTopK * 2];
    LocalTensor<uint32_t> sorted_indices =
        source_tokens[kMaxTopK * 3].ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> old_tokens = source_tokens[kMaxTopK * 8];
    LocalTensor<int32_t> old_slots = source_tokens[kMaxTopK * 9];
    LocalTensor<int32_t> slot_workspace = source_tokens[kMaxTopK * 10];
    LocalTensor<int32_t> new_slots = source_tokens[kMaxTopK * 11];
    LocalTensor<int32_t> sparse_indices = source_tokens[kMaxTopK * 12];
    LocalTensor<int32_t> selection_block_row = source_tokens[kMaxTopK * 13];
    LocalTensor<int32_t> metadata = source_tokens[kMaxTopK * kBufferCount];

    if constexpr (PreserveInactiveResident) {
      if (row.visible_kv_length == 0) {
        // Padding can alias a live stable state row. Publish empty query
        // outputs without reading or updating that row's persistent state.
        Duplicate(sparse_indices, -1, kMaxTopK);
        Duplicate(metadata, 0, 8);
        PipeBarrier<PIPE_V>();
        set_wait_flag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        const int64_t topk_offset =
            static_cast<int64_t>(active_row) * topk_count_;
        const DataCopyExtParams empty_indices_params{
            1, static_cast<uint32_t>(topk_count_ * kInt32Bytes), 0, 0, 0};
        const DataCopyExtParams empty_length_params{
            1, static_cast<uint32_t>(kInt32Bytes), 0, 0, 0};
        DataCopyPad(selection_sparse_indices_[topk_offset],
                    sparse_indices,
                    empty_indices_params);
        if constexpr (CompactIndices) {
          DataCopyPad(compact_indices_[topk_offset],
                      sparse_indices,
                      empty_indices_params);
        }
        if constexpr (ExportPlan) {
          DataCopyPad(
              plan_tokens_[topk_offset], sparse_indices, empty_indices_params);
          DataCopyPad(
              plan_slots_[topk_offset], sparse_indices, empty_indices_params);
        }
        DataCopyPad(
            selection_actual_seq_[active_row], metadata, empty_length_params);
        write_selection_block_table(
            active_row, row.state_row, selection_block_row);
        set_wait_flag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        return;
      }
    }

    const int32_t old_count = selection_valid_count_.GetValue(row.state_row);
    Duplicate(new_slots, -1, kMaxTopK);
    Duplicate(sparse_indices, -1, kMaxTopK);
    ArithProgression(source_index_storage, 0, 1, kMaxTopK);
    PipeBarrier<PIPE_V>();

    const int64_t topk_offset = static_cast<int64_t>(active_row) * topk_count_;
    const DataCopyPadExtParams<int32_t> int_pad_params{false, 0, 0, 0};
    const DataCopyExtParams topk_copy_params{
        1, static_cast<uint32_t>(topk_count_ * kInt32Bytes), 0, 0, 0};
    DataCopyPad(source_tokens,
                topk_indices_[topk_offset],
                topk_copy_params,
                int_pad_params);

    if (old_count > 0) {
      const DataCopyExtParams old_copy_params{
          1, static_cast<uint32_t>(old_count * kInt32Bytes), 0, 0, 0};
      const int64_t state_offset =
          static_cast<int64_t>(row.state_row) * kMaxTopK;
      DataCopyPad(old_tokens,
                  resident_token_ids_[state_offset],
                  old_copy_params,
                  int_pad_params);
      DataCopyPad(old_slots,
                  selection_slot_indices_[state_offset],
                  old_copy_params,
                  int_pad_params);
    }
    set_wait_flag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

    // The kPool tail occupies the last three positions of the 2051-wide
    // indexer output, after potentially invalid scored positions. Scan every
    // physical position instead of taking the first min(KV length, top-k).
    int32_t valid_count = 0;
    if constexpr (CompactIndices) {
      // NoPE uses the same FP32 token range already required by Sort. Reuse
      // sort scratch before Sort starts; keep original top-k ranks unchanged.
      LocalTensor<float> tokens_float = sort_temp;
      LocalTensor<uint8_t> positive_mask =
          sorted_pairs.ReinterpretCast<uint8_t>();
      LocalTensor<uint8_t> visible_mask = positive_mask[288];
      LocalTensor<uint8_t> valid_mask = positive_mask[576];
      LocalTensor<float> count_scratch = sorted_pairs[kMaxTopK];
      LocalTensor<float> invalid_values = sorted_pairs[kMaxTopK * 2];
      set_wait_flag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
      Cast(tokens_float, source_tokens, RoundMode::CAST_ROUND, topk_count_);
      Duplicate(invalid_values, -1.0F, topk_count_);
      PipeBarrier<PIPE_V>();
      CompareScalar(
          positive_mask, tokens_float, 0.0F, CMPMODE::GE, topk_count_);
      CompareScalar(visible_mask,
                    tokens_float,
                    static_cast<float>(row.visible_kv_length),
                    CMPMODE::LT,
                    topk_count_);
      PipeBarrier<PIPE_V>();
      And(valid_mask.ReinterpretCast<uint16_t>(),
          positive_mask.ReinterpretCast<uint16_t>(),
          visible_mask.ReinterpretCast<uint16_t>(),
          144);
      PipeBarrier<PIPE_V>();
      uint64_t gathered_count = 0;
      GatherMaskParams gather_params;
      gather_params.repeatTimes = 1;
      gather_params.src0BlockStride = 1;
      gather_params.src0RepeatStride = 8;
      gather_params.src1RepeatStride = 8;
      GatherMask(count_scratch,
                 tokens_float,
                 valid_mask.ReinterpretCast<uint32_t>(),
                 true,
                 static_cast<uint32_t>(topk_count_),
                 gather_params,
                 gathered_count);
      PipeBarrier<PIPE_V>();
      Select(tokens_float,
             valid_mask,
             tokens_float,
             invalid_values,
             SELMODE::VSEL_TENSOR_TENSOR_MODE,
             static_cast<uint32_t>(topk_count_));
      PipeBarrier<PIPE_V>();
      Cast(source_tokens, tokens_float, RoundMode::CAST_ROUND, topk_count_);
      PipeBarrier<PIPE_V>();
      valid_count = static_cast<int32_t>(gathered_count);

    } else {
      for (int32_t rank = 0; rank < topk_count_; ++rank) {
        const int32_t token = source_tokens.GetValue(rank);
        if (token >= 0 && token < row.visible_kv_length) {
          ++valid_count;
        } else {
          source_tokens.SetValue(rank, -1);
        }
      }
      set_wait_flag<HardEvent::S_V>(HardEvent::S_V);
    }
    if (old_count > 0 && old_count != valid_count) {
      Duplicate(slot_workspace, 0, kMaxTopK);
    }
    if (valid_count > 0) {
      sort_topk(source_tokens,
                source_indices,
                sort_temp,
                sorted_pairs,
                sorted_tokens,
                sorted_indices,
                topk_count_);
    }
    set_wait_flag<HardEvent::V_S>(HardEvent::V_S);

    // Serve consumes <= -2 as a miss slot (-value - 2).
    if (old_count == 0) {
      for (int32_t sorted_index = 0; sorted_index < valid_count;
           ++sorted_index) {
        assign_miss_slot(sorted_index,
                         sorted_index,
                         sorted_indices,
                         new_slots,
                         sparse_indices);
      }
    } else if (old_count == valid_count) {
      int32_t old_cursor = 0;
      int32_t sorted_cursor = 0;
      // Positive balance stores free slots; negative stores sorted indices
      // waiting for a slot. Both lists use slot_workspace as a stack.
      int32_t pending_balance = 0;
      int32_t old_token = old_tokens.GetValue(0);
      int32_t sorted_token = sorted_tokens.GetValue(0);
      while (old_cursor < old_count && sorted_cursor < valid_count) {
        if (old_token > sorted_token) {
          const int32_t selection_slot = old_slots.GetValue(old_cursor);
          if (pending_balance < 0) {
            const int32_t pending =
                slot_workspace.GetValue(-pending_balance - 1);
            const int32_t sorted_index = pending;
            assign_miss_slot(sorted_index,
                             selection_slot,
                             sorted_indices,
                             new_slots,
                             sparse_indices);
            ++pending_balance;
          } else {
            slot_workspace.SetValue(pending_balance, selection_slot);
            ++pending_balance;
          }
          ++old_cursor;
          if (old_cursor < old_count) {
            old_token = old_tokens.GetValue(old_cursor);
          }
          continue;
        }
        if (old_token < sorted_token) {
          if (pending_balance > 0) {
            const int32_t selection_slot =
                slot_workspace.GetValue(pending_balance - 1);
            assign_miss_slot(sorted_cursor,
                             selection_slot,
                             sorted_indices,
                             new_slots,
                             sparse_indices);
            --pending_balance;
          } else {
            slot_workspace.SetValue(-pending_balance, sorted_cursor);
            --pending_balance;
          }
          ++sorted_cursor;
          if (sorted_cursor < valid_count) {
            sorted_token = sorted_tokens.GetValue(sorted_cursor);
          }
          continue;
        }

        const int32_t selection_slot = old_slots.GetValue(old_cursor);
        if (sorted_token < row.current_start) {
          const int32_t topk_rank =
              static_cast<int32_t>(sorted_indices.GetValue(sorted_cursor));
          new_slots.SetValue(sorted_cursor, selection_slot);
          sparse_indices.SetValue(topk_rank, selection_slot);
        } else {
          assign_miss_slot(sorted_cursor,
                           selection_slot,
                           sorted_indices,
                           new_slots,
                           sparse_indices);
        }
        ++old_cursor;
        ++sorted_cursor;
        if (old_cursor < old_count) {
          old_token = old_tokens.GetValue(old_cursor);
        }
        if (sorted_cursor < valid_count) {
          sorted_token = sorted_tokens.GetValue(sorted_cursor);
        }
      }
      while (old_cursor < old_count) {
        const int32_t selection_slot = old_slots.GetValue(old_cursor);
        if (pending_balance < 0) {
          const int32_t pending = slot_workspace.GetValue(-pending_balance - 1);
          const int32_t sorted_index = pending;
          assign_miss_slot(sorted_index,
                           selection_slot,
                           sorted_indices,
                           new_slots,
                           sparse_indices);
          ++pending_balance;
        } else {
          slot_workspace.SetValue(pending_balance, selection_slot);
          ++pending_balance;
        }
        ++old_cursor;
      }
      while (sorted_cursor < valid_count) {
        if (pending_balance > 0) {
          const int32_t selection_slot =
              slot_workspace.GetValue(pending_balance - 1);
          assign_miss_slot(sorted_cursor,
                           selection_slot,
                           sorted_indices,
                           new_slots,
                           sparse_indices);
          --pending_balance;
        } else {
          slot_workspace.SetValue(-pending_balance, sorted_cursor);
          --pending_balance;
        }
        ++sorted_cursor;
      }
    } else {
      int32_t old_cursor = 0;
      for (int32_t sorted_index = 0; sorted_index < valid_count;
           ++sorted_index) {
        const int32_t token_id = sorted_tokens.GetValue(sorted_index);
        while (old_cursor < old_count &&
               old_tokens.GetValue(old_cursor) > token_id) {
          ++old_cursor;
        }
        if (old_cursor < old_count &&
            old_tokens.GetValue(old_cursor) == token_id &&
            token_id < row.current_start) {
          const int32_t selection_slot = old_slots.GetValue(old_cursor);
          // A shorter effective top-k cannot address a resident slot beyond
          // its new KV length. Reload just these former hits into a compact
          // slot instead of reusing an out-of-range selection position.
          if (selection_slot >= 0 && selection_slot < valid_count) {
            const int32_t topk_rank =
                static_cast<int32_t>(sorted_indices.GetValue(sorted_index));
            new_slots.SetValue(sorted_index, selection_slot);
            slot_workspace.SetValue(selection_slot, 1);
            sparse_indices.SetValue(topk_rank, selection_slot);
            ++old_cursor;
          }
        }
      }

      int32_t free_slot = 0;
      for (int32_t sorted_index = 0; sorted_index < valid_count;
           ++sorted_index) {
        if (new_slots.GetValue(sorted_index) >= 0) {
          continue;
        }
        while (slot_workspace.GetValue(free_slot) != 0) {
          ++free_slot;
        }
        assign_miss_slot(
            sorted_index, free_slot, sorted_indices, new_slots, sparse_indices);
        slot_workspace.SetValue(free_slot, 1);
        ++free_slot;
      }
    }

    metadata.SetValue(kMetadataValidCountOffset, valid_count);

    set_wait_flag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    const int64_t state_offset = static_cast<int64_t>(row.state_row) * kMaxTopK;
    if (valid_count > 0) {
      const DataCopyExtParams state_copy_params{
          1, static_cast<uint32_t>(valid_count * kInt32Bytes), 0, 0, 0};
      DataCopyPad(
          resident_token_ids_[state_offset], sorted_tokens, state_copy_params);
      DataCopyPad(
          selection_slot_indices_[state_offset], new_slots, state_copy_params);
    }
    if constexpr (ExportPlan) {
      // Independent immutable plan. Serve never clears the reload encoding.
      DataCopyPad(plan_slots_[topk_offset], sparse_indices, topk_copy_params);
      DataCopyPad(source_tokens,
                  topk_indices_[topk_offset],
                  topk_copy_params,
                  int_pad_params);
      set_wait_flag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
      DataCopyPad(plan_tokens_[topk_offset], source_tokens, topk_copy_params);
      set_wait_flag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
      for (int32_t rank = 0; rank < topk_count_; ++rank) {
        const int32_t value = sparse_indices.GetValue(rank);
        if (value < -1) sparse_indices.SetValue(rank, -value - 2);
      }
      // The immutable miss plan is already published. Order the scalar
      // decoding before MTE3 publishes the separate attention indices.
      set_wait_flag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    }
    DataCopyPad(selection_sparse_indices_[topk_offset],
                sparse_indices,
                topk_copy_params);
    if constexpr (CompactIndices) {
      // Keep the original positional miss encoding for Serve. Compact only
      // the independent attention output, preserving duplicate multiplicity
      // and original top-k order. All reused sort scratch is now published.
      set_wait_flag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
      set_wait_flag<HardEvent::S_V>(HardEvent::S_V);
      LocalTensor<float> encoded = source_tokens.ReinterpretCast<float>();
      LocalTensor<float> compact =
          source_index_storage.ReinterpretCast<float>();
      LocalTensor<float> decoded_misses = sort_temp;
      LocalTensor<uint8_t> valid_mask = sorted_pairs.ReinterpretCast<uint8_t>();
      LocalTensor<uint8_t> miss_mask = valid_mask[288];
      Cast(encoded, sparse_indices, RoundMode::CAST_ROUND, topk_count_);
      Duplicate(compact, -1.0F, topk_count_);
      PipeBarrier<PIPE_V>();
      CompareScalar(valid_mask, encoded, -1.0F, CMPMODE::NE, topk_count_);
      CompareScalar(miss_mask, encoded, -2.0F, CMPMODE::LE, topk_count_);
      Muls(decoded_misses, encoded, -1.0F, topk_count_);
      PipeBarrier<PIPE_V>();
      Adds(decoded_misses, decoded_misses, -2.0F, topk_count_);
      PipeBarrier<PIPE_V>();
      Select(encoded,
             miss_mask,
             decoded_misses,
             encoded,
             SELMODE::VSEL_TENSOR_TENSOR_MODE,
             topk_count_);
      PipeBarrier<PIPE_V>();
      uint64_t compact_count = 0;
      GatherMaskParams gather_params;
      gather_params.repeatTimes = 1;
      gather_params.src0BlockStride = 1;
      gather_params.src0RepeatStride = 8;
      gather_params.src1RepeatStride = 8;
      GatherMask(compact,
                 encoded,
                 valid_mask.ReinterpretCast<uint32_t>(),
                 true,
                 static_cast<uint32_t>(topk_count_),
                 gather_params,
                 compact_count);
      PipeBarrier<PIPE_V>();
      Cast(source_index_storage, compact, RoundMode::CAST_ROUND, topk_count_);
      PipeBarrier<PIPE_V>();
      set_wait_flag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
      DataCopyPad(compact_indices_[topk_offset],
                  source_index_storage,
                  topk_copy_params);
    }
    const DataCopyExtParams metadata_copy_params{
        1, static_cast<uint32_t>(kInt32Bytes), 0, 0, 0};
    DataCopyPad(selection_valid_count_[row.state_row],
                metadata[kMetadataValidCountOffset],
                metadata_copy_params);
    DataCopyPad(selection_actual_seq_[active_row],
                metadata[kMetadataValidCountOffset],
                metadata_copy_params);

    write_selection_block_table(active_row, row.state_row, selection_block_row);
    set_wait_flag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    if constexpr (CompactIndices) {
      // A core may process another query row. The compact output reuses the
      // next row's MTE2 input and scalar sort scratch; order both consumers
      // before reloading or rewriting either range.
      set_wait_flag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
      set_wait_flag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }
  }

  template <HardEvent Event>
  __aicore__ inline void set_wait_flag(HardEvent event) {
    const event_t event_id =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<Event>(event_id);
    AscendC::WaitFlag<Event>(event_id);
  }

  TPipe* pipe_;
  int32_t worker_count_;
  int32_t query_token_count_;
  int32_t topk_head_count_;
  int32_t topk_count_;

  GlobalTensor<int32_t> query_row_descriptor_;
  GlobalTensor<int32_t> topk_indices_;
  GlobalTensor<int32_t> selection_sparse_indices_;
  GlobalTensor<int32_t> selection_actual_seq_;
  GlobalTensor<int32_t> selection_block_table_;
  GlobalTensor<int32_t> resident_token_ids_;
  GlobalTensor<int32_t> selection_slot_indices_;
  GlobalTensor<int32_t> selection_valid_count_;

  GlobalTensor<int32_t> compact_indices_;
  GlobalTensor<int32_t> plan_tokens_;
  GlobalTensor<int32_t> plan_slots_;
  TBuf<QuePosition::VECCALC> work_buffer_;
};

template <typename T, int32_t kRopeDim, bool ImmutablePlan>
class SparseKvServe final {
 public:
  __aicore__ inline SparseKvServe(TPipe* pipe,
                                  int32_t worker_count,
                                  int32_t query_token_count,
                                  int32_t topk_head_count,
                                  int32_t topk_count,
                                  int32_t current_row_map_stride,
                                  int32_t full_block_table_stride)
      : pipe_(pipe),
        worker_count_(worker_count),
        query_token_count_(query_token_count),
        topk_head_count_(topk_head_count),
        topk_count_(topk_count),
        current_row_map_stride_(current_row_map_stride),
        full_block_table_stride_(full_block_table_stride) {}

  __aicore__ inline void init(GM_ADDR full_kv_cache,
                              GM_ADDR full_block_table,
                              GM_ADDR query_row_descriptor,
                              GM_ADDR current_row_map,
                              GM_ADDR topk_indices,
                              GM_ADDR slot_mapping,
                              GM_ADDR current_kv,
                              GM_ADDR selection_kv_cache,
                              GM_ADDR selection_sparse_indices,
                              GM_ADDR full_k_rope = nullptr,
                              GM_ADDR current_k_rope = nullptr,
                              GM_ADDR selection_k_rope = nullptr) {
    full_kv_cache_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(full_kv_cache));

    full_block_table_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(full_block_table));
    query_row_descriptor_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(query_row_descriptor));
    current_row_map_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(current_row_map));
    topk_indices_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(topk_indices));
    slot_mapping_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(slot_mapping));
    current_kv_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(current_kv));

    selection_kv_cache_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(selection_kv_cache));

    selection_sparse_indices_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(selection_sparse_indices));

    if constexpr (kRopeDim > 0) {
      full_k_rope_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(full_k_rope));
      current_k_rope_.SetGlobalBuffer(
          reinterpret_cast<__gm__ T*>(current_k_rope));
      selection_k_rope_.SetGlobalBuffer(
          reinterpret_cast<__gm__ T*>(selection_k_rope));
    }
    pipe_->InitBuffer(kv_queue_, 2, (kKvDim + kRopeDim) * sizeof(T));
    pipe_->InitBuffer(index_queue_, 1, kCopyIndexTileSize * kInt32Bytes);
  }

  __aicore__ inline void process() {
    const int32_t task_id = static_cast<int32_t>(GetBlockIdx());
    for (int32_t token_row = task_id; token_row < query_token_count_;
         token_row += worker_count_) {
      copy_current_to_host(token_row);
    }

    const int32_t active_row_count = query_token_count_ * topk_head_count_;
    const int64_t work_item_count =
        static_cast<int64_t>(active_row_count) * topk_count_;
    const int64_t tile_count =
        (work_item_count + kCopyIndexTileSize - 1) / kCopyIndexTileSize;
    int32_t cached_active_row = -1;
    int32_t cache_row = 0;
    int32_t current_start = 0;
    int32_t visible_kv_length = 0;
    int32_t state_row = 0;
    for (int64_t tile_index = task_id; tile_index < tile_count;
         tile_index += worker_count_) {
      const int64_t tile_start = tile_index * kCopyIndexTileSize;
      const int64_t remaining_item_count = work_item_count - tile_start;
      const int32_t tile_size = static_cast<int32_t>(
          remaining_item_count < kCopyIndexTileSize ? remaining_item_count
                                                    : kCopyIndexTileSize);
      LocalTensor<int32_t> sparse_indices = index_queue_.AllocTensor<int32_t>();
      const DataCopyPadExtParams<int32_t> pad_params{false, 0, 0, 0};
      const DataCopyExtParams copy_params{
          1, static_cast<uint32_t>(tile_size * kInt32Bytes), 0, 0, 0};
      DataCopyPad(sparse_indices,
                  selection_sparse_indices_[tile_start],
                  copy_params,
                  pad_params);
      index_queue_.EnQue(sparse_indices);  // codespell:ignore enque
      sparse_indices = index_queue_.DeQue<int32_t>();
      set_wait_flag<HardEvent::MTE2_S>(HardEvent::MTE2_S);

      bool has_miss = false;
      for (int32_t tile_offset = 0; tile_offset < tile_size; ++tile_offset) {
        const int32_t pending_slot = sparse_indices.GetValue(tile_offset);
        if (pending_slot >= -1) {
          continue;
        }
        has_miss = true;
        const int64_t work_index = tile_start + tile_offset;
        const int32_t active_row =
            static_cast<int32_t>(work_index / topk_count_);
        if (active_row != cached_active_row) {
          const QueryRow row = resolve_query_row</*ResolveCacheRow=*/true,
                                                 /*ResolveLengths=*/true>(
              active_row, topk_head_count_, query_row_descriptor_);
          cache_row = row.cache_row;
          current_start = row.current_start;
          visible_kv_length = row.visible_kv_length;
          state_row = row.state_row;
          cached_active_row = active_row;
        }
        const int32_t selection_slot = -pending_slot - 2;
        const int32_t logical_token = topk_indices_.GetValue(work_index);
        copy_miss(logical_token,
                  selection_slot,
                  active_row / topk_head_count_,
                  cache_row,
                  current_start,
                  visible_kv_length,
                  state_row);
        if constexpr (!ImmutablePlan)
          sparse_indices.SetValue(tile_offset, selection_slot);
      }

      if (!has_miss) {
        index_queue_.FreeTensor(sparse_indices);
        continue;
      }
      if constexpr (!ImmutablePlan) {
        set_wait_flag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(
            selection_sparse_indices_[tile_start], sparse_indices, copy_params);
      }
      index_queue_.FreeTensor(sparse_indices);
    }
  }

 private:
  __aicore__ inline void copy_current_to_host(int32_t token_row) {
    const int32_t append_slot = slot_mapping_.GetValue(token_row);
    if (append_slot < 0) {
      return;
    }

    LocalTensor<T> current = kv_queue_.AllocTensor<T>();
    const DataCopyPadExtParams<T> pad_params{false, 0, 0, 0};
    const DataCopyExtParams kv_params{
        1, static_cast<uint32_t>(kKvDim * sizeof(T)), 0, 0, 0};
    DataCopyPad(current,
                current_kv_[static_cast<int64_t>(token_row) * kKvDim],
                kv_params,
                pad_params);

    if constexpr (kRopeDim > 0) {
      const DataCopyExtParams rope_params{1, kRopeDim * sizeof(T), 0, 0, 0};
      DataCopyPad(current[kKvDim],
                  current_k_rope_[static_cast<int64_t>(token_row) * kRopeDim],
                  rope_params,
                  pad_params);
    }
    kv_queue_.EnQue(current);  // codespell:ignore enque
    current = kv_queue_.DeQue<T>();
    DataCopyPad(full_kv_cache_[static_cast<int64_t>(append_slot) * kKvDim],
                current,
                kv_params);

    if constexpr (kRopeDim > 0) {
      const DataCopyExtParams rope_params{1, kRopeDim * sizeof(T), 0, 0, 0};
      DataCopyPad(full_k_rope_[static_cast<int64_t>(append_slot) * kRopeDim],
                  current[kKvDim],
                  rope_params);
    }
    kv_queue_.FreeTensor(current);
  }

  __aicore__ inline void copy_miss(int32_t logical_token,
                                   int32_t selection_slot,
                                   int32_t token_row,
                                   int32_t cache_row,
                                   int32_t current_start,
                                   int32_t visible_kv_length,
                                   int32_t state_row) {
    LocalTensor<T> data = kv_queue_.AllocTensor<T>();
    const DataCopyPadExtParams<T> pad_params{false, 0, 0, 0};
    const DataCopyExtParams kv_params{
        1, static_cast<uint32_t>(kKvDim * sizeof(T)), 0, 0, 0};

    const bool use_current =
        logical_token >= current_start && logical_token < visible_kv_length;
    const int32_t current_offset = logical_token - current_start;
    const int32_t current_token_row =
        use_current
            ? current_row_map_.GetValue(static_cast<int64_t>(token_row) *
                                            current_row_map_stride_ +
                                        current_offset)
            : -1;
    // Current KV stays readable in HBM while other cores append it to Host.
    // Read it directly here without waiting for those Host writes.
    if (use_current) {
      DataCopyPad(data,
                  current_kv_[static_cast<int64_t>(current_token_row) * kKvDim],
                  kv_params,
                  pad_params);
      if constexpr (kRopeDim > 0) {
        const DataCopyExtParams rope_params{1, kRopeDim * sizeof(T), 0, 0, 0};
        DataCopyPad(
            data[kKvDim],
            current_k_rope_[static_cast<int64_t>(current_token_row) * kRopeDim],
            rope_params,
            pad_params);
      }
    } else {
      const int32_t logical_block = logical_token / kPageSize;
      const int32_t token_in_block = logical_token % kPageSize;
      const int32_t physical_block = full_block_table_.GetValue(
          static_cast<int64_t>(cache_row) * full_block_table_stride_ +
          logical_block);
      const int64_t physical_token =
          static_cast<int64_t>(physical_block) * kPageSize + token_in_block;
      DataCopyPad(
          data, full_kv_cache_[physical_token * kKvDim], kv_params, pad_params);
      if constexpr (kRopeDim > 0) {
        const DataCopyExtParams rope_params{1, kRopeDim * sizeof(T), 0, 0, 0};
        DataCopyPad(data[kKvDim],
                    full_k_rope_[physical_token * kRopeDim],
                    rope_params,
                    pad_params);
      }
    }
    kv_queue_.EnQue(data);  // codespell:ignore enque
    data = kv_queue_.DeQue<T>();

    const int64_t physical_selection_slot =
        static_cast<int64_t>(state_row) * kMaxTopK + selection_slot;
    DataCopyPad(
        selection_kv_cache_[physical_selection_slot * kKvDim], data, kv_params);

    if constexpr (kRopeDim > 0) {
      const DataCopyExtParams rope_params{1, kRopeDim * sizeof(T), 0, 0, 0};
      DataCopyPad(selection_k_rope_[physical_selection_slot * kRopeDim],
                  data[kKvDim],
                  rope_params);
    }
    kv_queue_.FreeTensor(data);
  }

  template <HardEvent Event>
  __aicore__ inline void set_wait_flag(HardEvent event) {
    const event_t event_id =
        static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<Event>(event_id);
    AscendC::WaitFlag<Event>(event_id);
  }

  TPipe* pipe_;
  int32_t worker_count_;
  int32_t query_token_count_;
  int32_t topk_head_count_;
  int32_t topk_count_;
  int32_t current_row_map_stride_;
  int32_t full_block_table_stride_;

  GlobalTensor<T> full_k_rope_;
  GlobalTensor<T> current_k_rope_;
  GlobalTensor<T> selection_k_rope_;
  GlobalTensor<T> full_kv_cache_;
  GlobalTensor<int32_t> full_block_table_;
  GlobalTensor<int32_t> query_row_descriptor_;
  GlobalTensor<int32_t> current_row_map_;
  GlobalTensor<int32_t> topk_indices_;
  GlobalTensor<int32_t> slot_mapping_;
  GlobalTensor<T> current_kv_;
  GlobalTensor<T> selection_kv_cache_;
  GlobalTensor<int32_t> selection_sparse_indices_;

  TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> kv_queue_;
  TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> index_queue_;
};

}  // namespace xllm_ops::dsa_kv_offloading
