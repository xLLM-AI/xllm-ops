/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class MtpPrepareNextDraftKernel {
 public:
  __aicore__ inline MtpPrepareNextDraftKernel() = default;

  __aicore__ inline void Init(
      GM_ADDR accepted_tokens,
      GM_ADDR accepted_embeddings,
      GM_ADDR embedding_placeholder,
      GM_ADDR base_positions,
      GM_ADDR base_kv_seq_lens,
      GM_ADDR block_tables,
      GM_ADDR draft_token_ids,
      GM_ADDR draft_embeddings,
      GM_ADDR draft_positions,
      GM_ADDR draft_kv_seq_lens,
      GM_ADDR draft_cache_slots,
      const MtpPrepareNextDraftTilingData* tiling) {
    tiling_ = *tiling;
    accepted_tokens_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int64_t*>(accepted_tokens));
    accepted_embeddings_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(accepted_embeddings));
    embedding_placeholder_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(embedding_placeholder));
    base_positions_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(base_positions));
    base_kv_seq_lens_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(base_kv_seq_lens));
    block_tables_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(block_tables));
    draft_token_ids_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(draft_token_ids));
    draft_embeddings_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(draft_embeddings));
    draft_positions_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(draft_positions));
    draft_kv_seq_lens_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(draft_kv_seq_lens));
    draft_cache_slots_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(draft_cache_slots));

    const uint32_t embedding_bytes =
        ((tiling_.hiddenSize * sizeof(T) + 31U) / 32U) * 32U;
    pipe_.InitBuffer(embedding_buffer_, embedding_bytes);
  }

  __aicore__ inline void Process() {
    if (GetBlockIdx() != 0) {
      return;
    }

    mte2_to_mte3_event_ =
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3);
    mte3_to_mte2_event_ =
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);

    // Scalar SetValue stores from different AIV cores can target the same
    // 32-byte GM cache line and overwrite adjacent rows.  This preparation is
    // launch-bound and the batch is small, so one core processes every row to
    // keep the five compact outputs deterministic while retaining one fused
    // launch.
    for (row_ = 0; row_ < tiling_.batchSize; ++row_) {
      ProcessRow();
    }

    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(
        mte2_to_mte3_event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(
        mte3_to_mte2_event_);
  }

 private:
  __aicore__ inline void ProcessRow() {

    const uint32_t token_row_offset = row_ * tiling_.speculativeWidth;
    int64_t accepted_length = 0;
    for (uint32_t i = 0; i < tiling_.speculativeWidth; ++i) {
      if (accepted_tokens_gm_.GetValue(token_row_offset + i) >= 0) {
        ++accepted_length;
      }
    }

    const uint32_t last_index =
        accepted_length > 0 ? static_cast<uint32_t>(accepted_length - 1) : 0U;
    const uint32_t previous_index =
        accepted_length > 1 ? static_cast<uint32_t>(accepted_length - 2) : 0U;
    const int64_t last_token =
        accepted_tokens_gm_.GetValue(token_row_offset + last_index);
    const int64_t previous_token =
        accepted_length > 1
            ? accepted_tokens_gm_.GetValue(token_row_offset + previous_index)
            : last_token;

    const uint32_t output_row = row_ * 2U;
    draft_token_ids_gm_.SetValue(output_row,
                                 static_cast<int32_t>(previous_token));
    draft_token_ids_gm_.SetValue(output_row + 1U,
                                 static_cast<int32_t>(last_token));

    const int32_t base_position =
        base_positions_gm_.GetValue(row_) +
        static_cast<int32_t>(accepted_length);
    const int32_t previous_position = base_position - 1;
    draft_positions_gm_.SetValue(output_row, previous_position);
    draft_positions_gm_.SetValue(output_row + 1U, base_position);
    draft_kv_seq_lens_gm_.SetValue(
        row_,
        base_kv_seq_lens_gm_.GetValue(row_) +
            static_cast<int32_t>(accepted_length));
    const int32_t previous_cache_position =
        accepted_length == tiling_.speculativeWidth
            ? previous_position
            : base_position + 1;
    draft_cache_slots_gm_.SetValue(
        output_row, PositionToCacheSlot(previous_cache_position));
    draft_cache_slots_gm_.SetValue(output_row + 1U,
                                   PositionToCacheSlot(base_position));

    const uint32_t last_embedding_offset =
        (token_row_offset + last_index) * tiling_.hiddenSize;
    const uint32_t previous_embedding_offset =
        (token_row_offset + previous_index) * tiling_.hiddenSize;
    if (accepted_length > 1) {
      CopyEmbedding(accepted_embeddings_gm_[previous_embedding_offset],
                    draft_embeddings_gm_[output_row * tiling_.hiddenSize]);
    } else {
      CopyEmbedding(embedding_placeholder_gm_,
                    draft_embeddings_gm_[output_row * tiling_.hiddenSize]);
    }
    CopyEmbedding(accepted_embeddings_gm_[last_embedding_offset],
                  draft_embeddings_gm_[(output_row + 1U) *
                                       tiling_.hiddenSize]);
  }

  __aicore__ inline int32_t PositionToCacheSlot(int32_t position) const {
    // Graph warmup and batch-transition templates can contain placeholder
    // positions before a real KV block has been assigned. Do not pass an
    // invalid slot to reshape_and_cache; steady-state decode positions and
    // block ids are non-negative, so this guard does not alter real requests.
    if (position < 0) {
      return 0;
    }
    const int32_t block_index = position / tiling_.blockSize;
    if (block_index < 0 ||
        block_index >= static_cast<int32_t>(tiling_.numBlocksPerSequence)) {
      return 0;
    }
    const int32_t block_id = block_tables_gm_.GetValue(
        row_ * tiling_.numBlocksPerSequence +
        static_cast<uint32_t>(block_index));
    if (block_id < 0) {
      return 0;
    }
    return block_id * tiling_.blockSize + position % tiling_.blockSize;
  }

  __aicore__ inline void CopyEmbedding(GlobalTensor<T> source,
                                       GlobalTensor<T> destination) {
    LocalTensor<T> local = embedding_buffer_.Get<T>();
    DataCopy(local, source, tiling_.hiddenSize);
    SetFlag<HardEvent::MTE2_MTE3>(mte2_to_mte3_event_);
    WaitFlag<HardEvent::MTE2_MTE3>(mte2_to_mte3_event_);
    DataCopy(destination, local, tiling_.hiddenSize);
    SetFlag<HardEvent::MTE3_MTE2>(mte3_to_mte2_event_);
    WaitFlag<HardEvent::MTE3_MTE2>(mte3_to_mte2_event_);
  }

  MtpPrepareNextDraftTilingData tiling_;
  uint32_t row_ = 0;
  TPipe pipe_;
  TBuf<TPosition::VECCALC> embedding_buffer_;
  TEventID mte2_to_mte3_event_ = 0;
  TEventID mte3_to_mte2_event_ = 0;
  GlobalTensor<int64_t> accepted_tokens_gm_;
  GlobalTensor<T> accepted_embeddings_gm_;
  GlobalTensor<T> embedding_placeholder_gm_;
  GlobalTensor<int32_t> base_positions_gm_;
  GlobalTensor<int32_t> base_kv_seq_lens_gm_;
  GlobalTensor<int32_t> block_tables_gm_;
  GlobalTensor<int32_t> draft_token_ids_gm_;
  GlobalTensor<T> draft_embeddings_gm_;
  GlobalTensor<int32_t> draft_positions_gm_;
  GlobalTensor<int32_t> draft_kv_seq_lens_gm_;
  GlobalTensor<int32_t> draft_cache_slots_gm_;
};
