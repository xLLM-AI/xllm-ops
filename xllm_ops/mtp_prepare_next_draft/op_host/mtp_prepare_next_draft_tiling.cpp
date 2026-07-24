/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include "mtp_prepare_next_draft_tiling.h"

#include "register/op_def_registry.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* token_shape = context->GetInputShape(0);
  const gert::StorageShape* embedding_shape = context->GetInputShape(1);
  const gert::StorageShape* block_table_shape = context->GetInputShape(5);
  if (token_shape == nullptr || embedding_shape == nullptr ||
      block_table_shape == nullptr) {
    return ge::GRAPH_FAILED;
  }

  const gert::Shape& tokens = token_shape->GetStorageShape();
  const gert::Shape& embeddings = embedding_shape->GetStorageShape();
  const gert::Shape& block_tables = block_table_shape->GetStorageShape();
  if (tokens.GetDimNum() != 2 || embeddings.GetDimNum() != 3 ||
      block_tables.GetDimNum() != 2) {
    return ge::GRAPH_FAILED;
  }

  const int64_t batch = tokens.GetDim(0);
  const int64_t speculative_width = tokens.GetDim(1);
  const int64_t hidden = embeddings.GetDim(2);
  const int64_t num_blocks = block_tables.GetDim(1);
  const auto* attrs = context->GetAttrs();
  const int64_t* block_size_attr =
      attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(0);
  if (batch <= 0 || speculative_width <= 0 || hidden <= 0 ||
      num_blocks <= 0 || block_size_attr == nullptr ||
      *block_size_attr <= 0) {
    return ge::GRAPH_FAILED;
  }

  MtpPrepareNextDraftTilingData tiling;
  tiling.set_batchSize(static_cast<uint32_t>(batch));
  tiling.set_speculativeWidth(static_cast<uint32_t>(speculative_width));
  tiling.set_hiddenSize(static_cast<uint32_t>(hidden));
  tiling.set_numBlocksPerSequence(static_cast<uint32_t>(num_blocks));
  tiling.set_blockSize(static_cast<int32_t>(*block_size_attr));
  context->SetBlockDim(1);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MtpPrepareNextDraft).Tiling(TilingFunc);

}  // namespace optiling
