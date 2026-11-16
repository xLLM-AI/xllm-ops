/* Copyright 2026 The xLLM Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied.
*/
#pragma once

#include "dsa_kv_offloading_validate.h"

namespace xllm_ops::dsa_kv_offloading {
inline int64_t StableSelectionRows(const aclTensor* selection) {
  if (!Tensor(selection, op::DataType::DT_BF16)) {
    return 0;
  }
  const auto& shape = selection->GetViewShape();
  if (shape.GetDimNum() != 4 || shape.GetDim(0) <= 0 ||
      shape.GetDim(0) % 17 != 0 ||
      shape.GetDim(0) / 17 > std::numeric_limits<int32_t>::max() / 2176 ||
      !Shape(selection, {shape.GetDim(0), 128, 1, 512})) {
    return 0;
  }
  return shape.GetDim(0) / 17;
}

inline bool StableCommon(aclTensor* host,
                         const aclTensor* table,
                         const aclTensor* descriptor,
                         const aclTensor* map,
                         const aclTensor* tokens,
                         const aclTensor* append,
                         const aclTensor* current,
                         const aclTensor* encoded,
                         aclTensor* selection) {
  for (const aclTensor* tensor :
       {host, const_cast<aclTensor*>(current), selection}) {
    if (!Tensor(tensor, op::DataType::DT_BF16)) {
      return false;
    }
  }
  for (const aclTensor* tensor :
       {table, descriptor, map, tokens, append, encoded}) {
    if (!Tensor(tensor, op::DataType::DT_INT32)) {
      return false;
    }
  }
  const auto& topk_shape = tokens->GetViewShape();
  if (topk_shape.GetDimNum() != 3 || topk_shape.GetDim(0) <= 0 ||
      topk_shape.GetDim(1) != 1 || topk_shape.GetDim(2) != 2176) {
    return false;
  }
  const int64_t rows = topk_shape.GetDim(0);
  const int64_t capacity = StableSelectionRows(selection);
  if (rows > capacity) {
    return false;
  }
  const auto& host_shape = host->GetViewShape();
  const auto& block_table = table->GetViewShape();
  const auto& row_map = map->GetViewShape();
  if (host_shape.GetDimNum() != 4 || host_shape.GetDim(0) <= 0 ||
      host_shape.GetDim(0) > std::numeric_limits<int32_t>::max() / 128 ||
      !Shape(host, {host_shape.GetDim(0), 128, 1, 512})) {
    return false;
  }
  if (block_table.GetDimNum() != 2 || block_table.GetDim(0) <= 0 ||
      block_table.GetDim(1) <= 0 ||
      block_table.GetDim(1) > std::numeric_limits<int32_t>::max()) {
    return false;
  }
  if (row_map.GetDimNum() != 2 || row_map.GetDim(1) <= 0 ||
      row_map.GetDim(1) > rows || !Shape(map, {rows, row_map.GetDim(1)})) {
    return false;
  }
  return Shape(descriptor, {rows, 4}) && Shape(current, {rows, 1, 512}) &&
         Shape(append, {rows}) && Shape(encoded, {rows, 1, 2176});
}

inline bool StableState(const aclTensor* tokens,
                        const aclTensor* selection,
                        const aclTensor* resident_tokens,
                        const aclTensor* resident_slots,
                        const aclTensor* resident_count,
                        const aclTensor* indices,
                        const aclTensor* lengths,
                        const aclTensor* table) {
  for (const aclTensor* tensor : {tokens,
                                  resident_tokens,
                                  resident_slots,
                                  resident_count,
                                  indices,
                                  lengths,
                                  table}) {
    if (!Tensor(tensor, op::DataType::DT_INT32)) {
      return false;
    }
  }
  const auto& topk_shape = tokens->GetViewShape();
  if (topk_shape.GetDimNum() != 3 || topk_shape.GetDim(0) <= 0 ||
      topk_shape.GetDim(1) != 1 || topk_shape.GetDim(2) != 2176) {
    return false;
  }
  const int64_t rows = topk_shape.GetDim(0);
  const int64_t capacity = StableSelectionRows(selection);
  if (capacity < rows) {
    return false;
  }
  return Shape(resident_tokens, {capacity, 1, 2176}) &&
         Shape(resident_slots, {capacity, 1, 2176}) &&
         Shape(resident_count, {capacity}) && Shape(indices, {rows, 1, 2176}) &&
         Shape(lengths, {rows}) && Shape(table, {rows, 17});
}

inline bool StableRope(aclTensor* host,
                       const aclTensor* current,
                       aclTensor* selection,
                       const aclTensor* latent,
                       const aclTensor* tokens,
                       const aclTensor* latent_selection) {
  if (!Tensor(host, op::DataType::DT_BF16) ||
      !Tensor(current, op::DataType::DT_BF16) ||
      !Tensor(selection, op::DataType::DT_BF16) ||
      !Tensor(latent, op::DataType::DT_BF16) ||
      !Tensor(tokens, op::DataType::DT_INT32)) {
    return false;
  }
  const int64_t capacity = StableSelectionRows(latent_selection);
  const auto& topk_shape = tokens->GetViewShape();
  const auto& latent_shape = latent->GetViewShape();
  if (capacity <= 0 || topk_shape.GetDimNum() != 3 ||
      latent_shape.GetDimNum() != 4 || topk_shape.GetDim(0) <= 0 ||
      topk_shape.GetDim(0) > capacity) {
    return false;
  }
  const int64_t rows = topk_shape.GetDim(0);
  return Shape(host, {latent_shape.GetDim(0), 128, 1, 64}) &&
         Shape(current, {rows, 1, 64}) &&
         Shape(selection, {capacity * 17, 128, 1, 64});
}
}  // namespace xllm_ops::dsa_kv_offloading
