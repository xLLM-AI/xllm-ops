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
#include <initializer_list>
#include <limits>

#include "opdev/common_types.h"
#include "opdev/tensor_view_utils.h"
namespace xllm_ops::dsa_kv_offloading {
inline bool Tensor(const aclTensor* t, op::DataType dtype) {
  return t != nullptr && t->GetData() != nullptr && t->GetDataType() == dtype &&
         t->GetStorageFormat() == op::Format::FORMAT_ND &&
         op::IsContiguous(t) && t->GetPlacement() == gert::kOnDeviceHbm;
}
inline bool Shape(const aclTensor* t, std::initializer_list<int64_t> dims) {
  if (t == nullptr || t->GetViewShape().GetDimNum() != dims.size())
    return false;
  size_t axis = 0;
  for (const auto dim : dims)
    if (t->GetViewShape().GetDim(axis++) != dim) return false;
  return true;
}
inline bool Distinct(std::initializer_list<const aclTensor*> tensors) {
  // Writable and independently consumed buffers must not overlap, even via
  // different views of the same allocation. Empty tensors are not accepted.
  for (auto i = tensors.begin(); i != tensors.end(); ++i) {
    if (*i == nullptr || (*i)->Numel() <= 0) return false;
    const auto begin = reinterpret_cast<uintptr_t>((*i)->GetData());
    const uint64_t elementBytes =
        (*i)->GetDataType() == op::DataType::DT_BF16 ? 2 : 4;
    if (static_cast<uint64_t>((*i)->Numel()) >
        std::numeric_limits<uintptr_t>::max() / elementBytes)
      return false;
    const uintptr_t bytes =
        static_cast<uintptr_t>((*i)->Numel()) * elementBytes;
    if (begin > std::numeric_limits<uintptr_t>::max() - bytes) return false;
    for (auto j = tensors.begin(); j != i; ++j) {
      const auto other = reinterpret_cast<uintptr_t>((*j)->GetData());
      // Earlier iterations already validated this range and its arithmetic.
      const uintptr_t otherBytes =
          static_cast<uintptr_t>((*j)->Numel()) *
          ((*j)->GetDataType() == op::DataType::DT_BF16 ? 2 : 4);
      if (begin < other + otherBytes && other < begin + bytes) return false;
    }
  }
  return true;
}
}  // namespace xllm_ops::dsa_kv_offloading
