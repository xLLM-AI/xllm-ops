/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace ecr_compact {

constexpr size_t kInputProb = 0;
constexpr size_t kInputTauVal = 1;
constexpr size_t kInputTauIdx = 2;
constexpr size_t kInputSums = 3;
constexpr size_t kInputCnts = 4;
constexpr size_t kInputCount = 5;
constexpr size_t kOutputIds = 0;
constexpr size_t kOutputWeights = 1;
constexpr size_t kOutputCount = 2;
constexpr size_t kAttrSeqLengths = 0;
constexpr size_t kAttrTopP = 1;
constexpr size_t kAttrEps = 2;
constexpr int64_t kMaxExpertCount = 64;
constexpr uint32_t kMaxSequenceCount = 32;
constexpr int64_t kMaxSequenceLength = 2097152;
constexpr uint32_t kTokenTile = 512;

struct Contract {
  int64_t e = -1;
  int64_t s = -1;
  int64_t capacity_c = -1;
  uint32_t b = 1;
  uint32_t s3_total_tasks = 0;
  float top_p = 1.0f;
  float eps = 1.0e-9f;
  uint32_t seq_base[kMaxSequenceCount] = {};
  uint32_t seq_len[kMaxSequenceCount] = {};
  uint32_t seq_k[kMaxSequenceCount] = {};
  uint32_t c_offset[kMaxSequenceCount] = {};
  uint32_t s3_task_base[kMaxSequenceCount] = {};
};

inline ge::DataType InputType(size_t index) {
  return index == kInputTauIdx ? ge::DT_INT32 : ge::DT_FLOAT;
}

inline ge::DataType OutputType(size_t index) {
  return index == kOutputIds ? ge::DT_INT32 : ge::DT_FLOAT;
}

inline bool UnknownRank(const gert::Shape& shape) {
  return shape.GetDimNum() == 1 && shape.GetDim(0) == -2;
}

inline bool Matches(const gert::Shape& shape,
                    int64_t first,
                    int64_t second,
                    bool vector,
                    bool dynamic) {
  if (UnknownRank(shape)) {
    return dynamic;
  }
  if (shape.GetDimNum() != (vector ? 1U : 2U)) {
    return false;
  }
  const int64_t expected[2] = {first, second};
  for (size_t i = 0; i < shape.GetDimNum(); ++i) {
    const int64_t dim = shape.GetDim(i);
    if (dim < -1 || (!dynamic && dim < 0)) {
      return false;
    }
    if (dim >= 0 && expected[i] >= 0 && dim != expected[i]) {
      return false;
    }
  }
  return true;
}

template <typename Context>
inline bool BuildContract(Context* ctx,
                          const gert::Shape& prob,
                          bool dynamic,
                          Contract& contract) {
  if (ctx == nullptr) {
    return false;
  }
  if (UnknownRank(prob)) {
    if (!dynamic) {
      return false;
    }
  } else {
    if (prob.GetDimNum() != 2) {
      return false;
    }
    contract.e = prob.GetDim(0);
    contract.s = prob.GetDim(1);
    if (contract.e < -1 || contract.e == 0 || contract.e > kMaxExpertCount ||
        contract.s < -1 || contract.s > INT32_MAX ||
        (!dynamic && (contract.e < 1 || contract.s < 0))) {
      return false;
    }
  }

  const auto* attrs = ctx->GetAttrs();
  const auto* lengths =
      attrs == nullptr ? nullptr : attrs->GetListInt(kAttrSeqLengths);
  const size_t count = lengths == nullptr ? 0 : lengths->GetSize();
  if (count > kMaxSequenceCount) {
    return false;
  }
  if (attrs != nullptr) {
    const float* top_p = attrs->template GetAttrPointer<float>(kAttrTopP);
    if (top_p != nullptr) {
      contract.top_p = *top_p;
    }
    const float* eps = attrs->template GetAttrPointer<float>(kAttrEps);
    if (eps != nullptr) {
      contract.eps = *eps;
    }
  }
  if (!std::isfinite(contract.top_p) || contract.top_p <= 0.0f ||
      contract.top_p > 1.0f || !std::isfinite(contract.eps) ||
      contract.eps <= 0.0f) {
    return false;
  }

  contract.b = count > 1 ? static_cast<uint32_t>(count) : 1U;
  if (count <= 1 && contract.s < 0) {
    return dynamic;
  }
  uint64_t base = 0;
  uint64_t capacity = 0;
  uint64_t tasks = 0;
  const int64_t* values = count > 1 ? lengths->GetData() : nullptr;
  if (count > 1 && values == nullptr) {
    return false;
  }
  for (uint32_t sample = 0; sample < contract.b; ++sample) {
    const int64_t len = count > 1 ? values[sample] : contract.s;
    if (len < 0 || len > kMaxSequenceLength) {
      return false;
    }
    const double rounded = std::ceil(static_cast<double>(len) *
                                         static_cast<double>(contract.top_p) -
                                     1.0e-12);
    const uint32_t k =
        len == 0 ? 0U
                 : static_cast<uint32_t>(std::min(static_cast<double>(len),
                                                  std::max(1.0, rounded)));
    contract.seq_base[sample] = static_cast<uint32_t>(base);
    contract.seq_len[sample] = static_cast<uint32_t>(len);
    contract.seq_k[sample] = k;
    contract.c_offset[sample] = static_cast<uint32_t>(capacity);
    contract.s3_task_base[sample] = static_cast<uint32_t>(tasks);
    base += static_cast<uint64_t>(len);
    capacity += k;
    tasks += (static_cast<uint64_t>(len) + kTokenTile - 1) / kTokenTile;
    if (base > INT32_MAX || capacity > INT32_MAX || tasks > UINT32_MAX) {
      return false;
    }
  }
  if (contract.s >= 0 && base != static_cast<uint64_t>(contract.s)) {
    return false;
  }
  contract.capacity_c = static_cast<int64_t>(capacity);
  contract.s3_total_tasks = static_cast<uint32_t>(tasks);
  return true;
}

}  // namespace ecr_compact
}  // namespace ops
