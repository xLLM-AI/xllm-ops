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
namespace ecr_threshold {

constexpr size_t kInputProb = 0;
constexpr size_t kInputCount = 1;
constexpr size_t kOutputTauVal = 0;
constexpr size_t kOutputTauIdx = 1;
constexpr size_t kOutputNGt = 2;
constexpr size_t kOutputCount = 3;
constexpr size_t kAttrSeqLengths = 0;
constexpr size_t kAttrTopP = 1;
constexpr int64_t kMaxExpertCount = 64;
constexpr uint32_t kMaxSequenceCount = 32;
constexpr int64_t kMaxSequenceLength = 2097152;

struct Contract {
  int64_t e = -1;
  int64_t s = -1;
  uint32_t b = 1;
  float top_p = 1.0f;
  uint32_t seq_base[kMaxSequenceCount] = {};
  uint32_t seq_len[kMaxSequenceCount] = {};
  uint32_t seq_k[kMaxSequenceCount] = {};
};

inline ge::DataType InputType(size_t index) {
  (void)index;
  return ge::DT_FLOAT;
}

inline ge::DataType OutputType(size_t index) {
  return index == kOutputTauVal ? ge::DT_FLOAT : ge::DT_INT32;
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
      if (!std::isfinite(*top_p) || *top_p <= 0.0f || *top_p > 1.0f) {
        return false;
      }
      contract.top_p = *top_p;
    }
  }

  contract.b = count > 1 ? static_cast<uint32_t>(count) : 1U;
  if (count <= 1 && contract.s < 0) {
    return dynamic;
  }
  uint64_t base = 0;
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
    base += static_cast<uint64_t>(len);
    if (base > INT32_MAX) {
      return false;
    }
  }
  if (contract.s >= 0 && base != static_cast<uint64_t>(contract.s)) {
    return false;
  }
  return true;
}

}  // namespace ecr_threshold
}  // namespace ops
