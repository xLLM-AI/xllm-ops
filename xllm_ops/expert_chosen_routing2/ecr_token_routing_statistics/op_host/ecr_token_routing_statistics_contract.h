/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace ecr_statistics {

constexpr size_t kInputProb = 0;
constexpr size_t kInputTauVal = 1;
constexpr size_t kInputTauIdx = 2;
constexpr size_t kInputCount = 3;
constexpr size_t kOutputSums = 0;
constexpr size_t kOutputCnts = 1;
constexpr size_t kOutputCount = 2;
constexpr size_t kAttrSeqLengths = 0;
constexpr int64_t kMaxExpertCount = 64;
constexpr uint32_t kMaxSequenceCount = 32;
constexpr int64_t kMaxSequenceLength = 2097152;
constexpr uint32_t kTokenTile = 512;

struct Contract {
  int64_t e = -1;
  int64_t s = -1;
  uint32_t b = 1;
  uint32_t total_tasks = 0;
  uint32_t seq_base[kMaxSequenceCount] = {};
  uint32_t seq_len[kMaxSequenceCount] = {};
  uint32_t task_base[kMaxSequenceCount] = {};
};

inline ge::DataType InputType(size_t index) {
  return index == kInputTauIdx ? ge::DT_INT32 : ge::DT_FLOAT;
}

inline ge::DataType OutputType(size_t index) {
  (void)index;
  return ge::DT_FLOAT;
}

inline bool UnknownRank(const gert::Shape& shape) {
  return shape.GetDimNum() == 1 && shape.GetDim(0) == -2;
}

// `first`/`second` may be -1 to accept any extent on that axis.
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
                          const gert::Shape* tau_val,
                          const gert::Shape* tau_idx,
                          bool dynamic,
                          Contract& contract) {
  if (ctx == nullptr || tau_val == nullptr || tau_idx == nullptr) {
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

  contract.b = count > 1 ? static_cast<uint32_t>(count) : 1U;
  if (!Matches(*tau_val, contract.e, static_cast<int64_t>(contract.b), false,
               dynamic) ||
      !Matches(*tau_idx, contract.e, static_cast<int64_t>(contract.b), false,
               dynamic)) {
    return false;
  }
  if (count <= 1 && contract.s < 0) {
    return dynamic;
  }
  const int64_t* values = count > 1 ? lengths->GetData() : nullptr;
  if (count > 1 && values == nullptr) {
    return false;
  }
  uint64_t base = 0;
  uint64_t tasks = 0;
  for (uint32_t sample = 0; sample < contract.b; ++sample) {
    const int64_t len = count > 1 ? values[sample] : contract.s;
    if (len < 0 || len > kMaxSequenceLength) {
      return false;
    }
    const uint64_t tiles =
        (static_cast<uint64_t>(len) + kTokenTile - 1) / kTokenTile;
    contract.seq_base[sample] = static_cast<uint32_t>(base);
    contract.seq_len[sample] = static_cast<uint32_t>(len);
    contract.task_base[sample] = static_cast<uint32_t>(tasks);
    base += static_cast<uint64_t>(len);
    tasks += tiles;
    if (base > INT32_MAX || tasks > UINT32_MAX) {
      return false;
    }
  }
  if (contract.s >= 0 && base != static_cast<uint64_t>(contract.s)) {
    return false;
  }
  contract.total_tasks = static_cast<uint32_t>(tasks);
  return true;
}

}  // namespace ecr_statistics
}  // namespace ops
