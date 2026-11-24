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

#include <cstdint>

// ACLNN appends the TensorList descriptors after tiling. Their fields require
// 64-bit alignment.
struct alignas(uint64_t) KdaReplayssmFoldTilingData {
  int32_t sequences;
  int32_t layers;
  int32_t workers;
  int32_t reserved = 0;
};

namespace kda_replayssm {

constexpr int32_t kHeads = 8;
constexpr int32_t kWindow = 8;
constexpr int32_t kKeyDimension = 128;
constexpr int32_t kValueDimension = 128;
constexpr int32_t kVectorLanes = 64;
constexpr uint8_t kStateRepeatStride = 16;
constexpr int32_t kFeatures = kHeads * kKeyDimension;
constexpr int32_t kStateElements = kValueDimension * kKeyDimension;
constexpr int32_t kRecordElements = kWindow * kKeyDimension;
constexpr int32_t kStateBytes = kStateElements * sizeof(float);
constexpr int32_t kRecordBytes = kRecordElements * sizeof(float);
constexpr int32_t kBufferBytes = kStateBytes + 3 * kRecordBytes;
constexpr int32_t kUbBytes = 2 * kBufferBytes + kRecordBytes;

}  // namespace kda_replayssm
