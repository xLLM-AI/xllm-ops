/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#pragma once

#include "kernel_operator.h"

namespace ecr_routing_normalize_compact_impl {

constexpr uint32_t kSimtThreads = 1024;

// Contiguous thread intervals plus an exclusive prefix preserve token order.
__simt_vf__ __aicore__ LAUNCH_BOUND(kSimtThreads) inline void OutputCompactVf(
    __gm__ float* prob,
    __gm__ float* tau_values,
    __gm__ int32_t* tau_indices,
    __gm__ float* sums,
    __gm__ float* cnts,
    __gm__ int32_t* ids,
    __gm__ float* weights,
    __ubuf__ uint32_t* scan_buffer,
    uint64_t prob_offset,
    uint64_t output_offset,
    uint32_t task,
    uint32_t base,
    uint32_t length,
    uint32_t selected_limit,
    float eps) {
  const uint32_t thread_idx = AscendC::Simt::GetThreadIdx();
  const uint32_t thread_num = AscendC::Simt::GetThreadNum();
  // The kernel only writes selected slots; define every unused slot first.
  for (uint32_t slot = thread_idx; slot < selected_limit; slot += thread_num) {
    ids[output_offset + slot] = -1;
    weights[output_offset + slot] = 0.0f;
  }
  __syncthreads();

  const uint32_t segment_length = (length + thread_num - 1) / thread_num;
  const uint32_t begin = thread_idx * segment_length;
  const uint32_t candidate_end = begin + segment_length;
  const uint32_t end = candidate_end < length ? candidate_end : length;
  const float tau_value = tau_values[task];
  const int32_t tau_index = tau_indices[task];
  uint32_t selected_count = 0;
  for (uint32_t token = begin; token < end; ++token) {
    const float p = prob[prob_offset + token];
    if (p > tau_value ||
        (p == tau_value && static_cast<int32_t>(token) <= tau_index)) {
      ++selected_count;
    }
  }

  __ubuf__ uint32_t* scan_read = scan_buffer;
  __ubuf__ uint32_t* scan_write = scan_buffer + kSimtThreads;
  scan_read[thread_idx] = selected_count;
  __syncthreads();
  // Separate source/destination arrays avoid read/write races. Empty interval
  // threads must participate in every barrier, without an early return.
  for (uint32_t offset = 1; offset < thread_num; offset *= 2) {
    uint32_t prefix = scan_read[thread_idx];
    if (thread_idx >= offset) {
      prefix += scan_read[thread_idx - offset];
    }
    scan_write[thread_idx] = prefix;
    __syncthreads();
    __ubuf__ uint32_t* scan_swap = scan_read;
    scan_read = scan_write;
    scan_write = scan_swap;
  }
  uint32_t output_index = scan_read[thread_idx] - selected_count;
  for (uint32_t token = begin; token < end; ++token) {
    const float p = prob[prob_offset + token];
    if (p > tau_value ||
        (p == tau_value && static_cast<int32_t>(token) <= tau_index)) {
      // Preserve defensive truncation for externally supplied thresholds.
      if (output_index < selected_limit) {
        const uint32_t global_token = base + token;
        const float sum = sums[global_token];
        const float denominator = sum > eps ? sum : eps;
        const float normalized = p / denominator;
        const float count_root = sqrt(cnts[global_token]);
        ids[output_offset + output_index] = static_cast<int32_t>(global_token);
        weights[output_offset + output_index] = normalized * count_root;
      }
      ++output_index;
    }
  }
}

}  // namespace ecr_routing_normalize_compact_impl
