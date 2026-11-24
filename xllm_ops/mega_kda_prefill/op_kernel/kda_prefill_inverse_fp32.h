/* Copyright 2025 JD.com
Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// KDA implementation introduced in xllm_ops commit be99d50 (2026-09-15).
#pragma once
#include <pto/pto-inst.hpp>

#include "kda_kernel_utils.h"

namespace mega_kda_prefill_inverse_fp32 {
using namespace pto;
using namespace mega_kda_prefill_utils;

#ifdef __CCE_AICORE__
using GS = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
using ST = pto::Stride<1, 1, 1, DYNAMIC, 1>;
using GM = GlobalTensor<float, GS, ST>;
template <int R, int C, int VR = R>
using Vec = Tile<TileType::Vec,
                 float,
                 R,
                 C,
                 BLayout::RowMajor,
                 VR,
                 C,
                 SLayout::NoneBox,
                 512>;
#endif

#if defined(__DAV_VEC__)
AICORE inline void subtract_pair_tree(__ubuf__ float* result,
                                      __ubuf__ float* rhs,
                                      __ubuf__ float* products,
                                      __ubuf__ float* scratch,
                                      int count) {
  constexpr int kCols = 64;
  constexpr int kRowBlocks = 8;
  auto* src = products;
  auto* dst = scratch;
  while (count > 1) {
    const int pairs = count / 2;
    vadd(dst,
         src,
         src + kCols,
         pairs,
         1,
         1,
         1,
         kRowBlocks,
         2 * kRowBlocks,
         2 * kRowBlocks);
    PipeBarrierVec();
    if (count % 2 != 0) {
      vadd(dst,
           dst,
           src + (count - 1) * kCols,
           1,
           1,
           1,
           1,
           kRowBlocks,
           kRowBlocks,
           kRowBlocks);
      PipeBarrierVec();
    }
    auto* next = src;
    src = dst;
    dst = next;
    count = pairs;
  }
  vsub(result, rhs, src, 1, 1, 1, 1, kRowBlocks, kRowBlocks, kRowBlocks);
  PipeBarrierVec();
}
#endif

template <int C, bool ScaleFullAkk = false>
AICORE inline void inverse_fp32(__gm__ float* lower,
                                __gm__ float* out,
                                __gm__ float* identity,
                                __gm__ float* workspace,
                                __gm__ const int32_t* cu,
                                int64_t batches,
                                int64_t tokens,
                                int32_t heads,
                                __gm__ float* beta = nullptr) {
#if defined(__DAV_VEC__)
  // One sub-block owns a 64-column inverse. A 128-column inverse uses
  // both sub-blocks of the same core.
  constexpr int N = 64;
  static_assert(C == N || C == 2 * N);
  constexpr int subblocks_per_task = C / N;
  constexpr int tasks_per_core = 2 / subblocks_per_task;
  const int subblock = get_subblockid();
  const int core =
      get_block_idx() * tasks_per_core + subblock / subblocks_per_task;
  const int cores = get_block_num() * tasks_per_core;
  const int col0 = (subblock % subblocks_per_task) * N;
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t total_chunks = 0;
  for (int seq = 0; seq < batches; ++seq) {
    total_chunks += (cu[seq + 1] - cu[seq] + C - 1) / C;
  }
  Vec<C, N> eye;
  TASSIGN(eye, 147456);
  GM geye(identity + col0, GS(1, 1, 1, C, N), ST(128));
  TLOAD(eye, geye);
  for (int64_t task = core; task < total_chunks * heads; task += cores) {
    const int head = task % heads;
    int64_t chunk = task / heads;
    int64_t bos = 0;
    int64_t len = 0;
    for (int seq = 0; seq < batches; ++seq) {
      bos = cu[seq];
      len = cu[seq + 1] - bos;
      const int64_t count = (len + C - 1) / C;
      if (chunk < count) {
        break;
      }
      chunk -= count;
    }
    const int valid = len - chunk * C < C ? len - chunk * C : C;
    const int64_t start = bos + chunk * C;
    Vec<C, C, DYNAMIC> low(valid);
    Vec<C, N> x;
    Vec<64, N> reduction;
    Vec<C, 8> broadcast;
    TASSIGN(low, 0);
    TASSIGN(x, 65536);
    TASSIGN(reduction, 131072);
    TASSIGN(broadcast, 180224);
    GM src(lower + (start * heads + head) * C,
           GS(1, 1, 1, valid, C),
           ST(heads * C));
    TLOAD(low, src);
    if constexpr (ScaleFullAkk) {
      if (valid == C) {
        // Cube publishes the unscaled FP32 dot products for full chunks.
        // Apply the same row beta as the vector path before solving.
        static_assert(C == 64);
        constexpr int kBetaAddr = 180224 + C * 8 * sizeof(float);
        Vec<1, C> row_beta;
        Vec<C, 8> beta_broadcast;
        Tile<TileType::Vec, float, C, 1, BLayout::ColMajor> beta_column;
        TASSIGN(row_beta, kBetaAddr);
        TASSIGN(beta_column, kBetaAddr);
        TASSIGN(beta_broadcast, 180224);
        GM weights(beta + int64_t(head) * tokens + start,
                   GS(1, 1, 1, 1, C), ST(C));
        TLOAD(row_beta, weights);
        SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
        TROWEXPANDMUL(low, low, beta_column, beta_broadcast);
        PipeBarrierVec();
      } else {
        SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
      }
    } else {
      SetWaitFlag<PIPE_MTE2, PIPE_V>(0);
    }
    TMOV(x, eye);
    PipeBarrierVec();
    // X[i,:] = I[i,:] - sum_{j<i} L[i,j] X[j,:]. Each AIV owns its RHS.
    for (int row = 1; row < valid; ++row) {
      Vec<C, N, DYNAMIC> solved(row), products(row);
      Tile<TileType::Vec, float, C, 1, BLayout::ColMajor, DYNAMIC, 1> coeff(
          row);
      Vec<1, N> result, rhs;
      TASSIGN(solved, 65536);
      TASSIGN(products, 98304);
      TASSIGN(coeff, row * C * 4);
      TASSIGN(result, 65536 + row * N * 4);
      TASSIGN(rhs, 147456 + row * N * 4);
      TROWEXPANDMUL(products, solved, coeff, broadcast);
      PipeBarrierVec();
      subtract_pair_tree(
          result.data(), rhs.data(), products.data(), reduction.data(), row);
    }
    Vec<C, N, DYNAMIC> store(valid);
    TASSIGN(store, 65536);
    GM dst(out + (start * heads + head) * C + col0,
           GS(1, 1, 1, valid, N),
           ST(heads * C));
    SetWaitFlag<PIPE_V, PIPE_MTE3>(0);
    TSTORE(dst, store);
    // Return both the solution and lower buffers before the next task reuses
    // UB.
    pipe_barrier(PIPE_ALL);
  }
#endif
}

}  // namespace mega_kda_prefill_inverse_fp32
