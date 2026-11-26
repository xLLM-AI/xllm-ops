/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "kernel_operator.h"
using namespace AscendC;

class T5SplitQkvAppendCacheKernel {
 public:
  __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR pastKey, GM_ADDR pastValue,
                             GM_ADDR query, GM_ADDR key, GM_ADDR value,
                             uint32_t batch, uint32_t heads, uint32_t length,
                             uint32_t dim) {
    qkv_.SetGlobalBuffer((__gm__ uint16_t*)qkv);
    pastKey_.SetGlobalBuffer((__gm__ uint16_t*)pastKey);
    pastValue_.SetGlobalBuffer((__gm__ uint16_t*)pastValue);
    query_.SetGlobalBuffer((__gm__ uint16_t*)query);
    key_.SetGlobalBuffer((__gm__ uint16_t*)key);
    keyCached_.SetGlobalBuffer((__gm__ uint16_t*)key);
    value_.SetGlobalBuffer((__gm__ uint16_t*)value);
    valueCached_.SetGlobalBuffer((__gm__ uint16_t*)value);
    // Leave history reads at the hardware default. Re-read cache lines may
    // remain resident; freshly allocated streaming outputs should not evict them.
    key_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    value_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    batch_ = batch; heads_ = heads; dim_ = dim;
    oldSize_ = static_cast<uint64_t>(length) * dim;
    newSize_ = oldSize_ + dim;
    // Divide the output-cache budget across every K and V head.
    // Whole chunks keep cached and streaming transfers on separate jobs.
    constexpr uint64_t usable = (192ULL - 24ULL) << 20;
    const uint64_t nHeads = static_cast<uint64_t>(batch) * heads;
    const uint64_t resident = (2 * nHeads * oldSize_ + 4 * nHeads * dim) * 2;
    uint64_t chunkBudget = resident < usable ? (usable - resident) / (2 * kChunk) : 0;
    cachedChunksBase_ = chunkBudget / (2 * nHeads);
    cachedChunksExtra_ = chunkBudget % (2 * nHeads);
    qChunks_ = (heads * dim + kChunk - 1) / kChunk;
    cacheChunks_ = (newSize_ + kChunk - 1) / kChunk;
    pipe_.InitBuffer(buffer_, 2, kChunk * sizeof(uint16_t));
  }

  __aicore__ inline void Process() {
    uint64_t qJobs = static_cast<uint64_t>(batch_) * qChunks_;
    uint64_t cacheJobs = static_cast<uint64_t>(batch_) * heads_ * cacheChunks_;
    uint64_t jobs = qJobs + 2 * cacheJobs;
    uint64_t begin = jobs * GetBlockIdx() / GetBlockNum();
    uint64_t end = jobs * (GetBlockIdx() + 1) / GetBlockNum();
    for (uint64_t job = begin; job < end; ++job) {
      if (job < qJobs) {
        uint64_t batch = job / qChunks_;
        uint64_t offset = (job % qChunks_) * kChunk;
        uint32_t size = Min(kChunk, heads_ * dim_ - offset);
        Copy(query_, batch * heads_ * dim_ + offset, qkv_,
             batch * 3 * heads_ * dim_ + offset, size);
      } else {
        uint64_t local = job - qJobs;
        bool isValue = local >= cacheJobs;
        if (isValue) local -= cacheJobs;
        uint64_t head = local / cacheChunks_;
        uint64_t offset = (local % cacheChunks_) * kChunk;
        uint32_t count = Min(kChunk, newSize_ - offset);
        uint64_t nHeads = static_cast<uint64_t>(batch_) * heads_;
        uint64_t globalHead = head + (isValue ? nHeads : 0);
        uint64_t remainder = (globalHead * cachedChunksExtra_) % (2 * nHeads);
        uint64_t extra = remainder + cachedChunksExtra_ >= 2 * nHeads;
        bool cached = offset / kChunk < cachedChunksBase_ + extra;
        auto out = isValue ? (cached ? valueCached_ : value_) : (cached ? keyCached_ : key_);
        auto past = isValue ? pastValue_ : pastKey_;
        uint32_t history = offset < oldSize_ ? Min(count, oldSize_ - offset) : 0;
        if (history) Copy(out, head * newSize_ + offset, past,
                          head * oldSize_ + offset, history);
        uint32_t tail = count - history;
        if (tail) {
          uint64_t batch = head / heads_;
          uint64_t headInBatch = head % heads_;
          uint64_t source = batch * 3 * heads_ * dim_ +
              (isValue ? 2 : 1) * heads_ * dim_ + headInBatch * dim_ +
              offset + history - oldSize_;
          Copy(out, head * newSize_ + offset + history, qkv_, source, tail);
        }
      }
    }
    Flush();
  }

 private:
  __aicore__ inline uint32_t Min(uint64_t a, uint64_t b) { return a < b ? a : b; }
  __aicore__ inline void Copy(GlobalTensor<uint16_t>& dst, uint64_t dstOffset,
                             GlobalTensor<uint16_t>& src, uint64_t srcOffset,
                             uint32_t size) {
    auto local = buffer_.AllocTensor<uint16_t>();
    DataCopyExtParams copy{1, size * 2, 0, 0, 0};
    DataCopyPadExtParams<uint16_t> padding{false, 0, 0, 0};
    DataCopyPad(local, src[srcOffset], copy, padding);
    buffer_.EnQue(local);
    // Start the next read before waiting for the preceding write. Two
    // independent UB slots keep MTE2 and MTE3 active on different chunks.
    Flush();
    pendingDst_ = dst;
    pendingOffset_ = dstOffset;
    pendingSize_ = size;
    pending_ = true;
  }
  __aicore__ inline void Flush() {
    if (!pending_) return;
    auto ready = buffer_.DeQue<uint16_t>();
    DataCopyExtParams copy{1, pendingSize_ * 2, 0, 0, 0};
    DataCopyPad(pendingDst_[pendingOffset_], ready, copy);
    buffer_.FreeTensor(ready);
    pending_ = false;
  }
  static constexpr uint32_t kChunk = 32768;
  TPipe pipe_;
  TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> buffer_;
  GlobalTensor<uint16_t> qkv_, pastKey_, pastValue_, query_, key_, keyCached_, value_, valueCached_;
  GlobalTensor<uint16_t> pendingDst_;
  uint64_t pendingOffset_ = 0;
  uint64_t cachedChunksBase_ = 0, cachedChunksExtra_ = 0;
  uint32_t pendingSize_ = 0;
  bool pending_ = false;
  uint32_t batch_, heads_, dim_, qChunks_;
  uint64_t oldSize_, newSize_, cacheChunks_;
};

