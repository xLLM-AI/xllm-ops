#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the x_flash_attention_infer operator (paged-KV flash decoding).

x_flash_attention_infer is a paged-KV flash-decoding attention kernel with a
TND layout and (kernel-internal) causal masking. This test targets the
DECODE scenario (qSeqlen == 1 per batch), which is both the simplest and the
most common inference case:

  * query        : [numTokens, qHead, headDim]  (numTokens = batch * 1)  fp16/bf16
  * key_cache /   : [numBlocks, blockSize, kvHead, headDim]  (paged, ND/TND)
    value_cache
  * block_table  : [batch, maxBlocksPerBatch]   int32
  * actual_q_lens: int32 cumulative prefix-sum (TND);  qSeqlen==1 -> [1,2,...,batch]
  * actual_kv_lens: int32 per-batch kv length (PAGED, NOT prefix-summed)

Single-core extra_tiling layout is built by custom_ops.x_flash_attention_infer_npu.

Single-core constraints (see custom_ops._build_xfa_extra_tiling):
  * every batch shares the SAME kv_seqlen (curKSBlockNum must be uniform)
  * kv_seqlen <= 512 (MAX_KV_STACK_LEN) -> curKSBlockNum == 1 == endS2Idx
  * maxQSeqlen(=1) * qHead < 128 -> guarantees the FD kernel branch

For qSeqlen == 1 the causal triangle is empty (the single query sees every kv
position), so the golden is a plain paged full-attention.
"""

import math
import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def _paged_gather_kv(cache, block_table, kv_seqlen, batch, block_size):
    """Reassemble contiguous per-batch KV from a paged cache.

    cache       : [numBlocks, blockSize, kvHead, headDim]
    block_table : [batch, maxBlocksPerBatch] int32
    returns     : [batch, kv_seqlen, kvHead, headDim]
    """
    kv_head = cache.shape[2]
    head_dim = cache.shape[3]
    out = torch.zeros((batch, kv_seqlen, kv_head, head_dim), dtype=cache.dtype)
    for b in range(batch):
        pos = 0
        blk = 0
        while pos < kv_seqlen:
            block_id = int(block_table[b, blk].item())
            take = min(block_size, kv_seqlen - pos)
            out[b, pos:pos + take] = cache[block_id, :take]
            pos += take
            blk += 1
    return out


def _xfa_decode_golden(query, key, value, batch, q_head, kv_head, scale):
    """Reference decode attention (qSeqlen == 1) in fp32.

    query : [batch, q_head, head_dim]      (one query token per batch)
    key   : [batch, kv_seqlen, kv_head, head_dim]
    value : [batch, kv_seqlen, kv_head, head_dim]
    returns attn_out [batch, q_head, head_dim] fp32
    """
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    group = q_head // kv_head
    head_dim = q.shape[-1]
    out = torch.zeros((batch, q_head, head_dim), dtype=torch.float32)
    for b in range(batch):
        for h in range(q_head):
            kvh = h // group
            qh = q[b, h]                       # [head_dim]
            kh = k[b, :, kvh]                  # [kv_seqlen, head_dim]
            vh = v[b, :, kvh]                  # [kv_seqlen, head_dim]
            scores = (kh @ qh) * scale         # [kv_seqlen]
            scores = scores - scores.max()
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = probs @ vh
    return out


def _build_causal_mask():
    """Kernel expects an int8 [2048, 2048] causal mask table (upper-tri masked)."""
    m = torch.triu(torch.ones(2048, 2048, dtype=torch.int8), diagonal=1)
    return m


@pytest.mark.parametrize(
    "dtype, batch, q_head, kv_head, head_dim, kv_seqlen, block_size",
    [
        (torch.float16, 1, 8, 8, 128, 128, 128),
        (torch.float16, 2, 8, 8, 128, 128, 128),
        (torch.float16, 2, 16, 8, 128, 256, 128),
    ],
)
def test_x_flash_attention_infer(dtype, batch, q_head, kv_head, head_dim,
                                 kv_seqlen, block_size):
    torch.manual_seed(0)
    scale = 1.0 / math.sqrt(head_dim)
    q_seqlen = 1
    num_tokens = batch * q_seqlen

    # blocks needed per batch, laid out contiguously per batch in block_table.
    blocks_per_batch = (kv_seqlen + block_size - 1) // block_size
    num_blocks = batch * blocks_per_batch

    query = torch.randn(num_tokens, q_head, head_dim, dtype=dtype)
    key_cache = torch.randn(num_blocks, block_size, kv_head, head_dim, dtype=dtype)
    value_cache = torch.randn(num_blocks, block_size, kv_head, head_dim, dtype=dtype)

    block_table = torch.zeros(batch, blocks_per_batch, dtype=torch.int32)
    for b in range(batch):
        for j in range(blocks_per_batch):
            block_table[b, j] = b * blocks_per_batch + j

    # actual_q_lens: cumulative prefix-sum of qSeqlen(=1) per batch.
    actual_q_lens = torch.arange(1, batch + 1, dtype=torch.int32)
    # actual_kv_lens: per-batch kv length (PAGED, not prefix-summed).
    actual_kv_lens = torch.full((batch,), kv_seqlen, dtype=torch.int32)

    mask = _build_causal_mask()

    # ---- golden (fp32) ----
    gathered_k = _paged_gather_kv(key_cache, block_table, kv_seqlen, batch, block_size)
    gathered_v = _paged_gather_kv(value_cache, block_table, kv_seqlen, batch, block_size)
    query_bhd = query.view(batch, q_seqlen, q_head, head_dim)[:, 0]  # [batch, q_head, head_dim]
    golden = _xfa_decode_golden(query_bhd, gathered_k, gathered_v,
                                batch, q_head, kv_head, scale)

    # ---- device ----
    out = custom_ops.x_flash_attention_infer_npu(
        query.npu(), key_cache.npu(), value_cache.npu(), block_table.npu(),
        actual_q_lens.npu(), actual_kv_lens.npu(),
        q_head, kv_head, scale, batch, kv_seqlen,
        mask=mask.npu(), layout="TND",
    )
    out = out.cpu().view(batch, q_head, head_dim).to(torch.float32)

    torch.testing.assert_close(out, golden, atol=6e-2, rtol=6e-2)