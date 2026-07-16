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
"""Tests for the multi_latent_attention operator (DeepSeek MLA, split-cache decode).

MLA (Multi-Latent Attention) compresses KV into a shared latent (NoPE) cache and
keeps a separate RoPE cache. Q and KV are each split into two parts:

  * NoPE (latent) part  : hidden dim = 512 (kernel-hardcoded embedding size)
  * RoPE part           : hidden dim = 64  (kernel-hardcoded)

The score uses BOTH parts; the value reuses ONLY the NoPE latent:

  K = [kvCache | kvCacheRope]      (576)
  Q = [query   | queryRope]        (576)
  score = (q_nope @ k_nope^T + q_rope @ k_rope^T) * tor
  attn  = softmax(score) @ v_nope   (v = kvCache NoPE, 512)

This test targets the DECODE case (one query token per batch, qSeqLen == 1),
which is the simplest and most common inference scenario. Host tiling is fully
automatic (no extra_tiling needed).

Layout (ND, fp16, non-quant, non-ring -> TilingKey == 0):
  * query       : [numTokens, q_head, 512]    fp16   (numTokens = batch * 1)
  * queryRope   : [numTokens, q_head, 64]      fp16
  * kvCache     : [numBlocks, blockSize, kv_head, 512]  fp16  (paged latent)
  * kvCacheRope : [numBlocks, blockSize, kv_head, 64]   fp16  (paged rope)
  * block_tables: [batch, maxBlocksPerBatch]   int32
  * contextLens : [batch]                      int32  (per-batch kv length)

Constraints for the simplest path:
  * type = 0 (SPLIT_CACHE), maskType = 0 (NONE), isRing = 0 (no lse output)
  * q_head < 128 to avoid the MTP TP1 special path (mtpTp1Flag)
  * MLA standard uses kv_head = 1 (shared latent).
"""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

NOPE_DIM = 512
ROPE_DIM = 64


def _paged_gather(cache, block_table, kv_seqlen, batch, block_size):
    """Reassemble contiguous per-batch KV from a paged cache.

    cache       : [numBlocks, blockSize, kv_head, dim]
    returns     : [batch, kv_seqlen, kv_head, dim]
    """
    kv_head = cache.shape[2]
    dim = cache.shape[3]
    out = torch.zeros((batch, kv_seqlen, kv_head, dim), dtype=cache.dtype)
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


def _mla_decode_golden(q_nope, q_rope, k_nope, k_rope, v_nope,
                       batch, q_head, kv_head, tor):
    """Reference MLA decode attention (qSeqLen == 1) in fp32.

    q_nope : [batch, q_head, 512]     q_rope : [batch, q_head, 64]
    k_nope : [batch, kv_seqlen, kv_head, 512]   k_rope : [.., .., .., 64]
    v_nope : [batch, kv_seqlen, kv_head, 512]
    returns attn_out [batch, q_head, 512] fp32
    """
    qn = q_nope.to(torch.float32)
    qr = q_rope.to(torch.float32)
    kn = k_nope.to(torch.float32)
    kr = k_rope.to(torch.float32)
    vn = v_nope.to(torch.float32)
    group = q_head // kv_head
    out = torch.zeros((batch, q_head, NOPE_DIM), dtype=torch.float32)
    for b in range(batch):
        for h in range(q_head):
            kvh = h // group
            qn_h = qn[b, h]                       # [512]
            qr_h = qr[b, h]                        # [64]
            kn_h = kn[b, :, kvh]                   # [kv_seqlen, 512]
            kr_h = kr[b, :, kvh]                   # [kv_seqlen, 64]
            vn_h = vn[b, :, kvh]                   # [kv_seqlen, 512]
            scores = (kn_h @ qn_h + kr_h @ qr_h) * tor   # [kv_seqlen]
            scores = scores - scores.max()
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = probs @ vn_h
    return out


@pytest.mark.parametrize(
    "dtype, batch, q_head, kv_head, kv_seqlen, block_size",
    [
        (torch.float16, 1, 16, 1, 128, 128),
        (torch.float16, 2, 16, 1, 128, 128),
        (torch.float16, 2, 32, 1, 256, 128),
    ],
)
def test_multi_latent_attention(dtype, batch, q_head, kv_head, kv_seqlen, block_size):
    torch.manual_seed(0)
    tor = 1.0 / math.sqrt(NOPE_DIM + ROPE_DIM)
    q_seqlen = 1
    num_tokens = batch * q_seqlen

    blocks_per_batch = (kv_seqlen + block_size - 1) // block_size
    num_blocks = batch * blocks_per_batch

    query = torch.randn(num_tokens, q_head, NOPE_DIM, dtype=dtype)
    query_rope = torch.randn(num_tokens, q_head, ROPE_DIM, dtype=dtype)
    kv_cache = torch.randn(num_blocks, block_size, kv_head, NOPE_DIM, dtype=dtype)
    kv_cache_rope = torch.randn(num_blocks, block_size, kv_head, ROPE_DIM, dtype=dtype)

    block_table = torch.zeros(batch, blocks_per_batch, dtype=torch.int32)
    for b in range(batch):
        for j in range(blocks_per_batch):
            block_table[b, j] = b * blocks_per_batch + j

    context_lens = torch.full((batch,), kv_seqlen, dtype=torch.int32)

    # ---- golden (fp32) ----
    gk = _paged_gather(kv_cache, block_table, kv_seqlen, batch, block_size)
    gkr = _paged_gather(kv_cache_rope, block_table, kv_seqlen, batch, block_size)
    q_nope = query.view(batch, q_seqlen, q_head, NOPE_DIM)[:, 0]
    q_rope = query_rope.view(batch, q_seqlen, q_head, ROPE_DIM)[:, 0]
    golden = _mla_decode_golden(q_nope, q_rope, gk, gkr, gk,
                                batch, q_head, kv_head, tor)

    # ---- device ----
    out = custom_ops.multi_latent_attention_npu(
        query.npu(), query_rope.npu(), kv_cache.npu(), kv_cache_rope.npu(),
        block_table.npu(), context_lens.npu(),
        q_head, kv_head, tor, [kv_seqlen] * batch,
    )
    out = out.cpu().view(batch, q_head, NOPE_DIM).to(torch.float32)

    torch.testing.assert_close(out, golden, atol=6e-2, rtol=6e-2)