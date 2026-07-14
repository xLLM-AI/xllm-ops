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

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


BLOCK_SIZE = 128
C0 = 16  # fp16 fractal inner dim


def _nd2nz_block(block):
    """Convert one [BLOCK_SIZE, token_size] ND block to NZ (FRACTAL_NZ) layout,
    returned flattened in the exact order the kernel writes back in-place.

    ND element (n, d) with n in [0, BLOCK_SIZE), d in [0, token_size) maps to
    NZ linear offset:
        (d // C0) * (BLOCK_SIZE * C0) + n * C0 + (d % C0)

    token_size is padded up to a multiple of C0; padding columns are zero,
    matching the Nd2Nz hardware behaviour (dValue rounded up to C0).
    """
    n_rows, token_size = block.shape
    assert n_rows == BLOCK_SIZE
    d_pad = (token_size + C0 - 1) // C0 * C0
    out = torch.zeros(BLOCK_SIZE * d_pad, dtype=block.dtype)
    for d in range(token_size):
        base = (d // C0) * (BLOCK_SIZE * C0) + (d % C0)
        # rows n=0..127 stride C0
        out[base:base + BLOCK_SIZE * C0:C0] = block[:, d]
    return out


def convert_kv_cache_format_golden(k_cache, v_cache, kv_cache_offset,
                                   kv_seq_len, is_prefill,
                        num_kv_heads, head_size_k, head_size_v):
    """CPU reference matching the AICore kernel: in-place ND2NZ rewrite.

    k_cache: [max_tokens, num_kv_heads, head_size_k] fp16
    v_cache: [max_tokens, num_kv_heads, head_size_v] fp16
    kv_cache_offset: [num_batches] int64, token offset (in tokens) per batch
    kv_seq_len: [num_batches] int32, valid token count per batch

    prefill: for each batch, from front to back, every full BLOCK_SIZE-token
             segment is rewritten in NZ layout; the tail (< BLOCK_SIZE) is
             untouched.
    decode:  for each batch, only when kv_seqlen > 0 and kv_seqlen %
             BLOCK_SIZE == 0, the last BLOCK_SIZE block is rewritten.
    """
    token_size_k = num_kv_heads * head_size_k
    token_size_v = num_kv_heads * head_size_v

    k = k_cache.clone().reshape(-1)  # flat over [max_tokens * token_size_k]
    v = v_cache.clone().reshape(-1)
    num_batches = kv_cache_offset.shape[0]

    def rewrite(flat, base_tok, blk_start_tok, token_size):
        # flat element start of the block
        start = (base_tok + blk_start_tok) * token_size
        block = flat[start:start + BLOCK_SIZE * token_size].reshape(
            BLOCK_SIZE, token_size)
        nz = _nd2nz_block(block)
        # kernel writes back block_size * token_size elements contiguously
        flat[start:start + BLOCK_SIZE * token_size] = nz[:BLOCK_SIZE * token_size]

    for b in range(num_batches):
        seqlen = int(kv_seq_len[b].item())
        if seqlen == 0:
            continue
        base = int(kv_cache_offset[b].item())
        if is_prefill:
            remain = seqlen
            blk = 0
            while remain >= BLOCK_SIZE:
                rewrite(k, base, blk, token_size_k)
                rewrite(v, base, blk, token_size_v)
                blk += BLOCK_SIZE
                remain -= BLOCK_SIZE
        else:
            if seqlen > 0 and seqlen % BLOCK_SIZE == 0:
                blk = seqlen - BLOCK_SIZE
                rewrite(k, base, blk, token_size_k)
                rewrite(v, base, blk, token_size_v)

    return (k.reshape(k_cache.shape), v.reshape(v_cache.shape))


# 10 shapes:
# (is_prefill, seq_lens(list), num_kv_heads, head_size_k, head_size_v)
CASES = [
    (True,  [128],            2, 16, 16),
    (True,  [256],            2, 16, 16),
    (True,  [384],            4, 32, 32),
    (True,  [130],            2, 64, 64),   # tail < 128 untouched
    (True,  [128, 256],       2, 16, 16),   # multi-batch
    (True,  [200, 128, 300],  4, 16, 48),   # k/v different head_size
    (False, [128],            2, 16, 16),   # decode aligned
    (False, [256],            4, 32, 32),   # decode aligned, rewrite last blk
    (False, [130],            2, 16, 16),   # decode not %128 -> untouched
    (False, [128, 200, 384],  2, 48, 16),   # multi-batch decode mix
]


def _make_inputs(seq_lens, num_kv_heads, head_size_k, head_size_v):
    num_batches = len(seq_lens)
    token_size_k = num_kv_heads * head_size_k
    token_size_v = num_kv_heads * head_size_v

    offsets = []
    acc = 0
    for s in seq_lens:
        offsets.append(acc)
        acc += s
    max_tokens = acc

    k_cache = torch.randn(max_tokens, num_kv_heads, head_size_k,
                          dtype=torch.float16)
    v_cache = torch.randn(max_tokens, num_kv_heads, head_size_v,
                          dtype=torch.float16)
    kv_cache_offset = torch.tensor(offsets, dtype=torch.int64)
    kv_seq_len = torch.tensor(seq_lens, dtype=torch.int32)
    return k_cache, v_cache, kv_cache_offset, kv_seq_len


@pytest.mark.parametrize(
    "is_prefill,seq_lens,num_kv_heads,head_size_k,head_size_v", CASES)
def test_convert_kv_cache_format(is_prefill, seq_lens, num_kv_heads,
                                 head_size_k, head_size_v):
    torch.manual_seed(2026)
    k_cache, v_cache, kv_cache_offset, kv_seq_len = _make_inputs(
        seq_lens, num_kv_heads, head_size_k, head_size_v)

    k_ref, v_ref = convert_kv_cache_format_golden(
        k_cache, v_cache, kv_cache_offset, kv_seq_len,
        is_prefill, num_kv_heads, head_size_k, head_size_v)

    k_npu = k_cache.npu()
    v_npu = v_cache.npu()
    off_npu = kv_cache_offset.npu()
    seq_npu = kv_seq_len.npu()

    custom_ops.convert_kv_cache_format_npu(
        k_npu, v_npu, off_npu, seq_npu,
        is_prefill, num_kv_heads, head_size_k, head_size_v)

    k_out = k_npu.cpu()
    v_out = v_npu.cpu()

    assert torch.equal(k_out, k_ref), (
        f"k_cache mismatch prefill={is_prefill} seq_lens={seq_lens}")
    assert torch.equal(v_out, v_ref), (
        f"v_cache mismatch prefill={is_prefill} seq_lens={seq_lens}")