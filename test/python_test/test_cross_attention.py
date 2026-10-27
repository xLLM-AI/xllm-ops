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

import math

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def ceil(x, y):
    return (x + y - 1) // y


def compare_tensors(a: torch.Tensor, b: torch.Tensor,
                    atol: float = 1e-6, rtol: float = 1e-4,
                    max_display: int = 10):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    diff = torch.abs(a - b)
    condition = (diff > atol) & (diff > rtol * torch.maximum(torch.abs(a), torch.abs(b)))

    indices = torch.where(condition)
    total_violations = condition.sum().item()

    print(f"Total violations: {total_violations}")
    print(f"Absolute tolerance (atol): {atol}")
    print(f"Relative tolerance (rtol): {rtol}")
    print("-" * 80)

    if total_violations == 0:
        print("All elements satisfy the tolerance!")
        return

    count = min(max_display, total_violations)
    print(f"Showing first {count} violations:")
    print(f"{'Index':<30} {'Tensor A':<20} {'Tensor B':<20} {'Diff':<20} {'Tolerance':<20}")
    print("-" * 80)

    for i in range(count):
        idx = tuple(indices[dim][i].item() for dim in range(len(indices)))
        val_a = a[idx].item()
        val_b = b[idx].item()
        val_diff = diff[idx].item()
        tol = atol + rtol * max(abs(val_a), abs(val_b))
        print(f"{str(idx):<30} {val_a:<20.10f} {val_b:<20.10f} {val_diff:<20.10f} {tol:<20.10f}")


def gen_input_data(batch, q_len, kv_len, max_block_num_per_req, block_size, block_num,
                   q_head_num, kv_head_num, head_dim, dtype=torch.bfloat16):
    actual_kv_lens = torch.Tensor([kv_len] * batch).to(torch.int64)
    query = torch.randn((batch, q_len, q_head_num, head_dim), dtype=dtype)
    key_cache = torch.randn((block_num, block_size, kv_head_num, head_dim), dtype=dtype)
    value_cache = torch.randn((block_num, block_size, kv_head_num, head_dim), dtype=dtype)
    block_tables = torch.zeros((batch, max_block_num_per_req), dtype=torch.int32)

    block_nums = []
    for kv_len in actual_kv_lens:
        block_nums.append(ceil(kv_len, block_size))

    total_block_num = sum(block_nums)
    tables = torch.randperm(block_num)[:total_block_num]

    block_start = 0
    for b_idx, block_num in enumerate(block_nums):
        block_tables[b_idx, :block_num] = torch.Tensor(tables[block_start: block_start + block_num]).to(torch.int64)
        block_start += block_num

    scale = 1 / math.sqrt(head_dim)

    return query, key_cache, value_cache, block_tables, actual_kv_lens, scale


def gather_kv_from_cache(kv_len, block_table, key_cache, value_cache):
    _, block_size, kv_head_num, head_dim = key_cache.shape
    key = torch.empty((kv_len, kv_head_num, head_dim), dtype=key_cache.dtype, device=key_cache.device)
    value = torch.empty((kv_len, kv_head_num, head_dim), dtype=value_cache.dtype, device=value_cache.device)
    seq_start = 0
    block_num = ceil(kv_len, block_size)

    for i in range(block_num):
        block_idx = block_table[i]
        copy_len = block_size
        if i == block_num - 1:
            copy_len = kv_len - i * block_size
        key[seq_start: seq_start + copy_len] = key_cache[block_idx, :copy_len, ...]
        value[seq_start: seq_start + copy_len] = value_cache[block_idx, :copy_len, ...]
        seq_start += copy_len

    return key, value


def online_softmax(input, in_max, in_sum, update):
    if update:
        x_max = torch.max(torch.cat([in_max, input], dim=-1), dim=-1, keepdims=True).values
        dst = torch.exp(input - x_max)
        exp_max = torch.exp(in_max - x_max)
        x_sum = torch.sum(dst, dim=-1, keepdims=True)
        x_sum = x_sum + exp_max * in_sum
        return dst, x_max, x_sum, exp_max
    else:
        x_max = torch.max(input, dim=-1, keepdims=True).values
        x_sub = input - x_max
        dst = torch.exp(x_sub)
        x_sum = torch.sum(dst, dim=-1, keepdims=True)
        exp_max = None
        return dst, x_max, x_sum, exp_max


def cpu_cross_attention(query, key_cache, value_cache, block_tables, actual_kv_lens, scale):
    kv_tile = 512
    batch, q_len, q_head_num, head_dim = query.shape
    block_num, block_size, kv_head_num, head_dim = key_cache.shape
    group_size = q_head_num // kv_head_num

    attn_out = torch.empty_like(query)
    for b_idx in range(batch):
        kv_len = actual_kv_lens[b_idx]
        key, value = gather_kv_from_cache(kv_len, block_tables[b_idx], key_cache, value_cache)

        kv_tile_num = ceil(kv_len, kv_tile)
        for q_head_idx in range(q_head_num):
            kv_head_idx = q_head_idx // group_size

            q_tile = query[b_idx, :, q_head_idx, :]
            in_max, in_sum, exp_max, attn_out_tile = None, None, None, None
            for kv_idx in range(kv_tile_num):
                kv_tile_len = (kv_len - kv_idx * kv_tile) if kv_idx == kv_tile_num - 1 else kv_tile
                key_tile = key[kv_idx * kv_tile: kv_idx * kv_tile + kv_tile_len, kv_head_idx, :]
                value_tile = value[kv_idx * kv_tile: kv_idx * kv_tile + kv_tile_len, kv_head_idx, :]
                update = kv_idx > 0
                qk = torch.matmul(q_tile.to(torch.float32), key_tile.transpose(0, 1).to(torch.float32))

                qk_scale = qk * scale
                p, in_max, in_sum, exp_max = online_softmax(qk_scale, in_max, in_sum, update)
                p = p.to(q_tile.dtype)

                pv = torch.matmul(p.to(torch.float32), value_tile.to(torch.float32))

                if update:
                    attn_out_tile = pv + exp_max * attn_out_tile
                else:
                    attn_out_tile = pv
            attn_out_tile = attn_out_tile / in_sum
            attn_out[b_idx, :, q_head_idx, :] = attn_out_tile.to(query.dtype)
    return attn_out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("q_head_num, kv_head_num", [(16, 4)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch", [1, 6, 8])
@pytest.mark.parametrize("decoder_prompt_len", [3])
@pytest.mark.parametrize("encoder_prompt_len", [68, 128, 256, 300, 460, 700])
def test_cross_attention(dtype, q_head_num, kv_head_num, head_dim, batch,
                         decoder_prompt_len, encoder_prompt_len):
    # Device selection (skip if no NPU available)
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    torch.manual_seed(1234)

    max_model_len = 1024
    block_size = 128
    block_num = 100
    max_block_num_per_req = ceil(max_model_len, block_size)
    query, key_cache, value_cache, block_tables, actual_kv_lens, scale = gen_input_data(
        batch, decoder_prompt_len, encoder_prompt_len, max_block_num_per_req,
        block_size, block_num, q_head_num, kv_head_num, head_dim, dtype)
    attn_golden = cpu_cross_attention(query, key_cache, value_cache, block_tables, actual_kv_lens, scale)

    q_npu = query.npu()
    key_npu = key_cache.npu()
    value_npu = value_cache.npu()
    block_table_npu = block_tables.npu()
    kv_len_npu = actual_kv_lens.npu()

    if dtype == torch.bfloat16:
        atol = 1e-2
        rtol = 1e-2
    else:
        atol = 1e-3
        rtol = 1e-3

    npu_res = custom_ops.cross_attention_npu(q_npu, key_npu, value_npu, block_table_npu, kv_len_npu, scale)
    npu_res = npu_res.cpu()

    flag = torch.allclose(attn_golden, npu_res, atol=atol, rtol=rtol, equal_nan=True)
    if not flag:
        compare_tensors(attn_golden, npu_res, atol, rtol)
    assert flag

