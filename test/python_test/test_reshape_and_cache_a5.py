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


CASES = [
    (torch.float16, 1, 2, 128, [0]),
    (torch.float16, 7, 3, 17, [0, 127, 128, 255, 256, 383, 511]),
    (torch.float16, 17, 4, 128, [255, 3, 129, 64, 0, 130, 127, 17,
                                18, 19, 20, 21, 22, 23, 24, 25, 26]),
    (torch.float16, 129, 4, 128, None),
    (torch.bfloat16, 1, 4, 128, [511]),
    (torch.bfloat16, 33, 8, 128, None),
]


@pytest.mark.parametrize("dtype,num_tokens,num_heads,head_dim,slots", CASES)
def test_reshape_and_cache_a5(dtype, num_tokens, num_heads, head_dim, slots):
    torch.manual_seed(7)
    num_blocks = 4
    block_size = 128
    total_slots = num_blocks * block_size
    if slots is None:
        slots = torch.randperm(total_slots)[:num_tokens].tolist()
    slot_mapping_cpu = torch.tensor(slots, dtype=torch.int32)

    key_cpu = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype)
    value_cpu = torch.randn_like(key_cpu)
    key_cache_cpu = torch.randn(
        num_blocks, block_size, num_heads, head_dim, dtype=dtype)
    value_cache_cpu = torch.randn_like(key_cache_cpu)
    expected_key = key_cache_cpu.clone().view(total_slots, num_heads, head_dim)
    expected_value = value_cache_cpu.clone().view(
        total_slots, num_heads, head_dim)
    expected_key[slot_mapping_cpu.long()] = key_cpu
    expected_value[slot_mapping_cpu.long()] = value_cpu

    key = key_cpu.npu()
    value = value_cpu.npu()
    key_cache = key_cache_cpu.npu()
    value_cache = value_cache_cpu.npu()
    slot_mapping = slot_mapping_cpu.npu()
    custom_ops.reshape_and_cache_a5_npu(
        key, value, key_cache, value_cache, slot_mapping)
    torch_npu.npu.synchronize()

    torch.testing.assert_close(key_cache.cpu(), expected_key.view_as(key_cache_cpu),
                               rtol=0, atol=0)
    torch.testing.assert_close(
        value_cache.cpu(), expected_value.view_as(value_cache_cpu),
        rtol=0, atol=0)


def test_reshape_and_cache_a5_skips_negative_slots():
    key = torch.randn(3, 2, 128, dtype=torch.float16)
    value = torch.randn_like(key)
    key_cache = torch.zeros(2, 128, 2, 128, dtype=torch.float16)
    value_cache = torch.zeros_like(key_cache)
    slots = torch.tensor([0, -1, 255], dtype=torch.int32)

    key_npu = key.npu()
    value_npu = value.npu()
    key_cache_npu = key_cache.npu()
    value_cache_npu = value_cache.npu()
    custom_ops.reshape_and_cache_a5_npu(
        key_npu, value_npu, key_cache_npu, value_cache_npu, slots.npu())
    torch_npu.npu.synchronize()

    expected_key = key_cache.view(256, 2, 128)
    expected_value = value_cache.view(256, 2, 128)
    expected_key[0] = key[0]
    expected_key[255] = key[2]
    expected_value[0] = value[0]
    expected_value[255] = value[2]
    torch.testing.assert_close(key_cache_npu.cpu(), key_cache, rtol=0, atol=0)
    torch.testing.assert_close(value_cache_npu.cpu(), value_cache,
                               rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reshape_and_cache_a5_preserves_unmapped_cache(dtype):
    key = torch.randn(2, 2, 64, dtype=dtype)
    value = torch.randn_like(key)
    key_cache = torch.randn(2, 128, 2, 64, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()
    slots = torch.tensor([1, 254], dtype=torch.int32)
    expected_key.view(256, 2, 64)[slots.long()] = key
    expected_value.view(256, 2, 64)[slots.long()] = value

    key_cache_npu = key_cache.npu()
    value_cache_npu = value_cache.npu()
    custom_ops.reshape_and_cache_a5_npu(
        key.npu(), value.npu(), key_cache_npu, value_cache_npu, slots.npu())
    torch_npu.npu.synchronize()

    torch.testing.assert_close(key_cache_npu.cpu(), expected_key, rtol=0, atol=0)
    torch.testing.assert_close(
        value_cache_npu.cpu(), expected_value, rtol=0, atol=0)
