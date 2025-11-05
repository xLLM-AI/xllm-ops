#!/usr/bin/env python3
# Copyright 2025 The xLLM Authors. All Rights Reserved.
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


def _select_unshared_kv(
    global_index: torch.Tensor,
    unshared_kv_cache: torch.Tensor,
    token_idxs: torch.Tensor,
    decode_step: int,
    *,
    batch: int,
    beam_size: int,
    head_num: int,
    head_dim: int,
):
    # Store token_idx of this step into global_index
    global_index[:, :, decode_step - 1] = token_idxs.reshape([batch, beam_size])

    select_kv_cache = torch.zeros(
        [batch * beam_size, head_num, decode_step, head_dim],
        dtype=unshared_kv_cache.dtype,
    )

    # Iteratively search back in time
    for batch_idx in range(batch):
        prev_step_token_index = token_idxs[batch_idx * beam_size : batch_idx * beam_size + beam_size]
        for i in range(decode_step, 0, -1):
            real_idx = i - 1
            # Find indices for this step
            if i == decode_step:
                cur_step_token_index = prev_step_token_index
            else:
                cur_step_token_index = global_index[batch_idx, prev_step_token_index, real_idx]
                prev_step_token_index = cur_step_token_index
            # Gather KV for this step based on indices
            batch_kv_cache = unshared_kv_cache[batch_idx * beam_size : batch_idx * beam_size + beam_size]
            cur_step_kv = batch_kv_cache[cur_step_token_index, :, real_idx, :]
            select_kv_cache[
                batch_idx * beam_size : batch_idx * beam_size + beam_size, :, real_idx, :
            ] = cur_step_kv

    return select_kv_cache


def _calc_group_num(token_idxs: torch.Tensor, *, batch: int, beam_size: int) -> torch.Tensor:
    group_num = torch.zeros([batch, beam_size], dtype=torch.int32)
    for b_idx in range(batch):
        for token_idx in token_idxs[b_idx * beam_size : b_idx * beam_size + beam_size]:
            group_num[b_idx][token_idx] += 1
        # Prefix-sum in-place
        for i in range(1, beam_size):
            group_num[b_idx][i] = group_num[b_idx][i - 1] + group_num[b_idx][i]
    return group_num.reshape([batch * beam_size])


def _generate_beam_idxs(*, batch: int, beam_size: int) -> torch.Tensor:
    beam_idxs = torch.randint(0, beam_size, [batch, beam_size], dtype=torch.int32)
    # Indices in beam_idxs must be sorted ascending
    sorted_beam_idxs, _ = torch.sort(beam_idxs)
    return sorted_beam_idxs.reshape([batch * beam_size])

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_select_unshared_kv_npu(dtype):
    # Device selection (skip if no NPU available)
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    torch.manual_seed(1234)

    batch = 10
    max_decode_step = 3
    beam_size = 512
    head_num = 8
    head_dim = 128

    x_key_block_npu = torch.zeros(
        [batch * beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    ).npu()
    x_value_block_npu = torch.zeros(
        [batch * beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    ).npu()
    x_key_block = torch.zeros(
        [batch * beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    )
    x_value_block = torch.zeros(
        [batch * beam_size, head_num, max_decode_step, head_dim], dtype=dtype
    )

    atol_div = 1e-6
    global_index = torch.zeros([batch, beam_size, max_decode_step], dtype=torch.int32)

    for decode_step in range(1, max_decode_step + 1):
        beam_idxs = _generate_beam_idxs(batch=batch, beam_size=beam_size)
        curr_key = torch.rand([batch * beam_size, head_num, head_dim], dtype=dtype)
        curr_value = torch.rand([batch * beam_size, head_num, head_dim], dtype=dtype)
        real_index = decode_step - 1

        # Fill key/value for this step
        x_key_block[:, :, real_index, :] = curr_key
        x_value_block[:, :, real_index, :] = curr_value
        x_key_block_npu[:, :, real_index, :] = curr_key.npu()
        x_value_block_npu[:, :, real_index, :] = curr_value.npu()

        select_key_golden = _select_unshared_kv(
            global_index, x_key_block, beam_idxs, decode_step,
            batch=batch, beam_size=beam_size, head_num=head_num, head_dim=head_dim
        )
        select_value_golden = _select_unshared_kv(
            global_index, x_value_block, beam_idxs, decode_step,
            batch=batch, beam_size=beam_size, head_num=head_num, head_dim=head_dim
        )

        # Compute group_num
        group_num = _calc_group_num(beam_idxs, batch=batch, beam_size=beam_size)

        # Run NPU kernel (in-place on x_key_block_npu/x_value_block_npu)
        custom_ops.select_unshared_kv_npu(
            beam_idxs.npu(),
            x_key_block_npu,
            x_value_block_npu,
            group_num.npu(),
            decode_step,
            beam_size,
        )

        select_key_npu = x_key_block_npu.cpu()[:, :, :decode_step, :]
        select_value_npu = x_value_block_npu.cpu()[:, :, :decode_step, :]

        assert torch.allclose(select_key_npu, select_key_golden, atol=atol_div, rtol=0)
        assert torch.allclose(select_value_npu, select_value_golden, atol=atol_div, rtol=0)
