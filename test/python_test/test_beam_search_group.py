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

import os
import pytest
import torch
import pdb
import ctypes
import sys
torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

torch.manual_seed(10)
def beam_search_torch(request_num, beam_width, log_probs, top_tokens, top_probs, sequence, current_step, top_k):
    # top_tokens/top_probs shape: [request_num * beam_width, beam_width]
    beam_width = top_tokens.shape[-1]
    device = top_tokens.device
    # Prefill stage: current_step == 0, directly write the first column of top_tokens to the last dimension of the 0th column
    if current_step == 0:
        tokens_flat = top_tokens.view(request_num * beam_width, -1)
        first_col = tokens_flat[:, 0].to(torch.int32)
        origin_seq_t = sequence.clone()
        sequence[:, :, 0] = first_col.view(request_num, beam_width)
        out_token_ids = first_col.view(request_num, beam_width)
        out_token_index = torch.arange(request_num * beam_width, dtype=torch.int32, device=device).view(request_num, beam_width)
        out_log_probs = torch.full((request_num, beam_width), float('-inf'), dtype=torch.float32, device=device)
        out_beam_count_prefix_sums = torch.zeros((request_num, beam_width), dtype=torch.int32, device=device)
        
        return out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, sequence, origin_seq_t
    # Expand log_probs across beam_width candidates
    expanded_log_probs = log_probs.repeat(1, beam_width)  # [request_num, beam_width * beam_width]
    expanded_log_probs = expanded_log_probs.view(request_num, beam_width * beam_width)
    # Flatten top_probs to match expanded_log_probs
    candidate_scores = expanded_log_probs + top_probs.view(request_num, beam_width * beam_width)
    # Select top beam_width candidates per request
    topk_scores, topk_indices = torch.topk(candidate_scores, top_k, dim=1, largest=True, sorted=True)
    selected_beam = torch.div(topk_indices, beam_width, rounding_mode='floor').to(torch.int32)
    selected_within_top = torch.remainder(topk_indices, beam_width).to(torch.int32)             # [request_num, beam_width]
    # Gather next tokens from top_tokens using computed indices
    request_ids = torch.arange(request_num, dtype=torch.int32, device=device).view(-1, 1)
    base_indices = (request_ids * beam_width)
    orig_seq_indices = (base_indices + selected_beam).reshape(-1).to(torch.long)  # [request_num*beam_width]

    selected_top = top_tokens.index_select(0, orig_seq_indices)
    next_tokens = selected_top.gather(1, selected_within_top.reshape(-1, 1).to(torch.long)) \
                             .reshape(request_num, top_k).to(torch.int32)

    beam_ids = selected_beam.reshape(request_num, top_k)
    scores = topk_scores.to(torch.float32).reshape(request_num, top_k)
    # Outputs
    out_token_ids = torch.zeros((request_num, top_k), dtype=torch.int32, device=device)
    out_token_index = torch.full((request_num, top_k), -1, dtype=torch.int32, device=device)
    out_log_probs = torch.full((request_num, top_k), float('-inf'), dtype=torch.float32, device=device)
    out_beam_count_prefix_sums = torch.zeros((request_num, beam_width), dtype=torch.int32, device=device)
    
    # 按照beam id从小到大重排所有的tokens，同一个beam的按照score降序排列
    # Per-request bucketing by beam id with stable order
    for r in range(request_num):
        # counts: 所有token来源beam的计数
        counts = torch.zeros((beam_width,), dtype=torch.int32, device=device)
        for j in range(top_k):
            b = int(beam_ids[r, j])
            if 0 <= b < beam_width:
                counts[b] += 1
        # prefix: token来源beam的累加和
        prefix = torch.cumsum(counts, dim=0)  # exclusive starts are prefix - counts
        # exclusive prefix: 每个beam的起始位置
        starts = prefix - counts
        cursor = starts.clone()
        token_sorted = torch.zeros((top_k,), dtype=torch.int32, device=device)
        score_sorted = torch.full((top_k,), float('-inf'), dtype=torch.float32, device=device)
        index_sorted = torch.full((top_k,), -1, dtype=torch.int32, device=device)
        for j in range(top_k):
            b = int(beam_ids[r, j])
            if 0 <= b < beam_width:
                pos = int(cursor[b])
                if pos < top_k:
                    token_sorted[pos] = next_tokens[r, j]
                    score_sorted[pos] = scores[r, j]
                    index_sorted[pos] = b  # local beam id; offset added after rewrite
                    cursor[b] = cursor[b] + 1
        out_token_ids[r] = token_sorted
        out_log_probs[r] = score_sorted
        # Add request offset to out_token_index to match kernel behavior
        out_token_index[r] = index_sorted + r * beam_width
        # global prefix sum like C++: prefix + r * beam_width
        out_beam_count_prefix_sums[r] = prefix + r * beam_width
    origin_seq = sequence

    # Decode stage: first reorder prefix according to out_token_index, then write the new token to the current_step column
    index_flat = out_token_index.view(-1).to(torch.long)
    flat_sequence = sequence[:, :beam_width, :].reshape(request_num * beam_width, -1)
    reordered_rows = flat_sequence.index_select(0, index_flat)
    view_shape = sequence.shape
    if top_k != beam_width:
        flat_sequence = torch.zeros((request_num * top_k, sequence.shape[-1]), dtype=sequence.dtype).to(sequence.device)
        view_shape = (request_num, top_k, sequence.shape[-1])
       
    if current_step > 0:
        flat_sequence[:, :current_step] = reordered_rows[:, :current_step]
    flat_sequence[:, current_step] = out_token_ids.view(-1)
    sequence = flat_sequence.view(view_shape)
    return out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, sequence, origin_seq

@pytest.mark.parametrize(
    "request_num, beam_width, top_k, current_step",
    [
        # beam_width == top_k && top_k % 8 == 0
        # (1, 8, 8, 1),
        # (1, 8, 8, 2),
        # (1, 8, 16, 1),
        # (1, 8, 16, 2),
        # (1, 16, 16, 1),
        # (1, 16, 16, 2),
        # (2, 32, 32, 1),
        # (2, 32, 32, 2),
        # (2, 512, 512, 1),
        # (2, 512, 512, 2),
        # (2, 512, 1024, 1),
        # (2, 512, 1024, 2),
        # (1, 1024, 1024, 1),
        # (1, 1024, 1024, 2),
        # (4, 32, 32, 1),
        # (4, 32, 32, 2),
        # (4, 512, 512, 1),
        # (4, 512, 512, 2),
        # (4, 1024, 1024, 1),
        # (4, 1024, 1024, 2),

        # beam_width != top_k && top_k % 8 == 0 && beam_width % 8 == 0 && top_k < step_size * beam_width
        # 默认step_size为8
        # (1, 8, 16, 1),
        # (2, 16, 8, 1),
        # (2, 16, 32, 1),
        # (2, 16, 32, 1),
        # (2, 32, 64, 1),
        # (2, 64, 32, 1),
        # (2, 64, 128, 1),
        # (2, 128, 256, 2),
        # (2, 256, 512, 2),
        # (2, 512, 1024, 2),

        # beam_width != top_k && top_k % 8 != 0 && beam_width % 8 == 0 && top_k <= step_size * beam_width
        # 默认step_size为8
        # (1, 8, 15, 2),
        # (2, 8, 15, 2),
        # (2, 8, 17, 1),

        # beam_width != top_k && top_k % 8 != 0 && beam_width % 8 != 0 && top_k <= step_size * beam_width
        # 默认step_size为8
        # (1, 9, 7, 1),
        # (1, 9, 15, 1),
        # (1, 15, 7, 2),
        # (1, 15, 31, 1),
        # (1, 17, 31, 1),
    ],
)
def test_beam_search_group_npu(request_num, beam_width, top_k, current_step):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    # 配合pytest -s输出ascend c代码中的标准输出
    libc = ctypes.CDLL(None)

    # 设置 stdout 无缓冲
    libc.setvbuf(
        ctypes.c_void_p.in_dll(libc, "stdout"),
        None,
        2,
        0
    )


    if top_k > beam_width * beam_width:
        top_k = beam_width * beam_width

    atol_div = 1e-4
    eff_top_k = 1 if current_step == 0 else top_k

    log_probs = torch.rand((request_num * beam_width, 1), dtype=torch.float32)
    top_tokens = torch.randint(0, 10, (request_num * beam_width, beam_width), dtype=torch.int32)
    top_probs = torch.rand((request_num * beam_width, beam_width), dtype=torch.float32)
    sequence = torch.randint(0, 10, (request_num, beam_width, current_step + 1), dtype=torch.int32)
    origin_seq = sequence.clone()
    print("sequence", sequence)
    out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, sequence_cpu, _ = beam_search_torch(
        request_num, beam_width, log_probs, top_tokens, top_probs, sequence, current_step, top_k
    )

    out_token_ids_npu, out_token_index_npu, out_log_probs_npu, out_beam_count_prefix_sums_npu, sequence_npu = custom_ops.beam_search_group_npu(
        log_probs.npu(), top_tokens.npu(), top_probs.npu(), origin_seq.npu(), current_step, top_k
    )

    # 可能存在token_id不同，但是out_log_probs一样的情况，应属正常情况，此时可以只比较out_token_index， out_log_probs和out_beam_count_prefix_sums
    if current_step != 0:
        assert torch.allclose(out_token_ids.flatten(), out_token_ids_npu.cpu().flatten(), atol=atol_div, rtol=0)
        assert torch.allclose(out_token_index.flatten(), out_token_index_npu.cpu().flatten(), atol=atol_div, rtol=0)
        assert torch.allclose(out_log_probs.flatten(), out_log_probs_npu.cpu().flatten(), atol=atol_div, rtol=0)
        assert torch.allclose(out_beam_count_prefix_sums.flatten(), out_beam_count_prefix_sums_npu.cpu().flatten(), atol=atol_div, rtol=0)
    if current_step == 0:
        assert torch.allclose(out_token_ids.flatten(), sequence_npu.cpu()[:, :, 0].flatten(), atol=atol_div, rtol=0)
    else:
        assert torch.allclose(sequence_cpu[:, :, : current_step + 1].flatten(), sequence_npu.cpu()[:, :, : current_step + 1].flatten(), atol=atol_div, rtol=0)