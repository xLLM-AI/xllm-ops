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

def beam_search_torch(request_num, beam_width, log_probs, top_tokens, top_probs, sequence, current_step):
    # top_tokens/top_probs shape: [request_num * beam_width, top_k]
    top_k = top_tokens.shape[-1]
    device = top_tokens.device
    # Expand log_probs across top_k candidates
    expanded_log_probs = log_probs.repeat(1, top_k)  # [request_num, beam_width * top_k]
    expanded_log_probs = expanded_log_probs.view(request_num, beam_width * top_k)
    # print("expanded_log_probs:", expanded_log_probs)
    # Flatten top_probs to match expanded_log_probs
    candidate_scores = expanded_log_probs + top_probs.view(request_num, beam_width * top_k)
    # print("candidate_scores:", candidate_scores)
    # Select top beam_width candidates per request
    topk_scores, topk_indices = torch.topk(candidate_scores, beam_width, dim=1, largest=True, sorted=True)
    # print("topk_scores:", topk_scores)
    # print("topk_indices:", topk_indices)
    selected_beam = torch.div(topk_indices, top_k, rounding_mode='floor').to(torch.int32)
    # print("selected_beam:", selected_beam)
    selected_within_top = torch.remainder(topk_indices, top_k).to(torch.int32)             # [request_num, beam_width]
    # print("selected_within_top:", selected_within_top)
    # Gather next tokens from top_tokens using computed indices
    request_ids = torch.arange(request_num, dtype=torch.int32, device=device).view(-1, 1)
    base_indices = (request_ids * beam_width)
    orig_seq_indices = (base_indices + selected_beam).reshape(-1).to(torch.long)  # [request_num*beam_width]
    # print("ori_seq_indices:", orig_seq_indices)
    selected_top = top_tokens.index_select(0, orig_seq_indices)
    # print("selected_top:", selected_top)                   # [request_num*beam_width, top_k]
    next_tokens = selected_top.gather(1, selected_within_top.reshape(-1, 1).to(torch.long)) \
                             .reshape(request_num, beam_width).to(torch.int32)
    # print("next_tokens:", next_tokens)
    beam_ids = selected_beam.reshape(request_num, beam_width)
    scores = topk_scores.to(torch.float32).reshape(request_num, beam_width)
    # Outputs
    out_token_ids = torch.zeros((request_num, beam_width), dtype=torch.int32, device=device)
    out_token_index = torch.full((request_num, beam_width), -1, dtype=torch.int32, device=device)
    out_log_probs = torch.full((request_num, beam_width), float('-inf'), dtype=torch.float32, device=device)
    out_beam_count_prefix_sums = torch.zeros((request_num, beam_width), dtype=torch.int32, device=device)
    # Per-request bucketing by beam id with stable order
    for r in range(request_num):
        counts = torch.zeros((beam_width,), dtype=torch.int32, device=device)
        for j in range(beam_width):
            b = int(beam_ids[r, j])
            if 0 <= b < beam_width:
                counts[b] += 1
        prefix = torch.cumsum(counts, dim=0)  # exclusive starts are prefix - counts
        starts = prefix - counts
        cursor = starts.clone()
        token_sorted = torch.zeros((beam_width,), dtype=torch.int32, device=device)
        score_sorted = torch.full((beam_width,), float('-inf'), dtype=torch.float32, device=device)
        index_sorted = torch.full((beam_width,), -1, dtype=torch.int32, device=device)
        for j in range(beam_width):
            b = int(beam_ids[r, j])
            if 0 <= b < beam_width:
                pos = int(cursor[b])
                if pos < beam_width:
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
        # Rewrite sequence per out_token_index (using local beam ids before offset)
        for j in range(beam_width):
            beam_local = int(index_sorted[j])  # local beam id
            if 0 <= beam_local < beam_width:
                sequence[r, j] = sequence[r, beam_local]
        # 根据current step 把新生成的token拼接到sequence中
        for j in range(beam_width):
            sequence[r, j, current_step - 1] = out_token_ids[r, j]
    return out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, sequence

@pytest.mark.parametrize("dtype", [torch.int32])
def test_beam_search_group_npu(dtype):
    # Device selection (skip if no NPU available)
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    torch.manual_seed(1234)

    request_num = 2
    beam_width = 2
    top_k = 2
    atol_div = 1e-4
    log_probs = torch.rand((request_num*beam_width,1), dtype=torch.float32)
    top_tokens = torch.randint(0, 10, (request_num * beam_width, top_k), dtype=torch.int32)
    top_probs = torch.rand((request_num * beam_width, top_k), dtype=torch.float32)
    sequence = torch.randint(0, 10, (request_num,beam_width,3), dtype=torch.int32)
    # print("log_probs:", log_probs)
    # print("top_tokens:", top_tokens)
    # print("top_probs:", top_probs)
    # print("sequence:", sequence)
    current_step = 2
    out_token_ids, out_token_index, out_log_probs, out_beam_count_prefix_sums, sequence = beam_search_torch(request_num, beam_width, log_probs, top_tokens, top_probs, sequence,current_step)
    # print("out_token_ids:", out_token_ids)
    # print("out_token_index:", out_token_index)
    # print("out_log_probs:", out_log_probs)
    # print("out_beam_count_prefix_sums:", out_beam_count_prefix_sums)
    # print("sequence:", sequence)
    sequence_golden = sequence.clone()
    log_probs_npu = log_probs.npu()
    top_tokens_npu = top_tokens.npu()
    top_probs_npu = top_probs.npu()
    sequence_npu = sequence.npu()
    out_token_ids_npu, out_token_index_npu, out_log_probs_npu, out_beam_count_prefix_sums_npu, sequence_npu = custom_ops.beam_search_group_npu(log_probs_npu, top_tokens_npu, top_probs_npu, sequence_npu, current_step)
    assert torch.allclose(out_token_ids, out_token_ids_npu.cpu(), atol=atol_div, rtol=0)
    assert torch.allclose(out_token_index, out_token_index_npu.cpu(), atol=atol_div, rtol=0)
    assert torch.allclose(out_log_probs, out_log_probs_npu.cpu(), atol=atol_div, rtol=0)
    assert torch.allclose(out_beam_count_prefix_sums, out_beam_count_prefix_sums_npu.cpu(), atol=atol_div, rtol=0)
    assert torch.allclose(sequence, sequence_golden, atol=atol_div, rtol=0)