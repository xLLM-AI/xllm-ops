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

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def _beam_search_rec_final_select_torch(
    log_probs: torch.Tensor,
    top_tokens: torch.Tensor,
    top_probs: torch.Tensor,
    sequence: torch.Tensor,
    current_step: int,
    result_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    request_num, active_beam_width, total_rounds = sequence.shape
    candidate_top_k = top_tokens.shape[-1]
    combined_probs = (
        log_probs.view(request_num, active_beam_width, 1)
        + top_probs.view(request_num, active_beam_width, candidate_top_k)
    ).view(request_num, active_beam_width * candidate_top_k)
    out_log_probs, flat_indices = torch.topk(
        combined_probs, result_width, dim=-1, largest=True, sorted=True
    )
    parent_beam = torch.div(flat_indices, candidate_top_k, rounding_mode="floor")
    token_in_beam = torch.remainder(flat_indices, candidate_top_k)
    request_idx = torch.arange(request_num, device=top_tokens.device).unsqueeze(1)
    out_token_ids = top_tokens.view(
        request_num, active_beam_width, candidate_top_k
    )[request_idx, parent_beam, token_in_beam].to(torch.int32)
    out_token_index = (parent_beam + request_idx * active_beam_width).to(
        torch.int32
    )
    out_sequence = torch.zeros(
        (request_num, result_width, total_rounds),
        dtype=torch.int32,
        device=sequence.device,
    )
    out_sequence[:, :, :current_step] = sequence[
        request_idx.expand_as(parent_beam),
        parent_beam,
        :current_step,
    ]
    out_sequence[:, :, current_step] = out_token_ids
    return (
        out_token_ids.reshape(-1, 1),
        out_token_index.reshape(-1, 1),
        out_log_probs.reshape(-1, 1),
        out_sequence,
    )


@pytest.mark.parametrize(
    "request_num, active_beam_width, candidate_top_k, result_width, current_step",
    [
        (1, 8, 32, 32, 1),
        (1, 16, 128, 64, 2),
        (2, 16, 64, 64, 2),
        (1, 128, 256, 256, 2),
        (1, 256, 512, 512, 2),
    ],
)
def test_beam_search_rec_final_select_npu(
    request_num: int,
    active_beam_width: int,
    candidate_top_k: int,
    result_width: int,
    current_step: int,
) -> None:
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    torch.manual_seed(2026)
    total_rounds = current_step + 1
    log_probs = torch.randn((request_num * active_beam_width, 1), dtype=torch.float32)
    top_tokens = torch.randint(
        0,
        10000,
        (request_num * active_beam_width, candidate_top_k),
        dtype=torch.int32,
    )
    top_probs = torch.randn(
        (request_num * active_beam_width, candidate_top_k), dtype=torch.float32
    )
    sequence = torch.randint(
        0,
        10000,
        (request_num, active_beam_width, total_rounds),
        dtype=torch.int32,
    )

    expected = _beam_search_rec_final_select_torch(
        log_probs, top_tokens, top_probs, sequence, current_step, result_width
    )
    actual = custom_ops.beam_search_rec_final_select_npu(
        log_probs.npu(),
        top_tokens.npu(),
        top_probs.npu(),
        sequence.npu(),
        current_step,
        result_width,
    )

    assert torch.equal(expected[0], actual[0].cpu())
    assert torch.equal(expected[1], actual[1].cpu())
    assert torch.allclose(expected[2], actual[2].cpu(), atol=1e-5, rtol=0)
    assert torch.equal(expected[3], actual[3].cpu())
