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

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def beam_search_torch(log_probs, top_tokens, top_probs, request_num, beam_width, top_k):
    """PyTorch reference implementation for beam_search operator.

    Args:
        log_probs: [num_sequences, sequence_length] float32 - cumulative log probs per sequence
        top_tokens: [num_sequences, top_k] int32 - top-k candidate token ids per sequence
        top_probs: [num_sequences, top_k] float32 - top-k candidate log probs per sequence
        request_num: number of requests
        beam_width: beam width (= top_k)
        top_k: number of top candidates (= beam_width)

    Returns:
        out_token_ids: [num_sequences] int32
        out_token_index: [num_sequences] int32
        out_log_probs: [num_sequences] float32
    """
    device = log_probs.device
    num_sequences = request_num * beam_width

    # Reshape log_probs to [request_num, beam_width]
    log_probs_2d = log_probs.view(request_num, beam_width)

    # Reshape top_tokens and top_probs to [request_num, beam_width, top_k]
    top_tokens_3d = top_tokens.view(request_num, beam_width, top_k)
    top_probs_3d = top_probs.view(request_num, beam_width, top_k)

    # Expand log_probs: [request_num, beam_width, 1] + top_probs: [request_num, beam_width, top_k]
    # => combined: [request_num, beam_width, top_k]
    combined = log_probs_2d.unsqueeze(2) + top_probs_3d

    # Flatten to [request_num, beam_width * top_k] for topk selection
    combined_flat = combined.view(request_num, beam_width * top_k)

    # Select top beam_width candidates per request
    topk_scores, topk_indices = torch.topk(combined_flat, beam_width, dim=1, largest=True, sorted=True)

    # Decode parent beam index and within-beam token index
    parent_beam = torch.div(topk_indices, top_k, rounding_mode="floor").to(torch.int32)
    token_in_beam = torch.remainder(topk_indices, top_k).to(torch.int32)

    # Gather selected token ids
    request_idx = torch.arange(request_num, device=device).unsqueeze(1)
    out_token_ids = top_tokens_3d[request_idx, parent_beam, token_in_beam].to(torch.int32)

    # Compute output token index (parent beam with request offset)
    out_token_index = (parent_beam + request_idx * beam_width).to(torch.int32)

    # Output log probs
    out_log_probs = topk_scores.to(torch.float32)

    return out_token_ids.view(-1), out_token_index.view(-1), out_log_probs.view(-1)


def _get_device():
    """Get NPU device, skip test if unavailable."""
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
        return True
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")
        return False


# Test Case 1: Basic small beam search (beam_width=4, request_num=1)
@pytest.mark.parametrize(
    "request_num, beam_width, top_k",
    [
        (1, 4, 4),
    ],
)
def test_beam_search_basic(request_num, beam_width, top_k):
    """Test beam_search with basic small configuration."""
    if not _get_device():
        return

    torch.manual_seed(42)
    num_sequences = request_num * beam_width
    sequence_length = 1

    log_probs = torch.randn((num_sequences, sequence_length), dtype=torch.float32)
    top_tokens = torch.randint(0, 100, (num_sequences, top_k), dtype=torch.int32)
    top_probs = torch.randn((num_sequences, top_k), dtype=torch.float32)

    # CPU reference
    ref_token_ids, ref_token_index, ref_log_probs = beam_search_torch(
        log_probs, top_tokens, top_probs, request_num, beam_width, top_k
    )

    # NPU computation
    npu_token_ids, npu_token_index, npu_log_probs = custom_ops.beam_search_npu(
        log_probs.npu(), top_tokens.npu(), top_probs.npu()
    )

    atol = 1e-4
    assert torch.equal(ref_token_ids, npu_token_ids.cpu()), "token_ids mismatch"
    assert torch.equal(ref_token_index, npu_token_index.cpu()), "token_index mismatch"
    assert torch.allclose(ref_log_probs, npu_log_probs.cpu(), atol=atol, rtol=0), "log_probs mismatch"


# Test Case 2: Multiple requests (request_num=2, beam_width=8)
@pytest.mark.parametrize(
    "request_num, beam_width, top_k",
    [
        (2, 8, 8),
    ],
)
def test_beam_search_multi_request(request_num, beam_width, top_k):
    """Test beam_search with multiple requests."""
    if not _get_device():
        return

    torch.manual_seed(123)
    num_sequences = request_num * beam_width
    sequence_length = 1

    log_probs = torch.randn((num_sequences, sequence_length), dtype=torch.float32)
    top_tokens = torch.randint(0, 1000, (num_sequences, top_k), dtype=torch.int32)
    top_probs = torch.randn((num_sequences, top_k), dtype=torch.float32)

    ref_token_ids, ref_token_index, ref_log_probs = beam_search_torch(
        log_probs, top_tokens, top_probs, request_num, beam_width, top_k
    )

    npu_token_ids, npu_token_index, npu_log_probs = custom_ops.beam_search_npu(
        log_probs.npu(), top_tokens.npu(), top_probs.npu()
    )

    atol = 1e-4
    assert torch.equal(ref_token_ids, npu_token_ids.cpu()), "token_ids mismatch"
    assert torch.equal(ref_token_index, npu_token_index.cpu()), "token_index mismatch"
    assert torch.allclose(ref_log_probs, npu_log_probs.cpu(), atol=atol, rtol=0), "log_probs mismatch"


# Test Case 3: Larger beam width (beam_width=32, request_num=2)
@pytest.mark.parametrize(
    "request_num, beam_width, top_k",
    [
        (2, 32, 32),
    ],
)
def test_beam_search_large_beam(request_num, beam_width, top_k):
    """Test beam_search with larger beam width."""
    if not _get_device():
        return

    torch.manual_seed(2025)
    num_sequences = request_num * beam_width
    sequence_length = 1

    log_probs = torch.randn((num_sequences, sequence_length), dtype=torch.float32)
    top_tokens = torch.randint(0, 50000, (num_sequences, top_k), dtype=torch.int32)
    top_probs = torch.randn((num_sequences, top_k), dtype=torch.float32)

    ref_token_ids, ref_token_index, ref_log_probs = beam_search_torch(
        log_probs, top_tokens, top_probs, request_num, beam_width, top_k
    )

    npu_token_ids, npu_token_index, npu_log_probs = custom_ops.beam_search_npu(
        log_probs.npu(), top_tokens.npu(), top_probs.npu()
    )

    atol = 1e-4
    assert torch.equal(ref_token_ids, npu_token_ids.cpu()), "token_ids mismatch"
    assert torch.equal(ref_token_index, npu_token_index.cpu()), "token_index mismatch"
    assert torch.allclose(ref_log_probs, npu_log_probs.cpu(), atol=atol, rtol=0), "log_probs mismatch"


# Test Case 4: Very large beam width (beam_width=512, request_num=1)
@pytest.mark.parametrize(
    "request_num, beam_width, top_k",
    [
        (1, 512, 512),
    ],
)
def test_beam_search_very_large_beam(request_num, beam_width, top_k):
    """Test beam_search with very large beam width."""
    if not _get_device():
        return

    torch.manual_seed(2026)
    num_sequences = request_num * beam_width
    sequence_length = 1

    log_probs = torch.randn((num_sequences, sequence_length), dtype=torch.float32)
    top_tokens = torch.randint(0, 10000, (num_sequences, top_k), dtype=torch.int32)
    top_probs = torch.randn((num_sequences, top_k), dtype=torch.float32)

    ref_token_ids, ref_token_index, ref_log_probs = beam_search_torch(
        log_probs, top_tokens, top_probs, request_num, beam_width, top_k
    )

    npu_token_ids, npu_token_index, npu_log_probs = custom_ops.beam_search_npu(
        log_probs.npu(), top_tokens.npu(), top_probs.npu()
    )

    atol = 1e-4
    assert torch.equal(ref_token_ids, npu_token_ids.cpu()), "token_ids mismatch"
    assert torch.equal(ref_token_index, npu_token_index.cpu()), "token_index mismatch"
    assert torch.allclose(ref_log_probs, npu_log_probs.cpu(), atol=atol, rtol=0), "log_probs mismatch"


# Test Case 5: Multiple requests with large beam (request_num=4, beam_width=128)
@pytest.mark.parametrize(
    "request_num, beam_width, top_k",
    [
        (4, 128, 128),
    ],
)
def test_beam_search_multi_request_large(request_num, beam_width, top_k):
    """Test beam_search with multiple requests and large beam width."""
    if not _get_device():
        return

    torch.manual_seed(999)
    num_sequences = request_num * beam_width
    sequence_length = 1

    log_probs = torch.randn((num_sequences, sequence_length), dtype=torch.float32)
    top_tokens = torch.randint(0, 30000, (num_sequences, top_k), dtype=torch.int32)
    top_probs = torch.randn((num_sequences, top_k), dtype=torch.float32)

    ref_token_ids, ref_token_index, ref_log_probs = beam_search_torch(
        log_probs, top_tokens, top_probs, request_num, beam_width, top_k
    )

    npu_token_ids, npu_token_index, npu_log_probs = custom_ops.beam_search_npu(
        log_probs.npu(), top_tokens.npu(), top_probs.npu()
    )

    atol = 1e-4
    assert torch.equal(ref_token_ids, npu_token_ids.cpu()), "token_ids mismatch"
    assert torch.equal(ref_token_index, npu_token_index.cpu()), "token_index mismatch"
    assert torch.allclose(ref_log_probs, npu_log_probs.cpu(), atol=atol, rtol=0), "log_probs mismatch"