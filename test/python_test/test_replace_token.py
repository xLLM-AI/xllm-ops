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


def replace_token_golden(forked_token_ids, last_step_output_token_ids):
    """CPU reference matching the AICore kernel forward logic.

    For each element in ``forked_token_ids`` (int32):
      - if the value is negative, it is a 1-based negative reference into
        ``last_step_output_token_ids``: out = last_step[-v - 1]
        (cast from int64 to int32).
      - otherwise the value is kept unchanged.

    Equivalent to the torch reference:
        neg_mask = forked < 0
        idx = (-forked) - 1
        replacement = last_step[idx]
        out = where(neg_mask, replacement, forked)
    """
    forked = forked_token_ids.to(torch.int64)
    neg_mask = forked < 0
    clamped_idx = torch.clamp(-forked, min=0) - 1
    clamped_idx = torch.clamp(clamped_idx, min=0)
    replacement = last_step_output_token_ids[clamped_idx]
    out = torch.where(neg_mask, replacement, forked)
    return out.to(torch.int32)


# 10 shapes: (sequence_length, b_length)
# sequence_length is the number of forked token ids;
# b_length is the number of last-step output token ids that can be referenced.
CASES = [
    (8,      4),
    (16,     8),
    (32,    16),
    (64,    16),
    (128,   32),
    (256,   64),
    (1000,  128),
    (37,     5),
    (511,    50),
    (10001, 100),  # crosses the kernel's max_tokens=10000 chunk boundary
]


def _make_inputs(sequence_length, b_length):
    # last-step output tokens: arbitrary positive int64 token ids.
    last_step = torch.randint(0, 100000, (b_length,), dtype=torch.int64)

    # forked tokens: a mix of non-negative token ids and negative references.
    # A negative value v means "take last_step[-v - 1]", so valid negatives
    # are in [-b_length, -1].
    forked = torch.randint(0, 100000, (sequence_length,), dtype=torch.int32)

    # Randomly turn ~40% of positions into valid negative references.
    neg_choice = torch.rand(sequence_length) < 0.4
    neg_refs = -(torch.randint(1, b_length + 1, (sequence_length,),
                               dtype=torch.int32))
    forked = torch.where(neg_choice, neg_refs, forked)
    return forked, last_step


@pytest.mark.parametrize("sequence_length,b_length", CASES)
def test_replace_token(sequence_length, b_length):
    torch.manual_seed(2026)
    forked, last_step = _make_inputs(sequence_length, b_length)

    out_ref = replace_token_golden(forked, last_step)

    forked_npu = forked.npu()
    last_step_npu = last_step.npu()
    out_npu = custom_ops.replace_token_npu(forked_npu, last_step_npu)

    out_cpu = out_npu.cpu().to(torch.int32)

    assert torch.equal(out_cpu, out_ref), (
        f"replace_token mismatch for shape "
        f"(seq={sequence_length}, b={b_length})\n"
        f"ref={out_ref}\nout={out_cpu}")