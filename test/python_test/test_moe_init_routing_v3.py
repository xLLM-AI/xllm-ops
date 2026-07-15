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

import numpy as np
import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


# Attribute enums (match the aclnn interface / kernel).
GATHER = 0
SCATTER = 1
CUMSUM = 0
COUNT = 1
DROPLESS = 0
QUANT_MODE_UNQUANT = -1
ACTIVE_NUM_DROPLESS = -1


def moe_init_routing_v3_golden(x, expert_idx, expert_num):
    """CPU reference for the dropless + unquant + GATHER + COUNT config.

    MoeInitRoutingV3 shares the same routing semantics as
    moe_init_routing_custom for this minimal config:

    Sorting stage:
      flat = expert_idx.reshape(-1)                # length bs*k
      sorted_idx = stable_argsort(flat)            # ascending by expert id
    Gather (GATHER mode, row_idx_type=0):
      expanded_x[i]              = x[sorted_idx[i] // k]   # move token rows
      expanded_row_idx[sorted_idx[i]] = i                 # src -> dst mapping
    Count:
      expert_tokens_count[e]     = count(flat == e)  for e in [0, expert_num)

    Returns (expanded_x, expanded_row_idx, expert_tokens_count).
    """
    bs, h = x.shape
    k = expert_idx.shape[1]
    flat = expert_idx.reshape(-1).astype(np.int64)          # (bs*k,)
    # Stable ascending sort by expert id.
    sorted_idx = np.argsort(flat, kind="stable")            # (bs*k,)

    expanded_x = x[sorted_idx // k]                          # (bs*k, h)

    expanded_row_idx = np.empty(bs * k, dtype=np.int32)
    expanded_row_idx[sorted_idx] = np.arange(bs * k, dtype=np.int32)

    expert_tokens_count = np.zeros(expert_num, dtype=np.int64)
    for e in range(expert_num):
        expert_tokens_count[e] = np.sum(flat == e)

    return expanded_x, expanded_row_idx, expert_tokens_count


# (bs, h, k, expert_num)
CASES = [
    (4,    64, 2,  4),
    (8,   128, 2,  8),
    (16,  256, 4,  8),
    (32,  512, 2, 16),
    (2,    64, 1,  4),
    (10,  128, 3,  8),
    (64,  256, 2, 32),
    (7,   768, 4, 16),
    (128, 512, 2, 64),
    (5,  1024, 6,  8),
]


def _make_inputs(bs, h, k, expert_num, dtype):
    rng = np.random.default_rng(2026)
    x = (rng.random((bs, h), dtype=np.float32) - 0.5)
    # Each row selects k experts (allow repeats across rows, unique within a row).
    expert_idx = np.zeros((bs, k), dtype=np.int32)
    for i in range(bs):
        expert_idx[i] = rng.choice(expert_num, size=k, replace=False)
    x_t = torch.from_numpy(x).to(dtype)
    expert_idx_t = torch.from_numpy(expert_idx).to(torch.int32)
    return x, x_t, expert_idx, expert_idx_t


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bs,h,k,expert_num", CASES)
def test_moe_init_routing_v3(bs, h, k, expert_num, dtype):
    torch.manual_seed(2026)

    x_np, x_t, expert_idx_np, expert_idx_t = _make_inputs(bs, h, k, expert_num, dtype)

    # golden uses the (possibly down-cast) x values so precision matches the kernel.
    x_ref_np = x_t.float().numpy()
    expanded_x_ref, expanded_row_idx_ref, count_ref = moe_init_routing_v3_golden(
        x_ref_np, expert_idx_np, expert_num
    )

    expanded_x, expanded_row_idx, count_or_cumsum, _scale = \
        custom_ops.moe_init_routing_v3_npu(
            x_t.npu(),
            expert_idx_t.npu(),
            active_num=ACTIVE_NUM_DROPLESS,
            expert_num=expert_num,
            drop_pad_mode=DROPLESS,
            expert_tokens_num_type=COUNT,
            expert_tokens_num_flag=True,
            quant_mode=QUANT_MODE_UNQUANT,
            row_idx_type=GATHER,
        )

    expanded_x_out = expanded_x.cpu().float().numpy()
    expanded_row_idx_out = expanded_row_idx.cpu().numpy().astype(np.int32)
    count_out = count_or_cumsum.cpu().numpy().astype(np.int64)

    if dtype == torch.float16:
        atol = rtol = 4e-3
    elif dtype == torch.bfloat16:
        atol = rtol = 1e-2
    else:
        atol = rtol = 1e-5

    # expert token count must match exactly.
    np.testing.assert_array_equal(count_out, count_ref)
    # row idx (src -> dst) must match exactly.
    np.testing.assert_array_equal(expanded_row_idx_out, expanded_row_idx_ref)
    # gathered rows must match within tolerance.
    np.testing.assert_allclose(
        expanded_x_out, expanded_x_ref, atol=atol, rtol=rtol
    )