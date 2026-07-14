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


def moe_fused_add_topk_golden(x, add_num, group_num, group_topk, top_n,
                              top_k, is_norm, scale):
    """CPU reference matching the AICore kernel forward logic.

    Flow (per row):
      sig      = sigmoid(x)                       # pure sigmoid, fp32
      scores   = sig + add_num                    # used for group/topk scoring
      group scoring: split scores into group_num groups; each group's score
        is the sum of its top_n largest elements.
      group topk: pick the group_topk highest-scoring groups; elements that
        belong to the non-selected groups are zeroed out in scores.
      topk: on the masked scores, take top_k -> indices
      y = gather(sig, indices)                    # y comes from pure sigmoid
      if is_norm: y = y / sum(y) * scale
    """
    batch, expert = x.shape
    assert expert % group_num == 0
    group_eles = expert // group_num

    xf = x.float()
    add_f = add_num.float()

    sig = torch.sigmoid(xf)
    scores = sig + add_f  # [batch, expert]

    scores_g = scores.reshape(batch, group_num, group_eles)
    # each group score = sum of its top_n largest values
    top_n_vals, _ = torch.topk(scores_g, top_n, dim=-1, largest=True, sorted=True)
    group_score = top_n_vals.sum(dim=-1)  # [batch, group_num]

    # select group_topk groups
    _, sel_group = torch.topk(group_score, group_topk, dim=-1,
                              largest=True, sorted=True)
    group_mask = torch.zeros(batch, group_num, dtype=torch.float32)
    group_mask.scatter_(1, sel_group, 1.0)
    elem_mask = group_mask.unsqueeze(-1).expand(batch, group_num,
                                                group_eles).reshape(batch, expert)

    masked_scores = scores * elem_mask  # non-selected groups -> 0

    # global top_k over masked scores -> indices
    _, indices = torch.topk(masked_scores, top_k, dim=-1,
                            largest=True, sorted=True)

    # gather y from pure sigmoid
    y = torch.gather(sig, 1, indices)  # [batch, top_k]

    if is_norm:
        y = y / y.sum(dim=-1, keepdim=True) * scale

    return y.to(torch.float32), indices.to(torch.int32)


# 10 shapes covering different batch / expert / group / topk / norm / dtype
CASES = [
    # (batch, expert, group_num, group_topk, top_n, top_k, is_norm, scale)
    (1,   8,  2, 1, 2, 2, False, 1.0),
    (2,  16,  4, 2, 2, 4, True,  1.0),
    (4,  32,  8, 4, 2, 8, True,  2.0),
    (8,  64,  8, 3, 2, 6, False, 1.0),
    (16, 64,  4, 2, 4, 8, True,  1.5),
    (32, 128, 8, 4, 4, 8, True,  1.0),
    (3,  48,  6, 3, 2, 6, False, 1.0),
    (5,  24,  3, 2, 3, 4, True,  0.5),
    (7,  96,  8, 5, 3, 10, True, 1.0),
    (10, 128, 16, 8, 2, 16, False, 1.0),
]


def _make_inputs(batch, expert, dtype):
    # use distinct values (small unique perturbation) to avoid tie-break
    # ambiguity in topk/sort between golden and kernel.
    base = torch.randn(batch, expert, dtype=torch.float32)
    tie_break = torch.arange(batch * expert, dtype=torch.float32).reshape(
        batch, expert) * 1e-4
    x = (base + tie_break).to(dtype)
    add_num = (torch.randn(expert, dtype=torch.float32) * 0.1).to(dtype)
    return x, add_num


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,expert,group_num,group_topk,top_n,top_k,is_norm,scale", CASES)
def test_moe_fused_add_topk(batch, expert, group_num, group_topk, top_n,
                            top_k, is_norm, scale, dtype):
    torch.manual_seed(2026)
    x, add_num = _make_inputs(batch, expert, dtype)

    y_ref, idx_ref = moe_fused_add_topk_golden(
        x, add_num, group_num, group_topk, top_n, top_k, is_norm, scale)

    x_npu = x.npu()
    add_npu = add_num.npu()
    y_npu, idx_npu = custom_ops.moe_fused_add_topk_npu(
        x_npu, add_npu, group_num, group_topk, top_n, top_k,
        activate_type=0, is_norm=is_norm, scale=scale)

    y_out = y_npu.cpu().float()
    idx_out = idx_npu.cpu().to(torch.int32)

    # indices must match exactly (data is constructed tie-free)
    assert torch.equal(idx_out, idx_ref), (
        f"indices mismatch\nref={idx_ref}\nout={idx_out}")

    atol = 4e-3 if dtype == torch.float16 else 1e-2
    rtol = 4e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(y_out, y_ref, atol=atol, rtol=rtol)