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


def moe_gating_top_k_golden(x, bias, k, renorm, norm_type,
                            routed_scaling_factor, eps):
    """CPU reference matching the without-group AICore kernel logic.

    Flow (per row, group_count=1 / k_group=1 / group_select_mode=0):
      norm_type == 0: xnorm = softmax(x)          (fp32)
      norm_type == 1: xnorm = sigmoid(x)          (fp32)
      score_with_bias = xnorm + bias  (if bias) else xnorm
      expert_idx = argsort(score_with_bias, desc)[:k]     # top-k experts
      y = gather(xnorm, expert_idx)               # score from pure xnorm
      needRenorm = (norm_type == 1) or (norm_type == 0 and renorm == 1)
      if needRenorm: y = y / (sum(y) + eps)
      y = y * routed_scaling_factor
      out = xnorm   (full expert dim, always returned)
    """
    xf = x.float()

    if norm_type == 1:
        xnorm = torch.sigmoid(xf)
    else:
        xnorm = torch.softmax(xf, dim=-1)

    if bias is not None:
        score = xnorm + bias.float().unsqueeze(0)
    else:
        score = xnorm

    # top-k experts by score (descending)
    _, expert_idx = torch.topk(score, k, dim=-1, largest=True, sorted=True)

    y = torch.gather(xnorm, 1, expert_idx)

    need_renorm = (norm_type == 1) or (norm_type == 0 and renorm == 1)
    if need_renorm:
        y = y / (y.sum(dim=-1, keepdim=True) + eps)
    y = y * routed_scaling_factor

    return (y.to(torch.float32),
            expert_idx.to(torch.int32),
            xnorm.to(torch.float32))


# 10 shapes: (rows, expert_num, k, renorm, norm_type, scale, has_bias)
CASES = [
    (1,    8,  2, 0, 0, 1.0, False),
    (2,   16,  4, 1, 0, 1.0, True),
    (4,   32,  8, 0, 1, 2.0, False),
    (8,   64,  6, 1, 1, 1.5, True),
    (16,  64,  8, 1, 0, 1.0, False),
    (32, 128,  8, 0, 0, 2.5, True),
    (3,   48,  6, 1, 1, 1.0, False),
    (5,   96, 10, 0, 1, 0.5, True),
    (7,  128, 16, 1, 0, 1.0, True),
    (10, 256, 12, 0, 0, 1.0, False),
]


def _make_inputs(rows, expert, has_bias, dtype):
    # distinct values (tiny monotone perturbation) to avoid tie-break
    # ambiguity in topk/sort between golden and kernel.
    base = torch.randn(rows, expert, dtype=torch.float32)
    tie_break = torch.arange(rows * expert, dtype=torch.float32).reshape(
        rows, expert) * 1e-4
    x = (base + tie_break).to(dtype)
    bias = None
    if has_bias:
        bias = (torch.randn(expert, dtype=torch.float32) * 0.1).to(dtype)
    return x, bias


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rows,expert,k,renorm,norm_type,scale,has_bias", CASES)
def test_moe_gating_top_k(rows, expert, k, renorm, norm_type, scale,
                          has_bias, dtype):
    torch.manual_seed(2026)
    eps = 1e-20
    x, bias = _make_inputs(rows, expert, has_bias, dtype)

    y_ref, idx_ref, out_ref = moe_gating_top_k_golden(
        x, bias, k, renorm, norm_type, scale, eps)

    x_npu = x.npu()
    bias_npu = bias.npu() if bias is not None else None
    y_npu, idx_npu, out_npu = custom_ops.moe_gating_top_k_npu(
        x_npu, k,
        k_group=1, group_count=1, group_select_mode=0,
        renorm=renorm, norm_type=norm_type, out_flag=True,
        routed_scaling_factor=scale, eps=eps, bias=bias_npu)

    y_out = y_npu.cpu().float()
    idx_out = idx_npu.cpu().to(torch.int32)
    out_out = out_npu.cpu().float()

    if dtype == torch.float16:
        atol = rtol = 4e-3
    elif dtype == torch.float32:
        atol = rtol = 1e-4
    else:
        atol = rtol = 1e-2

    # out is the full-expert-dim softmax/sigmoid, order-independent.
    torch.testing.assert_close(out_out, out_ref, atol=atol, rtol=rtol)

    # For the selected top-k experts, the exact choice may differ from golden
    # at the top-k / (top-k+1) boundary when two experts share (numerically)
    # equal scores (a tie). This is a legitimate divergence, not a bug.
    # Validate in a tie-robust way per row:
    #   - experts selected by both: their gathered scores must match
    #   - experts selected by only one side: the score they carry must equal
    #     the boundary score, i.e. selecting either is equally correct.
    idx_out_l = idx_out.long()
    idx_ref_l = idx_ref.long()
    for r in range(rows):
        set_out = set(idx_out_l[r].tolist())
        set_ref = set(idx_ref_l[r].tolist())
        if set_out == set_ref:
            continue
        # symmetric difference must be ties: the full-dim scores of the
        # experts each side uniquely picked must be equal within tolerance.
        only_out = sorted(set_out - set_ref)
        only_ref = sorted(set_ref - set_out)
        s_out = out_out[r, only_out]
        s_ref = out_out[r, only_ref]
        s_out_sorted, _ = torch.sort(s_out)
        s_ref_sorted, _ = torch.sort(s_ref)
        torch.testing.assert_close(
            s_out_sorted, s_ref_sorted, atol=atol, rtol=rtol,
            msg=f"row {r}: selected experts differ but scores are NOT a "
                f"tie\nonly_out={only_out} scores={s_out.tolist()}\n"
                f"only_ref={only_ref} scores={s_ref.tolist()}")

    # gathered top-k score multiset must match regardless of ordering
    y_out_sorted, _ = torch.sort(y_out, dim=-1)
    y_ref_sorted, _ = torch.sort(y_ref, dim=-1)
    torch.testing.assert_close(y_out_sorted, y_ref_sorted,
                               atol=atol, rtol=rtol)