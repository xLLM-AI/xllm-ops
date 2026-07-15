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


def _xnorm(xf, norm_type):
    # norm_type: SOFTMAX=0 / SIGMOID=1 / SOFTPLUS=2
    if norm_type == 0:
        return torch.softmax(xf, dim=-1)
    if norm_type == 1:
        return torch.sigmoid(xf)
    # softplus variant used by the kernel: sqrt(ln(1 + exp(x)))
    return torch.sqrt(torch.log1p(torch.exp(xf)))


def moe_gating_top_k_hash_golden(x, bias, input_ids, tid2eid, k,
                                 norm_type, routed_scaling_factor, eps):
    """CPU reference matching the without-group hash AICore kernel.

    Per row (group_count=1 / k_group=1 / group_select_mode=0 / renorm=0):
      xnorm = softmax/sigmoid/softplus(x)  (fp32)
      hashFlag (input_ids & tid2eid given):
          expert_idx[row] = tid2eid[input_ids[row]*k : input_ids[row]*k + k]
      else (topk):
          score = xnorm + bias (if bias) ; expert_idx = topk(score, k) desc
      y = gather(xnorm, expert_idx)
      if norm_type in (1, 2): y = y / (y.sum(-1) + eps)   # renorm
      y = y * routed_scaling_factor
      out = xnorm  (rows, expert_num) fp32
    """
    xf = x.float()
    xnorm = _xnorm(xf, norm_type)

    if input_ids is not None and tid2eid is not None:
        ids = input_ids.long()
        t2e = tid2eid.long().reshape(-1)
        rows = xnorm.shape[0]
        expert_idx = torch.stack(
            [t2e[ids[r] * k: ids[r] * k + k] for r in range(rows)], dim=0
        ).to(torch.int32)
    else:
        if bias is not None:
            score = xnorm + bias.float().unsqueeze(0)
        else:
            score = xnorm
        _, expert_idx = torch.topk(score, k, dim=-1, largest=True, sorted=True)
        expert_idx = expert_idx.to(torch.int32)

    y = torch.gather(xnorm, 1, expert_idx.long())

    if norm_type in (1, 2):
        y = y / (y.sum(dim=-1, keepdim=True) + eps)
    y = y * routed_scaling_factor

    return (y.to(torch.float32),
            expert_idx.to(torch.int32),
            xnorm.to(torch.float32))


# 10 shapes: (rows, expert_num, k, norm_type, scale, has_bias, use_hash)
CASES = [
    (1,    8,  2, 0, 1.0, False, False),
    (2,   16,  4, 1, 1.0, True,  False),
    (4,   32,  8, 0, 2.0, False, True),
    (8,   64,  6, 1, 1.5, True,  True),
    (16,  64,  8, 0, 1.0, False, True),
    (32, 128,  8, 1, 2.5, True,  False),
    (3,   48,  6, 0, 1.0, False, True),
    (5,   96, 10, 1, 0.5, True,  True),
    (7,  128, 16, 0, 1.0, True,  False),
    (10, 256, 12, 1, 1.0, False, True),
]


def _make_inputs(rows, expert, k, has_bias, use_hash, dtype):
    # distinct values (tiny monotone perturbation) to avoid tie-break
    # ambiguity in topk/sort between golden and kernel.
    base = torch.randn(rows, expert, dtype=torch.float32)
    tie_break = torch.arange(rows * expert, dtype=torch.float32).reshape(
        rows, expert) * 1e-4
    x = (base + tie_break).to(dtype)

    bias = None
    if has_bias:
        bias = (torch.randn(expert, dtype=torch.float32) * 0.1).to(dtype)

    input_ids = None
    tid2eid = None
    if use_hash:
        # token id -> expert-id table. Use a modest vocab; each token maps to
        # k distinct experts in [0, expert). int32 to hit <int32,int32> branch.
        vocab = max(rows * 2, 8)
        input_ids = torch.randint(0, vocab, (rows,), dtype=torch.int32)
        table = torch.empty(vocab, k, dtype=torch.int32)
        for v in range(vocab):
            table[v] = torch.randperm(expert)[:k].to(torch.int32)
        tid2eid = table.reshape(-1).contiguous()
    return x, bias, input_ids, tid2eid


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rows,expert,k,norm_type,scale,has_bias,use_hash", CASES)
def test_moe_gating_top_k_hash(rows, expert, k, norm_type, scale,
                               has_bias, use_hash, dtype):
    torch.manual_seed(2026)
    eps = 1e-20
    x, bias, input_ids, tid2eid = _make_inputs(
        rows, expert, k, has_bias, use_hash, dtype)

    y_ref, idx_ref, out_ref = moe_gating_top_k_hash_golden(
        x, bias, input_ids, tid2eid, k, norm_type, scale, eps)

    x_npu = x.npu()
    bias_npu = bias.npu() if bias is not None else None
    ids_npu = input_ids.npu() if input_ids is not None else None
    t2e_npu = tid2eid.npu() if tid2eid is not None else None

    y_npu, idx_npu, out_npu = custom_ops.moe_gating_top_k_hash_npu(
        x_npu, k, bias=bias_npu, input_ids=ids_npu, tid2eid=t2e_npu,
        k_group=1, group_count=1, group_select_mode=0,
        renorm=0, norm_type=norm_type, out_flag=True,
        routed_scaling_factor=scale, eps=eps)

    y_out = y_npu.cpu().float()
    idx_out = idx_npu.cpu().to(torch.int32)
    out_out = out_npu.cpu().float()

    # indices must match exactly (tie-free inputs / deterministic hash table)
    assert torch.equal(idx_out, idx_ref), (
        f"indices mismatch\nref={idx_ref}\nout={idx_out}")

    if dtype == torch.float16:
        atol = rtol = 4e-3
    elif dtype == torch.float32:
        atol = rtol = 1e-4
    else:
        atol = rtol = 1e-2

    torch.testing.assert_close(y_out, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_out, out_ref, atol=atol, rtol=rtol)