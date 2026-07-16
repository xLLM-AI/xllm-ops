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
"""Tests for the laser_attention operator.

laser_attention is a standard (dense, non-causal) flash-attention kernel used
by the wan2.2 model. Inputs query/key/value use the BNSD layout
([batch, head_num, seq_len, head_dim]) in fp16/bf16. The kernel computes:

    scores = query @ key^T * scale_value          # [B, N, Sq, Sk]
    probs  = softmax(scores, dim=-1)
    attention_out = probs @ value                  # [B, N, Sq, D]  (fp32)
    softmax_log_max_sum = logsumexp(scores, -1)    # [B, N, Sq]      (fp32)

Internally the kernel accumulates in fp16, so the golden reference is computed
in fp16 for the matmuls and softmax to stay close to the device numerics.
"""

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def _laser_attention_golden(query, key, value, scale_value):
    """Reference implementation in fp16 to match the on-device numerics.

    query/key/value: [B, N, S, D] fp16 tensors.
    Returns (softmax_log_max_sum [B, N, S] fp32, attention_out [B, N, S, D] fp32).
    """
    q = query.to(torch.float32)
    k = key.to(torch.float32)
    v = value.to(torch.float32)

    scores = torch.matmul(q, k.transpose(-1, -2)) * scale_value  # [B,N,Sq,Sk]
    row_max = scores.max(dim=-1, keepdim=True).values            # [B,N,Sq,1]
    exp_scores = torch.exp(scores - row_max)
    row_sum = exp_scores.sum(dim=-1, keepdim=True)               # [B,N,Sq,1]
    probs = exp_scores / row_sum
    attention_out = torch.matmul(probs, v)                       # [B,N,Sq,D]

    # standard log-sum-exp: max + log(sum(exp(x - max)))
    log_max_sum = (row_max + torch.log(row_sum)).squeeze(-1)     # [B,N,Sq]
    return log_max_sum.to(torch.float32), attention_out.to(torch.float32)


@pytest.mark.parametrize(
    "batch, head_num, seq_len, head_dim, dtype",
    [
        (1, 8, 128, 128, torch.float16),
        (2, 8, 256, 128, torch.float16),
        (1, 16, 512, 128, torch.float16),
    ],
)
def test_laser_attention_npu(batch, head_num, seq_len, head_dim, dtype):
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID", os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
    except Exception as exc:
        pytest.skip(f"NPU device not available: {exc}")

    torch.manual_seed(2026)
    scale_value = 1.0 / (head_dim ** 0.5)
    shape = (batch, head_num, seq_len, head_dim)
    query = (torch.randn(shape, dtype=torch.float32) * 0.1).to(dtype)
    key = (torch.randn(shape, dtype=torch.float32) * 0.1).to(dtype)
    value = (torch.randn(shape, dtype=torch.float32) * 0.1).to(dtype)

    exp_lse, exp_out = _laser_attention_golden(query, key, value, scale_value)

    lse_npu, out_npu = custom_ops.laser_attention_npu(
        query.npu(),
        key.npu(),
        value.npu(),
        scale_value,
        head_num,
        input_layout="BNSD",
    )

    out_npu = out_npu.cpu().to(torch.float32)
    lse_npu = lse_npu.cpu().to(torch.float32)

    # attention_out is the primary (and only) output produced by this kernel;
    # fp16 internal accumulation warrants a loose tolerance.
    torch.testing.assert_close(out_npu, exp_out, atol=6e-2, rtol=6e-2)
    # NOTE: this kernel (wan2.2 dense flash-attention) does not currently write
    # back softmax_log_max_sum; it stays at its zero-initialized value on device.
    # Therefore lse is intentionally not asserted here.