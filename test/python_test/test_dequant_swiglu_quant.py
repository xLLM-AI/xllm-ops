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


def _silu(x):
    # silu(x) = x * sigmoid(x) = x / (1 + exp(-x)); matches kernel beta=-1.0.
    return x / (1.0 + torch.exp(-x))


def dequant_swiglu_quant_golden(x, activate_left=False):
    """CPU reference for the bf16/fp16 dynamic-quant path (no bias/scale).

    x:[rows, H] (H even). Split along last dim into two halves A/B.
    kernel BaseProcess: activate_left==0 (default) -> gate=A=x[:, H/2:],
                        up=B=x[:, 0:H/2].
    swiglu = silu(gate) * up            -> [rows, H/2]  (computed in fp32)
    dynamic per-token quant:
        value = max(|swiglu|, axis=-1) / 127     (this is the output scale)
        y = round_half_even(swiglu / value) -> clip[-128,127] -> int8
    Returns (y_int8[rows, H/2], scale_fp32[rows]).
    """
    xf = x.float()
    H = xf.shape[-1]
    half = H // 2
    if activate_left:
        gate = xf[..., :half]
        up = xf[..., half:]
    else:
        gate = xf[..., half:]
        up = xf[..., :half]
    swi = _silu(gate) * up                      # [rows, half] fp32
    amax = swi.abs().amax(dim=-1)               # [rows]
    value = amax / 127.0                        # output scale (per-token)
    scaled = swi / value.unsqueeze(-1)
    # kernel casts with RINT (round half to even) then to int8.
    q = torch.round(scaled)                     # torch.round is round-half-even
    q = torch.clamp(q, -128, 127)
    y = q.to(torch.int8)
    return y, value


def dequant_swiglu_quant_int32_golden(x, weight_scale, activation_scale,
                                      activate_left=False):
    """CPU reference for the int32 dequant path (dynamic quant, no bias).

    x:[rows, H] int32 (matmul accumulator). Dequant per element:
        deq = x * weight_scale[col] * activation_scale[row]
    then split A/B, swiglu, dynamic per-token quant (same as above).
    weight_scale:[H] fp32, activation_scale:[rows] fp32.
    """
    xf = x.float()
    ws = weight_scale.float().unsqueeze(0)          # [1, H]
    act = activation_scale.float().unsqueeze(-1)    # [rows, 1]
    deq = xf * ws * act                             # [rows, H]
    return dequant_swiglu_quant_golden(deq, activate_left=activate_left)


# (rows, H)  -- H must be even; gate/up each H/2.
CASES = [
    (1,    256),
    (2,    512),
    (4,    128),
    (8,   1024),
    (16,   768),
    (32,   256),
    (128,  512),
    (3,    320),
    (7,   2048),
    (64,  1536),
]


def _run(x, activate_left, quant_mode="dynamic"):
    y_npu, scale_npu = custom_ops.dequant_swiglu_quant_npu(
        x.npu(),
        activate_left=activate_left,
        quant_mode=quant_mode,
    )
    return y_npu.cpu(), scale_npu.cpu().float()


def _assert_quant_close(y_out, y_ref, scale_out, scale_ref):
    # scale (fp32 per-token max/127) should match tightly.
    torch.testing.assert_close(scale_out, scale_ref.float(), atol=2e-2, rtol=2e-2)
    # int8 output: allow off-by-one on a small fraction of elements (rounding
    # boundary differences between kernel RINT and torch round).
    diff = (y_out.int() - y_ref.int()).abs()
    mismatch = (diff > 1).float().mean().item()
    assert mismatch < 5e-3, f"too many int8 mismatches > 1: {mismatch}"
    max_diff = diff.max().item()
    assert max_diff <= 2, f"int8 max abs diff too large: {max_diff}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,H", CASES)
def test_dequant_swiglu_quant_float(rows, H, dtype):
    torch.manual_seed(2026)
    x = (torch.rand(rows, H, dtype=torch.float32) * 4.0 - 2.0).to(dtype)

    y_ref, scale_ref = dequant_swiglu_quant_golden(x, activate_left=False)
    y_out, scale_out = _run(x, activate_left=False)

    assert list(y_out.shape) == [rows, H // 2]
    assert list(scale_out.shape) == [rows]
    _assert_quant_close(y_out, y_ref, scale_out, scale_ref)


@pytest.mark.parametrize("rows,H", CASES[:5])
def test_dequant_swiglu_quant_activate_left(rows, H):
    torch.manual_seed(7)
    x = (torch.rand(rows, H, dtype=torch.float32) * 4.0 - 2.0).to(torch.bfloat16)

    y_ref, scale_ref = dequant_swiglu_quant_golden(x, activate_left=True)
    y_out, scale_out = _run(x, activate_left=True)

    _assert_quant_close(y_out, y_ref, scale_out, scale_ref)