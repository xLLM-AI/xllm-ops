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
    return x * torch.sigmoid(x)


def layer_norm_fwd_golden(x, weight, bias=None, z=None, eps=1e-6,
                          group_size=-1, norm_before_gate=True,
                          is_rms_norm=False):
    """CPU reference matching the AICore kernel forward logic.

    Layout: x is [..., full_n], flattened to m rows. Each row is split into
    ngroups groups of length group_size (full_n = ngroups * group_size).
    All math is done in fp32, results cast back to input dtype.
    mean/rstd are laid out group-major: index = group * m + row.
    """
    orig_dtype = x.dtype
    full_n = x.shape[-1]
    m = 1
    for s in x.shape[:-1]:
        m *= s
    gs = full_n if group_size <= 0 else group_size
    assert full_n % gs == 0
    ngroups = full_n // gs

    xf = x.reshape(m, ngroups, gs).float()
    wf = weight.reshape(ngroups, gs).float()
    bf = bias.reshape(ngroups, gs).float() if bias is not None else None
    zf = z.reshape(m, ngroups, gs).float() if z is not None else None

    mean_out = torch.zeros(ngroups, m, dtype=torch.float32)
    rstd_out = torch.zeros(ngroups, m, dtype=torch.float32)
    y = torch.empty(m, ngroups, gs, dtype=torch.float32)

    for r in range(m):
        for g in range(ngroups):
            row = xf[r, g].clone()
            if zf is not None and not norm_before_gate:
                row = row * _silu(zf[r, g])
            if not is_rms_norm:
                mean_val = row.mean()
                mean_out[g, r] = mean_val
                row = row - mean_val
            var = (row * row).mean()
            rstd_val = 1.0 / torch.sqrt(var + eps)
            rstd_out[g, r] = rstd_val
            row = row * rstd_val * wf[g]
            if bf is not None:
                row = row + bf[g]
            if zf is not None and norm_before_gate:
                row = row * _silu(zf[r, g])
            y[r, g] = row

    y = y.reshape(x.shape).to(orig_dtype)
    mean_flat = mean_out.reshape(-1) if not is_rms_norm else torch.zeros(0)
    rstd_flat = rstd_out.reshape(-1)
    return y, mean_flat, rstd_flat


# (batch, full_n, group_size, has_bias, has_z, norm_before_gate, is_rms_norm)
SHAPES = [
    (4, 128, -1, False, False, True, False),
    (8, 128, -1, True, False, True, False),
    (2, 256, 128, False, True, True, False),
    (16, 512, 128, True, True, True, False),
    (4, 128, -1, False, False, True, True),
    (8, 256, 128, False, True, True, True),
    (32, 128, -1, True, True, True, False),
    (6, 384, 128, False, False, True, True),
    (10, 512, 256, True, True, False, False),
    (3, 1024, 128, False, True, True, True),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,full_n,group_size,has_bias,has_z,norm_before_gate,is_rms_norm",
    SHAPES,
)
def test_layer_norm_fwd_npu(dtype, batch, full_n, group_size, has_bias,
                            has_z, norm_before_gate, is_rms_norm):
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    torch.manual_seed(2026)
    eps = 1e-5

    x = torch.randn(batch, full_n, dtype=dtype)
    weight = torch.randn(full_n, dtype=dtype)
    bias = torch.randn(full_n, dtype=dtype) if has_bias else None
    z = torch.randn(batch, full_n, dtype=dtype) if has_z else None

    y_ref, mean_ref, rstd_ref = layer_norm_fwd_golden(
        x, weight, bias, z, eps, group_size, norm_before_gate, is_rms_norm)

    y_npu, mean_npu, rstd_npu = custom_ops.layer_norm_fwd_npu(
        x.npu(),
        weight.npu(),
        bias.npu() if bias is not None else None,
        z.npu() if z is not None else None,
        eps,
        group_size,
        norm_before_gate,
        is_rms_norm,
    )

    y_npu = y_npu.cpu().float()
    rstd_npu = rstd_npu.cpu().float()

    atol = 4e-3 if dtype == torch.float16 else 1e-2
    rtol = 4e-3 if dtype == torch.float16 else 1e-2

    assert torch.allclose(y_npu, y_ref.float(), atol=atol, rtol=rtol)
    assert torch.allclose(rstd_npu, rstd_ref, atol=1e-2, rtol=1e-2)
    if not is_rms_norm:
        mean_npu = mean_npu.cpu().float()
        assert torch.allclose(mean_npu, mean_ref, atol=1e-2, rtol=1e-2)