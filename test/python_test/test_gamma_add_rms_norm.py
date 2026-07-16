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


def gamma_add_rms_norm_golden(x1, x2, gamma, eps, add_gamma_offset):
    dtype = x1.dtype
    if dtype == torch.float16:
        x = x1 + x2
    else:
        x = (x1.float() + x2.float()).to(dtype)

    adjusted_gamma = gamma
    if add_gamma_offset:
        adjusted_gamma = (gamma + 1).to(dtype)

    x_float = x.float()
    variance = (x_float * x_float).mean(
        dim=tuple(range(x.dim() - gamma.dim(), x.dim())), keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    normalized = (x_float * rstd).to(dtype)
    y = (normalized * adjusted_gamma).to(dtype)
    return y, rstd.float(), x


# (x shape, gamma shape) covers decode, prefill, and multi-dimensional gamma.
CASES = [
    ((1, 1024), (1024,)),
    ((4, 2048), (2048,)),
    ((16, 5120), (5120,)),
    ((2, 3, 256), (3, 256)),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("add_gamma_offset", [False, True])
@pytest.mark.parametrize("x_shape,gamma_shape", CASES)
def test_gamma_add_rms_norm(x_shape, gamma_shape, add_gamma_offset, dtype):
    torch.manual_seed(20260716)
    eps = 1e-6
    x1 = (torch.rand(x_shape, dtype=torch.float32) - 0.5).to(dtype)
    x2 = (torch.rand(x_shape, dtype=torch.float32) - 0.5).to(dtype)
    gamma = (torch.rand(gamma_shape, dtype=torch.float32) - 0.5).to(dtype)

    y_ref, rstd_ref, x_ref = gamma_add_rms_norm_golden(
        x1, x2, gamma, eps, add_gamma_offset)
    y_npu, rstd_npu, x_npu = custom_ops.gamma_add_rms_norm_npu(
        x1.npu(),
        x2.npu(),
        gamma.npu(),
        eps,
        add_gamma_offset,
    )
    torch.npu.synchronize()

    if dtype == torch.float16:
        atol = rtol = 1e-3
    elif dtype == torch.bfloat16:
        atol = rtol = 5e-3
    else:
        atol = rtol = 1e-5

    assert tuple(rstd_npu.shape) == tuple(rstd_ref.shape)
    torch.testing.assert_close(
        x_npu.cpu().float(), x_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        rstd_npu.cpu().float(), rstd_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        y_npu.cpu().float(), y_ref.float(), atol=atol, rtol=rtol)
