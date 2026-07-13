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


def add_rms_norm_bias_golden(x1, x2, gamma, beta, eps):
    """CPU reference matching the AICore AddRmsNormBias kernel.

    x    = x1 + x2                         (same dtype as x1)
    rstd = 1 / sqrt(mean(x^2, last dim) + eps)   (fp32, last dim -> 1)
    y    = x * rstd * gamma + beta         (cast back to x1 dtype)

    All reductions are computed in fp32, results cast back to input dtype.
    Returns (y, rstd, x).
    """
    orig_dtype = x1.dtype
    xf = x1.float() + x2.float()
    var = (xf * xf).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    yf = xf * rstd * gamma.float()
    if beta is not None:
        yf = yf + beta.float()
    y = yf.to(orig_dtype)
    x = xf.to(orig_dtype)
    return y, rstd, x


# (batch, D, has_beta)
CASES = [
    (1,    64, True),
    (2,   128, False),
    (4,   256, True),
    (8,   512, False),
    (16, 1024, True),
    (3,   768, False),
    (5,  2048, True),
    (7,   320, False),
    (10, 4096, True),
    (6,  1536, False),
]


def _make_inputs(batch, d, has_beta, dtype):
    x1 = (torch.rand(batch, d, dtype=torch.float32) - 0.5).to(dtype)
    x2 = (torch.rand(batch, d, dtype=torch.float32) - 0.5).to(dtype)
    gamma = (torch.rand(d, dtype=torch.float32) - 0.5).to(dtype)
    beta = (torch.rand(d, dtype=torch.float32) - 0.5).to(dtype) if has_beta else None
    return x1, x2, gamma, beta


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,d,has_beta", CASES)
def test_add_rms_norm_bias(batch, d, has_beta, dtype):
    torch.manual_seed(2026)
    eps = 1e-6

    x1, x2, gamma, beta = _make_inputs(batch, d, has_beta, dtype)

    y_ref, rstd_ref, x_ref = add_rms_norm_bias_golden(x1, x2, gamma, beta, eps)

    y_npu, rstd_npu, x_npu = custom_ops.add_rms_norm_bias_npu(
        x1.npu(),
        x2.npu(),
        gamma.npu(),
        beta.npu() if beta is not None else None,
        eps,
    )

    y_out = y_npu.cpu().float()
    rstd_out = rstd_npu.cpu().float()
    x_out = x_npu.cpu().float()

    if dtype == torch.float16:
        atol = rtol = 4e-3
    elif dtype == torch.bfloat16:
        atol = rtol = 1e-2
    else:
        atol = rtol = 1e-4

    torch.testing.assert_close(x_out, x_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(rstd_out, rstd_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_out, y_ref.float(), atol=atol, rtol=rtol)