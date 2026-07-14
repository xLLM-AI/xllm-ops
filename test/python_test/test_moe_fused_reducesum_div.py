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


def moe_fused_reducesum_div_golden(x):
    """CPU reference matching the AICore kernel forward logic.

    The kernel casts the input to fp32, computes the sum along the last
    dimension for every row, then multiplies each element by 1/sum
    (i.e. row-wise normalization: output = x / sum(x, dim=-1)).
    The result is cast back to the input dtype on output.
    """
    xf = x.float()
    row_sum = xf.sum(dim=-1, keepdim=True)
    out = xf * (1.0 / row_sum)
    return out


# 10 shapes covering different row / column combinations.
# The last dim (n) is the reduce/normalize axis; leading dims are flattened
# to rows by the kernel (it sums all leading dims to form m).
CASES = [
    (1,    8),
    (2,   16),
    (4,   32),
    (8,   64),
    (16, 128),
    (32, 256),
    (3,   24),
    (5,   48),
    (7,   96),
    (10, 200),
]


def _make_inputs(rows, n, dtype):
    # Use strictly positive values so the row sum is well away from zero,
    # keeping the reciprocal numerically stable for fp16/bf16.
    base = torch.rand(rows, n, dtype=torch.float32) + 0.5
    return base.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows,n", CASES)
def test_moe_fused_reducesum_div(rows, n, dtype):
    torch.manual_seed(2026)
    x = _make_inputs(rows, n, dtype)

    y_ref = moe_fused_reducesum_div_golden(x)

    x_npu = x.npu()
    y_npu = custom_ops.moe_fused_reducesum_div_npu(x_npu)

    y_out = y_npu.cpu().float()

    atol = 4e-3 if dtype == torch.float16 else 1e-2
    rtol = 4e-3 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(y_out, y_ref, atol=atol, rtol=rtol)