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


# The HcPost kernel fuses a per-token combination step.  For every batch/token
# element b and output head hc it computes (all math done in fp32):
#
#   y[b, hc, d] = sum_j residual[b, j, d] * comb[b, j, hc]
#               + x[b, d] * post[b, hc]
#
# where the last (hc) dimension has a fixed width of HC = 4 (the kernel loads
# exactly 4 comb / residual registers).  Shapes:
#   x        : [bs, d]
#   residual : [bs, hc, d]
#   post     : [bs, hc]
#   comb     : [bs, hc, hc]     (comb[b, j, hc] -> row j, col hc)
#   y        : [bs, hc, d]      (same shape / dtype as residual)
HC = 4


def hc_post_golden(x, residual, post, comb):
    xf = x.float()             # [bs, d]
    rf = residual.float()      # [bs, hc, d]
    pf = post.float()          # [bs, hc]
    cf = comb.float()          # [bs, hc(j), hc(out)]

    # sum_j residual[b, j, d] * comb[b, j, out] -> [bs, out, d]
    # einsum: b j d, b j o -> b o d
    combined = torch.einsum("bjd,bjo->bod", rf, cf)
    # x[b, d] * post[b, out] -> [bs, out, d]
    x_term = torch.einsum("bd,bo->bod", xf, pf)
    return combined + x_term


# 10 shapes: vary batch (bs) and feature dim (d); hc fixed at HC=4.
# NOTE: the kernel processes the feature (d) dimension in units of 64 fp32
# elements (256B vector repeat); d must therefore be a multiple of 64 for the
# result to be bit-accurate.  Real hidden dims are always 64-aligned, so every
# test shape uses a multiple of 64.  d also spans below / around / above the
# 2048 tiling split.
CASES = [
    (1,    64),
    (2,   128),
    (4,   192),
    (8,   256),
    (16,  512),
    (3,  1024),
    (5,  2048),
    (7,   320),
    (10, 3072),
    (6,  4096),
]


def _make_inputs(bs, d, dtype):
    torch.manual_seed(2026)
    x = (torch.rand(bs, d, dtype=torch.float32) - 0.5)
    residual = (torch.rand(bs, HC, d, dtype=torch.float32) - 0.5)
    post = (torch.rand(bs, HC, dtype=torch.float32) - 0.5)
    comb = (torch.rand(bs, HC, HC, dtype=torch.float32) - 0.5)
    return (x.to(dtype), residual.to(dtype), post.to(dtype), comb.to(dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bs,d", CASES)
def test_hc_post(bs, d, dtype):
    x, residual, post, comb = _make_inputs(bs, d, dtype)

    y_ref = hc_post_golden(x, residual, post, comb)

    y_npu = custom_ops.hc_post_npu(
        x.npu(), residual.npu(), post.npu(), comb.npu())
    y_out = y_npu.cpu().float()

    if dtype == torch.float16:
        atol, rtol = 4e-3, 4e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-4, 1e-4
    torch.testing.assert_close(y_out, y_ref, atol=atol, rtol=rtol)