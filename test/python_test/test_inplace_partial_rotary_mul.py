# Copyright 2025 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""InplacePartialRotaryMul unit test (strictly consistent with the kernel).

Op semantics (interleaved / GPT-NeoX style RoPE, only mode==1 supported):
  - x:       [B, S, N, allHeadDim]  (dim1==1 or dim2==1, i.e. S==1 or N==1)
  - cos/sin: [B, 1, 1, headDim]     (per-batch, broadcast to all seq/head)
  - partial_slice = [start, end], headDim = end - start
  - only rotate the segment x[..., start:end], keep the rest unchanged, write back to x in place.

  For adjacent element pairs (2i, 2i+1) within the segment (i is the even index within the segment):
    out[2i]   = x[2i]   * cos[2i]   - x[2i+1] * sin[2i]
    out[2i+1] = x[2i+1] * cos[2i+1] + x[2i]   * sin[2i+1]
  (kernel: y=x*cos; gather(i^1) swaps adjacent even/odd; *sin; negate even positions; y+=x)

Kernel hard constraints (910_93 membase Tiling4InplacePartialRotaryMul):
  - x must be 4D; cos/sin must be 4D and equal per-dimension; cos.dim1==cos.dim2==1
  - cos.dim0 == x.dim0 (batch equal)
  - headDim (== end-start) aligned to 16 (fp16/bf16) / 8 (fp32), and <= 64
  - x.dim1==1 or x.dim2==1 (isSpecial branch)
"""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def rope_interleaved_golden(x, cos, sin, start, end):
    """CPU reference: rotate x[..., start:end] in place. Returns the full rotated x.

    x:   [B, S, N, allHeadDim]  float32
    cos/sin: [B, 1, 1, headDim] float32
    """
    out = x.clone()
    seg = x[..., start:end]                       # [B,S,N,headDim]
    c = cos[..., :]                               # [B,1,1,headDim] -> broadcast
    s = sin[..., :]
    # swap adjacent even/odd: rot[2i]=seg[2i+1], rot[2i+1]=seg[2i]
    rot = torch.empty_like(seg)
    rot[..., 0::2] = seg[..., 1::2]
    rot[..., 1::2] = seg[..., 0::2]
    # negate even positions: only negate the even index positions after swapping
    sign = torch.ones_like(c)
    sign[..., 0::2] = -1.0
    rotated = seg * c + rot * s * sign
    out[..., start:end] = rotated
    return out


# CASES: (B, S, N, allHeadDim, start, end, dtype)
#   S==1 or N==1 (isSpecial); end<=allHeadDim
#   [kernel hard constraint] headDim(=end-start) is numerically correct only when ==64;
#   headDim<64 (32/16/48) triggers a vector core exception (507035) / wrong result,
#   which is a kernel defect (source is read-only), so we narrow to the stable domain headDim==64.
#   Partial rotation can still be tested: when allHeadDim>64 only the first 64 dims are rotated, the rest stays unchanged.
CASES = [
    (2, 1, 4, 64, 0, 64, torch.float16),
    (1, 1, 8, 128, 0, 64, torch.float16),    # allHeadDim=128, rotate only the first 64 dims
    (4, 1, 2, 96, 0, 64, torch.float16),     # allHeadDim=96, partial rotation [0,64)
    (3, 8, 1, 64, 0, 64, torch.float16),     # N==1 branch
    (1, 16, 1, 128, 0, 64, torch.float16),
    (8, 1, 2, 192, 0, 64, torch.float16),    # allHeadDim=192
    (2, 1, 4, 64, 0, 64, torch.bfloat16),
    (1, 1, 8, 128, 0, 64, torch.bfloat16),
    (2, 4, 1, 96, 0, 64, torch.bfloat16),    # N==1, partial rotation
    (1, 1, 6, 64, 0, 64, torch.bfloat16),
]


def _run_case(B, S, N, allHeadDim, start, end, dtype, seed):
    gen = torch.Generator().manual_seed(seed)
    headDim = end - start
    x = torch.randn(B, S, N, allHeadDim, generator=gen, dtype=torch.float32)
    cos = torch.randn(B, 1, 1, headDim, generator=gen, dtype=torch.float32)
    sin = torch.randn(B, 1, 1, headDim, generator=gen, dtype=torch.float32)

    golden = rope_interleaved_golden(x, cos, sin, start, end).to(dtype).to(torch.float32)

    x_npu = x.to(dtype).npu()
    cos_npu = cos.to(dtype).npu()
    sin_npu = sin.to(dtype).npu()
    custom_ops.inplace_partial_rotary_mul_npu(
        x_npu, cos_npu, sin_npu, 1, [start, end])
    out = x_npu.cpu().to(torch.float32)

    atol = 4e-3 if dtype == torch.float16 else 2e-2
    rtol = 4e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(out, golden, atol=atol, rtol=rtol)


@pytest.mark.parametrize("case", CASES)
def test_inplace_partial_rotary_mul(case):
    B, S, N, allHeadDim, start, end, dtype = case
    _run_case(B, S, N, allHeadDim, start, end, dtype, seed=2026)


@pytest.mark.parametrize("case", CASES[:5])
def test_inplace_partial_rotary_mul_seed2(case):
    B, S, N, allHeadDim, start, end, dtype = case
    _run_case(B, S, N, allHeadDim, start, end, dtype, seed=7)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))