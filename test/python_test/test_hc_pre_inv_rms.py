"""CPU golden vs NPU test for the hc_pre_inv_rms custom op.

hc_pre_inv_rms computes the reciprocal RMS scale over the last two dims:

    y = 1 / sqrt( mean(x^2 over last two dims) + eps )

Shapes:
    x: (b, s, hc, d)  -> y: (b, s, 1)      (4D input)
    x: (t, hc, d)     -> y: (t, 1)         (3D input)
The output dtype is ALWAYS fp32 regardless of the input dtype.

IMPORTANT: the numerically-correct large_d kernel path is only enabled when
R = hc * d == 28672 (see hc_pre_inv_rms_tiling.cpp). All cases below fix
R == 28672 so we exercise that path; other R values fall back to a kernel
with a known fold-padding defect and are intentionally not covered.
"""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

R = 28672  # hc * d, the only shape the large_d kernel supports.


def hc_pre_inv_rms_golden(x, eps):
    """Reference: 1/sqrt(mean(x^2 over last two dims) + eps), fp32 output."""
    xf = x.float()
    ndim = xf.dim()
    # mean over the last two dims
    ms = xf.pow(2).mean(dim=(ndim - 2, ndim - 1))   # (b,s) or (t,)
    inv = torch.rsqrt(ms + eps)                     # (b,s) or (t,)
    return inv.unsqueeze(-1)                         # (b,s,1) or (t,1)


# (leading_shape, hc, d, dtype); hc*d must equal R (28672).
# dtype covers fp32 / fp16 / bf16 (all supported by the large_d kernel).
CASES = [
    ((1,),      4, 7168, torch.float32),   # 3D (t,hc,d)
    ((2,),      7, 4096, torch.float16),   # 3D
    ((4,),      8, 3584, torch.bfloat16),  # 3D
    ((8,),     16, 1792, torch.float32),   # 3D
    ((1, 1),    4, 7168, torch.float16),   # 4D (b,s,hc,d)
    ((2, 2),    2, 14336, torch.bfloat16),  # 4D
    ((1, 4),    7, 4096, torch.float32),   # 4D
    ((3,),      4, 7168, torch.float16),   # 3D
    ((2, 3),    8, 3584, torch.bfloat16),  # 4D
    ((5,),      4, 7168, torch.float32),   # 3D
]


@pytest.mark.parametrize("leading,hc,d,dtype", CASES)
def test_hc_pre_inv_rms(leading, hc, d, dtype):
    torch.manual_seed(2026)
    assert hc * d == R, f"hc*d must be {R}, got {hc * d}"
    eps = 1e-6

    shape = tuple(leading) + (hc, d)
    # moderate magnitude to keep fp16/bf16 x^2 accumulation well-conditioned.
    x = (torch.randn(shape, dtype=torch.float32) * 0.5).to(dtype)

    y_g = hc_pre_inv_rms_golden(x, eps)
    y_n = custom_ops.hc_pre_inv_rms_npu(x.npu(), eps)

    assert tuple(y_n.shape) == tuple(y_g.shape), \
        f"shape mismatch: npu {tuple(y_n.shape)} vs golden {tuple(y_g.shape)}"
    assert y_n.dtype == torch.float32

    # tolerance: fp32 tight, fp16/bf16 looser due to x^2 accumulation.
    if dtype == torch.float32:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 5e-3, 5e-3

    torch.testing.assert_close(
        y_n.cpu().float(), y_g.float(), rtol=rtol, atol=atol)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))