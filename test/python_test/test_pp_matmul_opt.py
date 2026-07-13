"""CPU golden vs NPU test for the pp_matmul_opt custom op.

pp_matmul_opt computes c = a @ b^T with fp32 accumulation:
  Inputs
    a  bf16/fp16 (M, K)
    b  bf16/fp16 (N, K)   (stored transposed; kernel multiplies a @ b^T)
  Output
    c  bf16/fp16 (M, N)   c[i, j] = sum_k a[i, k] * b[j, k]

IMPORTANT: this is a SPECIALIZED (not general) matmul kernel. It only works
inside a narrow design domain:

1. Core/tile assignment is hard-coded: 24 AI cores process exactly
   20*42 + 4*40 = 1000 tiles of size 256x256, so the tile count must match:
       (K / 256) * (N / 256) == 1000
   A shape with tile count != 1000 indexes out of bounds (MTE address out of
   range / ACL 507015).
2. K % 256 == 0 (k0 = 256, rows = K / 256), N % 256 == 0 (n0 = 256).
3. M <= 16. The Mmad baseM is hard-coded to 16, so only the first 16 output
   rows are actually computed; M >= 17 yields garbage in the extra rows.
4. bf16 only. Although the op def also lists fp16, the kernel casts the GM
   pointers to bfloat16_t unconditionally, so fp16 inputs are mis-interpreted.

Accumulation is done in fp32 then cast (RINT) back to bf16.
"""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def pp_matmul_opt_golden(a, b):
    """Reference: c = a @ b^T computed in fp32, cast back to a.dtype."""
    c = torch.matmul(a.float(), b.float().transpose(-1, -2))
    return c.to(a.dtype)


# (M, K, N, dtype). Every case satisfies (K/256)*(N/256) == 1000 and M <= 16.
# bf16 only (the kernel casts GM pointers to bf16 regardless of the op dtype).
# (rows, cols) factor pairs of 1000 used below:
#   (8,125) (10,100) (20,50) (25,40) (40,25) (50,20) (100,10) (125,8)
CASES = [
    (1, 2048, 32000, torch.bfloat16),   # rows=8,   cols=125
    (4, 2560, 25600, torch.bfloat16),   # rows=10,  cols=100
    (8, 5120, 12800, torch.bfloat16),   # rows=20,  cols=50
    (12, 6400, 10240, torch.bfloat16),  # rows=25,  cols=40
    (16, 10240, 6400, torch.bfloat16),  # rows=40,  cols=25
    (16, 12800, 5120, torch.bfloat16),  # rows=50,  cols=20
    (8, 25600, 2560, torch.bfloat16),   # rows=100, cols=10
    (4, 32000, 2048, torch.bfloat16),   # rows=125, cols=8
    (2, 6400, 10240, torch.bfloat16),   # rows=25,  cols=40
    (16, 5120, 12800, torch.bfloat16),  # rows=20,  cols=50
]


@pytest.mark.parametrize("M,K,N,dtype", CASES)
def test_pp_matmul_opt(M, K, N, dtype):
    torch.manual_seed(2026)
    # Keep magnitudes small so bf16 accumulation error over large K stays bounded.
    a = torch.randn(M, K, dtype=torch.float32) * 0.05
    b = torch.randn(N, K, dtype=torch.float32) * 0.05
    a = a.to(dtype)
    b = b.to(dtype)

    c_golden = pp_matmul_opt_golden(a, b)

    c_npu = custom_ops.pp_matmul_opt_npu(a.npu(), b.npu()).cpu()

    assert c_npu.shape == (M, N)
    assert c_npu.dtype == dtype

    # bf16 matmul with large K: compare in fp32 with relaxed tolerance.
    rtol, atol = 8e-3, 8e-3
    torch.testing.assert_close(
        c_npu.float(), c_golden.float(), rtol=rtol, atol=atol)