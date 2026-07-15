"""CPU golden vs NPU test for the moe_grouped_matmul custom op.

moe_grouped_matmul performs a grouped (per-expert) matmul with NO quantization
(KernelMoeGMMNoQuant), fp32 accumulation cast back to the input dtype.

  Inputs
    x           bf16/fp16 (M, K)     ND. M = total tokens across all groups.
    weight      bf16/fp16 (G, K, N)  ND. One (K, N) weight per group/expert.
                (transpose_weight=True expects (G, N, K).)
    group_list  int32/int64 (G, 2)   key-value mode: each row is
                [group_idx, count], where count is the number of x rows that
                belong to expert group_idx. Rows of x are consumed in the
                order the group_list rows appear (contiguous slices), and
                sum(count) must equal M.
  Attr
    transpose_weight  bool (default False)
  Output
    y           bf16/fp16 (M, N)

Semantics (transpose_weight = False):
    row = 0
    for (gid, cnt) in group_list:
        if cnt <= 0: break
        y[row:row+cnt] = x[row:row+cnt] @ weight[gid]      # (cnt,K)@(K,N)
        row += cnt

Only bf16/fp16 are supported (tiling rejects other dtypes). group_list must be
2-D (key-value mode). Accumulation is fp32, so we relax tolerance for large K.
"""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def moe_grouped_matmul_golden(x, weight, group_list, transpose_weight=False):
    """Reference grouped matmul computed in fp32, cast back to x.dtype."""
    M, K = x.shape
    N = weight.shape[1] if transpose_weight else weight.shape[2]
    y = torch.zeros((M, N), dtype=torch.float32)
    row = 0
    gl = group_list.tolist()
    for gid, cnt in gl:
        if cnt <= 0:
            break
        w = weight[gid].float()
        if transpose_weight:
            w = w.transpose(-1, -2)          # (N,K) -> (K,N)
        xg = x[row:row + cnt].float()        # (cnt, K)
        y[row:row + cnt] = torch.matmul(xg, w)
        row += cnt
    return y.to(x.dtype)


def make_group_list(counts, dtype=torch.int64):
    """Build a (G, 2) key-value group_list: [group_idx, count]."""
    rows = [[gid, cnt] for gid, cnt in enumerate(counts)]
    return torch.tensor(rows, dtype=dtype)


# (counts_per_group, K, N, transpose_weight, dtype)
# G = len(counts); M = sum(counts). bf16 only for reliability.
bf16 = torch.bfloat16
CASES = [
    ([4, 4], 128, 256, False, bf16),
    ([8], 256, 128, False, bf16),
    ([2, 3, 3], 256, 256, False, bf16),
    ([16, 16], 512, 256, False, bf16),
    ([10, 6, 8], 128, 512, False, bf16),
    ([1, 1, 1, 1], 256, 128, False, bf16),
    ([8, 8], 256, 256, True, bf16),
    ([5, 11], 512, 128, True, bf16),
    ([32], 1024, 256, False, bf16),
    ([12, 4, 8, 8], 256, 384, False, bf16),
]


@pytest.mark.parametrize("counts, K, N, transpose_weight, dtype", CASES)
def test_moe_grouped_matmul(counts, K, N, transpose_weight, dtype):
    torch.manual_seed(0)
    G = len(counts)
    M = sum(counts)

    x = (torch.randn(M, K) * 0.05).to(dtype)
    if transpose_weight:
        weight = (torch.randn(G, N, K) * 0.05).to(dtype)
    else:
        weight = (torch.randn(G, K, N) * 0.05).to(dtype)
    group_list = make_group_list(counts, dtype=torch.int64)

    golden = moe_grouped_matmul_golden(x, weight, group_list, transpose_weight)

    x_npu = x.npu()
    weight_npu = weight.npu()
    group_list_npu = group_list.npu()
    y_npu = custom_ops.moe_grouped_matmul_npu(
        x_npu, weight_npu, group_list_npu, transpose_weight)
    y = y_npu.cpu()

    torch.testing.assert_close(
        y.float(), golden.float(), rtol=8e-3, atol=8e-3)