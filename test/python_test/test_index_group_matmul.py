"""CPU golden vs NPU test for the index_group_matmul custom op.

index_group_matmul performs an int8 grouped (per-expert) matmul followed by
per-channel + per-token dequantization (TilingKey=0, KernelMatmulInt128Mlt512).

  Inputs
    a               int8   (M, K)    ND. M = total tokens across all groups.
    b               int8   (G, K, N) ND passed (framework transforms to NZ).
    scale           bf16   (G, N)    per-group per-channel dequant scale.
    per_token_scale float  (M,)      per-row dequant scale.
    group_list      int64  (G,)      CUMULATIVE offsets. group g owns rows
                    [group_list[g-1], group_list[g]); group_list[-1] == M.
  Output
    c               bf16   (M, N)

Semantics:
    prev = 0
    for g in range(G):
        cur = group_list[g]
        cnt = cur - prev
        acc = a[prev:cur].int32 @ b[g].int32            # (cnt,K)@(K,N) int32
        c[prev:cur] = (acc * scale[g] * per_token_scale[prev:cur, None]).bf16
        prev = cur

Kernel constraints (tiling Init): a 2-D, b 3-D, scale 2-D (G,N),
per_token_scale 1-D (M,), group_list 1-D (G,). baseN=256 / baseK=128 are
hard-coded so N must be a multiple of 256 and K a multiple of 128.

scale is bf16 (only ~8 mantissa bits) and accumulation is a large int32, so we
relax tolerance.
"""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def index_group_matmul_golden(a, b, scale, per_token_scale, group_list):
    """Reference int8 grouped matmul + dequant, computed in fp32 -> bf16."""
    M, K = a.shape
    G, _, N = b.shape
    c = torch.zeros((M, N), dtype=torch.float32)
    gl = group_list.tolist()
    prev = 0
    for g in range(G):
        cur = int(gl[g])
        cnt = cur - prev
        if cnt <= 0:
            prev = cur
            continue
        ag = a[prev:cur].to(torch.int32)             # (cnt, K)
        bg = b[g].to(torch.int32)                    # (K, N)
        acc = torch.matmul(ag, bg).float()           # (cnt, N) int matmul
        sc = scale[g].float().view(1, N)             # per-channel
        pts = per_token_scale[prev:cur].float().view(cnt, 1)  # per-token
        c[prev:cur] = acc * sc * pts
        prev = cur
    return c.to(torch.bfloat16)


def make_group_list(counts, dtype=torch.int64):
    """Build cumulative-offset group_list from per-group counts."""
    offs = []
    acc = 0
    for cnt in counts:
        acc += cnt
        offs.append(acc)
    return torch.tensor(offs, dtype=dtype)


bf16 = torch.bfloat16
# (counts_per_group, K, N). G = len(counts); M = sum(counts).
# K must be a multiple of 128, N a multiple of 256.
CASES = [
    ([16, 16], 128, 256),
    ([16], 256, 256),
    ([16, 16, 16], 128, 256),
    ([32, 32], 256, 512),
    ([16, 16], 128, 512),
    ([16, 16, 16, 16], 256, 256),
    ([48], 128, 256),
    ([16, 32], 384, 256),
    ([32], 512, 512),
    ([16, 16, 16], 256, 768),
]


@pytest.mark.parametrize("counts, K, N", CASES)
def test_index_group_matmul(counts, K, N):
    torch.manual_seed(0)
    G = len(counts)
    M = sum(counts)

    a = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    b = torch.randint(-8, 8, (G, K, N), dtype=torch.int8)
    # small scale so the fp32 output magnitude stays reasonable / bf16-friendly.
    scale = (torch.randn(G, N) * 0.01).to(bf16)
    per_token_scale = (torch.rand(M) * 0.5 + 0.75).to(torch.float32)
    group_list = make_group_list(counts, dtype=torch.int64)

    golden = index_group_matmul_golden(a, b, scale, per_token_scale, group_list)

    a_npu = a.npu()
    b_npu = b.npu()
    scale_npu = scale.npu()
    pts_npu = per_token_scale.npu()
    group_list_npu = group_list.npu()

    c_npu = custom_ops.index_group_matmul_npu(
        a_npu, b_npu, scale_npu, pts_npu, group_list_npu)
    c = c_npu.cpu()

    torch.testing.assert_close(
        c.float(), golden.float(), rtol=2e-2, atol=2e-2)