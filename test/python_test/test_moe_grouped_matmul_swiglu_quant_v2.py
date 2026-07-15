"""CPU golden vs NPU test for grouped_matmul_swiglu_quant_v2.

V2 differs from V1 only in the interface shape of the grouped inputs:
  * weight / weight_scale are TensorLists (one tensor per expert), not a single
    packed (G, ...) tensor.
  * group_list is a 1D int64 tensor of length G (groupListType=0 -> per-group
    token count), instead of V1's (G, 2) [expertIdx, count] rows.
The int8 pertoken compute chain is identical to V1 (verified closed-loop):

  Inputs (pertoken int8xint8 main path)
    x            int8   (M, K)      ND. M = total tokens across groups.
    weight       list of int8 (K, 2N) per expert (fed as NZ physical layout).
    weight_scale list of float (2N,) per expert (per-channel dequant scale).
    x_scale      float  (M,)        per-token dequant scale.
    group_list   int64  (G,)        per-group token count (groupListType=0).
  Outputs
    y            int8   (M, N)      N = (2N) / 2
    y_scale      float  (M,)        per-token quant scale.

Compute chain (per group e, rows consecutive):
    acc = x_g.int32 @ weight[e].int32                      # (cnt, 2N) int32
    deq = acc * weight_scale[e] * x_scale[row]             # fp32
    gate = deq[:, :N]; up = deq[:, N:]                     # gate=first half
    swi  = gate * sigmoid(gate) * up                       # SiLU gate
    y_scale = max(|swi|, axis=-1) / 127
    y = round(swi / y_scale) -> int8

Constraints (matmul baseN=256 baseK=256): K multiple of 128, 2N multiple of 256.

The wrapper takes packed weight (G,K,2N) / weight_scale (G,2N) for convenience
and splits them into a per-expert TensorList inside the C++ impl.
"""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def weight_to_nz(weight):
    """(G,K,2N) int8 ND -> FRACTAL_NZ physical layout, reshape back to 3D.

    Each expert's (K,2N) block is rearranged independently so a later
    per-expert select(0,g) yields a valid NZ 2D block.
    """
    G, K, W = weight.shape
    K0, N0 = 16, 32
    K1, N1 = K // K0, W // N0
    nz = weight.reshape(G, K1, K0, N1, N0).permute(0, 3, 1, 2, 4).contiguous()
    return nz.reshape(G, K, W).contiguous()


def golden(x, weight, weight_scale, x_scale, group_list):
    """int8 grouped matmul + dequant + swiglu + dynamic int8 quant.

    group_list is 1D per-group token count (groupListType=0), rows consecutive.
    """
    M, K = x.shape
    G, _, W = weight.shape          # W = 2N
    N = W // 2
    y = torch.zeros((M, N), dtype=torch.int8)
    y_scale = torch.zeros((M,), dtype=torch.float32)
    counts = group_list.tolist()
    prev = 0
    for e in range(G):
        cnt = int(counts[e])
        if cnt <= 0:
            continue
        cur = prev + cnt
        xg = x[prev:cur].to(torch.int32)            # (cnt, K)
        wg = weight[e].to(torch.int32)              # (K, 2N)
        acc = torch.matmul(xg, wg).float()          # (cnt, 2N)
        sc = weight_scale[e].float().view(1, W)     # per-channel
        pts = x_scale[prev:cur].float().view(cnt, 1)  # per-token
        deq = acc * sc * pts
        gate = deq[:, :N]                           # first half
        up = deq[:, N:]                             # second half
        swi = gate * torch.sigmoid(gate) * up       # SiLU gate
        value = swi.abs().amax(dim=-1) / 127.0      # (cnt,) = y_scale
        yg = torch.round(swi / value.unsqueeze(-1)).clamp(-128, 127)
        y[prev:cur] = yg.to(torch.int8)
        y_scale[prev:cur] = value
        prev = cur
    return y, y_scale


# (counts_per_group, K, 2N). G = len(counts); M = sum(counts).
# K multiple of 128; 2N multiple of 256.
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


def _assert_quant_close(y, golden_y, scale, golden_scale):
    torch.testing.assert_close(scale.float(), golden_scale.float(),
                               rtol=2e-2, atol=2e-2)
    diff = (y.to(torch.int32) - golden_y.to(torch.int32)).abs()
    max_diff = int(diff.max()) if diff.numel() else 0
    bad_ratio = float((diff > 1).float().mean()) if diff.numel() else 0.0
    assert max_diff <= 2, f"int8 max diff {max_diff} > 2"
    assert bad_ratio < 5e-3, f"int8 diff>1 ratio {bad_ratio} too high"


@pytest.mark.parametrize("counts, K, W", CASES)
def test_grouped_matmul_swiglu_quant_v2(counts, K, W):
    torch.manual_seed(0)
    G = len(counts)
    M = sum(counts)

    x = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    weight = torch.randint(-8, 8, (G, K, W), dtype=torch.int8)
    weight_scale = (torch.randn(G, W) * 0.01).to(torch.float32)
    x_scale = (torch.rand(M) * 0.5 + 0.75).to(torch.float32)
    group_list = torch.tensor(counts, dtype=torch.int64)  # 1D count

    golden_y, golden_scale = golden(x, weight, weight_scale, x_scale,
                                    group_list)

    # weight is fed as raw ND (G,K,2N); the wrapper casts each expert to
    # FRACTAL_NZ via npu_format_cast, which does the physical repack for us.
    y_npu, scale_npu = custom_ops.grouped_matmul_swiglu_quant_v2_npu(
        x.npu(), weight.npu(), weight_scale.npu(), x_scale.npu(),
        group_list.npu(), 0)

    _assert_quant_close(y_npu.cpu(), golden_y, scale_npu.cpu(), golden_scale)