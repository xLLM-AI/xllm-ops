"""CPU golden vs NPU test for the hc_pre_sinkhorn custom op.

hc_pre_sinkhorn per-token fused gating:
  Inputs
    mixes    fp32 (bs, hcMix)   hcMix = 2*M + M*M, split into [pre(M) | post(M) | comb(M*M)]
    rsqrt    fp32 (bs,)         per-row scalar
    hc_scale fp32 (3,)          [scale_pre, scale_post, scale_comb]
    hc_base  fp32 (hcMix,)      split [base_pre(M) | base_post(M) | base_comb(M*M)]
    x        bf16 (bs, M, d)    per-(row, mult) feature
  Attr
    hc_mult (M), hc_sinkhorn_iters (iterTimes), hc_eps (eps)
  Outputs
    y         bf16 (bs, d)          sum_m pre[b,m] * x[b,m,:]
    post      fp32 (bs, M)          sigmoid(mix1*rsqrt*scale1 + base1) * 2
    comb_frag fp32 (bs, M, M)       sinkhorn-normalized softmax matrix
"""
import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def hc_pre_sinkhorn_golden(mixes, rsqrt, hc_scale, hc_base, x,
                           hc_mult, iters, eps):
    """Reference implementation mirroring the AscendC kernel exactly (fp32)."""
    M = hc_mult
    bs = mixes.shape[0]
    m32 = mixes.float()
    r = rsqrt.float().view(bs, 1)                   # (bs,1)
    s0, s1, s2 = [float(v) for v in hc_scale.float().tolist()]
    base = hc_base.float()
    base0 = base[0:M].view(1, M)                       # pre
    base1 = base[M:2 * M].view(1, M)                   # post
    base2 = base[2 * M:2 * M + M * M].view(1, M, M)    # comb

    mix0 = m32[:, 0:M]                                 # (bs,M) pre
    mix1 = m32[:, M:2 * M]                             # (bs,M) post
    mix2 = m32[:, 2 * M:2 * M + M * M].view(bs, M, M)  # (bs,M,M) comb

    # ---- pre = sigmoid(mix0*rsqrt*scale0 + base0) + eps ----
    pre = torch.sigmoid(mix0 * r * s0 + base0) + eps   # (bs,M)

    # ---- post = sigmoid(mix1*rsqrt*scale1 + base1) * 2 ----
    post = torch.sigmoid(mix1 * r * s1 + base1) * 2.0  # (bs,M)

    # ---- y = sum_m pre[b,m] * x[b,m,:] ----
    xf = x.float()                                     # (bs,M,d)
    y = (xf * pre.view(bs, M, 1)).sum(dim=1)           # (bs,d)

    # ---- comb: t = mix2*rsqrt*scale2 + base2 ----
    t = mix2 * r.view(bs, 1, 1) * s2 + base2           # (bs,M,M)
    # softmax over last dim, then +eps (matches SoftmaxFP32Perf)
    t = torch.softmax(t, dim=-1) + eps
    # column normalize: divide each column by sum over dim1 (+eps)
    col = t.sum(dim=1, keepdim=True) + eps             # (bs,1,M)
    comb = t / col
    # (iters-1) sinkhorn iterations: row-normalize then column-normalize
    for _ in range(iters - 1):
        row = comb.sum(dim=-1, keepdim=True) + eps     # (bs,M,1)
        comb = comb / row
        col = comb.sum(dim=1, keepdim=True) + eps      # (bs,1,M)
        comb = comb / col

    return y.to(x.dtype), post, comb


# (bs, hc_mult, d)
CASES = [
    (1, 4, 64),
    (2, 4, 128),
    (4, 4, 256),
    (8, 4, 512),
    (16, 4, 128),
    (3, 2, 64),
    (5, 8, 256),
    (7, 4, 320),
    (6, 4, 1024),
    (10, 4, 192),
]


@pytest.mark.parametrize("bs,hc_mult,d", CASES)
def test_hc_pre_sinkhorn(bs, hc_mult, d):
    torch.manual_seed(2026)
    M = hc_mult
    hc_mix = 2 * M + M * M
    iters = 20
    eps = 1e-6

    mixes = torch.randn(bs, hc_mix, dtype=torch.float32)
    rsqrt = torch.rand(bs, dtype=torch.float32) * 0.5 + 0.5
    hc_scale = torch.rand(3, dtype=torch.float32) * 0.5 + 0.5
    hc_base = torch.randn(hc_mix, dtype=torch.float32) * 0.1
    x = torch.randn(bs, M, d, dtype=torch.bfloat16)

    y_g, post_g, comb_g = hc_pre_sinkhorn_golden(
        mixes, rsqrt, hc_scale, hc_base, x, M, iters, eps)

    y_n, post_n, comb_n = custom_ops.hc_pre_sinkhorn_npu(
        mixes.npu(), rsqrt.npu(), hc_scale.npu(), hc_base.npu(), x.npu(),
        M, iters, eps)

    torch.testing.assert_close(
        post_n.cpu().float(), post_g.float(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        comb_n.cpu().float(), comb_g.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        y_n.cpu().float(), y_g.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-x", "-q"]))