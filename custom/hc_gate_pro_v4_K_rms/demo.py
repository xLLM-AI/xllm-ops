#!/usr/bin/env python3
# coding: utf-8
"""Minimal invocation demo for hc_gate_pro (v4-K + RmsNorm fused).

Usage (inside the A5 container, from the repo root, with the CANN 9.2.0
environment already sourced):
    TILE_FWK_DEVICE_ID=0 PTO_TILE_LIB_CODE_PATH=third_party/pto-isa \
        python3 custom/hc_gate_pro_v4_K_rms/demo.py [t]

Self-contained: inputs are built in place and compared against a torch
reference chain (5-op semantics + RmsNorm).
"""

import os
import sys

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hc_gate_pro_impl import hc_gate_pro  # noqa: E402

HC, D, MIX_HC, HC_D = 4, 4096, 24, 16384
EPS, RMS_EPS = 1e-6, 1e-6


def reference(x, hc_fn, hc_scale, hc_base, gamma):
    """Torch reference chain: fused gating (fp32 compute chain) + RmsNorm."""
    xf = x.float().reshape(x.shape[0], HC_D)
    r = torch.rsqrt((xf * xf).mean(dim=-1, keepdim=True) + EPS)
    res = (xf @ hc_fn.float().T) * r
    pre = torch.sigmoid(res[:, 0:HC] * hc_scale[0] + hc_base[0:HC]) + EPS
    post = torch.sigmoid(res[:, HC:2 * HC] * hc_scale[1]
                         + hc_base[HC:2 * HC]) * 2.0
    comb_flag = (res[:, 2 * HC:].reshape(-1, HC, HC) * hc_scale[2]
                 + hc_base[2 * HC:].reshape(1, HC, HC))
    y = (x.float() * pre.reshape(-1, HC, 1)).sum(dim=1).to(torch.bfloat16)
    t0 = torch.softmax(comb_flag, dim=-1) + EPS
    comb = t0 / (t0.sum(dim=-2, keepdim=True) + EPS)
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + EPS)
    yf = y.float()
    rstd = torch.rsqrt((yf * yf).mean(dim=-1, keepdim=True) + RMS_EPS)
    y_norm = (yf * rstd * gamma.float()).to(torch.bfloat16)
    return y_norm, rstd, post, comb


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dev = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}"
    torch.npu.set_device(dev)

    # ---- Build inputs (random data, for demo purposes) ----
    g = torch.Generator(device="cpu").manual_seed(42)
    x = torch.empty((t, HC, D), dtype=torch.bfloat16).uniform_(-1, 1, generator=g)
    hc_fn = torch.empty((MIX_HC, HC_D), dtype=torch.bfloat16).uniform_(
        -1, 1, generator=g)
    hc_scale = torch.empty((3,), dtype=torch.float32).uniform_(-1, 1, generator=g)
    hc_base = torch.empty((MIX_HC,), dtype=torch.float32).uniform_(-1, 1, generator=g)
    gamma = torch.empty((D,), dtype=torch.bfloat16).uniform_(0.5, 1.5, generator=g)
    x, hc_fn = x.to(dev), hc_fn.to(dev)
    hc_scale, hc_base, gamma = (hc_scale.to(dev), hc_base.to(dev), gamma.to(dev))

    # ---- Invoke the operator (first call includes JIT compilation) ----
    y, rstd, post, comb = hc_gate_pro(x, hc_fn, hc_scale, hc_base, gamma)
    torch.npu.synchronize()
    print(f"t={t} | y_norm {tuple(y.shape)}/{y.dtype} | rstd {tuple(rstd.shape)}"
          f" | post {tuple(post.shape)} | comb {tuple(comb.shape)}")

    # ---- Compare against the reference chain (tolerances follow the
    # precision test in SPEC section 7) ----
    yr, rr, pr, cr = reference(x.cpu(), hc_fn.cpu(), hc_scale.cpu(),
                               hc_base.cpu(), gamma.cpu())
    checks = (("y_norm", y.float().cpu(), yr.float(), 1e-2, 1e-2),
              ("rstd", rstd.cpu(), rr, 1e-4, 5e-3),
              ("post", post.cpu(), pr, 2.5e-5, 5e-3),
              ("comb", comb.cpu(), cr, 2.5e-5, 5e-3))
    for name, got, ref, atol, rtol in checks:
        ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
        diff = (got - ref).abs().max().item()
        print(f"  {name:6s}: {'PASS' if ok else 'FAIL'}  max_abs={diff:.3e}")
    print("[DEMO_" + ("PASS" if all(torch.allclose(a, b, atol=c, rtol=d)
                                     for _, a, b, c, d in checks) else "FAIL") + "]")


if __name__ == "__main__":
    main()
