#!/usr/bin/env python3
"""Precision + wall-clock check of the 20 cann-bench cases via custom_ops."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

torch_npu = __import__("torch_npu")
ROOT = Path("/home/h00801112/codex/my-xllm-ops/xllm-ops")
sys.path.insert(0, str(ROOT / "test/python_test"))
import custom_ops  # noqa: E402
from test_hc_pre_sinkhorn import hc_pre_sinkhorn_golden  # noqa: E402

CASES_CSV = Path("/home/h00801112/codex/bestline4/operators/hc_pre_sinkhorn/round2/cases.csv")


def make_inputs(row):
    bs = int(row["bs"])
    m = int(row["hc_mult"])
    d = int(row["d"])
    iters = int(row["hc_sinkhorn_iters"])
    eps = float(row["hc_eps"])
    hc_mix = 2 * m + m * m
    vr = row.get("value_range", "[-1,1]")
    g = torch.Generator().manual_seed(2026 + int(row["case_id"]))

    if "nan" in vr:
        mixes = torch.full((bs, hc_mix), float("nan"))
        rsqrt = torch.full((bs,), float("nan"))
        hc_scale = torch.full((3,), float("nan"))
        hc_base = torch.full((hc_mix,), float("nan"))
        x = torch.full((bs, m, d), float("nan"), dtype=torch.bfloat16)
    elif "inf" in vr:
        mixes = torch.empty(bs, hc_mix).uniform_(-2, 2, generator=g)
        mixes.view(-1)[0] = float("inf")
        mixes.view(-1)[1] = float("-inf")
        rsqrt = torch.empty(bs).uniform_(0.5, 1.0, generator=g)
        hc_scale = torch.empty(3).uniform_(0.5, 1.0, generator=g)
        hc_base = torch.empty(hc_mix).uniform_(-0.1, 0.1, generator=g)
        x = torch.randn(bs, m, d, generator=g).to(torch.bfloat16)
    else:
        mixes = torch.empty(bs, hc_mix).uniform_(-1, 1, generator=g)
        rsqrt = torch.empty(bs).uniform_(0.5, 1.0, generator=g)
        hc_scale = torch.empty(3).uniform_(0.5, 1.0, generator=g)
        hc_base = torch.empty(hc_mix).uniform_(-0.1, 0.1, generator=g)
        x = torch.randn(bs, m, d, generator=g).to(torch.bfloat16)
    return mixes, rsqrt, hc_scale, hc_base, x, m, iters, eps


def max_err(a, b):
    if not torch.isfinite(a).any() and not torch.isfinite(b).any():
        return 0.0, 0.0
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().clamp_min(1e-8)
    return float(diff.max()), float((diff / denom).max())


def time_op(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def main():
    rows = list(csv.DictReader(CASES_CSV.open()))
    print(f"{'case':>4} {'bs':>6} {'M':>3} {'d':>5} {'iter':>5} {'wall_us':>9} {'y':>8} {'post':>8} {'comb':>8} {'ok':>5}")
    fails = 0
    recs = []
    for row in rows:
        mixes, rsqrt, hc_scale, hc_base, x, m, iters, eps = make_inputs(row)
        y_g, post_g, comb_g = hc_pre_sinkhorn_golden(
            mixes, rsqrt, hc_scale, hc_base, x, m, iters, eps)

        def launch():
            return custom_ops.hc_pre_sinkhorn_npu(
                mixes.npu(), rsqrt.npu(), hc_scale.npu(), hc_base.npu(), x.npu(),
                m, iters, eps)

        y_n, post_n, comb_n = launch()
        ye, yr = max_err(y_n.cpu(), y_g)
        pe, pr = max_err(post_n.cpu(), post_g)
        ce, cr = max_err(comb_n.cpu(), comb_g)
        y_ok = ye <= 1e-2 or yr <= 1e-2 or (not torch.isfinite(y_g).any())
        p_ok = pe <= 1e-4 or pr <= 1e-4 or (not torch.isfinite(post_g).any())
        c_ok = ce <= 1e-3 or cr <= 1e-3 or (not torch.isfinite(comb_g).any())
        ok = y_ok and p_ok and c_ok
        fails += int(not ok)
        us = time_op(launch)
        print(f"{row['case_id']:>4} {row['bs']:>6} {m:>3} {row['d']:>5} {iters:>5} "
              f"{us:>9.2f} {ye:>8.1e} {pe:>8.1e} {ce:>8.1e} {'PASS' if ok else 'FAIL':>5}")
        recs.append({"case_id": int(row["case_id"]), "ok": ok, "wall_us": us,
                     "y_abs": ye, "post_abs": pe, "comb_abs": ce})
        sys.stdout.flush()
    print(f"\n{len(rows) - fails}/{len(rows)} PASS")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
