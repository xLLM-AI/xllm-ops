#!/usr/bin/env python3
"""On-device kernel time for the 20 cann-bench cases via torch_npu profiler."""
from __future__ import annotations

import csv
import glob
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch_npu

ROOT = Path("/home/h00801112/codex/my-xllm-ops/xllm-ops")
sys.path.insert(0, str(ROOT / "test/python_test"))
import custom_ops  # noqa: E402

CASES_CSV = Path("/home/h00801112/codex/bestline4/operators/hc_pre_sinkhorn/round2/cases.csv")
OUT = ROOT / "xllm_ops/moe/hc_pre_sinkhorn/docs/perf-integrate/my_xllm_20cases_perf.json"
WORK = Path("/tmp/hc_pre_sinkhorn_prof_my_xllm")


def parse_kernel_csv(work: Path) -> dict:
    stats = defaultdict(lambda: [0, 0.0])
    candidates = glob.glob(str(work / "**" / "kernel_details.csv"), recursive=True)
    if not candidates:
        candidates = glob.glob(str(work / "**" / "op_statistic*.csv"), recursive=True)
    for path in candidates:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("Name") or row.get("OP Type") or "").strip()
                dur = row.get("Duration(us)") or row.get("Total Time(us)") or "0"
                if not name:
                    continue
                try:
                    val = float(dur)
                except ValueError:
                    continue
                stats[name][0] += 1
                stats[name][1] += val
    return {k: {"count": v[0], "total_us": round(v[1], 3),
                "avg_us": round(v[1] / max(v[0], 1), 3)}
            for k, v in stats.items()}


def pick(stats: dict, needle: str):
    for name, val in stats.items():
        if needle.lower() in name.lower():
            return {"kernel": name, **val}
    return None


def profile_once(fn, tag: str, reps: int = 20) -> dict:
    work = WORK / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    exp_cfg = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    )
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=exp_cfg,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(work)),
    ) as prof:
        for _ in range(reps):
            fn()
            prof.step()
        torch.npu.synchronize()
    return parse_kernel_csv(work)


def main():
    torch_npu.npu.set_device(0)
    rows = list(csv.DictReader(CASES_CSV.open()))
    results = []
    print(f"{'case':>4} {'bs':>6} {'M':>3} {'iter':>5} {'kernel_us':>10} {'kernel':>24}")
    for row in rows:
        cid = int(row["case_id"])
        bs, m, d = int(row["bs"]), int(row["hc_mult"]), int(row["d"])
        iters, eps = int(row["hc_sinkhorn_iters"]), float(row["hc_eps"])
        hc_mix = 2 * m + m * m
        g = torch.Generator().manual_seed(2026 + cid)
        mixes = torch.empty(bs, hc_mix).uniform_(-1, 1, generator=g).npu()
        rsqrt = torch.empty(bs).uniform_(0.5, 1.0, generator=g).npu()
        hc_scale = torch.empty(3).uniform_(0.5, 1.0, generator=g).npu()
        hc_base = torch.empty(hc_mix).uniform_(-0.1, 0.1, generator=g).npu()
        x = torch.randn(bs, m, d, generator=g).to(torch.bfloat16).npu()

        def fn():
            return custom_ops.hc_pre_sinkhorn_npu(
                mixes, rsqrt, hc_scale, hc_base, x, m, iters, eps)

        stats = profile_once(fn, f"c{cid}", reps=20)
        hit = pick(stats, "HcPreSinkhorn") or pick(stats, "hc_pre_sinkhorn")
        us = hit["avg_us"] if hit else float("nan")
        name = hit["kernel"] if hit else "n/a"
        results.append({"case_id": cid, "bs": bs, "M": m, "d": d, "iters": iters,
                        "kernel_us": us, "kernel": name})
        print(f"{cid:>4} {bs:>6} {m:>3} {iters:>5} {us:>10} {name:>24}")
        sys.stdout.flush()

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwritten {OUT}")


if __name__ == "__main__":
    main()
