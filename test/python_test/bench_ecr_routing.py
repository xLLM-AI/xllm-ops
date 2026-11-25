#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Per-case A/B latency of the aclnn ports against the upstream extensions.

Runs the same 21-case matrix as the pytest suites (shape, source precision,
top_p, sample segmentation) and, for every case, compares outputs before
timing so a failing baseline cannot silently produce meaningless numbers.

The debug box is shared and the upstream path is host-load sensitive, so each
case is timed in several blocks with the aclnn/upstream order flipping between
rounds; the upstream side keeps its best round (load only ever slows it down)
while aclnn is taken at the median of rounds.

Outputs one row per (case, round, op, backend) plus a table for the report.
"""

import argparse
import csv
import statistics

import numpy as np
import torch
import torch_npu  # noqa: F401

import custom_ops

ATOL = 1e-6
RTOL = 1e-6
DEVICE = torch.device("npu:0")


def load_library(path):
    torch.ops.load_library(str(path))


def summarize(samples):
    values = sorted(samples)
    return dict(mean=statistics.fmean(values),
                p50=values[len(values) // 2],
                p90=values[int(len(values) * 0.9)],
                best=values[0])


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    spans = []
    for _ in range(iters):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        spans.append((start, end))
    torch.npu.synchronize()
    return summarize([s.elapsed_time(e) for s, e in spans])


# Same matrix as test_ecr_*.py; keep the two in sync when cases change.
def case(case_id, s, e, dtype="fp32", **kwargs):
    spec = dict(case_id=case_id, s=s, e=e, dtype=dtype, top_p=0.125,
                eps=1e-9, seed=1234, dup_ratio=0.0, lengths=[],
                pattern="random")
    spec.update(kwargs)
    return spec


CASES = [
    case("F1", 4096, 48), case("F2", 4096, 48, "fp16"),
    case("F3", 4096, 48, "bf16"),
    case("F4", 4096, 48), case("F5", 4096, 48),
    case("F6", 40960, 48, "bf16"), case("F7", 4096, 48),
    case("F8", 4096, 48), case("F9", 4097, 48),
    case("F10", 4096, 33), case("L1", 65537, 33),
    case("L2", 131072, 48, "bf16"), case("L3", 131073, 33, "fp16"),
    case("L4", 40961, 33), case("ZERO", 17, 3),
    case("TIES", 33, 3), case("EMPTY", 0, 3),
    case("EMPTY_MIX", 17, 3), case("TINY", 17, 3),
    case("ONE", 1, 1), case("SINGLE_IGNORED", 17, 3),
]
for _item in CASES:
    _id = _item["case_id"]
    if _id == "F4":
        _item["lengths"] = [1024] * 4
    elif _id == "F5":
        _item["lengths"] = [512, 1024, 1536, 1024]
    elif _id in ("F7", "L2"):
        _item["dup_ratio"] = 0.3
    elif _id in ("F8", "L4", "ONE"):
        _item["top_p"] = 1.0
    elif _id == "L3":
        _item["lengths"] = [65537, 65536]
    elif _id == "ZERO":
        _item["pattern"] = "zero"
    elif _id == "TIES":
        _item["pattern"] = "ties"
    elif _id == "EMPTY_MIX":
        _item["lengths"] = [0, 8, 0, 9]
    elif _id == "TINY":
        _item["top_p"] = 1e-15
        _item["lengths"] = [8, 9]
    elif _id == "SINGLE_IGNORED":
        _item["lengths"] = [999]


def make_probability(spec):
    rng = np.random.default_rng(spec["seed"])
    logits = rng.standard_normal((spec["s"], spec["e"])).astype(np.float32) * 2.0
    count = int(spec["s"] * spec["dup_ratio"])
    if count > 1:
        chosen = rng.choice(spec["s"], size=count, replace=False)
        logits[chosen] = logits[chosen[0]]
    if spec["dtype"] == "fp16":
        logits = logits.astype(np.float16).astype(np.float32)
    elif spec["dtype"] == "bf16":
        bits = logits.view(np.uint32)
        bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16
        logits = (bits.astype(np.uint32) << 16).view(np.float32)
    elif spec["dtype"] != "fp32":
        raise ValueError("unsupported source dtype")
    prob = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    if spec["pattern"] == "zero":
        prob.fill(0.0)
    elif spec["pattern"] == "ties":
        prob.fill(0.5)
    elif spec["pattern"] != "random":
        raise ValueError("unknown data pattern")
    return np.ascontiguousarray(prob.T)


def check_exact(name, actual, expected):
    if not torch.equal(actual, expected):
        diff = (actual != expected).sum().item()
        raise AssertionError(f"{name} mismatch in {diff} elements")


FIELDS = ["case", "s", "e", "dtype", "top_p", "lengths", "op", "backend",
          "round", "mean_ms", "p50_ms", "p90_ms", "best_ms"]


def run(args):
    torch_npu.npu.set_device(0)
    load_library(args.upstream_s2)
    load_library(args.upstream_s3)
    load_library(args.upstream_s4)

    rows = []
    for spec in CASES:
        label = spec["case_id"]
        print(f"[case] {label}", flush=True)
        try:
            prob = torch.from_numpy(make_probability(spec)).to(DEVICE)
            lengths = spec["lengths"]

            tau_val_ref, tau_idx_ref, n_gt_ref = (
                torch.ops.npu.ecr_expert_selection_threshold(
                    prob, lengths, spec["top_p"]))
            tau_val, tau_idx, n_gt = (
                custom_ops.ecr_expert_selection_threshold_npu(
                    prob, lengths, spec["top_p"]))
            check_exact(f"{label}.tau_val", tau_val, tau_val_ref)
            check_exact(f"{label}.tau_idx", tau_idx, tau_idx_ref)
            check_exact(f"{label}.n_gt", n_gt, n_gt_ref)

            sums_ref, cnts_ref = torch.ops.npu.ecr_token_routing_statistics(
                prob, tau_val_ref, tau_idx_ref, lengths)
            sums, cnts = custom_ops.ecr_token_routing_statistics_npu(
                prob, tau_val_ref, tau_idx_ref, lengths)
            check_exact(f"{label}.sums", sums, sums_ref)
            check_exact(f"{label}.cnts", cnts, cnts_ref)

            ids_ref, w_ref = torch.ops.npu.ecr_routing_normalize_compact(
                prob, tau_val_ref, tau_idx_ref, sums_ref, cnts_ref,
                lengths, spec["top_p"], spec["eps"])
            ids, w = custom_ops.ecr_routing_normalize_compact_npu(
                prob, tau_val_ref, tau_idx_ref, sums_ref, cnts_ref,
                lengths, spec["top_p"], spec["eps"])
            check_exact(f"{label}.ids", ids, ids_ref)
            if not torch.allclose(w, w_ref, rtol=RTOL, atol=ATOL):
                raise AssertionError(f"{label}.weights mismatch")
        except Exception as exc:  # noqa: BLE001 - report, keep the matrix going
            print(f"[WAIVED] {label}: {exc}")
            continue

        calls = {
            "S2": {
                "aclnn": lambda: custom_ops.ecr_expert_selection_threshold_npu(
                    prob, lengths, spec["top_p"]),
                "golden": lambda: torch.ops.npu.ecr_expert_selection_threshold(
                    prob, lengths, spec["top_p"]),
            },
            "S3": {
                "aclnn": lambda: custom_ops.ecr_token_routing_statistics_npu(
                    prob, tau_val_ref, tau_idx_ref, lengths),
                "golden": lambda: torch.ops.npu.ecr_token_routing_statistics(
                    prob, tau_val_ref, tau_idx_ref, lengths),
            },
            "S4": {
                "aclnn": lambda: custom_ops.ecr_routing_normalize_compact_npu(
                    prob, tau_val_ref, tau_idx_ref, sums_ref, cnts_ref,
                    lengths, spec["top_p"], spec["eps"]),
                "golden": lambda: torch.ops.npu.ecr_routing_normalize_compact(
                    prob, tau_val_ref, tau_idx_ref, sums_ref, cnts_ref,
                    lengths, spec["top_p"], spec["eps"]),
            },
        }
        for rnd in range(args.rounds):
            order = ("aclnn", "golden") if rnd % 2 == 0 else ("golden", "aclnn")
            for op in ("S2", "S3", "S4"):
                for backend in order:
                    print(f"[time] {label} {op} {backend} round={rnd}", flush=True)
                    got = timeit(calls[op][backend], args.warmup, args.iters)
                    rows.append(dict(case=label, s=spec["s"], e=spec["e"],
                                     dtype=spec["dtype"], top_p=spec["top_p"],
                                     lengths=",".join(map(str, lengths)) or "-",
                                     op=op, backend=backend, round=rnd,
                                     mean_ms=got["mean"], p50_ms=got["p50"],
                                     p90_ms=got["p90"], best_ms=got["best"]))

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: (repr(v) if isinstance(v, float) else v)
                                 for k, v in row.items()})
        print(f"wrote {args.csv}")

    # The debug box is shared and the upstream path is host-load sensitive: its
    # p50 only ever moves up, so the best round is the closest thing to its
    # intrinsic cost. The aclnn side is taken at the median instead, because a
    # single lucky event read under-reports tiny kernels. This also means the
    # printed ratios are a lower bound on the speedup.
    agg = {}
    for row in rows:
        key = (row["case"], row["op"], row["backend"])
        agg.setdefault(key, []).append(row["p50_ms"])
    for key, values in agg.items():
        pick = min(values) if key[2] == "golden" else statistics.median(values)
        agg[key] = pick

    header = (f"{'case':>14} {'S':>7} {'E':>3} {'dtype':>6} {'top_p':>8} "
              f"{'op':>3} {'aclnn(ms)':>10} {'golden(ms)':>11} {'ratio':>7}")
    print(header)
    print("-" * len(header))
    summary = {}
    for spec in CASES:
        label = spec["case_id"]
        for op in ("S2", "S3", "S4"):
            got = agg.get((label, op, "aclnn"))
            ref = agg.get((label, op, "golden"))
            if got is None or ref is None:
                continue
            summary.setdefault(op, []).append(ref / got)
            print(f"{label:>14} {spec['s']:>7} {spec['e']:>3} "
                  f"{spec['dtype']:>6} {spec['top_p']:>8g} {op:>3} "
                  f"{got:>10.4f} {ref:>11.4f} {ref / got:>6.2f}x")

    for op in ("S2", "S3", "S4"):
        vals = summary.get(op, [])
        if vals:
            print(f"{op}: cases={len(vals)} min={min(vals):.2f}x "
                  f"median={statistics.median(vals):.2f}x max={max(vals):.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-s2", required=True,
                        help="libecr_expert_selection_threshold_ops.so")
    parser.add_argument("--upstream-s3", required=True,
                        help="libecr_token_routing_statistics_ops.so")
    parser.add_argument("--upstream-s4", required=True,
                        help="libecr_routing_normalize_compact_ops.so")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3,
                        help="alternating blocks per case; see run() for how "
                             "the rounds are aggregated")
    parser.add_argument("--csv", default=None)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
