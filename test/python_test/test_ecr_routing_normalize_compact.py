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
"""S4 ecr_routing_normalize_compact: NPU precision checks.

Ported from xllm/tests/routing_three_stage_test.py (shared case matrix and
CPU oracle) and ops/ecr_routing_normalize_compact/scripts/bench_golden.py
(the torch golden). Only the S4 stage is exercised here: tau_val/tau_idx and
sums/cnts come from the CPU oracle instead of the S2/S3 kernels, so this
file tests one operator independently.
"""

import math

import numpy as np
import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

ATOL = 1e-6
RTOL = 1e-5


def case(case_id, s, e, dtype="fp32", **kwargs):
    spec = dict(case_id=case_id, s=s, e=e, dtype=dtype, top_p=0.125,
                eps=1e-9, seed=1234, dup_ratio=0.0, lengths=[],
                pattern="random")
    spec.update(kwargs)
    return spec


# One shared matrix: historical F1-F10 / four long shapes plus boundaries.
# Every base case then runs with fp32 and bf16 source values; "dtype" is the
# precision the generated values are rounded to, the kernel itself stays fp32.
_BASE_CASES = [
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
for _item in _BASE_CASES:
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

# fp32 and bf16 for every case; the two fp16 cases keep their fp16 run too.
CASES = []
for _item in _BASE_CASES:
    for _dtype in ("fp32", "bf16", "fp16"):
        if _dtype == "fp16" and _item["dtype"] != "fp16":
            continue
        CASES.append(dict(_item,
                          case_id="{}-{}".format(_item["case_id"], _dtype),
                          dtype=_dtype))

CASE_IDS = [item["case_id"] for item in CASES]


def effective_lengths(spec):
    lens = list(spec["lengths"])
    if len(lens) <= 1:
        return [spec["s"]]
    if any(x < 0 for x in lens) or sum(lens) != spec["s"]:
        raise ValueError("invalid sample lengths")
    return lens


def make_probability(spec):
    """Seeded generator, contiguous float32 [E,S], values in [0,1]."""
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


def cpu_reference(prob, spec):
    """Stable-sort oracle independent of the extension and of the golden.

    tauIdx is sample-local and ties go to the smaller token. Nonempty samples
    always select at least one token, including for a tiny positive top_p.
    Empty thresholds use the finite sentinel 2/-1. Sums accumulate in float64.
    """
    e, s = prob.shape
    p = float(np.float32(spec["top_p"]))
    if not 0 < p <= 1:
        raise ValueError("top_p must remain positive after float32 rounding")
    lens = effective_lengths(spec)
    ks = [min(n, max(1, math.ceil(n * p - 1e-12))) if n else 0 for n in lens]
    tau_val = np.full((e, len(lens)), 2.0, np.float32)
    tau_idx = np.full((e, len(lens)), -1, np.int32)
    mask = np.zeros((e, s), bool)
    base = 0
    for b, (n, k) in enumerate(zip(lens, ks)):
        if k:
            row = prob[:, base:base + n]
            order = np.argsort(-row, axis=1, kind="stable")[:, :k]
            for expert in range(e):
                local = order[expert]
                mask[expert, base + local] = True
                tau_val[expert, b] = row[expert, local[-1]]
                tau_idx[expert, b] = local[-1]
        base += n
    sums = np.where(mask, prob, 0).sum(axis=0, dtype=np.float64).astype(np.float32)
    cnts = mask.sum(axis=0).astype(np.float32)
    ids = np.full((e, sum(ks)), -1, np.int32)
    weights = np.zeros(ids.shape, np.float32)
    for expert in range(e):
        tokens = np.flatnonzero(mask[expert])
        ids[expert, :len(tokens)] = tokens
        weights[expert, :len(tokens)] = (
            prob[expert, tokens] / np.maximum(sums[tokens], np.float32(spec["eps"]))
            * np.sqrt(cnts[tokens]))
    return dict(tau_val=tau_val, tau_idx=tau_idx, sums=sums, cnts=cnts,
                ids=ids, weights=weights)


def golden_reference(prob, tau_val, tau_idx, sums, cnts, spec):
    """bench_golden.py S4: mask, token sort, gather and normalize on CPU."""
    lens = effective_lengths(spec)
    ratio = float(np.float32(spec["top_p"]))
    eps = float(np.float32(spec["eps"]))
    ids, weights = [], []
    base = 0
    for b, n in enumerate(lens):
        k = min(n, max(1, math.ceil(n * ratio - 1e-12))) if n else 0
        row = prob[:, base:base + n]
        local = torch.arange(n, dtype=torch.int32)
        mask = ((row > tau_val[:, b, None]) |
                ((row == tau_val[:, b, None]) &
                 (local[None, :] <= tau_idx[:, b, None])))
        selected = torch.where(mask, local[None, :], n).sort(dim=1).values[:, :k]
        gather_idx = selected.to(torch.int64)
        chosen_prob = row.gather(1, gather_idx)
        global_idx = gather_idx + base
        chosen_sums = sums[global_idx].clamp_min(eps)
        chosen_counts = cnts[global_idx].sqrt()
        ids.append(selected + base)
        weights.append(chosen_prob / chosen_sums * chosen_counts)
        base += n
    return torch.cat(ids, dim=1), torch.cat(weights, dim=1)


def check_array(name, actual, expected, exact=False):
    value = actual.detach().cpu().numpy() if hasattr(actual, "detach") else actual
    assert value.shape == expected.shape, (
        f"{name}: shape {value.shape} != {expected.shape}")
    assert value.dtype == expected.dtype, (
        f"{name}: dtype {value.dtype} != {expected.dtype}")
    if value.size == 0:
        return
    assert np.isfinite(value).all(), f"{name}: non-finite values"
    if exact:
        bad = value != expected
    else:
        bad = np.abs(value - expected) > ATOL + RTOL * np.abs(expected)
    assert not bad.any(), (
        f"{name}: {np.count_nonzero(bad)} mismatches; "
        f"first={tuple(np.argwhere(bad)[0])}")


@pytest.mark.parametrize("spec", CASES, ids=CASE_IDS)
def test_ecr_routing_normalize_compact_npu(spec):
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    prob_np = make_probability(spec)
    ref = cpu_reference(prob_np, spec)
    device = torch.device("npu:0")
    ids, weights = custom_ops.ecr_routing_normalize_compact_npu(
        torch.from_numpy(prob_np).to(device),
        torch.from_numpy(ref["tau_val"]).to(device),
        torch.from_numpy(ref["tau_idx"]).to(device),
        torch.from_numpy(ref["sums"]).to(device),
        torch.from_numpy(ref["cnts"]).to(device),
        spec["lengths"],
        spec["top_p"],
        spec["eps"],
    )
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32
    assert tuple(ids.shape) == ref["ids"].shape
    check_array("ids", ids, ref["ids"], exact=True)
    check_array("weights", weights, ref["weights"])


@pytest.mark.parametrize("spec", CASES, ids=CASE_IDS)
def test_ecr_routing_normalize_compact_golden(spec):
    prob_np = make_probability(spec)
    ref = cpu_reference(prob_np, spec)
    prob = torch.from_numpy(prob_np)
    ids, weights = golden_reference(
        prob,
        torch.from_numpy(ref["tau_val"]),
        torch.from_numpy(ref["tau_idx"]),
        torch.from_numpy(ref["sums"]),
        torch.from_numpy(ref["cnts"]),
        spec,
    )
    check_array("golden.ids", ids, ref["ids"], exact=True)
    check_array("golden.weights", weights, ref["weights"])


@pytest.mark.parametrize("lengths", ["list", "int32", "int64"])
def test_ecr_routing_normalize_compact_seq_lengths_forms(lengths):
    """List and both CPU integer tensor spellings produce the same output."""
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    spec = next(item for item in CASES if item["case_id"] == "F4-fp32")
    values = effective_lengths(spec)
    arg = values
    if lengths == "int32":
        arg = torch.tensor(values, dtype=torch.int32)
    elif lengths == "int64":
        arg = torch.tensor(values, dtype=torch.int64)

    prob_np = make_probability(spec)
    ref = cpu_reference(prob_np, spec)
    device = torch.device("npu:0")
    ids, weights = custom_ops.ecr_routing_normalize_compact_npu(
        torch.from_numpy(prob_np).to(device),
        torch.from_numpy(ref["tau_val"]).to(device),
        torch.from_numpy(ref["tau_idx"]).to(device),
        torch.from_numpy(ref["sums"]).to(device),
        torch.from_numpy(ref["cnts"]).to(device),
        arg,
        spec["top_p"],
        spec["eps"],
    )
    check_array("ids", ids, ref["ids"], exact=True)
    check_array("weights", weights, ref["weights"])


def test_ecr_routing_normalize_compact_rejects_bad_lengths():
    """Lengths that do not sum to S are rejected before the launch."""
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    device = torch.device("npu:0")
    prob = torch.zeros(2, 16, dtype=torch.float32, device=device)
    tau_val = torch.zeros(2, 2, dtype=torch.float32, device=device)
    tau_idx = torch.zeros(2, 2, dtype=torch.int32, device=device)
    sums = torch.zeros(16, dtype=torch.float32, device=device)
    cnts = torch.zeros(16, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError):
        custom_ops.ecr_routing_normalize_compact_npu(
            prob, tau_val, tau_idx, sums, cnts, [8, 4], 0.5, 1e-9)
