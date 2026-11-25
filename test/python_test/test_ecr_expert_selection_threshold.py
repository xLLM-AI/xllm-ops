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
"""S2 ecr_expert_selection_threshold: NPU precision checks.

Ported from xllm/tests/routing_three_stage_test.py (shared case matrix and
CPU oracle) and ops/ecr_expert_selection_threshold/scripts/bench_golden.py
(the torch golden). Only the S2 stage is exercised here, so the threshold,
its token index and the strictly-greater count are compared directly instead
of feeding S3/S4.
"""

import math

import numpy as np
import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def case(case_id, s, e, dtype="fp32", **kwargs):
    spec = dict(case_id=case_id, s=s, e=e, dtype=dtype, top_p=0.125,
                eps=1e-9, seed=1234, dup_ratio=0.0, lengths=[],
                pattern="random")
    spec.update(kwargs)
    return spec


# One shared matrix: historical F1-F10 / four long shapes plus boundaries.
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
    Empty thresholds use the finite sentinel 2/-1/0.
    """
    e, s = prob.shape
    p = float(np.float32(spec["top_p"]))
    if not 0 < p <= 1:
        raise ValueError("top_p must remain positive after float32 rounding")
    lens = effective_lengths(spec)
    ks = [min(n, max(1, math.ceil(n * p - 1e-12))) if n else 0 for n in lens]
    tau_val = np.full((e, len(lens)), 2.0, np.float32)
    tau_idx = np.full((e, len(lens)), -1, np.int32)
    n_gt = np.zeros((e, len(lens)), np.int32)
    base = 0
    for b, (n, k) in enumerate(zip(lens, ks)):
        if k:
            row = prob[:, base:base + n]
            order = np.argsort(-row, axis=1, kind="stable")[:, :k]
            for expert in range(e):
                boundary = int(order[expert, -1])
                tau_val[expert, b] = row[expert, boundary]
                tau_idx[expert, b] = boundary
                n_gt[expert, b] = np.count_nonzero(
                    row[expert] > row[expert, boundary])
        base += n
    return dict(tau_val=tau_val, tau_idx=tau_idx, n_gt=n_gt)


def threshold_mask(prob, tau_val, tau_idx, lengths):
    """Selection implied by (tauVal, tauIdx): greater, plus ties up to tauIdx."""
    e, s = prob.shape
    mask = np.zeros((e, s), bool)
    base = 0
    for b, n in enumerate(lengths):
        if n:
            row = prob[:, base:base + n]
            local = np.arange(n)[None, :]
            mask[:, base:base + n] = (
                (row > tau_val[:, b, None]) |
                ((row == tau_val[:, b, None]) &
                 (local <= tau_idx[:, b, None])))
        base += n
    return mask


def golden_reference(prob, spec):
    """bench_golden.py S2: framework topk, comparisons and reductions."""
    lens = effective_lengths(spec)
    ratio = float(np.float32(spec["top_p"]))
    values, indices, counts = [], [], []
    base = 0
    for n in lens:
        if not n:
            values.append(torch.full((prob.shape[0],), 2.0))
            indices.append(torch.full((prob.shape[0],), -1, dtype=torch.int32))
            counts.append(torch.zeros((prob.shape[0],), dtype=torch.int32))
            continue
        k = min(n, max(1, math.ceil(n * ratio - 1e-12)))
        row = prob[:, base:base + n]
        threshold = torch.topk(row, k, dim=1, sorted=True).values[:, -1]
        greater = (row > threshold[:, None]).sum(dim=1, dtype=torch.int32)
        equal = row == threshold[:, None]
        rank = equal.to(torch.int32).cumsum(dim=1, dtype=torch.int32)
        boundary = equal & (rank == (k - greater)[:, None])
        local = torch.arange(n, dtype=torch.int32)
        index = torch.where(boundary, local[None, :], -1).max(dim=1).values
        values.append(threshold)
        indices.append(index)
        counts.append(greater)
        base += n
    return (torch.stack(values, dim=1), torch.stack(indices, dim=1),
            torch.stack(counts, dim=1))


def check_array(name, actual, expected):
    value = actual.detach().cpu().numpy() if hasattr(actual, "detach") else actual
    assert value.shape == expected.shape, (
        f"{name}: shape {value.shape} != {expected.shape}")
    assert value.dtype == expected.dtype, (
        f"{name}: dtype {value.dtype} != {expected.dtype}")
    if value.size == 0:
        return
    bad = value != expected
    assert not bad.any(), (
        f"{name}: {np.count_nonzero(bad)} mismatches; "
        f"first={tuple(np.argwhere(bad)[0])}")


@pytest.mark.parametrize("spec", CASES, ids=CASE_IDS)
def test_ecr_expert_selection_threshold_npu(spec):
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    prob_np = make_probability(spec)
    ref = cpu_reference(prob_np, spec)
    device = torch.device("npu:0")
    tau_val, tau_idx, n_gt = custom_ops.ecr_expert_selection_threshold_npu(
        torch.from_numpy(prob_np).to(device),
        spec["lengths"],
        spec["top_p"],
    )
    assert tau_val.dtype == torch.float32
    assert tau_idx.dtype == torch.int32 and n_gt.dtype == torch.int32
    assert tuple(tau_val.shape) == ref["tau_val"].shape
    check_array("tau_val", tau_val, ref["tau_val"])
    check_array("tau_idx", tau_idx, ref["tau_idx"])
    check_array("n_gt", n_gt, ref["n_gt"])
    check_array(
        "mask",
        threshold_mask(prob_np, tau_val.detach().cpu().numpy(),
                       tau_idx.detach().cpu().numpy(),
                       effective_lengths(spec)),
        threshold_mask(prob_np, ref["tau_val"], ref["tau_idx"],
                       effective_lengths(spec)))


@pytest.mark.parametrize("spec", CASES, ids=CASE_IDS)
def test_ecr_expert_selection_threshold_golden(spec):
    prob_np = make_probability(spec)
    ref = cpu_reference(prob_np, spec)
    values, indices, counts = golden_reference(torch.from_numpy(prob_np), spec)
    check_array("golden.tau_val", values, ref["tau_val"])
    check_array("golden.tau_idx", indices, ref["tau_idx"])
    check_array("golden.n_gt", counts, ref["n_gt"])


@pytest.mark.parametrize("lengths", ["list", "int32", "int64"])
def test_ecr_expert_selection_threshold_seq_lengths_forms(lengths):
    """List and both CPU integer tensor spellings produce the same output."""
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    spec = next(item for item in CASES if item["case_id"] == "F4")
    values = effective_lengths(spec)
    arg = values
    if lengths == "int32":
        arg = torch.tensor(values, dtype=torch.int32)
    elif lengths == "int64":
        arg = torch.tensor(values, dtype=torch.int64)

    prob_np = make_probability(spec)
    ref = cpu_reference(prob_np, spec)
    device = torch.device("npu:0")
    tau_val, tau_idx, n_gt = custom_ops.ecr_expert_selection_threshold_npu(
        torch.from_numpy(prob_np).to(device),
        arg,
        spec["top_p"],
    )
    check_array("tau_val", tau_val, ref["tau_val"])
    check_array("tau_idx", tau_idx, ref["tau_idx"])
    check_array("n_gt", n_gt, ref["n_gt"])


def test_ecr_expert_selection_threshold_rejects_bad_lengths():
    """Lengths that do not sum to S are rejected before the launch."""
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    prob = torch.zeros(2, 16, dtype=torch.float32, device="npu:0")
    with pytest.raises(RuntimeError):
        custom_ops.ecr_expert_selection_threshold_npu(prob, [8, 4], 0.5)
