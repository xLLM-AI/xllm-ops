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
"""S3 ecr_token_routing_statistics: NPU precision checks.

Ported from xllm/tests/routing_three_stage_test.py (shared case matrix and CPU
oracles) and ops/ecr_token_routing_statistics/scripts/bench_golden.py (the
torch golden). The S2 thresholds are produced by the local stable-sort oracle,
so this file only exercises the statistics op.
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
                seed=1234, dup_ratio=0.0, lengths=[], pattern="random")
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


def s2_reference(prob, spec):
    """Stable-sort S2 oracle: sample-local k-th value and its token index."""
    e, s = prob.shape
    p = float(np.float32(spec["top_p"]))
    if not 0 < p <= 1:
        raise ValueError("top_p must remain positive after float32 rounding")
    lens = effective_lengths(spec)
    ks = [min(n, max(1, math.ceil(n * p - 1e-12))) if n else 0 for n in lens]
    tau_val = np.full((e, len(lens)), 2.0, np.float32)
    tau_idx = np.full((e, len(lens)), -1, np.int32)
    base = 0
    for b, (n, k) in enumerate(zip(lens, ks)):
        if k:
            row = prob[:, base:base + n]
            order = np.argsort(-row, axis=1, kind="stable")[:, :k]
            for expert in range(e):
                boundary = int(order[expert, -1])
                tau_val[expert, b] = row[expert, boundary]
                tau_idx[expert, b] = boundary
        base += n
    return dict(tau_val=tau_val, tau_idx=tau_idx)


def s3_reference(prob, tau_val, tau_idx, lengths):
    """Per-token selection statistics; sums reduce in float64 then cast."""
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
    sums = np.where(mask, prob, 0).sum(axis=0, dtype=np.float64).astype(np.float32)
    cnts = mask.sum(axis=0).astype(np.float32)
    return sums, cnts


def golden_reference(prob, spec):
    """bench_golden.py S2+S3: framework topk/comparisons and reductions."""
    lens = effective_lengths(spec)
    ratio = float(np.float32(spec["top_p"]))
    values, indices = [], []
    base = 0
    for n in lens:
        if not n:
            values.append(torch.full((prob.shape[0],), 2.0))
            indices.append(torch.full((prob.shape[0],), -1, dtype=torch.int32))
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
        base += n
    tau_val = torch.stack(values, dim=1)
    tau_idx = torch.stack(indices, dim=1)

    sums, counts = [], []
    base = 0
    for b, n in enumerate(lens):
        row = prob[:, base:base + n]
        local = torch.arange(n, dtype=torch.int32)
        mask = ((row > tau_val[:, b, None]) |
                ((row == tau_val[:, b, None]) &
                 (local[None, :] <= tau_idx[:, b, None])))
        sums.append(torch.where(mask, row, 0.0).sum(dim=0))
        counts.append(mask.to(torch.float32).sum(dim=0))
        base += n
    return torch.cat(sums), torch.cat(counts)


def check_array(name, actual, expected, exact=False):
    value = actual.detach().cpu().numpy() if hasattr(actual, "detach") else actual
    assert value.shape == expected.shape, (
        f"{name}: shape {value.shape} != {expected.shape}")
    assert value.dtype == expected.dtype, (
        f"{name}: dtype {value.dtype} != {expected.dtype}")
    if value.size == 0:
        return
    if exact:
        bad = value != expected
    else:
        bad = np.abs(value - expected) > ATOL + RTOL * np.abs(expected)
    assert not bad.any(), (
        f"{name}: {np.count_nonzero(bad)} mismatches; "
        f"first={tuple(np.argwhere(bad)[0])} "
        f"actual={value[tuple(np.argwhere(bad)[0])]} "
        f"expected={expected[tuple(np.argwhere(bad)[0])]}")


@pytest.mark.parametrize("spec", CASES, ids=CASE_IDS)
def test_ecr_token_routing_statistics_npu(spec):
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    prob_np = make_probability(spec)
    lens = effective_lengths(spec)
    ref = s2_reference(prob_np, spec)
    sums_ref, cnts_ref = s3_reference(prob_np, ref["tau_val"], ref["tau_idx"],
                                      lens)
    device = torch.device("npu:0")
    sums, cnts = custom_ops.ecr_token_routing_statistics_npu(
        torch.from_numpy(prob_np).to(device),
        torch.from_numpy(ref["tau_val"]).to(device),
        torch.from_numpy(ref["tau_idx"]).to(device),
        spec["lengths"],
    )
    assert sums.dtype == torch.float32 and cnts.dtype == torch.float32
    assert tuple(sums.shape) == (prob_np.shape[1],)
    check_array("sums", sums, sums_ref)
    check_array("cnts", cnts, cnts_ref, exact=True)


@pytest.mark.parametrize("spec", CASES, ids=CASE_IDS)
def test_ecr_token_routing_statistics_golden(spec):
    prob_np = make_probability(spec)
    lens = effective_lengths(spec)
    ref = s2_reference(prob_np, spec)
    sums_ref, cnts_ref = s3_reference(prob_np, ref["tau_val"], ref["tau_idx"],
                                      lens)
    sums, cnts = golden_reference(torch.from_numpy(prob_np), spec)
    check_array("golden.sums", sums, sums_ref)
    check_array("golden.cnts", cnts, cnts_ref, exact=True)


@pytest.mark.parametrize("lengths", ["list", "int32", "int64"])
def test_ecr_token_routing_statistics_seq_lengths_forms(lengths):
    """List and both CPU integer tensor spellings produce the same output."""
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    spec = next(item for item in CASES if item["case_id"] == "F5-fp32")
    values = effective_lengths(spec)
    arg = values
    if lengths == "int32":
        arg = torch.tensor(values, dtype=torch.int32)
    elif lengths == "int64":
        arg = torch.tensor(values, dtype=torch.int64)

    prob_np = make_probability(spec)
    ref = s2_reference(prob_np, spec)
    sums_ref, cnts_ref = s3_reference(prob_np, ref["tau_val"], ref["tau_idx"],
                                      values)
    device = torch.device("npu:0")
    sums, cnts = custom_ops.ecr_token_routing_statistics_npu(
        torch.from_numpy(prob_np).to(device),
        torch.from_numpy(ref["tau_val"]).to(device),
        torch.from_numpy(ref["tau_idx"]).to(device),
        arg,
    )
    check_array("sums", sums, sums_ref)
    check_array("cnts", cnts, cnts_ref, exact=True)


def test_ecr_token_routing_statistics_rejects_bad_lengths():
    """Lengths that do not sum to S are rejected before the launch."""
    try:
        torch_npu.npu.set_device(0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NPU device not available: {exc}")

    prob = torch.zeros(2, 16, dtype=torch.float32, device="npu:0")
    tau_val = torch.zeros(2, 2, dtype=torch.float32, device="npu:0")
    tau_idx = torch.zeros(2, 2, dtype=torch.int32, device="npu:0")
    with pytest.raises(RuntimeError):
        custom_ops.ecr_token_routing_statistics_npu(prob, tau_val, tau_idx,
                                                   [8, 4])
