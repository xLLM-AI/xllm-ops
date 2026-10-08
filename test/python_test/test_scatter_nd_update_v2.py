# Copyright 2025 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ScatterNdUpdateV2 unit test.

Op semantics (strictly matching the kernel implementation):
  - indexDim   = indices.shape[-1]
  - scatterLength = prod(var.shape[indexDim:])            (product of remaining dims)
  - for each index row:
        gmIdx = sum(indices[row][i] * strides[i] for i in range(indexDim))
        start = gmIdx * scatterLength                     (physical element offset)
        var.flat[start : start+scatterLength] = updates[row].flatten()
  - for duplicate index mapping to the same gmIdx, keep the "last row in original order" of updates
    (kernel de-dupes via ascending sort + descending traversal + lastProcessedIdx;
     golden naturally realizes "later write overrides earlier" via sequential traversal)

  strides semantics: offset multiple in units of scatterLength blocks. For row-major var, the standard value is
        strides[i] = prod(var.shape[i+1 : indexDim])
"""
import json
import os
import time

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


def scatter_nd_update_v2_golden(var, indices, updates,strides):
    """CPU-side reference implementation. var is updated in place and returned.

    var:     tensor of arbitrary shape
    indices: [num_indices, indexDim] (int32/int64)
    updates: [num_indices, *var.shape[indexDim:]]
    strides: list[int], length = indexDim
    """
    index_dim = indices.shape[-1]
    scatter_length = 1
    for d in range(index_dim, var.dim()):
        scatter_length *= var.shape[d]

    var_flat = var.reshape(-1).clone()
    idx = indices.to(torch.int64)
    upd_flat = updates.reshape(updates.shape[0], -1)
    num_indices = idx.shape[0]

    # sequential traversal: later write overrides earlier => duplicate index keeps last row
    for row in range(num_indices):
        gm_idx = 0
        for i in range(index_dim):
            gm_idx += int(idx[row, i].item()) * strides[i]
        # strides are element-level offsets (include scatterLength factor); gm_idx is the physical element offset
        start = gm_idx
        var_flat[start:start + scatter_length] = upd_flat[row]

    return var_flat.reshape(var.shape)


def make_row_major_strides(var_shape, index_dim):
    """Row-major element-level strides (matching the kernel physical offset).

    kernel: outOffset = sum(idx[i]*strides[i]); no extra *scatterLength,
    so strides must be element-level offsets including the scatterLength factor:
        strides[i] = prod(var_shape[i+1 : ])   (traverse to the end of var)
    """
    strides = []
    for i in range(index_dim):
        s = 1
        for j in range(i + 1, len(var_shape)):
            s *= var_shape[j]
        strides.append(s)
    return strides


def unique_indices(num_indices, dim_ranges, generator):
    """Generate num_indices unique index rows (each column ranges over dim_ranges[i])."""
    seen = set()
    rows = []
    max_total = 1
    for r in dim_ranges:
        max_total *= r
    num_indices = min(num_indices, max_total)
    while len(rows) < num_indices:
        row = tuple(int(torch.randint(0, dim_ranges[i], (1,), generator=generator).item())
                    for i in range(len(dim_ranges)))
        if row not in seen:
            seen.add(row)
            rows.append(list(row))
    return rows


# CASES: (var_shape, index_dim, num_indices, dup_pairs, var_dtype, idx_dtype)
#   dup_pairs: number of appended duplicate rows (tests "keep last duplicate"); appended rows reuse existing index
CASES = [
    ((8, 16),           1, 4,  0, torch.float16, torch.int32),
    ((16, 32),          1, 6,  2, torch.float32, torch.int32),
    ((8, 8, 16),        2, 5,  0, torch.float16, torch.int64),
    ((10, 6, 32),       2, 8,  3, torch.float32, torch.int32),
    ((4, 4, 4, 8),      3, 6,  0, torch.float16, torch.int64),
    ((6, 5, 3, 16),     3, 10, 4, torch.float32, torch.int32),
    ((32, 64),          1, 12, 5, torch.float16, torch.int32),
    ((12, 12, 8),       2, 9,  2, torch.float32, torch.int64),
    ((5, 7),            1, 3,  1, torch.float16, torch.int32),
    ((3, 4, 5, 6, 4),   4, 7,  0, torch.float32, torch.int32),
    # A5 moe perf shapes: indices (8,1) int64, indexDim=1, small-fast-path candidates
    ((20736, 512),      1, 8,  0, torch.bfloat16, torch.int64),
    ((1241088, 1),      1, 8,  0, torch.float16,  torch.int64),
    ((1241088, 128),    1, 8,  0, torch.int8,     torch.int64),
]


def _build_inputs(var_shape, index_dim, num_indices, dup_pairs, var_dtype, idx_dtype, gen):
    strides = make_row_major_strides(var_shape, index_dim)
    dim_ranges = [var_shape[i] for i in range(index_dim)]

    base_rows = unique_indices(num_indices, dim_ranges, gen)
    # append duplicate rows: reuse existing index but with different updates => verify keeping the last write
    dup_pairs = min(dup_pairs, len(base_rows))
    rows = list(base_rows)
    for k in range(dup_pairs):
        rows.append(list(base_rows[k]))

    total = len(rows)
    indices = torch.tensor(rows, dtype=idx_dtype)

    scatter_shape = tuple(var_shape[index_dim:])
    if var_dtype == torch.int8:
        updates = torch.randint(-128, 128, (total, *scatter_shape),
                                generator=gen).to(torch.int8)
        var = torch.randint(-128, 128, tuple(var_shape), generator=gen).to(torch.int8)
    else:
        updates = torch.randn((total, *scatter_shape), generator=gen).to(var_dtype)
        var = torch.randn(var_shape, generator=gen).to(var_dtype)
    return var, indices, updates, strides


@pytest.mark.parametrize(
    "var_shape,index_dim,num_indices,dup_pairs,var_dtype,idx_dtype", CASES)
def test_scatter_nd_update_v2(var_shape, index_dim, num_indices, dup_pairs,
                              var_dtype, idx_dtype):
    gen = torch.Generator()
    gen.manual_seed(2026)

    var, indices, updates, strides = _build_inputs(
        var_shape, index_dim, num_indices, dup_pairs, var_dtype, idx_dtype, gen)

    out_ref = scatter_nd_update_v2_golden(var, indices, updates, strides)

    var_npu = var.clone().npu()
    custom_ops.scatter_nd_update_v2_npu(
        var_npu, indices.npu(), updates.npu(), strides)

    assert torch.equal(var_npu.cpu(), out_ref), (
        f"mismatch: shape={var_shape} index_dim={index_dim} "
        f"dtype={var_dtype}/{idx_dtype}")


# ---------------------------------------------------------------------------
# Performance: all functional CASES above (10 original + 3 A5 moe shapes)
# Two metrics per case:
#   e2e_us    : host loop timing over PERF_ITERS submits + final sync
#   device_us : NPU kernel time from torch_npu profiler (device side)
# Result JSON path: env SCATTER_PERF_OUT (default ./scatter_perf_result.json)
# ---------------------------------------------------------------------------
PERF_WARMUP = 20
PERF_ITERS = 1000


def _make_npu_inputs(var_shape, index_dim, num_indices, dup_pairs, var_dtype, idx_dtype):
    gen = torch.Generator()
    gen.manual_seed(2026)
    var, indices, updates, strides = _build_inputs(
        var_shape, index_dim, num_indices, dup_pairs, var_dtype, idx_dtype, gen)
    return (var.clone().npu(), indices.npu(), updates.npu(), strides)


def _perf_e2e_us(npu_inputs):
    var_npu, indices_npu, updates_npu, strides = npu_inputs
    for _ in range(PERF_WARMUP):
        custom_ops.scatter_nd_update_v2_npu(var_npu, indices_npu, updates_npu, strides)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(PERF_ITERS):
        custom_ops.scatter_nd_update_v2_npu(var_npu, indices_npu, updates_npu, strides)
    torch.npu.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / PERF_ITERS * 1e6


def _perf_device_us(npu_inputs):
    import tempfile
    from torch_npu.profiler import profile, ProfilerActivity

    var_npu, indices_npu, updates_npu, strides = npu_inputs
    for _ in range(PERF_WARMUP):
        custom_ops.scatter_nd_update_v2_npu(var_npu, indices_npu, updates_npu, strides)
    torch.npu.synchronize()
    trace_path = tempfile.mktemp(prefix="scatter_perf_", suffix=".json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        for _ in range(PERF_ITERS):
            custom_ops.scatter_nd_update_v2_npu(var_npu, indices_npu, updates_npu, strides)
        torch.npu.synchronize()
    prof.export_chrome_trace(trace_path)
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", []) if isinstance(trace, dict) else trace
    kernel_us = 0.0
    for evt in events:
        if not isinstance(evt, dict):
            continue
        name = str(evt.get("name", ""))
        if "AiCore" in name and "ScatterNdUpdate" in name:
            kernel_us += float(evt.get("dur", 0.0))
    try:
        os.remove(trace_path)
    except OSError:
        pass
    return kernel_us / PERF_ITERS if kernel_us > 0 else None


def test_scatter_nd_update_v2_perf():
    tag = os.environ.get("SCATTER_PERF_TAG", "untagged")
    results = []
    for var_shape, index_dim, num_indices, dup_pairs, var_dtype, idx_dtype in CASES:
        npu_inputs = _make_npu_inputs(
            var_shape, index_dim, num_indices, dup_pairs, var_dtype, idx_dtype)
        e2e_us = _perf_e2e_us(npu_inputs)
        try:
            device_us = _perf_device_us(npu_inputs)
        except Exception as exc:  # profiler unavailable in this env
            print(f"[perf][warn] device profiling failed: {exc}")
            device_us = None
        shape_str = "x".join(map(str, var_shape))
        dtype_str = f"{str(var_dtype).replace('torch.', '')}/{str(idx_dtype).replace('torch.', '')}"
        dev_str = f"{device_us:.3f}" if device_us is not None else "N/A"
        print(f"[perf][{tag}] {shape_str} {dtype_str} e2e={e2e_us:.3f}us device={dev_str}us")
        results.append({
            "tag": tag,
            "var_shape": list(var_shape),
            "index_dim": index_dim,
            "num_indices": num_indices,
            "var_dtype": str(var_dtype),
            "idx_dtype": str(idx_dtype),
            "warmup": PERF_WARMUP,
            "iters": PERF_ITERS,
            "e2e_us": round(e2e_us, 4),
            "device_us": round(device_us, 4) if device_us is not None else None,
        })

    out_path = os.environ.get("SCATTER_PERF_OUT", "scatter_perf_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[perf][{tag}] result written to {out_path}")
    try:
        import shutil
        shutil.rmtree("export_only_prof_dir", ignore_errors=True)
    except Exception:
        pass