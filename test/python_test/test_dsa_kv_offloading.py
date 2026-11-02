# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0
"""Public vendor ACLNN contracts. No model, mocked kernel or hand-built AOT.

DSA_KV_OFFLOADING_VENDOR_LIBRARY must name the installed libcust_opapi.so. The caller must
exclusively assign one A3 device. Host aliases own real ACL pinned mappings.
"""

from __future__ import annotations

import ctypes as C
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

PAGE, CAPACITY, LATENT, ROPE = 128, 2176, 512, 64


def _expected_plan_tokens(
    topk: "torch.Tensor", descriptor: "torch.Tensor"
) -> "torch.Tensor":
    """An inactive query publishes an empty plan without touching resident KV."""
    expected = topk.clone()
    expected[descriptor[:, 2] == 0] = -1
    return expected


def test_dummy_rope_plan_token_oracle_cpu() -> None:
    import torch as cpu_torch

    topk = cpu_torch.tensor([[[3, 4, -1]], [[7, 8, -1]]], dtype=cpu_torch.int32)
    descriptor = cpu_torch.tensor([[0, 0, 0, 7], [0, 1, 9, 5]], dtype=cpu_torch.int32)
    expected = _expected_plan_tokens(topk, descriptor)
    assert expected[0].tolist() == [[-1, -1, -1]]
    assert cpu_torch.equal(expected[1], topk[1])
    assert topk[0].tolist() == [[3, 4, -1]]


@pytest.fixture(scope="module")
def runtime():
    global torch
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    library = os.environ.get("DSA_KV_OFFLOADING_VENDOR_LIBRARY")
    if not library:
        pytest.skip("DSA_KV_OFFLOADING_VENDOR_LIBRARY must identify the installed vendor library")
    torch.npu.set_device(0)
    return Runtime(Path(library).resolve(strict=True))


@dataclass
class HostAlias:
    tensor: object
    pointer: int


class Runtime:
    def __init__(self, library):
        # Match the framework adapter's RTLD_LOCAL loading. Promoting CANN
        # aggregate libraries globally can interpose their private C++ state.
        self.acl = C.CDLL("libascendcl.so")
        self.api = C.CDLL(str(library))
        self.meta = C.CDLL("libnnopbase.so")
        self.meta.aclCreateTensor.restype = C.c_void_p
        self.meta.aclCreateTensor.argtypes = [
            C.POINTER(C.c_int64),
            C.c_uint64,
            C.c_int,
            C.POINTER(C.c_int64),
            C.c_int64,
            C.c_int,
            C.POINTER(C.c_int64),
            C.c_uint64,
            C.c_void_p,
        ]
        self.meta.aclDestroyTensor.argtypes = [C.c_void_p]
        self.acl.aclrtMallocHost.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        self.acl.aclrtHostRegister.argtypes = [
            C.c_void_p,
            C.c_uint64,
            C.c_int,
            C.POINTER(C.c_void_p),
        ]
        self.acl.aclrtHostUnregister.argtypes = [C.c_void_p]
        self.acl.aclrtFreeHost.argtypes = [C.c_void_p]
        self.keepalive = []
        self.api.aclnnDsaKvOffloadingSelectionCapacity.restype = C.c_int64
        assert self.api.aclnnDsaKvOffloadingSelectionCapacity() == CAPACITY
        for name, count in (
            ("DsaKvOffloadingUpdateNope", 15),
            ("DsaKvOffloadingUpdateRopeWithPlan", 19),
            ("DsaKvOffloadingServeRopePlanned", 12),
        ):
            get = getattr(self.api, "aclnn" + name + "GetWorkspaceSize")
            get.argtypes = [C.c_void_p] * count + [
                C.POINTER(C.c_uint64),
                C.POINTER(C.c_void_p),
            ]
            run = getattr(self.api, "aclnn" + name)
            run.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]

    @contextmanager
    def host(self, pages, dim):
        size = pages * PAGE * dim * 2
        alignment = os.sysconf("SC_PAGESIZE")
        raw, mapped = C.c_void_p(), C.c_void_p()
        assert self.acl.aclrtMallocHost(C.byref(raw), size + alignment - 1) == 0
        aligned = (raw.value + alignment - 1) // alignment * alignment
        registered = False
        try:
            assert self.acl.aclrtHostRegister(aligned, size, 0, C.byref(mapped)) == 0
            registered = True
            assert mapped.value
            storage = (C.c_uint8 * size).from_address(aligned)
            tensor = torch.frombuffer(storage, dtype=torch.bfloat16).view(
                pages, PAGE, 1, dim
            )
            yield HostAlias(tensor, mapped.value)
            torch.npu.synchronize()
        finally:
            if registered:
                assert self.acl.aclrtHostUnregister(aligned) == 0
            assert self.acl.aclrtFreeHost(raw) == 0

    def view(self, tensor):
        alias = isinstance(tensor, HostAlias)
        t = tensor.tensor if alias else tensor
        dtype = {torch.int32: 3, torch.bfloat16: 27, torch.float32: 0}[t.dtype]
        dims = (C.c_int64 * t.ndim)(*t.shape)
        strides = (C.c_int64 * t.ndim)(*t.stride())
        pointer = tensor.pointer if alias else t.data_ptr()
        # data_ptr already includes the view offset; storage metadata is the
        # view itself. Negative strides are not supported by these contracts.
        handle = self.meta.aclCreateTensor(
            dims, t.ndim, dtype, strides, 0, 2, dims, t.ndim, pointer
        )
        assert handle
        return handle

    def call(self, name, tensors, expect_error=False):
        handles = [self.view(t) for t in tensors]
        size, executor = C.c_uint64(), C.c_void_p()
        try:
            status = getattr(self.api, "aclnn" + name + "GetWorkspaceSize")(
                *handles, C.byref(size), C.byref(executor)
            )
            if expect_error:
                assert status != 0 and not executor.value and size.value == 0
                return status
            assert status == 0, (name, "GetWorkspaceSize", status)
            workspace = torch.empty(max(size.value, 1), dtype=torch.uint8, device="npu")
            self.keepalive.append(workspace)
            stream = torch.npu.current_stream().npu_stream
            status = getattr(self.api, "aclnn" + name)(
                workspace.data_ptr(), size.value, executor, stream
            )
            assert status == 0, (name, "Run", status)
        finally:
            for handle in handles:
                assert self.meta.aclDestroyTensor(handle) == 0


def npu(value, dtype=None):
    return torch.as_tensor(value, dtype=dtype or torch.int32, device="npu")


class Case:
    def __init__(self, runtime, rope=False, rows=4, pages=8, capacity=None):
        self.runtime, self.rope, self.rows = runtime, rope, rows
        self.capacity = rows if capacity is None else capacity
        assert self.capacity >= rows
        self.stack = ExitStack()
        self.layers = []
        for layer in range(3 if rope else 1):
            host = self.stack.enter_context(runtime.host(pages, LATENT))
            self.fill(host.tensor, layer, 0)
            data = {
                "host": host,
                "current": npu(
                    self.values(rows, LATENT, layer, 101).unsqueeze(1), torch.bfloat16
                ),
                "selection": torch.zeros(
                    self.capacity * 17,
                    PAGE,
                    1,
                    LATENT,
                    dtype=torch.bfloat16,
                    device="npu",
                ),
            }
            if rope:
                hr = self.stack.enter_context(runtime.host(pages, ROPE))
                self.fill(hr.tensor, layer, 31)
                data.update(
                    host_rope=hr,
                    current_rope=npu(
                        self.values(rows, ROPE, layer, 151).unsqueeze(1), torch.bfloat16
                    ),
                    selection_rope=torch.zeros(
                        self.capacity * 17,
                        PAGE,
                        1,
                        ROPE,
                        dtype=torch.bfloat16,
                        device="npu",
                    ),
                )
            self.layers.append(data)
        self.table = npu(
            [
                list(reversed(range(pages // 2))),
                list(reversed(range(pages // 2, pages))),
            ]
        )
        self.desc = npu([[0, i + 1, 260 + i, i] for i in range(rows)])
        self.map = npu([list(range(rows)) for _ in range(rows)])
        self.topk = torch.full((rows, 1, CAPACITY), -1, dtype=torch.int32, device="npu")
        self.append = npu([self.physical(0, 259 + i) for i in range(rows)])
        self.state = {
            "indices": torch.empty_like(self.topk),
            "compact": torch.empty_like(self.topk),
            "lengths": torch.empty(rows, dtype=torch.int32, device="npu"),
            "table": torch.empty(rows, 17, dtype=torch.int32, device="npu"),
            "resident": torch.full(
                (self.capacity, 1, CAPACITY), -1, dtype=torch.int32, device="npu"
            ),
            "slots": torch.full(
                (self.capacity, 1, CAPACITY), -1, dtype=torch.int32, device="npu"
            ),
            "count": torch.zeros(self.capacity, dtype=torch.int32, device="npu"),
            "plan_tokens": torch.empty_like(self.topk),
            "plan_slots": torch.empty_like(self.topk),
        }
        self.set_tokens([0, 127, 128, 258, 259, 260, 263, -1, 9999])

    @staticmethod
    def values(rows, dim, layer, bias):
        return (
            (torch.arange(rows * dim).reshape(rows, dim) % 251) + layer * 256 + bias
        ).to(torch.bfloat16)

    def fill(self, tensor, layer, bias):
        tensor.copy_(
            self.values(tensor.shape[0] * PAGE, tensor.shape[-1], layer, bias).view_as(
                tensor
            )
        )

    def close(self):
        torch.npu.synchronize()
        self.runtime.keepalive.clear()
        self.stack.close()

    def physical(self, req, token):
        return int(self.table.cpu()[req, token // PAGE]) * PAGE + token % PAGE

    def set_tokens(self, tokens):
        self.topk.fill_(-1)
        self.topk[:, 0, : len(tokens)].copy_(npu(tokens))
        # GLM Flash's kPool tail is at physical ranks 2048..2050, behind holes.
        self.topk[:, 0, 2048:2051].copy_(npu([251, 252, 253]))

    def args(self):
        s, layer = self.state, self.layers[0]
        args = [
            layer["host"],
            self.table,
            self.desc,
            self.map,
            self.topk,
            self.append,
            layer["current"],
            layer["selection"],
            s["indices"],
            s["lengths"],
            s["table"],
            s["resident"],
            s["slots"],
            s["count"],
        ]
        if self.rope:
            args.extend(
                [
                    layer["host_rope"],
                    layer["current_rope"],
                    layer["selection_rope"],
                    s["plan_tokens"],
                    s["plan_slots"],
                ]
            )
        else:
            args.append(s["compact"])
        return args

    def execute(self, check_readonly=True):
        name = "DsaKvOffloadingUpdateRopeWithPlan" if self.rope else "DsaKvOffloadingUpdateNope"
        self.runtime.call(name, self.args())
        if self.rope:
            if check_readonly:
                torch.npu.synchronize()
                plans = {
                    key: self.state[key].cpu().clone()
                    for key in ("plan_tokens", "plan_slots")
                }
            for layer in self.layers[1:]:
                self.runtime.call(
                    "DsaKvOffloadingServeRopePlanned",
                    [
                        layer["host"],
                        layer["host_rope"],
                        self.table,
                        self.desc,
                        self.map,
                        self.state["plan_tokens"],
                        self.state["plan_slots"],
                        self.append,
                        layer["current"],
                        layer["current_rope"],
                        layer["selection"],
                        layer["selection_rope"],
                    ],
                )
                if check_readonly:
                    torch.npu.synchronize()
                    for key, saved in plans.items():
                        assert torch.equal(self.state[key].cpu(), saved)

    def snapshots(self):
        return [
            {
                key: value.tensor.clone()
                for key, value in layer.items()
                if isinstance(value, HostAlias)
            }
            for layer in self.layers
        ]

    def check(self, before):
        torch.npu.synchronize()
        ids = self.state["indices"].cpu()[:, 0]
        desc, tokens, table = self.desc.cpu(), self.topk.cpu()[:, 0], self.table.cpu()
        rowmap, append = self.map.cpu(), self.append.cpu()
        for row, (req, window, visible, state_row) in enumerate(desc.tolist()):
            valid = (tokens[row] >= 0) & (tokens[row] < visible)
            assert int(self.state["lengths"].cpu()[row]) == int(valid.sum())
            assert torch.equal(ids[row][~valid], torch.full_like(ids[row][~valid], -1))
            slots = ids[row][valid].long()
            assert bool(((slots >= 0) & (slots < valid.sum())).all())
            assert len(slots.unique()) == len(slots)
            assert torch.equal(
                self.state["table"].cpu()[row],
                torch.arange(state_row * 17, (state_row + 1) * 17),
            )
            if visible > 0:
                count = int(self.state["count"].cpu()[state_row])
                assert count == int(valid.sum())
                assert torch.equal(
                    self.state["resident"].cpu()[state_row, 0, :count],
                    tokens[row][valid].sort(descending=True).values,
                )
            for layer, saved in zip(self.layers, before):
                for hostkey, currentkey, selectionkey, dim in (
                    ("host", "current", "selection", LATENT),
                    ("host_rope", "current_rope", "selection_rope", ROPE),
                ):
                    if hostkey not in layer:
                        continue
                    selected = (
                        layer[selectionkey].cpu().view(self.capacity, CAPACITY, dim)
                    )
                    current = layer[currentkey].cpu()[:, 0]
                    want = []
                    for token in tokens[row][valid].tolist():
                        if token >= visible - window:
                            want.append(
                                current[rowmap[row, token - (visible - window)]]
                            )
                        else:
                            want.append(
                                saved[hostkey][
                                    table[req, token // PAGE], token % PAGE, 0
                                ]
                            )
                    if want:
                        assert torch.equal(
                            selected[state_row, slots], torch.stack(want)
                        )
                    # Host append is unconditional for nonnegative physical
                    # slots, independently of whether this query has misses.
                    if append[row] >= 0:
                        actual = layer[hostkey].tensor.view(-1, dim)[append[row]]
                        assert torch.equal(actual, current[row])
        if self.rope:
            plan = self.state["plan_slots"].cpu()
            decoded = torch.where(plan < -1, -plan - 2, plan)
            assert torch.equal(decoded, self.state["indices"].cpu())
            assert torch.equal(
                self.state["plan_tokens"].cpu(),
                _expected_plan_tokens(self.topk.cpu(), desc),
            )


@pytest.fixture
def case_factory(runtime):
    cases = []

    def factory(**kwargs):
        case = Case(runtime, **kwargs)
        cases.append(case)
        return case

    yield factory
    for case in cases:
        case.close()


@pytest.mark.parametrize("rope", [False, True])
def test_public_update_kv_and_resident_transitions(case_factory, rope):
    c = case_factory(rope=rope)
    for tokens in (
        [0, 127, 128, 258, 259, 260, 263, -1, 9999],
        [1, 127, 129, 258, 259, 260, 263, -1, 9999],
        [127, 259],
        [127, 128, 129, 259, 260, 261, 262],
        [127, 128, 129, 259, 260, 261, 262],
    ):
        c.set_tokens(tokens)
        before = c.snapshots()
        c.execute()
        c.check(before)
    # All historical hits still append the current row. Reuse at a new request
    # requires the framework to clear the logical resident count first.
    c.desc[:, 1].zero_()
    c.set_tokens([0, 127, 128])
    before = c.snapshots()
    c.execute()
    c.check(before)
    before = c.snapshots()
    c.execute()
    c.check(before)
    c.state["count"].zero_()
    c.desc.copy_(npu([[1, i + 1, 260 + i, i] for i in range(c.rows)]))
    c.append.copy_(npu([c.physical(1, 259 + i) for i in range(c.rows)]))
    before = c.snapshots()
    c.execute()
    c.check(before)


@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_verify_rollback_current_refresh_and_history(case_factory, rope, accepted):
    c = case_factory(rope=rope)
    c.execute()
    torch.npu.synchronize()
    # A new verify replaces rejected latents at the same logical positions.
    # Framework reassigns each query descriptor; historical accepted prefix
    # remains in Host, current window must never use stale selection payload.
    start = 259 + accepted
    c.desc.copy_(npu([[0, i + 1, start + i + 1, i] for i in range(c.rows)]))
    c.append.copy_(npu([c.physical(0, start + i) for i in range(c.rows)]))
    c.set_tokens([0, 127, 128, 258, 259, 260, 261, 262, 263, 264, 265])
    for layer in c.layers:
        layer["current"].add_(17)
        if rope:
            layer["current_rope"].add_(19)
    before = c.snapshots()
    c.execute()
    c.check(before)
    # Force reread from Host after dropping this token from the resident set.
    c.state["count"].zero_()
    c.desc.copy_(npu([[0, 0, start + c.rows, i] for i in range(c.rows)]))
    c.append.fill_(-1)
    before = c.snapshots()
    c.execute()
    c.check(before)


@pytest.mark.parametrize("rope", [False, True])
def test_graph_dummy_busy_dummy_and_request_row_remap(case_factory, rope):
    c = case_factory(rope=rope)
    c.desc[:, 1:3].zero_()
    c.append.fill_(-1)
    # Warm up standard vendor before capture, no metadata frozen on CPU.
    for _ in range(3):
        c.execute()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        c.execute(check_readonly=False)
    for busy in (True, False, True, False):
        desc = [[1, i + 1, 260 + i, c.rows - i - 1] for i in range(c.rows)]
        if not busy:
            desc = [[1, 0, 0, c.rows - i - 1] for i in range(c.rows)]
        c.desc.copy_(npu(desc))
        c.append.copy_(
            npu([c.physical(1, 259 + i) if busy else -1 for i in range(c.rows)])
        )
        before = c.snapshots()
        resident_before = {
            key: c.state[key].cpu().clone() for key in ("resident", "slots", "count")
        }
        graph.replay()
        c.check(before)
        if not busy:
            for key, saved in resident_before.items():
                assert torch.equal(c.state[key].cpu(), saved)
            for layer, saved in zip(c.layers, before):
                assert torch.equal(layer["host"].tensor, saved["host"])
                if rope:
                    assert torch.equal(layer["host_rope"].tensor, saved["host_rope"])


@pytest.mark.parametrize("invalid", ["dtype", "shape", "strides", "alias"])
def test_public_api_rejects_invalid_views(case_factory, invalid):
    c = case_factory()
    args = c.args()
    if invalid == "dtype":
        args[4] = c.topk.float()
    elif invalid == "shape":
        args[4] = c.topk[:, :, :2051].contiguous()
    elif invalid == "strides":
        args[4] = torch.empty(c.rows, 1, CAPACITY * 2, dtype=torch.int32, device="npu")[
            :, :, ::2
        ]
    else:
        args[11] = args[8]
    c.runtime.call("DsaKvOffloadingUpdateNope", args, expect_error=True)


@pytest.mark.parametrize("rope", [False, True])
def test_selection_full_capacity(case_factory, rope):
    c = case_factory(rope=rope, rows=1, pages=36)
    c.topk.copy_(npu(list(reversed(range(CAPACITY)))).view_as(c.topk))
    c.desc.copy_(npu([[0, 1, CAPACITY, 0]]))
    c.append.copy_(npu([c.physical(0, CAPACITY - 1)]))
    before = c.snapshots()
    c.execute()
    c.check(before)


@pytest.mark.parametrize("accepted", [0, 1, 3])
def test_compact_indices_order_duplicates_and_rollback(case_factory, accepted):
    c = case_factory()
    compact = torch.empty_like(c.topk)
    for tokens in (
        [127, 127, -1, 0, 259, 259, 261, 9999],
        [127, 0, 127, -1, 259, 259, 261, 9999],
        [127, 127],
        [127, 127, 128, 128, 259, 259],
    ):
        c.set_tokens(tokens)
        before = c.snapshots()
        c.runtime.call("DsaKvOffloadingUpdateNope", c.args()[:-1] + [compact])
        c.check(before)
        _check_compact(c, compact)
    start = 259 + accepted
    c.desc.copy_(npu([[0, i + 1, start + i + 1, i] for i in range(c.rows)]))
    c.append.copy_(npu([c.physical(0, start + i) for i in range(c.rows)]))
    c.layers[0]["current"].add_(17)
    before = c.snapshots()
    c.runtime.call("DsaKvOffloadingUpdateNope", c.args()[:-1] + [compact])
    c.check(before)
    _check_compact(c, compact)


def _check_compact(c, compact):
    torch.npu.synchronize()
    positional = c.state["indices"].cpu()[:, 0]
    output = compact.cpu()[:, 0]
    for row in range(c.rows):
        selected = positional[row][positional[row] >= 0]
        assert torch.equal(output[row, : len(selected)], selected)
        assert bool((output[row, len(selected) :] == -1).all())
        assert int(c.state["lengths"].cpu()[row]) == len(selected)


def test_compact_graph_dummy_busy_and_full_capacity(case_factory):
    c = case_factory(rows=1, pages=36)
    compact = torch.empty_like(c.topk)
    c.topk.copy_(npu(list(reversed(range(CAPACITY)))).view_as(c.topk))
    c.desc.copy_(npu([[0, 0, 0, 0]]))
    c.append.fill_(-1)
    for _ in range(3):
        c.runtime.call("DsaKvOffloadingUpdateNope", c.args()[:-1] + [compact])
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        c.runtime.call("DsaKvOffloadingUpdateNope", c.args()[:-1] + [compact])
    for visible in (CAPACITY, 0, CAPACITY, 132, 0):
        c.desc.copy_(npu([[0, 0, visible, 0]]))
        before = c.snapshots()
        graph.replay()
        c.check(before)
        _check_compact(c, compact)


@pytest.mark.parametrize("invalid", ["alias", "dtype", "shape", "strides"])
def test_compact_output_rejects_invalid_views(case_factory, invalid):
    c = case_factory()
    if invalid == "alias":
        compact = c.state["indices"]
    elif invalid == "dtype":
        compact = c.topk.float()
    elif invalid == "shape":
        compact = c.topk[:, :, :2051].contiguous()
    else:
        compact = torch.empty(c.rows, 1, CAPACITY * 2, dtype=torch.int32, device="npu")[
            :, :, ::2
        ]
    c.runtime.call("DsaKvOffloadingUpdateNope", c.args()[:-1] + [compact], expect_error=True)


@pytest.mark.parametrize("rope", [False, True])
def test_capacity_exceeds_query_rows_high_slot_graph_reorder(case_factory, rope):
    """Resident addresses are independent of graph output/query row numbers."""
    c = case_factory(rope=rope, rows=2, capacity=8)
    c.desc[:, 3].copy_(npu([7, 5]))
    before = c.snapshots()
    c.execute()
    c.check(before)
    pointers = {key: value.data_ptr() for key, value in c.state.items()}
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        c.execute(check_readonly=False)
    for state_rows in ([5, 7], [7, 5]):
        c.desc[:, 3].copy_(npu(state_rows))
        before = c.snapshots()
        graph.replay()
        c.check(before)
    saved_resident = {
        key: c.state[key].cpu().clone() for key in ("resident", "slots", "count")
    }
    saved_payload = [
        {
            key: value.cpu().clone()
            for key, value in layer.items()
            if key in ("selection", "selection_rope")
        }
        for layer in c.layers
    ]
    c.desc[:, 1:3].zero_()
    c.append.fill_(-1)
    before = c.snapshots()
    graph.replay()
    c.check(before)
    for key, saved in saved_resident.items():
        assert torch.equal(c.state[key].cpu(), saved)
    for layer, saved in zip(c.layers, saved_payload):
        for key, value in saved.items():
            assert torch.equal(layer[key].cpu(), value)
    assert pointers == {key: value.data_ptr() for key, value in c.state.items()}
