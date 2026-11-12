# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""CPU contract checks and opt-in real ACLNN precision/replay tests."""
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from mega_kda_decode_reference import OUTPUTS, make_case, reference
from mega_kda_test_utils import GLM_KDA_HEADS, TP_SIZES, bitwise_equal, error_metrics


ROOT = Path(__file__).resolve().parents[2]
DECODE_MODES = [(0, 1)] + [(1, capacity) for capacity in range(1, 18)]
NPU_TEST = pytest.mark.skipif(os.getenv("MEGA_KDA_TEST_NPU") is None,
                              reason="native validation requires an explicitly selected exclusive NPU")


def test_native_host_policy(tmp_path):
    source = tmp_path / "policy.cpp"
    source.write_text(r'''
#include <cassert>
#include "mega_kda_decode_policy.h"
int main() {
    using namespace mega_kda;
    TilingData t{2, 2, 8, 1, 3, 4, 7, 5, 9};
    assert(SelectKey(0, false, t) == 1);
    assert(SelectKey(1, true, t) == 2);
    assert(SelectKey(0, true, t) == 0);
    assert(SelectKey(1, false, t) == 0);
    assert(SelectKey(2, false, t) == 0);
    for (int tp : {1, 2, 4, 8, 16, 32, 64}) {
        t.heads = 64 / tp;
        assert(SelectKey(0, false, t) == 1);
        assert(SelectKey(1, true, t) == 2);
    }
    t.heads = 8;
    t.max_query_tokens = 4; t.conv_history = 6; t.tokens = 7;
    assert(SelectKey(1, true, t) == 2);
    assert(SelectKey(0, false, t) == 0);
    t.tokens = 9;
    assert(SelectKey(1, true, t) == 0);
    t.tokens = 7; t.conv_history = 3;
    assert(SelectKey(1, true, t) == 0);
    t.conv_history = 6; t.heads = 129;
    assert(SelectKey(1, true, t) == 0);
    t.heads = 8;
    for (int capacity = 1; capacity <= 17; ++capacity) {
        t.max_query_tokens = capacity;
        t.conv_history = capacity + 2;
        t.tokens = t.batch * capacity;
        assert(SelectKey(1, true, t) == 2);
    }
    t.max_query_tokens = 18; t.conv_history = 20; t.tokens = 36;
    assert(SelectKey(1, true, t) == 0);
    assert(!FitsBytes({9223372036854775807LL, 2}));
    assert(!FitsBytes({-1, 2}));
}
''')
    exe = tmp_path / "policy"
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source),
                    "-I", str(ROOT / "xllm_ops/mega_kda_decode/op_host"), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_single_registered_entry():
    op = ROOT / "xllm_ops/mega_kda_decode"
    definitions = "\n".join(path.read_text() for path in op.rglob("*.cpp"))
    assert definitions.count("OP_ADD(MegaKdaDecode)") == 1
    assert definitions.count('__global__ __aicore__ void mega_kda_decode(') == 1
    assert "TILING_KEY_IS(1)" in definitions and "TILING_KEY_IS(2)" in definitions
    assert "GetRequiredInputShape" in definitions and "GetOptionalInputShape" in definitions
    assert "mega_kda_decode" in (ROOT / "xllm_ops/build_aclnn.sh").read_text()


@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
@pytest.mark.parametrize("bias", [False, True])
def test_reference_state_contract(mode, capacity, bias):
    data = make_case(batch=3, capacity=capacity, mode=mode, bias=bias, ragged=True, padding=True)
    snapshots = {k: v.clone() for k, v in data.items() if isinstance(v, torch.Tensor)}
    result = reference(data)
    for key, before in snapshots.items():
        assert torch.equal(before, data[key]), key
    out, conv, ssm = result
    assert [x.dtype for x in result] == [torch.bfloat16, torch.bfloat16, torch.float32]
    assert all(torch.isfinite(x).all() for x in result)
    assert torch.equal(conv[0], data["conv_state_out"][0])
    assert torch.equal(ssm[0], data["ssm_state_out"][0])
    for row in range(2):
        begin, end = data["cu_seqlens"][row:row + 2].tolist()
        accepted = int(data["num_accepted_tokens"][row]) if mode else 1
        slot = int(data["conv_write_indices"][row])
        assert torch.equal(conv[slot, :2], data["conv_state_in"][row, accepted:accepted + 2])
        assert torch.equal(conv[slot, 2:2 + end - begin], data["qkv"][begin:end])
        assert torch.equal(conv[slot, 2 + end - begin:], data["conv_state_out"][slot, 2 + end - begin:])
    for _ in range(20):
        assert all(bitwise_equal(a, b) for a, b in zip(result, reference(data)))


def test_reference_checkpoint_restore_and_discard():
    data = make_case(mode=1, capacity=4)
    first = reference(data)
    data["num_accepted_tokens"][0] = 1
    restored = reference(data)
    assert not torch.equal(first[0][0], restored[0][0])
    data["ssm_write_indices"].fill_(-1)
    data["conv_write_indices"].fill_(-1)
    discarded = reference(data)
    assert torch.equal(restored[0], discarded[0])
    assert torch.equal(discarded[1], data["conv_state_out"])
    assert torch.equal(discarded[2], data["ssm_state_out"])


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
def test_tp_reference_shape(tp, mode, capacity):
    heads = GLM_KDA_HEADS // tp
    data = make_case(heads=heads, mode=mode, capacity=capacity, bias=True)
    result = reference(data)
    assert data["qkv"].shape == (2 * capacity, heads * 384)
    assert data["gate"].shape == (2 * capacity, heads, 128)
    for name, value in zip(OUTPUTS, result):
        assert value.shape == data[name].shape and value.dtype == data[name].dtype
        assert torch.isfinite(value).all(), name
    assert result[2].shape[1:] == (heads, 128, 128)


@pytest.mark.parametrize("batch", [3, 4, 128])
def test_plain_padding_keeps_one_token_per_row(batch):
    data = make_case(batch=batch, heads=1, mode=0, ragged=True, padding=True)
    assert data["qkv"].shape[0] == batch
    assert torch.equal(data["cu_seqlens"], torch.arange(batch + 1, dtype=torch.int32))
    assert int(data["conv_read_indices"][-1]) == -1
    out, _, _ = reference(data)
    assert torch.equal(out[-1], data["output"][-1])


@pytest.mark.parametrize("invalid", ["duplicate_conv", "duplicate_ssm", "accepted", "offsets", "slot"])
def test_reference_rejects_invalid_metadata(invalid):
    data = make_case(mode=1, capacity=4)
    if invalid == "duplicate_conv":
        data["conv_write_indices"].fill_(0)
    elif invalid == "duplicate_ssm":
        data["ssm_write_indices"].fill_(0)
    elif invalid == "accepted":
        data["num_accepted_tokens"][0] = 0
    elif invalid == "offsets":
        data["cu_seqlens"][-1] += 1
    else:
        data["conv_write_indices"][0] = 999
    with pytest.raises(ValueError):
        reference(data)


def npu_case(data):
    device = os.getenv("MEGA_KDA_TEST_NPU")
    if device is None:
        pytest.skip("set MEGA_KDA_TEST_NPU to an exclusively available NPU for native tests")
    import torch_npu  # noqa: F401
    import custom_ops_lib
    torch.npu.set_device(int(device))
    tensors = {k: v.to(f"npu:{device}") if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    return custom_ops_lib, tensors


def assert_native(data, actual, input_snapshots=None, expected=None):
    if expected is None:
        expected = reference(data)
    records = {}
    for name, want in zip(OUTPUTS, expected):
        got = actual[name].cpu()
        assert got.shape == want.shape and got.dtype == want.dtype, name
        assert torch.isfinite(got).all() and torch.isfinite(want).all(), name
        if name == "conv_state_out":
            assert bitwise_equal(got, want), name
            continue
        rtol, atol = (1e-2, 1e-3) if name == "output" else (1e-3, 1e-4)
        active = want != 37
        stats = error_metrics(got[active], want[active], near=0.001, relative=rtol, absolute=atol)
        records[name] = stats
        assert stats["finite"], (name, stats)
        # Retain the existing pointwise gate; split metrics are additional diagnostics.
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
        # Pool padding and discarded writes are exact, even when active arithmetic is approximate.
        untouched = want == 37
        assert bitwise_equal(got[untouched], want[untouched]), name
    for key, value in (data if input_snapshots is None else input_snapshots).items():
        if isinstance(value, torch.Tensor) and key not in OUTPUTS:
            assert bitwise_equal(actual[key].cpu(), value), key
    print(json.dumps({"mode": data["mode"], "capacity": data["max_query_tokens"],
                      "shape": list(data["gate"].shape), "errors": records}), flush=True)
    return expected


def check_fixed20(data):
    ops, actual = npu_case(data)
    for _ in range(3):
        ops.mega_kda_decode(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = ops.mega_kda_decode(**actual)
    assert all(a.data_ptr() == actual[name].data_ptr() for a, name in zip(result, OUTPUTS))
    expected = reference(data)
    first = None
    for _ in range(20):
        graph.replay()
        assert_native(data, actual, expected=expected)
        current = [actual[name].cpu() for name in OUTPUTS]
        if first is None:
            first = current
        assert all(bitwise_equal(a, b) for a, b in zip(first, current))


@pytest.mark.parametrize("batch,heads", [(1, 1), (2, 2), (4, 2), (8, 8), (16, 8), (32, 16), (64, 16), (128, 32)])
@pytest.mark.parametrize("mode,capacity", [(0, 1), (1, 1), (1, 4)])
@NPU_TEST
def test_native_fixed20(batch, heads, mode, capacity):
    data = make_case(batch=batch, heads=heads, capacity=capacity, mode=mode,
                     bias=True, ragged=True, padding=batch > 1)
    check_fixed20(data)


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
@NPU_TEST
def test_native_tp_fixed20(tp, mode, capacity):
    data = make_case(heads=GLM_KDA_HEADS // tp, capacity=capacity, mode=mode, bias=True)
    check_fixed20(data)


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
@NPU_TEST
def test_native_graph_changed20_carry20(tp, mode, capacity):
    data = make_case(heads=GLM_KDA_HEADS // tp, capacity=capacity, mode=mode)
    ops, actual = npu_case(data)
    for _ in range(3):
        ops.mega_kda_decode(**actual)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        ops.mega_kda_decode(**actual)
    for carry in (False, True):
        for step in range(20):
            data["qkv"].add_(0.001)
            actual["qkv"].copy_(data["qkv"])
            if mode:
                data["num_accepted_tokens"].fill_(step % capacity + 1)
                actual["num_accepted_tokens"].copy_(data["num_accepted_tokens"])
            # Change destinations without replacing bindings or recapturing.
            data["conv_write_indices"].copy_(torch.tensor([1, 2] if step % 2 else [2, 1], dtype=torch.int32))
            actual["conv_write_indices"].copy_(data["conv_write_indices"])
            snapshots = {k: v.cpu() for k, v in actual.items()
                         if isinstance(v, torch.Tensor) and k not in OUTPUTS}
            graph.replay()
            expected = assert_native(data, actual, snapshots)
            for name, result in zip(OUTPUTS, expected):
                data[name].copy_(result)
            if carry:
                for row in range(2):
                    cr, cw = int(data["conv_read_indices"][row]), int(data["conv_write_indices"][row])
                    data["conv_state_in"][cr].copy_(expected[1][cw])
                    actual["conv_state_in"][cr].copy_(actual["conv_state_out"][cw])
                    for token in range(capacity):
                        sr, sw = int(data["ssm_read_indices"][row, token]), int(data["ssm_write_indices"][row, token])
                        data["ssm_state_in"][sr].copy_(expected[2][sw])
                        actual["ssm_state_in"][sr].copy_(actual["ssm_state_out"][sw])


@NPU_TEST
def test_native_rejects_input_output_alias():
    data = make_case()
    ops, actual = npu_case(data)
    actual["conv_state_out"] = actual["conv_state_in"]
    with pytest.raises(RuntimeError, match="must not alias"):
        ops.mega_kda_decode(**actual)
