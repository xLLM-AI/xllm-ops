# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""ACLNN Decode precision and graph replay tests on NPU."""
import json
import os

import pytest
import torch

from mega_kda_decode_reference import (
    MTP, NATIVE, OUTPUTS, PUBLIC_COMPAT, make_case, reference, reference_910b,
)
from mega_kda_test_utils import (
    GLM_KDA_HEADS, TP_SIZES, bitwise_equal, checked_written_rows, error_metrics, is_a5_device,
)


DECODE_MODES = [(NATIVE, 1), (PUBLIC_COMPAT, 1)] + [(MTP, capacity) for capacity in range(1, 18)]
NPU_TEST = pytest.mark.skipif(os.getenv("MEGA_KDA_TEST_NPU") is None,
                              reason="native validation requires an explicitly selected exclusive NPU")


def _public_discard_case() -> dict:
    data = make_case(batch=3, heads=8, mode=PUBLIC_COMPAT)
    data["conv_read_indices"][2] = -1
    data["ssm_read_indices"][1] = -1
    data["conv_write_indices"][0] = -1
    data["ssm_write_indices"][0] = -1
    return data


def _public_case(batch: int, heads: int, variant: str) -> dict:
    data = make_case(batch=batch, heads=heads, mode=PUBLIC_COMPAT)
    channels_per_part = heads * 128
    if variant == "zero":
        data["conv_weight"][:, :2 * channels_per_part].zero_()
    elif variant == "tiny":
        data["qkv"].fill_(1)
        data["conv_state_in"].fill_(1)
        data["ssm_state_in"].fill_(1)
        data["conv_weight"][:, :2 * channels_per_part].fill_(1e-7)
    elif variant == "wide":
        # BF16 values above FP16's finite range, with finite small Conv products.
        data["qkv"].fill_(70000)
        data["conv_state_in"].fill_(70000)
        data["conv_weight"].fill_(1e-6)
    elif variant == "gate_extremes":
        data["gate"][:, :, ::2] = -80
        data["gate"][:, :, 1::2] = 80
        data["gate_bias"].zero_()
        data["a_log"].zero_()
    else:
        raise ValueError(variant)
    return data


@NPU_TEST
@pytest.mark.parametrize(("mode", "expected"), [(MTP, 13.0)])
def test_a5_decode_silu_rounding_at_bf16_midpoint(mode, expected):
    # Four BF16 taps from the xLLM main Conv replay. Route their activation
    # through V with a single unit K and zero initial state to expose it in SSM.
    data = make_case(batch=1, heads=1, mode=mode)
    data["qkv"].zero_()
    data["conv_weight"].zero_()
    data["conv_state_in"].zero_()
    data["ssm_state_in"].zero_()
    data["beta"].fill_(64)
    data["qkv"][0, 128] = 32
    data["conv_weight"][-1, 128] = 1
    taps = torch.tensor([-0.43359375, 0.435546875, -3.1875, 1.2578125])
    data["conv_state_in"][0, :3, 256] = taps[:3]
    data["qkv"][0, 256] = taps[-1]
    data["conv_weight"][:, 256] = torch.tensor([1.59375, 3.578125, -3.3125, 1.2265625])
    lib, actual = npu_case(data)
    if not is_a5_device(actual["qkv"].device):
        pytest.skip("A5 Conv/SiLU rounding")
    lib.mega_kda_decode(**actual)
    slot = int(data["ssm_write_indices"][0, 0])
    assert actual["ssm_state_out"][slot, 0, 0, 0].item() == expected


def npu_case(data, *, require_supported_mode=True):
    device = os.getenv("MEGA_KDA_TEST_NPU")
    if device is None:
        pytest.skip("set MEGA_KDA_TEST_NPU to an exclusively available NPU for native tests")
    import torch_npu  # noqa: F401
    import custom_ops_lib
    torch.npu.set_device(int(device))
    if require_supported_mode and data["mode"] == PUBLIC_COMPAT and is_a5_device(torch.device(f"npu:{device}")):
        pytest.skip("Public compatibility mode is supported on Ascend 910B and 910C")
    tensors = {k: v.to(f"npu:{device}") if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    return custom_ops_lib, tensors


def device_reference(data, actual):
    device = actual["qkv"].device
    if device.type == "cpu":
        return reference(data)
    if is_a5_device(device):
        return reference(data, a5=True)
    return reference_910b(data)


def decode_write_masks(data):
    tokens = torch.zeros(data["output"].size(0), dtype=torch.bool)
    slots = torch.zeros(data["ssm_state_out"].size(0), dtype=torch.bool)
    offsets = data["cu_seqlens"].tolist()
    accepted = data["num_accepted_tokens"]
    for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        if begin == end:
            continue
        checkpoint = 0 if accepted is None else int(accepted[row]) - 1
        if data["conv_read_indices"][row] < 0 or data["ssm_read_indices"][row, checkpoint] < 0:
            continue
        tokens[begin:end] = True
        destinations = data["ssm_write_indices"][row, :end - begin]
        slots[destinations[destinations >= 0].long()] = True
    return {"output": tokens, "ssm_state_out": slots}


def assert_native(data, actual, input_snapshots=None, expected=None):
    if expected is None:
        expected = device_reference(data, actual)
    written = decode_write_masks(data)
    before = data if input_snapshots is None else input_snapshots
    records = {}
    for name, want in zip(OUTPUTS, expected):
        got = actual[name].cpu()
        assert got.shape == want.shape and got.dtype == want.dtype, name
        assert torch.isfinite(got).all() and torch.isfinite(want).all(), name
        if name == "conv_state_out":
            assert bitwise_equal(got, want), name
            continue
        rtol, atol = (1e-2, 1e-3) if name == "output" else (1e-3, 1e-4)
        got, want = checked_written_rows(got, want, before.get(name, data[name]), written[name], name)
        stats = error_metrics(got, want, near=0.001, relative=rtol, absolute=atol)
        records[name] = stats
        assert stats["finite"], (name, stats)
        # Retain the existing pointwise gate; split metrics are additional diagnostics.
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)
    for key, value in before.items():
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
    expected = device_reference(data, actual)
    first = None
    for _ in range(20):
        graph.replay()
        assert_native(data, actual, expected=expected)
        current = [actual[name].cpu() for name in OUTPUTS]
        if first is None:
            first = current
        assert all(bitwise_equal(a, b) for a, b in zip(first, current))


@pytest.mark.parametrize("batch,heads", [(1, 1), (2, 2), (4, 2), (8, 8), (16, 8), (32, 16), (64, 16), (128, 32)])
@pytest.mark.parametrize("mode,capacity", [(NATIVE, 1), (PUBLIC_COMPAT, 1), (MTP, 1), (MTP, 4)])
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


def check_changed20_carry20(data: dict) -> None:
    mode, capacity = data["mode"], data["max_query_tokens"]
    batch = data["conv_read_indices"].numel()
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
            if mode == MTP:
                data["num_accepted_tokens"].fill_(step % capacity + 1)
                actual["num_accepted_tokens"].copy_(data["num_accepted_tokens"])
            # Change destinations without replacing bindings or recapturing.
            destinations = torch.arange(1, batch + 1, dtype=torch.int32)
            data["conv_write_indices"].copy_(destinations if step % 2 else destinations.flip(0))
            actual["conv_write_indices"].copy_(data["conv_write_indices"])
            snapshots = {k: v.cpu() for k, v in actual.items()
                         if isinstance(v, torch.Tensor) and k != "conv_state_out"}
            graph.replay()
            expected = assert_native(data, actual, snapshots)
            for name, result in zip(OUTPUTS, expected):
                data[name].copy_(result)
            if carry:
                for row in range(batch):
                    cr, cw = int(data["conv_read_indices"][row]), int(data["conv_write_indices"][row])
                    data["conv_state_in"][cr].copy_(expected[1][cw])
                    actual["conv_state_in"][cr].copy_(actual["conv_state_out"][cw])
                    for token in range(capacity):
                        sr, sw = int(data["ssm_read_indices"][row, token]), int(data["ssm_write_indices"][row, token])
                        data["ssm_state_in"][sr].copy_(expected[2][sw])
                        actual["ssm_state_in"][sr].copy_(actual["ssm_state_out"][sw])


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("mode,capacity", DECODE_MODES)
@NPU_TEST
def test_native_graph_changed20_carry20(tp, mode, capacity):
    data = make_case(heads=GLM_KDA_HEADS // tp, capacity=capacity, mode=mode)
    check_changed20_carry20(data)


@pytest.mark.parametrize("batch,heads", [(1, 8), (4, 8), (5, 8), (2, 3)])
@NPU_TEST
def test_public_compat_specialized_fixed20_changed20_carry20(batch: int, heads: int) -> None:
    data = make_case(batch=batch, heads=heads, mode=PUBLIC_COMPAT, bias=True)
    check_fixed20(data)
    check_changed20_carry20(data)


@pytest.mark.parametrize("batch,heads", [(1, 8), (4, 8), (5, 8), (2, 3)])
@pytest.mark.parametrize("variant", ["zero", "tiny", "wide", "gate_extremes"])
@NPU_TEST
def test_public_compat_numerical_edges(batch: int, heads: int, variant: str) -> None:
    data = _public_case(batch, heads, variant)
    ops, actual = npu_case(data)
    ops.mega_kda_decode(**actual)
    assert_native(data, actual)


@NPU_TEST
def test_public_compat_row_dispatch() -> None:
    # Set this from the validation device's PlatformAscendC GetCoreNumAiv value.
    # The launch uses min(AIV cores, 8*B*H); record the actual launch calculation.
    core_count = os.getenv("MEGA_KDA_TEST_AIV_CORES")
    if core_count is None:
        pytest.skip("set MEGA_KDA_TEST_AIV_CORES from the selected device for dispatch coverage")
    cores = int(core_count)
    assert cores > 2
    rows_seen = set()
    for batch, heads in ((1, 1), ((cores + 1) // 2, 1)):
        blocks = min(cores, 8 * batch * heads)
        rows = 64 if 2 * batch * heads >= blocks else 32
        data = make_case(batch=batch, heads=heads, mode=PUBLIC_COMPAT)
        ops, actual = npu_case(data)
        ops.mega_kda_decode(**actual)
        assert_native(data, actual)
        rows_seen.add(rows)
        print(json.dumps({"aiv_cores": cores, "launched_blocks": blocks, "batch": batch,
                          "heads": heads, "v_rows": rows, "mode": PUBLIC_COMPAT}), flush=True)
    assert rows_seen == {32, 64}


@pytest.mark.parametrize("invalid", ["accepted", "capacity"])
@NPU_TEST
def test_public_compat_binding_rejects_mtp_contract(invalid: str) -> None:
    data = make_case(mode=PUBLIC_COMPAT)
    if invalid == "accepted":
        data["num_accepted_tokens"] = torch.ones(2, dtype=torch.int32)
    else:
        data["max_query_tokens"] = 2
    ops, actual = npu_case(data, require_supported_mode=False)
    with pytest.raises(RuntimeError, match="mode/capacity/accepted-count"):
        ops.mega_kda_decode(**actual)


@NPU_TEST
def test_public_compat_nonunit_offsets_do_not_publish() -> None:
    data = make_case(mode=PUBLIC_COMPAT)
    data["cu_seqlens"][:] = torch.tensor([0, 0, 2], dtype=torch.int32)
    ops, actual = npu_case(data)
    ops.mega_kda_decode(**actual)
    for name in OUTPUTS:
        assert bitwise_equal(actual[name].cpu(), data[name])


@NPU_TEST
def test_public_compat_native_inactive_and_discard() -> None:
    data = _public_discard_case()
    ops, actual = npu_case(data)
    ops.mega_kda_decode(**actual)
    assert_native(data, actual)


@NPU_TEST
def test_native_rejects_input_output_alias():
    data = make_case()
    ops, actual = npu_case(data)
    actual["conv_state_out"] = actual["conv_state_in"]
    with pytest.raises(RuntimeError, match="must not alias"):
        ops.mega_kda_decode(**actual)
