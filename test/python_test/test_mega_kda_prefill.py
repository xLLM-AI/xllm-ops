# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""ACLNN Prefill precision and graph replay tests on NPU."""
import json
import os

import pytest
import torch

from mega_kda_prefill_reference import OUTPUTS, make_case, reference, reference_910b
from mega_kda_test_utils import (
    GLM_KDA_HEADS, TP_SIZES, bitwise_equal, checked_written_rows, error_metrics, is_a5_device,
)

NATIVE = pytest.mark.skipif(os.getenv("MEGA_KDA_PREFILL_TEST_NPU") is None,
                           reason="select an exclusive NPU explicitly")


def native_case(data):
    import torch_npu  # noqa: F401
    import custom_ops_lib
    device = int(os.environ["MEGA_KDA_PREFILL_TEST_NPU"])
    torch.npu.set_device(device)
    actual = {k: v.to(f"npu:{device}") if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    return custom_ops_lib, actual


def check(data, actual, label, input_snapshots=None):
    device = actual["qkv"].device
    if device.type == "cpu":
        expected = reference(data)
    elif is_a5_device(device):
        expected = reference(data, a5=True)
    else:
        expected = reference_910b(data)
    offsets = data["cu_seqlens"]
    destinations = data["ssm_write_indices"][offsets[1:] > offsets[:-1]]
    written = torch.zeros(data["ssm_state_out"].size(0), dtype=torch.bool)
    written[destinations[destinations >= 0].long()] = True
    before = data if input_snapshots is None else input_snapshots
    records = {}
    for name, want in zip(OUTPUTS, expected):
        got = actual[name].cpu()
        assert got.shape == want.shape and got.dtype == want.dtype, (label, name)
        assert torch.isfinite(got).all() and torch.isfinite(want).all(), (label, name)
        if name == "conv_state_out":
            assert bitwise_equal(got, want), name
            continue
        if name == "ssm_state_out":
            got, want = checked_written_rows(got, want, before.get(name, data[name]), written, name)
        stats = error_metrics(got, want, near=0.001, absolute=0.00001)
        records[name] = stats
        assert stats["finite"], (label, name, stats)
        assert stats["normal_over_fraction"] <= 0.001, (label, name, stats)
        assert stats["near_over_fraction"] <= 0.001, (label, name, stats)
    for k, x in before.items():
        if isinstance(x, torch.Tensor) and k not in OUTPUTS:
            assert bitwise_equal(x, actual[k].cpu()), k
    print(json.dumps({"case": label, "errors": records}), flush=True)
    return expected


@NATIVE
@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("lengths", [(1,), (2, 0, 3), (127,), (128,), (129, 0, 3), (257, 5)])
@pytest.mark.parametrize("bias", [False, True])
def test_native_accuracy(tp, lengths, bias):
    heads = GLM_KDA_HEADS // tp
    data = make_case(lengths, heads=heads, bias=bias)
    lib, actual = native_case(data)
    for _ in range(3): lib.mega_kda_prefill(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        lib.mega_kda_prefill(**actual)
    graph.replay()
    check(data, actual, f"TP={tp}/H={heads}/{lengths}/bias={bias}")


@NATIVE
@pytest.mark.parametrize(("tokens", "heads"), [(1, 8), (131, 1)])
def test_a5_prefill_conv_preserves_small_bf16_inputs(tokens, heads):
    # Both BF16 pairs have the same exact FP32 product. Narrowing the
    # smaller input to FP16 would round it to zero.
    data = make_case((tokens,), heads=heads)
    lib, actual = native_case(data)
    if not torch.npu.get_device_name().lower().startswith(("ascend950", "ascend910_95")):
        pytest.skip("A5 Conv input rounding")
    data["qkv"].fill_(1)
    data["conv_weight"].zero_()
    data["conv_weight"][-1].fill_(1)
    v_channels = slice(2 * heads * 128, None)
    results = []
    for value, weight in [(0.5, 1.0), (2**-25, 2**24)]:
        data["qkv"][:, v_channels] = value
        data["conv_weight"][-1, v_channels] = weight
        actual["qkv"].copy_(data["qkv"])
        actual["conv_weight"].copy_(data["conv_weight"])
        lib.mega_kda_prefill(**actual)
        output = actual["output"].cpu()
        ssm = actual["ssm_state_out"][int(data["ssm_write_indices"][0])].cpu()
        results.append((output, ssm))
    for baseline, small_input in zip(*results):
        assert torch.isfinite(baseline).all() and torch.count_nonzero(baseline) > 0
        torch.testing.assert_close(small_input, baseline, rtol=0, atol=0)


@NATIVE
@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5])
def test_a5_short_prefill_conv_cancellation_matches_general(tokens):
    heads, dim = 8, 128
    data = make_case((5,), heads=heads, bias=True)
    data["qkv"].fill_(1)
    data["conv_state_in"].fill_(1)
    data["conv_read_indices"].fill_(0)
    data["gate"].fill_(-40)
    data["beta"].zero_()
    data["beta"][0].fill_(1)
    weights = data["conv_weight"].view(4, 3, heads, dim).zero_()
    bias = data["conv_bias"].view(3, heads, dim).zero_()
    bias[:2, :, 0] = 32  # Q and K normalize to a unit vector in BF16.
    bias[2, :, 0] = 2**24
    weights[0, 2, :, 0] = 0.5
    weights[1, 2, :, 0] = -(2**24)
    weights[2, 2, :, 0] = 0.5

    # The exact Conv sum is 1. Only the first token writes a new value to SSM.
    value = torch.nn.functional.silu(torch.tensor(1.0)).bfloat16().float()
    slot = int(data["ssm_write_indices"][0])
    expected_state = torch.zeros_like(data["ssm_state_out"][slot])
    expected_state[:, 0, 0] = value
    expected_output = torch.zeros_like(data["output"])
    expected_output[:, :, 0] = (value * dim**-0.5).bfloat16()
    case = dict(data, cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32))
    for name in ("qkv", "gate", "beta", "output"):
        case[name] = data[name][:tokens]
    lib, actual = native_case(case)
    if not is_a5_device(actual["qkv"].device):
        pytest.skip("A5 short Prefill Conv accumulation")
    lib.mega_kda_prefill(**actual)
    torch.testing.assert_close(actual["ssm_state_out"][slot].cpu(), expected_state, rtol=0, atol=0)
    torch.testing.assert_close(actual["output"].cpu(), expected_output[:tokens], rtol=0, atol=0)


@NATIVE
@pytest.mark.parametrize("heads", [1, 3, 8])
def test_910b_prefill_conv_rounding_is_independent_of_head_count(heads):
    # A unit K and zero initial state expose the BF16 SiLU result in SSM.
    data = make_case((1,), heads=heads)
    for name in ("qkv", "conv_weight", "conv_state_in", "ssm_state_in"):
        data[name].zero_()
    data["conv_read_indices"].fill_(0)
    data["beta"].fill_(1)
    k_lane, v_lane = heads * 128, 2 * heads * 128
    data["qkv"][0, k_lane] = 32
    data["conv_weight"][-1, k_lane] = 1
    data["qkv"][0, v_lane] = 1
    data["conv_weight"][-1, v_lane] = 1
    data["conv_state_in"][0, 0, v_lane] = 1
    data["conv_weight"][0, v_lane] = 1 / 256
    lib, actual = native_case(data)
    if is_a5_device(actual["qkv"].device):
        pytest.skip("Conv rounding on Ascend 910B and 910C")
    lib.mega_kda_prefill(**actual)
    expected = torch.nn.functional.silu(torch.tensor(1 + 1 / 256)).bfloat16()
    slot = int(data["ssm_write_indices"][0])
    assert actual["ssm_state_out"][slot, 0, 0, 0].item() == expected.item()


@NATIVE
def test_native_alias_and_shape_rejection():
    data = make_case((1,), heads=1)
    lib, actual = native_case(data)
    bad = dict(actual, conv_state_out=actual["conv_state_in"])
    with pytest.raises(RuntimeError, match="alias"):
        lib.mega_kda_prefill(**bad)
    bad = dict(actual, gate=actual["gate"].float())
    with pytest.raises(RuntimeError):
        lib.mega_kda_prefill(**bad)


@NATIVE
def test_native_dynamic_invalid_metadata():
    data = make_case((1, 0, 2), heads=1)
    lib, actual = native_case(data)
    lib.mega_kda_prefill(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph): lib.mega_kda_prefill(**actual)
    graph.replay()
    before = [actual[name].cpu() for name in OUTPUTS]
    for name, values in (
        ("cu_seqlens", [0, 2, 1, 3]),
        ("conv_read_indices", [-2, 0, 0]),
        ("ssm_write_indices", [1, 0, 1]),
        ("conv_write_indices", [999, -1, -1]),
    ):
        actual[name].copy_(torch.tensor(values, dtype=torch.int32))
        graph.replay()
        assert all(torch.equal(actual[n].cpu(), want) for n, want in zip(OUTPUTS, before))
        actual[name].copy_(data[name])
    # Restore valid metadata, then exercise discarded writes in the same captured graph.
    for family in ("conv", "ssm"):
        name = f"{family}_write_indices"
        data[name].fill_(-1)
        actual[name].copy_(data[name])
    graph.replay()
    for name, want in zip(OUTPUTS[1:], before[1:]):
        assert torch.equal(actual[name].cpu(), want)


@NATIVE
@pytest.mark.parametrize(
    ("tokens", "tp"),
    [
        pytest.param(tokens, tp, id=f"{tokens}-tp{tp}")
        for tokens in (1024, 2048, 4096, 8192, 16384, 32768)
        for tp in TP_SIZES
        if (tokens, tp) not in {
            (1024, 16),
            (2048, 1),
            (8192, 8),
            (8192, 16),
            (32768, 16),
            (32768, 64),
        }
    ],
)
def test_native_common_lengths(tp, tokens):
    heads = GLM_KDA_HEADS // tp
    data = make_case((tokens,), heads=heads, bias=True)
    # Raise output magnitude so relative-error checks cover normal values too.
    data["qkv"].mul_(5)
    data["conv_weight"].mul_(2)
    lib, actual = native_case(data)
    for _ in range(3): lib.mega_kda_prefill(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph): lib.mega_kda_prefill(**actual)
    graph.replay()
    check(data, actual, f"TP={tp}/T={tokens}/H={heads}")
    first = [actual[name].cpu() for name in OUTPUTS]
    for iteration in range(20):
        graph.replay()
        matches = {name: bitwise_equal(actual[name].cpu(), want)
                   for name, want in zip(OUTPUTS, first)}
        assert all(matches.values()), (tp, tokens, iteration, matches)


@NATIVE
@pytest.mark.parametrize("batch,heads", [(4, 8), (8, 8), (1, 32)])
def test_native_core_recycling(batch, heads):
    data = make_case(tuple(129 if i % 2 == 0 else 3 for i in range(batch)), heads=heads, bias=True)
    lib, actual = native_case(data)
    lib.mega_kda_prefill(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph): lib.mega_kda_prefill(**actual)
    graph.replay()
    check(data, actual, f"batch={batch}/heads={heads}")


@NATIVE
@pytest.mark.parametrize("gate_bias,beta", [(-10.0, 0.0), (0.0, 1.0), (4.0, 0.5)])
def test_native_gate_range(gate_bias, beta):
    data = make_case((129, 1), heads=2, bias=True)
    data["gate_bias"].fill_(gate_bias)
    data["beta"].fill_(beta)
    lib, actual = native_case(data)
    lib.mega_kda_prefill(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph): lib.mega_kda_prefill(**actual)
    graph.replay()
    check(data, actual, f"gate_bias={gate_bias}/beta={beta}")


@NATIVE
@pytest.mark.parametrize("tp", [tp for tp in TP_SIZES if tp != 8], ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("carry", [False, True])
def test_native_graph_100(tp, carry):
    heads = GLM_KDA_HEADS // tp
    data = make_case((3, 1), heads=heads, bias=True)
    lib, actual = native_case(data)
    for _ in range(3): lib.mega_kda_prefill(**actual)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        lib.mega_kda_prefill(**actual)
    fixed = None
    for iteration in range(100):
        snapshots = {k: v.cpu() for k, v in actual.items()
                     if isinstance(v, torch.Tensor) and k not in ("output", "conv_state_out")}
        graph.replay()
        expected = check(data, actual, f"TP={tp}/H={heads}/carry={carry}/{iteration}", snapshots)
        current = [actual[name].cpu() for name in OUTPUTS]
        if not carry:
            if fixed is None: fixed = current
            assert all(bitwise_equal(a, b) for a, b in zip(fixed, current)), iteration
        else:
            # Advance CPU and NPU state independently, including changed Device mappings.
            for b in range(2):
                for i, family in enumerate(("conv", "ssm"), 1):
                    slot = int(data[f"{family}_write_indices"][b])
                    data[f"{family}_state_in"][b].copy_(expected[i][slot])
                    actual[f"{family}_state_in"][b].copy_(actual[OUTPUTS[i]][slot])
                    data[f"{family}_read_indices"][b] = b
                data["conv_write_indices"][b] = (iteration + 1) % 2 * 2 + b
                data["ssm_write_indices"][b] = (iteration + 1) % 2 * 2 + b
            data["qkv"] = -data["qkv"]
            actual["qkv"].copy_(data["qkv"])
            data["cu_seqlens"][1] = 2 + iteration % 2
            actual["cu_seqlens"].copy_(data["cu_seqlens"])
            for family in ("conv", "ssm"):
                for mode in ("read", "write"):
                    name = f"{family}_{mode}_indices"
                    actual[name].copy_(data[name])
            for name, value in zip(OUTPUTS, expected):
                data[name].copy_(value)
