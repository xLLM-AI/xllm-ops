"""Opt-in 500-step true state recurrence, including cross-chunk graph replay."""
import json
import os
import time

import pytest
import torch

from mega_kda_prefill_reference import OUTPUTS, make_case
from test_mega_kda_prefill import NATIVE, check, native_case


@NATIVE
@pytest.mark.parametrize("length", [1, 67])
@pytest.mark.parametrize("kind", ["cancel_small", "cancel_regular", "large_norm_61", "large_norm_127"])
def test_native_frontend_finite_boundaries(length, kind):
    # Finite public inputs can have substantial cancellation or a Q/K norm
    # whose naive FP32 square sum overflows. Cover scalar and batched rows.
    data = make_case((length,), heads=1, bias=True, seed=71)
    data["qkv"].fill_(1)
    data["conv_state_in"].fill_(1)
    data["conv_read_indices"].fill_(0)
    data["conv_weight"].zero_()
    if kind.startswith("large_norm"):
        data["conv_bias"][:256].fill_(2.0 ** int(kind.rsplit("_", 1)[1]))
        data["conv_bias"][256:].fill_(1)
    else:
        data["conv_bias"].fill_(2.0 ** 24)
        data["conv_weight"][0].fill_(0.5)
        data["conv_weight"][1].fill_(-(2.0 ** 24))
        if kind == "cancel_regular":
            data["conv_weight"][2].fill_(0.5)
    lib, actual = native_case(data)
    lib.mega_kda_prefill(**actual)
    check(data, actual, f"finite_boundary/{kind}/T={length}/eager")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        lib.mega_kda_prefill(**actual)
    for _ in range(10):
        graph.replay()
    check(data, actual, f"finite_boundary/{kind}/T={length}/graph")


@NATIVE
@pytest.mark.parametrize("heads", [1, 8, 64])
@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0, 4.0])
def test_native_activation_range(heads, scale):
    data = make_case((129, 65, 3, 0), heads=heads, bias=True, seed=91)
    data["conv_weight"].mul_(scale)
    data["conv_bias"].mul_(scale)
    lib, actual = native_case(data)
    lib.mega_kda_prefill(**actual)
    check(data, actual, f"activation_range/H={heads}/scale={scale}/eager")
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        lib.mega_kda_prefill(**actual)
    for _ in range(10):
        graph.replay()
    check(data, actual, f"activation_range/H={heads}/scale={scale}/graph")


@NATIVE
@pytest.mark.parametrize("heads", [1, 8, 64])
@pytest.mark.parametrize("gate_bias", [-4.0], ids=["normal_decay"])
def test_native_carry_500(heads, gate_bias):
    # 129 exercises a second chunk; 65 crosses the inverse split; one empty
    # sequence checks that unused state slots remain untouched.
    lengths = (129, 65, 0)
    data = make_case(lengths, heads=heads, bias=True, seed=17)
    data["gate_bias"].fill_(gate_bias)
    banks = [make_case(lengths, heads=heads, bias=True, seed=seed)
             for seed in (101, 203, 307, 409)]
    lib, actual = native_case(data)
    for _ in range(3):
        lib.mega_kda_prefill(**actual)
    run_mode = os.environ.get("KDA_CARRY500_MODE", "graph")
    assert run_mode in ("graph", "eager"), run_mode
    graph = None
    if run_mode == "graph":
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            lib.mega_kda_prefill(**actual)
    started = time.monotonic()
    for iteration in range(500):
        # Addresses stay fixed for graph replay; input contents change.
        bank = banks[iteration % len(banks)]
        for name in ("qkv", "gate", "beta"):
            data[name].copy_(bank[name])
            actual[name].copy_(data[name])
        snapshots = {name: value.cpu() for name, value in actual.items()
                     if isinstance(value, torch.Tensor) and name not in ("output", "conv_state_out")}
        if graph is None:
            lib.mega_kda_prefill(**actual)
        else:
            graph.replay()
        expected = check(data, actual,
                         f"carry500/H={heads}/gate_bias={gate_bias}/step={iteration + 1}/mode={run_mode}",
                         snapshots)
        # Advance independently: never replace the NPU state by CPU golden.
        for b, length in enumerate(lengths):
            if not length:
                continue
            for index, family in enumerate(("conv", "ssm"), 1):
                slot = int(data[f"{family}_write_indices"][b])
                data[f"{family}_state_in"][b].copy_(expected[index][slot])
                actual[f"{family}_state_in"][b].copy_(actual[OUTPUTS[index]][slot])
                data[f"{family}_read_indices"][b] = b
                # Rotate disjoint write slots without changing graph addresses.
                data[f"{family}_write_indices"][b] = ((iteration + 1) % 2) * 3 + b
        for family in ("conv", "ssm"):
            for mode in ("read", "write"):
                name = f"{family}_{mode}_indices"
                actual[name].copy_(data[name])
        for name, value in zip(OUTPUTS, expected):
            data[name].copy_(value)
        if (iteration + 1) % 25 == 0:
            print(json.dumps(dict(progress=iteration + 1, total=500, heads=heads,
                                  gate_bias=gate_bias,
                                  elapsed_s=time.monotonic() - started)), flush=True)
