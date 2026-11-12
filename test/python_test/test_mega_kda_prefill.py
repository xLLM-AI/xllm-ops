# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Contract, FP64 oracle and opt-in ACLNN graph replay checks."""
import json
import os
from pathlib import Path
import re
import subprocess

import pytest
import torch

from mega_kda_prefill_reference import OUTPUTS, make_case, reference
from mega_kda_test_utils import GLM_KDA_HEADS, TP_SIZES, bitwise_equal, error_metrics

ROOT = Path(__file__).resolve().parents[2]
NATIVE = pytest.mark.skipif(os.getenv("MEGA_KDA_PREFILL_TEST_NPU") is None,
                           reason="select an exclusive NPU explicitly")


def test_host_policy(tmp_path):
    source = tmp_path / "policy.cpp"
    source.write_text(r'''
#include <cassert>
#include "mega_kda_prefill_policy.h"
int main() {
    using namespace kda_prefill;
    TilingData t{3, 132, 2, 4, 6, 5, 7, 24, 1};
    assert(ValidTiling(t));
    Workspace w(t);
    assert(w.q == 0 && w.k > w.q && w.bytes % 512 == 0);
    assert(w.eye + 16384 * 4 == w.bytes);
    assert(w.wy_k - w.wy_a == t.blocks * 16384 * sizeof(float));
    for (int tp : {1, 2, 4, 8, 16, 32, 64}) {
        t.heads = 64 / tp;
        assert(ValidTiling(t));
    }
    t.heads = 2;
    t.ffts_addr = 0; assert(!ValidTiling(t)); t.ffts_addr = 1;
    t.blocks = 33; assert(!ValidTiling(t)); t.blocks = 24;
    t.tokens = 0; assert(!ValidTiling(t));
    t.tokens = 32768; assert(ValidTiling(t));
    t.tokens = 131073; assert(!ValidTiling(t)); t.tokens = 1;
    t.heads = 129; assert(!ValidTiling(t)); t.heads = 2;
    t.conv_output_slots = -1; assert(!ValidTiling(t));
}
''')
    exe = tmp_path / "policy"
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source),
                    "-I", str(ROOT / "xllm_ops/mega_kda_prefill/op_host"), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_single_entry():
    op = ROOT / "xllm_ops/mega_kda_prefill"
    source = "\n".join(p.read_text() for p in op.rglob("*.cpp"))
    assert source.count("OP_ADD(MegaKdaPrefill)") == 1
    assert source.count("__global__ __aicore__ void mega_kda_prefill(") == 1
    assert "KERNEL_TYPE_MIX_AIC_1_2" in source
    stage_headers = (
        "kda_prefill_kkt_rows.h", "kda_prefill_inverse_fp32.h", "kda_prefill_wy.h",
        "kda_prefill_chunk_h.h", "kda_prefill_chunk_o.h",
    )
    entry = (op / "op_kernel/mega_kda_prefill.cpp").read_text()
    for name in stage_headers:
        path = op / "op_kernel" / name
        assert path.is_file(), name
        assert f'#include "{name}"' in entry, name
        assert "__global__" not in path.read_text()
        assert "call_kernel" not in path.read_text()


@pytest.mark.parametrize("operator", ["mega_kda_prefill", "mega_kda_decode"])
def test_source_filenames_are_descriptive(operator):
    paths = [path for path in (ROOT / "xllm_ops" / operator).rglob("*")
             if path.is_file() and path.suffix in {".h", ".cpp"}]
    assert paths
    for path in paths:
        assert re.search(r"(?:^|_)r\d+(?:_|$)", path.stem, re.IGNORECASE) is None, path.name


@pytest.mark.parametrize("lengths", [(1,), (2, 0, 3), (127, 1), (128,), (129, 0, 3), (257,)])
@pytest.mark.parametrize("bias", [False, True])
def test_oracle_contract(lengths, bias):
    data = make_case(lengths, heads=1, bias=bias)
    before = {k: x.clone() for k, x in data.items() if isinstance(x, torch.Tensor)}
    result = reference(data)
    assert all(torch.isfinite(x).all() for x in result)
    assert [x.dtype for x in result] == [torch.bfloat16, torch.bfloat16, torch.float32]
    for name, value in before.items():
        assert torch.equal(data[name], value), name
    for idx, family in enumerate(("conv", "ssm"), 1):
        writes = {int(data[f"{family}_write_indices"][i]) for i, length in enumerate(lengths) if length}
        for slot in range(len(result[idx])):
            if slot not in writes:
                assert torch.equal(result[idx][slot], data[OUTPUTS[idx]][slot])
    # Zero history is not an inactive sequence.
    assert not torch.equal(result[0][0], data["output"][0])
    for family in ("conv", "ssm"):
        data[f"{family}_write_indices"].fill_(-1)
    discarded = reference(data)
    assert torch.equal(discarded[0], result[0])
    assert torch.equal(discarded[1], data["conv_state_out"])
    assert torch.equal(discarded[2], data["ssm_state_out"])


@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
def test_tp_reference_shape(tp):
    heads = GLM_KDA_HEADS // tp
    data = make_case((3, 0, 1), heads=heads, bias=True)
    result = reference(data)
    assert data["qkv"].shape == (4, heads * 384)
    assert data["gate"].shape == (4, heads, 128)
    for name, value in zip(OUTPUTS, result):
        assert value.shape == data[name].shape and value.dtype == data[name].dtype
        assert torch.isfinite(value).all(), name
    assert result[2].shape[1:] == (heads, 128, 128)


@pytest.mark.parametrize("invalid", ["offsets", "read", "write", "duplicate", "alias"])
def test_invalid_metadata(invalid):
    data = make_case((2, 3), heads=1)
    if invalid == "offsets": data["cu_seqlens"][-1] += 1
    elif invalid == "read": data["ssm_read_indices"][0] = -2
    elif invalid == "write": data["conv_write_indices"][0] = 999
    elif invalid == "duplicate": data["ssm_write_indices"].fill_(0)
    else: data["conv_state_out"] = data["conv_state_in"]
    with pytest.raises(ValueError):
        reference(data)


def native_case(data):
    import torch_npu  # noqa: F401
    import custom_ops_lib
    device = int(os.environ["MEGA_KDA_PREFILL_TEST_NPU"])
    torch.npu.set_device(device)
    actual = {k: v.to(f"npu:{device}") if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    return custom_ops_lib, actual


def check(data, actual, label, input_snapshots=None):
    expected = reference(data)
    records = {}
    for name, want in zip(OUTPUTS, expected):
        got = actual[name].cpu()
        assert got.shape == want.shape and got.dtype == want.dtype, (label, name)
        assert torch.isfinite(got).all() and torch.isfinite(want).all(), (label, name)
        if name == "conv_state_out":
            assert bitwise_equal(got, want), name
            continue
        active = want != 37
        stats = error_metrics(got[active], want[active], near=0.001, absolute=0.00001)
        records[name] = stats
        assert stats["finite"], (label, name, stats)
        assert stats["normal_over_fraction"] <= 0.001, (label, name, stats)
        assert stats["near_over_fraction"] <= 0.001, (label, name, stats)
        untouched = want == 37
        assert bitwise_equal(got[untouched], want[untouched]), name
    for k, x in (data if input_snapshots is None else input_snapshots).items():
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
@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("tokens", [1024, 2048, 4096, 8192, 16384, 32768])
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
    # Long TP1 replay catches delayed corruption when chunk state UB is reused.
    replays = 1000 if tp == 1 and tokens == 2048 else 20
    for iteration in range(replays):
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
@pytest.mark.parametrize("tp", TP_SIZES, ids=lambda tp: f"tp{tp}")
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
        snapshots = {k: v.cpu() for k, v in actual.items() if isinstance(v, torch.Tensor) and k not in OUTPUTS}
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
