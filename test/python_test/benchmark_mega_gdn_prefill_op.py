import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu

from test_mega_gdn_prefill_op import (
    _make_constants,
    _make_cpu_inputs,
    _run_baseline,
    _run_e2e,
    _to_device,
)


CASES = {
    "batch2-equal": ((256, 256), 24, 8, 1),
    "batch4-ragged": ((129, 257, 63, 511), 12, 4, 2),
    "batch8-ragged": (
        (17, 33, 65, 127, 128, 129, 255, 257),
        8,
        4,
        5,
    ),
}


def _make_runner(case, state_mode, use_fused, device):
    seq_lens, value_heads, key_heads, checkpoint_stride = case
    cpu_inputs = _make_cpu_inputs(
        value_heads,
        key_heads,
        checkpoint_stride=checkpoint_stride,
        state_mode=state_mode,
        seq_lens=seq_lens,
    )
    tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(
        device,
        tokens=sum(seq_lens),
        seq_lens=seq_lens,
    )
    conv_read_slots = cpu_inputs["conv_state_read_indices"].tolist()
    conv_write_slots = cpu_inputs["conv_state_write_indices"].tolist()

    if use_fused:
        return lambda: _run_e2e(tensors, constants)

    return lambda: _run_baseline(
        tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
        checkpoint_stride=checkpoint_stride,
        conv_read_slots=conv_read_slots,
        conv_write_slots=conv_write_slots,
    )


def _measure_round(runner, iterations):
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        runner()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / iterations
    device_ms = start.elapsed_time(end) / iterations
    return {"device_ms": device_ms, "wall_ms": wall_ms}


def _summarize(samples):
    return {
        "device_ms_median": statistics.median(
            sample["device_ms"] for sample in samples
        ),
        "device_ms_min": min(sample["device_ms"] for sample in samples),
        "wall_ms_median": statistics.median(
            sample["wall_ms"] for sample in samples
        ),
        "wall_ms_min": min(sample["wall_ms"] for sample in samples),
        "rounds": samples,
    }


def _speedup(baseline_ms, fused_ms):
    return (baseline_ms / fused_ms - 1.0) * 100.0


def benchmark_case(name, case, state_mode, warmup, iterations, rounds, device):
    baseline = _make_runner(case, state_mode, False, device)
    fused = _make_runner(case, state_mode, True, device)
    for _ in range(warmup):
        baseline()
        fused()
    torch.npu.synchronize()

    samples = {"baseline": [], "fused": []}
    for round_index in range(rounds):
        order = ("baseline", "fused")
        if round_index & 1:
            order = tuple(reversed(order))
        for implementation in order:
            runner = baseline if implementation == "baseline" else fused
            samples[implementation].append(_measure_round(runner, iterations))

    baseline_summary = _summarize(samples["baseline"])
    fused_summary = _summarize(samples["fused"])
    return {
        "case": name,
        "state_mode": state_mode,
        "seq_lens": list(case[0]),
        "value_heads": case[1],
        "key_heads": case[2],
        "checkpoint_stride": case[3],
        "baseline": baseline_summary,
        "fused": fused_summary,
        "device_speedup_percent": _speedup(
            baseline_summary["device_ms_median"],
            fused_summary["device_ms_median"],
        ),
        "wall_speedup_percent": _speedup(
            baseline_summary["wall_ms_median"],
            fused_summary["wall_ms_median"],
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=list(CASES)
    )
    parser.add_argument(
        "--state-modes",
        nargs="+",
        choices=("no_initial", "inplace", "out_of_place"),
        default=("no_initial", "inplace"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(f"npu:{args.device}")
    torch_npu.npu.set_device(device)
    results = []
    with torch.no_grad():
        for case_name in args.cases:
            for state_mode in args.state_modes:
                result = benchmark_case(
                    case_name,
                    CASES[case_name],
                    state_mode,
                    args.warmup,
                    args.iterations,
                    args.rounds,
                    device,
                )
                results.append(result)
                print(
                    f"{case_name}/{state_mode}: "
                    f"device {result['baseline']['device_ms_median']:.3f} -> "
                    f"{result['fused']['device_ms_median']:.3f} ms "
                    f"({result['device_speedup_percent']:+.2f}%), "
                    f"wall {result['baseline']['wall_ms_median']:.3f} -> "
                    f"{result['fused']['wall_ms_median']:.3f} ms "
                    f"({result['wall_speedup_percent']:+.2f}%)"
                )

    payload = {
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
