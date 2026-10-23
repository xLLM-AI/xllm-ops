# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# http://www.apache.org/licenses/LICENSE-2.0

"""Run each build in a separate process; select an idle, healthy A5 explicitly."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def _load(path):
    spec = importlib.util.spec_from_file_location("mx_adapter_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dequant(q, scale):
    values = q.cpu().float()
    exponents = scale.cpu().view(torch.uint8).reshape(values.shape[0], -1)
    return values * torch.pow(2.0, exponents.int().repeat_interleave(32, 1) - 127)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=Path)
    parser.add_argument("--device", type=int)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.device is not None:
        import torch_npu

    module = _load(args.module)
    result = {"module": str(args.module), "checks": [], "hashes": {}}
    if args.partial:
        assert not module.available()
        result["checks"].append("aggregate MXFP8 disabled")
    assert module.core_scalar_dtype(module.e8m0_enum) == 37
    assert module.core_scalar_dtype(-1) == -1
    assert module.core_scalar_dtype(255) == -1
    result["checks"].append("core scalar E8M0 and invalid enums")
    for dtype, acl_dtype in [
        (torch.float8_e8m0fnu, 37),
        (torch.float8_e5m2, 35),
        (torch.float8_e4m3fn, 36),
        (torch.float32, 0),
        (torch.bfloat16, 27),
    ]:
        for shape in [(3, 2, 2), (2, 2, 128, 2)]:
            tensor = torch.empty(shape, dtype=dtype)
            assert module.test_descriptor_dtype(tensor) == acl_dtype
    result["checks"].append("test bridge actual ACL descriptors")
    if args.disabled:
        print(json.dumps(result), flush=True)
        module.matmul4(
            torch.empty((2, 128)), torch.empty((2, 2, 2)),
            torch.empty((64, 32)), torch.empty((2, 32, 2)),
            torch.bfloat16, 292, None,
        )
        raise AssertionError("missing matmul capability did not reject the call")
    if args.device is None:
        print(json.dumps(result), flush=True)
        return
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    for shape in [(3, 2, 2), (2, 2, 128, 2)]:
        tensor = torch.empty(shape, dtype=torch.float8_e8m0fnu, device=device)
        assert tensor.element_size() == 1
        assert module.core_descriptor_dtype(tensor) == 37
        assert module.test_descriptor_dtype(tensor) == 37
    result["checks"].append("core/test NPU E8M0 descriptors")

    m, n, k = 3, 32, 128
    x_cpu = ((torch.arange(m * k).reshape(m, k) % 11) - 5).bfloat16() / 2
    packed_cpu = (torch.arange(n * k // 2) % 256).byte().reshape(n, k // 2)
    packed = packed_cpu.to(device)
    scales = torch.full((n, k // 32), 127, dtype=torch.uint8, device=device)
    q, x_scale = module.dynamic(x_cpu.to(device))
    scale_view = scales.reshape(n, k // 64, 2).transpose(0, 1)
    lut = torch.tensor([
        0., .5, 1., 1.5, 2., 3., 4., 6.,
        -0., -.5, -1., -1.5, -2., -3., -4., -6.,
    ])
    decoded = lut[torch.stack((packed_cpu & 15, packed_cpu >> 4), -1).long()]
    decoded = decoded.reshape(n, k)
    reference = _dequant(q, x_scale) @ decoded.T
    for with_bias in (False, True):
        bias_cpu = torch.arange(n).bfloat16() / 2 if with_bias else None
        bias = bias_cpu.to(device) if with_bias else None
        output = module.matmul4(
            q, x_scale, packed.T, scale_view, torch.bfloat16, 292, bias,
        )
        expected = reference + bias_cpu.float() if with_bias else reference
        torch.testing.assert_close(
            output.cpu().float(), expected.bfloat16().float(), rtol=.01, atol=.02,
        )
        result["hashes"][f"w4_bias_{with_bias}"] = hashlib.sha256(
            output.cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    result["checks"].append("W4A8 matmul, with/without bias, CPU oracle")
    expanded = module.prefill(packed)
    unpacked = torch_npu.npu_format_cast(expanded, 2).cpu().float()
    torch.testing.assert_close(unpacked, decoded.T, rtol=0, atol=0)
    output = module.matmul8(
        q, x_scale, expanded, scale_view.contiguous(), None, torch.bfloat16,
    )
    torch.testing.assert_close(
        output.cpu().float(), reference.bfloat16().float(), rtol=.01, atol=.02,
    )
    result["checks"].append("lossless prefill expansion and FP8-cache matmul")
    module.prepack(packed)
    assert torch_npu.get_npu_format(packed) == 29
    first = torch_npu.npu_format_cast(packed, 2).cpu()
    module.prepack(packed)
    torch.testing.assert_close(torch_npu.npu_format_cast(packed, 2).cpu(), first)
    result["checks"].append("NZ prepack idempotence")
    output = module.matmul4(
        q, x_scale, packed.T, scale_view, torch.bfloat16, 292, None,
    )
    torch.testing.assert_close(
        output.cpu().float(), reference.bfloat16().float(), rtol=.01, atol=.02,
    )
    result["checks"].append("NZ W4A8 matmul CPU oracle")
    gate_cpu = ((torch.arange(m * 256).reshape(m, 256) % 17) - 8).float() / 8
    q_gate, gate_scale = module.swiglu4(gate_cpu.bfloat16().to(device), True, 0.)
    left, right = gate_cpu.chunk(2, dim=-1)
    expected_gate = torch.nn.functional.silu(left) * right
    torch.testing.assert_close(
        _dequant(q_gate, gate_scale), expected_gate, rtol=.15, atol=.03,
    )
    result["checks"].append("partial-build SwiGLU CPU oracle")
    if args.grouped:
        gx_cpu = ((torch.arange(32 * k).reshape(32, k) % 7) - 3).bfloat16()
        gq, gs = module.dynamic(gx_cpu.to(device))
        weights_nd = torch.stack([
            torch.full((k, 256), 1 / 16),
            torch.full((k, 256), 2 / 16),
        ]).to(torch.float8_e4m3fn).to(device)
        weights_nz = torch_npu.npu_format_cast(weights_nd, 29)
        assert torch_npu.get_npu_format(weights_nd) == 2
        assert torch_npu.get_npu_format(weights_nz) == 29
        weight_scales = torch.full((2, k // 64, 256, 2), 127, dtype=torch.uint8)
        weight_scales = weight_scales.view(torch.float8_e8m0fnu).to(device)
        expected = torch.cat([
            gx_cpu[:16].float().sum(1) / 16,
            gx_cpu[16:].float().sum(1) / 8,
        ]).reshape(32, 1).expand(32, 128)
        expected = torch.nn.functional.silu(expected) * expected
        for weights in (weights_nd, weights_nz):
            y, sy = module.grouped(
                gq, [weights], [weight_scales],
                gs.view(torch.float8_e8m0fnu),
                torch.tensor([16, 16], dtype=torch.int64, device=device),
            )
            assert sy.dtype == torch.float8_e8m0fnu
            torch.testing.assert_close(
                _dequant(y, sy), expected, rtol=.15, atol=.03,
            )
        result["checks"].append("grouped MX ND and WeightNz E8M0 workspace")
    torch.npu.synchronize()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
