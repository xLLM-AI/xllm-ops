# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""AdaRmsNorm via the repository custom_ops / ACLNN integration."""
import itertools

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

PROFILE_SHAPES = [
    (259, 128), (414, 128), (868, 128), (1346, 128), (2072, 128),
    (2604, 128), (3312, 128), (4038, 128), (23616, 128), (25830, 128),
    (70848, 128), (77490, 128), (259, 5120), (414, 5120), (673, 5120),
    (1, 434, 6144), (1, 673, 6144), (1, 11808, 6144), (1, 12915, 6144),
    (434, 6144), (11808, 6144),
]
TOLERANCES = {
    torch.bfloat16: (2**-6, 2**-6, 1.0),
    torch.float16: (2**-9, 2**-9, 0.1),
    torch.float32: (2**-16, 2**-10, 0.01),
}


def golden(x, scale, shift, weight, eps):
    x = x.double()
    scale, shift = scale.double(), shift.double()
    if x.ndim == 3:
        if scale.ndim == 2:
            scale = scale.unsqueeze(1)
        if shift.ndim == 2:
            shift = shift.unsqueeze(1)
    out = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.double()
    return out * (1.0 + scale) + shift


def check_result(actual, expected):
    dtype = actual.dtype
    actual = actual.cpu().double()
    typed = expected.to(dtype).double()
    for mask in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(mask(actual), mask(typed))
    finite = torch.isfinite(actual) & torch.isfinite(typed)
    errors = (actual[finite] - expected[finite]).abs()
    if errors.numel():
        atol, rtol, maximum = TOLERANCES[dtype]
        assert ((errors <= atol + rtol * expected[finite].abs()).double().mean() >= 0.99).item()
        assert errors.max().item() <= maximum


def run_case(shape, dtype, scale_mode="D", shift_mode="D", has_weight=False, eps=1e-6):
    generator = torch.Generator().manual_seed(2026)
    d = shape[-1]

    def make(shape):
        return torch.randn(shape, generator=generator).to(dtype)

    def parameter(mode):
        if mode == "D":
            return make((d,))
        if mode == "full":
            return make(shape)
        if mode == "BD":
            return make((shape[0], d))
        return make((1, shape[1], d))

    cpu = [make(shape), parameter(scale_mode), parameter(shift_mode), make((d,)) if has_weight else None]
    inputs = [tensor.npu() if tensor is not None else None for tensor in cpu]
    result = custom_ops.ada_rms_norm_npu(*inputs, eps=eps)
    assert tuple(result.shape) == tuple(shape) and result.dtype == dtype
    check_result(result, golden(*cpu, eps))
    for original, tensor in zip(cpu, inputs):
        if original is not None:
            assert torch.equal(original.view(torch.uint8), tensor.cpu().view(torch.uint8))


@pytest.mark.parametrize("shape", PROFILE_SHAPES)
def test_profile_shapes(shape):
    run_case(shape, torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("d", [1, 17, 33, 127, 128, 129, 5119, 6144, 8192])
def test_dtype_and_tails(dtype, d):
    run_case((67, d), dtype, shift_mode="full", has_weight=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 3, 33), (2, 7, 6144)])
@pytest.mark.parametrize("scale,shift,weight", list(itertools.product(("D", "BD", "full"), ("D", "1SD", "full"), (False, True))))
def test_ada_broadcast(dtype, shape, scale, shift, weight):
    run_case(shape, dtype, scale, shift, weight)


@pytest.mark.parametrize("shape", [(0, 33), (0, 5, 33), (3, 0, 33)])
def test_empty(shape):
    run_case(shape, torch.bfloat16)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf"), 1e-300, 1e300])
def test_invalid_epsilon(eps):
    x = torch.ones((3, 33), device="npu", dtype=torch.bfloat16)
    parameter = torch.ones(33, device="npu", dtype=x.dtype)
    with pytest.raises(RuntimeError, match="eps"):
        custom_ops.ada_rms_norm_npu(x, parameter, parameter, eps=eps)


def test_cpp_binding_matches_python_wrapper():
    import custom_ops_lib
    x = torch.randn((3, 7, 6144), device="npu", dtype=torch.bfloat16)
    scale = torch.randn((3, 6144), device="npu", dtype=x.dtype)
    shift = torch.randn((6144,), device="npu", dtype=x.dtype)
    torch.testing.assert_close(custom_ops_lib.ada_rms_norm(x, scale, shift),
                               custom_ops.ada_rms_norm_npu(x, scale, shift), rtol=0, atol=0)
