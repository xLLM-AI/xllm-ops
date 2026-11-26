# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""RmsNormChannels via the repository custom_ops / ACLNN integration."""
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

# The 25 shapes of the source cann_bench task set, which route through every
# tiling key (SMALL_C / SPLIT_C / STREAM / SINGLE_CHANNEL / SIMT_WARP).
CASES = [
    ([1, 5120, 414, 1, 1], torch.bfloat16),
    ([1, 128, 3312, 1, 1], torch.bfloat16),
    ([1, 6144, 12915, 1, 1], torch.bfloat16),
    ([1, 128, 77490, 1, 1], torch.bfloat16),
    ([1, 128, 25830, 1, 1], torch.bfloat16),
    ([1, 6144, 11808, 1, 1], torch.bfloat16),
    ([1, 6144, 434, 1, 1], torch.bfloat16),
    ([1, 5120, 673, 1, 1], torch.bfloat16),
    ([1, 128, 4038, 1, 1], torch.bfloat16),
    ([1, 128, 1346, 1, 1], torch.bfloat16),
    ([1, 128, 2072, 1, 1], torch.bfloat16),
    ([1, 128, 868, 1, 1], torch.bfloat16),
    ([1, 128, 259, 1, 1], torch.bfloat16),
    ([2, 5120, 3, 4, 5], torch.bfloat16),
    ([3, 256, 5, 7, 13], torch.bfloat16),
    ([1, 128, 7, 9, 11], torch.float16),
    ([1, 5120, 2, 3, 7], torch.float32),
    ([1, 1, 16, 1, 1], torch.bfloat16),
    ([1, 128, 1, 1, 1], torch.bfloat16),
    ([1, 128, 16, 1, 1], torch.float32),
    ([1, 128, 16, 1, 1], torch.float16),
    ([1, 256, 63, 1, 1], torch.bfloat16),
    ([1, 256, 65, 1, 1], torch.bfloat16),
    ([1, 256, 127, 1, 1], torch.bfloat16),
    ([1, 256, 129, 1, 1], torch.bfloat16),
]
TOLERANCES = {
    torch.bfloat16: (2**-6, 2**-6, 1.0),
    torch.float16: (2**-9, 2**-9, 0.1),
    torch.float32: (2**-16, 2**-10, 0.01),
}


def golden(value, gamma, eps):
    """FP64 channel-axis reference, no transpose."""
    x = value.double()
    channels = value.size(1)
    mean_sq = x.pow(2).mean(dim=1, keepdim=True)
    rstd = torch.rsqrt(mean_sq + eps)
    scale_shape = [1, channels] + [1] * (value.dim() - 2)
    return x * rstd * gamma.double().view(scale_shape)


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


def run_case(shape, dtype, eps=1e-12, check_inputs=True):
    generator = torch.Generator().manual_seed(2026)
    value = torch.randn(shape, generator=generator).to(dtype)
    gamma = (torch.rand((shape[1],), generator=generator) + 0.5).to(dtype)
    value_npu, gamma_npu = value.npu(), gamma.npu()
    result = custom_ops.rms_norm_channels_npu(value_npu, gamma_npu, eps=eps)
    assert tuple(result.shape) == tuple(shape) and result.dtype == dtype
    check_result(result, golden(value, gamma, eps))
    if check_inputs:
        assert torch.equal(value.view(torch.uint8), value_npu.cpu().view(torch.uint8))
        assert torch.equal(gamma.view(torch.uint8), gamma_npu.cpu().view(torch.uint8))


@pytest.mark.parametrize("shape,dtype", CASES,
                         ids=[f"{s}-{str(d).replace('torch.', '')}" for s, d in CASES])
def test_source_cases(shape, dtype):
    run_case(shape, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_dtype_matrix(dtype):
    run_case([2, 128, 7, 9, 11], dtype)


@pytest.mark.parametrize("shape", [
    [1, 128, 2048, 1, 1],    # SMALL_C
    [1, 5120, 414, 1, 1],    # SPLIT_C
    [1, 6144, 12915, 1, 1],  # STREAM
    [1, 1, 16, 1, 1],        # SINGLE_CHANNEL
    [1, 128, 16, 1, 1],      # SIMT_WARP
    [1, 256, 129, 1, 1],     # SMALL_C / SPLIT_C boundary
    [1, 512, 160, 1, 1],     # SPLIT_C small-C branch
])
def test_tiling_key_families(shape):
    run_case(shape, torch.bfloat16)


def test_eps_default_matches_golden():
    run_case([1, 128, 259, 1, 1], torch.bfloat16, eps=1e-12, check_inputs=False)


def test_rejects_non_5d_input():
    value_npu = torch.randn((1, 128, 33), dtype=torch.bfloat16).npu()
    gamma_npu = torch.rand((128,), dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        custom_ops.rms_norm_channels_npu(value_npu, gamma_npu)


def test_rejects_gamma_mismatch():
    value_npu = torch.randn((1, 128, 33, 1, 1), dtype=torch.bfloat16).npu()
    gamma_npu = torch.rand((64,), dtype=torch.bfloat16).npu()
    with pytest.raises(RuntimeError):
        custom_ops.rms_norm_channels_npu(value_npu, gamma_npu)
