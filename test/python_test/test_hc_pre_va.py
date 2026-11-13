# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The xLLM Authors.
"""Both the implicit and external-pre-mix cases exercise the four-output hc_pre_va API."""

import ctypes
import gc
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import pytest

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


@pytest.fixture(scope="module", autouse=True)
def _prepare_custom_op() -> None:
    if not torch_npu.npu.is_available():
        pytest.skip("An Ascend NPU is required")
    if torch_npu._C._npu_get_soc_version() != 253:
        pytest.skip("hc_pre_va tests require an Ascend A3")
    torch_npu.npu.config.allow_internal_format = True


HC_MULT = 4
DSV4_FLASH_HIDDEN_SIZE = 4096
HIDDEN_SIZE = 5120  # DeepSeek-V4.1-Flash checkpoint hidden size
EXTENDED_HIDDEN_SIZE = 7168
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995


def _make_hc_pre_inputs(shape: tuple[int, ...]):
    torch.manual_seed(1024)
    hidden_size = shape[-1]
    fan_in = HC_MULT * hidden_size
    x = (torch.rand(shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = (
        torch.rand(
            MIX_HC,
            fan_in,
            dtype=torch.float32,
        )
        / fan_in
    )
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return x, hc_fn, hc_scale, hc_base


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    dropped_mantissa_bits = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mantissa_mask = ~((1 << dropped_mantissa_bits) - 1)
    bits = tensor.contiguous().view(torch.int32)
    return (bits & mantissa_mask).view(torch.float32)


def _hc_pre_cpu(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)

    # HcPre performs this matmul in HF32 round-toward-zero mode.
    mixes = F.linear(_to_hf32(x_flat), _to_hf32(hc_fn)) * inv_rms
    pre, post, comb_frag = mixes.split(
        [HC_MULT, HC_MULT, HC_MULT * HC_MULT],
        dim=-1,
    )
    comb_frag = comb_frag.unflatten(-1, (HC_MULT, HC_MULT))

    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb_frag = comb_frag * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)

    comb_frag = comb_frag.softmax(-1) + HC_EPS
    comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        comb_frag = comb_frag / (comb_frag.sum(-1, keepdim=True) + HC_EPS)
        comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)

    mix_for_y = pre_mix.float() if pre_mix is not None else pre
    y = (mix_for_y.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, post, comb_frag, pre


def _assert_close_with_pass_rate(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    diff_threshold: float,
    required_pass_rate: float,
):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny)
        <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    assert pass_rate >= required_pass_rate, (
        f"pass rate {pass_rate:.2%} is below {required_pass_rate:.2%}; max absolute difference: {max_abs_diff}"
    )


def _compare_hc_pre_with_cpu(shape: tuple[int, ...]):
    x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    expected_y, expected_post, expected_comb_frag, expected_pre = _hc_pre_cpu(
        x,
        hc_fn,
        hc_scale,
        hc_base,
    )
    y, post, comb_frag, pre = custom_ops.hc_pre_va_npu(
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        None,
        HC_MULT,
        HC_SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
    )

    batch_shape = shape[:-2]
    assert y.shape == (*batch_shape, shape[-1])
    assert post.shape == (*batch_shape, HC_MULT)
    assert comb_frag.shape == (*batch_shape, HC_MULT, HC_MULT)
    assert y.dtype == torch.bfloat16
    assert post.dtype == torch.float32
    assert comb_frag.dtype == torch.float32
    _assert_close_with_pass_rate(
        y,
        expected_y,
        diff_threshold=Y_DIFF_THRESHOLD,
        required_pass_rate=Y_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        post,
        expected_post,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )
    _assert_close_with_pass_rate(
        comb_frag,
        expected_comb_frag,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )

    assert pre.shape == (*batch_shape, HC_MULT)
    assert pre.dtype == torch.float32
    _assert_close_with_pass_rate(
        pre,
        expected_pre,
        diff_threshold=AUX_DIFF_THRESHOLD,
        required_pass_rate=AUX_REQUIRED_PASS_RATE,
    )


@torch.inference_mode()
def test_hc_pre_va_without_external_mix_bf16_3d_input():
    _compare_hc_pre_with_cpu((2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_hc_pre_va_without_external_mix_bf16_4d_input():
    _compare_hc_pre_with_cpu((1, 2, HC_MULT, HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_hc_pre_va_without_external_mix_bf16_dsv4_flash_hidden_size():
    _compare_hc_pre_with_cpu((4, HC_MULT, DSV4_FLASH_HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
def test_hc_pre_va_without_external_mix_bf16_extended_hidden_size():
    _compare_hc_pre_with_cpu((2, HC_MULT, EXTENDED_HIDDEN_SIZE))
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@torch.inference_mode()
@pytest.mark.parametrize(
    "shape",
    [
        (1, HC_MULT, 5120),
        (6, HC_MULT, 5120),
        (33, HC_MULT, 5120),
        (1, 6, HC_MULT, 5120),
    ],
)
def test_hc_pre_va_uses_external_pre_mix(shape):
    x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(shape)
    pre_mix = torch.rand(shape[:-1], dtype=torch.float32)
    expected_y, expected_post, expected_comb_frag, expected_pre = _hc_pre_cpu(
        x, hc_fn, hc_scale, hc_base, pre_mix
    )
    y, post, comb_frag, pre = custom_ops.hc_pre_va_npu(
        x.npu(),
        hc_fn.npu(),
        hc_scale.npu(),
        hc_base.npu(),
        pre_mix.npu(),
        hc_mult=HC_MULT,
        hc_sinkhorn_iters=HC_SINKHORN_ITERS,
        norm_eps=NORM_EPS,
        hc_eps=HC_EPS,
    )
    assert y.shape == (*shape[:-2], shape[-1])
    assert post.shape == (*shape[:-2], HC_MULT)
    assert comb_frag.shape == (*shape[:-2], HC_MULT, HC_MULT)
    assert pre.shape == (*shape[:-2], HC_MULT)
    assert y.dtype == torch.bfloat16
    assert post.dtype == torch.float32
    assert comb_frag.dtype == torch.float32
    assert pre.dtype == torch.float32
    for actual, expected, diff_threshold, required_pass_rate in (
        (y, expected_y, Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE),
        (post, expected_post, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
        (comb_frag, expected_comb_frag, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
        (pre, expected_pre, AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
    ):
        _assert_close_with_pass_rate(
            actual,
            expected,
            diff_threshold=diff_threshold,
            required_pass_rate=required_pass_rate,
        )
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


@pytest.mark.parametrize(
    "tokens,width",
    [
        (1, 5120),
        (6, 5120),
        (33, 5120),
        (129, 128),
        (1025, 128),
        (129, 127),
        (129, 129),
        (8193, 32),
    ],
)
@pytest.mark.parametrize("external_mix", [False, True])
@torch.inference_mode()
def test_hc_pre_va_graph_replay_and_determinism(
    tokens: int, width: int, external_mix: bool
) -> None:
    x, fn, scale, base = (
        value.npu() for value in _make_hc_pre_inputs((tokens, HC_MULT, width))
    )
    mix = torch.rand(tokens, HC_MULT, device=x.device) if external_mix else None

    def execute() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return custom_ops.hc_pre_va_npu(
            x, fn, scale, base, mix, HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS
        )

    expected = execute()
    repeated = execute()
    for actual, reference in zip(repeated, expected):
        assert torch.equal(actual, reference)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = execute()
    graph.replay()
    torch.npu.synchronize()
    for actual, reference in zip(captured, expected):
        assert torch.equal(actual, reference)
    x.mul_(0.75)
    if mix is not None:
        mix.mul_(0.5)
    updated = execute()
    graph.replay()
    torch.npu.synchronize()
    for actual, reference in zip(captured, updated):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("external_mix", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (129, HC_MULT, 128),
        (1025, HC_MULT, 128),
        (4097, HC_MULT, 128),
        (8193, HC_MULT, 32),
        (129, HC_MULT, 127),
        (129, HC_MULT, 129),
        (17, HC_MULT, 7),
        (17, HC_MULT, 17),
        (2, 65, HC_MULT, 128),
        (129, HC_MULT, HIDDEN_SIZE),
    ],
)
@torch.inference_mode()
def test_hc_pre_va_multiline_and_tail_tiles(
    shape: tuple[int, ...], external_mix: bool
) -> None:
    x, fn, scale, base = _make_hc_pre_inputs(shape)
    # Different signed magnitudes expose incorrect per-row RMS and gather phases.
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x.mul_(
        torch.linspace(0.125, 2, x.numel() // (HC_MULT * shape[-1])).reshape(
            *shape[:-2], 1, 1
        )
    )
    fn.normal_().div_((HC_MULT * shape[-1]) ** 0.5)
    # Exact HF32 operands isolate layout/RMS errors from conversion rounding.
    fn = fn.to(torch.bfloat16).float()
    mix = torch.rand(shape[:-1]) if external_mix else None
    expected = _hc_pre_cpu(x, fn, scale, base, mix)
    inputs = tuple(value.npu() for value in (x, fn, scale, base))
    actual = custom_ops.hc_pre_va_npu(
        *inputs, pre_mix=mix.npu() if mix is not None else None
    )
    for index, (value, reference) in enumerate(zip(actual, expected)):
        assert torch.isfinite(value).all()
        _assert_close_with_pass_rate(
            value,
            reference,
            diff_threshold=Y_DIFF_THRESHOLD if index == 0 else AUX_DIFF_THRESHOLD,
            required_pass_rate=Y_REQUIRED_PASS_RATE
            if index == 0
            else AUX_REQUIRED_PASS_RATE,
        )
    if mix is None:
        explicit = custom_ops.hc_pre_va_npu(*inputs, pre_mix=actual[3])
        for value, reference in zip(explicit, actual):
            assert torch.equal(value, reference)


@pytest.mark.parametrize("hidden_size", [128, HIDDEN_SIZE])
@pytest.mark.parametrize("external_mix", [False, True])
@pytest.mark.parametrize("base_mode", ["zero", "normal", "small"])
@pytest.mark.parametrize("iterations", [1, 2, 4, 20])
@torch.inference_mode()
def test_hc_pre_va_identical_rows_match_single_token(
    hidden_size: int, external_mix: bool, base_mode: str, iterations: int
) -> None:
    torch.manual_seed(101)
    hidden = torch.randn(1, HC_MULT, hidden_size).to(torch.bfloat16).npu()
    # Remove GEMM and RMS rounding from the coefficients under test.
    fn = torch.zeros(MIX_HC, HC_MULT * hidden_size, device=hidden.device)
    scale = torch.zeros(3, device=hidden.device)
    base = torch.zeros(MIX_HC) if base_mode == "zero" else torch.randn(MIX_HC)
    if base_mode == "small":
        base *= 0.01
    base = base.npu()
    mix = torch.rand(1, HC_MULT, device=hidden.device) if external_mix else None
    single = tuple(
        value.cpu()
        for value in custom_ops.hc_pre_va_npu(
            hidden, fn, scale, base, mix, HC_MULT, iterations, NORM_EPS, HC_EPS
        )
    )

    def _comb_reference(tokens: int) -> torch.Tensor:
        def _sum_four(value: torch.Tensor, dim: int) -> torch.Tensor:
            first, second, third, fourth = value.unbind(dim)
            if dim == -1 and tokens <= 32:
                return ((first + second) + (third + fourth)).unsqueeze(dim)
            return (((first + second) + third) + fourth).unsqueeze(dim)

        logits = base[2 * HC_MULT :].view(1, HC_MULT, HC_MULT)
        expected = (logits - logits.amax(-1, keepdim=True)).exp()
        expected = expected / _sum_four(expected, -1) + HC_EPS
        expected = expected / (_sum_four(expected, -2) + HC_EPS)
        for _ in range(iterations - 1):
            expected = expected / (_sum_four(expected, -1) + HC_EPS)
            expected = expected / (_sum_four(expected, -2) + HC_EPS)
        return expected.cpu()

    # Cover one-row calls, multi-row tiles, full tiles and partial final tiles.
    # Small widths permit multi-row stage-2 tiles; 5120 covers the checkpoint.
    token_counts = (1, 3, 33, 65, 127, 128, 129, 4097)
    if hidden_size == 128 and iterations == HC_SINKHORN_ITERS:
        # Stage 2 permits at most 63 rows/tile. This exceeds even 96 cores
        # times that limit, forcing repeated gather-mask resets on each core.
        token_counts += (65537,)
    for tokens in token_counts:
        actual = custom_ops.hc_pre_va_npu(
            hidden.repeat(tokens, 1, 1),
            fn,
            scale,
            base,
            mix.repeat(tokens, 1) if mix is not None else None,
            HC_MULT,
            iterations,
            NORM_EPS,
            HC_EPS,
        )
        for name, value, reference in zip(("y", "post", "comb", "pre"), actual, single):
            if name == "comb":
                reference = _comb_reference(tokens)
            assert torch.equal(value.cpu(), reference.expand_as(value)), (
                f"{name} changes with row position or partition: "
                f"tokens={tokens}, hidden={hidden_size}, iterations={iterations}"
            )


@pytest.mark.parametrize("hidden_size", [128, HIDDEN_SIZE])
@pytest.mark.parametrize("external_mix", [False, True])
@torch.inference_mode()
def test_hc_pre_va_identical_coefficients_after_graph_input_update(
    hidden_size: int, external_mix: bool
) -> None:
    torch.manual_seed(101)
    tokens = 129
    hidden = torch.randn(1, HC_MULT, hidden_size).to(torch.bfloat16).npu()
    hidden = hidden.repeat(tokens, 1, 1)
    fn = torch.zeros(MIX_HC, HC_MULT * hidden_size, device=hidden.device)
    scale = torch.zeros(3, device=hidden.device)
    base = torch.randn(MIX_HC, device=hidden.device)
    mix = torch.rand(1, HC_MULT, device=hidden.device) if external_mix else None
    mix = mix.repeat(tokens, 1) if mix is not None else None

    def execute() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return custom_ops.hc_pre_va_npu(hidden, fn, scale, base, mix)

    execute()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = execute()
    for factor in (1.0, 0.25):
        base.mul_(factor)
        graph.replay()
        fresh = execute()
        torch.npu.synchronize()
        for value, reference in zip(captured, fresh):
            assert torch.equal(value, reference)
            assert torch.equal(value, value[:1].expand_as(value))


@pytest.mark.parametrize(
    "invalid",
    [
        {"hc_sinkhorn_iters": 0},
        {"hc_sinkhorn_iters": -1},
        {"norm_eps": 0.0},
        {"norm_eps": -1.0},
        {"norm_eps": float("nan")},
        {"norm_eps": float("inf")},
        {"hc_eps": 0.0},
        {"hc_eps": -1.0},
        {"hc_eps": float("nan")},
        {"hc_eps": float("inf")},
    ],
)
def test_hc_pre_va_rejects_invalid_attributes(invalid: dict[str, object]) -> None:
    hidden, fn, scale, base = _make_hc_pre_inputs((2, HC_MULT, 128))
    arguments = dict(
        zip(
            ("hidden", "hc_fn", "hc_scale", "hc_base"),
            (v.npu() for v in (hidden, fn, scale, base)),
        )
    )
    with pytest.raises(RuntimeError, match="iterations and finite epsilon"):
        custom_ops.hc_pre_va_npu(**arguments, **invalid)


def _workspace_status(
    shape: tuple[int, ...],
    hc_mult: int = 4,
    iterations: int = 20,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
) -> int:
    vendor_paths = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
    libraries = [
        Path(path) / "op_api/lib/libcust_opapi.so" for path in vendor_paths if path
    ]
    library = next((path for path in libraries if path.is_file()), None)
    if library is None:
        raise RuntimeError(
            "ASCEND_CUSTOM_OPP_PATH must select the isolated test operator package"
        )
    metadata = ctypes.CDLL("libopapi.so")
    operation = ctypes.CDLL(str(library))
    pointer = ctypes.c_void_p
    dimensions_pointer = ctypes.POINTER(ctypes.c_int64)
    metadata.aclCreateTensor.argtypes = [
        dimensions_pointer,
        ctypes.c_uint64,
        ctypes.c_int,
        dimensions_pointer,
        ctypes.c_int64,
        ctypes.c_int,
        dimensions_pointer,
        ctypes.c_uint64,
        pointer,
    ]
    metadata.aclCreateTensor.restype = pointer
    metadata.aclDestroyTensor.argtypes = [pointer]
    metadata.aclDestroyTensor.restype = ctypes.c_int
    workspace_size = operation.aclnnHcPreVaGetWorkspaceSize
    workspace_size.argtypes = (
        [pointer] * 5
        + [ctypes.c_int64] * 2
        + [ctypes.c_double] * 2
        + [pointer] * 4
        + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(pointer)]
    )
    workspace_size.restype = ctypes.c_int
    tensors = (
        torch.empty(shape, dtype=torch.bfloat16, device="npu"),
        torch.empty((24, shape[-2] * shape[-1]), dtype=torch.float32, device="npu"),
        torch.empty(3, dtype=torch.float32, device="npu"),
        torch.empty(24, dtype=torch.float32, device="npu"),
        torch.empty((*shape[:-2], shape[-1]), dtype=torch.bfloat16, device="npu"),
        torch.empty((*shape[:-2], HC_MULT), dtype=torch.float32, device="npu"),
        torch.empty((*shape[:-2], HC_MULT, HC_MULT), dtype=torch.float32, device="npu"),
        torch.empty((*shape[:-2], HC_MULT), dtype=torch.float32, device="npu"),
    )
    descriptors = []
    try:
        for tensor in tensors:
            dimensions = (ctypes.c_int64 * tensor.dim())(*tensor.shape)
            strides = (ctypes.c_int64 * tensor.dim())(*tensor.stride())
            descriptor = metadata.aclCreateTensor(
                dimensions,
                tensor.dim(),
                27 if tensor.dtype == torch.bfloat16 else 0,
                strides,
                tensor.storage_offset(),
                2,
                dimensions,
                tensor.dim(),
                tensor.data_ptr(),
            )
            assert descriptor is not None
            descriptors.append(descriptor)
        size = ctypes.c_uint64()
        executor = pointer()
        return workspace_size(
            *descriptors[:4],
            None,
            hc_mult,
            iterations,
            hc_eps,
            norm_eps,
            *descriptors[4:],
            ctypes.byref(size),
            ctypes.byref(executor),
        )
    finally:
        for descriptor in descriptors:
            metadata.aclDestroyTensor(descriptor)


@pytest.mark.parametrize(
    "invalid",
    (
        {"shape": (2, 3, 128)},
        {"shape": (2, 4, 128), "hc_mult": 3},
        {"shape": (2, 4, 128), "iterations": 0},
        {"shape": (2, 4, 128), "iterations": -1},
        {"shape": (2, 4, 128), "norm_eps": 0.0},
        {"shape": (2, 4, 128), "hc_eps": float("nan")},
    ),
)
def test_hc_pre_va_direct_aclnn_rejects_invalid_attributes(
    invalid: dict[str, object],
) -> None:
    assert _workspace_status(**invalid) != 0


@pytest.mark.parametrize("shape", ((2, 4, 128), (1, 2, 4, 128)))
def test_hc_pre_va_direct_aclnn_accepts_valid_attributes(
    shape: tuple[int, ...],
) -> None:
    assert _workspace_status(shape) == 0


@pytest.mark.parametrize("soc", [220, 250])
def test_hc_pre_va_skips_non_a3(soc: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch_npu.npu, "is_available", lambda: True)
    monkeypatch.setattr(torch_npu._C, "_npu_get_soc_version", lambda: soc)
    with pytest.raises(pytest.skip.Exception, match="Ascend A3"):
        _prepare_custom_op.__wrapped__()


@pytest.mark.parametrize("shape", [(129, HC_MULT, 128), (1024, HC_MULT, 4096)])
@torch.inference_mode()
def test_hc_pre_va_pre_output_buffer_reuse(shape: tuple[int, ...]) -> None:
    hidden, fn, scale, base = _make_hc_pre_inputs(shape)
    hidden.normal_()
    fn.normal_().div_((HC_MULT * shape[-1]) ** 0.5)
    arguments = tuple(value.npu() for value in (hidden, fn, scale, base))
    baseline = None
    for _ in range(5):
        outputs = custom_ops.hc_pre_va_npu(*arguments, hc_sinkhorn_iters=1)
        torch.npu.synchronize()
        actual = tuple(value.cpu() for value in outputs)
        assert all(torch.isfinite(value).all() for value in actual)
        if baseline is not None:
            for value, reference in zip(actual, baseline):
                assert torch.equal(value, reference)
        baseline = actual


@pytest.mark.parametrize(
    "tokens,hidden_size", [(1, 128), (3, 128), (32, 128), (33, 128), (129, 128), (3, 4096)]
)
@torch.inference_mode()
def test_hc_pre_va_sinkhorn_preserves_reference_row_sums(
    tokens: int, hidden_size: int
) -> None:
    """Check the FP32 addition order before BF16 residual recombination."""
    torch.manual_seed(27)
    hidden = torch.randn(tokens, HC_MULT, hidden_size).bfloat16().npu()
    fn = torch.zeros(MIX_HC, HC_MULT * hidden_size, device=hidden.device)
    scale = torch.zeros(3, device=hidden.device)
    # No projection/RMS rounding enters the matrix under test.
    base = (torch.randn(MIX_HC) * 2).npu()
    _, _, actual, _ = custom_ops.hc_pre_va_npu(
        hidden,
        fn,
        scale,
        base,
        None,
        HC_MULT,
        HC_SINKHORN_ITERS,
        NORM_EPS,
        HC_EPS,
    )

    def _sum_four(value: torch.Tensor, dim: int) -> torch.Tensor:
        first, second, third, fourth = value.unbind(dim)
        if dim == -1 and tokens <= 32:
            return ((first + second) + (third + fourth)).unsqueeze(dim)
        return (((first + second) + third) + fourth).unsqueeze(dim)

    logits = base[2 * HC_MULT :].view(1, HC_MULT, HC_MULT)
    expected = (logits - logits.amax(-1, keepdim=True)).exp()
    expected = expected / _sum_four(expected, -1) + HC_EPS
    expected = expected / (_sum_four(expected, -2) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        expected = expected / (_sum_four(expected, -1) + HC_EPS)
        expected = expected / (_sum_four(expected, -2) + HC_EPS)
    assert torch.equal(actual.cpu(), expected.expand_as(actual).cpu())
