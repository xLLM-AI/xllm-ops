# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The xLLM Authors.
"""Regression coverage for optional inputs on the ungrouped V2 route."""

import pytest
import torch


torch_npu = pytest.importorskip("torch_npu")


@pytest.fixture(scope="module", autouse=True)
def _prepare_v2_op() -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an NPU")
    if "swiglu_mode" not in str(torch.ops.npu.npu_dequant_swiglu_quant._schemas):
        pytest.skip("requires torch_npu with the V2 SwiGLU attributes")
    torch.npu.set_device(0)
    torch.use_deterministic_algorithms(True)


@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float16, torch.float32, torch.int32])
@pytest.mark.parametrize("quant_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("quant_mode", "quant_rank"),
    [("dynamic", 1), ("static_scalar", 1), ("static_scalar", 2),
     ("static_channel", 1), ("static_channel", 2)],
)
@pytest.mark.parametrize("activate_left", [False, True])
@torch.inference_mode()
def test_ungrouped_int32_optional_buffers(
    bias_dtype: torch.dtype | None,
    quant_dtype: torch.dtype,
    quant_mode: str,
    quant_rank: int,
    activate_left: bool,
) -> None:
    torch.manual_seed(2026)
    rows, width = 3, 128
    x = torch.randint(-16, 17, (rows, 2 * width), dtype=torch.int32)
    weight_scale = torch.linspace(0.125, 0.75, 2 * width)
    activation_scale = torch.tensor([0.25, 0.5, 1.0])
    bias = None
    if bias_dtype is not None:
        bias = ((torch.arange(2 * width) % 7) - 3).to(bias_dtype)
        if bias_dtype != torch.int32:
            bias *= 0.25
    if quant_mode == "static_scalar":
        quant_scale = torch.tensor([2.0], dtype=quant_dtype)
        quant_offset = torch.tensor([3.0], dtype=quant_dtype)
    else:
        quant_scale = (1 + torch.arange(width) % 3).to(quant_dtype)
        quant_offset = ((torch.arange(width) % 5) - 2).to(quant_dtype)
        if quant_mode == "dynamic":
            quant_offset = None

    dequant = x.float()
    if bias_dtype == torch.int32:
        dequant += bias.float()
    dequant = dequant * weight_scale * activation_scale[:, None]
    if bias is not None and bias_dtype != torch.int32:
        dequant += bias.float()
    left, right = dequant.chunk(2, dim=-1)
    gate, up = (left, right) if activate_left else (right, left)
    gate = gate.clamp(max=1.5)
    values = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-1.5, 1.5) + 1.0)
    if quant_mode == "dynamic":
        values *= quant_scale.float()
        expected_scale = values.abs().amax(-1) / 127.0
        expected = (values / expected_scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    else:
        values /= quant_scale.float()
        expected = (values + quant_offset.float()).round().clamp(-128, 127).to(torch.int8)

    def _to_npu(value: torch.Tensor | None) -> torch.Tensor | None:
        return value.npu() if value is not None else None

    if quant_rank == 2:
        quant_scale = quant_scale.unsqueeze(0)
        quant_offset = quant_offset.unsqueeze(0)
    actual, actual_scale = torch_npu.npu_dequant_swiglu_quant(
        x.npu(),
        weight_scale=weight_scale.npu(),
        activation_scale=activation_scale.npu(),
        bias=_to_npu(bias),
        quant_scale=quant_scale.npu(),
        quant_offset=_to_npu(quant_offset),
        group_index=None,
        activate_left=activate_left,
        quant_mode=int(quant_mode == "dynamic"),
        swiglu_mode=1,
        clamp_limit=1.5,
        glu_alpha=1.702,
        glu_bias=1.0,
    )
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    # Static quantization has no defined per-token output scale.
    if quant_mode == "dynamic":
        torch.testing.assert_close(actual_scale.cpu(), expected_scale, rtol=1e-6, atol=1e-8)
