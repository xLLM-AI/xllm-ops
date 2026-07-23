# Copyright 2026 The xLLM Authors. All Rights Reserved.

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops_lib")


HEAD_DIM = 128
LOCAL_HEAD_SHAPES = [
    (num_k_heads, num_k_heads * ratio)
    for ratio in range(1, 5)
    for num_k_heads in (1, 2, 4, 8, 16)
]


def _has_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _run_case(num_k_heads, num_v_heads, batch_size):
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    bf16_options = {"device": "npu:0", "dtype": torch.bfloat16}
    float_options = {"device": "npu:0", "dtype": torch.float32}

    qkv = torch.ones((batch_size, conv_dim), **bf16_options)
    z = torch.zeros((batch_size, num_v_heads, HEAD_DIM), **bf16_options)
    b = torch.zeros((batch_size, num_v_heads), **bf16_options)
    a = torch.zeros((batch_size, num_v_heads), **bf16_options)
    conv_weight = torch.zeros((4, conv_dim), **bf16_options)
    conv_state = torch.zeros((batch_size, 3, conv_dim), **bf16_options)
    a_log = torch.zeros((num_v_heads,), **float_options)
    dt_bias = torch.zeros((num_v_heads,), **float_options)
    ssm_state = torch.ones(
        (batch_size, num_v_heads, HEAD_DIM, HEAD_DIM), **float_options
    )
    state_indices = torch.arange(batch_size, device="npu:0", dtype=torch.int32)
    norm_weight = torch.ones((HEAD_DIM,), **bf16_options)

    outputs = custom_ops.qwen35_gdn_decode_super_op(
        qkv,
        z,
        b,
        a,
        conv_weight,
        conv_state,
        a_log,
        dt_bias,
        ssm_state,
        state_indices,
        norm_weight,
    )
    torch.npu.synchronize()

    assert torch.equal(outputs[3], torch.zeros_like(outputs[3]))
    assert torch.equal(conv_state[:, :2], torch.zeros_like(conv_state[:, :2]))
    assert torch.equal(conv_state[:, 2], qkv)
    assert torch.allclose(
        ssm_state,
        torch.full_like(ssm_state, 0.5),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("num_k_heads", "num_v_heads"),
    LOCAL_HEAD_SHAPES,
)
def test_all_qwen35_local_head_shapes(num_k_heads, num_v_heads):
    _run_case(num_k_heads, num_v_heads, batch_size=1)


@pytest.mark.parametrize(
    ("num_k_heads", "num_v_heads", "batch_size"),
    [
        (8, 16, 2),
        (8, 24, 3),
        (8, 24, 4),
        (8, 24, 32),
        (16, 64, 2),
    ],
)
def test_representative_batch_shapes(num_k_heads, num_v_heads, batch_size):
    _run_case(num_k_heads, num_v_heads, batch_size)
