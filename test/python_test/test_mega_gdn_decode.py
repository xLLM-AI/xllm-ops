# Copyright 2026 The xLLM Authors. All Rights Reserved.

import pytest
import torch
import torch.nn.functional as F

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


def _reference_decode(
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
    num_k_heads,
    num_v_heads,
):
    batch_size, conv_dim = qkv.shape
    conv_out = torch.empty_like(qkv)
    expected_conv_state = conv_state.clone()
    expected_ssm_state = ssm_state.clone()

    for batch_idx in range(batch_size):
        state_idx = state_indices[batch_idx].item()
        history = expected_conv_state[state_idx].float()
        x = qkv[batch_idx].float()
        weights = conv_weight.float()
        conv_acc = weights[0] * history[0]
        conv_acc = conv_acc + weights[1] * history[1]
        conv_acc = conv_acc + weights[2] * history[2]
        conv_acc = conv_acc + weights[3] * x
        conv_out[batch_idx] = F.silu(conv_acc).to(torch.bfloat16)
        expected_conv_state[state_idx, 0] = expected_conv_state[state_idx, 1]
        expected_conv_state[state_idx, 1] = expected_conv_state[state_idx, 2]
        expected_conv_state[state_idx, 2] = qkv[batch_idx]

    q, k, v = torch.split(
        conv_out,
        [num_k_heads * HEAD_DIM, num_k_heads * HEAD_DIM, num_v_heads * HEAD_DIM],
        dim=-1,
    )
    q = q.view(batch_size, num_k_heads, HEAD_DIM).float()
    k = k.view(batch_size, num_k_heads, HEAD_DIM).float()
    v = v.view(batch_size, num_v_heads, HEAD_DIM).float()
    q = q / torch.sqrt(torch.sum(q.square(), dim=-1, keepdim=True) + 1e-6)
    k = k / torch.sqrt(torch.sum(k.square(), dim=-1, keepdim=True) + 1e-6)
    q = q / (HEAD_DIM**0.5)
    q = q.repeat_interleave(num_v_heads // num_k_heads, dim=1)
    k = k.repeat_interleave(num_v_heads // num_k_heads, dim=1)

    decay = torch.exp(
        -torch.exp(a_log).unsqueeze(0)
        * F.softplus(a.float() + dt_bias.unsqueeze(0), beta=1.0, threshold=20.0)
    )
    beta = torch.sigmoid(b.float())
    attention_out = torch.empty_like(v)
    for batch_idx in range(batch_size):
        state_idx = state_indices[batch_idx].item()
        for head_idx in range(num_v_heads):
            state = expected_ssm_state[state_idx, head_idx] * decay[
                batch_idx, head_idx
            ]
            prediction = torch.sum(state * k[batch_idx, head_idx, :, None], dim=0)
            delta = (v[batch_idx, head_idx] - prediction) * beta[
                batch_idx, head_idx
            ]
            state = state + k[batch_idx, head_idx, :, None] * delta[None, :]
            expected_ssm_state[state_idx, head_idx] = state
            attention_out[batch_idx, head_idx] = torch.sum(
                state * q[batch_idx, head_idx, :, None], dim=0
            )

    attention_out = attention_out.to(torch.bfloat16).float()
    rms = torch.rsqrt(
        torch.mean(attention_out.square(), dim=-1, keepdim=True) + 1e-6
    )
    out = (
        attention_out * rms * norm_weight.float() * F.silu(z.float())
    ).to(torch.bfloat16)
    return conv_out, expected_conv_state, expected_ssm_state, out


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

    outputs = custom_ops.mega_gdn_decode(
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


@pytest.mark.parametrize(
    ("num_k_heads", "num_v_heads", "batch_size"),
    [(1, 2, 1), (2, 4, 2)],
)
def test_nonzero_inputs_match_reference(num_k_heads, num_v_heads, batch_size):
    generator = torch.Generator().manual_seed(20260727)
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM
    cache_slots = 3

    def randn(shape, scale, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    qkv = randn((batch_size, conv_dim), 0.2)
    z = randn((batch_size, num_v_heads, HEAD_DIM), 0.2)
    b = randn((batch_size, num_v_heads), 0.3)
    a = randn((batch_size, num_v_heads), 0.2)
    conv_weight = randn((4, conv_dim), 0.15)
    conv_state = randn((cache_slots, 3, conv_dim), 0.2)
    a_log = randn((num_v_heads,), 0.1, torch.float32) - 0.5
    dt_bias = randn((num_v_heads,), 0.1, torch.float32)
    ssm_state = randn(
        (cache_slots, num_v_heads, HEAD_DIM, HEAD_DIM), 0.02, torch.float32
    )
    state_indices = torch.tensor([2, 0][:batch_size], dtype=torch.int32)
    norm_weight = 1 + randn((HEAD_DIM,), 0.05)

    expected = _reference_decode(
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
        num_k_heads,
        num_v_heads,
    )
    npu_inputs = [
        tensor.to("npu:0")
        for tensor in (
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
    ]
    outputs = custom_ops.mega_gdn_decode(*npu_inputs)
    torch.npu.synchronize()
    actual = tuple(tensor.cpu() for tensor in outputs)

    torch.testing.assert_close(actual[0], expected[0], rtol=0.03, atol=0.003)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[2], expected[2], rtol=0.003, atol=0.0003)
    torch.testing.assert_close(actual[3], expected[3], rtol=0.05, atol=0.005)
