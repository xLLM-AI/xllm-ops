import pytest
import torch
import torch.nn.functional as F


torch_npu = pytest.importorskip("torch_npu")
custom_ops_lib = pytest.importorskip("custom_ops_lib")

HEAD_DIM = 128
WIDTH = 4


def _cumulative_lengths(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return tuple(result)


def _conv_reference(
    x,
    weight,
    conv_state,
    lengths,
    cache_indices,
    initial_state_mode,
):
    outputs = []
    token_offset = 0
    weight_fp32 = weight.float()
    for batch_idx, length in enumerate(lengths):
        slot = cache_indices[batch_idx]
        sequence = x[token_offset : token_offset + length]
        history = (
            conv_state[slot].clone()
            if initial_state_mode[batch_idx]
            else torch.zeros((WIDTH - 1, x.size(1)), dtype=x.dtype)
        )
        padded = torch.cat((history, sequence), dim=0)
        accumulator = torch.zeros_like(sequence, dtype=torch.float32)
        for tap in range(WIDTH):
            accumulator.add_(
                padded[tap : tap + length].float() * weight_fp32[tap]
            )
        outputs.append(F.silu(accumulator).to(x.dtype))
        conv_state[slot].copy_(padded[-(WIDTH - 1) :])
        token_offset += length
    return torch.cat(outputs, dim=0)


def _packed_reference(conv_output, q_dim, k_dim, v_dim, output_dtype):
    q, k, v = torch.split(conv_output, (q_dim, k_dim, v_dim), dim=-1)

    def normalize_qk(value):
        heads = value.view(value.size(0), -1, HEAD_DIM).float()
        norm = torch.sqrt((heads * heads).sum(dim=-1, keepdim=True) + 1.0e-6)
        return (heads / norm).to(torch.bfloat16).to(output_dtype)

    return torch.cat(
        (
            normalize_qk(q).reshape(-1),
            normalize_qk(k).reshape(-1),
            v.to(output_dtype).reshape(-1),
        )
    ).view_as(conv_output)


@pytest.mark.parametrize(
    ("q_heads", "v_heads", "lengths", "initial_state_mode"),
    [
        pytest.param(16, 48, (64,), (1,), id="tp1"),
        pytest.param(8, 24, (127, 129), (0, 1), id="tp2-ragged"),
        pytest.param(4, 12, (256,), (0,), id="tp4"),
        pytest.param(2, 6, (63, 65), (1, 0), id="tp8-ragged"),
        pytest.param(1, 3, (129,), (1,), id="tp16"),
    ],
)
@pytest.mark.parametrize(
    "output_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
def test_causal_conv1d_qkv_general(
    q_heads,
    v_heads,
    lengths,
    initial_state_mode,
    output_dtype,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260716 + q_heads + v_heads)
    q_dim = q_heads * HEAD_DIM
    k_dim = q_dim
    v_dim = v_heads * HEAD_DIM
    conv_dim = q_dim + k_dim + v_dim
    total_tokens = sum(lengths)
    num_slots = len(lengths) + 2
    cache_indices = tuple(range(1, len(lengths) + 1))

    x = (torch.randn((total_tokens, conv_dim), generator=generator) * 0.25).to(
        torch.bfloat16
    )
    weight = (torch.randn((WIDTH, conv_dim), generator=generator) * 0.25).to(
        torch.bfloat16
    )
    conv_state = (
        torch.randn((num_slots, WIDTH - 1, conv_dim), generator=generator) * 0.1
    ).to(torch.bfloat16)
    conv_state_ref = conv_state.clone()

    conv_output = _conv_reference(
        x,
        weight,
        conv_state_ref,
        lengths,
        cache_indices,
        initial_state_mode,
    )
    expected = _packed_reference(
        conv_output, q_dim, k_dim, v_dim, output_dtype
    )

    device = torch.device("npu")
    x_npu = x.to(device)
    weight_npu = weight.to(device)
    conv_state_npu = conv_state.to(device)
    op = (
        custom_ops_lib.causal_conv1d_packed_qkv_general_bf16
        if output_dtype == torch.bfloat16
        else custom_ops_lib.causal_conv1d_packed_qkv_general
    )
    actual = op(
        x_npu,
        weight_npu,
        conv_state_npu,
        _cumulative_lengths(lengths),
        cache_indices,
        initial_state_mode,
        q_dim,
        k_dim,
        v_dim,
        HEAD_DIM,
    )
    torch.npu.synchronize()

    assert actual.dtype == output_dtype
    atol = 1.0e-2 if output_dtype == torch.bfloat16 else 1.0e-3
    rtol = atol
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        conv_state_npu.cpu(), conv_state_ref, atol=0.0, rtol=0.0
    )
