from typing import Tuple

import pytest
import torch
import torch.nn.functional as F


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

PAD_SLOT_ID = -1


def validate_cmp(y_cal: torch.Tensor, y_ref: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.float16:
        torch.testing.assert_close(y_cal, y_ref, rtol=3e-3, atol=1e-2, equal_nan=True)
    elif dtype == torch.bfloat16:
        torch.testing.assert_close(y_cal, y_ref, rtol=1e-2, atol=1e-2, equal_nan=True)
    elif dtype == torch.float32:
        torch.testing.assert_close(y_cal, y_ref, rtol=1e-3, atol=4e-3, equal_nan=True)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def to_int64_tuple(t: torch.Tensor) -> Tuple[int, ...]:
    t = t.to(torch.int64)
    if t.dim() == 0:
        return (t.item(),)
    return tuple(t.tolist())


def causal_conv1d_update_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation_mode: int = 0,
    pad_slot_id: int = PAD_SLOT_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = weight.shape[0]
    out = torch.empty_like(x, dtype=torch.float32)
    conv_state_out = conv_state.clone().to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    bias_fp32 = None if bias is None else bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)

    for seq in range(cache_indices.numel()):
        cache_idx = int(cache_indices[seq])
        if cache_idx == pad_slot_id:
            continue
        start = int(query_start_loc[seq])
        end = int(query_start_loc[seq + 1])
        history = conv_state_out[cache_idx].clone()
        for token_idx in range(start, end):
            window = torch.cat([history, x_fp32[token_idx : token_idx + 1]], dim=0)
            token_out = (window * weight_fp32).sum(dim=0)
            if bias_fp32 is not None:
                token_out = token_out + bias_fp32
            if activation_mode == 1:
                token_out = F.silu(token_out)
            out[token_idx] = token_out
            history = torch.cat([history[1:], x_fp32[token_idx : token_idx + 1]], dim=0)
        conv_state_out[cache_idx] = history

    return out, conv_state_out


def causal_conv1d_spec_update_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    cache_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation_mode: int = 0,
    pad_slot_id: int = PAD_SLOT_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = weight.shape[0]
    batch, seq_len, _ = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    conv_state_out = conv_state.clone().to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    bias_fp32 = None if bias is None else bias.to(torch.float32)
    x_fp32 = x.to(torch.float32)

    for seq in range(batch):
        cache_idx = int(cache_indices[seq])
        if cache_idx == pad_slot_id:
            continue
        accepted_offset = max(int(num_accepted_tokens[seq]) - 1, 0)
        history = conv_state_out[
            cache_idx, accepted_offset : accepted_offset + width - 1
        ].clone()
        for token_idx in range(seq_len):
            window = torch.cat([history, x_fp32[seq, token_idx : token_idx + 1]], dim=0)
            token_out = (window * weight_fp32).sum(dim=0)
            if bias_fp32 is not None:
                token_out = token_out + bias_fp32
            if activation_mode == 1:
                token_out = F.silu(token_out)
            out[seq, token_idx] = token_out
            history = torch.cat(
                [history[1:], x_fp32[seq, token_idx : token_idx + 1]], dim=0
            )
        conv_state_out[cache_idx] = torch.cat(
            [
                conv_state_out[
                    cache_idx,
                    accepted_offset + 1 : accepted_offset + width - 1,
                ],
                x_fp32[seq],
            ],
            dim=0,
        )

    return out, conv_state_out


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("activation_mode", [0])
@pytest.mark.parametrize("has_bias", [False, True])
def test_ascend_causal_conv1d_update(dtype: torch.dtype, activation_mode: int, has_bias: bool) -> None:
    torch.random.manual_seed(0)

    device = "npu"
    dim = 16
    width = 4
    seq_len = [8, 4]
    num_seq = len(seq_len)
    state_len = width - 1

    x = torch.randn(sum(seq_len), dim, device=device, dtype=dtype)
    weight = torch.randn(width, dim, device=device, dtype=dtype)
    conv_state = torch.randn(num_seq, state_len, dim, device=device, dtype=dtype)
    conv_state_before = conv_state.clone()
    query_start_loc = torch.cumsum(torch.tensor([0] + seq_len, device=device, dtype=torch.int32), dim=0)
    cache_indices = torch.arange(num_seq, device=device, dtype=torch.int32)
    bias = torch.randn(dim, device=device, dtype=dtype) if has_bias else None

    out = custom_ops.causal_conv1d_npu(
        x,
        weight,
        conv_state=conv_state,
        bias_opt=bias,
        query_start_loc_opt=to_int64_tuple(query_start_loc),
        cache_indices_opt=to_int64_tuple(cache_indices),
        initial_state_mode_opt=[],
        num_accepted_tokens_opt=[],
        activation_mode=activation_mode,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=1,
    )

    out_ref, conv_state_ref = causal_conv1d_update_ref(
        x.cpu(),
        weight.cpu(),
        conv_state_before.cpu(),
        query_start_loc.cpu(),
        cache_indices.cpu(),
        bias=None if bias is None else bias.cpu(),
        activation_mode=activation_mode,
    )

    validate_cmp(out.cpu().float(), out_ref, torch.float32)
    validate_cmp(conv_state.cpu().float(), conv_state_ref, torch.float32)


@pytest.mark.parametrize("dim", [16, 2560])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ascend_causal_conv1d_spec_update_reused_slot(
    dtype: torch.dtype, dim: int
) -> None:
    torch.random.manual_seed(1)

    device = "npu"
    batch = 1
    width = 4
    speculative_tokens = 8
    seq_len = speculative_tokens + 1
    state_len = speculative_tokens + width - 1
    cache_indices = torch.tensor([0], device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    weight = torch.randn(width, dim, device=device, dtype=dtype)
    conv_state = torch.randn(1, state_len, dim, device=device, dtype=dtype)

    # Leave a complete speculative window in the slot, as a previous request
    # would, then emulate fresh prefill by replacing only its valid history.
    previous_accepted = torch.tensor([5], device=device, dtype=torch.int32)
    for _ in range(4):
        previous_x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
        custom_ops.causal_conv1d_npu(
            previous_x,
            weight,
            conv_state=conv_state,
            bias_opt=None,
            query_start_loc_opt=to_int64_tuple(query_start_loc),
            cache_indices_opt=to_int64_tuple(cache_indices),
            initial_state_mode_opt=[],
            num_accepted_tokens_opt=to_int64_tuple(previous_accepted),
            activation_mode=1,
            pad_slot_id=PAD_SLOT_ID,
            run_mode=1,
        )

    fresh_history = torch.randn(width - 1, dim, device=device, dtype=dtype)
    conv_state[0, : width - 1].copy_(fresh_history)
    conv_state_before = conv_state.clone()
    x = torch.randn(batch, seq_len, dim, device=device, dtype=dtype)
    num_accepted_tokens = torch.ones(batch, device=device, dtype=torch.int32)

    out = custom_ops.causal_conv1d_npu(
        x,
        weight,
        conv_state=conv_state,
        bias_opt=None,
        query_start_loc_opt=to_int64_tuple(query_start_loc),
        cache_indices_opt=to_int64_tuple(cache_indices),
        initial_state_mode_opt=[],
        num_accepted_tokens_opt=to_int64_tuple(num_accepted_tokens),
        activation_mode=1,
        pad_slot_id=PAD_SLOT_ID,
        run_mode=1,
    )

    out_ref, conv_state_ref = causal_conv1d_spec_update_ref(
        x.cpu(),
        weight.cpu(),
        conv_state_before.cpu(),
        cache_indices.cpu(),
        num_accepted_tokens.cpu(),
        activation_mode=1,
    )

    validate_cmp(out.cpu().float(), out_ref, dtype)
    validate_cmp(conv_state.cpu().float(), conv_state_ref, dtype)
