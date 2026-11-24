# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""FP64 recurrent oracle and the shared 910B/910C FP32 Decode reference."""
from typing import Any

import torch
from mega_kda_test_utils import round_bf16, silu_bf16


OUTPUTS = ("output", "conv_state_out", "ssm_state_out")
NATIVE, MTP, PUBLIC_COMPAT = 0, 1, 3


def reference(data: dict[str, Any], *, a5: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return independent output pools, preserving padding and discarded writes."""
    if any(x.device.type != "cpu" for x in data.values() if isinstance(x, torch.Tensor)):
        raise ValueError("the oracle must run on CPU")
    mode, capacity = data["mode"], data["max_query_tokens"]
    accepted = data["num_accepted_tokens"]
    if a5 and mode == PUBLIC_COMPAT:
        raise ValueError("public compatibility mode is not supported on A5")
    if not ((mode in (NATIVE, PUBLIC_COMPAT) and capacity == 1 and accepted is None) or
            (mode == MTP and 1 <= capacity <= 17 and accepted is not None)):
        raise ValueError("invalid route")
    qkv = data["qkv"]
    heads = data["gate"].shape[1]
    conv_in, ssm_in = data["conv_state_in"], data["ssm_state_in"]
    out, conv_out, ssm_out = (data[name].clone() for name in OUTPUTS)
    offsets = data["cu_seqlens"].tolist()
    if (offsets[0] != 0 or offsets[-1] != qkv.shape[0]
            or any(b < a for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("invalid offsets")
    if mode != MTP and offsets != list(range(len(offsets))):
        raise ValueError("ordinary decode requires unit offsets")
    expected_history = capacity + 2 if mode == MTP else 3
    if conv_in.shape[1] != expected_history or conv_out.shape[1] != expected_history:
        raise ValueError("invalid Conv history")
    conv_destinations, ssm_destinations = set(), set()
    for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        count = end - begin
        if count == 0:
            continue
        if count > capacity or (mode != MTP and (count != 1 or begin != row)):
            raise ValueError("invalid query length")
        naccepted = int(accepted[row]) if mode == MTP else 1
        if not 1 <= naccepted <= capacity:
            raise ValueError("invalid accepted count")
        cr, cw = int(data["conv_read_indices"][row]), int(data["conv_write_indices"][row])
        sr = int(data["ssm_read_indices"][row, naccepted - 1])
        if cr == -1 or sr == -1:
            continue
        if not (0 <= cr < len(conv_in) and 0 <= sr < len(ssm_in) and -1 <= cw < len(conv_out)):
            raise ValueError("invalid state slot")
        if cw >= 0:
            if cw in conv_destinations:
                raise ValueError("duplicate Conv destination")
            conv_destinations.add(cw)
        writes = data["ssm_write_indices"][row, :count].tolist()
        for sw in writes:
            if not -1 <= sw < len(ssm_out):
                raise ValueError("invalid SSM destination")
            if sw >= 0:
                if sw in ssm_destinations:
                    raise ValueError("duplicate SSM destination")
                ssm_destinations.add(sw)
        history = conv_in[cr, naccepted - 1:naccepted + 2]
        sequence = torch.cat((history, qkv[begin:end]))
        state = ssm_in[sr].double().clone()
        for i, token in enumerate(range(begin, end)):
            x = sequence[i:i + 4]
            narrow_conv_input = mode == MTP or (not a5 and mode == NATIVE)
            x = (x.half() if narrow_conv_input else x).double()
            acc = torch.zeros_like(x[0]) if data["conv_bias"] is None else data["conv_bias"].double().clone()
            for tap in range(4):
                acc = acc + x[tap] * data["conv_weight"][tap].double()
            silu_input = acc if a5 and mode == NATIVE else round_bf16(acc).double()
            conv = silu_bf16(silu_input).double().view(3, heads, 128)
            q, k, v = conv.unbind(0)
            if a5:
                q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
                k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            else:
                q_norm = torch.sqrt((q * q).sum(-1, keepdim=True))
                k_norm = torch.sqrt((k * k).sum(-1, keepdim=True))
                q = q * torch.where(q_norm > 0, q_norm, 1.0).reciprocal() * 128 ** -0.5
                k = k * torch.where(k_norm > 0, k_norm, 1.0).reciprocal()
            raw_g = data["gate"][token].double() + data["gate_bias"].double()
            decay = (-5.0 * torch.sigmoid(raw_g * data["a_log"].double().exp()[:, None])).exp()
            raw_beta = data["beta"][token]
            if a5:
                beta = torch.sigmoid(raw_beta.double()).float().double()
            else:
                beta = torch.sigmoid(raw_beta.float()).double()
            state = state * decay[:, None, :]
            delta = (v - (state * k[:, None, :]).sum(-1)) * beta[:, None]
            state = state + delta[:, :, None] * k[:, None, :]
            out[token] = round_bf16((state * q[:, None, :]).sum(-1))
            if writes[i] >= 0:
                ssm_out[writes[i]] = state
        if cw >= 0:
            conv_out[cw, :count + 2] = sequence[1:]
    return out, conv_out, ssm_out


def reference_910b(data):
    """CPU FP32 recurrence retaining every public FP16/BF16 cast boundary."""
    assert all(v.device.type == "cpu" for v in data.values() if isinstance(v, torch.Tensor))
    out, conv_out, ssm_out = (data[n].clone() for n in OUTPUTS)
    heads = data["gate"].shape[1]
    mode = data["mode"]
    offsets = data["cu_seqlens"].tolist()
    for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        if begin == end:
            continue
        count = end - begin
        accepted = int(data["num_accepted_tokens"][row]) if mode == MTP else 1
        cr, cw = int(data["conv_read_indices"][row]), int(data["conv_write_indices"][row])
        sr = int(data["ssm_read_indices"][row, accepted - 1])
        if cr < 0 or sr < 0:
            continue
        history = data["conv_state_in"][cr, accepted - 1:accepted + 2]
        seq = torch.cat((history, data["qkv"][begin:end]))
        state = data["ssm_state_in"][sr].float().clone()
        x = (seq if mode == PUBLIC_COMPAT else seq.half()).float()
        acc = torch.zeros((count, x.shape[1]), dtype=torch.float32)
        if data["conv_bias"] is not None:
            acc += data["conv_bias"].float()
        for tap in range(4):
            acc = acc + x[tap:tap + count] * data["conv_weight"][tap].float()
        pre_silu = acc.bfloat16().float()
        activation = (pre_silu / (1.0 + torch.exp(-pre_silu))).bfloat16().float()
        q, k, v = activation.reshape(count, 3, heads, 128).unbind(1)
        qn, kn = q.square().sum(-1, keepdim=True), k.square().sum(-1, keepdim=True)
        q_norm, k_norm = qn.sqrt(), kn.sqrt()
        q = q * torch.where(q_norm > 0, q_norm, 1.0).reciprocal()
        k = k * torch.where(k_norm > 0, k_norm, 1.0).reciprocal()
        q = q * 128 ** -0.5
        g = (data["gate"][begin:end].float() + data["gate_bias"].float()) * data["a_log"].float().exp()[None, :, None]
        decay = (-5.0 / (1.0 + torch.exp(-g))).exp()
        beta = torch.sigmoid(data["beta"][begin:end].float())
        for i, token in enumerate(range(begin, end)):
            state = state * decay[i, :, None, :]
            delta = (v[i] - (state * k[i, :, None, :]).sum(-1)) * beta[i, :, None]
            state = state + delta[:, :, None] * k[i, :, None, :]
            y = (state * q[i, :, None, :]).sum(-1)
            out[token] = y.bfloat16()
            sw = int(data["ssm_write_indices"][row, i])
            if sw >= 0:
                ssm_out[sw] = state
        if cw >= 0:
            conv_out[cw, :count + 2] = seq[1:]
    return out, conv_out, ssm_out


def public_compat_stages(data: dict[str, Any], sequence: torch.Tensor,
                         token: int = 0) -> dict[str, torch.Tensor]:
    """Evaluate FP32 frontend boundaries, independently of the FP64 recurrence.

    This checks operation order and zero handling; Torch's CPU reductions and exp
    are not an instruction-level model of the Ascend implementation.
    """
    heads = data["gate"].shape[1]
    accumulation = (torch.zeros_like(sequence[0], dtype=torch.float32)
                    if data["conv_bias"] is None else data["conv_bias"].float().clone())
    for tap, weight in zip(sequence.float().unbind(0), data["conv_weight"].float().unbind(0)):
        accumulation = accumulation + tap * weight
    pre_silu = accumulation.bfloat16().float()
    activation = (pre_silu / (1.0 + torch.exp(-pre_silu))).bfloat16().float()
    q, k, v = activation.reshape(3, heads, 128).unbind(0)
    normalized = []
    for vector in (q, k):
        length = torch.sqrt(vector.square().sum(dim=-1, keepdim=True))
        reciprocal = 1.0 / torch.where(length > 0, length, torch.ones_like(length))
        normalized.append(vector * reciprocal)
    scaled_gate = (data["gate"][token].float() + data["gate_bias"].float()) * data["a_log"].float().exp()[:, None]
    sigmoid = 1.0 / (1.0 + torch.exp(-scaled_gate))
    gate = sigmoid * -5.0
    beta = torch.sigmoid(data["beta"][token].float())
    return {"pre_silu": pre_silu, "conv": activation, "q": normalized[0] * 128 ** -0.5,
            "k": normalized[1], "v": v, "gate": gate, "decay": gate.exp(), "beta": beta}


def make_case(batch: int = 2, heads: int = 2, capacity: int = 1, mode: int = NATIVE,
              *, bias: bool = False, ragged: bool = False,
              padding: bool = False) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(7301)
    lengths = [capacity] * batch
    if ragged and capacity > 1:
        lengths[0] = 1
    if ragged and mode == MTP and batch > 2:
        lengths[-1] = 0
    offsets = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
    tokens, channels = int(offsets[-1]), heads * 384
    history = capacity + 2 if mode == MTP else 3
    slots = batch * capacity + 3

    def rand(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * 0.1).to(dtype)

    data = dict(
        qkv=rand((tokens, channels)), gate=rand((tokens, heads, 128)), beta=rand((tokens, heads)),
        a_log=torch.full((heads,), -2.0), gate_bias=rand((heads, 128), torch.float32),
        conv_weight=rand((4, channels)), conv_bias=rand((channels,)) if bias else None,
        conv_state_in=rand((batch + 3, history, channels)),
        ssm_state_in=rand((slots, heads, 128, 128), torch.float32),
        cu_seqlens=offsets, conv_read_indices=torch.arange(batch, dtype=torch.int32),
        conv_write_indices=torch.arange(batch - 1, -1, -1, dtype=torch.int32) + 1,
        ssm_read_indices=torch.arange(batch * capacity, dtype=torch.int32).view(batch, capacity),
        ssm_write_indices=torch.arange(batch * capacity, dtype=torch.int32).view(batch, capacity) + 2,
        num_accepted_tokens=torch.tensor([capacity] + [1] * (batch - 1), dtype=torch.int32) if mode == MTP else None,
        mode=mode, max_query_tokens=capacity,
        output=torch.full((tokens, heads, 128), 37.0, dtype=torch.bfloat16),
        conv_state_out=torch.full((batch + 5, history, channels), 37.0, dtype=torch.bfloat16),
        ssm_state_out=torch.full((slots + 2, heads, 128, 128), 37.0),
    )
    if padding:
        data["conv_read_indices"][-1] = -1
    return data
