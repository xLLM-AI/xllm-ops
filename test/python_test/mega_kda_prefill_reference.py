# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Independent CPU references for the 910B/910C and 950 KDA Prefill paths."""
import torch
from mega_kda_test_utils import round_bf16, silu_bf16

OUTPUTS = ("output", "conv_state_out", "ssm_state_out")


def validate(data):
    if any(x.device.type != "cpu" for x in data.values() if isinstance(x, torch.Tensor)):
        raise ValueError("CPU oracle only")
    if any(data[o].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
           for o in OUTPUTS for k, x in data.items()
           if isinstance(x, torch.Tensor) and k != o):
        raise ValueError("outputs must not alias inputs or each other")
    offsets = data["cu_seqlens"].tolist()
    if offsets[0] != 0 or offsets[-1] != len(data["qkv"]) or any(b < a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("invalid offsets")
    for family in ("conv", "ssm"):
        writes = set()
        for b, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            read, write = (int(data[f"{family}_{mode}_indices"][b]) for mode in ("read", "write"))
            if not -1 <= read < len(data[f"{family}_state_in"]) or not -1 <= write < len(data[f"{family}_state_out"]):
                raise ValueError("invalid state slot")
            if end > begin and write >= 0:
                if write in writes:
                    raise ValueError("duplicate destination")
                writes.add(write)
    return offsets


def reference(data, *, return_prepared=False, a5=False):
    """Accumulate in FP64 and apply the selected architecture's boundaries."""
    offsets = validate(data)
    qkv, gate = data["qkv"], data["gate"]
    heads = gate.shape[1]
    out, conv_out, ssm_out = (data[name].clone() for name in OUTPUTS)
    prepared = {name: torch.empty((len(qkv), heads, 128), dtype=torch.bfloat16)
                for name in ("q", "k", "v")}
    prepared["g"] = torch.empty((len(qkv), heads, 128), dtype=torch.float32)
    prepared["initial"] = torch.zeros((len(offsets) - 1, heads, 128, 128), dtype=torch.float32)
    for b, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        if begin == end:
            continue
        cr, cw, sr, sw = (int(data[name][b]) for name in (
            "conv_read_indices", "conv_write_indices", "ssm_read_indices", "ssm_write_indices"))
        history = torch.zeros_like(data["conv_state_in"][0]) if cr < 0 else data["conv_state_in"][cr]
        sequence = torch.cat((history, qkv[begin:end]))
        state = torch.zeros_like(data["ssm_state_in"][0], dtype=torch.float64) if sr < 0 else data["ssm_state_in"][sr].double().clone()
        prepared["initial"][b] = state.float()
        for i, token in enumerate(range(begin, end)):
            x = sequence[i:i + 4]
            x = (x if a5 else x.half()).double()
            acc = (x * data["conv_weight"].double()).sum(0)
            if data["conv_bias"] is not None:
                acc += data["conv_bias"].double()
            silu_input = acc if a5 else round_bf16(acc).double()
            conv = silu_bf16(silu_input).double().view(3, heads, 128)
            q, k, v = conv.unbind(0)
            q = round_bf16(q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)).double()
            k = round_bf16(k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)).double()
            raw = gate[token].double() + data["gate_bias"].double()
            g = -5 * torch.sigmoid(raw * data["a_log"].double().exp()[:, None])
            for name, value in zip(("q", "k", "v", "g"), (q, k, v, g)):
                prepared[name][token] = value
            state *= g.exp()[:, None, :]
            delta = (v - (state * k[:, None, :]).sum(-1)) * data["beta"][token].double()[:, None]
            state += delta[:, :, None] * k[:, None, :]
            out[token] = round_bf16((state * q[:, None, :]).sum(-1) * 128 ** -0.5)
        if cw >= 0:
            conv_out[cw] = sequence[-3:]
        if sw >= 0:
            ssm_out[sw] = state.float()
    result = (out, conv_out, ssm_out)
    return (result, prepared) if return_prepared else result


# 910B uses the xLLM 64-token numerical path. This independent CPU reference
# keeps FP32 arithmetic and marks BF16 operands at their semantic boundaries.
MAIN64_CHUNK = 64
MAIN64_SCORE_BAND = 32
MAIN64_LN2 = 0.6931471805599453
MAIN64_INV_LN2 = 1.4426950408889634
MAIN64_SCALE = 128 ** -0.5


def _bf16_boundary(value):
    return value.to(torch.bfloat16).to(torch.float32)


def _main64_exp(log2_value, *, minimum, maximum):
    exponent = log2_value * MAIN64_LN2
    return exponent.clamp(minimum * MAIN64_LN2, maximum * MAIN64_LN2).exp()


def _main64_gate_prefix(raw_gate, a_exp, gate_bias):
    logits = (raw_gate.float() + gate_bias) * a_exp
    steps = (-5.0 * (1.0 / (1.0 + torch.exp(-logits)))) * MAIN64_INV_LN2
    prefix = torch.empty_like(steps)
    accumulated = torch.zeros_like(steps[0])
    for i, step in enumerate(steps):
        accumulated = accumulated + step
        prefix[i] = accumulated
    return prefix.transpose(0, 1).contiguous()


def _main64_chunk(q, k, v, gate_prefix, beta, state):
    """Apply the chunk equations with FP32 matrices holding BF16 operands."""
    length = q.shape[1]
    raw_qk = q.new_zeros((q.shape[0], length, length))
    raw_kk = torch.zeros_like(raw_qk)
    for begin in range(0, length, MAIN64_SCORE_BAND):
        end = min(begin + MAIN64_SCORE_BAND, length)
        pivot_row = begin + (end - begin) // 2
        pivot = gate_prefix[:, pivot_row:pivot_row + 1]
        row_decay = _main64_exp(gate_prefix[:, begin:end] - pivot, minimum=-126, maximum=120)
        col_decay = _main64_exp(pivot - gate_prefix[:, :end], minimum=-126, maximum=120)
        right = _bf16_boundary(k[:, :end] * col_decay).transpose(-1, -2)
        raw_kk[:, begin:end, :end] = _bf16_boundary(k[:, begin:end] * row_decay) @ right
        raw_qk[:, begin:end, :end] = _bf16_boundary(q[:, begin:end] * row_decay) @ right

    lower = (raw_kk * beta).tril(diagonal=-1)
    eye = torch.eye(length, dtype=torch.float32).expand(q.shape[0], -1, -1)
    inverse = _bf16_boundary(
        torch.linalg.solve_triangular(lower + eye, eye, upper=False, unitriangular=True))
    decay = _main64_exp(gate_prefix, minimum=-80, maximum=80)
    w_seed = _bf16_boundary(_bf16_boundary(k * decay) * beta)
    u_seed = _bf16_boundary(v * beta)
    w = _bf16_boundary(inverse @ w_seed)
    u = _bf16_boundary(inverse @ u_seed)
    snapshot = _bf16_boundary(state)
    corrected_v = _bf16_boundary(u - w @ snapshot)
    local_score = _bf16_boundary(raw_qk.tril() * MAIN64_SCALE)
    q_state = _bf16_boundary(_bf16_boundary(q * decay) * MAIN64_SCALE)
    output = (q_state @ snapshot + local_score @ corrected_v).to(torch.bfloat16)
    last_gate = gate_prefix[:, -1:]
    update_key = _bf16_boundary(k * _main64_exp(last_gate - gate_prefix, minimum=-80, maximum=80))
    next_state = state * (last_gate * MAIN64_LN2).exp().transpose(-1, -2)
    next_state = next_state + update_key.transpose(-1, -2) @ corrected_v
    return output, next_state


def _main64_normalize(x):
    # 910B reduces D=128 by adding the two halves, then summing adjacent pairs.
    squares = x.square()
    partial = squares[..., :64] + squares[..., 64:]
    while partial.shape[-1] > 1:
        partial = partial[..., 0::2] + partial[..., 1::2]
    inv_norm = (partial + 1e-6).sqrt().reciprocal()
    return _bf16_boundary(x * inv_norm)


def reference_910b(data):
    offsets = validate(data)
    out, conv_out, ssm_out = (data[name].clone() for name in OUTPUTS)
    heads = data["gate"].shape[1]
    weights = data["conv_weight"].float()
    bias = data["conv_bias"]
    bias = bias.float() if bias is not None else None
    gate_bias = data["gate_bias"].float()
    a_exp = data["a_log"].float().exp()[:, None]

    for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        if begin == end:
            continue
        cr = int(data["conv_read_indices"][row])
        cw = int(data["conv_write_indices"][row])
        sr = int(data["ssm_read_indices"][row])
        sw = int(data["ssm_write_indices"][row])
        history = (torch.zeros_like(data["conv_state_in"][0]) if cr < 0
                   else data["conv_state_in"][cr])
        seq = torch.cat((history, data["qkv"][begin:end]))
        saved_state = (torch.zeros_like(data["ssm_state_in"][0]) if sr < 0
                       else data["ssm_state_in"][sr])
        state = saved_state.transpose(-1, -2).float().contiguous()
        for start in range(begin, end, MAIN64_CHUNK):
            stop = min(start + MAIN64_CHUNK, end)
            count = stop - start
            window = seq[start - begin:stop - begin + 3].float()
            acc = (window[:count] * weights[0] if bias is None
                   else torch.addcmul(bias, window[:count], weights[0]))
            for tap in range(1, 4):
                acc = torch.addcmul(acc, window[tap:tap + count], weights[tap])
            activation = _bf16_boundary(acc / (1.0 + torch.exp(-acc)))
            q, k, v = activation.reshape(count, 3, heads, 128).unbind(1)
            q = _main64_normalize(q)
            k = _main64_normalize(k)
            gate_prefix = _main64_gate_prefix(data["gate"][start:stop], a_exp, gate_bias)
            beta = data["beta"][start:stop].float().transpose(0, 1).unsqueeze(-1)
            output, state = _main64_chunk(q.transpose(0, 1), k.transpose(0, 1),
                                          v.transpose(0, 1), gate_prefix, beta, state)
            out[start:stop] = output.transpose(0, 1)
        if cw >= 0:
            conv_out[cw] = seq[-3:]
        if sw >= 0:
            ssm_out[sw] = state.transpose(-1, -2)
    return out, conv_out, ssm_out


def make_case(lengths=(129, 3, 0), heads=2, bias=False, seed=20260915):
    generator = torch.Generator().manual_seed(seed)
    batch, tokens, channels = len(lengths), sum(lengths), heads * 384

    def rand(shape, scale=0.1, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=generator) * scale).to(dtype)

    data = dict(
        qkv=rand((tokens, channels)), gate=rand((tokens, heads, 128)),
        beta=torch.sigmoid(rand((tokens, heads), dtype=torch.float32)),
        a_log=torch.zeros(heads), gate_bias=torch.full((heads, 128), -4.0),
        conv_weight=rand((4, channels)), conv_bias=rand((channels,)) if bias else None,
        conv_state_in=rand((batch + 1, 3, channels)),
        ssm_state_in=rand((batch + 2, heads, 128, 128), dtype=torch.float32),
        cu_seqlens=torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0).int(),
        conv_read_indices=torch.arange(batch, dtype=torch.int32),
        conv_write_indices=torch.arange(batch - 1, -1, -1, dtype=torch.int32) + 1,
        ssm_read_indices=torch.arange(batch - 1, -1, -1, dtype=torch.int32),
        ssm_write_indices=torch.arange(batch, dtype=torch.int32) + 2,
        output=torch.full((tokens, heads, 128), 37.0, dtype=torch.bfloat16),
        conv_state_out=torch.full((batch + 3, 3, channels), 37.0, dtype=torch.bfloat16),
        ssm_state_out=torch.full((batch + 4, heads, 128, 128), 37.0),
    )
    data["conv_read_indices"][0] = -1
    data["ssm_read_indices"][0] = -1
    return data
