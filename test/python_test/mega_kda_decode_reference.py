# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""FP64 CPU formula oracle for the GLM vector-gate KDA decode ABI."""
import torch


OUTPUTS = ("output", "conv_state_out", "ssm_state_out")


def reference(data):
    """Return independent output pools, preserving padding and discarded writes."""
    if any(x.device.type != "cpu" for x in data.values() if isinstance(x, torch.Tensor)):
        raise ValueError("the oracle must run on CPU")
    mode, capacity = data["mode"], data["max_query_tokens"]
    accepted = data["num_accepted_tokens"]
    if not ((mode == 0 and capacity == 1 and accepted is None) or
            (mode == 1 and 1 <= capacity <= 17 and accepted is not None)):
        raise ValueError("invalid route")
    qkv = data["qkv"]
    heads = data["gate"].shape[1]
    conv_in, ssm_in = data["conv_state_in"], data["ssm_state_in"]
    out, conv_out, ssm_out = (data[name].clone() for name in OUTPUTS)
    offsets = data["cu_seqlens"].tolist()
    if offsets[0] != 0 or offsets[-1] != qkv.shape[0] or any(b < a for a, b in zip(offsets, offsets[1:])):
        raise ValueError("invalid offsets")
    conv_destinations, ssm_destinations = set(), set()
    for row, (begin, end) in enumerate(zip(offsets, offsets[1:])):
        count = end - begin
        if count == 0:
            continue
        if count > capacity or (mode == 0 and (count != 1 or begin != row)):
            raise ValueError("invalid query length")
        naccepted = int(accepted[row]) if mode else 1
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
            # Preserve the public activation casts, not device accumulation rounding.
            x = sequence[i:i + 4].half().double()
            acc = torch.zeros_like(x[0]) if data["conv_bias"] is None else data["conv_bias"].double().clone()
            for tap in range(4):
                acc = acc + x[tap] * data["conv_weight"][tap].double()
            conv = torch.nn.functional.silu(acc).bfloat16().double().view(3, heads, 128)
            q, k, v = conv.unbind(0)
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * 128 ** -0.5
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            raw_g = data["gate"][token].double() + data["gate_bias"].double()
            decay = (-5.0 * torch.sigmoid(raw_g * data["a_log"].double().exp()[:, None])).exp()
            beta = torch.sigmoid(data["beta"][token].double())
            state = state * decay[:, None, :]
            delta = (v - (state * k[:, None, :]).sum(-1)) * beta[:, None]
            state = state + delta[:, :, None] * k[:, None, :]
            out[token] = (state * q[:, None, :]).sum(-1).bfloat16()
            if writes[i] >= 0:
                ssm_out[writes[i]] = state
        if cw >= 0:
            conv_out[cw, :count + 2] = sequence[1:]
    return out, conv_out, ssm_out


def make_case(batch=2, heads=2, capacity=1, mode=0, *, bias=False, ragged=False, padding=False):
    generator = torch.Generator().manual_seed(7301)
    lengths = [capacity] * batch
    if ragged and capacity > 1:
        lengths[0] = 1
    if ragged and mode == 1 and batch > 2:
        lengths[-1] = 0
    offsets = torch.tensor([0] + lengths, dtype=torch.int32).cumsum(0).int()
    tokens, channels = int(offsets[-1]), heads * 384
    history = capacity + 2 if mode else 3
    slots = batch * capacity + 3

    def rand(shape, dtype=torch.bfloat16):
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
        num_accepted_tokens=torch.tensor([capacity] + [1] * (batch - 1), dtype=torch.int32) if mode else None,
        mode=mode, max_query_tokens=capacity,
        output=torch.full((tokens, heads, 128), 37.0, dtype=torch.bfloat16),
        conv_state_out=torch.full((batch + 5, history, channels), 37.0, dtype=torch.bfloat16),
        ssm_state_out=torch.full((slots + 2, heads, 128, 128), 37.0),
    )
    if padding:
        data["conv_read_indices"][-1] = -1
    return data
