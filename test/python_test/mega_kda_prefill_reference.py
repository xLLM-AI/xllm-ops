# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""FP64 CPU recurrent oracle; independent of the device's chunk/WY algorithm."""
import torch

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


def reference(data, *, return_prepared=False):
    """Round only public Conv/QK/output boundaries; accumulate oracle math in FP64."""
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
            # Preserve the existing Ascend Conv activation cast, then use FP64 accumulation.
            x = sequence[i:i + 4].half().double()
            acc = (x * data["conv_weight"].double()).sum(0)
            if data["conv_bias"] is not None:
                acc += data["conv_bias"].double()
            conv = torch.nn.functional.silu(acc).bfloat16().double().view(3, heads, 128)
            q, k, v = conv.unbind(0)
            q = (q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)).bfloat16().double()
            k = (k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)).bfloat16().double()
            raw = gate[token].double() + data["gate_bias"].double()
            g = -5 * torch.sigmoid(raw * data["a_log"].double().exp()[:, None])
            for name, value in zip(("q", "k", "v", "g"), (q, k, v, g)):
                prepared[name][token] = value
            state *= g.exp()[:, None, :]
            delta = (v - (state * k[:, None, :]).sum(-1)) * data["beta"][token].double()[:, None]
            state += delta[:, :, None] * k[:, None, :]
            out[token] = ((state * q[:, None, :]).sum(-1) * 128 ** -0.5).bfloat16()
        if cw >= 0:
            conv_out[cw] = sequence[-3:]
        if sw >= 0:
            ssm_out[sw] = state.float()
    result = (out, conv_out, ssm_out)
    return (result, prepared) if return_prepared else result


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
