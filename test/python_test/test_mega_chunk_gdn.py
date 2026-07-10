import pytest
import torch
import torch.nn.functional as F

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


CHUNK_SIZE = 128


def _has_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


SUPPORTED_HEAD_CONFIGS = [
    pytest.param(2, 1, id="H2-Hg1"),
    pytest.param(4, 2, id="H4-Hg2"),
    pytest.param(6, 2, id="H6-Hg2"),
    pytest.param(16, 4, id="H16-Hg4"),
    pytest.param(16, 8, id="H16-Hg8"),
    pytest.param(16, 16, id="H16-Hg16"),
    pytest.param(24, 8, id="H24-Hg8"),
    pytest.param(32, 4, id="H32-Hg4"),
    pytest.param(32, 8, id="H32-Hg8"),
    pytest.param(32, 16, id="H32-Hg16"),
    pytest.param(32, 32, id="H32-Hg32"),
    pytest.param(48, 8, id="H48-Hg8"),
    pytest.param(48, 12, id="H48-Hg12"),
    pytest.param(48, 16, id="H48-Hg16"),
    pytest.param(64, 4, id="H64-Hg4"),
    pytest.param(64, 8, id="H64-Hg8"),
    pytest.param(64, 16, id="H64-Hg16"),
]


def _assert_close(name, actual, expected):
    diff = (actual.float().cpu() - expected.float()).abs()
    max_abs = diff.max().item()
    if max_abs <= 1e-2:
        return

    rmse = torch.sqrt((diff.flatten() ** 2).mean()).item()
    base = torch.sqrt((expected.float().flatten() ** 2).mean()).item()
    ratio = rmse / max(base, 1e-8)
    assert ratio < 0.05, f"{name} max_abs={max_abs:.6f} rmse_ratio={ratio:.6f}"


def _inv_tril_inplace(a):
    assert a.shape[-2] == a.shape[-1]
    chunk_size = a.shape[-1]
    for i in range(1, chunk_size):
        row = a[..., i, :i].clone()
        sub = a[..., :i, :i].clone()
        a[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    eye = torch.eye(chunk_size, dtype=a.dtype, device=a.device)
    return a + eye


def _chunk_gated_delta_rule_native(q, k, v, g, beta, initial_state=None, output_final_state=False):
    initial_dtype = q.dtype
    q, k, v, beta, g = [x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g)]

    batch_size, num_heads, seq_len, head_dim = k.shape
    pad_size = (CHUNK_SIZE - seq_len % CHUNK_SIZE) % CHUNK_SIZE
    q = F.pad(q, (0, 0, 0, pad_size))
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    padded_len = seq_len + pad_size

    q = q * (head_dim ** -0.5)
    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    q, k, v, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, -1, CHUNK_SIZE, head_dim) for x in (q, k, v, k_beta, v_beta)
    ]
    g = g.reshape(batch_size, num_heads, -1, CHUNK_SIZE)

    mask_diag = torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=q.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)
    attn = _inv_tril_inplace(attn)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    if initial_state is None:
        last_state = torch.zeros(batch_size, num_heads, head_dim, head_dim, dtype=value.dtype, device=value.device)
    else:
        last_state = initial_state.to(value)

    out = torch.zeros_like(value)
    mask_strict = torch.triu(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(padded_len // CHUNK_SIZE):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_strict, 0)
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
        out[:, :, i] = attn_inter + attn @ v_new
        last_state = (
            last_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    final_state = last_state if output_final_state else None
    out = out.reshape(batch_size, num_heads, -1, head_dim)[:, :, :seq_len]
    return out.transpose(1, 2).contiguous().to(initial_dtype), final_state


def _native_reference(q, k, v, g, beta, cu_seqlens=None, initial_state=None, output_final_state=False):
    if q.shape[2] != v.shape[2]:
        assert v.shape[2] % q.shape[2] == 0
        group_size = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(group_size, dim=2)
        k = k.repeat_interleave(group_size, dim=2)

    if cu_seqlens is None:
        return _chunk_gated_delta_rule_native(q, k, v, g, beta, initial_state, output_final_state)

    outs = []
    final_states = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        cur_initial_state = None if initial_state is None else initial_state[seq_idx : seq_idx + 1]
        out, final_state = _chunk_gated_delta_rule_native(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            cur_initial_state,
            output_final_state,
        )
        outs.append(out)
        if output_final_state:
            final_states.append(final_state)
    return torch.cat(outs, dim=1), torch.cat(final_states, dim=0) if output_final_state else None


def _make_inputs(total_tokens, num_value_heads, num_key_heads, seed):
    torch.manual_seed(seed)
    head_dim = 128
    q = F.normalize(torch.randn(1, total_tokens, num_key_heads, head_dim), p=2, dim=-1).half()
    k = F.normalize(torch.randn(1, total_tokens, num_key_heads, head_dim), p=2, dim=-1).half()
    v = torch.randn(1, total_tokens, num_value_heads, head_dim, dtype=torch.float16)
    g = F.logsigmoid(torch.randn(1, total_tokens, num_value_heads, dtype=torch.float32))
    beta = torch.rand(1, total_tokens, num_value_heads, dtype=torch.float16)
    return q, k, v, g, beta


def _run_mega(q_cpu, k_cpu, v_cpu, g_cpu, beta_cpu, cu_list=None, initial_state_cpu=None, output_final_state=False):
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cu = None if cu_list is None else torch.tensor(cu_list, dtype=torch.long, device=device)
    result = custom_ops.mega_chunk_gdn_npu(
        q_cpu.to(device),
        k_cpu.to(device),
        v_cpu.to(device),
        g_cpu.to(device),
        beta_cpu.to(device),
        scale=128 ** -0.5,
        initial_state=None if initial_state_cpu is None else initial_state_cpu.to(device),
        output_final_state=output_final_state,
        cu_seqlens=cu,
    )
    torch_npu.npu.synchronize()
    return result[1].cpu(), None if result[3] is None else result[3].cpu()


@pytest.mark.parametrize(
    ("total_tokens", "cu_list", "num_value_heads", "num_key_heads"),
    [
        pytest.param(129, None, 2, 1, id="single-H2-Hg1"),
        pytest.param(129, None, 4, 2, id="single-H4-Hg2"),
        pytest.param(129, None, 6, 2, id="single-H6-Hg2"),
        pytest.param(129, None, 16, 4, id="single-H16-Hg4"),
        pytest.param(256, [0, 96, 128, 256], 32, 8, id="varlen-H32-Hg8"),
        pytest.param(512, [0, 96, 128, 512], 48, 16, id="long-varlen-H48-Hg16"),
    ],
)
def test_mega_chunk_gdn_e2e(total_tokens, cu_list, num_value_heads, num_key_heads):
    q, k, v, g, beta = _make_inputs(total_tokens, num_value_heads, num_key_heads, seed=0)
    actual, _ = _run_mega(q, k, v, g, beta, cu_list)
    expected, _ = _native_reference(q, k, v, g, beta, cu_list)
    _assert_close("mega_vs_native", actual, expected)


@pytest.mark.parametrize(
    ("total_tokens", "cu_list", "num_value_heads", "num_key_heads"),
    [
        pytest.param(129, None, 16, 4, id="single-H16-Hg4"),
        pytest.param(256, [0, 96, 128, 256], 64, 16, id="varlen-H64-Hg16"),
    ],
)
@pytest.mark.parametrize("state_kind", ["zero", "random"])
def test_mega_chunk_gdn_initial_state(total_tokens, cu_list, num_value_heads, num_key_heads, state_kind):
    q, k, v, g, beta = _make_inputs(total_tokens, num_value_heads, num_key_heads, seed=1)
    num_sequences = 1 if cu_list is None else len(cu_list) - 1
    if state_kind == "zero":
        h0 = torch.zeros(num_sequences, num_value_heads, 128, 128, dtype=torch.float16)
    else:
        h0 = (0.1 * torch.randn(num_sequences, num_value_heads, 128, 128)).half()

    actual, actual_final_state = _run_mega(q, k, v, g, beta, cu_list, h0, output_final_state=True)
    expected, expected_final_state = _native_reference(q, k, v, g, beta, cu_list, h0, output_final_state=True)
    _assert_close(f"mega_vs_native_{state_kind}", actual, expected)
    _assert_close(f"mega_final_state_vs_native_{state_kind}", actual_final_state, expected_final_state)


@pytest.mark.parametrize(("num_value_heads", "num_key_heads"), SUPPORTED_HEAD_CONFIGS)
def test_mega_chunk_gdn_supported_head_configs(num_value_heads, num_key_heads):
    total_tokens = 129
    cu_list = [0, 64, total_tokens]
    q, k, v, g, beta = _make_inputs(total_tokens, num_value_heads, num_key_heads, seed=2)
    h0 = (0.05 * torch.randn(len(cu_list) - 1, num_value_heads, 128, 128)).half()

    actual, actual_final_state = _run_mega(q, k, v, g, beta, cu_list, h0, output_final_state=True)
    expected, expected_final_state = _native_reference(q, k, v, g, beta, cu_list, h0, output_final_state=True)
    _assert_close(f"mega_all_configs_H{num_value_heads}_Hg{num_key_heads}", actual, expected)
    _assert_close(
        f"mega_all_configs_final_state_H{num_value_heads}_Hg{num_key_heads}",
        actual_final_state,
        expected_final_state,
    )


def test_mega_chunk_gdn_prefill_warmup_h4_hg2_smoke():
    total_tokens = 8192
    q, k, v, g, beta = _make_inputs(total_tokens, 4, 2, seed=3)

    out, final_state = _run_mega(q, k, v, g, beta, output_final_state=True)

    assert out.shape == (1, total_tokens, 4, 128)
    assert final_state.shape == (1, 4, 128, 128)
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(final_state.float()).all()
