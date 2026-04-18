import os
import pytest
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

def compare_tensors_by_ratio(
    golden: torch.Tensor,
    actual: torch.Tensor,
    name: str = "tensor",
    rtol: float = 0.01,
    atol: float = 0.004,
) -> bool:
    """
    Compare two tensors using combined tolerance: pass if |diff| <= atol + rtol * |golden|.
    Default: rtol=1%, atol=0.004 (roughly 1 ULP for bf16 near zero).
    Returns True if all points pass.
    """
    if golden.shape != actual.shape:
        print(f"  [{name}] shape mismatch: golden {golden.shape} vs actual {actual.shape}")
        return False

    golden_f = golden.float()
    actual_f = actual.float()
    diff = torch.abs(golden_f - actual_f)
    threshold = atol + rtol * torch.abs(golden_f)
    mask = diff > threshold
    failed_count = int(mask.sum().item())
    total_count = golden.numel()

    max_abs = diff.max().item()
    # Relative error only where golden is significant
    significant = torch.abs(golden_f) > atol
    if significant.any():
        rel_on_sig = (diff[significant] / torch.abs(golden_f[significant]))
        max_rel_sig = rel_on_sig.max().item()
        mean_rel_sig = rel_on_sig.mean().item()
    else:
        max_rel_sig = 0.0
        mean_rel_sig = 0.0

    if failed_count == 0:
        print(f"  [{name}] PASS  total={total_count}  "
              f"max_abs={max_abs:.6f}  "
              f"max_rel(significant)={max_rel_sig:.6f}  "
              f"mean_rel(significant)={mean_rel_sig:.6f}")
        return True
    else:
        print(f"  [{name}] FAIL  total={total_count}  "
              f"failed={failed_count} ({failed_count/total_count*100:.4f}%)  "
              f"max_abs={max_abs:.6f}  "
              f"max_rel(significant)={max_rel_sig:.6f}")
        failed_indices = torch.nonzero(mask, as_tuple=False)
        for i in range(min(5, failed_count)):
            idx = tuple(failed_indices[i].tolist())
            g = golden_f[idx].item()
            a = actual_f[idx].item()
            d = diff[idx].item()
            t = threshold[idx].item()
            print(f"    pos{idx}: golden={g:.8f}  actual={a:.8f}  "
                  f"diff={d:.8f}  threshold={t:.8f}")
        return False

def make_inputs(bs, mtp, nk, nv, dk, dv, use_g=True, use_gk=False,
                use_accepted_tokens=False, seed=42):
    """Generate inputs on CPU, matching the kernel's expected shapes."""
    torch.manual_seed(seed)

    actual_seq_lengths = torch.ones(bs, dtype=torch.int32) * mtp
    t = int(actual_seq_lengths.sum().item())

    state = torch.rand((t, nv, dv, dk), dtype=torch.bfloat16)
    query = torch.nn.functional.normalize(
        torch.rand((t, nk, dk), dtype=torch.bfloat16), p=2, dim=-1
    )
    key = torch.nn.functional.normalize(
        torch.rand((t, nk, dk), dtype=torch.bfloat16), p=2, dim=-1
    )
    value = torch.rand((t, nv, dv), dtype=torch.bfloat16)
    beta = torch.rand((t, nv), dtype=torch.bfloat16)
    scale = dk ** -0.5

    ssm_state_indices = torch.arange(t, dtype=torch.int32)

    g = None
    if use_g:
        g = -torch.rand((t, nv), dtype=torch.float32)

    gk = None
    if use_gk:
        gk = -torch.rand((t, nv, dk), dtype=torch.float32)

    num_accepted_tokens = None
    if use_accepted_tokens and mtp > 1:
        num_accepted_tokens = torch.randint(1, mtp + 1, (bs,), dtype=torch.int32)

    return {
        "query": query,
        "key": key,
        "value": value,
        "state": state,
        "beta": beta,
        "scale": scale,
        "actual_seq_lengths": actual_seq_lengths,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
        "g": g,
        "gk": gk,
    }

def recurrent_gated_delta_rule_golden(
    query: torch.Tensor,             # [T, NK, dk]  bf16
    key: torch.Tensor,               # [T, NK, dk]  bf16
    value: torch.Tensor,             # [T, NV, dv]  bf16
    state: torch.Tensor,             # [T, NV, dv, dk]  bf16
    beta: torch.Tensor,              # [T, NV]  bf16
    scale: float,
    actual_seq_lengths: torch.Tensor, # [B]  int32
    ssm_state_indices: torch.Tensor,  # [T]  int32
    num_accepted_tokens: Optional[torch.Tensor] = None,  # [B] int32
    g: Optional[torch.Tensor] = None,   # [T, NV] fp32
    gk: Optional[torch.Tensor] = None,  # [T, NK, dk] fp32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        attn_out:    [T, NV, dv]     bf16
        final_state: [T, NV, dv, dk] bf16
    """
    T, NK, dk = query.shape
    _, NV, dv = value.shape
    B = actual_seq_lengths.shape[0]

    # Cast all to float32 for precision
    q_f = query.float() * scale
    k_f = key.float()
    v_f = value.float()
    beta_f = beta.float()
    state_f = state.float()

    alpha = torch.exp(g.float()) if g is not None else None
    alpha_k = torch.exp(gk.float()) if gk is not None else None

    attn_out = torch.zeros(T, NV, dv, dtype=torch.float32)
    final_state = state_f.clone()

    seq_ptr = int(actual_seq_lengths[0].item())
    for b in range(1, B):
        seq_len = int(actual_seq_lengths[b].item())
        seq0 = seq_ptr
        seq1 = seq0 + seq_len
        seq_ptr = seq1

        # Determine which state position to load as initial state
        state_token_idx = seq0
        if num_accepted_tokens is not None:
            accepted = int(num_accepted_tokens[b - 1].item())
            state_token_idx = seq0 + accepted - 1

        state_offset = int(ssm_state_indices[state_token_idx].item())

        for h_v in range(NV):
            h_k = h_v // (NV // NK)

            # Load initial state for this head: [dv, dk]
            S = final_state[state_offset, h_v].clone()

            for seq_i in range(seq0, seq1):
                q_h = q_f[seq_i, h_k]       # [dk]
                k_h = k_f[seq_i, h_k]       # [dk]
                v_h = v_f[seq_i, h_v]       # [dv]
                b_h = beta_f[seq_i, h_v]    # scalar

                # Apply gama decay
                if alpha is not None:
                    S = S * alpha[seq_i, h_v]

                # Apply gamaK decay (element-wise along dk dimension)
                if alpha_k is not None:
                    S = S * alpha_k[seq_i, h_v].unsqueeze(0)

                # delta = v - S @ k
                Sk = torch.mv(S, k_h)       # [dv]
                delta = v_h - Sk
                delta = delta * b_h

                # Rank-1 update: S += delta * k^T
                S = S + torch.outer(delta, k_h)

                # Attention output: o = S @ q_scaled
                attn = torch.mv(S, q_h)     # [dv]

                # Store
                attn_out[seq_i, h_v] = attn
                final_state[int(ssm_state_indices[seq_i].item()), h_v] = S

    return attn_out.to(torch.bfloat16), final_state.to(torch.bfloat16)

def run_golden(inp):
    """Run CPU golden implementation."""
    return recurrent_gated_delta_rule_golden(
        query=inp["query"],
        key=inp["key"],
        value=inp["value"],
        state=inp["state"].clone(),
        beta=inp["beta"],
        scale=inp["scale"],
        actual_seq_lengths=inp["actual_seq_lengths"],
        ssm_state_indices=inp["ssm_state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        g=inp["g"],
        gk=inp["gk"],
    )

def run_npu(inp, device):
    """Run NPU operator."""
    q_npu = inp["query"].to(device)
    k_npu = inp["key"].to(device)
    v_npu = inp["value"].to(device)
    s_npu = inp["state"].clone().to(device)
    b_npu = inp["beta"].to(device)
    asl_npu = inp["actual_seq_lengths"].to(device)
    ssi_npu = inp["ssm_state_indices"].to(device)

    g_npu = inp["g"].to(device) if inp["g"] is not None else None
    gk_npu = inp["gk"].to(device) if inp["gk"] is not None else None
    nat_npu = inp["num_accepted_tokens"].to(device) if inp["num_accepted_tokens"] is not None else None

    result = custom_ops.recurrent_gated_delta_rule_npu(
        q_npu, k_npu, v_npu, s_npu,
        beta=b_npu,
        scale=inp["scale"],
        actual_seq_lengths=asl_npu,
        ssm_state_indices=ssi_npu,
        num_accepted_tokens=nat_npu,
        g=g_npu,
        gk=gk_npu,
    )
    torch_npu.npu.synchronize()

    # result is (attn_out, final_state)
    if isinstance(result, (tuple, list)):
        attn_out = result[0].cpu()
        final_state = result[1].cpu()
    else:
        attn_out = result.cpu()
        final_state = s_npu.cpu()

    return attn_out, final_state

def run_test_case(desc, bs, mtp, nk, nv, dk, dv, device,
                  use_g=True, use_gk=False, use_accepted_tokens=False,
                  seed=42, rtol=0.01, atol=0.004):
    """Run a single test case and report result."""
    print(f"\n{'='*60}")
    print(f"TEST: {desc}")
    print(f"  bs={bs}, mtp={mtp}, nk={nk}, nv={nv}, dk={dk}, dv={dv}")
    print(f"  use_g={use_g}, use_gk={use_gk}, use_accepted_tokens={use_accepted_tokens}")
    print(f"  seed={seed}, rtol={rtol}, atol={atol}")
    print(f"{'='*60}")

    inp = make_inputs(bs, mtp, nk, nv, dk, dv,
                      use_g=use_g, use_gk=use_gk,
                      use_accepted_tokens=use_accepted_tokens, seed=seed)

    print("  Running CPU golden ...")
    golden_attn, golden_state = run_golden(inp)

    print("  Running NPU ...")
    npu_attn, npu_state = run_npu(inp, device)

    print(f"  NPU attn  shape={npu_attn.shape}  dtype={npu_attn.dtype}")
    print(f"  NPU state shape={npu_state.shape}  dtype={npu_state.dtype}")

    torch.testing.assert_close(golden_attn, npu_attn, rtol=rtol, atol=atol, equal_nan=True)
    torch.testing.assert_close(golden_state, npu_state, rtol=rtol, atol=atol, equal_nan=True)

    attn_pass = compare_tensors_by_ratio(golden_attn, npu_attn, "attn_out", rtol=rtol, atol=atol)
    state_pass = compare_tensors_by_ratio(golden_state, npu_state, "final_state", rtol=rtol, atol=atol)

    overall = attn_pass and state_pass
    status = "PASS" if overall else "FAIL"
    print(f"\n  >> {desc}: {status}")
    return overall

@pytest.mark.parametrize(
    ("B", "MTP", "H", "HV", "DK", "DV"),
    [
        pytest.param(
            *test,
            id="B{}-MTP{}-H{}-HV{}-DK{}-DV{}".format(*test)
        )
        for test in [
            (2, 1, 4, 8, 128, 128),
        ]
    ],
)
def test_accuracy_recurrent(
    B: int,
    MTP: int,
    H: int,
    HV: int,
    DK: int,
    DV: float,
):
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)

    results = []

    # --- Core test cases (matching xx.py default shapes) ---
    results.append(run_test_case(
        "basic_bs2_mtp1",
        bs=2, mtp=1, nk=4, nv=8, dk=128, dv=128, device=device,
        use_g=True, use_accepted_tokens=False,
    ))
