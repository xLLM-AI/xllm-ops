import pytest
import torch
import torch.nn.functional as F


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")
custom_ops_lib = pytest.importorskip("custom_ops_lib")


CHUNK_SIZE = 128
HEAD_DIM = 128
TP_SIZES = (1, 2, 4, 8, 16)
MTP_SPECULATIVE_TOKENS = (1, 2, 4, 8, 16)
TOKENS = 512
READ_SLOT = 1
WRITE_SLOT = 2
STATE_MODES = ("no_initial", "inplace", "out_of_place")

# (global value heads, global key heads), taken from the official model configs.
QWEN35_MODEL_HEADS = {
    "0.8B": (16, 16),
    "2B": (16, 16),
    "4B": (32, 16),
    "9B": (32, 16),
    "27B": (48, 16),
    "35B-A3B": (32, 16),
    "122B-A10B": (64, 16),
    "397B-A17B": (64, 16),
}


def _has_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _unique_local_head_configs():
    configs = set()
    for value_heads, key_heads in QWEN35_MODEL_HEADS.values():
        for tp_size in TP_SIZES:
            assert value_heads % tp_size == 0
            assert key_heads % tp_size == 0
            configs.add((value_heads // tp_size, key_heads // tp_size))
    return [
        pytest.param(
            value_heads,
            key_heads,
            id=f"V{value_heads}-K{key_heads}",
        )
        for value_heads, key_heads in sorted(configs)
    ]


QWEN35_LOCAL_HEAD_CONFIGS = _unique_local_head_configs()


def _make_cpu_inputs(
    value_heads,
    key_heads,
    checkpoint_stride=1,
    state_mode="no_initial",
    tokens=TOKENS,
    conv_read_slot=None,
    conv_write_slot=WRITE_SLOT,
    seq_lens=None,
):
    assert state_mode in STATE_MODES
    if seq_lens is None:
        seq_lens = (tokens,)
    else:
        seq_lens = tuple(seq_lens)
        assert seq_lens and all(seq_len > 0 for seq_len in seq_lens)
        tokens = sum(seq_lens)
    batch_size = len(seq_lens)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260728 + value_heads * 100 + key_heads)
    conv_dim = (2 * key_heads + value_heads) * HEAD_DIM
    slots = max(4, 2 * batch_size)
    conv_state_len = checkpoint_stride + 2
    if batch_size == 1:
        if conv_read_slot is None:
            conv_read_slot = {
                "no_initial": -1,
                "inplace": conv_write_slot,
                "out_of_place": READ_SLOT,
            }[state_mode]
        conv_read_slots = [conv_read_slot]
        conv_write_slots = [conv_write_slot]
    else:
        assert conv_read_slot is None
        if state_mode == "no_initial":
            conv_read_slots = [-1] * batch_size
            conv_write_slots = list(range(batch_size))
        elif state_mode == "inplace":
            conv_read_slots = list(range(batch_size))
            conv_write_slots = list(range(batch_size))
        else:
            conv_read_slots = list(range(batch_size))
            conv_write_slots = list(range(batch_size, 2 * batch_size))
    ssm_read_slots = [
        -1 if slot < 0 else slot * checkpoint_stride
        for slot in conv_read_slots
    ]
    ssm_write_slots = [
        slot * checkpoint_stride for slot in conv_write_slots
    ]
    return {
        "mixed_qkv": (
            torch.randn(tokens, conv_dim, generator=generator) * 0.12
        ).bfloat16(),
        "z": (
            torch.randn(
                tokens,
                value_heads,
                HEAD_DIM,
                generator=generator,
            )
            * 0.3
        ).bfloat16(),
        "b": (
            torch.randn(tokens, value_heads, generator=generator) * 0.4
        ).bfloat16(),
        "a": (
            torch.randn(tokens, value_heads, generator=generator) * 0.4
        ).bfloat16(),
        "conv_weight": (
            torch.randn(4, conv_dim, generator=generator) * 0.05
        ).bfloat16(),
        "conv_state": (
            torch.randn(
                slots,
                conv_state_len,
                conv_dim,
                generator=generator,
            )
            * 0.1
        ).bfloat16(),
        "a_log": torch.linspace(
            -2.0,
            0.25,
            value_heads,
            dtype=torch.float32,
        ),
        "dt_bias": torch.linspace(
            -0.75,
            0.5,
            value_heads,
            dtype=torch.float32,
        ),
        "ssm_state": (
            torch.randn(
                slots * checkpoint_stride,
                value_heads,
                HEAD_DIM,
                HEAD_DIM,
                generator=generator,
            )
            * 0.02
        ).float(),
        "norm_weight": (
            1.0
            + torch.randn(HEAD_DIM, generator=generator, dtype=torch.float32)
            * 0.05
        ).bfloat16(),
        "conv_state_read_indices": torch.tensor(
            conv_read_slots, dtype=torch.int32
        ),
        "conv_state_write_indices": torch.tensor(
            conv_write_slots, dtype=torch.int32
        ),
        "ssm_state_read_indices": torch.tensor(
            ssm_read_slots, dtype=torch.int32
        ),
        "ssm_state_write_indices": torch.tensor(
            ssm_write_slots, dtype=torch.int32
        ),
    }


def _to_device(cpu_inputs, device):
    return {
        name: tensor.to(device=device).contiguous()
        for name, tensor in cpu_inputs.items()
    }


def _make_constants(device, tokens=TOKENS, seq_lens=None):
    if seq_lens is None:
        seq_lens = (tokens,)
    assert sum(seq_lens) == tokens
    cu_seqlens = [0]
    for seq_len in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    mask_lower = torch.tril(
        torch.ones(
            CHUNK_SIZE,
            CHUNK_SIZE,
            device=device,
            dtype=torch.float32,
        ),
        diagonal=-1,
    )
    mask_full = torch.tril(
        torch.ones(
            CHUNK_SIZE,
            CHUNK_SIZE,
            device=device,
            dtype=torch.float32,
        )
    )
    minus_identity = torch.zeros(
        CHUNK_SIZE,
        CHUNK_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    minus_identity.diagonal().fill_(-1)
    return {
        "mask_lower": mask_lower,
        "mask_full": mask_full,
        "minus_identity": minus_identity,
        "cu_seqlens": torch.tensor(
            cu_seqlens, device=device, dtype=torch.int32
        ),
        "seq_lens": tuple(seq_lens),
    }


def _run_baseline(
    tensors,
    constants,
    value_heads,
    key_heads,
    state_mode,
    checkpoint_stride=1,
    conv_read_slot=None,
    conv_write_slot=WRITE_SLOT,
    conv_read_slots=None,
    conv_write_slots=None,
):
    tokens = tensors["mixed_qkv"].size(0)
    seq_lens = constants["seq_lens"]
    batch_size = len(seq_lens)
    q_dim = key_heads * HEAD_DIM
    k_dim = q_dim
    v_dim = value_heads * HEAD_DIM
    if conv_read_slots is not None or conv_write_slots is not None:
        assert conv_read_slots is not None and conv_write_slots is not None
        assert len(conv_read_slots) == batch_size
        assert len(conv_write_slots) == batch_size
    elif batch_size == 1 and conv_read_slot is not None:
        conv_read_slots = [conv_read_slot]
        conv_write_slots = [conv_write_slot]
    else:
        conv_read_slots = tensors["conv_state_read_indices"].cpu().tolist()
        conv_write_slots = tensors["conv_state_write_indices"].cpu().tolist()
    has_initial_states = [slot >= 0 for slot in conv_read_slots]
    for read_slot, write_slot in zip(conv_read_slots, conv_write_slots):
        if read_slot >= 0 and read_slot != write_slot:
            tensors["conv_state"][write_slot].copy_(
                tensors["conv_state"][read_slot]
            )

    query_start_loc = [0]
    for seq_len in seq_lens:
        query_start_loc.append(query_start_loc[-1] + seq_len)

    conv_out = custom_ops.causal_conv1d_npu(
        tensors["mixed_qkv"],
        tensors["conv_weight"],
        conv_state=tensors["conv_state"],
        bias_opt=None,
        query_start_loc_opt=query_start_loc,
        cache_indices_opt=conv_write_slots,
        initial_state_mode_opt=[int(value) for value in has_initial_states],
        num_accepted_tokens_opt=[],
        activation_mode=1,
        pad_slot_id=-1,
        run_mode=0,
    )

    g = -torch.exp(tensors["a_log"]).view(1, 1, value_heads) * F.softplus(
        tensors["a"].float().view(1, tokens, value_heads)
        + tensors["dt_bias"].view(1, 1, value_heads),
        beta=1.0,
        threshold=20.0,
    )
    if checkpoint_stride > 1:
        g = g.to(tensors["a"].dtype).float()
    beta = (
        torch.sigmoid(tensors["b"].float())
        .to(torch.bfloat16)
        .view(1, tokens, value_heads)
        .contiguous()
    )

    q_raw, k_raw, v_raw = torch.split(
        conv_out,
        (q_dim, k_dim, v_dim),
        dim=-1,
    )
    q = q_raw.float().view(1, tokens, key_heads, HEAD_DIM)
    k = k_raw.float().view(1, tokens, key_heads, HEAD_DIM)
    q = (q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1.0e-6))
    k = (k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1.0e-6))
    q = q.to(torch.bfloat16).contiguous()
    k = k.to(torch.bfloat16).contiguous()
    v = v_raw.view(1, tokens, value_heads, HEAD_DIM).contiguous()

    initial_state = None
    if all(has_initial_states):
        initial_state = tensors["ssm_state"].index_select(
            0, tensors["ssm_state_read_indices"].long()
        )
    else:
        assert not any(has_initial_states), (
            "the standalone MegaChunkGdn baseline only supports a uniform "
            "initial-state mode"
        )

    mega_outputs = custom_ops.mega_chunk_gdn_npu(
        q,
        k,
        v,
        g.contiguous(),
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=constants["cu_seqlens"],
    )
    tensors["ssm_state"].index_copy_(
        0, tensors["ssm_state_write_indices"].long(), mega_outputs[3]
    )

    core = mega_outputs[1].float()
    normalized = core * torch.rsqrt(
        core.square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    normalized = normalized * tensors["norm_weight"].float()
    out = normalized * F.silu(
        tensors["z"].view(1, tokens, value_heads, HEAD_DIM).float()
    )
    return {
        "out": out.to(torch.bfloat16).squeeze(0),
        "conv_state": tensors["conv_state"],
        "ssm_state": tensors["ssm_state"],
    }


def _run_e2e(tensors, constants):
    num_matrices = sum(
        (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE
        for seq_len in constants["seq_lens"]
    ) * tensors["b"].size(-1)
    out = custom_ops_lib.mega_gdn_prefill_op(
        tensors["mixed_qkv"],
        tensors["b"],
        tensors["a"],
        tensors["z"],
        tensors["conv_weight"],
        tensors["conv_state"],
        tensors["a_log"],
        tensors["dt_bias"],
        tensors["conv_state_read_indices"],
        tensors["conv_state_write_indices"],
        tensors["ssm_state_read_indices"],
        tensors["ssm_state_write_indices"],
        tensors["ssm_state"],
        constants["mask_lower"],
        constants["mask_full"],
        constants["minus_identity"],
        constants["cu_seqlens"],
        tensors["norm_weight"],
        num_matrices,
    )
    return {
        "out": out,
        "conv_state": tensors["conv_state"],
        "ssm_state": tensors["ssm_state"],
    }


def _error_metrics(actual, expected):
    actual_f32 = actual.float().cpu()
    expected_f32 = expected.float().cpu()
    diff = actual_f32 - expected_f32
    max_abs = diff.abs().max().item()
    rmse = torch.sqrt(diff.square().mean()).item()
    reference_rms = torch.sqrt(expected_f32.square().mean()).item()
    return max_abs, rmse / max(reference_rms, 1.0e-8)


def _assert_bf16_close(name, actual, expected):
    max_abs, rmse_ratio = _error_metrics(actual, expected)
    assert max_abs <= 1.5625e-2 or rmse_ratio < 5.0e-2, (
        f"{name}: max_abs={max_abs:.8f}, rmse_ratio={rmse_ratio:.8f}"
    )


def _assert_fp32_close(name, actual, expected):
    max_abs, rmse_ratio = _error_metrics(actual, expected)
    assert max_abs <= 2.0e-4 or rmse_ratio < 1.0e-3, (
        f"{name}: max_abs={max_abs:.8f}, rmse_ratio={rmse_ratio:.8f}"
    )


def _assert_results(
    e2e,
    baseline,
    write_state_index,
    prior_write_state_indices=(),
):
    _assert_bf16_close("out", e2e["out"], baseline["out"])
    torch.testing.assert_close(
        e2e["conv_state"].cpu(),
        baseline["conv_state"].cpu(),
        atol=0.0,
        rtol=0.0,
    )
    _assert_fp32_close(
        "ssm_state_write_slot",
        e2e["ssm_state"][write_state_index],
        baseline["ssm_state"][write_state_index],
    )
    for index in prior_write_state_indices:
        _assert_fp32_close(
            f"ssm_state_prior_write_slot_{index}",
            e2e["ssm_state"][index],
            baseline["ssm_state"][index],
        )
    written_state_indices = {write_state_index, *prior_write_state_indices}
    untouched_slots = torch.tensor(
        [
            index
            for index in range(e2e["ssm_state"].size(0))
            if index not in written_state_indices
        ],
        device=e2e["ssm_state"].device,
    )
    torch.testing.assert_close(
        e2e["ssm_state"].index_select(0, untouched_slots).cpu(),
        baseline["ssm_state"].index_select(0, untouched_slots).cpu(),
        atol=0.0,
        rtol=0.0,
    )


def _assert_multi_batch_results(e2e, baseline, write_state_indices):
    _assert_bf16_close("out", e2e["out"], baseline["out"])
    torch.testing.assert_close(
        e2e["conv_state"].cpu(),
        baseline["conv_state"].cpu(),
        atol=0.0,
        rtol=0.0,
    )
    written = set(write_state_indices)
    for index in sorted(written):
        _assert_fp32_close(
            f"ssm_state_write_slot_{index}",
            e2e["ssm_state"][index],
            baseline["ssm_state"][index],
        )
    untouched = torch.tensor(
        [
            index
            for index in range(e2e["ssm_state"].size(0))
            if index not in written
        ],
        device=e2e["ssm_state"].device,
    )
    torch.testing.assert_close(
        e2e["ssm_state"].index_select(0, untouched).cpu(),
        baseline["ssm_state"].index_select(0, untouched).cpu(),
        atol=0.0,
        rtol=0.0,
    )


def test_qwen35_model_tp_matrix_has_20_unique_local_shapes():
    assert len(QWEN35_MODEL_HEADS) * len(TP_SIZES) == 40
    assert len(QWEN35_LOCAL_HEAD_CONFIGS) == 20


@pytest.mark.parametrize(
    ("value_heads", "key_heads"),
    QWEN35_LOCAL_HEAD_CONFIGS,
)
def test_mega_gdn_prefill_op_all_model_tp_shapes(
    value_heads,
    key_heads,
):
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    state_mode = "no_initial"
    cpu_inputs = _make_cpu_inputs(
        value_heads, key_heads, state_mode=state_mode
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(device)
    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
    )
    e2e = _run_e2e(e2e_tensors, constants)
    torch_npu.npu.synchronize()

    _assert_results(e2e, baseline, WRITE_SLOT)


@pytest.mark.parametrize("state_mode", STATE_MODES)
def test_mega_gdn_prefill_op_state_modes(state_mode):
    value_heads = 24
    key_heads = 8
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cpu_inputs = _make_cpu_inputs(
        value_heads, key_heads, state_mode=state_mode
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(device)

    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
    )
    e2e = _run_e2e(e2e_tensors, constants)
    torch_npu.npu.synchronize()

    _assert_results(e2e, baseline, WRITE_SLOT)


@pytest.mark.parametrize("tokens", (2047, 2048, 2049, 4097))
def test_mega_gdn_prefill_op_gate_bf16_midpoint_regression(tokens):
    value_heads = 24
    key_heads = 8
    checkpoint_stride = 5
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cpu_inputs = _make_cpu_inputs(
        value_heads,
        key_heads,
        checkpoint_stride=checkpoint_stride,
        tokens=tokens,
        state_mode="no_initial",
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(device, tokens=tokens)

    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        "no_initial",
        checkpoint_stride=checkpoint_stride,
    )
    e2e = _run_e2e(e2e_tensors, constants)
    torch_npu.npu.synchronize()

    _assert_results(e2e, baseline, WRITE_SLOT * checkpoint_stride)


@pytest.mark.parametrize(
    ("seq_lens", "value_heads", "key_heads", "checkpoint_stride"),
    (
        ((256, 256), 24, 8, 1),
        ((129, 257, 63, 511), 12, 4, 2),
        ((17, 33, 65, 127, 128, 129, 255, 257), 8, 4, 5),
    ),
    ids=("batch2-equal", "batch4-ragged", "batch8-ragged"),
)
@pytest.mark.parametrize("state_mode", STATE_MODES)
def test_mega_gdn_prefill_op_multi_batch(
    seq_lens,
    value_heads,
    key_heads,
    checkpoint_stride,
    state_mode,
):
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cpu_inputs = _make_cpu_inputs(
        value_heads,
        key_heads,
        checkpoint_stride=checkpoint_stride,
        state_mode=state_mode,
        seq_lens=seq_lens,
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(
        device,
        tokens=sum(seq_lens),
        seq_lens=seq_lens,
    )

    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
        checkpoint_stride=checkpoint_stride,
    )
    e2e = _run_e2e(e2e_tensors, constants)
    torch_npu.npu.synchronize()

    write_state_indices = cpu_inputs[
        "ssm_state_write_indices"
    ].tolist()
    _assert_multi_batch_results(
        e2e,
        baseline,
        write_state_indices,
    )


@pytest.mark.parametrize("state_mode", ("no_initial", "inplace"))
def test_mega_gdn_prefill_op_batch32_qwen35_tp2_geometry(
    state_mode: str,
) -> None:
    value_heads = 24
    key_heads = 8
    seq_lens = (32,) * 32
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cpu_inputs = _make_cpu_inputs(
        value_heads,
        key_heads,
        state_mode=state_mode,
        seq_lens=seq_lens,
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(
        device,
        tokens=sum(seq_lens),
        seq_lens=seq_lens,
    )

    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
    )
    e2e = _run_e2e(e2e_tensors, constants)
    torch_npu.npu.synchronize()

    write_state_indices = cpu_inputs[
        "ssm_state_write_indices"
    ].tolist()
    _assert_multi_batch_results(
        e2e,
        baseline,
        write_state_indices,
    )


def test_mega_gdn_prefill_op_group_qk_underfilled_grid_regression():
    value_heads = 12
    key_heads = 4
    seq_lens = (454,)
    state_mode = "no_initial"
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cpu_inputs = _make_cpu_inputs(
        value_heads,
        key_heads,
        state_mode=state_mode,
        seq_lens=seq_lens,
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(
        device,
        tokens=sum(seq_lens),
        seq_lens=seq_lens,
    )

    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
    )
    for _ in range(20):
        e2e = _run_e2e(e2e_tensors, constants)
        torch_npu.npu.synchronize()
        _assert_results(e2e, baseline, WRITE_SLOT)


def test_mega_gdn_prefill_op_async_layer_queue_regression():
    """Keep each A5 software-sync reset adjacent to its queued kernel.

    Qwen3.5-27B enqueues this operator once for each of its 48 linear-attention
    layers before synchronizing.  A reset submitted directly by the caller can
    overtake an earlier OpCommand custom handler and leave the next kernel with
    stale generation counters.  A single-call test, or synchronizing after
    every call, cannot reproduce that ordering failure.
    """
    value_heads = 24
    key_heads = 8
    state_mode = "no_initial"
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cpu_inputs = _make_cpu_inputs(
        value_heads,
        key_heads,
        state_mode=state_mode,
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(device)
    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
    )

    for _ in range(48):
        e2e = _run_e2e(e2e_tensors, constants)
    torch_npu.npu.synchronize()

    _assert_results(e2e, baseline, WRITE_SLOT)


@pytest.mark.parametrize(
    ("value_heads", "key_heads"),
    QWEN35_LOCAL_HEAD_CONFIGS,
)
@pytest.mark.parametrize(
    "num_speculative_tokens",
    MTP_SPECULATIVE_TOKENS,
    ids=lambda value: f"mtp{value}",
)
@pytest.mark.parametrize(
    "state_mode",
    STATE_MODES,
)
def test_mega_gdn_prefill_op_all_model_tp_mtp_cache_layout(
    value_heads,
    key_heads,
    num_speculative_tokens,
    state_mode,
):
    checkpoint_stride = num_speculative_tokens + 1
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    cpu_inputs = _make_cpu_inputs(
        value_heads,
        key_heads,
        checkpoint_stride=checkpoint_stride,
        state_mode=state_mode,
    )
    baseline_tensors = _to_device(cpu_inputs, device)
    e2e_tensors = _to_device(cpu_inputs, device)
    constants = _make_constants(device)

    baseline = _run_baseline(
        baseline_tensors,
        constants,
        value_heads,
        key_heads,
        state_mode,
        checkpoint_stride=checkpoint_stride,
    )
    e2e = _run_e2e(e2e_tensors, constants)
    torch_npu.npu.synchronize()

    final_state_index = 2 * checkpoint_stride
    _assert_results(e2e, baseline, final_state_index)


def test_mega_gdn_prefill_op_mtp4_state_carry_1024_to_801():
    value_heads = 12
    key_heads = 4
    checkpoint_stride = 5
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)

    def run_path(use_e2e):
        shared_conv_state = None
        shared_ssm_state = None
        outputs = []
        final_result = None
        segments = (
            (1024, "no_initial", -1, READ_SLOT),
            (801, "out_of_place", READ_SLOT, WRITE_SLOT),
        )
        for tokens, state_mode, read_slot, write_slot in segments:
            cpu_inputs = _make_cpu_inputs(
                value_heads,
                key_heads,
                checkpoint_stride=checkpoint_stride,
                state_mode=state_mode,
                tokens=tokens,
                conv_read_slot=read_slot,
                conv_write_slot=write_slot,
            )
            tensors = _to_device(cpu_inputs, device)
            if shared_conv_state is not None:
                tensors["conv_state"] = shared_conv_state
                tensors["ssm_state"] = shared_ssm_state
            constants = _make_constants(device, tokens=tokens)
            if use_e2e:
                final_result = _run_e2e(tensors, constants)
            else:
                final_result = _run_baseline(
                    tensors,
                    constants,
                    value_heads,
                    key_heads,
                    state_mode,
                    checkpoint_stride=checkpoint_stride,
                    conv_read_slot=read_slot,
                    conv_write_slot=write_slot,
                )
            outputs.append(final_result["out"].clone())
            shared_conv_state = tensors["conv_state"]
            shared_ssm_state = tensors["ssm_state"]
        return outputs, final_result

    baseline_outputs, baseline = run_path(use_e2e=False)
    e2e_outputs, e2e = run_path(use_e2e=True)
    torch_npu.npu.synchronize()

    _assert_bf16_close("segment_1024_out", e2e_outputs[0], baseline_outputs[0])
    _assert_results(
        e2e,
        baseline,
        WRITE_SLOT * checkpoint_stride,
        prior_write_state_indices=(READ_SLOT * checkpoint_stride,),
    )
