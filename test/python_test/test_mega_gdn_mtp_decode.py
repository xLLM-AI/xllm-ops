# Copyright 2026 The xLLM Authors. All Rights Reserved.

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops_lib")


HEAD_DIM = 128
SUPPORTED_SPECULATIVE_TOKENS = tuple(range(1, 17))


@dataclass(frozen=True)
class MegaGdnMtpResult:
    conv_out: torch.Tensor
    conv_state: torch.Tensor
    ssm_state: torch.Tensor
    out: torch.Tensor


def _has_npu() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _has_npu(), reason="NPU is required")


def _make_inputs(
    speculative_tokens: int,
    *,
    batch_size: int = 1,
    num_k_heads: int = 1,
    num_v_heads: int = 1,
    same_slot: bool = False,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(
        20260804
        + speculative_tokens
        + 10 * batch_size
        + 100 * num_k_heads
        + 1000 * num_v_heads
    )
    sequence_length = speculative_tokens + 1
    num_state_slots = batch_size if same_slot else batch_size + 1
    conv_dim = (2 * num_k_heads + num_v_heads) * HEAD_DIM

    def rand_bfloat16(*shape: int) -> torch.Tensor:
        return (
            torch.randn(*shape, generator=generator, dtype=torch.float32)
            .mul_(0.1)
            .to(torch.bfloat16)
        )

    if same_slot:
        state_indices = torch.arange(batch_size, dtype=torch.int32)
        read_state_indices = state_indices
        write_state_indices = state_indices.clone()
    else:
        read_state_indices = torch.zeros(batch_size, dtype=torch.int32)
        write_state_indices = torch.arange(
            1, batch_size + 1, dtype=torch.int32
        )

    return {
        "qkv": rand_bfloat16(batch_size, sequence_length, conv_dim),
        "z": rand_bfloat16(
            batch_size, sequence_length, num_v_heads, HEAD_DIM
        ),
        "b": rand_bfloat16(batch_size, sequence_length, num_v_heads),
        "a": rand_bfloat16(batch_size, sequence_length, num_v_heads),
        "conv_weight": rand_bfloat16(4, conv_dim),
        "conv_state": rand_bfloat16(
            num_state_slots, sequence_length + 2, conv_dim
        ),
        "a_log": torch.full((num_v_heads,), -1.0, dtype=torch.float32),
        "dt_bias": torch.zeros(num_v_heads, dtype=torch.float32),
        "ssm_state": torch.randn(
            num_state_slots * sequence_length,
            num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.1),
        "read_state_indices": read_state_indices,
        "write_state_indices": write_state_indices,
        "num_accepted_tokens": torch.full(
            (batch_size,),
            (sequence_length + 1) // 2,
            dtype=torch.int32,
        ),
        "norm_weight": rand_bfloat16(HEAD_DIM),
    }


def _l2_normalize(value: torch.Tensor) -> torch.Tensor:
    return value * torch.rsqrt(
        value.square().sum(dim=-1, keepdim=True) + 1e-6
    )


def _reference(
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    read_state_indices: torch.Tensor,
    write_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    norm_weight: torch.Tensor,
    conv_output: torch.Tensor | None = None,
) -> MegaGdnMtpResult:
    batch_size, sequence_length, conv_dim = qkv.shape
    num_v_heads = z.size(2)
    num_k_heads = (
        conv_dim - num_v_heads * HEAD_DIM
    ) // (2 * HEAD_DIM)
    state_stride = sequence_length

    conv_state_out = conv_state.clone()
    ssm_state_out = ssm_state.clone()
    conv_outputs = []
    outputs = []

    read_conv_snapshots = conv_state.index_select(
        0, read_state_indices.to(torch.int64)
    ).clone()
    read_checkpoints = (
        read_state_indices.to(torch.int64) * state_stride
        + num_accepted_tokens.to(torch.int64)
        - 1
    )
    read_ssm_snapshots = ssm_state.index_select(
        0, read_checkpoints
    ).clone()

    for batch_idx in range(batch_size):
        write_slot = int(write_state_indices[batch_idx])
        accepted = int(num_accepted_tokens[batch_idx])
        read_conv = read_conv_snapshots[batch_idx]
        if conv_output is None:
            history = read_conv[accepted - 1 : accepted + 2].float()
            token_conv_outputs = []
            for token_idx in range(sequence_length):
                token = qkv[batch_idx, token_idx].float()
                conv_acc = (
                    (history * conv_weight[:3].float()).sum(dim=0)
                    + token * conv_weight[3].float()
                )
                conv_fp32 = conv_acc * torch.reciprocal(
                    torch.exp(-conv_acc) + 1.0
                )
                token_conv_outputs.append(conv_fp32.to(torch.bfloat16))
                history = torch.cat(
                    (history[1:], token.unsqueeze(0)), dim=0
                )
            batch_conv = torch.stack(token_conv_outputs)
        else:
            batch_conv = conv_output[batch_idx]
        conv_outputs.append(batch_conv)
        conv_state_out[write_slot, :2] = read_conv[
            accepted : accepted + 2
        ]
        conv_state_out[write_slot, 2 : sequence_length + 2] = qkv[
            batch_idx
        ]

        q = batch_conv[:, : num_k_heads * HEAD_DIM].reshape(
            sequence_length, num_k_heads, HEAD_DIM
        )
        k = batch_conv[
            :, num_k_heads * HEAD_DIM : 2 * num_k_heads * HEAD_DIM
        ].reshape(sequence_length, num_k_heads, HEAD_DIM)
        v = batch_conv[:, 2 * num_k_heads * HEAD_DIM :].reshape(
            sequence_length, num_v_heads, HEAD_DIM
        )
        q = _l2_normalize(q.float()) / HEAD_DIM**0.5
        k = _l2_normalize(k.float())
        repeats = num_v_heads // num_k_heads
        q = q.repeat_interleave(repeats, dim=1)
        k = k.repeat_interleave(repeats, dim=1)

        state = read_ssm_snapshots[batch_idx].float()
        token_outputs = []
        for token_idx in range(sequence_length):
            g = (
                -torch.exp(a_log)
                * F.softplus(a[batch_idx, token_idx].float() + dt_bias)
            ).to(torch.bfloat16).float()
            decay = torch.exp(g)
            beta = torch.sigmoid(
                b[batch_idx, token_idx].float()
            ).to(torch.bfloat16).float()
            state = state * decay[:, None, None]
            prediction = torch.einsum("hkv,hk->hv", state, k[token_idx])
            delta = (v[token_idx].float() - prediction) * beta[:, None]
            state = state + torch.einsum(
                "hk,hv->hkv", k[token_idx], delta
            )
            readout = torch.einsum("hkv,hk->hv", state, q[token_idx])

            checkpoint = write_slot * state_stride + token_idx
            ssm_state_out[checkpoint] = state

            norm_input = readout.to(torch.bfloat16).float()
            rms_inv = torch.rsqrt(
                norm_input.square().mean(dim=-1, keepdim=True) + 1e-6
            )
            norm_output = norm_input * rms_inv * norm_weight.float()
            norm_output = norm_output * F.silu(
                z[batch_idx, token_idx].float()
            )
            token_outputs.append(norm_output.to(torch.bfloat16))
        outputs.append(torch.stack(token_outputs))

    return MegaGdnMtpResult(
        conv_out=torch.stack(conv_outputs),
        conv_state=conv_state_out,
        ssm_state=ssm_state_out,
        out=torch.stack(outputs),
    )


def _run_unfused_conv(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    batch_size, sequence_length, _ = inputs["qkv"].shape
    read_state_indices = inputs["read_state_indices"].to(torch.int64)
    conv_state = inputs["conv_state"].index_select(
        0, read_state_indices
    ).clone()
    query_start_loc = tuple(
        index * sequence_length for index in range(batch_size + 1)
    )
    cache_indices = tuple(range(batch_size))
    num_accepted_tokens = tuple(
        int(value) for value in inputs["num_accepted_tokens"]
    )

    conv_output = custom_ops.causal_conv1d(
        inputs["qkv"].to("npu:0"),
        inputs["conv_weight"].to("npu:0"),
        conv_state.to("npu:0"),
        None,
        query_start_loc,
        cache_indices,
        (),
        num_accepted_tokens,
        1,
        -1,
        1,
    )
    torch.npu.synchronize()
    return conv_output.cpu()


def _expected_from_unfused_conv(
    inputs: dict[str, torch.Tensor],
) -> MegaGdnMtpResult:
    conv_output = _run_unfused_conv(inputs)
    return _reference(**inputs, conv_output=conv_output)


def _run_npu(inputs: dict[str, torch.Tensor]) -> MegaGdnMtpResult:
    argument_names = (
        "qkv",
        "z",
        "b",
        "a",
        "conv_weight",
        "conv_state",
        "a_log",
        "dt_bias",
        "ssm_state",
        "read_state_indices",
        "write_state_indices",
        "num_accepted_tokens",
        "norm_weight",
    )
    device_inputs = {
        name: tensor.to("npu:0") for name, tensor in inputs.items()
    }
    actual = custom_ops.mega_gdn_mtp_decode(
        *(device_inputs[name] for name in argument_names)
    )
    torch.npu.synchronize()
    return MegaGdnMtpResult(*(tensor.cpu() for tensor in actual))


def _assert_matches_reference(
    actual: MegaGdnMtpResult,
    expected: MegaGdnMtpResult,
) -> None:
    torch.testing.assert_close(
        actual.conv_out, expected.conv_out, rtol=8.0e-3, atol=1.0e-6
    )
    torch.testing.assert_close(
        actual.conv_state, expected.conv_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.ssm_state,
        expected.ssm_state,
        rtol=5.0e-3,
        atol=2.5e-5,
    )
    torch.testing.assert_close(
        actual.out, expected.out, rtol=5.0e-3, atol=2.0e-2
    )


@pytest.mark.parametrize(
    "speculative_tokens", SUPPORTED_SPECULATIVE_TOKENS
)
def test_k1_to_k16_matches_reference(speculative_tokens: int) -> None:
    # Alternating ownership covers both decode and Prefix Cache fork paths
    # without doubling the full K=1..16 device matrix.
    same_slot = speculative_tokens % 2 == 0
    inputs = _make_inputs(speculative_tokens, same_slot=same_slot)

    expected = _expected_from_unfused_conv(inputs)
    actual = _run_npu(inputs)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("speculative_tokens", (1, 8, 16))
@pytest.mark.parametrize("accepted_position", ("first", "middle", "last"))
@pytest.mark.parametrize("same_slot", (False, True), ids=("fork", "same"))
def test_accepted_checkpoint_boundaries(
    speculative_tokens: int,
    accepted_position: str,
    same_slot: bool,
) -> None:
    inputs = _make_inputs(speculative_tokens, same_slot=same_slot)
    sequence_length = speculative_tokens + 1
    accepted_values = {
        "first": 1,
        "middle": (sequence_length + 1) // 2,
        "last": sequence_length,
    }
    inputs["num_accepted_tokens"].fill_(
        accepted_values[accepted_position]
    )

    expected = _expected_from_unfused_conv(inputs)
    actual = _run_npu(inputs)

    _assert_matches_reference(actual, expected)


def test_k8_qk_group_cache_prefix_fork_matches_reference() -> None:
    inputs = _make_inputs(
        8,
        batch_size=4,
        num_k_heads=8,
        num_v_heads=24,
        same_slot=False,
    )
    inputs["num_accepted_tokens"] = torch.tensor(
        (1, 3, 6, 9), dtype=torch.int32
    )

    expected = _expected_from_unfused_conv(inputs)
    actual = _run_npu(inputs)

    _assert_matches_reference(actual, expected)


@pytest.mark.parametrize("aiv_count", (56, 64, 72))
def test_k8_two_owner_schedule_covers_each_head_once(
    aiv_count: int,
) -> None:
    group_count = 4 * 8
    active_cores = min(aiv_count, 2 * group_count)
    singleton_owner_count = active_cores - group_count
    owner_tasks: list[list[tuple[int, int]]] = [
        [] for _ in range(active_cores)
    ]

    for core_idx in range(active_cores):
        if core_idx < group_count:
            owner_tasks[core_idx].extend(
                ((core_idx, 0), (core_idx, 1))
            )
            continue
        singleton_idx = core_idx - group_count
        for group_idx in range(
            singleton_idx, group_count, singleton_owner_count
        ):
            owner_tasks[core_idx].append((group_idx, 2))

    actual_tasks = sorted(
        task for tasks in owner_tasks for task in tasks
    )
    expected_tasks = [
        (group_idx, value_head_idx)
        for group_idx in range(group_count)
        for value_head_idx in range(3)
    ]
    assert actual_tasks == expected_tasks
    assert max(map(len, owner_tasks)) == 2
    assert active_cores == min(aiv_count, 64)


@pytest.mark.parametrize(
    ("num_k_heads", "num_v_heads"),
    (
        pytest.param(16, 16, id="qwen35-0p8b-2b"),
        pytest.param(16, 32, id="qwen35-4b-9b-35b"),
        pytest.param(16, 48, id="qwen35-27b"),
        pytest.param(16, 64, id="qwen35-122b-397b"),
    ),
)
def test_qwen35_model_head_geometries(
    num_k_heads: int,
    num_v_heads: int,
) -> None:
    inputs = _make_inputs(
        3,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        same_slot=True,
    )

    expected = _expected_from_unfused_conv(inputs)
    actual = _run_npu(inputs)

    _assert_matches_reference(actual, expected)
