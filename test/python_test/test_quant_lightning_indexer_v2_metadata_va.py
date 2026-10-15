# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""QuantLightningIndexerV2MetadataVa unit test.

The va metadata producer is an AICPU-only op that builds the 1024 x int32
per-core tiling metadata consumed by the fused QLI v3 candidate kernel.

Layout (matching op_kernel/quant_lightning_indexer_v2_metadata_va.h):
  - QLI section (AIC): meta[core * 8 + field] for core in [0, 36), fields
    [core_enable, bn2_start, m_start, s2_start, bn2_end, m_end, s2_end,
     first_qld_workspace_idx]
  - QLD section (AIV): meta[36 * 8 + core * 8 + field] for core in [0, 72),
    fields [core_enable, bn2_idx, m_idx, workspace_idx, workspace_num,
     m_start, m_num, unused]

There is no independent golden for the split/balance math, so this test pins:
  - the exact 1024 int32 shape,
  - structural invariants of every enabled core (monotone ranges, valid
    workspace/m offsets, enable flags strictly 0/1),
  - 3x replay determinism (byte-identical output across repeated launches).
"""
from __future__ import annotations

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

AIC_CORE_MAX_NUM = 36
AIV_CORE_MAX_NUM = 72
QLI_META_SIZE = 8
QLD_META_SIZE = 8
TOTAL_SIZE = 1024

# QLI field indices
QLI_ENABLE = 0
QLI_BN2_START = 1
QLI_M_START = 2
QLI_S2_START = 3
QLI_BN2_END = 4
QLI_M_END = 5
QLI_S2_END = 6
QLI_FIRST_QLD_WS = 7

# QLD field indices
QLD_ENABLE = 0
QLD_BN2_IDX = 1
QLD_M_IDX = 2
QLD_WS_IDX = 3
QLD_WS_NUM = 4
QLD_M_START = 5
QLD_M_NUM = 6


def _make_valid_inputs(batch: int = 2, seq_q: int = 512, seq_k: int = 1024):
    """A3-valid inputs: layout_k=PA_BBND, quant_mode=2, cmp_ratio=4."""
    device = "npu"
    cu_seqlens_q = torch.tensor(
        [0] + [seq_q * (i + 1) for i in range(batch)], dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.tensor(
        [0] + [seq_k * (i + 1) for i in range(batch)], dtype=torch.int32, device=device
    )
    seqused_q = torch.full((batch,), seq_q, dtype=torch.int32, device=device)
    seqused_k = torch.full((batch,), seq_k, dtype=torch.int32, device=device)
    cmp_residual_k = torch.tensor([i % 4 for i in range(batch)], dtype=torch.int32, device=device)
    return cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k


def _call_metadata(batch: int = 2, seq_q: int = 512, seq_k: int = 1024) -> torch.Tensor:
    cu_q, cu_k, sq_q, sq_k, cmp_k = _make_valid_inputs(batch, seq_q, seq_k)
    return custom_ops.quant_lightning_indexer_v2_metadata_va_npu(
        64, 1, 128, 512, 2,  # num_heads_q, num_heads_k, head_dim, topk, quant_mode
        cu_q, cu_k, sq_q, sq_k, cmp_k,
        batch, seq_q, seq_k,  # batch_size, max_seqlen_q, max_seqlen_k
        "BSND", "PA_BBND", 3, 4,  # layout_q, layout_k, mask_mode, cmp_ratio
    )


def _check_structural_invariants(meta: torch.Tensor) -> None:
    """Every enabled core must hold a valid, monotone, in-bounds range."""
    assert meta.shape == (TOTAL_SIZE,), f"metadata shape {tuple(meta.shape)} != (1024,)"
    assert meta.dtype == torch.int32, f"metadata dtype {meta.dtype} != int32"
    m = meta.cpu()
    qli = m[: AIC_CORE_MAX_NUM * QLI_META_SIZE].view(AIC_CORE_MAX_NUM, QLI_META_SIZE)
    # QLD section occupies the next 72*8 entries; the tail (1024 - 864) is padding.
    qld_start = AIC_CORE_MAX_NUM * QLI_META_SIZE
    qld = m[qld_start : qld_start + AIV_CORE_MAX_NUM * QLD_META_SIZE].view(
        AIV_CORE_MAX_NUM, QLD_META_SIZE
    )

    enabled_qli = 0
    for core in range(AIC_CORE_MAX_NUM):
        enable = int(qli[core, QLI_ENABLE])
        assert enable in (0, 1), f"QLI core {core} enable={enable} not in {{0,1}}"
        if enable:
            enabled_qli += 1
            bn2_s, bn2_e = int(qli[core, QLI_BN2_START]), int(qli[core, QLI_BN2_END])
            m_s, m_e = int(qli[core, QLI_M_START]), int(qli[core, QLI_M_END])
            s2_s, s2_e = int(qli[core, QLI_S2_START]), int(qli[core, QLI_S2_END])
            # end == 0 is the kernel's "process to the end" sentinel
            # (quant_lightning_indexer_v2_va_kernel_arch22.h:586/721-731);
            # otherwise the range must be monotone.
            assert 0 <= bn2_s and (bn2_e == 0 or bn2_s <= bn2_e), (
                f"QLI core {core} bn2 range [{bn2_s},{bn2_e}]"
            )
            assert 0 <= m_s and (m_e == 0 or m_s <= m_e), (
                f"QLI core {core} m range [{m_s},{m_e}]"
            )
            assert 0 <= s2_s and (s2_e == 0 or s2_s <= s2_e), (
                f"QLI core {core} s2 range [{s2_s},{s2_e}]"
            )
            assert 0 <= int(qli[core, QLI_FIRST_QLD_WS]) < AIV_CORE_MAX_NUM, (
                f"QLI core {core} first_qld_ws out of AIV range"
            )
    assert enabled_qli > 0, "no QLI core enabled"

    enabled_qld = 0
    for core in range(AIV_CORE_MAX_NUM):
        enable = int(qld[core, QLD_ENABLE])
        assert enable in (0, 1), f"QLD core {core} enable={enable} not in {{0,1}}"
        if enable:
            enabled_qld += 1
            ws_num = int(qld[core, QLD_WS_NUM])
            m_num = int(qld[core, QLD_M_NUM])
            m_start = int(qld[core, QLD_M_START])
            assert ws_num > 0, f"QLD core {core} workspace_num={ws_num} <= 0"
            assert m_num > 0, f"QLD core {core} m_num={m_num} <= 0"
            assert m_start >= 0, f"QLD core {core} m_start={m_start} < 0"
            assert 0 <= int(qld[core, QLD_BN2_IDX]) and 0 <= int(qld[core, QLD_M_IDX])
            assert 0 <= int(qld[core, QLD_WS_IDX]) < AIV_CORE_MAX_NUM, (
                f"QLD core {core} workspace_idx out of AIV range"
            )
    # FD (QLD/AIV) reduction is A5-only (supportFd_ requires Ascend950,
    # aicpu.cpp:330-338); on A3 the whole QLD section is legitimately disabled,
    # so no enabled_qld > 0 assertion here.


def test_metadata_shape_and_structure() -> None:
    meta = _call_metadata()
    _check_structural_invariants(meta)


def test_metadata_replay_deterministic() -> None:
    """3x replay must be byte-identical (deterministic split/balance)."""
    first = _call_metadata().cpu()
    for _ in range(2):
        again = _call_metadata().cpu()
        assert torch.equal(first, again), "metadata differs across replays"


def test_metadata_larger_batch_structure() -> None:
    meta = _call_metadata(batch=8, seq_q=1024, seq_k=2048)
    _check_structural_invariants(meta)