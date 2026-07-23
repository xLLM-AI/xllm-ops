# Copyright 2025 The xLLM Authors. All Rights Reserved.
# AICPU metadata op test for quant_lightning_indexer_metadata.
# Verifies: no crash, correct output shape/dtype, output self-consistency.

from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401
import custom_ops

META_SIZE = 1024


@dataclass(frozen=True)
class QuantLightningIndexerMetadataCase:
    name: str
    batch_size: int
    max_seq_q: int
    max_seq_k: int
    num_heads_q: int
    num_heads_k: int
    head_dim: int
    query_quant_mode: int = 0
    key_quant_mode: int = 0
    layout_query: str = "BSND"
    layout_key: str = "PA_BSND"
    sparse_count: int = 8
    sparse_mode: int = 3
    pre_token: int = 9223372036854775807
    next_token: int = 9223372036854775807
    cmp_ratio: int = 1


CASES = [
    QuantLightningIndexerMetadataCase(
        name="decode_b1_sq1_sk128_n64_nk1_d128",
        batch_size=1, max_seq_q=1, max_seq_k=128,
        num_heads_q=64, num_heads_k=1, head_dim=128,
    ),
    QuantLightningIndexerMetadataCase(
        name="decode_b4_sq1_sk256_n64_nk1_d128",
        batch_size=4, max_seq_q=1, max_seq_k=256,
        num_heads_q=64, num_heads_k=1, head_dim=128,
    ),
    QuantLightningIndexerMetadataCase(
        name="decode_b1_sq1_sk512_n64_nk1_d128_sparse16",
        batch_size=1, max_seq_q=1, max_seq_k=512,
        num_heads_q=64, num_heads_k=1, head_dim=128,
        sparse_count=16,
    ),
    QuantLightningIndexerMetadataCase(
        name="decode_b2_sq1_sk1024_n64_nk1_d128",
        batch_size=2, max_seq_q=1, max_seq_k=1024,
        num_heads_q=64, num_heads_k=1, head_dim=128,
    ),
    QuantLightningIndexerMetadataCase(
        name="decode_b1_sq4_sk128_n64_nk1_d128_prefill",
        batch_size=1, max_seq_q=4, max_seq_k=128,
        num_heads_q=64, num_heads_k=1, head_dim=128,
    ),
]


def _make_inputs(case: QuantLightningIndexerMetadataCase):
    """Create actual_seq_lengths tensors on NPU."""
    B = case.batch_size
    device = "npu"
    # actual_seq_lengths_query: (B,) int32, each = max_seq_q
    aslq = torch.full((B,), case.max_seq_q, dtype=torch.int32, device=device)
    # actual_seq_lengths_key: (B,) int32, each = max_seq_k
    aslk = torch.full((B,), case.max_seq_k, dtype=torch.int32, device=device)
    return aslq, aslk


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_quant_lightning_indexer_metadata(case: QuantLightningIndexerMetadataCase):
    torch.npu.set_device(0)
    aslq, aslk = _make_inputs(case)

    meta_t = custom_ops.quant_lightning_indexer_metadata_npu(
        aslq, aslk,
        num_heads_q=case.num_heads_q, num_heads_k=case.num_heads_k,
        head_dim=case.head_dim,
        query_quant_mode=case.query_quant_mode,
        key_quant_mode=case.key_quant_mode,
        batch_size=case.batch_size,
        max_seq_q=case.max_seq_q, max_seq_k=case.max_seq_k,
        layout_query=case.layout_query, layout_key=case.layout_key,
        sparse_count=case.sparse_count, sparse_mode=case.sparse_mode,
        pre_token=case.pre_token, next_token=case.next_token,
        cmp_ratio=case.cmp_ratio,
    )
    torch.npu.synchronize()

    # Verify output shape and dtype
    assert meta_t.shape == (META_SIZE,), f"Expected shape ({META_SIZE},), got {meta_t.shape}"
    assert meta_t.dtype == torch.int32, f"Expected dtype int32, got {meta_t.dtype}"

    # Verify determinism: call again and compare
    meta_t2 = custom_ops.quant_lightning_indexer_metadata_npu(
        aslq, aslk,
        num_heads_q=case.num_heads_q, num_heads_k=case.num_heads_k,
        head_dim=case.head_dim,
        query_quant_mode=case.query_quant_mode,
        key_quant_mode=case.key_quant_mode,
        batch_size=case.batch_size,
        max_seq_q=case.max_seq_q, max_seq_k=case.max_seq_k,
        layout_query=case.layout_query, layout_key=case.layout_key,
        sparse_count=case.sparse_count, sparse_mode=case.sparse_mode,
        pre_token=case.pre_token, next_token=case.next_token,
        cmp_ratio=case.cmp_ratio,
    )
    torch.npu.synchronize()
    assert torch.equal(meta_t.cpu(), meta_t2.cpu()), "Metadata output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])