# Copyright 2025 The xLLM Authors. All Rights Reserved.
# AICPU metadata op test for sparse_attn_sharedkv_metadata.
# Verifies: no crash, correct output shape/dtype, output self-consistency.

from dataclasses import dataclass

import pytest
import torch
import torch_npu  # noqa: F401
import custom_ops

META_SIZE = 1024


@dataclass(frozen=True)
class SparseAttnSharedkvMetadataCase:
    name: str
    batch_size: int
    max_seq_q: int
    max_seq_kv: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    ori_mask_mode: int = 4
    cmp_mask_mode: int = 3
    ori_top_k: int = 0
    cmp_top_k: int = 0
    cmp_ratio: int = 1
    ori_win_left: int = 127
    ori_win_right: int = 0
    layout_q: str = "BSND"
    layout_kv: str = "PA_ND"
    has_ori_kv: bool = True
    has_cmp_kv: bool = False


CASES = [
    SparseAttnSharedkvMetadataCase(
        name="decode_b1_s1_kv16_n64_kvn1_d512",
        batch_size=1, max_seq_q=1, max_seq_kv=16,
        num_heads_q=64, num_heads_kv=1, head_dim=512,
    ),
    SparseAttnSharedkvMetadataCase(
        name="decode_b4_s1_kv128_n64_kvn1_d512",
        batch_size=4, max_seq_q=1, max_seq_kv=128,
        num_heads_q=64, num_heads_kv=1, head_dim=512,
    ),
    SparseAttnSharedkvMetadataCase(
        name="prefill_b1_s4_kv16_n64_kvn1_d512",
        batch_size=1, max_seq_q=4, max_seq_kv=16,
        num_heads_q=64, num_heads_kv=1, head_dim=512,
    ),
    SparseAttnSharedkvMetadataCase(
        name="decode_b2_s1_kv256_n64_kvn1_d512_swa",
        batch_size=2, max_seq_q=1, max_seq_kv=256,
        num_heads_q=64, num_heads_kv=1, head_dim=512,
        ori_mask_mode=4, ori_win_left=127, ori_win_right=0,
    ),
    SparseAttnSharedkvMetadataCase(
        name="decode_b1_s1_kv64_n64_kvn1_d512_has_cmp",
        batch_size=1, max_seq_q=1, max_seq_kv=64,
        num_heads_q=64, num_heads_kv=1, head_dim=512,
        has_ori_kv=True, has_cmp_kv=True, cmp_ratio=4,
    ),
]


def _make_inputs(case: SparseAttnSharedkvMetadataCase):
    """Create input tensors for metadata op on NPU."""
    B = case.batch_size
    device = "npu"
    # cu_seq_lens: cumulative sequence lengths, shape (B+1,)
    seq_q_lens = torch.full((B,), case.max_seq_q, dtype=torch.int32)
    cu_seq_q = torch.zeros(B + 1, dtype=torch.int32)
    cu_seq_q[1:] = torch.cumsum(seq_q_lens, dim=0)

    seq_kv_lens = torch.full((B,), case.max_seq_kv, dtype=torch.int32)
    cu_seq_ori_kv = torch.zeros(B + 1, dtype=torch.int32)
    cu_seq_ori_kv[1:] = torch.cumsum(seq_kv_lens, dim=0)

    cu_seq_cmp_kv = torch.zeros(B + 1, dtype=torch.int32)
    if case.has_cmp_kv and case.cmp_ratio > 0:
        cmp_kv_lens = torch.full((B,), case.max_seq_kv // case.cmp_ratio, dtype=torch.int32)
        cu_seq_cmp_kv[1:] = torch.cumsum(cmp_kv_lens, dim=0)

    seqused_q = seq_q_lens.clone()
    seqused_kv = seq_kv_lens.clone()

    return (
        cu_seq_q.to(device), cu_seq_ori_kv.to(device), cu_seq_cmp_kv.to(device),
        seqused_q.to(device), seqused_kv.to(device),
    )


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_sparse_attn_sharedkv_metadata(case: SparseAttnSharedkvMetadataCase):
    torch.npu.set_device(0)
    cu_seq_q, cu_seq_ori_kv, cu_seq_cmp_kv, seqused_q, seqused_kv = _make_inputs(case)

    meta_t = custom_ops.sparse_attn_sharedkv_metadata_npu(
        cu_seq_q, cu_seq_ori_kv, cu_seq_cmp_kv, seqused_q, seqused_kv,
        num_heads_q=case.num_heads_q, num_heads_kv=case.num_heads_kv,
        head_dim=case.head_dim,
        batch_size=case.batch_size, max_seq_q=case.max_seq_q, max_seq_kv=case.max_seq_kv,
        ori_top_k=case.ori_top_k, cmp_top_k=case.cmp_top_k, cmp_ratio=case.cmp_ratio,
        ori_mask_mode=case.ori_mask_mode, cmp_mask_mode=case.cmp_mask_mode,
        ori_win_left=case.ori_win_left, ori_win_right=case.ori_win_right,
        layout_q=case.layout_q, layout_kv=case.layout_kv,
        has_ori_kv=case.has_ori_kv, has_cmp_kv=case.has_cmp_kv,
    )
    torch.npu.synchronize()

    # Verify output shape and dtype
    assert meta_t.shape == (META_SIZE,), f"Expected shape ({META_SIZE},), got {meta_t.shape}"
    assert meta_t.dtype == torch.int32, f"Expected dtype int32, got {meta_t.dtype}"

    # Verify determinism: call again and compare
    meta_t2 = custom_ops.sparse_attn_sharedkv_metadata_npu(
        cu_seq_q, cu_seq_ori_kv, cu_seq_cmp_kv, seqused_q, seqused_kv,
        num_heads_q=case.num_heads_q, num_heads_kv=case.num_heads_kv,
        head_dim=case.head_dim,
        batch_size=case.batch_size, max_seq_q=case.max_seq_q, max_seq_kv=case.max_seq_kv,
        ori_top_k=case.ori_top_k, cmp_top_k=case.cmp_top_k, cmp_ratio=case.cmp_ratio,
        ori_mask_mode=case.ori_mask_mode, cmp_mask_mode=case.cmp_mask_mode,
        ori_win_left=case.ori_win_left, ori_win_right=case.ori_win_right,
        layout_q=case.layout_q, layout_kv=case.layout_kv,
        has_ori_kv=case.has_ori_kv, has_cmp_kv=case.has_cmp_kv,
    )
    torch.npu.synchronize()
    assert torch.equal(meta_t.cpu(), meta_t2.cpu()), "Metadata output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])