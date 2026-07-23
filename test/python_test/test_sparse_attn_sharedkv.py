#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest test for sparse_attn_sharedkv (SWA single-path + attention sink).
Golden: SWA sliding-window attention + sink term, unpaged reference.
NPU: PA_ND paged kv + metadata two-stage.
"""
import numpy as np
import pytest
import torch

from custom_ops import sparse_attn_sharedkv_npu

# ============ Golden reference (SWA + sink, numpy CPU float32) ============

def golden_swa(q, kv, sinks, B, S1, S2, N1, KV_N, D, win_left, win_right, scale):
    """
    q:    [B, S1, N1, D] float32
    kv:   [B, S2, KV_N, D] float32 (KV_N=1 broadcast)
    sinks:[N1] float32
    return: out [B, S1, N1, D] float32
    """
    out = np.zeros((B, S1, N1, D), dtype=np.float32)
    for b in range(B):
        for s1_idx in range(S1):
            diag = S2 - S1 + s1_idx
            mask_right = diag + win_right
            mask_left = max(diag - win_left, 0)
            if mask_right < mask_left:
                continue
            j_range = np.arange(mask_left, mask_right + 1)
            k = kv[b, j_range, 0, :]  # [J, D]
            for n1 in range(N1):
                logit = (q[b, s1_idx, n1, :] @ k.T) * scale  # [J]
                sink = sinks[n1]
                m = max(float(logit.max()), float(sink))
                exp_l = np.exp(logit - m)
                exp_s = np.exp(sink - m)
                denom = float(exp_l.sum()) + float(exp_s)
                p = exp_l / denom
                out[b, s1_idx, n1, :] = p @ k
    return out


def to_paged(kv, B, S2, KV_N, D, block_size):
    """
    kv: [B, S2, KV_N, D] -> kv_pa[block_num, block_size, KV_N, D] + block_table[B, max_blocks]
    """
    blocks_per_b = (S2 + block_size - 1) // block_size
    block_num = B * blocks_per_b
    kv_pa = np.zeros((block_num, block_size, KV_N, D), dtype=kv.dtype)
    block_table = np.zeros((B, blocks_per_b), dtype=np.int32)
    for b in range(B):
        for blk in range(blocks_per_b):
            gblk = b * blocks_per_b + blk
            block_table[b, blk] = gblk
            for tok in range(block_size):
                t = blk * block_size + tok
                if t < S2:
                    kv_pa[gblk, tok, :, :] = kv[b, t, :, :]
    return kv_pa, block_table


# ============ Test cases ============

CASES = [
    # (B, S1, S2, N1, KV_N, D, block_size, win_left, win_right, dtype_str)
    (1, 4, 16, 64, 1, 512, 16, 127, 0, "fp16"),
]


@pytest.mark.parametrize("B,S1,S2,N1,KV_N,D,BLOCK_SIZE,WIN_LEFT,WIN_RIGHT,dtype_str", CASES)
def test_sparse_attn_sharedkv(B, S1, S2, N1, KV_N, D, BLOCK_SIZE, WIN_LEFT, WIN_RIGHT, dtype_str):
    rng = np.random.default_rng(1234)
    scale = 1.0 / np.sqrt(D)

    # Generate data in float32
    q_f32 = rng.standard_normal((B, S1, N1, D)).astype(np.float32) * 0.1
    kv_f32 = rng.standard_normal((B, S2, KV_N, D)).astype(np.float32) * 0.1
    sinks_f32 = rng.standard_normal((N1,)).astype(np.float32) * 0.1
    seqused_kv = np.array([S2] * B, dtype=np.int32)

    # Cast to target dtype for quantization alignment
    torch_dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16
    q_t = torch.from_numpy(q_f32).to(torch_dtype)
    kv_t = torch.from_numpy(kv_f32).to(torch_dtype)

    # Golden uses the quantized values (fp16/bf16 -> fp32)
    q_ref = q_t.float().numpy()
    kv_ref = kv_t.float().numpy()
    golden_out = golden_swa(q_ref, kv_ref, sinks_f32, B, S1, S2, N1, KV_N, D,
                            WIN_LEFT, WIN_RIGHT, scale)
    # Cast golden to target dtype for comparison
    golden_out_t = torch.from_numpy(golden_out).to(torch_dtype).float().numpy()

    # Page the kv
    kv_np = kv_t.float().numpy()
    kv_pa, block_table = to_paged(kv_np, B, S2, KV_N, D, BLOCK_SIZE)
    kv_pa_t = torch.from_numpy(kv_pa).to(torch_dtype)

    # Prepare NPU inputs
    query_npu = q_t.npu()
    ori_kv_npu = kv_pa_t.npu()
    ori_bt_npu = torch.from_numpy(block_table).int().npu()
    seqused_kv_npu = torch.from_numpy(seqused_kv).int().npu()
    sinks_npu = torch.from_numpy(sinks_f32).float().npu()

    # Run NPU
    npu_out = sparse_attn_sharedkv_npu(
        query_npu, ori_kv_npu, ori_bt_npu, seqused_kv_npu, sinks_npu,
        n1=N1, kv_n=KV_N, d=D, s1=S1, s2=S2,
        ori_mask_mode=4, cmp_mask_mode=3,
        ori_win_left=WIN_LEFT, ori_win_right=WIN_RIGHT,
        softmax_scale=scale, cmp_ratio=1,
        layout_q="BSND", layout_kv="PA_ND"
    )
    npu_result = npu_out.cpu().float().numpy()

    # Compare
    atol = 2e-2
    rtol = 2e-2
    np.testing.assert_allclose(npu_result, golden_out_t, atol=atol, rtol=rtol,
                               err_msg="sparse_attn_sharedkv NPU vs golden mismatch")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])