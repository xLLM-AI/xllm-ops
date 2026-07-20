#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparse_attn_sharedkv SWA (Sliding Window Attention + attention sink) single-path golden + PA_ND data generation.

Kernel semantics (precisely locked by swa_kernel.h CalcParams / tiling.cpp CheckFeature):
  For each batch b, each query absolute position s1, each q head n1 (total N1=64):
    diag      = actKvS2 - actS1 + s1                (causal alignment)
    maskRight = diag + oriWinRight                  (closed-interval upper bound, default 0)
    maskLeft  = max(diag - oriWinLeft, 0)           (closed-interval lower bound, default 127)
    j in [maskLeft, maskRight]
    logit[j]  = (Q[s1,n1,:] . K[j,:]) * softmax_scale
    m         = max(max_j logit[j], sinks[n1])
    denom     = sum_j exp(logit[j]-m) + exp(sinks[n1]-m)   (sink as an extra softmax term)
    out[s1,n1,:] = sum_j (exp(logit[j]-m)/denom) * V[j,:]  (sink value=0, no contribution to numerator)

Hard constraints: N1(qHead)=64; D=512; KV_N(kvHead)=1; layout_q=BSND; layout_kv=PA_ND;
        ori_mask_mode=4; cmp_mask_mode=3; ori_win_left=127; ori_win_right=0; qType==oriKvType in {fp16,bf16}.

PA_ND: golden uses unpaged kv_bnsd as reference; NPU uses paged kv_pa[block_num,block_size,kvH=1,d] + block_table, mapped from the same source by slicing.
"""
import os
import sys
import numpy as np

DATA_DIR = "/tmp/sparse_attn_sharedkv_data"

# ------------------ Case parameters (SWA single-path, minimal scale) ------------------
B = 1              # batch
S1 = 4             # query sequence length (per batch)
S2 = 16            # kv sequence length (per batch)
N1 = 64            # q head count (hard constraint =64)
KV_N = 1           # kv head count (hard constraint =1)
D = 512            # head dim (hard constraint =512)
BLOCK_SIZE = 16    # tokens per PA_ND block
ORI_WIN_LEFT = 127
ORI_WIN_RIGHT = 0
SOFTMAX_SCALE = 1.0 / np.sqrt(D)


def golden_swa(q, kv, sinks, actS1, actS2, win_left, win_right, scale):
    """
    q:    [B, S1, N1, D]  float32
    kv:   [B, S2, KV_N, D] float32 (unpaged, KV_N=1 broadcast to N1)
    sinks:[N1]            float32
    return: out [B, S1, N1, D] float32, lse [B, S1, N1] float32
    """
    out = np.zeros((B, S1, N1, D), dtype=np.float32)
    lse = np.zeros((B, S1, N1), dtype=np.float32)
    for b in range(B):
        for s1 in range(actS1):
            diag = actS2 - actS1 + s1
            mask_right = diag + win_right
            mask_left = max(diag - win_left, 0)
            if mask_right < mask_left:
                # no valid kv, sink only
                for n1 in range(N1):
                    out[b, s1, n1, :] = 0.0
                    lse[b, s1, n1] = sinks[n1]  # log(exp(sink-sink))+sink = sink
                continue
            j_range = np.arange(mask_left, mask_right + 1)  # closed interval
            k = kv[b, j_range, 0, :]                         # [J, D] (kvHead=0 broadcast)
            for n1 in range(N1):
                logit = (q[b, s1, n1, :] @ k.T) * scale      # [J]
                sink = sinks[n1]
                m = max(float(logit.max()), float(sink))
                exp_l = np.exp(logit - m)                    # [J]
                exp_s = np.exp(sink - m)
                denom = float(exp_l.sum()) + float(exp_s)
                p = exp_l / denom                            # [J]
                out[b, s1, n1, :] = p @ k                    # sink value=0
                lse[b, s1, n1] = np.log(denom) + m
    return out, lse


def to_paged(kv, block_size):
    """
    kv: [B, S2, KV_N, D] -> paged kv[block_num, block_size, KV_N, D] + block_table[B, max_blocks]
    Same-source slice mapping: token (b, t) -> global block = b*blocks_per_b + t//block_size
    """
    blocks_per_b = (S2 + block_size - 1) // block_size
    max_blocks = blocks_per_b
    block_num = B * blocks_per_b
    kv_pa = np.zeros((block_num, block_size, KV_N, D), dtype=kv.dtype)
    block_table = np.zeros((B, max_blocks), dtype=np.int32)
    for b in range(B):
        for blk in range(blocks_per_b):
            gblk = b * blocks_per_b + blk
            block_table[b, blk] = gblk
            for tok in range(block_size):
                t = blk * block_size + tok
                if t < S2:
                    kv_pa[gblk, tok, :, :] = kv[b, t, :, :]
    return kv_pa, block_table


def main():
    dtype_str = sys.argv[1] if len(sys.argv) > 1 else "fp16"
    np_dtype = np.float16 if dtype_str == "fp16" else np.uint16  # bf16 stored as uint16 bits
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(1234)

    # ---- generate data (float32 reference) ----
    q = rng.standard_normal((B, S1, N1, D)).astype(np.float32) * 0.1
    kv = rng.standard_normal((B, S2, KV_N, D)).astype(np.float32) * 0.1
    sinks = rng.standard_normal((N1,)).astype(np.float32) * 0.1
    seqused_kv = np.array([S2] * B, dtype=np.int32)

    # ---- golden (unpaged) ----
    out, lse = golden_swa(q, kv, sinks, S1, S2, ORI_WIN_LEFT, ORI_WIN_RIGHT, SOFTMAX_SCALE)

    # ---- paged (NPU input) ----
    kv_pa_f32, block_table = to_paged(kv, BLOCK_SIZE)

    # ---- dtype conversion and write to disk ----
    def cast(x):
        if dtype_str == "fp16":
            return x.astype(np.float16)
        # bf16: take the high 16 bits of float32
        u32 = x.astype(np.float32).view(np.uint32)
        return ((u32 + 0x8000) >> 16).astype(np.uint16)

    cast(q).tofile(os.path.join(DATA_DIR, "q.bin"))
    cast(kv_pa_f32).tofile(os.path.join(DATA_DIR, "kv_pa.bin"))
    sinks.astype(np.float32).tofile(os.path.join(DATA_DIR, "sinks.bin"))
    block_table.tofile(os.path.join(DATA_DIR, "block_table.bin"))
    seqused_kv.tofile(os.path.join(DATA_DIR, "seqused_kv.bin"))
    cast(out).tofile(os.path.join(DATA_DIR, "golden_out.bin"))
    lse.astype(np.float32).tofile(os.path.join(DATA_DIR, "golden_lse.bin"))

    # ---- meta info ----
    with open(os.path.join(DATA_DIR, "meta.txt"), "w") as f:
        f.write(f"dtype={dtype_str}\n")
        f.write(f"B={B} S1={S1} S2={S2} N1={N1} KV_N={KV_N} D={D}\n")
        f.write(f"BLOCK_SIZE={BLOCK_SIZE} block_num={kv_pa_f32.shape[0]} max_blocks={block_table.shape[1]}\n")
        f.write(f"softmax_scale={SOFTMAX_SCALE}\n")
        f.write(f"ori_win_left={ORI_WIN_LEFT} ori_win_right={ORI_WIN_RIGHT}\n")

    print(f"[gen] dtype={dtype_str} q{q.shape} kv_pa{kv_pa_f32.shape} block_table{block_table.shape}")
    print(f"[gen] out{out.shape} lse{lse.shape} -> {DATA_DIR}")


if __name__ == "__main__":
    main()