#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compressor(AICORE MIX_AIC) fused-op golden + BSH data generation.

Scenario (priority: non-OVERLAP, direct semantics):
  {cmp_ratio=128, coff=1, head_dim=512}  (one of the 3 legal combos of CheckScenarioConsistency)
  B=1, S=128 -> Sr=ceil(S/cmp_ratio)=1, hidden_size=1024 (512-aligned, in [1024,10240]),
  start_pos=0, block_size>=S -> no state history dependency (head/tail holders all 0, no r/w on kv/score_state).

Fused formula (precisely locked from compressor_block_cube.h ComputeMm1 + block_vec.h Vec1/Vec2):
  coffD = coff*head_dim = 512
  kv    = x @ wkv^T                        # (S,hidden) @ (coffD,hidden)^T -> (S, coffD)
  score = x @ wgate^T                      # (S, coffD)
  score = score + ape                      # ape=(cmp_ratio, coffD), add positional encoding per row within group
  # softmax within group (cmp_ratio rows) along "row/group" dim (each column independent); coff=1 => ReduceSize=cmp_ratio
  p     = softmax(score, axis=row-within-group)   # (cmp_ratio, coffD) per group
  cmp   = sum_within_group(p * kv)         # (Sr, coffD) weighted-sum compression within group
  norm  = RmsNorm(cmp, norm_weight(head_dim), eps)   # per-row rmsnorm
  out   = HALF-rope (on last rope_head_dim=64 dims), first head_dim-64 dims unchanged
  cmp_kv = out.reshape(B, Sr, head_dim)    # BSH
"""
import os
import sys
import numpy as np

DATA_DIR = "/tmp/compressor_data"

# ------------------ case params (non-OVERLAP minimal scenario) ------------------
B = 1
S = 128
CMP_RATIO = 128
COFF = 1
HEAD_DIM = 512
COFF_D = COFF * HEAD_DIM        # 512
HIDDEN = 1024                   # in [1024,10240] and 512-aligned
ROPE_HEAD_DIM = 64
NORM_EPS = 1e-6
BLOCK_SIZE = 128                # >= S, single block holds all
SR = (S + CMP_RATIO - 1) // CMP_RATIO   # 1


def rms_norm(x, weight, eps):
    """x:(rows, head_dim) weight:(head_dim,) per-row rmsnorm"""
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * weight


def half_rope(x, cos, sin):
    """
    HALF mode (rope.h RotaryPosEmb MODE==HALF): only acts on last rope_head_dim dims per row.
    x:(rows, head_dim); cos/sin:(rows, rope_head_dim)
    rotate_half(v) = [-v[half:], v[:half]]  (half = rope_head_dim/2)
    out_rope = v*cos + rotate_half(v)*sin
    first head_dim-rope_head_dim dims unchanged.
    """
    out = x.astype(np.float32).copy()
    r = ROPE_HEAD_DIM
    base = HEAD_DIM - r
    v = out[:, base:base + r]
    half = r // 2
    rot = np.concatenate([-v[:, half:], v[:, :half]], axis=-1)
    out[:, base:base + r] = v * cos + rot * sin
    return out


def golden(x, wkv, wgate, ape, norm_weight, rope_cos, rope_sin):
    """
    x:(B,S,hidden) wkv/wgate:(coffD,hidden) ape:(cmp_ratio,coffD)
    norm_weight:(head_dim) rope_cos/sin:(B,Sr,rope_head_dim)
    return cmp_kv:(B,Sr,head_dim) float32
    """
    cmp_kv = np.zeros((B, SR, HEAD_DIM), dtype=np.float32)
    for b in range(B):
        kv = x[b] @ wkv.T            # (S, coffD)
        score = x[b] @ wgate.T       # (S, coffD)
        for g in range(SR):
            r0 = g * CMP_RATIO
            r1 = min(r0 + CMP_RATIO, S)
            n = r1 - r0
            sc = score[r0:r1] + ape[:n]           # (n, coffD)
            kvg = kv[r0:r1]                        # (n, coffD)
            # softmax within group (along row dim, each column independent)
            m = np.max(sc, axis=0, keepdims=True)
            e = np.exp(sc - m)
            p = e / np.sum(e, axis=0, keepdims=True)   # (n, coffD)
            cmp = np.sum(p * kvg, axis=0, keepdims=True)  # (1, coffD)
            norm = rms_norm(cmp, norm_weight, NORM_EPS)   # (1, head_dim) coffD==head_dim
            out = half_rope(norm, rope_cos[b, g:g + 1], rope_sin[b, g:g + 1])
            cmp_kv[b, g] = out[0]
    return cmp_kv


def main():
    dtype_str = sys.argv[1] if len(sys.argv) > 1 else "fp16"
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(2025)

    x = (rng.standard_normal((B, S, HIDDEN)) * 0.1).astype(np.float32)
    wkv = (rng.standard_normal((COFF_D, HIDDEN)) * 0.05).astype(np.float32)
    wgate = (rng.standard_normal((COFF_D, HIDDEN)) * 0.05).astype(np.float32)
    ape = (rng.standard_normal((CMP_RATIO, COFF_D)) * 0.1).astype(np.float32)
    norm_weight = (rng.standard_normal((HEAD_DIM,)) * 0.1 + 1.0).astype(np.float32)
    rope_cos = (rng.standard_normal((B, SR, ROPE_HEAD_DIM)) * 0.1).astype(np.float32)
    rope_sin = (rng.standard_normal((B, SR, ROPE_HEAD_DIM)) * 0.1).astype(np.float32)

    # state inputs (zero-initialized in-place; not used in minimal scenario)
    # kv_state/score_state shape: (block_num, block_size, coffD) paged; block_num=1 in minimal scenario
    block_num = 1
    kv_state = np.zeros((block_num, BLOCK_SIZE, COFF_D), dtype=np.float32)
    score_state = np.zeros((block_num, BLOCK_SIZE, COFF_D), dtype=np.float32)

    # block_table: (batchSize, maxBlockNumPerBatch) int32, points to physical block index
    # maxBlockNumPerBatch = ceil(S/BLOCK_SIZE) = 1; batch 0 uses block 0
    max_block = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_block_table = np.zeros((B, max_block), dtype=np.int32)
    score_block_table = np.zeros((B, max_block), dtype=np.int32)

    gold = golden(x, wkv, wgate, ape, norm_weight, rope_cos, rope_sin)

    def cast(a):
        if dtype_str == "fp16":
            return a.astype(np.float16)
        u32 = a.astype(np.float32).view(np.uint32)
        return ((u32 + 0x8000) >> 16).astype(np.uint16)

    cast(x).tofile(os.path.join(DATA_DIR, "x.bin"))
    cast(wkv).tofile(os.path.join(DATA_DIR, "wkv.bin"))
    cast(wgate).tofile(os.path.join(DATA_DIR, "wgate.bin"))
    ape.astype(np.float32).tofile(os.path.join(DATA_DIR, "ape.bin"))
    kv_state.tofile(os.path.join(DATA_DIR, "kv_state.bin"))
    score_state.tofile(os.path.join(DATA_DIR, "score_state.bin"))
    kv_block_table.tofile(os.path.join(DATA_DIR, "kv_block_table.bin"))
    score_block_table.tofile(os.path.join(DATA_DIR, "score_block_table.bin"))
    cast(norm_weight).tofile(os.path.join(DATA_DIR, "norm_weight.bin"))
    cast(rope_cos).tofile(os.path.join(DATA_DIR, "rope_cos.bin"))
    cast(rope_sin).tofile(os.path.join(DATA_DIR, "rope_sin.bin"))
    cast(gold).tofile(os.path.join(DATA_DIR, "golden_cmp_kv.bin"))

    with open(os.path.join(DATA_DIR, "meta.txt"), "w") as f:
        f.write(f"dtype={dtype_str}\n")
        f.write(f"B={B} S={S} SR={SR} HIDDEN={HIDDEN} HEAD_DIM={HEAD_DIM} COFF_D={COFF_D}\n")
        f.write(f"CMP_RATIO={CMP_RATIO} COFF={COFF} ROPE_HEAD_DIM={ROPE_HEAD_DIM}\n")
        f.write(f"BLOCK_SIZE={BLOCK_SIZE} block_num={block_num} NORM_EPS={NORM_EPS}\n")

    print(f"[gen] dtype={dtype_str} x{x.shape} wkv{wkv.shape} ape{ape.shape}")
    print(f"[gen] cmp_kv(golden){gold.shape} -> {DATA_DIR}")


if __name__ == "__main__":
    main()