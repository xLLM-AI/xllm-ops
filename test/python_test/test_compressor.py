import pytest
import numpy as np
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")


# ---- Golden (CPU fp32 reference, inlined from compressor_gen.py) ----
def rms_norm(x, weight, eps):
    """x:(rows, head_dim) weight:(head_dim,) per-row rmsnorm"""
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * weight


def half_rope(x, cos, sin, head_dim, rope_head_dim):
    """HALF mode: only acts on last rope_head_dim dims per row."""
    out = x.astype(np.float32).copy()
    r = rope_head_dim
    base = head_dim - r
    v = out[:, base:base + r]
    half = r // 2
    rot = np.concatenate([-v[:, half:], v[:, :half]], axis=-1)
    out[:, base:base + r] = v * cos + rot * sin
    return out


def compressor_golden(x, wkv, wgate, ape, norm_weight, rope_cos, rope_sin,
                      cmp_ratio, head_dim, rope_head_dim, norm_eps):
    """
    x:(B,S,hidden) wkv/wgate:(coffD,hidden) ape:(cmp_ratio,coffD)
    norm_weight:(head_dim) rope_cos/sin:(B,Sr,rope_head_dim)
    return cmp_kv:(B,Sr,head_dim) float32
    """
    B, S, _ = x.shape
    SR = (S + cmp_ratio - 1) // cmp_ratio
    cmp_kv = np.zeros((B, SR, head_dim), dtype=np.float32)
    for b in range(B):
        kv = x[b] @ wkv.T
        score = x[b] @ wgate.T
        for g in range(SR):
            r0 = g * cmp_ratio
            r1 = min(r0 + cmp_ratio, S)
            n = r1 - r0
            sc = score[r0:r1] + ape[:n]
            kvg = kv[r0:r1]
            m = np.max(sc, axis=0, keepdims=True)
            e = np.exp(sc - m)
            p = e / np.sum(e, axis=0, keepdims=True)
            cmp = np.sum(p * kvg, axis=0, keepdims=True)
            norm = rms_norm(cmp, norm_weight, norm_eps)
            out = half_rope(norm, rope_cos[b, g:g + 1], rope_sin[b, g:g + 1],
                            head_dim, rope_head_dim)
            cmp_kv[b, g] = out[0]
    return cmp_kv


# ---- Test cases ----
# Using minimal scenario: S=cmp_ratio so SR=1; single block holds all.
CASES = [
    # (B, S, HIDDEN, HEAD_DIM, CMP_RATIO, COFF, ROPE_HEAD_DIM, BLOCK_SIZE)
    (1, 128, 1024, 512, 128, 1, 64, 128),
]


@pytest.mark.parametrize("B,S,HIDDEN,HEAD_DIM,CMP_RATIO,COFF,ROPE_HD,BLOCK_SIZE", CASES)
def test_compressor(B, S, HIDDEN, HEAD_DIM, CMP_RATIO, COFF, ROPE_HD, BLOCK_SIZE):
    torch.manual_seed(2025)
    np.random.seed(2025)
    COFF_D = COFF * HEAD_DIM
    SR = (S + CMP_RATIO - 1) // CMP_RATIO
    NORM_EPS = 1e-6
    ROTARY_MODE = 1  # HALF

    # Generate inputs (fp32 reference)
    x_np = (np.random.randn(B, S, HIDDEN) * 0.1).astype(np.float32)
    wkv_np = (np.random.randn(COFF_D, HIDDEN) * 0.05).astype(np.float32)
    wgate_np = (np.random.randn(COFF_D, HIDDEN) * 0.05).astype(np.float32)
    ape_np = (np.random.randn(CMP_RATIO, COFF_D) * 0.1).astype(np.float32)
    norm_w_np = (np.random.randn(HEAD_DIM) * 0.1 + 1.0).astype(np.float32)
    rope_cos_np = (np.random.randn(B, SR, ROPE_HD) * 0.1).astype(np.float32)
    rope_sin_np = (np.random.randn(B, SR, ROPE_HD) * 0.1).astype(np.float32)

    # Golden
    ref = compressor_golden(x_np, wkv_np, wgate_np, ape_np, norm_w_np,
                            rope_cos_np, rope_sin_np,
                            CMP_RATIO, HEAD_DIM, ROPE_HD, NORM_EPS)

    # To half-precision tensors for NPU
    x_t = torch.from_numpy(x_np).half().npu()
    wkv_t = torch.from_numpy(wkv_np).half().npu()
    wgate_t = torch.from_numpy(wgate_np).half().npu()
    ape_t = torch.from_numpy(ape_np).float().npu()
    norm_w_t = torch.from_numpy(norm_w_np).half().npu()
    rope_sin_t = torch.from_numpy(rope_sin_np).half().npu()
    rope_cos_t = torch.from_numpy(rope_cos_np).half().npu()

    # In-place state (float32, paged)
    block_num = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_state_t = torch.zeros(block_num, BLOCK_SIZE, COFF_D, dtype=torch.float32).npu()
    score_state_t = torch.zeros(block_num, BLOCK_SIZE, COFF_D, dtype=torch.float32).npu()

    # Block tables: (B, maxBlock) int32
    max_block = block_num
    kv_bt = torch.zeros(B, max_block, dtype=torch.int32).npu()
    score_bt = torch.zeros(B, max_block, dtype=torch.int32).npu()

    out = custom_ops.compressor_npu(
        x_t, wkv_t, wgate_t, kv_state_t, score_state_t,
        ape_t, norm_w_t, rope_sin_t, rope_cos_t,
        kv_bt, score_bt,
        rope_head_dim=ROPE_HD, cmp_ratio=CMP_RATIO, coff=COFF,
        norm_eps=NORM_EPS, rotary_mode=ROTARY_MODE, enable_grad=False)

    out_np = out.cpu().float().numpy()
    ref_fp16 = ref.astype(np.float16).astype(np.float32)
    assert out_np.shape == ref_fp16.shape, f"{out_np.shape} vs {ref_fp16.shape}"

    atol, rtol = 2e-2, 2e-2
    diff = np.abs(out_np - ref_fp16)
    rel = diff / (np.abs(ref_fp16) + 1e-6)
    fail_count = int(np.sum((diff > atol) & (rel > rtol)))
    total = out_np.size
    assert fail_count == 0, (
        f"compressor mismatch: {fail_count}/{total} elems exceed tol, "
        f"maxAbs={diff.max():.5f} maxRel={rel.max():.5f}")