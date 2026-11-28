#!/usr/bin/env python3
# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

HEAD_DIM = 128
HALF_DIM = 64


def split_qkv_norm_rope_torch(qkv, q_gamma, k_gamma, cos, sin, num_q_heads,
                              num_kv_heads, eps=1e-6, gqa_expand=False,
                              layout_bnsd=False, rope_dim=None):
    """PyTorch reference for SplitQkvNormRope.

    Args:
        qkv: [B, S, stride] with q/k/v heads packed contiguously at row start.
        q_gamma/k_gamma: [D] RMSNorm gamma for q/k heads. D is derived from
            q_gamma (single source, same as host tiling).
        cos/sin: [B, S, rope_stride], only the first rope_dim/2 entries are
            used for rotation (NeoX half-split layout).
        num_q_heads/num_kv_heads: head counts; v heads == kv heads.
        rope_dim: None/0 = full-head rotation; otherwise only the first
            rope_dim dims are rotated (rope_dim/2 per half, pair-wise with the
            matching second-half chunk), the rest pass through after
            RMSNorm*gamma.
    Returns:
        (q, k, v) in BSND [B,S,H*D] or BNSD [B,H,S,D].
    """
    dtype = qkv.dtype
    b, s, _ = qkv.shape
    d = q_gamma.shape[-1]
    half = d // 2
    rd = (d if rope_dim is None or rope_dim == 0 else rope_dim) // 2
    num_v_heads = num_kv_heads
    used = (num_q_heads + num_kv_heads + num_v_heads) * d

    x = qkv[..., :used].float()
    q = x[..., :num_q_heads * d].reshape(b, s, num_q_heads, d)
    k_head = x[..., num_q_heads * d:(num_q_heads + num_kv_heads) * d]
    k = k_head.reshape(b, s, num_kv_heads, d)
    v_head = x[..., (num_q_heads + num_kv_heads) * d:]
    v = v_head.reshape(b, s, num_v_heads, d)

    def rms_norm(t, gamma):
        var = t.pow(2).mean(dim=-1, keepdim=True)
        return t * torch.rsqrt(var + eps) * gamma.float().view(1, 1, 1, d)

    def rope_neox(t, c, sn):
        if rd == half:
            t1 = t[..., :half]
            t2 = t[..., half:]
            return torch.cat([t1 * c - t2 * sn, t2 * c + t1 * sn], dim=-1)
        # partial rope: rotate pair-wise first rd of each half, rest passthrough
        t1 = t[..., :rd]
        t2 = t[..., half:half + rd]
        p1 = t[..., rd:half]
        p2 = t[..., half + rd:]
        r1 = t1 * c - t2 * sn
        r2 = t2 * c + t1 * sn
        return torch.cat([r1, p1, r2, p2], dim=-1)

    c = cos[..., :rd].float().unsqueeze(2)
    sn = sin[..., :rd].float().unsqueeze(2)

    q = rope_neox(rms_norm(q, q_gamma), c, sn)
    k = rope_neox(rms_norm(k, k_gamma), c, sn)

    if gqa_expand and num_q_heads != num_kv_heads:
        rep = num_q_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

    def to_layout(t):
        t = t.to(dtype)
        return t.transpose(1, 2).contiguous() if layout_bnsd else t.reshape(
            b, s, -1).contiguous()

    return to_layout(q), to_layout(k), to_layout(v)


def _get_device():
    """Get NPU device, skip test if unavailable."""
    try:
        device_id = int(os.getenv("NPU_DEVICE_ID",
                                  os.getenv("ASCEND_DEVICE_ID", "0")))
        torch_npu.npu.set_device(device_id)
        return True
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"NPU device not available: {e}")
        return False


# (batch, seq, nq, nkv, stride_tail, rope_stride, gqa_expand, layout_bnsd, head_dim)
CASES = [
    (1, 512, 6, 2, 6, 64, False, False, 128),   # JoyOV2 DiT main scenario
    (1, 128, 64, 16, 0, 64, False, False, 128),  # Nq+Nk=80 > 64 (dynamic sumsq bound)
    (1, 512, 6, 2, 6, 64, False, True, 128),    # BNSD layout
    (1, 512, 6, 2, 6, 64, True, False, 128),    # GQA inline expand
    (1, 512, 6, 2, 0, 64, False, False, 128),   # no tail columns
    (1, 1, 6, 2, 6, 64, False, False, 128),     # single token (refiner)
    (1, 7, 6, 2, 6, 64, False, False, 128),     # tail tile, S not multiple of 8
    (1, 259, 8, 2, 0, 64, False, False, 128),   # text-side 2 kv heads
    (2, 64, 6, 2, 6, 64, False, False, 128),    # B > 1
    (1, 128, 6, 2, 6, 128, False, False, 128),  # full-width rope table
    (1, 128, 8, 1, 0, 64, False, False, 128),   # Nk = Nv = 1
    (1, 128, 6, 2, 6, 64, True, True, 128),     # BNSD + gqa_expand
    # --- D = 256（Tier 1：VF chunk/pair 循环，nPairs=2）---
    (1, 512, 6, 2, 6, 128, False, False, 256),  # D=256 main scenario
    (1, 512, 6, 2, 6, 128, False, True, 256),   # D=256 BNSD
    (1, 512, 6, 2, 6, 128, True, False, 256),   # D=256 GQA inline expand
    (1, 1, 6, 2, 6, 128, False, False, 256),    # D=256 single token
    (1, 7, 6, 2, 6, 128, False, False, 256),    # D=256 tail tile
    (2, 64, 6, 2, 6, 128, False, False, 256),   # D=256 B > 1
    (1, 128, 8, 1, 0, 128, False, False, 256),  # D=256 Nk = Nv = 1
    # --- D = 64（Tier 2：HalfMasked 计数掩码路径，half=32 单 masked pair）---
    (1, 512, 6, 2, 6, 32, False, False, 64),   # D=64 main scenario
    (1, 512, 6, 2, 6, 32, False, True, 64),    # D=64 BNSD
    (1, 512, 6, 2, 6, 32, True, False, 64),    # D=64 GQA inline expand
    (1, 1, 6, 2, 6, 32, False, False, 64),     # D=64 single token
    (1, 7, 6, 2, 6, 32, False, False, 64),     # D=64 tail tile
    (1, 64, 6, 2, 6, 64, False, False, 64),    # D=64 full-width rope table
    (1, 128, 8, 1, 0, 32, False, False, 64),   # D=64 Nk = Nv = 1
    # --- D = 96（Tier 2：HalfMasked，half=48，pass1 尾块 32）---
    (1, 512, 6, 2, 6, 48, False, False, 96),   # D=96 main scenario
    (1, 512, 6, 2, 6, 48, False, True, 96),    # D=96 BNSD
    (1, 512, 6, 2, 6, 48, True, False, 96),    # D=96 GQA inline expand
    (1, 1, 6, 2, 6, 48, False, False, 96),     # D=96 single token
    (1, 7, 6, 2, 6, 48, False, False, 96),     # D=96 tail tile
    (1, 96, 6, 2, 6, 96, False, False, 96),    # D=96 full-width rope table
    (1, 128, 8, 1, 0, 48, False, False, 96),   # D=96 Nk = Nv = 1
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,nq,nkv,tail,rope_stride,gqa_expand,layout_bnsd,head_dim", CASES)
def test_split_qkv_norm_rope(batch, seq, nq, nkv, tail, rope_stride,
                             gqa_expand, layout_bnsd, head_dim, dtype):
    if not _get_device():
        return

    torch.manual_seed(20261106)
    eps = 1e-6
    half_dim = head_dim // 2
    stride = (nq + 2 * nkv) * head_dim + tail

    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)
    k_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)

    inv_freq = 1.0 / (10000.0**(torch.arange(
        0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    freqs = pos * inv_freq.view(1, 1, -1)
    cos = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    sin = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    cos[..., :half_dim] = freqs.cos().expand(batch, seq, half_dim)
    sin[..., :half_dim] = freqs.sin().expand(batch, seq, half_dim)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    q_ref, k_ref, v_ref = split_qkv_norm_rope_torch(
        qkv, q_gamma, k_gamma, cos, sin, nq, nkv, eps, gqa_expand, layout_bnsd)

    q_npu, k_npu, v_npu = custom_ops.split_qkv_norm_rope_npu(
        qkv.npu(),
        q_gamma.npu(),
        k_gamma.npu(),
        cos.npu(),
        sin.npu(),
        nq,
        nkv,
        eps,
        gqa_expand,
        layout_bnsd,
    )
    torch.npu.synchronize()

    if dtype == torch.float16:
        atol = rtol = 2e-3
    else:
        atol = rtol = 2e-2

    assert tuple(q_npu.shape) == tuple(q_ref.shape)
    assert tuple(k_npu.shape) == tuple(k_ref.shape)
    assert tuple(v_npu.shape) == tuple(v_ref.shape)

    torch.testing.assert_close(
        q_npu.cpu().float(), q_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        k_npu.cpu().float(), k_ref.float(), atol=atol, rtol=rtol)
    # v is a pure copy of the input columns, so it must match bit-exactly.
    torch.testing.assert_close(
        v_npu.cpu().float(), v_ref.float(), atol=0, rtol=0)


def _to_interleaved(qkv, num_q_heads, num_kv_heads, d=HEAD_DIM):
    """Repack ``[Q | K... | V... | tail]`` -> ``[Q | K0 V0 K1 V1 ... | tail]``."""
    b, s, _ = qkv.shape
    q_end = num_q_heads * d
    kv_end = q_end + 2 * num_kv_heads * d
    q = qkv[..., :q_end]
    k = qkv[..., q_end:q_end + num_kv_heads * d].reshape(
        b, s, num_kv_heads, d)
    v = qkv[..., q_end + num_kv_heads * d:kv_end].reshape(
        b, s, num_kv_heads, d)
    kv_interleaved = torch.stack([k, v], dim=3).reshape(
        b, s, 2 * num_kv_heads * d)
    return torch.cat([q, kv_interleaved, qkv[..., kv_end:]], dim=-1).contiguous()


# (batch, seq, nq, nkv, stride_tail, rope_stride, gqa_expand, layout_bnsd, head_dim)
INTERLEAVED_CASES = [
    (1, 512, 6, 2, 6, 64, False, False, 128),   # main scenario, K_iV_i interleaved
    (1, 512, 6, 2, 6, 64, True, False, 128),    # interleaved + GQA expand
    (1, 7, 6, 2, 6, 64, False, False, 128),     # interleaved tail tile
    (2, 64, 6, 2, 6, 64, False, True, 128),     # interleaved + BNSD + B > 1
    (1, 512, 6, 2, 6, 128, False, False, 256),  # D=256 interleaved main
    (1, 512, 6, 2, 6, 128, True, False, 256),   # D=256 interleaved + GQA expand
    (1, 512, 6, 2, 6, 32, False, False, 64),   # D=64 interleaved main
    (1, 512, 6, 2, 6, 32, True, False, 64),    # D=64 interleaved + GQA expand
    (1, 512, 6, 2, 6, 48, False, False, 96),   # D=96 interleaved main
    (1, 512, 6, 2, 6, 48, True, False, 96),    # D=96 interleaved + GQA expand
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,nq,nkv,tail,rope_stride,gqa_expand,layout_bnsd,head_dim", INTERLEAVED_CASES)
def test_split_qkv_norm_rope_kv_interleaved(
        batch, seq, nq, nkv, tail, rope_stride, gqa_expand, layout_bnsd,
        head_dim, dtype):
    if not _get_device():
        return

    torch.manual_seed(20261106)
    eps = 1e-6
    half_dim = head_dim // 2
    stride = (nq + 2 * nkv) * head_dim + tail

    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)
    k_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)

    inv_freq = 1.0 / (10000.0**(torch.arange(
        0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    freqs = pos * inv_freq.view(1, 1, -1)
    cos = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    sin = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    cos[..., :half_dim] = freqs.cos().expand(batch, seq, half_dim)
    sin[..., :half_dim] = freqs.sin().expand(batch, seq, half_dim)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    # Reference consumes the block layout; the kernel consumes the interleaved one.
    q_ref, k_ref, v_ref = split_qkv_norm_rope_torch(
        qkv, q_gamma, k_gamma, cos, sin, nq, nkv, eps, gqa_expand, layout_bnsd)
    qkvi = _to_interleaved(qkv, nq, nkv, head_dim)

    q_npu, k_npu, v_npu = custom_ops.split_qkv_norm_rope_npu(
        qkvi.npu(),
        q_gamma.npu(),
        k_gamma.npu(),
        cos.npu(),
        sin.npu(),
        nq,
        nkv,
        eps,
        gqa_expand,
        layout_bnsd,
        True,  # kv_interleaved
    )
    torch.npu.synchronize()

    if dtype == torch.float16:
        atol = rtol = 2e-3
    else:
        atol = rtol = 2e-2

    assert tuple(q_npu.shape) == tuple(q_ref.shape)
    assert tuple(k_npu.shape) == tuple(k_ref.shape)
    assert tuple(v_npu.shape) == tuple(v_ref.shape)

    torch.testing.assert_close(
        q_npu.cpu().float(), q_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        k_npu.cpu().float(), k_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        v_npu.cpu().float(), v_ref.float(), atol=0, rtol=0)


# 宽头数用例：验证 host 按 UB 预算自动降 tokenTile（8 -> 4 -> 2）
# (batch, seq, nq, nkv, rope_stride, gqa_expand, layout_bnsd, kv_interleaved)
WIDE_CASES = [
    # (batch, seq, nq, nkv, rope_stride, gqa_expand, layout_bnsd, kv_interleaved, head_dim)
    (1, 256, 16, 8, 64, True, False, True, 128),    # headTotal=32, stride=4096 -> tile 4
    (1, 256, 32, 8, 64, True, False, True, 128),    # headTotal=48, stride=6144 -> tile 4
    (1, 256, 48, 16, 64, True, False, True, 128),   # headTotal=80, stride=10288 -> tile 2
    (1, 256, 16, 8, 64, False, False, False, 128),  # 块状布局 + 大 head 数
    (1, 256, 16, 8, 128, True, False, True, 256),   # D=256 headTotal=32 -> tile 4（行宽翻倍降档）
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,nq,nkv,rope_stride,gqa_expand,layout_bnsd,kv_interleaved,head_dim", WIDE_CASES)
def test_split_qkv_norm_rope_wide_heads(
        batch, seq, nq, nkv, rope_stride, gqa_expand, layout_bnsd,
        kv_interleaved, head_dim, dtype):
    if not _get_device():
        return

    torch.manual_seed(20261106)
    eps = 1e-6
    half_dim = head_dim // 2
    stride = (nq + 2 * nkv) * head_dim

    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)
    k_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)

    inv_freq = 1.0 / (10000.0**(torch.arange(
        0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    freqs = pos * inv_freq.view(1, 1, -1)
    cos = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    sin = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    cos[..., :half_dim] = freqs.cos().expand(batch, seq, half_dim)
    sin[..., :half_dim] = freqs.sin().expand(batch, seq, half_dim)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    q_ref, k_ref, v_ref = split_qkv_norm_rope_torch(
        qkv, q_gamma, k_gamma, cos, sin, nq, nkv, eps, gqa_expand, layout_bnsd)
    qkvi = _to_interleaved(qkv, nq, nkv, head_dim) if kv_interleaved else qkv

    q_npu, k_npu, v_npu = custom_ops.split_qkv_norm_rope_npu(
        qkvi.npu(), q_gamma.npu(), k_gamma.npu(), cos.npu(), sin.npu(),
        nq, nkv, eps, gqa_expand, layout_bnsd, kv_interleaved)
    torch.npu.synchronize()

    if dtype == torch.float16:
        atol = rtol = 2e-3
    else:
        atol = rtol = 2e-2

    assert tuple(q_npu.shape) == tuple(q_ref.shape)
    assert tuple(k_npu.shape) == tuple(k_ref.shape)
    assert tuple(v_npu.shape) == tuple(v_ref.shape)
    torch.testing.assert_close(
        q_npu.cpu().float(), q_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        k_npu.cpu().float(), k_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        v_npu.cpu().float(), v_ref.float(), atol=0, rtol=0)


# 部分 rope（Tier 3）：D=256 时仅旋转前 rope_dim=128 维，其余经 norm*gamma 直通。
# head_dim=256、rope_dim=128 固定；变化 (batch, seq, nq, nkv, tail, gqa, bnsd, kv_int)
# (batch, seq, nq, nkv, stride_tail, gqa_expand, layout_bnsd, kv_interleaved)
PARTIAL_ROPE_CASES = [
    (1, 512, 6, 2, 6, False, False, False),   # D=256 rope_dim=128 main
    (1, 512, 6, 2, 6, True, False, False),    # + GQA inline expand
    (1, 1, 6, 2, 6, False, False, False),     # single token
    (1, 7, 6, 2, 6, False, False, False),     # tail tile
    (1, 512, 6, 2, 6, False, True, False),    # BNSD
    (1, 512, 6, 2, 6, False, False, True),    # kv_interleaved
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,nq,nkv,tail,gqa_expand,layout_bnsd,kv_interleaved", PARTIAL_ROPE_CASES)
def test_split_qkv_norm_rope_partial_rope(
        batch, seq, nq, nkv, tail, gqa_expand, layout_bnsd, kv_interleaved,
        dtype):
    if not _get_device():
        return

    torch.manual_seed(20261106)
    eps = 1e-6
    head_dim = 256
    rope_dim = 128
    half_dim = head_dim // 2
    rope_half = rope_dim // 2
    stride = (nq + 2 * nkv) * head_dim + tail

    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)
    k_gamma = (1.0 + 0.1 * torch.randn(head_dim, dtype=torch.float32)).to(dtype)

    # inv_freq 按参与旋转的 rope_dim 生成（只取前 rope_half 项）
    inv_freq = 1.0 / (10000.0**(torch.arange(
        0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    pos = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    freqs = pos * inv_freq.view(1, 1, -1)
    cos = torch.zeros(batch, seq, half_dim, dtype=torch.float32)
    sin = torch.zeros(batch, seq, half_dim, dtype=torch.float32)
    cos[..., :rope_half] = freqs.cos().expand(batch, seq, rope_half)
    sin[..., :rope_half] = freqs.sin().expand(batch, seq, rope_half)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    q_ref, k_ref, v_ref = split_qkv_norm_rope_torch(
        qkv, q_gamma, k_gamma, cos, sin, nq, nkv, eps, gqa_expand, layout_bnsd,
        rope_dim)
    qkvi = _to_interleaved(qkv, nq, nkv, head_dim) if kv_interleaved else qkv

    q_npu, k_npu, v_npu = custom_ops.split_qkv_norm_rope_npu(
        qkvi.npu(), q_gamma.npu(), k_gamma.npu(), cos.npu(), sin.npu(),
        nq, nkv, eps, gqa_expand, layout_bnsd, kv_interleaved, rope_dim)
    torch.npu.synchronize()

    if dtype == torch.float16:
        atol = rtol = 2e-3
    else:
        atol = rtol = 2e-2

    assert tuple(q_npu.shape) == tuple(q_ref.shape)
    assert tuple(k_npu.shape) == tuple(k_ref.shape)
    assert tuple(v_npu.shape) == tuple(v_ref.shape)
    torch.testing.assert_close(
        q_npu.cpu().float(), q_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        k_npu.cpu().float(), k_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        v_npu.cpu().float(), v_ref.float(), atol=0, rtol=0)


# 拒绝路径：host 校验应返回参数错误而不是回绕/静默越界
# (nq, nkv, batch, seq, rope_stride, eps, q_gamma_dim, k_gamma_dim, desc)
# 注：epsilon 负值校验暂无法在 aclnn eager 路径覆盖——生成的
# NnopbaseAddAttrWithDtype(kNnopbaseFloat) 会把 Float 属性槽写成 0，
# tiling 读到的永远是 0（见文档"已知问题"）。验证代码保留，待框架修复后补用例。
# 注：负头数两条用例在 eager 路径由 torch 的 at::empty 负维校验先行拒绝
# （Negative dimensions are not supported），不会触达 tiling 的 int64 守卫；
# 守卫作为图模式/非 eager 入口的纵深防御保留。
REJECT_CASES = [
    (-1, 2, 1, 8, 64, 1e-6, 128, 128, 0, "negative num_q_heads rejected (torch shape check before tiling)"),
    (6, -2, 1, 8, 64, 1e-6, 128, 128, 0, "negative num_kv_heads rejected (torch shape check before tiling)"),
    (6, 2, 1, 8, 65, 1e-6, 128, 128, 0, "rope_stride must be 16-aligned for LoadAlign"),
    (6, 2, 1, 8, 64, 1e-6, 512, 512, 0, "head_dim 512 outside support set {64,96,128,256}"),
    (6, 2, 1, 8, 64, 1e-6, 72, 72, 0, "head_dim 72 outside support set {64,96,128,256}"),
    (6, 2, 1, 8, 64, 1e-6, 192, 192, 0, "head_dim 192 outside support set {64,96,128,256}"),
    (6, 2, 1, 8, 64, 1e-6, 128, 256, 0, "k_gamma dim must match q_gamma"),
    (6, 2, 1, 8, 128, 1e-6, 256, 256, 192, "rope_dim must be a multiple of 128"),
    (6, 2, 1, 8, 128, 1e-6, 256, 256, 512, "rope_dim must not exceed head_dim"),
    (6, 2, 1, 8, 32, 1e-6, 64, 64, 32, "rope_dim must equal head_dim for D<128"),
]


@pytest.mark.parametrize(
    "nq,nkv,batch,seq,rope_stride,eps,q_gamma_dim,k_gamma_dim,rope_dim,desc", REJECT_CASES)
def test_split_qkv_norm_rope_reject_invalid(
        nq, nkv, batch, seq, rope_stride, eps, q_gamma_dim, k_gamma_dim,
        rope_dim, desc):
    if not _get_device():
        return

    torch.manual_seed(20261117)
    dtype = torch.bfloat16
    stride = (6 + 2 * 2) * HEAD_DIM
    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = torch.ones(q_gamma_dim, dtype=dtype)
    k_gamma = torch.ones(k_gamma_dim, dtype=dtype)
    cos = torch.zeros(batch, seq, rope_stride, dtype=dtype)
    sin = torch.zeros(batch, seq, rope_stride, dtype=dtype)

    with pytest.raises((RuntimeError, AssertionError)):
        custom_ops.split_qkv_norm_rope_npu(
            qkv.npu(), q_gamma.npu(), k_gamma.npu(), cos.npu(), sin.npu(),
            nq, nkv, eps, True, False, False, rope_dim)
        torch.npu.synchronize()
