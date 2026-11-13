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
                              layout_bnsd=False):
    """PyTorch reference for SplitQkvNormRope.

    Args:
        qkv: [B, S, stride] with q/k/v heads packed contiguously at row start.
        q_gamma/k_gamma: [D] RMSNorm gamma for q/k heads.
        cos/sin: [B, S, rope_stride], only the first HALF_DIM entries are used
            (NeoX half-split layout).
        num_q_heads/num_kv_heads: head counts; v heads == kv heads.
    Returns:
        (q, k, v) in BSND [B,S,H*D] or BNSD [B,H,S,D].
    """
    dtype = qkv.dtype
    b, s, _ = qkv.shape
    num_v_heads = num_kv_heads
    used = (num_q_heads + num_kv_heads + num_v_heads) * HEAD_DIM

    x = qkv[..., :used].float()
    q = x[..., :num_q_heads * HEAD_DIM].reshape(b, s, num_q_heads, HEAD_DIM)
    k_head = x[..., num_q_heads * HEAD_DIM:(num_q_heads + num_kv_heads) * HEAD_DIM]
    k = k_head.reshape(b, s, num_kv_heads, HEAD_DIM)
    v_head = x[..., (num_q_heads + num_kv_heads) * HEAD_DIM:]
    v = v_head.reshape(b, s, num_v_heads, HEAD_DIM)

    def rms_norm(t, gamma):
        var = t.pow(2).mean(dim=-1, keepdim=True)
        return t * torch.rsqrt(var + eps) * gamma.float().view(1, 1, 1, HEAD_DIM)

    def rope_half(t, c, sn):
        t1 = t[..., :HALF_DIM]
        t2 = t[..., HALF_DIM:]
        return torch.cat([t1 * c - t2 * sn, t2 * c + t1 * sn], dim=-1)

    c = cos[..., :HALF_DIM].float().unsqueeze(2)
    sn = sin[..., :HALF_DIM].float().unsqueeze(2)

    q = rope_half(rms_norm(q, q_gamma), c, sn)
    k = rope_half(rms_norm(k, k_gamma), c, sn)

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


# (batch, seq, nq, nkv, stride_tail, rope_stride, gqa_expand, layout_bnsd)
CASES = [
    (1, 512, 6, 2, 6, 64, False, False),   # JoyOV2 DiT main scenario
    (1, 512, 6, 2, 6, 64, False, True),    # BNSD layout
    (1, 512, 6, 2, 6, 64, True, False),    # GQA inline expand
    (1, 512, 6, 2, 0, 64, False, False),   # no tail columns
    (1, 1, 6, 2, 6, 64, False, False),     # single token (refiner)
    (1, 7, 6, 2, 6, 64, False, False),     # tail tile, S not multiple of 8
    (1, 259, 8, 2, 0, 64, False, False),   # text-side 2 kv heads
    (2, 64, 6, 2, 6, 64, False, False),    # B > 1
    (1, 128, 6, 2, 6, 128, False, False),  # full-width rope table
    (1, 128, 8, 1, 0, 64, False, False),   # Nk = Nv = 1
    (1, 128, 6, 2, 6, 64, True, True),     # BNSD + gqa_expand
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,nq,nkv,tail,rope_stride,gqa_expand,layout_bnsd", CASES)
def test_split_qkv_norm_rope(batch, seq, nq, nkv, tail, rope_stride,
                             gqa_expand, layout_bnsd, dtype):
    if not _get_device():
        return

    torch.manual_seed(20261106)
    eps = 1e-6
    stride = (nq + 2 * nkv) * HEAD_DIM + tail

    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = (1.0 + 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32)).to(dtype)
    k_gamma = (1.0 + 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32)).to(dtype)

    inv_freq = 1.0 / (10000.0**(torch.arange(
        0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    freqs = pos * inv_freq.view(1, 1, -1)
    cos = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    sin = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    cos[..., :HALF_DIM] = freqs.cos().expand(batch, seq, HALF_DIM)
    sin[..., :HALF_DIM] = freqs.sin().expand(batch, seq, HALF_DIM)
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


def _to_interleaved(qkv, num_q_heads, num_kv_heads):
    """Repack ``[Q | K... | V... | tail]`` -> ``[Q | K0 V0 K1 V1 ... | tail]``."""
    b, s, _ = qkv.shape
    q_end = num_q_heads * HEAD_DIM
    kv_end = q_end + 2 * num_kv_heads * HEAD_DIM
    q = qkv[..., :q_end]
    k = qkv[..., q_end:q_end + num_kv_heads * HEAD_DIM].reshape(
        b, s, num_kv_heads, HEAD_DIM)
    v = qkv[..., q_end + num_kv_heads * HEAD_DIM:kv_end].reshape(
        b, s, num_kv_heads, HEAD_DIM)
    kv_interleaved = torch.stack([k, v], dim=3).reshape(
        b, s, 2 * num_kv_heads * HEAD_DIM)
    return torch.cat([q, kv_interleaved, qkv[..., kv_end:]], dim=-1).contiguous()


# (batch, seq, nq, nkv, stride_tail, rope_stride, gqa_expand, layout_bnsd)
INTERLEAVED_CASES = [
    (1, 512, 6, 2, 6, 64, False, False),   # main scenario, K_iV_i interleaved
    (1, 512, 6, 2, 6, 64, True, False),    # interleaved + GQA expand
    (1, 7, 6, 2, 6, 64, False, False),     # interleaved tail tile
    (2, 64, 6, 2, 6, 64, False, True),     # interleaved + BNSD + B > 1
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,nq,nkv,tail,rope_stride,gqa_expand,layout_bnsd", INTERLEAVED_CASES)
def test_split_qkv_norm_rope_kv_interleaved(
        batch, seq, nq, nkv, tail, rope_stride, gqa_expand, layout_bnsd, dtype):
    if not _get_device():
        return

    torch.manual_seed(20261106)
    eps = 1e-6
    stride = (nq + 2 * nkv) * HEAD_DIM + tail

    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = (1.0 + 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32)).to(dtype)
    k_gamma = (1.0 + 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32)).to(dtype)

    inv_freq = 1.0 / (10000.0**(torch.arange(
        0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    freqs = pos * inv_freq.view(1, 1, -1)
    cos = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    sin = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    cos[..., :HALF_DIM] = freqs.cos().expand(batch, seq, HALF_DIM)
    sin[..., :HALF_DIM] = freqs.sin().expand(batch, seq, HALF_DIM)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    # Reference consumes the block layout; the kernel consumes the interleaved one.
    q_ref, k_ref, v_ref = split_qkv_norm_rope_torch(
        qkv, q_gamma, k_gamma, cos, sin, nq, nkv, eps, gqa_expand, layout_bnsd)
    qkvi = _to_interleaved(qkv, nq, nkv)

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
    (1, 256, 16, 8, 64, True, False, True),    # headTotal=32, stride=4096 -> tile 4
    (1, 256, 32, 8, 64, True, False, True),    # headTotal=48, stride=6144 -> tile 4
    (1, 256, 48, 16, 64, True, False, True),   # headTotal=80, stride=10288 -> tile 2
    (1, 256, 16, 8, 64, False, False, False),  # 块状布局 + 大 head 数
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,seq,nq,nkv,rope_stride,gqa_expand,layout_bnsd,kv_interleaved", WIDE_CASES)
def test_split_qkv_norm_rope_wide_heads(
        batch, seq, nq, nkv, rope_stride, gqa_expand, layout_bnsd,
        kv_interleaved, dtype):
    if not _get_device():
        return

    torch.manual_seed(20261106)
    eps = 1e-6
    stride = (nq + 2 * nkv) * HEAD_DIM

    qkv = (torch.randn(batch, seq, stride, dtype=torch.float32) * 0.5).to(dtype)
    q_gamma = (1.0 + 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32)).to(dtype)
    k_gamma = (1.0 + 0.1 * torch.randn(HEAD_DIM, dtype=torch.float32)).to(dtype)

    inv_freq = 1.0 / (10000.0**(torch.arange(
        0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    pos = torch.arange(seq, dtype=torch.float32).view(1, seq, 1)
    freqs = pos * inv_freq.view(1, 1, -1)
    cos = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    sin = torch.zeros(batch, seq, rope_stride, dtype=torch.float32)
    cos[..., :HALF_DIM] = freqs.cos().expand(batch, seq, HALF_DIM)
    sin[..., :HALF_DIM] = freqs.sin().expand(batch, seq, HALF_DIM)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    q_ref, k_ref, v_ref = split_qkv_norm_rope_torch(
        qkv, q_gamma, k_gamma, cos, sin, nq, nkv, eps, gqa_expand, layout_bnsd)
    qkvi = _to_interleaved(qkv, nq, nkv) if kv_interleaved else qkv

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
