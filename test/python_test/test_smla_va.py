# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the va project

import pytest
import torch
import torch_npu  # noqa: F401

import custom_ops_lib  # noqa: F401


@pytest.fixture(autouse=True)
def _require_a3():
    if not torch.npu.is_available():
        pytest.skip("smla_va requires an A3 NPU")
    soc = torch_npu._C._npu_get_soc_version()
    if not 250 < soc < 260:
        pytest.skip(f"smla_va is A3 only (SoC {soc})")


def test_band_visible_lengths_require_one_row_per_query():
    num_tokens, num_heads, head_dim, block_size = 2, 16, 512, 128
    band_mask_mode = 4
    query = torch.zeros(
        (num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device="npu"
    )
    kv = torch.ones((1, block_size, 1, head_dim), dtype=torch.bfloat16, device="npu")
    starts = torch.tensor([0, num_tokens], dtype=torch.int32, device="npu")
    lengths = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    visible = torch.ones((num_tokens, 1), dtype=torch.int32, device="npu")
    metadata = torch.ops.xllm_ops.smla_va_metadata(
        num_heads,
        1,
        head_dim,
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        batch_size=1,
        max_seqlen_q=num_tokens,
        max_seqlen_ori_kv=num_tokens,
        ori_mask_mode=band_mask_mode,
        ori_win_left=block_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_block_table=torch.tensor([[0]], dtype=torch.int32, device="npu"),
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        sinks=torch.zeros(num_heads, dtype=torch.float32, device="npu"),
        metadata=metadata,
        softmax_scale=head_dim**-0.5,
        ori_mask_mode=band_mask_mode,
        ori_win_left=block_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    output, _ = torch.ops.xllm_ops.smla_va(query, ori_topk_length=visible, **kwargs)
    torch.npu.synchronize()
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
    # Zero queries give equal weight to each visible KV row and the zero
    # attention sink. A one-row replay window must yield 1 / (1 + 1), even
    # for the second query; ignoring visible lengths would yield 2 / 3 there.
    torch.testing.assert_close(output, torch.full_like(output, 0.5))

    with pytest.raises(RuntimeError, match="aclnnSmlaVa failed"):
        torch.ops.xllm_ops.smla_va(query, ori_topk_length=visible[:1], **kwargs)
        torch.npu.synchronize()


def _sparse_case(tokens: int, heads: int, keys: int, picks: int):
    torch.manual_seed(42)
    dim, block = 512, 128
    q = (torch.randn(tokens, heads, dim) * 0.3).bfloat16().npu()
    kv = torch.randn((keys + block - 1) // block, block, 1, dim).bfloat16().npu()
    indices = (
        torch.stack([torch.randperm(keys)[:picks] for _ in range(tokens)]).int().npu()
    )
    indices[:, -1] = -1
    starts = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    lengths = torch.tensor([keys], dtype=torch.int32, device="npu")
    sinks = torch.linspace(-1, 1, heads, device="npu")
    valid_lengths = (indices >= 0).sum(-1, keepdim=True).int()
    metadata = torch.ops.xllm_ops.smla_va_metadata(
        heads,
        1,
        dim,
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        ori_topk_length=valid_lengths,
        batch_size=1,
        max_seqlen_q=tokens,
        max_seqlen_ori_kv=keys,
        ori_topk=picks,
        ori_mask_mode=0,
        ori_win_left=keys,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=False,
    )
    kwargs = dict(
        ori_kv=kv,
        ori_sparse_indices=indices.unsqueeze(1),
        ori_block_table=torch.arange(
            kv.shape[0], device="npu", dtype=torch.int32
        ).unsqueeze(0),
        cu_seqlens_q=starts,
        seqused_ori_kv=lengths,
        ori_topk_length=valid_lengths,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=dim**-0.5,
        ori_mask_mode=0,
        ori_win_left=keys,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    return q, kwargs, indices


def _reference(q, kwargs, indices):
    # Explicit indices reproduce the model's union of SWA and compressed rows.
    qc = q.cpu().float()
    kc = kwargs["ori_kv"].cpu().float().view(-1, 512)
    idx = indices.cpu().long()
    selected = kc[idx.clamp_min(0)]
    scores = torch.einsum("thd,tkd->thk", qc, selected) * 512**-0.5
    scores.masked_fill_(idx[:, None, :] < 0, float("-inf"))
    sinks = kwargs["sinks"].cpu()[None, :, None].expand(q.size(0), -1, -1)
    probs = torch.softmax(torch.cat((scores, sinks), dim=-1), dim=-1)[..., :-1]
    return torch.einsum("thk,tkd->thd", probs, selected).bfloat16()


@pytest.mark.parametrize(
    "tokens,heads,keys,picks",
    [(1, 16, 128, 32), (6, 16, 256, 128), (33, 16, 1024, 640)],
)
def test_sparse_indices_match_reference(tokens, heads, keys, picks):
    q, kwargs, indices = _sparse_case(tokens, heads, keys, picks)
    output, _ = torch.ops.xllm_ops.smla_va(q, **kwargs)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), _reference(q, kwargs, indices), rtol=0.02, atol=0.003
    )


def test_sparse_indices_graph_replay_uses_live_values():
    q, kwargs, indices = _sparse_case(1, 16, 256, 128)
    for _ in range(3):
        torch.ops.xllm_ops.smla_va(q, **kwargs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output, _ = torch.ops.xllm_ops.smla_va(q, **kwargs)
    q.mul_(0.5)
    # Keep valid indices packed, as required by the DSpark sparse path.
    indices[:, -33:] = -1
    kwargs["ori_topk_length"].fill_(95)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu(), _reference(q, kwargs, indices), rtol=0.02, atol=0.003
    )


@pytest.mark.parametrize(
    "tokens,context,ratio",
    [
        (1, 301, 2),
        (6, 33, 2),
        (33, 1025, 2),
        (33, 1025, 1),
        (1, 301, 0),
        (33, 1025, 0),
        (256, 301, 0),
    ],
)
def test_compressed_attention_matches_reference(tokens, context, ratio):
    torch.manual_seed(42)
    heads, dim, block, win = 16, 512, 128, 128
    ori_len = win + tokens
    q = (torch.randn(tokens, heads, dim) * 0.3).bfloat16().npu()
    ori = torch.randn((ori_len + block - 1) // block, block, 1, dim).bfloat16().npu()
    starts = torch.tensor([0, tokens], dtype=torch.int32, device="npu")
    ori_lens = torch.tensor([ori_len], dtype=torch.int32, device="npu")
    visible = (
        torch.arange(context - tokens + 1, context + 1, dtype=torch.int32, device="npu")
        .clamp(max=win)
        .view(-1, 1)
    )
    sinks = torch.linspace(-1, 1, heads, device="npu")
    kwargs = dict(
        ori_kv=ori,
        ori_block_table=torch.arange(
            ori.shape[0], device="npu", dtype=torch.int32
        ).view(1, -1),
        cu_seqlens_q=starts,
        seqused_ori_kv=ori_lens,
        ori_topk_length=visible,
        sinks=sinks,
        softmax_scale=dim**-0.5,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if ratio else 0,
        ori_win_left=win - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )
    meta_kwargs = dict(
        cu_seqlens_q=starts,
        seqused_ori_kv=ori_lens,
        ori_topk_length=visible,
        batch_size=1,
        max_seqlen_q=tokens,
        max_seqlen_ori_kv=ori_len,
        cmp_topk=512 if ratio else 0,
        cmp_ratio=ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3 if ratio else 0,
        ori_win_left=win - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
        has_ori_kv=True,
        has_cmp_kv=bool(ratio),
    )
    cmp = None
    if ratio:
        cmp_len = context // ratio
        cmp = (
            torch.randn((cmp_len + block - 1) // block, block, 1, dim).bfloat16().npu()
        )
        indices = torch.full((tokens, 1, 512), -1, dtype=torch.int32)
        for i in range(tokens):
            count = min((context - tokens + i + 1) // ratio, 512)
            indices[i, 0, :count] = torch.randperm((context - tokens + i + 1) // ratio)[
                :count
            ]
        indices = indices.npu()
        cmp_lens = torch.tensor([cmp_len], dtype=torch.int32, device="npu")
        residual = torch.tensor([context % ratio], dtype=torch.int32, device="npu")
        kwargs.update(
            cmp_kv=cmp,
            cmp_sparse_indices=indices,
            cmp_block_table=torch.arange(
                cmp.shape[0], device="npu", dtype=torch.int32
            ).view(1, -1),
            seqused_cmp_kv=cmp_lens,
            cmp_residual_kv=residual,
        )
        meta_kwargs.update(
            seqused_cmp_kv=cmp_lens, cmp_residual_kv=residual, max_seqlen_cmp_kv=cmp_len
        )
    kwargs["metadata"] = torch.ops.xllm_ops.smla_va_metadata(
        heads, 1, dim, **meta_kwargs
    )
    output, _ = torch.ops.xllm_ops.smla_va(q, **kwargs)
    torch.npu.synchronize()
    refs = []
    ori_cpu = ori.cpu().float().view(-1, dim)
    for i in range(tokens):
        end = win + i + 1
        selected = ori_cpu[end - int(visible[i].item()) : end]
        if cmp is not None:
            idx = indices[i, 0].cpu().long()
            selected = torch.cat(
                (selected, cmp.cpu().float().view(-1, dim)[idx[idx >= 0]])
            )
        scores = torch.einsum("hd,kd->hk", q[i].cpu().float(), selected) * dim**-0.5
        probs = torch.softmax(torch.cat((scores, sinks.cpu()[:, None]), -1), -1)[:, :-1]
        refs.append((probs @ selected).bfloat16())
    torch.testing.assert_close(output.cpu(), torch.stack(refs), rtol=0.02, atol=0.003)
