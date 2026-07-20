import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

INVALID_IDX = -1
HEAD_DIM = 128  # kernel hard constraint: query last dim must be 128


# lightning_indexer_quant golden (CPU fp32 reference).
#   query : int8  [B, S1, N, D]     (N = qHeadNum, D = 128)
#   key   : int8  [B, S2, 1, D]     (keyHeadNum must be 1)
#   weights      : fp16 [B, S1, N]
#   query_scale  : fp16 [B, S1, N]  (per query token & head)
#   key_scale    : fp16 [B, S2, 1]  (per key token)
#   score[b,s1,s2] = sum_n( weights[b,s1,n] * qScale[b,s1,n] * kScale[b,s2,0]
#                           * sum_d( q[b,s1,n,d] * k[b,s2,0,d] ) )
#   sparse_mode 0 -> all s2 valid; 3 -> causal, s2 in [0, s1].
#   output int32 [B, S1, 1, sparse_count]: topk (descending) s2 indices,
#   padded with -1 when valid count < sparse_count.
def lightning_indexer_quant_golden(query, key, weights, query_scale,
                                   key_scale, sparse_count, sparse_mode):
    B, S1, N, D = query.shape
    S2 = key.shape[1]
    q = query.to(torch.int64)
    k = key.to(torch.int64)[:, :, 0, :]                 # [B, S2, D]
    # int matmul: [B, S1, N, S2]
    qk = torch.einsum("bsnd,btd->bsnt", q, k).to(torch.float32)
    w = weights.to(torch.float32)                        # [B, S1, N]
    qs = query_scale.to(torch.float32)                   # [B, S1, N]
    ks = key_scale.to(torch.float32)[:, :, 0]            # [B, S2]
    coeff = (w * qs).unsqueeze(-1)                       # [B, S1, N, 1]
    weighted = (qk * coeff).sum(dim=2)            # [B, S1, S2]
    score = weighted * ks.unsqueeze(1)                   # [B, S1, S2]

    out = torch.full((B, S1, 1, sparse_count), INVALID_IDX, dtype=torch.int32)
    for b in range(B):
        for s1 in range(S1):
            valid = S2 if sparse_mode == 0 else min(s1 + 1, S2)
            row = score[b, s1, :valid]
            order = torch.argsort(row, descending=True)
            take = min(sparse_count, valid)
            out[b, s1, 0, :take] = order[:take].to(torch.int32)
    return out


def _valid_set(row):
    return set(int(x) for x in row.tolist() if int(x) != INVALID_IDX)


# (B, S1, S2, N)  N = qHeadNum, keyHeadNum fixed to 1, D fixed to 128.
CASES = [
    (1, 8, 8, 1),
    (1, 16, 16, 2),
    (2, 8, 8, 4),
    (1, 32, 32, 8),
]


@pytest.mark.parametrize("sparse_mode", [0, 3])
@pytest.mark.parametrize("B,S1,S2,N", CASES)
def test_lightning_indexer_quant(B, S1, S2, N, sparse_mode):
    torch.manual_seed(2026)
    D = HEAD_DIM
    # sparse_count >= S2 so every valid key is selected -> compare index sets
    # (avoids topk tie-break ambiguity on ordering).
    sparse_count = S2

    query = torch.randint(-8, 8, (B, S1, N, D), dtype=torch.int8)
    key = torch.randint(-8, 8, (B, S2, 1, D), dtype=torch.int8)
    weights = (torch.rand(B, S1, N) * 2 - 1).to(torch.float16)
    query_scale = (torch.rand(B, S1, N) * 0.5 + 0.1).to(torch.float16)
    key_scale = (torch.rand(B, S2, 1) * 0.5 + 0.1).to(torch.float16)

    ref = lightning_indexer_quant_golden(
        query, key, weights, query_scale, key_scale, sparse_count, sparse_mode)

    out = custom_ops.lightning_indexer_quant_npu(
        query.npu(), key.npu(), weights.npu(),
        query_scale.npu(), key_scale.npu(),
        query_quant_mode=0, key_quant_mode=0,
        layout_query="BSND", layout_key="BSND",
        sparse_count=sparse_count, sparse_mode=sparse_mode)
    out = out.cpu()

    assert out.shape == ref.shape, f"{out.shape} vs {ref.shape}"
    for b in range(B):
        for s1 in range(S1):
            got = _valid_set(out[b, s1, 0])
            exp = _valid_set(ref[b, s1, 0])
            assert got == exp, (
                f"mode={sparse_mode} b={b} s1={s1}: got {got} exp {exp}")