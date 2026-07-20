# -*- coding: utf-8 -*-
# Generate quant_lightning_indexer(PA_BSND) inputs + CPU golden(sparse_indices) written to .bin.
# Kernel hard constraint: layout_key only supports PA_BSND (paged KV).
# golden.forward uses unpaged key_bnsd; NPU uses paged key[block_num,block_size,kH,hd]+block_table.
import sys, types, os, math
for name in ("test", "custom_ops"):
    sys.modules[name] = types.ModuleType(name)

GOLDEN_DIR = "/export/home/weinan5/zhaoxingcheng/xllm_ops/xllm_ops/attention/quant_lightning_indexer/tests/pytest"
sys.path.insert(0, GOLDEN_DIR)

import numpy as np
import torch
import quant_lightning_indexer_golden as G

OUT = "/tmp/quant_lightning_indexer_data"
os.makedirs(OUT, exist_ok=True)
np.random.seed(2026); torch.manual_seed(2026)

# ---- Case (PA_BSND): B=1, q_seq=4, k_seq=128(=1 block), block_size=128 ----
batch_size = 1
q_seq = 4
block_size = 128
k_seq = 128
q_head_num = 64
k_head_num = 1
head_dim = 128
block_num = 1
qk_dtype = torch.int8
dequant_dtype = torch.float16
actual_seq_dtype = torch.int32
cmp_ratio = 1
act_seq_q = [q_seq]
act_seq_k = [k_seq * cmp_ratio]
query_quant_mode = 0
key_quant_mode = 0
layout_query = "BSND"
layout_key = "PA_BSND"
sparse_count = 8
sparse_mode = 3
q_t_size = q_seq; k_t_size = k_seq

qr = [-100, 100]; kr = [-100, 100]; wr = [-25, 25]
qsr = [0, 255]; ksr = [0, 65504]

indexer_op = G.GeneralizedQLI(batch_size, q_seq, k_seq, q_t_size, k_t_size,
                       q_head_num, k_head_num, head_dim, block_size, block_num,
                       qk_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k,
                       query_quant_mode, key_quant_mode, layout_query, layout_key,
                       sparse_count, sparse_mode, cmp_ratio)

# query / weights / q_scale (BSND)
query = torch.tensor(np.random.uniform(qr[0], qr[1], (batch_size, q_seq, q_head_num, head_dim))).to(qk_dtype)
weights = torch.tensor(np.random.uniform(wr[0], wr[1], (batch_size, q_seq, q_head_num))).to(dequant_dtype)
q_scale = torch.tensor(np.random.uniform(qsr[0], qsr[1], (batch_size, q_seq, q_head_num))).to(dequant_dtype)

# key_bnsd (for golden) : [B, kH, k_max_s2, hd]
k_max_s2 = math.floor(max(act_seq_k) / cmp_ratio)
key_bnsd = torch.tensor(np.random.uniform(kr[0], kr[1], (batch_size, k_head_num, k_max_s2, head_dim))).to(qk_dtype)
k_scale_bns = torch.tensor(np.random.uniform(ksr[0], ksr[1], (batch_size, k_head_num, k_max_s2))).to(dequant_dtype)

aslq = torch.tensor(act_seq_q).to(actual_seq_dtype)
aslk = torch.tensor(act_seq_k).to(actual_seq_dtype)

# ---- Build block_table + paged key / k_scale (for NPU) ----
k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size)
key_block_num_per_batch = []
key_block_num_sum = 0
for cur_act_k in act_seq_k:
    cur_cmp = math.floor(cur_act_k / cmp_ratio)
    n = math.ceil(cur_cmp / block_size)
    key_block_num_per_batch.append(n); key_block_num_sum += n
assert block_num >= key_block_num_sum, "block_num too small"

block_id_list = np.arange(block_num).astype(np.int32)  # no permutation, keep deterministic
block_table = np.full((batch_size, k_max_block_num_per_batch), -1, dtype=np.int32)
cur = 0
for bi, thr in enumerate(key_block_num_per_batch):
    for ib in range(thr):
        block_table[bi][ib] = block_id_list[cur]; cur += 1

# paged key: [block_num, block_size, kH, hd]
key_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim), dtype=qk_dtype)
key_expand[:, :, :k_max_s2, :] = key_bnsd
key_pa = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype=qk_dtype)
for ib_ in range(batch_size):
    for i_block, cbid in enumerate(block_table[ib_]):
        if cbid == -1: continue
        sp = i_block * block_size
        for i_n in range(k_head_num):
            key_pa[cbid, :, i_n, :] = key_expand[ib_, i_n, sp:sp + block_size, :]

ks_expand = torch.zeros((batch_size, k_head_num,k_max_block_num_per_batch * block_size), dtype=dequant_dtype)
ks_expand[:, :, :k_max_s2] = k_scale_bns
kscale_pa = torch.zeros((block_num, block_size, k_head_num), dtype=dequant_dtype)
for ib_ in range(batch_size):
    for i_block, cbid in enumerate(block_table[ib_]):
        if cbid == -1: continue
        sp = i_block * block_size
        for i_n in range(k_head_num):
            kscale_pa[cbid, :, i_n] = ks_expand[ib_, i_n, sp:sp + block_size]

def save(name, t, np_dtype):
    arr = np.ascontiguousarray(t.cpu().numpy().astype(np_dtype))
    arr.tofile(os.path.join(OUT, name))
    print(f"{name}: shape={arr.shape} dtype={arr.dtype} bytes={arr.nbytes}")

save("query.bin", query, np.int8)
save("key.bin", key_pa, np.int8)                 # paged key
save("weights.bin", weights, np.float16)
save("q_scale.bin", q_scale, np.float16)
save("k_scale.bin", kscale_pa, np.float16)       # paged k_scale
save("aslq.bin", aslq, np.int32)
save("aslk.bin", aslk, np.int32)
save("block_table.bin", torch.from_numpy(block_table), np.int32)

# CPU golden (uses unpaged key_bnsd / k_scale_bns)
y, y_value = indexer_op.forward(query, key_bnsd, weights, q_scale, k_scale_bns, aslq, aslk, torch.from_numpy(block_table))
save("golden_indices.bin", y, np.int32)
print("golden y shape:", tuple(y.shape))
for s in range(q_seq):
    print(f"golden y[0,{s},0,:]=", y[0, s, 0, :].tolist())

with open(os.path.join(OUT, "meta.txt"), "w") as f:
    f.write(f"batch_size={batch_size}\nq_seq={q_seq}\nk_seq={k_seq}\n")
    f.write(f"q_head_num={q_head_num}\nk_head_num={k_head_num}\nhead_dim={head_dim}\n")
    f.write(f"block_size={block_size}\nblock_num={block_num}\n")
    f.write(f"k_max_block_num_per_batch={k_max_block_num_per_batch}\n")
    f.write(f"sparse_count={sparse_count}\nsparse_mode={sparse_mode}\ncmp_ratio={cmp_ratio}\n")
    f.write(f"query_quant_mode={query_quant_mode}\nkey_quant_mode={key_quant_mode}\n")
print("meta written. DONE.")