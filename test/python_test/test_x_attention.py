#!/usr/bin/env python3
# Copyright 2025 The xLLM Authors. All Rights Reserved.
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
import numpy as np
import random
import copy
import pytest
import torch

from dataclasses import dataclass
from ml_dtypes import bfloat16


torch_npu = pytest.importorskip("torch_npu")
custom_ops = pytest.importorskip("custom_ops")

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(1)


def gen_seqlen(max_q_seqlen: int, max_kv_seqlen: int, is_varied_len: int, batch: int):
    q_seqlen_list = []
    kv_seqlen_list = []
    if is_varied_len == 0:
        q_seqlen_list = [max_q_seqlen] * batch
        kv_seqlen_list = [max_kv_seqlen] * batch
    else:
        for _ in range(batch):
            q_seq = random.randint(1, max_q_seqlen)
            kv_seq = random.randint(1, max_kv_seqlen)
            q_seqlen_list.append(q_seq)
            kv_seqlen_list.append(kv_seq)
    return q_seqlen_list, kv_seqlen_list


class TestFlashAttentionInfer:
    @dataclass
    class AttentionInputs:
        # [bs, 1, headnum, headdim]
        query: np.ndarray
        # [num_blocks, block_size, kv_head, headdim]
        key_cache: np.ndarray
        value_cache: np.ndarray
        # [bs (request_num * beam_size), kv_head, max_decode_step, headdim]
        unshared_k: np.ndarray
        # [bs (request_num * beam_size), kv_head, max_decode_step, headdim]
        unshared_v: np.ndarray
        # [bs, max_blocks_per_batch] max_blocks_per_batch = ceil(max_shared_kvlen / block_size)
        block_tables: list
        # [request_num] (1, 1, 1)
        q_seqlen_list: list
        # [request_num] (share_len1, share_len2)
        k_seqlen_list: list
        global_mask: any
        mask_type: int
        shape_param: any

    @dataclass
    class GenDataParams:
        q_seqlen_list: list
        k_seqlen_list: list
        beam_size: int
        unshared_kvlen: int
        num_heads: int
        kv_heads: int
        head_size: int
        num_blocks: int
        block_size: int
        mask_type: int
        dtype: any
        kv_dtype: int

    @classmethod
    def group_matmul(cls, head, kv_head, left, right, right_row=None, right_col=None):
        group_num = head // kv_head
        score = None
        for i in range(kv_head):
            current_right = right[i : (i + 1), :, :] if right_row is None else right[i : (i + 1), :right_row, :right_col]
            group_score = np.matmul(left[i * group_num : (i + 1) * group_num, :, :].astype(np.float32), current_right.astype(np.float32))
            if score is None:
                score = group_score
            else:
                score = np.concatenate((score, group_score), 0)
        return score

    @classmethod
    def softmax_numpy(cls, sim):
        row_max = np.max(sim, axis=-1, keepdims=True)
        sim_sub = sim - row_max
        sim_sub = np.exp(sim_sub)
        row_sum = np.sum(sim_sub, axis=-1, keepdims=True)
        soft_res = sim_sub  # no div rowsum
        return soft_res, row_max, row_sum

    def ref_masked_attention(self,
        query,  # (q_seqlen, num_heads, head_size)
        key,    # (k_seqlen, kv_heads, head_size)
        value,
        scale: float,
        mask    # (q_seqlen, k_seqlen)
    ):
        # Q * K.T
        query = np.transpose(query, (1, 0, 2))
        key = np.transpose(key, (1, 2, 0))
        sim_high = self.group_matmul(query.shape[0], key.shape[0], query, key)  # (head_num, q_seqlen, k_seqlen)
        sim_high = sim_high * scale
        # softmax
        p_high, gm, gl = self.softmax_numpy(sim_high)
        p = p_high.astype(query.dtype)
        p_high = p_high.astype(np.float32)
        value = np.transpose(value, (1, 0, 2))

        out_high = self.group_matmul(query.shape[0], key.shape[0], p_high, value)
        out = self.group_matmul(query.shape[0], key.shape[0], p, value)
        out_high = np.transpose(out_high, (1, 0, 2))
        out = np.transpose(out, (1, 0, 2))
        out = out.astype(query.dtype)
        return out, out_high, gm, gl

    def ref_single_query_unshared_kv_attention(self,
        attention_inputs: "TestFlashAttentionInfer.AttentionInputs",
        output: np.ndarray,
        true_out: np.ndarray,
        unshared_gl: np.ndarray,
        unshared_gm: np.ndarray,
    ) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size = attention_inputs.shape_param.head_size
        beam_size = attention_inputs.shape_param.beam_size
        request_num = len(attention_inputs.q_seqlen_list)
        batch = beam_size * request_num
        decode_step = attention_inputs.shape_param.unshared_kvlen
        max_decode_step = attention_inputs.unshared_k.shape[2]

        assert attention_inputs.query.shape == (batch, num_heads, head_size)
        assert attention_inputs.unshared_k.shape == (batch, kv_heads, max_decode_step, head_size)
        assert attention_inputs.unshared_v.shape == (batch, kv_heads, max_decode_step, head_size)

        scale = 1.0 / (head_size ** 0.5)

        for i in range(batch):
            q = attention_inputs.query[i : i + 1, :, :]
            k = attention_inputs.unshared_k[i, :, :, :]
            v = attention_inputs.unshared_v[i, :, :, :]
            # Transpose for group_matmul
            q_t = np.transpose(q, (1, 0, 2))
            k_t = np.transpose(k, (0, 2, 1))

            sim = self.group_matmul(num_heads, kv_heads, q_t, k_t, head_size, decode_step)  # [num_heads, 1, unshared_kvlen]
            sim = sim * scale

            # Softmax with stats
            p, gm, gl = self.softmax_numpy(sim)
            gm = np.transpose(gm, (1, 0, 2))  # (q_seqlen, num_heads, 1)
            gl = np.transpose(gl, (1, 0, 2))
            p_high = p.astype(np.float32)
            out_high = self.group_matmul(num_heads, kv_heads, p_high, v, decode_step, head_size)
            out_high = np.transpose(out_high, (1, 0, 2))
            p_low = p.astype(attention_inputs.query.dtype)
            out_low = self.group_matmul(num_heads, kv_heads, p_low, v, decode_step, head_size)
            out_low = np.transpose(out_low, (1, 0, 2))
            out_low = out_low.astype(attention_inputs.query.dtype)

            # Write outputs
            output[i : i + 1, :, :] = out_low
            true_out[i : i + 1, :, :] = out_high

            unshared_gm[i, :, :] = gm[:, :, :]
            unshared_gl[i, :, :] = gl[:, :, :]

    def ref_single_query_shared_kv_attention(
        self,
        attention_inputs: "TestFlashAttentionInfer.AttentionInputs",
        output,
        true_out,
        shared_gl,
        shared_gm,
    ) -> None:
        num_heads = attention_inputs.shape_param.num_heads
        kv_heads = attention_inputs.shape_param.kv_heads
        head_size_qk = attention_inputs.shape_param.head_size
        head_size_vo = attention_inputs.shape_param.head_size
        block_size = attention_inputs.shape_param.block_size
        beam_size = attention_inputs.shape_param.beam_size
        request_num = len(attention_inputs.shape_param.q_seqlen_list)
        cu_seqlen = 0
        layout = "TND"

        for i in range(request_num):
            q_seqlen = int(beam_size)
            k_seqlen = int(attention_inputs.k_seqlen_list[i])
            if layout == "TND":
                q = attention_inputs.query[cu_seqlen : (cu_seqlen + q_seqlen), :, :]
            elif layout == "BSND":
                q = attention_inputs.query[i, :, :, :]
            keys = []
            values = []
            block_table = attention_inputs.block_tables[i]
            for j in range(k_seqlen):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = attention_inputs.key_cache[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_qk)
                keys.append(k)

                v = attention_inputs.value_cache[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_vo)
                values.append(v)
            keys = np.stack(keys, axis=0)
            values = np.stack(values, axis=0)
            scale = 1.0 / (head_size_qk ** 0.5)
            mask = None
            out, out_high, gm, gl = self.ref_masked_attention(q, keys, values, scale, mask)
            out = out.reshape(-1, num_heads, head_size_vo)
            out_high = out_high.reshape(-1, num_heads, head_size_vo)
            gm = np.transpose(gm, (1, 0, 2))  # (q_seqlen, num_heads, 1)
            gl = np.transpose(gl, (1, 0, 2))
            output[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out
            true_out[cu_seqlen : cu_seqlen + q_seqlen, :, :] = out_high
            shared_gl[cu_seqlen : cu_seqlen + q_seqlen, :, :] = gl
            shared_gm[cu_seqlen : cu_seqlen + q_seqlen, :, :] = gm
            cu_seqlen += q_seqlen

    def call_device_op(self, q, k, v, unshared_k, unshared_v, block_tables, actual_shared_kvlen, decode_step):        
        def _to_npu(arr):
            if bfloat16 is not None and isinstance(arr, np.ndarray) and arr.dtype == bfloat16:
                # torch.from_numpy doesn't support numpy bfloat16; go via float32 then cast
                return torch.tensor(copy.deepcopy(arr).astype(np.float32), dtype=torch.bfloat16).npu()
            return torch.from_numpy(copy.deepcopy(arr)).npu()

        q = _to_npu(q)
        q = q.unsqueeze(1)  # (bs * beam_size, 1, head_num, head_dim)
        k = _to_npu(k)
        v = _to_npu(v)
        unshared_k = _to_npu(unshared_k)
        unshared_v = _to_npu(unshared_v)
        block_tables = torch.tensor(copy.deepcopy(block_tables), dtype=torch.int32).npu()
        actual_shared_kvlen = torch.tensor(actual_shared_kvlen, dtype=torch.int32).npu()
        decode_step_tensor = torch.tensor([decode_step], dtype=torch.int32).npu()
        attn_out = custom_ops.x_attention_npu(
            q, k, v, unshared_k, unshared_v, block_tables, actual_shared_kvlen, decode_step_tensor
        )
        return attn_out

    def calc_data(self, gen_data_params: "TestFlashAttentionInfer.GenDataParams"):
        head_size_qk = gen_data_params.head_size
        head_size_vo = gen_data_params.head_size
        q_min_range = -1.0
        q_max_range = 1.0
        kv_min_range = -1.0
        kv_max_range = 1.0
        beam_size = gen_data_params.beam_size
        request_num = len(gen_data_params.k_seqlen_list)
        decode_step = gen_data_params.unshared_kvlen
        max_decode_step = 3

        num_tokens = np.array(gen_data_params.q_seqlen_list).sum() * beam_size
        num_shared_kv = np.array(gen_data_params.k_seqlen_list).sum()

        batch_size = request_num * beam_size
        query = np.random.uniform(
            q_min_range, q_max_range, size=(num_tokens, gen_data_params.num_heads, head_size_qk)
        ).astype(gen_data_params.dtype)
        max_k_seqlen = max(gen_data_params.k_seqlen_list)
        block_tables = []  # (num_tokens, max_num_blocks_per_seq）
        key_cache = None
        value_cache = None
        if gen_data_params.kv_dtype == 1:
            key_cache = np.random.uniform(
                kv_min_range,
                kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads, head_size_qk),
            ).astype(gen_data_params.dtype)

            value_cache = np.random.uniform(
                kv_min_range,
                kv_max_range,
                size=(gen_data_params.num_blocks, gen_data_params.block_size, gen_data_params.kv_heads, head_size_vo),
            ).astype(gen_data_params.dtype)
            max_num_blocks_per_seq = (max_k_seqlen + gen_data_params.block_size - 1) // gen_data_params.block_size
            for i in range(request_num):
                block_table = [max_num_blocks_per_seq * i + j for j in range(max_num_blocks_per_seq)]
                block_tables.append(block_table)

            # Unshared KV
            unshared_key = np.random.uniform(
                kv_min_range,
                kv_max_range,
                size=(batch_size, gen_data_params.kv_heads, max_decode_step, head_size_qk),
            ).astype(gen_data_params.dtype)
            unshared_value = np.random.uniform(
                kv_min_range,
                kv_max_range,
                size=(batch_size, gen_data_params.kv_heads, max_decode_step, head_size_vo),
            ).astype(gen_data_params.dtype)

        shape_out = (num_tokens, gen_data_params.num_heads, head_size_vo)
        sum_max_shape_out = (num_tokens, gen_data_params.num_heads, 1)
        shared_ref_out = np.zeros(shape_out, dtype=gen_data_params.dtype)
        shared_true_out = np.zeros(shape_out, dtype=np.float32)
        shared_gl = np.zeros(sum_max_shape_out, dtype=np.float32)
        shared_gm = np.zeros(sum_max_shape_out, dtype=np.float32)

        unshared_ref_out = np.zeros(shape_out, dtype=gen_data_params.dtype)
        unshared_true_out = np.zeros(shape_out, dtype=np.float32)
        unshared_gl = np.zeros(sum_max_shape_out, dtype=np.float32)
        unshared_gm = np.zeros(sum_max_shape_out, dtype=np.float32)

        attention_inputs = self.AttentionInputs(
            query,
            key_cache,
            value_cache,
            unshared_key,
            unshared_value,
            block_tables,
            gen_data_params.q_seqlen_list,
            gen_data_params.k_seqlen_list,
            None,
            gen_data_params.mask_type,
            gen_data_params,
        )

        self.ref_single_query_shared_kv_attention(
            attention_inputs, shared_ref_out, shared_true_out, shared_gl, shared_gm
        )

        self.ref_single_query_unshared_kv_attention(
            attention_inputs, unshared_ref_out, unshared_true_out, unshared_gl, unshared_gm
        )

        gm = np.maximum(shared_gm, unshared_gm)
        update_shared_expgm = np.exp(shared_gm - gm)
        update_unshared_expgm = np.exp(unshared_gm - gm)
        gl = shared_gl * update_shared_expgm + unshared_gl * update_unshared_expgm
        tmp_shared_true = shared_true_out * update_shared_expgm
        tmp_unshared_true = unshared_true_out * update_unshared_expgm
        tmp_add = tmp_shared_true + tmp_unshared_true
        final_true_out = tmp_add / gl

        npu_res = self.call_device_op(
            query, key_cache, value_cache, unshared_key, unshared_value, block_tables, gen_data_params.k_seqlen_list, decode_step
        )
        golden_res = torch.from_numpy(final_true_out).unsqueeze(1)
        npu_res = npu_res.cpu().float()
        assert torch.allclose(npu_res, golden_res, atol=0.001, rtol=0.001)

@pytest.mark.parametrize("dtype", [np.float16])
def test_x_attention_npu(dtype):
    # Device selection (skip if no NPU available)
    try:
        torch_npu.npu.set_device(0)
    except Exception as e:
        pytest.skip(f"NPU device not available: {e}")

    request = 1
    beam_size = 64
    q_seqlen = 1  # must be 1
    kv_seqlen = 256  # shared_kv_len
    unshared_seqlen = 1
    num_head = 16
    kv_heads = 16
    embedding_size = 128
    block_size = 128
    is_varied_len = 0
    mask_type = 0
    kv_dtype = 1

    q_seqlen_list, kv_seqlen_list = gen_seqlen(q_seqlen, kv_seqlen, is_varied_len, request)
    max_kv_seqlen = max(kv_seqlen_list)
    num_blocks = request * ((max_kv_seqlen + block_size - 1) // block_size)

    test_obj = TestFlashAttentionInfer()
    gen_data_params = test_obj.GenDataParams(
        q_seqlen_list,
        kv_seqlen_list,
        beam_size,
        unshared_seqlen,
        num_head,
        kv_heads,
        embedding_size,
        num_blocks,
        block_size,
        mask_type,
        dtype,
        kv_dtype,
    )
    test_obj.calc_data(gen_data_params)
