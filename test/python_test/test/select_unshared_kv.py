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

import torch
import torch_npu
import pdb
import custom_ops
import numpy as np

def select_unshared_kv(
    global_index,
    unshared_kv_cache,
    curr_kv,
    token_idxs,
    decode_step
):
    # 本轮token_idx存储至global_index
    global_index[:, :, decode_step - 1] = token_idxs.reshape([batch, beam_size])
    
    select_kv_cache = torch.zeros([batch*beam_size, head_num, decode_step, head_dim], dtype=unshared_kv_cache.dtype)
    # pdb.set_trace()
    # 循环向上查找
    for batch_idx in range(batch):
        prev_step_token_index = token_idxs[batch_idx*beam_size:batch_idx*beam_size+beam_size]
        for i in range(decode_step, 0, -1):
            # pdb.set_trace()
            real_idx = i - 1
            # 找出本轮的index 
            if i == decode_step:
                cur_step_token_index = prev_step_token_index
            else:
                cur_step_token_index = global_index[batch_idx, prev_step_token_index, real_idx]
                prev_step_token_index = cur_step_token_index
            # 根据index获取本轮真实kv
            batch_kv_cache = unshared_kv_cache[batch_idx * beam_size : batch_idx * beam_size + beam_size]
            cur_step_kv = batch_kv_cache[cur_step_token_index, :, real_idx, :]
            select_kv_cache[batch_idx * beam_size : batch_idx * beam_size + beam_size, :, real_idx, :] = cur_step_kv
    
    return select_kv_cache


def calc_group_num(token_idxs):
    group_num = torch.zeros([batch, beam_size], dtype=torch.int32)
    for b_idx in range(batch):
        for token_idx in token_idxs[b_idx*beam_size: b_idx*beam_size+beam_size]:
            group_num[b_idx][token_idx] += 1
        # 计算前缀和
        for i in range(1, beam_size):
            group_num[b_idx][i] = group_num[b_idx][i - 1] + group_num[b_idx][i]
    return group_num.reshape([batch*beam_size])


def generate_beam_idxs():
    beam_idxs = torch.randint(0, beam_size, [batch, beam_size], dtype=torch.int32)
    # beam_idxs中的序号一定是升序的
    sorted_beam_idxs, sort_indices = torch.sort(beam_idxs) 
    return sorted_beam_idxs.reshape([batch*beam_size,])


if __name__ == '__main__':
    # torch.ops.load_library(f"../torch_plugin/2.6.0/select_unshared_kv/build/libselect_unshared_kv_ops.so")

    device_id = 6
    torch_npu.npu.set_device(device_id)
    torch.manual_seed(1234)
    batch = 10
    max_decode_step = 3
    beam_size = 512
    head_num = 8
    head_dim = 128
    dtype = torch.bfloat16
    x_key_block_npu = torch.zeros([batch*beam_size, head_num, max_decode_step, head_dim], dtype=dtype).npu()
    x_value_block_npu = torch.zeros([batch*beam_size, head_num, max_decode_step, head_dim], dtype=dtype).npu()
    x_key_block = torch.zeros([batch*beam_size, head_num, max_decode_step, head_dim], dtype=dtype)
    x_value_block = torch.zeros([batch*beam_size, head_num, max_decode_step, head_dim], dtype=dtype)

    atol_div = 1e-6
    global_index = torch.zeros([batch, beam_size, max_decode_step], dtype=torch.int32)
    for decode_step in range(1, max_decode_step + 1):

        beam_idxs = generate_beam_idxs()        
        curr_key = torch.rand([batch*beam_size, head_num, head_dim], dtype=dtype)
        curr_value = torch.rand([batch*beam_size, head_num, head_dim], dtype=dtype)
        real_index = decode_step - 1
        # 填充本轮的key、value
        x_key_block[:, :, real_index, :] = curr_key
        x_value_block[:, :, real_index, :] = curr_value

        x_key_block_npu[:, :, real_index, :] = curr_key.npu()
        x_value_block_npu[:, :, real_index, :] = curr_value.npu()

        select_key_golden = select_unshared_kv(global_index, x_key_block, curr_key, beam_idxs, decode_step)
        select_value_golden = select_unshared_kv(global_index, x_value_block, curr_value, beam_idxs, decode_step)

        # group_num计算
        group_num = calc_group_num(beam_idxs)
        
        # decode_step_tensor = torch.Tensor([decode_step]).to(torch.int32)
        # beam_size_tensor = torch.Tensor([beam_size]).to(torch.int32)
        # npu执行
        
        custom_ops.select_unshared_kv_npu(
            beam_idxs.npu(), 
            x_key_block_npu, 
            x_value_block_npu,
            group_num.npu(),
            decode_step,
            beam_size
        )
        
        select_key_npu = x_key_block_npu.cpu()[:, :, :decode_step, :]
        select_value_npu = x_value_block_npu.cpu()[:, :, :decode_step, :]
        
        key_flag = torch.allclose(select_key_npu, select_key_golden, atol=atol_div, rtol=0)
        value_flag = torch.allclose(select_value_npu, select_value_golden, atol=atol_div, rtol=0)
        if key_flag:
            print(f"decode_step {decode_step} key_block is wright")
        else:
            print(f"decode_step {decode_step} key_block is wrong")
        if value_flag:
            print(f"decode_step {decode_step} value_block is wright")
        else:
            print(f"decode_step {decode_step} value_block is wrong")

        # for beam_idx in range(beam_size):
        #     golden = select_key_golden[beam_idx]
        #     npu_res = select_key_npu[beam_idx]
        #     flag = torch.allclose(npu_res, golden, atol=atol_div, rtol=0)
        #     if flag:
        #         print(f"beam {beam_idx} is wright")
        #     else:
        #         print(f"beam {beam_idx} is wrong")
        #         print(golden)
        #         print(npu_res)
        #         break
        