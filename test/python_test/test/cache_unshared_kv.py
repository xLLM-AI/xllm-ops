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



if __name__ == '__main__':
    # torch.ops.load_library(f"../torch_plugin/2.6.0/select_unshared_kv/build/libselect_unshared_kv_ops.so")
    # torch.ops.load_library(f"../torch_plugin/2.6.0/cache_unshared_kv/build/libcache_unshared_kv.so")

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

        curr_key = torch.rand([batch*beam_size, head_num, head_dim], dtype=dtype)
        curr_value = torch.rand([batch*beam_size, head_num, head_dim], dtype=dtype)
        real_index = decode_step - 1
        # 填充本轮的key、value
        x_key_block[:, :, real_index, :] = curr_key
        x_value_block[:, :, real_index, :] = curr_value

        decode_step_tensor = torch.Tensor([decode_step]).to(torch.int32)
        # npu执行
        custom_ops.cache_unshared_kv_npu(
            x_key_block_npu, 
            x_value_block_npu,
            curr_key.npu(),
            curr_value.npu(),
            decode_step_tensor.npu()
        )
        
        npu_key_block = x_key_block_npu.cpu()
        npu_value_block = x_value_block_npu.cpu()

        key_flag = torch.allclose(npu_key_block, x_key_block, atol=atol_div, rtol=0)
        value_flag = torch.allclose(npu_value_block, x_value_block, atol=atol_div, rtol=0)
        if key_flag:
            print(f"decode_step {decode_step} key_block is wright")
        else:
            print(f"decode_step {decode_step} key_block is wrong")
        if value_flag:
            print(f"decode_step {decode_step} value_block is wright")
        else:
            print(f"decode_step {decode_step} value_block is wrong")
