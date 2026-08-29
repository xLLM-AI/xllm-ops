#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import pandas as pd 
import numpy as np
import torch
import torch_npu
import pytest
import random
import math
import ast
from cann_ops_transformer.ops import lightning_indexer, lightning_indexer_metadata

def test_liv2_process(filepath, device_id=0):
    # 加载测试数据
    test_data = torch.load(filepath, map_location="cpu")

    params = test_data['params']
    cpu_result = test_data['cpu_result']
    topk_value = test_data['topk_value']
    cpu_topk_value = test_data['cpu_topk_value']
    max_seqlen_q = test_data['max_seqlen_q']
    print("执行用例：", filepath)
    torch_npu.npu.set_device(device_id)

    query = test_data['query'].npu()
    key = test_data['key'].npu()
    batch_size = int(params[0])
    q_head_num = int(params[5])
    k_head_num = int(params[6])
    head_dim = int(params[7])
    block_size = int(params[8])
    block_num = int(params[9])
    if params[10] == 'FP16':
        if 'blockFusion' in test_data and test_data['blockFusion'] is not None:
            blockFusion = test_data['blockFusion'].to(dtype=torch.float16).npu()
    elif params[10] == 'BF16':
        if 'blockFusion' in test_data and test_data['blockFusion'] is not None:
            blockFusion = test_data['blockFusion'].to(dtype=torch.bfloat16).npu()
    topk = params[19]
    mask_mode = params[20]
    return_value = params[25]
    weights = test_data['weights'].npu()
    if 'blockFusion' in test_data and test_data['blockFusion'] is not None:
        key = blockFusion[:, :block_size * k_head_num * head_dim].view(block_num, block_size, k_head_num, head_dim)
    if test_data['seqused_q'] is not None:
        seqused_q = test_data['seqused_q'].npu()
    else:
        seqused_q = None
    if test_data['seqused_k'] is not None:
        seqused_k = test_data['seqused_k'].npu()
    else:
        seqused_k = None
    if test_data['output_idx_offset'] is not None:
        output_idx_offset = test_data['output_idx_offset'].npu()
    else:
        output_idx_offset = None
    if test_data['cu_seqlens_q'] is not None:
        cu_seqlens_q = test_data['cu_seqlens_q'].npu()
    else:
        cu_seqlens_q = None
    if test_data['cu_seqlens_k'] is not None:
        cu_seqlens_k = test_data['cu_seqlens_k'].npu()
    else:
        cu_seqlens_k = None
    if test_data['block_table'] is not None:
        block_table = test_data['block_table'].npu()
    else:
        block_table = None
    layout_query = test_data['layout_query']
    layout_key = test_data['layout_key']
    cmp_ratio = test_data['cmp_ratio']
    if test_data['cmp_residual_k_for_npu'] is not None:
        cmp_residual_k = test_data['cmp_residual_k_for_npu'].npu()
    else:
        cmp_residual_k = None
    max_seqlen_q_meta = test_data['max_seqlen_q_meta']
    max_seqlen_k_meta = test_data['max_seqlen_k_meta']

    metadata = lightning_indexer_metadata(
                    num_heads_q = q_head_num,
                    num_heads_k = k_head_num,
                    head_dim = head_dim,
                    topk = topk,
                    cu_seqlens_q = cu_seqlens_q,
                    cu_seqlens_k = cu_seqlens_k,
                    seqused_q = seqused_q,
                    seqused_k = seqused_k,
                    cmp_residual_k = cmp_residual_k,
                    batch_size = batch_size,
                    max_seqlen_q = max_seqlen_q_meta,
                    max_seqlen_k = max_seqlen_k_meta,
                    layout_q = layout_query,
                    layout_k = layout_key,
                    mask_mode = mask_mode,
                    cmp_ratio = cmp_ratio)

    metadata = metadata.npu()
    #调用li算子
    npu_result, npu_topk_value = lightning_indexer(query, key, weights,
                                             cu_seqlens_q = cu_seqlens_q,
                                             cu_seqlens_k = cu_seqlens_k,
                                             seqused_q = seqused_q,
                                             seqused_k = seqused_k,
                                             cmp_residual_k = cmp_residual_k,
                                             block_table = block_table,
                                             output_idx_offset = output_idx_offset,
                                             metadata = metadata,
                                             topk = topk,
                                             max_seqlen_q = max_seqlen_q,
                                             layout_q = layout_query,
                                             layout_k = layout_key,
                                             mask_mode = mask_mode,
                                             cmp_ratio = cmp_ratio,
                                             return_value = return_value)
    torch.npu.synchronize()
    npu_topk_value, _ = npu_topk_value.sort(dim=-1, descending=True)
    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value, output_idx_offset, params

def test_liv2_process_graph(filepath, device_id=0):
    """
    graph 模式：从 .pt 文件加载 pre-computed tensor，走 torch.compile + torchair 后端执行算子，
    跳过 generate_liv2_test_data 的随机数据重新生成和 CPU golden 重算。
    与 eager 模式共用相同的 .pt 数据，仅算子调用路径不同（compile vs eager）。
    """
    import lightning_indexer_v2_acl_graph
    test_data = torch.load(filepath, map_location="cpu")
    params = test_data['params']
    output_idx_offset = test_data.get('output_idx_offset', None)

    torch_npu.npu.set_device(device_id)
    cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value = \
        lightning_indexer_v2_acl_graph.liv2_output_acl_graph_from_pt(params, test_data)

    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value, output_idx_offset, params
