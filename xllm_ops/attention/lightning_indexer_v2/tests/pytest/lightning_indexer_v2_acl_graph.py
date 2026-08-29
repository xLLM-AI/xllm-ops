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

import test
import torch
import torch_npu
import pytest
import random
import numpy as np
import math
import ctypes
import copy
import torchair
import torch.nn as nn
from torchair.configs.compiler_config import CompilerConfig
import lightning_indexer_v2_golden

class LIV2Network(nn.Module):
    def __init__(self):
        super(LIV2Network, self).__init__()

    def forward(self, query, key, weights, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k, block_table,
                output_idx_offset, metadata, topk, max_seqlen_q, layout_q, layout_k, mask_mode, cmp_ratio, return_value):
        return torch.ops.cann_ops_transformer.lightning_indexer(query, key, weights,
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
                                                                layout_q = layout_q,
                                                                layout_k = layout_k,
                                                                mask_mode = mask_mode,
                                                                cmp_ratio = cmp_ratio,
                                                                return_value = return_value)

def _liv2_prepare_tensors_and_metadata(params, tensor_dict):
    """
    统一处理 tensor 准备和 metadata 构造（共用逻辑）。
    兼容两个来源：generate_liv2_test_data 返回值和 .pt 文件加载，二者都在 CPU。
    """
    qk_dtype = params[10]

    if qk_dtype == 'FP16' or qk_dtype == torch.float16:
        query = tensor_dict["query"].to(dtype=torch.float16).npu()
        key = tensor_dict["key"].to(dtype=torch.float16).npu()
        if 'blockFusion' in tensor_dict and tensor_dict['blockFusion'] is not None:
            blockFusion = tensor_dict['blockFusion'].to(dtype=torch.float16).npu()
    else:
        query = tensor_dict["query"].to(dtype=torch.bfloat16).npu()
        key = tensor_dict["key"].to(dtype=torch.bfloat16).npu()
        if 'blockFusion' in tensor_dict and tensor_dict['blockFusion'] is not None:
            blockFusion = tensor_dict['blockFusion'].to(dtype=torch.bfloat16).npu()

    weights = tensor_dict["weights"].npu()

    if 'blockFusion' in tensor_dict and tensor_dict['blockFusion'] is not None:
        block_num = int(params[9])
        block_size = int(params[8])
        head_dim = int(params[7])
        k_head_num = int(params[6])
        key = blockFusion[:, :block_size * k_head_num * head_dim].view(block_num, block_size, k_head_num, head_dim)

    cu_seqlens_query = tensor_dict["cu_seqlens_q"].npu() if tensor_dict["cu_seqlens_q"] is not None else None
    cu_seqlens_key = tensor_dict["cu_seqlens_k"].npu() if tensor_dict["cu_seqlens_k"] is not None else None
    seqused_q = tensor_dict["seqused_q"].npu() if tensor_dict["seqused_q"] is not None else None
    seqused_k = tensor_dict["seqused_k"].npu() if tensor_dict["seqused_k"] is not None else None
    output_idx_offset = tensor_dict["output_idx_offset"].npu() if tensor_dict["output_idx_offset"] is not None else None
    block_table = tensor_dict["block_table"].npu() if tensor_dict["block_table"] is not None else None
    cmp_residual_k_for_npu = tensor_dict["cmp_residual_k_for_npu"].npu() if tensor_dict.get("cmp_residual_k_for_npu") is not None else None

    layout_query = tensor_dict["layout_query"]
    layout_key = tensor_dict["layout_key"]
    topk = tensor_dict["topk"]
    mask_mode = tensor_dict["mask_mode"]
    cmp_ratio = tensor_dict["cmp_ratio"]
    max_seqlen_q_meta = tensor_dict["max_seqlen_q_meta"]
    max_seqlen_k_meta = tensor_dict["max_seqlen_k_meta"]

    q_head_num = int(params[5])
    k_head_num = int(params[6])
    head_dim = int(params[7])
    batch_size = int(params[0])

    metadata = torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        num_heads_q = q_head_num,
        num_heads_k = k_head_num,
        head_dim = head_dim,
        topk = topk,
        cu_seqlens_q = cu_seqlens_query,
        cu_seqlens_k = cu_seqlens_key,
        seqused_q = seqused_q,
        seqused_k = seqused_k,
        cmp_residual_k = cmp_residual_k_for_npu,
        batch_size = batch_size,
        max_seqlen_q = max_seqlen_q_meta,
        max_seqlen_k = max_seqlen_k_meta,
        layout_q = layout_query,
        layout_k = layout_key,
        mask_mode = mask_mode,
        cmp_ratio = cmp_ratio)
    metadata = metadata.npu()

    run_args = {
        "query": query,
        "key": key,
        "weights": weights,
        "cu_seqlens_q": cu_seqlens_query,
        "cu_seqlens_k": cu_seqlens_key,
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
        "cmp_residual_k": cmp_residual_k_for_npu,
        "output_idx_offset": output_idx_offset,
        "max_seqlen_q": params[26],
        "block_table": block_table,
        "metadata": metadata,
        "layout_q": layout_query,
        "layout_k": layout_key,
        "topk": topk,
        "mask_mode": mask_mode,
        "cmp_ratio": cmp_ratio,
        "return_value": params[25],
    }
    return run_args

def _liv2_run_compiled_graph(run_args):
    """
    通过 torch.compile + torchair 后端执行算子（共用逻辑）。
    """
    config = CompilerConfig()
    config.mode = "reduce-overhead"
    npu_backend = torchair.get_npu_backend(compiler_config=config)
    torch._dynamo.reset()
    npu_mode = torch.compile(LIV2Network().npu(), fullgraph=False, backend=npu_backend, dynamic=False)
    npu_result, npu_topk_value = npu_mode(**run_args)
    torch.npu.synchronize()
    npu_topk_value, _ = npu_topk_value.sort(dim=-1, descending=True)
    return npu_result, npu_topk_value

def liv2_output_acl_graph(params):
    tensor_dict = lightning_indexer_v2_golden.generate_liv2_test_data(params)
    cpu_result = tensor_dict["cpu_result"]
    topk_value = tensor_dict["topk_value"]
    cpu_topk_value = tensor_dict["cpu_topk_value"]

    run_args = _liv2_prepare_tensors_and_metadata(params, tensor_dict)
    npu_result, npu_topk_value = _liv2_run_compiled_graph(run_args)

    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value

def liv2_output_acl_graph_from_pt(params, tensor_dict):
    """
    graph 模式入口（batch 用例使用）：从 .pt 文件加载 pre-computed tensor，走 torch.compile 执行。
    跳过 generate_liv2_test_data 的随机数据重新生成和 CPU golden 重算。
    """
    cpu_result = tensor_dict["cpu_result"]
    topk_value = tensor_dict["topk_value"]
    cpu_topk_value = tensor_dict["cpu_topk_value"]

    run_args = _liv2_prepare_tensors_and_metadata(params, tensor_dict)
    npu_result, npu_topk_value = _liv2_run_compiled_graph(run_args)

    return cpu_result, npu_result, topk_value, cpu_topk_value, npu_topk_value
