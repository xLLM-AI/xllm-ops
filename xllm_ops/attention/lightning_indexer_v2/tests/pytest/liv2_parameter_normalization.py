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

"""Pure LI_V2 parameter normalization shared by pytest and TTK adapters."""


def normalize_liv2_params(params):
    """Apply the batch pytest scalar conversions without generating test data."""
    values = tuple(params)
    if len(values) != 27:
        raise ValueError(
            f"LI_V2 parameter count mismatch: got {len(values)}, expected 27"
        )

    (
        batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num,
        head_dim, block_size, block_num, qk_dtype, cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k, cmp_residual_k, output_idx_offset, layout_query,
        layout_key, topk, mask_mode, query_datarange, key_datarange,
        weights_datarange, cmp_ratio, return_value, max_seqlen_q,
    ) = values

    q_t_size = 0 if q_t_size is None else int(q_t_size)
    k_t_size = 0 if k_t_size is None else int(k_t_size)
    block_size = 0 if block_size is None else int(block_size)
    block_num = 0 if block_num is None else int(block_num)
    max_seqlen_q = -1 if max_seqlen_q is None else int(max_seqlen_q)

    return (
        int(batch_size), int(q_seq), int(k_seq), q_t_size, k_t_size,
        int(q_head_num), int(k_head_num), int(head_dim), block_size, block_num,
        qk_dtype, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k,
        cmp_residual_k, output_idx_offset, layout_query, layout_key, int(topk),
        int(mask_mode), query_datarange, key_datarange, weights_datarange,
        int(cmp_ratio), int(return_value), max_seqlen_q,
    )
