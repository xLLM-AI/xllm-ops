# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

LI_METADATA_SIZE = 1024
LI_METADATA_OP_NAME = "lightning_indexer_metadata"


class LightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(LightningIndexerOpBuilder, self).__init__(
            "lightning_indexer", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/lightning_indexer.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
            "Tensor? cmp_residual_k=None, int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_k=None, "
            "str? layout_q=None, str? layout_k=None, int? mask_mode=None, int? cmp_ratio=None) -> Tensor",
            "lightning_indexer(Tensor q, Tensor k, Tensor w, "
            "int topk, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None,"
            "Tensor? seqused_q=None, Tensor? seqused_k=None, "
            "Tensor? cmp_residual_k=None, Tensor? block_table=None, "
            "Tensor? output_idx_offset=None, Tensor? metadata=None, int max_seqlen_q=-1,"
            'str layout_q="BSND", str layout_k="BSND", int mask_mode=0, '
            "int cmp_ratio=1,"
            "int return_value=0) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + LI_METADATA_OP_NAME)
        def lightning_indexer_metadata_meta(
            num_heads_q: int,
            num_heads_k: int,
            head_dim: int,
            topk: int,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_k: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_k: Optional[torch.Tensor] = None,
            cmp_residual_k: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_k: Optional[int] = None,
            layout_q: Optional[str] = None,
            layout_k: Optional[str] = None,
            mask_mode: Optional[int] = None,
            cmp_ratio: Optional[int] = None,
        ):
            return torch.empty((LI_METADATA_SIZE), dtype=torch.int32, device="npu")

        @impl(get_as_library(), self.name, "Meta")
        def lightning_indexer_meta(
            q,
            k,
            w,
            topk,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            output_idx_offset=None,
            metadata=None,
            max_seqlen_q=-1,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
            return_value=0,
        ):
            key_head_num = k.shape[1] if layout_k == "TND" else k.shape[2]

            if layout_q == "BSND":
                sparse_indices_out = torch.empty(
                    [q.shape[0], q.shape[1], key_head_num, topk],
                    dtype=torch.int32,
                    device="meta",
                )
            else:
                sparse_indices_out = torch.empty(
                    [q.shape[0], key_head_num, topk], dtype=torch.int32, device="meta"
                )
            if return_value:
                if layout_q == "BSND":
                    sparse_values_out = torch.empty(
                        [q.shape[0], q.shape[1], key_head_num, topk],
                        dtype=torch.float,
                        device="meta",
                    )
                else:
                    sparse_values_out = torch.empty(
                        [q.shape[0], key_head_num, topk],
                        dtype=torch.float,
                        device="meta",
                    )
            else:
                sparse_values_out = torch.empty([0], dtype=torch.float, device="meta")
            return (sparse_indices_out, sparse_values_out)


# Instantiate the builder
lightning_indexer_op_builder = LightningIndexerOpBuilder()
lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), LI_METADATA_OP_NAME, "PrivateUse1")
def lightning_indexer_metadata(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_k: Optional[str] = None,
    mask_mode: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio

    op_module = lightning_indexer_op_builder.load()
    return op_module.lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


@torch.library.register_kernel("cann_ops_transformer::" + LI_METADATA_OP_NAME, None)
def lightning_indexer_metadata_fallback(
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    topk: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_k: Optional[str] = None,
    mask_mode: Optional[int] = None,
    cmp_ratio: Optional[int] = None,
):
    # 处理所有 tensor 都为 None 的情况
    # 调用 NPU 实现
    return lightning_indexer_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        topk,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


torch.compiler.allow_in_graph(lightning_indexer_metadata)


@impl(get_as_library(), lightning_indexer_op_builder.name, "PrivateUse1")
def lightning_indexer(
    q,
    k,
    w,
    topk,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = lightning_indexer_op_builder.load()
    return op_module.lightning_indexer(
        q,
        k,
        w,
        topk,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        metadata,
        max_seqlen_q,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
        return_value,
    )
