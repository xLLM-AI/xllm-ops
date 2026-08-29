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
"""Populate LightningIndexer V2 metadata without depending on TTK state."""

import importlib.util
import logging
import sys
from pathlib import Path

import torch


OPERATOR = "lightning_indexer_v2"
METADATA_INDEX = 10


def load_metadata_protocol():
    name = "li_v2_ttk_metadata_protocol"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("metadata_protocol.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def get_attribute(kwargs, name, default=None, aliases=()):
    for key in (name, f"pytest_{name}", *aliases):
        value = kwargs.get(key)
        if value is not None:
            return value
    return default


def get_values(kwargs, name, tensor):
    value = kwargs.get(f"{name}_values")
    if value is None:
        value = tensor
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def max_sequence(prefix, used, fallback):
    if used:
        return max(int(value) for value in used)
    if prefix and len(prefix) > 1:
        return max(
            int(prefix[index + 1]) - int(prefix[index])
            for index in range(len(prefix) - 1)
        )
    return int(fallback)


def build_metadata_arguments(
    q, k, topk, max_seqlen_q, layout_q, layout_k, mask_mode, cmp_ratio, kwargs
):
    q_shape = tuple(int(value) for value in q.shape)
    k_shape = tuple(int(value) for value in k.shape)
    num_heads_q = q_shape[2] if layout_q == "BSND" else q_shape[1]
    num_heads_k = k_shape[1] if layout_k == "TND" else k_shape[2]
    cu_q = kwargs.get("cu_seqlens_q")
    cu_k = kwargs.get("cu_seqlens_k")
    seq_q = kwargs.get("seqused_q")
    seq_k = kwargs.get("seqused_k")
    cu_q_values = get_values(kwargs, "cu_seqlens_q", cu_q)
    cu_k_values = get_values(kwargs, "cu_seqlens_k", cu_k)
    seq_q_values = get_values(kwargs, "seqused_q", seq_q)
    seq_k_values = get_values(kwargs, "seqused_k", seq_k)

    batch_size = get_attribute(kwargs, "batch_size")
    if batch_size is None:
        if seq_q_values is not None:
            batch_size = len(seq_q_values)
        elif cu_q_values is not None:
            batch_size = len(cu_q_values) - 1
        elif layout_q == "BSND":
            batch_size = q_shape[0]
        else:
            batch_size = 0

    q_fallback = q_shape[1] if layout_q == "BSND" else q_shape[0]
    if int(max_seqlen_q) >= 0:
        q_fallback = int(max_seqlen_q)
    if layout_k == "BSND":
        k_fallback = k_shape[1]
    elif layout_k == "TND":
        k_fallback = k_shape[0]
    else:
        k_fallback = int(get_attribute(kwargs, "max_seqlen_k", k_shape[1]))

    return {
        "num_heads_q": int(
            get_attribute(kwargs, "num_heads_q", num_heads_q, ("pytest_q_head_num",))
        ),
        "num_heads_k": int(
            get_attribute(kwargs, "num_heads_k", num_heads_k, ("pytest_k_head_num",))
        ),
        "head_dim": int(get_attribute(kwargs, "head_dim", q_shape[-1])),
        "topk": int(topk),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "seqused_q": seq_q,
        "seqused_k": seq_k,
        "cmp_residual_k": kwargs.get("cmp_residual_k"),
        "batch_size": int(batch_size),
        "max_seqlen_q": int(
            get_attribute(
                kwargs,
                "metadata_max_seqlen_q",
                max_sequence(cu_q_values, seq_q_values, q_fallback),
            )
        ),
        "max_seqlen_k": int(
            get_attribute(
                kwargs,
                "metadata_max_seqlen_k",
                max_sequence(cu_k_values, seq_k_values, k_fallback),
            )
        ),
        "layout_q": str(layout_q),
        "layout_k": str(layout_k),
        "mask_mode": int(mask_mode),
        "cmp_ratio": int(cmp_ratio),
    }


def move_to_device(value, target):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def run_metadata(arguments, metadata):
    return torch.ops.cann_ops_transformer.lightning_indexer_metadata(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_k"]),
        int(arguments["head_dim"]),
        int(arguments["topk"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), metadata),
        cu_seqlens_k=move_to_device(arguments.get("cu_seqlens_k"), metadata),
        seqused_q=move_to_device(arguments.get("seqused_q"), metadata),
        seqused_k=move_to_device(arguments.get("seqused_k"), metadata),
        cmp_residual_k=move_to_device(arguments.get("cmp_residual_k"), metadata),
        batch_size=int(arguments["batch_size"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_k=int(arguments["max_seqlen_k"]),
        layout_q=str(arguments["layout_q"]),
        layout_k=str(arguments["layout_k"]),
        mask_mode=int(arguments["mask_mode"]),
        cmp_ratio=int(arguments["cmp_ratio"]),
    )


def run(
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
    metadata=None,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    **kwargs,
):
    """Generate metadata once, or reuse a nonzero manual-data input."""
    del w
    if metadata is None:
        raise ValueError("LightningIndexer V2 npu_preprocess requires metadata")
    arguments_kwargs = dict(kwargs)
    arguments_kwargs.update(
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "seqused_q": seqused_q,
            "seqused_k": seqused_k,
            "cmp_residual_k": cmp_residual_k,
        }
    )
    protocol = load_metadata_protocol()
    testcase_name = kwargs.get("testcase_name")
    if protocol.metadata_is_materialized(metadata):
        logging.info("[%s] reuse nonzero LI_V2 metadata input", testcase_name)
        return None
    arguments = protocol.load_metadata_inputs(OPERATOR, testcase_name)
    if arguments is not None:
        source = "manual-data sidecar"
    else:
        arguments = build_metadata_arguments(
            q,
            k,
            topk,
            max_seqlen_q,
            layout_q,
            layout_k,
            mask_mode,
            cmp_ratio,
            arguments_kwargs,
        )
        source = "main API fallback (sidecar unavailable)"
    logging.info("[%s] build LI_V2 metadata from %s", testcase_name, source)
    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "LI_V2 metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    rewritten = protocol.rewrite_metadata_input(
        OPERATOR, testcase_name, METADATA_INDEX, metadata
    )
    if rewritten is not None:
        logging.info("[%s] rewrote LI_V2 metadata input: %s", testcase_name, rewritten)
    return None
