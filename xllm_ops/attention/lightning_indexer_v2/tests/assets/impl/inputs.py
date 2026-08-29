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

"""Input customization for LightningIndexer V2 TTK cases."""

import importlib.util
import sys
from pathlib import Path

import torch


class LightningIndexerV2InputAdapter:
    """Translate a TTK case and reuse the pytest input/golden generator."""

    @staticmethod
    def module_load_error(stage, path, exc):
        return RuntimeError(
            "Failed to load LightningIndexerV2 module; "
            f"stage={stage}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    def __init__(self):
        self.pytest_golden = None
        self.pytest_normalizer = None

    @staticmethod
    def load_golden_store():
        name = "li_v2_ttk_golden"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("golden.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise LightningIndexerV2InputAdapter.module_load_error(
                "assets Golden store", path, exc
            ) from exc
        return module

    @staticmethod
    def load_metadata_protocol():
        name = "li_v2_ttk_metadata_protocol"
        if name in sys.modules:
            return sys.modules[name]
        path = Path(__file__).with_name("metadata_protocol.py")
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(name, None)
            raise LightningIndexerV2InputAdapter.module_load_error(
                "assets metadata protocol", path, exc
            ) from exc
        return module

    def load_pytest_golden(self):
        if self.pytest_golden is not None:
            return self.pytest_golden
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "lightning_indexer_v2_golden.py"
        name = "li_v2_pytest_golden"
        inserted = str(pytest_dir) not in sys.path
        if inserted:
            sys.path.insert(0, str(pytest_dir))
        try:
            if name in sys.modules:
                module = sys.modules[name]
            else:
                spec = importlib.util.spec_from_file_location(name, path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.pytest_golden = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error("pytest Golden", path, exc) from exc
        finally:
            if inserted:
                sys.path.remove(str(pytest_dir))

    def load_pytest_normalizer(self):
        if self.pytest_normalizer is not None:
            return self.pytest_normalizer
        pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
        path = pytest_dir / "liv2_parameter_normalization.py"
        name = "li_v2_pytest_normalizer"
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create import spec for {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.pytest_normalizer = module
            return module
        except Exception as exc:
            sys.modules.pop(name, None)
            raise self.module_load_error(
                "pytest parameter normalizer", path, exc
            ) from exc

    @staticmethod
    def list_value(kwargs, name):
        value = kwargs.get(f"{name}_values")
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(item) for item in value]

    @staticmethod
    def prefix_lengths(value):
        if not value:
            return []
        return [
            int(value[index + 1]) - int(value[index]) for index in range(len(value) - 1)
        ]

    @staticmethod
    def data_range(input_ranges, index):
        if (
            input_ranges
            and index < len(input_ranges)
            and input_ranges[index] is not None
        ):
            return repr(list(input_ranges[index]))
        return None

    @staticmethod
    def qk_dtype_name(tensor):
        if tensor.dtype == torch.float16:
            return "FP16"
        if tensor.dtype == torch.bfloat16:
            return "BF16"
        raise ValueError(f"unsupported LI_V2 q/k dtype: {tensor.dtype}")

    def geometry(self, q, k, layout_q, layout_k, cu_q, cu_k, seq_q, seq_k):
        q_lengths = self.prefix_lengths(cu_q) or (seq_q or [])
        k_lengths = self.prefix_lengths(cu_k) or (seq_k or [])
        if layout_q == "BSND":
            batch_size, q_seq, q_head_num, head_dim = [int(item) for item in q.shape]
            q_t_size = 0
        elif layout_q == "TND":
            q_t_size, q_head_num, head_dim = [int(item) for item in q.shape]
            batch_size = len(q_lengths)
            q_seq = max(q_lengths, default=q_t_size)
        else:
            raise ValueError(f"unsupported LI_V2 query layout: {layout_q}")

        if layout_k == "BSND":
            _, k_seq, k_head_num, _ = [int(item) for item in k.shape]
            k_t_size = 0
            block_size = 0
            block_num = 0
        elif layout_k == "TND":
            k_t_size, k_head_num, _ = [int(item) for item in k.shape]
            k_seq = max(k_lengths, default=k_t_size)
            block_size = 0
            block_num = 0
        elif layout_k == "PA_BBND":
            block_num, block_size, k_head_num, _ = [int(item) for item in k.shape]
            capacity = block_num * block_size
            per_batch_capacity = (
                capacity // batch_size if batch_size > 0 else block_size
            )
            k_seq = max(k_lengths, default=per_batch_capacity)
            k_t_size = 0
        else:
            raise ValueError(f"unsupported LI_V2 key layout: {layout_k}")
        return (
            batch_size,
            q_seq,
            k_seq,
            q_t_size,
            k_t_size,
            q_head_num,
            k_head_num,
            head_dim,
            block_size,
            block_num,
        )

    def build_case_params(self, q, k, layout_q, layout_k, kwargs):
        cu_q = self.list_value(kwargs, "cu_seqlens_q")
        cu_k = self.list_value(kwargs, "cu_seqlens_k")
        seq_q = self.list_value(kwargs, "seqused_q")
        seq_k = self.list_value(kwargs, "seqused_k")
        residual = self.list_value(kwargs, "cmp_residual_k")
        geometry = self.geometry(q, k, layout_q, layout_k, cu_q, cu_k, seq_q, seq_k)
        input_ranges = kwargs.get("input_ranges") or ()
        output_range = kwargs.get("output_idx_offset_range")
        output_range = None if output_range is None else repr(list(output_range))
        max_seqlen_q = kwargs.get("max_seqlen_q")
        max_seqlen_q = None if max_seqlen_q is None else int(max_seqlen_q)
        params = geometry + (
            self.qk_dtype_name(q),
            cu_q,
            cu_k,
            seq_q,
            seq_k,
            residual,
            output_range,
            layout_q,
            layout_k,
            kwargs.get("topk"),
            kwargs.get("mask_mode"),
            self.data_range(input_ranges, 0),
            self.data_range(input_ranges, 1),
            self.data_range(input_ranges, 2),
            kwargs.get("cmp_ratio"),
            kwargs.get("return_value"),
            max_seqlen_q,
        )
        return self.load_pytest_normalizer().normalize_liv2_params(params)

    @staticmethod
    def copy_tensor(dst, src, name):
        if dst is None:
            if src is not None:
                raise ValueError(
                    f"{name} is absent from CSV but pytest generator produced a tensor"
                )
            return
        if src is None:
            raise ValueError(
                f"{name} is present in CSV but pytest generator returned None"
            )
        src_cpu = src.detach().cpu() if torch.is_tensor(src) else torch.as_tensor(src)
        if tuple(dst.shape) != tuple(src_cpu.shape):
            raise ValueError(
                f"{name} shape mismatch: TTK={tuple(dst.shape)} "
                f"pytest={tuple(src_cpu.shape)}"
            )
        dst.copy_(src_cpu.to(dtype=dst.dtype, device=dst.device))

    @staticmethod
    def tensor_values(tensor):
        if tensor is None:
            return None
        return [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]

    @staticmethod
    def restore_paged_key(key, block_table, batch_size, sequence_length):
        """Restore the logical BNSD key consumed by the pytest compare model."""
        physical = key.detach().cpu()
        table = block_table.detach().cpu().to(torch.int64)
        if physical.ndim != 4:
            raise ValueError(
                f"paged key must be 4-D, got shape {tuple(physical.shape)}"
            )
        block_size, head_num, head_dim = (
            int(physical.shape[1]),
            int(physical.shape[2]),
            int(physical.shape[3]),
        )
        logical = torch.zeros(
            (batch_size, head_num, sequence_length, head_dim), dtype=physical.dtype
        )
        for batch_idx in range(batch_size):
            for logical_block, block_id_value in enumerate(table[batch_idx].tolist()):
                if block_id_value < 0:
                    continue
                if block_id_value >= physical.shape[0]:
                    raise ValueError(
                        f"block id {block_id_value} exceeds key block count"
                    )
                start = logical_block * block_size
                if start >= sequence_length:
                    break
                count = min(block_size, sequence_length - start)
                logical[batch_idx, :, start : start + count, :] = physical[
                    block_id_value, :count
                ].permute(1, 0, 2)
        return logical

    @staticmethod
    def normalize_compare_attributes(compare_context):
        attributes = dict(compare_context.attributes)
        aliases = {
            "maxSeqlenQ": "max_seqlen_q",
            "layoutQOptional": "layout_q",
            "layoutKOptional": "layout_k",
            "maskMode": "mask_mode",
            "cmpRatio": "cmp_ratio",
            "returnValue": "return_value",
        }
        for source, target in aliases.items():
            if target not in attributes and source in attributes:
                attributes[target] = attributes[source]
        return attributes

    def rebuild_compare_data(self, compare_context):
        """Rebuild only the pytest TopK compare context from replayed inputs."""
        tensors = tuple(compare_context.input_tensors or ())
        if len(tensors) < 10:
            raise ValueError(
                "LightningIndexerV2 compare context requires ten input slots"
            )
        q, k, w, cu_q, cu_k, seq_q, seq_k, residual, block_table, offset = tensors[:10]
        attributes = self.normalize_compare_attributes(compare_context)
        for name, tensor in (
            ("cu_seqlens_q", cu_q),
            ("cu_seqlens_k", cu_k),
            ("seqused_q", seq_q),
            ("seqused_k", seq_k),
            ("cmp_residual_k", residual),
        ):
            values = self.tensor_values(tensor)
            if values is not None:
                attributes[f"{name}_values"] = values

        layout_q = attributes.get("layout_q")
        layout_k = attributes.get("layout_k")
        if layout_q is None or layout_k is None:
            raise ValueError(
                "LI_V2 replay compare requires layout_q and layout_k from attributes"
            )
        params = self.build_case_params(q, k, layout_q, layout_k, attributes)
        (
            batch_size,
            q_seq,
            k_seq,
            q_t_size,
            k_t_size,
            q_head_num,
            k_head_num,
            head_dim,
            block_size,
            block_num,
            _,
            cu_q_values,
            cu_k_values,
            seq_q_values,
            seq_k_values,
            residual_values,
            _,
            _,
            _,
            topk,
            mask_mode,
            _,
            _,
            _,
            cmp_ratio,
            return_value,
            max_seqlen_q,
        ) = params

        def as_int_tensor(value):
            return None if value is None else torch.tensor(value, dtype=torch.int32)

        cu_q_cpu = as_int_tensor(cu_q_values)
        cu_k_cpu = as_int_tensor(cu_k_values)
        seq_q_cpu = as_int_tensor(seq_q_values)
        seq_k_cpu = as_int_tensor(seq_k_values)
        residual_cpu = as_int_tensor(residual_values)
        model = self.load_pytest_golden().GeneralizedLIV2(
            batch_size,
            q_seq,
            k_seq,
            q_t_size,
            k_t_size,
            q_head_num,
            k_head_num,
            head_dim,
            block_size,
            block_num,
            q.dtype,
            cu_q_cpu,
            cu_k_cpu,
            seq_q_cpu,
            seq_k_cpu,
            residual_cpu,
            layout_q,
            layout_k,
            topk,
            max_seqlen_q,
            mask_mode,
            cmp_ratio,
            return_value,
        )

        key_for_cpu = k
        if layout_k == "PA_BBND":
            if block_table is None or not seq_k_values:
                raise ValueError(
                    "PA_BBND compare context requires block_table and seqused_k"
                )
            key_for_cpu = self.restore_paged_key(
                k, block_table, batch_size, max(seq_k_values)
            )
        _, scores, _ = model.forward(
            q,
            key_for_cpu,
            w,
            cu_q_cpu,
            cu_k_cpu,
            seq_q_cpu,
            seq_k_cpu,
            residual_cpu,
            block_table,
            offset,
        )
        return {
            "params": params,
            "scores": scores,
            "topk_value": scores,
            "output_idx_offset": None if offset is None else offset.detach().cpu(),
            "score_layout": layout_q,
            "cu_seqlens_q": cu_q_cpu,
        }

    def customize(
        self,
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        layout_q,
        layout_k,
        kwargs,
    ):
        params = self.build_case_params(q, k, layout_q, layout_k, kwargs)
        golden_store = self.load_golden_store().CASE_DATA
        golden_store.clear()
        data = self.load_pytest_golden().generate_liv2_test_data(
            params, generate_golden=False
        )
        for name, dst, src_name in (
            ("q", q, "query"),
            ("k", k, "key"),
            ("w", w, "weights"),
            ("cu_seqlens_q", cu_seqlens_q, "cu_seqlens_q"),
            ("cu_seqlens_k", cu_seqlens_k, "cu_seqlens_k"),
            ("seqused_q", seqused_q, "seqused_q"),
            ("seqused_k", seqused_k, "seqused_k"),
            ("cmp_residual_k", cmp_residual_k, "cmp_residual_k_for_npu"),
            ("block_table", block_table, "block_table"),
            ("output_idx_offset", output_idx_offset, "output_idx_offset"),
        ):
            self.copy_tensor(dst, data.get(src_name), name)
        return data


INPUT_ADAPTER = LightningIndexerV2InputAdapter()


def rebuild_li_v2_compare_data(compare_context):
    return INPUT_ADAPTER.rebuild_compare_data(compare_context)


def generate_li_v2_inputs(
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
    **kwargs,
):
    """Populate pytest-derived inputs; metadata is filled by npu_preprocess."""
    if metadata is None:
        raise ValueError("LI_V2 direct API CSV must reserve the metadata tensor slot")
    params = dict(kwargs)
    params.update(
        {
            "topk": topk,
            "max_seqlen_q": max_seqlen_q,
            "layout_q": layout_q,
            "layout_k": layout_k,
            "mask_mode": mask_mode,
            "cmp_ratio": cmp_ratio,
            "return_value": return_value,
        }
    )
    data = INPUT_ADAPTER.customize(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        output_idx_offset,
        layout_q,
        layout_k,
        params,
    )
    zero_metadata(metadata)
    case_data = INPUT_ADAPTER.load_golden_store().CASE_DATA
    testcase_name = params.get("testcase_name")
    case_data.put(testcase_name, data)
    INPUT_ADAPTER.load_metadata_protocol().save_metadata_inputs(
        "lightning_indexer_v2", testcase_name, data.get("metadata_input")
    )
    return data


def zero_metadata(metadata):
    if torch.is_tensor(metadata):
        metadata.zero_()
    else:
        metadata.fill(0)


def generate_aclnn_li_v2_inputs(
    q,
    k,
    w,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    cmp_residual_k,
    block_table,
    output_idx_offset,
    metadata,
    topk,
    max_seqlen_q,
    layout_q,
    layout_k,
    mask_mode,
    cmp_ratio,
    return_value,
    sparse_indices_out,
    sparse_values_out,
    **kwargs,
):
    """Map the ACLNN C signature to the canonical pytest input adapter."""
    del sparse_indices_out, sparse_values_out
    return generate_li_v2_inputs(
        q,
        k,
        w,
        topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=max_seqlen_q,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=return_value,
        **kwargs,
    )
