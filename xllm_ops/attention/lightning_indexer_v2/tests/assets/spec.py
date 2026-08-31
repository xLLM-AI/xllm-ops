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

"""TestSpec adapter for LightningIndexer V2 TTK assets."""

import importlib.util
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")


def load_impl_module(stem):
    name = f"li_v2_ttk_{stem}"
    if name in sys.modules:
        return sys.modules[name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise RuntimeError(
            "Failed to load LightningIndexerV2 assets module; "
            f"stage=impl/{stem}; module={path.resolve()}; "
            f"original error: {type(exc).__name__}: {exc}"
        ) from exc
    return module


npu_preprocess_module = load_impl_module("npu_preprocess")
golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
metadata_inputs_module = load_impl_module("metadata_inputs")
compare_module = load_impl_module("compare")


class LightningIndexerV2Spec:
    golden = golden_module.cpu_lightning_indexer_v2
    customize_inputs = inputs_module.generate_li_v2_inputs
    npu_preprocess = npu_preprocess_module.run
    tolerance = {
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }

    def compare(*outputs, compare_context=None, **kwargs):
        del kwargs
        testcase_name = (
            None if compare_context is None else compare_context.testcase_name
        )
        data = golden_module.get_compare_data(testcase_name)
        if data is None:
            if compare_context is None:
                raise RuntimeError(
                    "LightningIndexerV2 pytest compare requires compare_context"
                )
            data = inputs_module.rebuild_li_v2_compare_data(compare_context)
            golden_module.set_compare_data(compare_context.testcase_name, data)
        try:
            return compare_module.compare(*outputs, compare_data=data)
        finally:
            golden_module.discard_compare_data(data)


class AclnnLightningIndexerV2Spec(LightningIndexerV2Spec):
    golden = golden_module.cpu_aclnn_li_v2
    customize_inputs = inputs_module.generate_aclnn_li_v2_inputs
    npu_preprocess = None


class LightningIndexerMetadataSpec:
    customize_inputs = metadata_inputs_module.generate_lightning_indexer_metadata_inputs


__spec__ = {
    "torch.ops.cann_ops_transformer.lightning_indexer": "LightningIndexerV2Spec",
    "torch.ops.cann_ops_transformer.lightning_indexer_metadata": (
        "LightningIndexerMetadataSpec"
    ),
    "aclnnLightningIndexerV2": "AclnnLightningIndexerV2Spec",
}
