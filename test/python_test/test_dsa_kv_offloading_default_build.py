# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Ensure the standard release selector includes every DSA KV offloading internal stage."""

import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "xllm_ops/build_aclnn.sh"
DSA_KV_OFFLOADING_STAGES = {
    "dsa_kv_offloading_plan_nope",
    "dsa_kv_offloading_plan_rope",
    "dsa_kv_offloading_serve_nope",
    "dsa_kv_offloading_serve_rope",
}


def default_ops(soc: str) -> set[str]:
    source = SOURCE.read_text()
    branch = source.split(f'elif [[ "$SOC_VERSION" =~ ^{soc}', 1)[1].split(
        "elif [[", 1
    )[0]
    array = branch.split("CUSTOM_OPS_ARRAY=(", 1)[1].split(")", 1)[0]
    return set(re.findall(r'^\s*"([a-z0-9_]+)"', array, re.MULTILINE))


def test_a3_default_contains_dsa_kv_offloading_and_existing_ops() -> None:
    selected = default_ops("ascend910_93")
    assert selected >= DSA_KV_OFFLOADING_STAGES
    assert {
        "x_flash_attention_infer",
        "recurrent_gated_delta_rule",
        "mtp_prepare_next_draft",
    } <= selected


def test_a5_default_does_not_select_a3_only_c0() -> None:
    assert not DSA_KV_OFFLOADING_STAGES & default_ops("ascend950")
