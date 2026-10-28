# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""Shared CPU precision and bitwise checks for MegaKDA tests."""
import math

import torch


# Per-rank shapes for GLM-5.3-Flash, not a distributed communication test.
GLM_KDA_HEADS = 64
TP_SIZES = (1, 2, 4, 8, 16, 32, 64)


def bitwise_equal(actual, expected):
    return (actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def error_metrics(actual, expected, *, near=0.01, relative=0.01, absolute=0.0005):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError("precision comparison requires identical shapes and dtypes")
    if not (math.isfinite(near) and near > 0 and math.isfinite(relative) and relative >= 0
            and math.isfinite(absolute) and absolute >= 0):
        raise ValueError("invalid error thresholds")
    actual, expected = actual.double(), expected.double()
    error = (actual - expected).abs()
    valid = torch.isfinite(actual) & torch.isfinite(expected) & torch.isfinite(error)
    normal = expected.abs() >= near
    small = ~normal
    rel = error[normal] / expected[normal].abs()
    rel_valid = valid[normal] & torch.isfinite(rel)
    # Invalid values must count as violations, not disappear behind NaN comparisons.
    error = torch.where(valid, error, float("inf"))
    rel = torch.where(rel_valid, rel, float("inf"))
    abs_near = error[small]

    def maximum(x):
        return float(x.max()) if x.numel() else 0.0

    def fraction(x):
        return float(x.double().mean()) if x.numel() else 0.0

    return dict(finite=bool(valid.all() & rel_valid.all()), max_abs=maximum(error),
                normal_count=int(normal.sum()), near_count=int(small.sum()),
                normal_max_rel=maximum(rel), normal_over_fraction=fraction(rel > relative),
                near_max_abs=maximum(abs_near), near_over_fraction=fraction(abs_near > absolute))
