# Copyright 2026 The xLLM Authors. All Rights Reserved.
"""CPU regressions for fail-closed precision and replay checks."""
import pytest
import torch

from mega_kda_test_utils import bitwise_equal, error_metrics


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", ["actual", "expected", "both"])
def test_nonfinite_never_passes(bad, side):
    actual = torch.tensor([bad if side != "expected" else 0.0])
    expected = torch.tensor([bad if side != "actual" else 0.0])
    stats = error_metrics(actual, expected)
    assert not stats["finite"], stats
    assert stats["normal_over_fraction"] + stats["near_over_fraction"] > 0, stats


def test_finite_subtraction_overflow_never_passes():
    limit = torch.finfo(torch.float64).max
    stats = error_metrics(torch.tensor([limit], dtype=torch.float64),
                          torch.tensor([-limit], dtype=torch.float64))
    assert not stats["finite"], stats


def test_error_groups_and_thresholds():
    expected = torch.tensor([1.0, -2.0, 0.0, 0.0001], dtype=torch.float64)
    actual = expected + torch.tensor([0.02, -0.002, 0.001, 0.00001], dtype=torch.float64)
    stats = error_metrics(actual, expected)
    assert stats["finite"]
    assert stats["normal_count"] == stats["near_count"] == 2
    assert stats["normal_over_fraction"] == stats["near_over_fraction"] == 0.5
    assert stats["normal_max_rel"] == pytest.approx(0.02)
    assert stats["near_max_abs"] == pytest.approx(0.001)


@pytest.mark.parametrize("values", [[], [0.0], [1.0]])
def test_empty_error_groups(values):
    value = torch.tensor(values)
    stats = error_metrics(value, value)
    assert stats["finite"]
    assert stats["normal_count"] + stats["near_count"] == len(values)
    for name in ("max_abs", "normal_max_rel", "near_max_abs",
                 "normal_over_fraction", "near_over_fraction"):
        assert stats[name] == 0


def test_relative_overflow_never_passes():
    stats = error_metrics(torch.tensor([1e300], dtype=torch.float64),
                          torch.tensor([1e-300], dtype=torch.float64), near=1e-301)
    assert not stats["finite"]
    assert stats["normal_over_fraction"] == 1


@pytest.mark.parametrize("mismatch", ["shape", "dtype"])
def test_comparison_rejects_mismatch(mismatch):
    value = torch.ones(2)
    other = value.reshape(1, 2) if mismatch == "shape" else value.double()
    with pytest.raises(ValueError, match="shapes and dtypes"):
        error_metrics(value, other)
    assert not bitwise_equal(value, other)


@pytest.mark.parametrize("options", [{"near": 0}, {"near": float("nan")},
                                     {"relative": -1}, {"absolute": float("inf")}])
def test_comparison_rejects_invalid_thresholds(options):
    with pytest.raises(ValueError, match="thresholds"):
        error_metrics(torch.zeros(1), torch.zeros(1), **options)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
def test_bitwise_distinguishes_signed_zero(dtype):
    positive, negative = torch.tensor([0.0], dtype=dtype), torch.tensor([-0.0], dtype=dtype)
    assert torch.equal(positive, negative)
    assert not bitwise_equal(positive, negative)
    assert bitwise_equal(positive, positive.clone())


def test_bitwise_noncontiguous_and_scalar():
    value = torch.arange(6).view(2, 3).t()
    assert bitwise_equal(value, value.contiguous())
    assert bitwise_equal(torch.tensor(1.0), torch.tensor(1.0))
    assert bitwise_equal(torch.empty(0), torch.empty(0))


@pytest.mark.parametrize("operator", ["prefill", "decode"])
@pytest.mark.parametrize("name", ["output", "conv_state_out", "ssm_state_out"])
@pytest.mark.parametrize("side", ["actual", "expected", "both"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_native_gate_rejects_nonfinite(monkeypatch, operator, name, side, bad):
    import test_mega_kda_decode as decode
    import test_mega_kda_prefill as prefill

    module = prefill if operator == "prefill" else decode
    data = module.make_case(heads=1)
    expected = module.reference(data)
    actual = dict(data, **{key: value.clone() for key, value in zip(module.OUTPUTS, expected)})
    index = module.OUTPUTS.index(name)
    if side != "expected":
        actual[name].flatten()[0] = bad
    if side != "actual":
        expected[index].flatten()[0] = bad
    monkeypatch.setattr(module, "reference", lambda _: expected)
    with pytest.raises(AssertionError):
        if operator == "prefill":
            module.check(data, actual, "nonfinite")
        else:
            module.assert_native(data, actual)
