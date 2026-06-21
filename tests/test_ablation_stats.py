"""Pure-numpy ablation statistics."""

from __future__ import annotations

import math

import pytest

from lmf.ablation.stats import (
    analytic_confidence_interval,
    bootstrap_ci,
    cohens_d,
    compare_to_baseline,
    holm_adjust,
    mean_std_stderr,
    paired_bootstrap_interval,
    percentile,
    sign_test_paired,
    welch_t_test,
)


def test_mean_std_stderr():
    out = mean_std_stderr([1.0, 2.0, 3.0])
    assert out["n"] == 3
    assert math.isclose(out["mean"], 2.0)
    assert math.isclose(out["std"], 1.0)
    assert math.isclose(out["stderr"], 1.0 / math.sqrt(3))


def test_mean_std_stderr_single_value():
    out = mean_std_stderr([5.0])
    assert out["n"] == 1
    assert out["mean"] == 5.0
    assert out["std"] == 0.0
    assert out["stderr"] == 0.0


def test_welch_t_test_identical_samples():
    out = welch_t_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert math.isclose(out["t"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["p_value"], 1.0, abs_tol=1e-9)


def test_welch_t_test_clearly_different_samples():
    a = [10.0, 10.1, 9.9, 10.05, 9.95]
    b = [1.0, 1.1, 0.9, 1.05, 0.95]
    out = welch_t_test(a, b)
    assert out["t"] > 0
    assert out["p_value"] < 0.01


def test_cohens_d_zero_for_identical_means():
    assert math.isclose(cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0, abs_tol=1e-9)


def test_cohens_d_large_for_separated_samples():
    d = cohens_d([10.0, 11.0, 12.0], [1.0, 2.0, 3.0])
    assert d > 2.0


def test_bootstrap_ci_brackets_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi = bootstrap_ci(values, n_resamples=500, seed=0)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_compare_to_baseline_structure():
    out = compare_to_baseline([2.0, 2.2, 2.1], [1.0, 1.1, 0.9])
    assert out["delta_mean"] > 0
    assert "cohens_d" in out
    assert "welch_t_test" in out
    assert "delta_bootstrap_ci" in out
    assert out["cell"]["n"] == 3
    assert out["baseline"]["n"] == 3


def test_percentile_matches_known_quantiles():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert math.isclose(percentile(values, 0.0), 1.0)
    assert math.isclose(percentile(values, 1.0), 5.0)
    assert math.isclose(percentile(values, 0.5), 3.0)


def test_percentile_empty_is_nan():
    assert math.isnan(percentile([], 0.5))


def test_paired_bootstrap_interval_brackets_mean():
    differences = [1.0, 1.2, 0.8, 1.1, 0.9]
    out = paired_bootstrap_interval(differences, seed=0, samples=500)
    assert out["lower"] <= out["mean"] <= out["upper"]
    assert out["samples"] == 500


def test_paired_bootstrap_interval_requires_values():
    with pytest.raises(ValueError):
        paired_bootstrap_interval([], seed=0)


def test_holm_adjust_orders_by_significance():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    # Holm-adjusted p-values are monotonically non-decreasing in rank order.
    ordered = sorted(adjusted.items(), key=lambda item: {"a": 0.01, "b": 0.04, "c": 0.5}[item[0]])
    assert ordered[0][1] <= ordered[1][1] <= ordered[2][1]


def test_sign_test_paired_all_positive_is_significant():
    assert sign_test_paired([1.0, 2.0, 3.0, 4.0, 5.0]) < 0.1


def test_sign_test_paired_balanced_is_not_significant():
    assert math.isclose(sign_test_paired([1.0, -1.0, 2.0, -2.0]), 1.0)


def test_analytic_confidence_interval_single_value():
    out = analytic_confidence_interval([5.0])
    assert out == {"mean": 5.0, "lower": 5.0, "upper": 5.0}


def test_analytic_confidence_interval_three_seeds_uses_t_critical():
    out = analytic_confidence_interval([1.0, 2.0, 3.0])
    assert out["lower"] < out["mean"] < out["upper"]
