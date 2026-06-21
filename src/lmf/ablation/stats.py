"""Pure-numpy statistics for comparing ablation cells against a baseline.

No scipy dependency: ``welch_t_test``'s p-value is a normal-distribution
approximation (valid for the moderate-to-large degrees-of-freedom typical of
multi-seed ablation sweeps), documented as such below.
"""

from __future__ import annotations

import math
import statistics

import numpy as np


def mean_std_stderr(values: list[float]) -> dict[str, float]:
    """Sample mean, (ddof=1) standard deviation, and standard error of the mean."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean()) if n else float("nan")
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    stderr = std / math.sqrt(n) if n > 1 else 0.0
    return {"mean": mean, "std": std, "stderr": stderr, "n": n}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Two-sided Welch's t-test.

    ``p_value`` is computed via a normal-distribution approximation to the
    t-distribution (``2 * (1 - Phi(|t|))``) rather than the exact
    incomplete-beta-based Student-t CDF, to avoid a scipy dependency. This is
    a good approximation once ``df`` is more than a handful, which covers
    typical multi-seed ablation sweeps; for very small ``n`` it is
    conservative-ish but not exact — treat ``p_value`` as a triage signal.
    """
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = arr_a.size, arr_b.size
    if n_a < 2 or n_b < 2:
        return {"t": float("nan"), "df": float("nan"), "p_value": float("nan")}

    mean_a, mean_b = arr_a.mean(), arr_b.mean()
    var_a, var_b = arr_a.var(ddof=1), arr_b.var(ddof=1)
    se_a, se_b = var_a / n_a, var_b / n_b
    denom = math.sqrt(se_a + se_b)
    if denom == 0.0:
        return {"t": 0.0, "df": float(n_a + n_b - 2), "p_value": 1.0}

    t = float((mean_a - mean_b) / denom)
    df_num = (se_a + se_b) ** 2
    df_den = (se_a ** 2 / (n_a - 1)) + (se_b ** 2 / (n_b - 1))
    df = float(df_num / df_den) if df_den else float(n_a + n_b - 2)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(t)))
    return {"t": t, "df": df, "p_value": p_value}


def cohens_d(a: list[float], b: list[float]) -> float:
    """Pooled-standard-deviation effect size between samples ``a`` and ``b``."""
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = arr_a.size, arr_b.size
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a, var_b = arr_a.var(ddof=1), arr_b.var(ddof=1)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0.0:
        return 0.0
    return float((arr_a.mean() - arr_b.mean()) / pooled)


def bootstrap_ci(values: list[float], n_resamples: int = 2000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    resampled_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=arr.size, replace=True)
        resampled_means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(resampled_means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of ``values`` at ``fraction`` in ``[0, 1]``."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    position = float(fraction) * (len(ordered) - 1)
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return ordered[left]
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def paired_bootstrap_interval(
    differences: list[float],
    *,
    seed: int,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap CI of the mean of paired ``differences``, in the dict shape used
    by qualification/gate reports. Thin wrapper around ``bootstrap_ci``."""
    if not differences:
        raise ValueError("bootstrap requires at least one paired difference")
    lower, upper = bootstrap_ci(differences, n_resamples=samples, ci=confidence, seed=seed)
    return {
        "mean": float(np.mean(differences)),
        "lower": lower,
        "upper": upper,
        "samples": samples,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment for a family of named p-values."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * float(value)))
        adjusted[name] = running
    return adjusted


def sign_test_paired(values: list[float]) -> float:
    """Two-sided exact sign test p-value for a list of paired differences."""
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    count = positive + negative
    if count == 0:
        return 1.0
    tail = min(positive, negative)
    probability = sum(math.comb(count, index) for index in range(tail + 1)) / 2**count
    return min(1.0, 2.0 * probability)


def analytic_confidence_interval(values: list[float]) -> dict[str, float]:
    """Analytic (Student-t for n=3, else normal-approximation) confidence
    interval of the mean of a small sample. Prefer ``bootstrap_ci`` /
    ``paired_bootstrap_interval`` when the sample is large enough to resample;
    this is for the very-small-n (e.g. 3-seed) regime where a closed-form
    interval is the established convention in this repo's gate reports."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": mean, "lower": mean, "upper": mean}
    standard_error = statistics.stdev(values) / math.sqrt(len(values))
    critical = 4.303 if len(values) == 3 else 1.96  # Student-t df=2 vs normal approx.
    radius = critical * standard_error
    return {"mean": mean, "lower": mean - radius, "upper": mean + radius}


def compare_to_baseline(cell_values: list[float], baseline_values: list[float]) -> dict:
    """Bundle delta-mean, Cohen's d, Welch's t-test, and a bootstrap CI of the
    delta into one dict, ready for ``report.py``."""
    cell_stats = mean_std_stderr(cell_values)
    baseline_stats = mean_std_stderr(baseline_values)
    delta_mean = cell_stats["mean"] - baseline_stats["mean"]

    arr_cell, arr_base = np.asarray(cell_values, dtype=float), np.asarray(baseline_values, dtype=float)
    n = min(arr_cell.size, arr_base.size)
    delta_ci = bootstrap_ci(list((arr_cell[:n] - arr_base[:n])) if n else [], seed=0) if n else (float("nan"), float("nan"))

    return {
        "cell": cell_stats,
        "baseline": baseline_stats,
        "delta_mean": delta_mean,
        "cohens_d": cohens_d(cell_values, baseline_values),
        "welch_t_test": welch_t_test(cell_values, baseline_values),
        "delta_bootstrap_ci": delta_ci,
    }
