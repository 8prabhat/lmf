"""Pure-numpy statistics for comparing ablation cells against a baseline.

No scipy dependency: ``welch_t_test``'s p-value is a normal-distribution
approximation (valid for the moderate-to-large degrees-of-freedom typical of
multi-seed ablation sweeps), documented as such below.
"""

from __future__ import annotations

import math

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
