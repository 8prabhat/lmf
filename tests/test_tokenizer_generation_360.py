from __future__ import annotations

import pytest

from scripts.benchmark_tokenizer_generation import _make_span_example
from scripts.benchmark_tokenizer_generation_360 import (
    METRICS,
    PRIMARY_CONTRASTS,
    T_CRITICAL_95,
    VARIANTS,
    _evenly_spaced_eval_indices,
    _paired_summary,
    _summarize,
)


def test_ten_seed_interval_uses_student_t_not_normal_approximation():
    assert T_CRITICAL_95[9] == pytest.approx(2.262)
    summary = _paired_summary(list(range(10)), [0.0] * 10, "higher")
    assert summary["ci95"][1] - summary["mean"] > 1.96 * summary["stdev"] / 10**0.5


def test_evenly_spaced_evaluation_selection_covers_each_language():
    examples = [
        _make_span_example(language, f"alpha beta gamma {index}", 64, 4)
        for language in ("a", "b")
        for index in range(10)
    ]
    selected = _evenly_spaced_eval_indices(examples, list(range(20)), 3)
    assert selected == [0, 4, 9, 10, 14, 19]


def test_paired_summary_respects_lower_is_better():
    summary = _paired_summary([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], "lower")
    assert summary["wins"] == 1
    assert summary["ties"] == 1
    assert summary["losses"] == 1


def test_partial_report_summary_only_pairs_common_complete_seeds():
    metrics = {metric: 1.0 for metric in METRICS}
    report = {
        "seeds": [0, 1],
        "variants": {variant: {} for variant in VARIANTS},
        "tasks": {
            task: {
                "variants": {
                    variant: {
                        "runs": {
                            "0": metrics,
                            **({"1": metrics} if variant == PRIMARY_CONTRASTS[0][0] else {}),
                        }
                    }
                    for variant in VARIANTS
                }
            }
            for task in ("a", "b")
        },
    }
    _summarize(report)
    contrast = report["aggregate_macro_across_tasks"]["paired_primary_contrasts"][
        "multigear_compositional_vs_sentencepiece_bpe"
    ]
    assert contrast["exact_match_pct"]["by_seed"] == [0.0]
