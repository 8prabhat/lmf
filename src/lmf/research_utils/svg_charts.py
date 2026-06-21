"""Zero-dependency SVG bar charts for auditable research/gate reports.

Deliberately hand-rolled rather than matplotlib-based: gate-decision artifacts
(``finalize_*_gate.py``, ``summarize_*.py``) must always render even in an
environment without matplotlib installed. For ablation sweep plots, where
matplotlib being unavailable is an acceptable best-effort fallback, see
``lmf.ablation.report.maybe_plot`` instead -- the two are not merged because
they have different dependency-availability contracts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

_PALETTE = ("#2166ac", "#777777", "#b2182b")
_POSITIVE = "#b2182b"
_NEGATIVE = "#2166ac"


def bar_chart_svg(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    *,
    diverging: bool = False,
    baseline: float = 0.0,
    lower_is_better: bool | None = None,
) -> None:
    """Write a labeled bar chart to ``path`` as a standalone SVG file.

    ``diverging=True`` draws bars rising or falling from ``baseline`` (a
    zero-centered line), colored red above / blue below -- for paired
    differences against a reference. ``diverging=False`` (default) draws
    ordinary bars from a bottom axis, colored by a cycling palette, with an
    optional "lower/higher is better" footer when ``lower_is_better`` is set.
    """
    if diverging:
        _diverging_bar_chart_svg(path, labels, values, title, baseline=baseline)
    else:
        _ordinary_bar_chart_svg(path, labels, values, title, lower_is_better=lower_is_better)


def _diverging_bar_chart_svg(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    *,
    baseline: float,
) -> None:
    width, height, margin = 760, 360, 55
    maximum = max([abs(value - baseline) for value in values] + [1e-6])
    zero_y = height / 2
    bar_width = (width - 2 * margin) / max(len(values), 1)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{zero_y}" x2="{width-margin}" y2="{zero_y}" stroke="black"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin + index * bar_width + 0.15 * bar_width
        delta = value - baseline
        bar_height = abs(delta) / maximum * (height / 2 - 70)
        y = zero_y - bar_height if delta >= 0 else zero_y
        color = _POSITIVE if delta > 0 else _NEGATIVE
        elements.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{0.7*bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>',
                f'<text x="{x+0.35*bar_width:.1f}" y="{height-28}" text-anchor="middle" font-size="11">{label}</text>',
                f'<text x="{x+0.35*bar_width:.1f}" y="{y-5 if delta>=0 else y+bar_height+15:.1f}" text-anchor="middle" font-size="10">{value:.4g}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements))


def _ordinary_bar_chart_svg(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    *,
    lower_is_better: bool | None,
) -> None:
    width, height, margin = 760, 390, 70
    maximum = max(values) * 1.15
    plot_height = height - 2 * margin
    bar_width = (width - 2 * margin) / len(values)
    rows = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="black"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = margin + index * bar_width + 0.18 * bar_width
        bar_height = value / maximum * plot_height
        y = height - margin - bar_height
        rows.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{0.64*bar_width:.1f}" height="{bar_height:.1f}" fill="{_PALETTE[index % len(_PALETTE)]}"/>',
                f'<text x="{x+0.32*bar_width:.1f}" y="{height-margin+22}" text-anchor="middle" font-size="12">{label}</text>',
                f'<text x="{x+0.32*bar_width:.1f}" y="{y-7:.1f}" text-anchor="middle" font-size="12">{value:.4g}</text>',
            ]
        )
    if lower_is_better is not None:
        direction = "lower" if lower_is_better else "higher"
        rows.append(
            f'<text x="{width-10}" y="{height-10}" text-anchor="end" font-size="10">{direction} is better</text>'
        )
    rows.append("</svg>")
    path.write_text("\n".join(rows))
