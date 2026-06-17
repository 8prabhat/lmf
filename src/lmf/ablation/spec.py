"""Parse a standalone ``configs/ablations/*.yaml`` ablation spec.

An ablation spec is a *different* document from an experiment config: it
points at an experiment config (``base_config``/``base_block``/``base_env``)
and describes a sweep over it, rather than describing a single run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class AxisSpec:
    """One sweep axis.

    ``path`` is either a dotted config-override path (consumed by
    ``apply_overrides``, e.g. ``"model.dim"``), or a prefixed structural/
    loss-term axis (``"structural:<point-name>"`` / ``"loss_term:<term>"``,
    see ``lmf.ablation.points`` and ``lmf.ablation.loss_terms``).
    """

    path: str
    values: list[Any]
    label: str | None = None

    @property
    def kind(self) -> Literal["config", "structural", "loss_term"]:
        if self.path.startswith("structural:"):
            return "structural"
        if self.path.startswith("loss_term:"):
            return "loss_term"
        return "config"

    @property
    def target(self) -> str:
        """The part of ``path`` after the ``kind:`` prefix (or ``path`` itself)."""
        return self.path.split(":", 1)[1] if ":" in self.path else self.path

    @property
    def display_name(self) -> str:
        return self.label or self.path


@dataclass(frozen=True)
class VariantSpec:
    """One named variant for ``mode: named_variants``.

    ``overrides`` is an arbitrary nested-dict overlay, deep-merged onto the
    resolved base config — it may include ``{"model": {"name": "opet", ...}}``
    to swap the model family entirely.
    """

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    base_config: str
    base_block: str | None = None
    base_env: str | None = None
    mode: Literal["grid", "one_at_a_time", "named_variants", "pairwise"] = "grid"
    axes: list[AxisSpec] = field(default_factory=list)
    variants: list[VariantSpec] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    run: dict[str, Any] = field(default_factory=dict)
    eval: dict[str, Any] = field(default_factory=dict)
    metric: str = "bits_per_token"
    goal: Literal["minimize", "maximize"] = "minimize"
    budget: dict[str, Any] | None = None
    results_dir: str | None = None


def load_ablation_spec(path: str | Path) -> AblationSpec:
    """Load an ``AblationSpec`` from YAML.

    Accepts either a top-level ``ablation:`` key (preferred) or a flat
    document whose top level *is* the spec.
    """
    doc = yaml.safe_load(Path(path).read_text()) or {}
    block = doc.get("ablation", doc)

    axes = [AxisSpec(**a) for a in block.get("axes", [])]
    variants = [VariantSpec(**v) for v in block.get("variants", [])]

    return AblationSpec(
        name=block["name"],
        base_config=block["base_config"],
        base_block=block.get("base_block"),
        base_env=block.get("base_env"),
        mode=block.get("mode", "grid"),
        axes=axes,
        variants=variants,
        seeds=list(block.get("seeds", [0])),
        run=dict(block.get("run", {})),
        eval=dict(block.get("eval", {})),
        metric=block.get("metric", "bits_per_token"),
        goal=block.get("goal", "minimize"),
        budget=block.get("budget"),
        results_dir=block.get("results_dir"),
    )
