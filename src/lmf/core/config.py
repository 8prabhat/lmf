"""Config loading and deterministic deep-merge.

A single YAML file describes an experiment. The framework merges, in order:

  1. the ``base`` block (shared defaults), if present;
  2. an environment overlay block (``environments.<env>``), if present;
  3. the requested named block (e.g. ``smoke`` / ``production_v4``);
  4. CLI ``--set key=value`` overrides.

Later sources win. Nested dicts merge key-by-key; scalars and lists replace.
This keeps configuration the single source of truth (no behaviour hard-coded in
Python) while staying simple enough to reason about.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto a copy of ``base`` (overlay wins)."""
    result = deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _coerce(value: str) -> Any:
    """Parse a CLI override string into a Python scalar where possible."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.subkey=value`` dotted overrides to a config dict in place."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(raw)
    return cfg


class ExperimentConfig:
    """Resolved configuration for one experiment block."""

    def __init__(self, raw: dict[str, Any], block: str) -> None:
        self.raw = raw
        self.block = block

    @property
    def model(self) -> dict[str, Any]:
        return dict(self.raw.get("model", {}))

    @property
    def trainer(self) -> dict[str, Any]:
        return dict(self.raw.get("trainer", {}))

    @property
    def data(self) -> dict[str, Any]:
        return dict(self.raw.get("data", {}))

    @property
    def evaluation(self) -> dict[str, Any]:
        return dict(self.raw.get("evaluation", {}))

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load_config(
    path: str | Path,
    block: str | None = None,
    env: str | None = None,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Load and resolve an experiment config.

    The YAML's top level may contain a ``base`` block, an ``environments`` map,
    and any number of named experiment blocks. ``block`` selects which named
    block to overlay; if omitted, ``default_block`` from the file (or ``"smoke"``)
    is used.
    """
    doc = yaml.safe_load(Path(path).read_text()) or {}
    selected = block or doc.get("default_block", "smoke")

    merged: dict[str, Any] = {}
    if isinstance(doc.get("base"), dict):
        merged = deep_merge(merged, doc["base"])
    if env and isinstance(doc.get("environments", {}).get(env), dict):
        merged = deep_merge(merged, doc["environments"][env])
    if selected not in doc:
        raise KeyError(
            f"config block {selected!r} not found in {path}; "
            f"available: {[k for k in doc if k not in {'base', 'environments', 'default_block'}]}"
        )
    if not isinstance(doc[selected], dict):
        raise TypeError(f"config block {selected!r} must be a mapping")
    merged = deep_merge(merged, doc[selected])
    merged = apply_overrides(merged, overrides)
    return ExperimentConfig(merged, selected)
