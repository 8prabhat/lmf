"""Generic atomic file IO (DRY: one write-then-replace implementation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (write to ``.tmp`` then
    ``replace``), so a reader never observes a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=indent, sort_keys=True, default=str))
    tmp.replace(path)
