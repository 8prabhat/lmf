"""Content-hashing utilities for reproducibility and provenance (DRY: one
implementation of each hash, not five copies scattered across scripts)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(value: Any) -> str:
    """SHA-256 of ``value``'s canonical (sorted-key) JSON encoding."""
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_tree_sha256(
    paths: Sequence[str] = ("src", "scripts", "configs", "tests", "pyproject.toml"),
) -> str:
    """SHA-256 over the name+content of every tracked or untracked-but-not-ignored
    file under ``paths``, for certifying which exact code produced a result."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", *paths],
        check=False,
        capture_output=True,
        text=True,
    )
    digest = hashlib.sha256()
    for name in sorted(result.stdout.splitlines()):
        path = Path(name)
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()
