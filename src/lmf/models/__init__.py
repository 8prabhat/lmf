"""Model families. Importing this package registers every family's builders.

Adding a new family is a matter of creating a subpackage that calls
``MODELS.register`` / ``TRAINERS.register`` and importing it here — no framework
code changes (Open/Closed).
"""

from __future__ import annotations

from . import (  # noqa: F401  (import side-effect: registration)
    gear_transformer,
    gru,
    native,
    opet,
    pure_parallel_gear,
    pure_parallel_gear_v3,
    rhca,
    transformer,
)

__all__ = [
    "gear_transformer",
    "gru",
    "native",
    "opet",
    "pure_parallel_gear",
    "pure_parallel_gear_v3",
    "rhca",
    "transformer",
]
