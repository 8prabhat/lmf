"""Model families. Importing this package registers every family's builders.

Adding a new family is a matter of creating a subpackage that calls
``MODELS.register`` / ``TRAINERS.register`` and importing it here — no framework
code changes (Open/Closed).
"""

from __future__ import annotations

from . import (  # noqa: F401  (import side-effect: registration)
    bounded_hybrid_gear,
    gear_transformer,
    gru,
    native,
    opet,
    pure_parallel_gear,
    rhca,
    transformer,
)

__all__ = [
    "bounded_hybrid_gear",
    "gear_transformer",
    "gru",
    "native",
    "opet",
    "pure_parallel_gear",
    "rhca",
    "transformer",
]
