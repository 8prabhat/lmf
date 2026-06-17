"""Model families. Importing this package registers every family's builders.

Adding a new family is a matter of creating a subpackage that calls
``MODELS.register`` / ``TRAINERS.register`` and importing it here — no framework
code changes (Open/Closed).
"""

from __future__ import annotations

from . import gear_transformer, native, opet, rhca, transformer  # noqa: F401  (import side-effect: registration)

__all__ = ["gear_transformer", "native", "opet", "rhca", "transformer"]
