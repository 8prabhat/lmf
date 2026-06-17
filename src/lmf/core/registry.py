"""A tiny, generic registry — the framework's Open/Closed extension point.

Models, corpora, and trainers register themselves by name. New families are
added by writing a module and importing it; no dispatch code in the framework
needs to change. Each registry is an independent namespace so the same short
name (e.g. ``"rhca"``) can denote a model builder and a trainer builder without
collision.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name → factory mapping with a decorator-based registration API."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def _decorator(factory: Callable[..., T]) -> Callable[..., T]:
            key = name.lower()
            if key in self._entries:
                raise KeyError(f"{self._kind} {name!r} is already registered")
            self._entries[key] = factory
            return factory

        return _decorator

    def get(self, name: str) -> Callable[..., T]:
        key = name.lower()
        if key not in self._entries:
            raise KeyError(
                f"unknown {self._kind} {name!r}; registered: {sorted(self._entries)}"
            )
        return self._entries[key]

    def create(self, name: str, *args, **kwargs) -> T:
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._entries


# Framework-wide registries. Families populate these at import time.
MODELS: Registry = Registry("model")
TRAINERS: Registry = Registry("trainer")
CORPORA: Registry = Registry("corpus")
