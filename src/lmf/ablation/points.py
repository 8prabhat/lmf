"""Generic structural-ablation points: discover and bypass ``nn.Module`` submodules.

Operates purely on ``nn.Module`` structure (``named_modules()``, ``nn.ModuleList``
discovery, runtime tensor/tuple/dict shape inspection). No model family is ever
named here -- this module works unmodified for any architecture made of standard
PyTorch modules. An optional, duck-typed ``model.ablation_points()`` hook (see
``AblationAddressable`` in ``core.interfaces``) can contribute additional named
points; this module works fine when that hook is absent.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
from torch import nn


class BypassError(Exception):
    """Raised when a submodule cannot be safely bypassed/zeroed."""


@dataclass(frozen=True)
class PointSpec:
    """Describes one ablation point.

    ``mode`` is one of ``"identity"`` (replace the primary output tensor with
    the primary input tensor), ``"zero"`` (zero every tensor leaf of the
    output), or ``"custom"`` (delegate entirely to ``factory``).
    """

    name: str
    path: str
    mode: str
    factory: Callable[[nn.Module], contextlib.AbstractContextManager[None]] | None = None


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def _find_primary_tensor(args: tuple, kwargs: dict) -> tuple[str, Any] | None:
    """Locate the first ``Tensor`` among positional args (then kwargs)."""
    for i, a in enumerate(args):
        if _is_tensor(a):
            return ("args", i)
    for k, v in kwargs.items():
        if _is_tensor(v):
            return ("kwargs", k)
    return None


def _zero_like_structure(out: Any) -> Any:
    if _is_tensor(out):
        return torch.zeros_like(out)
    if isinstance(out, tuple):
        return tuple(_zero_like_structure(o) for o in out)
    if isinstance(out, list):
        return [_zero_like_structure(o) for o in out]
    if isinstance(out, dict):
        return {k: _zero_like_structure(v) for k, v in out.items()}
    return out


def _replace_primary_in_output(out: Any, primary_input: torch.Tensor, path: str) -> Any:
    """Replace the first ``Tensor`` leaf of ``out`` with ``primary_input``.

    Raises ``BypassError`` if no tensor leaf is found or its shape doesn't
    match ``primary_input`` -- in that case identity bypass isn't safe.
    """
    if _is_tensor(out):
        if out.shape != primary_input.shape:
            raise BypassError(
                f"{path}: output tensor shape {tuple(out.shape)} != "
                f"input tensor shape {tuple(primary_input.shape)}")
        return primary_input
    if isinstance(out, tuple):
        for i, o in enumerate(out):
            if _is_tensor(o):
                if o.shape != primary_input.shape:
                    raise BypassError(
                        f"{path}: output tensor shape {tuple(o.shape)} != "
                        f"input tensor shape {tuple(primary_input.shape)}")
                return out[:i] + (primary_input,) + out[i + 1:]
        raise BypassError(f"{path}: no Tensor found in tuple output for identity bypass")
    if isinstance(out, list):
        for i, o in enumerate(out):
            if _is_tensor(o):
                if o.shape != primary_input.shape:
                    raise BypassError(
                        f"{path}: output tensor shape {tuple(o.shape)} != "
                        f"input tensor shape {tuple(primary_input.shape)}")
                new = list(out)
                new[i] = primary_input
                return new
        raise BypassError(f"{path}: no Tensor found in list output for identity bypass")
    if isinstance(out, dict):
        for k, v in out.items():
            if _is_tensor(v):
                if v.shape != primary_input.shape:
                    raise BypassError(
                        f"{path}: output tensor shape {tuple(v.shape)} != "
                        f"input tensor shape {tuple(primary_input.shape)}")
                new = dict(out)
                new[k] = primary_input
                return new
        raise BypassError(f"{path}: no Tensor found in dict output for identity bypass")
    raise BypassError(f"{path}: unsupported output type {type(out)!r} for identity bypass")


@contextlib.contextmanager
def bypass_module(model: nn.Module, path: str, mode: str = "identity") -> Iterator[None]:
    """Monkeypatch ``model.get_submodule(path).forward`` for the ``with`` block.

    ``mode="identity"`` replaces the primary output tensor with the primary
    input tensor (passing the submodule through unchanged from the rest of
    the network's perspective). ``mode="zero"`` zeroes every tensor leaf of
    the output. Both modes still call the original ``forward`` to learn the
    output structure/shapes -- ``"zero"`` needs this to build correctly
    shaped zero tensors, and ``"identity"`` needs it to locate the primary
    output tensor to replace.

    Raises ``BypassError`` (synchronously, on first call inside the ``with``
    block) if no safe primary-tensor correspondence exists.
    """
    if mode not in ("identity", "zero"):
        raise ValueError(f"unknown bypass mode {mode!r}")

    submodule = model.get_submodule(path)
    original_forward = submodule.forward

    if mode == "identity":
        def patched_forward(*args: Any, **kwargs: Any) -> Any:
            loc = _find_primary_tensor(args, kwargs)
            if loc is None:
                raise BypassError(f"{path}: no Tensor argument found for identity bypass")
            kind, key = loc
            primary_input = args[key] if kind == "args" else kwargs[key]
            out = original_forward(*args, **kwargs)
            return _replace_primary_in_output(out, primary_input, path)
    else:
        def patched_forward(*args: Any, **kwargs: Any) -> Any:
            out = original_forward(*args, **kwargs)
            return _zero_like_structure(out)

    submodule.forward = patched_forward
    try:
        yield
    finally:
        submodule.forward = original_forward


def skip_listed_module(model: nn.Module, list_path: str, index: int) -> contextlib.AbstractContextManager[None]:
    """Skip element ``index`` of the ``nn.ModuleList`` at ``list_path`` (identity bypass)."""
    return bypass_module(model, f"{list_path}.{index}", mode="identity")


def discover_points(model: nn.Module) -> dict[str, PointSpec]:
    """Walk ``model.named_modules()`` and return all generically-discoverable points.

    - Every ``nn.ModuleList`` with length > 1 contributes one ``"<path>.skip[i]"``
      point per element (identity bypass of that element).
    - Every submodule with direct parameters contributes a ``"<path>.bypass"``
      (identity) and a ``"<path>.zero"`` point.
    - If ``model`` implements ``ablation_points()`` (see
      ``core.interfaces.AblationAddressable``), those named points are merged
      in last, overriding any name collision (with a warning).
    """
    points: dict[str, PointSpec] = {}
    for path, module in model.named_modules():
        if path == "":
            continue
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            for i in range(len(module)):
                name = f"{path}.skip[{i}]"
                points[name] = PointSpec(name=name, path=f"{path}.{i}", mode="identity")
        if any(True for _ in module.parameters(recurse=False)):
            bypass_name = f"{path}.bypass"
            points[bypass_name] = PointSpec(name=bypass_name, path=path, mode="identity")
            zero_name = f"{path}.zero"
            points[zero_name] = PointSpec(name=zero_name, path=path, mode="zero")

    custom_fn = getattr(model, "ablation_points", None)
    if callable(custom_fn):
        for name, factory in custom_fn().items():
            if name in points:
                warnings.warn(f"ablation point {name!r} from model.ablation_points() "
                               "overrides a generically-discovered point")
            points[name] = PointSpec(name=name, path="", mode="custom", factory=factory)

    return points


def apply_point(model: nn.Module, point: PointSpec) -> contextlib.AbstractContextManager[None]:
    """Return a context manager that activates ``point`` for its duration."""
    if point.mode == "custom":
        assert point.factory is not None
        return point.factory(model)
    return bypass_module(model, point.path, mode=point.mode)
