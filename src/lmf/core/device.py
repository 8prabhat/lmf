"""Device and precision policy — owned in one place (DRY).

This module encodes the hard-won MPS lessons from the v3 review:

* **bf16 by default** (review S1). On Apple silicon a uniform whole-model bf16
  cast roughly doubles the MPS-safe batch and gives ~2x matmul throughput. We do
  NOT use autocast on MPS, because mixing fp32/bf16 ops in one graph crashes the
  MPS backend; instead we cast the whole module to bf16 and selectively retain
  any model-declared overflow-prone modules in fp32.
* **Watermark-based eviction** (review S2) instead of a full
  ``synchronize()+empty_cache()`` every optimizer step. With uniform step shapes
  (carried-state training) the per-step allocator flush is no longer needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def resolve_device(name: str | None = None) -> torch.device:
    """Resolve a requested device name to an available torch.device."""
    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was explicitly requested but is not available; "
                "silent CPU fallback is disabled"
            )
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was explicitly requested but is not available in this "
                "Python/PyTorch environment; silent CPU fallback is disabled"
            )
        return torch.device("mps")
    raise ValueError(f"unknown device: {name!r}")


def sync(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class PrecisionPolicy:
    """Encapsulates how a model and training loop handle low precision.

    ``precision`` is one of {"fp32", "bf16"}. The policy knows the two distinct
    bf16 strategies: uniform whole-model cast (MPS) vs autocast (CUDA/CPU).
    """

    precision: str = "bf16"

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be 'fp32' or 'bf16'")

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.precision == "bf16" else torch.float32

    def cast_model(self, model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
        """Apply the device-appropriate precision strategy to a model.

        Under a bf16 model, any module the model declares via ``fp32_modules()`` is
        recast back to fp32 (review §8: keep overflow-prone heads in fp32). The
        model owns this knowledge; the policy merely honours it.
        """
        model = model.to(device)
        if getattr(model, "force_fp32_parameters", False):
            # Pure Parallel Gear keeps persistent dynamics, parameters, and
            # optimizer moments in FP32. This is deliberately conservative on
            # MPS, where mixed dtypes inside one graph remain fragile.
            return model.float()
        if self.precision == "bf16" and device.type == "mps":
            # Uniform cast: avoids the MPS graph type-mixing crash autocast triggers.
            model = model.to(torch.bfloat16)
            for module in (model.fp32_modules() if hasattr(model, "fp32_modules") else []):
                module.float()
        return model

    def autocast(self, device: torch.device):
        """Context manager for the forward/backward pass.

        Only CUDA/CPU use autocast; on MPS the model is already uniformly bf16,
        so this is a no-op there.
        """
        if self.precision == "bf16" and device.type in {"cuda", "cpu"}:
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        return _NullContext()


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class MemoryGovernor:
    """Watermark-based MPS allocator eviction (review S2).

    Replaces a per-step ``synchronize()+empty_cache()`` with a periodic / watermark
    trigger. On CUDA and CPU this is a cheap no-op.
    """

    def __init__(self, device: torch.device, every: int = 50, watermark: float = 0.8) -> None:
        self.device = device
        self.every = max(1, int(every))
        self.watermark = float(watermark)
        self._budget: int | None = None

    def _mps_budget(self) -> int | None:
        try:
            return int(torch.mps.recommended_max_memory())
        except Exception:
            return None

    def maybe_evict(self, step: int) -> None:
        if self.device.type != "mps":
            return
        over_watermark = False
        if self._budget is None:
            self._budget = self._mps_budget()
        if self._budget:
            try:
                over_watermark = torch.mps.driver_allocated_memory() > self.watermark * self._budget
            except Exception:
                over_watermark = False
        if step % self.every == 0 or over_watermark:
            torch.mps.synchronize()
            torch.mps.empty_cache()
