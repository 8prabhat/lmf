"""MCPM: MultiGear Constructive Program Machine baseline.

Surface model plus a zero-gated deterministic execution-trace adapter. Shares
its gear-hierarchy and training/generation scaffolding with mecm and mgcf via
``NativeCausalLM`` (see ``lmf.models._shared.causal_mesh_base``).
"""

from __future__ import annotations

from ...core.registry import MODELS
from .._shared.causal_mesh_base import NativeCausalLM, NativeLMConfig


class MultiGearConstructiveProgramMachineLM(NativeCausalLM):
    """MCPM baseline: surface model plus zero-gated deterministic trace adapter."""

    family_name = "mcpm"


@MODELS.register("mcpm")
def build_mcpm(
    model_cfg: dict, vocab_size: int | None = None
) -> MultiGearConstructiveProgramMachineLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    cfg["mesh_residual"] = True
    cfg["execution_residual"] = True
    return MultiGearConstructiveProgramMachineLM(NativeLMConfig(**cfg))
