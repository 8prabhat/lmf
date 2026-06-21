"""MECM: MultiGear Elastic Causal Mesh baseline.

Causal long-convolution trunk plus a zero-gated mesh residual summary. Shares
its gear-hierarchy and training/generation scaffolding with mcpm and mgcf via
``NativeCausalLM`` (see ``lmf.models._shared.causal_mesh_base``).
"""

from __future__ import annotations

from ...core.registry import MODELS
from .._shared.causal_mesh_base import NativeCausalLM, NativeLMConfig


class MultiGearElasticCausalMeshLM(NativeCausalLM):
    """MECM baseline: causal long-convolution trunk plus zero-gated mesh summary."""

    family_name = "mecm"


@MODELS.register("mecm")
def build_mecm(model_cfg: dict, vocab_size: int | None = None) -> MultiGearElasticCausalMeshLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    cfg["mesh_residual"] = True
    cfg["execution_residual"] = False
    return MultiGearElasticCausalMeshLM(NativeLMConfig(**cfg))
