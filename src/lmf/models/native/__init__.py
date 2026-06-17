"""MultiGear-native architecture baselines: MECM, MCPM, MGCF, and MRWT."""

from __future__ import annotations

from .components import MGCFConfig, MRWTConfig, NativeLMConfig
from .model import (
    MultiGearConstructiveProgramMachineLM,
    MultiGearElasticCausalMeshLM,
    MultiGearFractalCausalFieldLM,
    MultiGearResidualWorkbenchTransformerLM,
)
from .model import build_mcpm, build_mecm, build_mgcf, build_mrwt
from .trainer import (
    NativeLMTrainer,
    build_mcpm_trainer,
    build_mecm_trainer,
    build_mgcf_trainer,
    build_mrwt_trainer,
)

__all__ = [
    "NativeLMConfig",
    "MGCFConfig",
    "MRWTConfig",
    "MultiGearElasticCausalMeshLM",
    "MultiGearConstructiveProgramMachineLM",
    "MultiGearFractalCausalFieldLM",
    "MultiGearResidualWorkbenchTransformerLM",
    "NativeLMTrainer",
    "build_mecm",
    "build_mcpm",
    "build_mgcf",
    "build_mrwt",
    "build_mecm_trainer",
    "build_mcpm_trainer",
    "build_mgcf_trainer",
    "build_mrwt_trainer",
]
