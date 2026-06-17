"""OPET -- Oscillating Phase-Encoded Tokenization.

``OPETEmbedding`` is the reusable piece: a drop-in enrichment of any token
embedding with a learned, differentiable phase/oscillation signal (see
``embedding.py`` for the math). ``OPETTransformerLM`` shows how to plug it
into a standard transformer stack and is registered as the ``"opet"`` model
family.
"""

from __future__ import annotations

from .analysis import PhaseAnalyzer, compute_phase_entropy, format_analysis_table
from .embedding import (
    PHASE_DIM,
    ContextPhaseModulator,
    OPETEmbedding,
    OPETEmbeddingConfig,
    OscillationEncoder,
    PhaseFrequencyEmbedding,
)
from .losses import (
    AmplitudeEntropyLoss,
    BoundarySharpnessLoss,
    OPETLoss,
    PhaseCoherenceLoss,
    PhaseOrthogonalityLoss,
)
from .model import OPETTransformerConfig, OPETTransformerLM, build_opet
from .trainer import OPETTrainer, build_opet_trainer  # noqa: F401  (registers the trainer)

__all__ = [
    "PHASE_DIM",
    "OPETEmbeddingConfig",
    "OPETEmbedding",
    "PhaseFrequencyEmbedding",
    "ContextPhaseModulator",
    "OscillationEncoder",
    "OPETLoss",
    "PhaseCoherenceLoss",
    "BoundarySharpnessLoss",
    "PhaseOrthogonalityLoss",
    "AmplitudeEntropyLoss",
    "PhaseAnalyzer",
    "format_analysis_table",
    "compute_phase_entropy",
    "OPETTransformerConfig",
    "OPETTransformerLM",
    "build_opet",
    "OPETTrainer",
    "build_opet_trainer",
]
