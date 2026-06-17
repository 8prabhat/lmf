"""OPET: Oscillating Phase-Encoded Tokenization.

A fully differentiable embedding enrichment layer where each token carries a
sinusoidal phase state that evolves across the sequence. Phase encodes
semantic-boundary information as a continuous, learnable signal on top of a
standard token embedding.

Pipeline::

    token ids -> token embedding -> phase frequency params (omega, phi0, A)
                                  -> context phase modulation (psi)
                                  -> phase phi = omega*pos + psi + phi0
                                  -> oscillation z = [cos phi, sin phi,
                                                       A cos 2phi, A sin 2phi]
                                  -> enriched embedding = concat(token_emb, gate(z) * z)

``OPETEmbedding`` is a drop-in replacement for ``nn.Embedding`` that produces
``d_model + PHASE_DIM`` wide vectors; ``OPETEmbeddingConfig.output_dim`` gives
the resulting width so callers can size a projection into their model's
hidden dimension.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of dimensions in the oscillation encoding: [cos phi, sin phi, A cos 2phi, A sin 2phi]
PHASE_DIM = 4


@dataclass(frozen=True)
class OPETEmbeddingConfig:
    vocab_size: int = 32000
    d_model: int = 256               # token embedding dim
    context_window: int = 4          # preceding tokens used for phase modulation
    n_freq_bands: int = 8            # number of learned frequency bands per token
    phase_init_scale: float = 0.1    # how spread-out initial phase perturbations are
    dropout: float = 0.1

    @property
    def output_dim(self) -> int:
        """Width of the enriched embedding: ``d_model + PHASE_DIM``."""
        return self.d_model + PHASE_DIM

    def to_dict(self) -> dict:
        return asdict(self)


class PhaseFrequencyEmbedding(nn.Module):
    """
    Learns a base frequency omega_v and base phase offset phi0_v for each
    vocabulary token. These are the 'intrinsic oscillation' parameters
    of each token type independent of position.

    omega_v in (0, pi)  -- constrained via sigmoid so frequency is bounded
    phi0_v  in [0, 2pi) -- unconstrained, wraps naturally via sin/cos
    """

    def __init__(self, cfg: OPETEmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Raw learnable parameters; shaped per-vocab-entry
        self.omega_raw = nn.Embedding(cfg.vocab_size, cfg.n_freq_bands)
        self.phi0_raw = nn.Embedding(cfg.vocab_size, cfg.n_freq_bands)
        self.amp_raw = nn.Embedding(cfg.vocab_size, cfg.n_freq_bands)

        # Project n_freq_bands -> scalar omega, phi0, A per token
        self.freq_proj = nn.Linear(cfg.n_freq_bands, 1, bias=False)
        self.phi_proj = nn.Linear(cfg.n_freq_bands, 1, bias=False)
        self.amp_proj = nn.Linear(cfg.n_freq_bands, 1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.omega_raw.weight, -0.5, 0.5)
        nn.init.uniform_(self.phi0_raw.weight, -math.pi, math.pi)
        nn.init.constant_(self.amp_raw.weight, 0.0)  # sigmoid(0) = 0.5 -> A=1.0 after scale

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: (B, T) int64

        Returns:
            omega: (B, T) in (0, pi)  -- per-token frequency
            phi0:  (B, T)             -- per-token base phase
            amp:   (B, T) in (0, 2)   -- per-token amplitude
        """
        o_raw = self.omega_raw(token_ids)  # (B, T, n_freq_bands)
        p_raw = self.phi0_raw(token_ids)
        a_raw = self.amp_raw(token_ids)

        omega = torch.sigmoid(self.freq_proj(o_raw).squeeze(-1)) * math.pi  # (B, T)
        phi0 = self.phi_proj(p_raw).squeeze(-1)                              # (B, T) unbounded
        amp = 2.0 * torch.sigmoid(self.amp_proj(a_raw).squeeze(-1))         # (B, T) in (0, 2)

        return omega, phi0, amp


class ContextPhaseModulator(nn.Module):
    """
    Computes psi_i: a phase perturbation from the preceding-token context.
    Uses a lightweight depthwise causal 1D conv over embeddings to produce a
    per-position phase delta.

    Preceding tokens 'pull' or 'push' a token's phase, creating emergent phase
    waves across the sequence. The conv is causal (left-padded only) so this
    matches the information available during autoregressive generation.
    """

    def __init__(self, cfg: OPETEmbeddingConfig) -> None:
        super().__init__()
        self.context_window = cfg.context_window

        self.dw_conv = nn.Conv1d(
            cfg.d_model, cfg.d_model,
            kernel_size=cfg.context_window + 1,
            padding=0,
            groups=cfg.d_model,  # depthwise -- cheap
            bias=False,
        )
        self.proj = nn.Linear(cfg.d_model, 1, bias=True)
        self.scale = nn.Parameter(torch.tensor(cfg.phase_init_scale))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, T, d_model)
        Returns:
            psi: (B, T) -- phase perturbation
        """
        x = embeddings.transpose(1, 2)  # (B, d_model, T)
        x = F.pad(x, (self.context_window, 0))  # causal: left-pad only
        x = self.dw_conv(x)             # (B, d_model, T)
        x = F.gelu(x)
        x = x.transpose(1, 2)           # (B, T, d_model)
        x = self.drop(x)
        psi = self.proj(x).squeeze(-1)  # (B, T)
        return psi * self.scale


class OscillationEncoder(nn.Module):
    """
    Assembles the full phase phi_i and encodes it as a ``PHASE_DIM``-D
    oscillation vector:

        phi_i = omega_i * i + psi_i + phi0_i

        z_i = [cos(phi_i), sin(phi_i), A_i * cos(2 phi_i), A_i * sin(2 phi_i)]

    The second harmonic (2*phi) captures finer boundary structure. The
    amplitude A modulates confidence / boundary sharpness.
    """

    def forward(self, omega: torch.Tensor, phi0: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            omega, phi0, psi: each (B, T)

        Returns:
            phi: (B, T) -- raw phase (for loss computation)
        """
        B, T = omega.shape
        pos = torch.arange(T, device=omega.device, dtype=omega.dtype).unsqueeze(0)  # (1, T)
        return omega * pos + psi + phi0  # (B, T)

    def encode_phase(self, phi: torch.Tensor, amp: torch.Tensor) -> torch.Tensor:
        """
        Convert raw phase + amplitude into a ``PHASE_DIM``-D oscillation vector.

        Args:
            phi: (B, T)
            amp: (B, T)
        Returns:
            z: (B, T, PHASE_DIM)
        """
        c1 = torch.cos(phi)
        s1 = torch.sin(phi)
        c2 = amp * torch.cos(2 * phi)
        s2 = amp * torch.sin(2 * phi)
        return torch.stack([c1, s1, c2, s2], dim=-1)  # (B, T, PHASE_DIM)


class OPETEmbedding(nn.Module):
    """
    Full OPET embedding module.

    Input:  token_ids (B, T)
    Output: a dict with the enriched embedding ``(B, T, output_dim)`` plus the
            raw phase/amplitude signals needed by ``opet.losses.OPETLoss`` and
            ``opet.analysis.PhaseAnalyzer``.

    Any model can use this as its input embedding by projecting
    ``output_dim -> model_dim`` (see ``opet.model.OPETTransformerLM`` for an
    example).
    """

    def __init__(self, cfg: OPETEmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.phase_freq_emb = PhaseFrequencyEmbedding(cfg)
        self.context_modulator = ContextPhaseModulator(cfg)
        self.oscillation_encoder = OscillationEncoder()

        # Learned gate: how much oscillation signal to pass through.
        self.phase_gate = nn.Sequential(
            nn.Linear(PHASE_DIM, PHASE_DIM),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Returns a dict with:
          'embeddings'  : (B, T, output_dim)  -- concatenated output
          'phase'       : (B, T)              -- raw phase phi
          'amplitude'   : (B, T)              -- amplitude A
          'token_emb'   : (B, T, d_model)     -- base token embedding
          'oscillation' : (B, T, PHASE_DIM)   -- gated phase encoding z
          'omega'       : (B, T)
          'psi'         : (B, T)
        """
        del attention_mask  # not needed by the embedding itself; kept for interface symmetry

        tok_emb = self.token_embedding(token_ids)  # (B, T, d_model)
        tok_emb = self.drop(tok_emb)

        omega, phi0, amp = self.phase_freq_emb(token_ids)
        psi = self.context_modulator(tok_emb)
        phi = self.oscillation_encoder(omega, phi0, psi)
        osc = self.oscillation_encoder.encode_phase(phi, amp)  # (B, T, PHASE_DIM)

        # Gate learns on a detached copy of the signal so the gate's own
        # gradient doesn't fight the phase gradient flowing through `osc`.
        gate = self.phase_gate(osc.detach())
        gated_osc = osc * gate

        enriched = torch.cat([tok_emb, gated_osc], dim=-1)  # (B, T, output_dim)

        return {
            'embeddings': enriched,
            'phase': phi,
            'amplitude': amp,
            'token_emb': tok_emb,
            'oscillation': gated_osc,
            'omega': omega,
            'psi': psi,
        }

    def get_phase_stats(self, out: dict) -> dict:
        """Compute diagnostic statistics on phase output."""
        phi = out['phase']
        amp = out['amplitude']

        phase_vel = torch.diff(phi, dim=-1)        # (B, T-1)
        boundary_score = torch.abs(phase_vel)

        return {
            'mean_phase_velocity': phase_vel.mean().item(),
            'std_phase_velocity': phase_vel.std().item(),
            'mean_amplitude': amp.mean().item(),
            'boundary_score': boundary_score,      # (B, T-1)
        }
