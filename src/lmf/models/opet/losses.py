"""OPET auxiliary losses.

Three complementary losses that train the phase signal end-to-end, combined
with the downstream task loss by :class:`OPETLoss`:

1. ``PhaseCoherenceLoss``    -- semantically similar tokens should phase-align
2. ``BoundarySharpnessLoss`` -- phase transitions at boundaries should be crisp
3. ``PhaseOrthogonalityLoss`` -- different frequency bands shouldn't collapse
4. ``AmplitudeEntropyLoss``  -- keep the amplitude signal informative (not constant)

Mathematical intuition
----------------------
The phase signal phi lives on a circle. We want:
  * Similar tokens (high cosine sim in embedding space) -> |delta phi| small (co-phase)
  * Boundary tokens (low cosine sim) -> |delta phi| large (anti-phase)

This is inspired by oscillatory binding in neuroscience: objects bound
together oscillate in phase, separate objects anti-phase.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseCoherenceLoss(nn.Module):
    """
    Penalizes phase difference between adjacent tokens weighted by their
    embedding similarity.

    L_coh = -1/N sum_i cos(phi_i - phi_{i-1}) * sim(e_i, e_{i-1})

    - If e_i and e_{i-1} are similar -> high sim -> reward phase alignment
    - If e_i and e_{i-1} are dissimilar -> low sim -> phase can diverge freely
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        phi: torch.Tensor,         # (B, T)
        embeddings: torch.Tensor,  # (B, T, d_model)
        mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> torch.Tensor:
        delta_phi = phi[:, 1:] - phi[:, :-1]                   # (B, T-1)
        cos_delta = torch.cos(delta_phi / self.temperature)    # (B, T-1) in [-1, 1]

        e_curr = embeddings[:, 1:]
        e_prev = embeddings[:, :-1]
        sim = F.cosine_similarity(e_curr, e_prev, dim=-1)       # (B, T-1) in [-1, 1]
        sim_pos = (sim + 1.0) / 2.0                             # shift to [0, 1]

        coherence = cos_delta * sim_pos                         # (B, T-1)

        if mask is not None:
            valid = mask[:, 1:].float() * mask[:, :-1].float()
            coherence = coherence * valid
            return -coherence.sum() / (valid.sum() + 1e-8)

        return -coherence.mean()


class BoundarySharpnessLoss(nn.Module):
    """
    Encourages large phase transitions at semantic boundaries (positions of
    low embedding similarity).

    L_sharp = -1/N sum_i boundary_weight_i * |delta phi_i|

    where boundary_weight_i = (1 - sim(e_i, e_{i-1})) / 2  (high at boundaries)

    This creates an adversarial dynamic with ``PhaseCoherenceLoss``: coherence
    pushes phases together when similar, sharpness pushes them apart at
    boundaries.
    """

    def __init__(self, sharpness_scale: float = 1.0, max_delta: float = 3.14) -> None:
        super().__init__()
        self.sharpness_scale = sharpness_scale
        self.max_delta = max_delta  # cap to avoid degenerate solutions

    def forward(
        self,
        phi: torch.Tensor,         # (B, T)
        embeddings: torch.Tensor,  # (B, T, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        delta_phi = phi[:, 1:] - phi[:, :-1]
        abs_delta = torch.abs(delta_phi).clamp(max=self.max_delta)

        e_curr = embeddings[:, 1:]
        e_prev = embeddings[:, :-1]
        sim = F.cosine_similarity(e_curr, e_prev, dim=-1)

        boundary_weight = (1.0 - sim).clamp(min=0.0) / 2.0     # (B, T-1) in [0, 1]
        sharpness = boundary_weight * abs_delta * self.sharpness_scale

        if mask is not None:
            valid = mask[:, 1:].float() * mask[:, :-1].float()
            sharpness = sharpness * valid
            return -sharpness.sum() / (valid.sum() + 1e-8)

        return -sharpness.mean()


class PhaseOrthogonalityLoss(nn.Module):
    """
    Prevents the ``n_freq_bands`` from all learning the same frequency by
    softly repelling per-band mean frequencies across the vocabulary.

    L_orth = sum_{j!=k} exp(-|omega_j - omega_k|^2 / sigma^2)
    """

    def __init__(self, sigma: float = 0.5) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, omega_embedding: nn.Embedding) -> torch.Tensor:
        """
        Args:
            omega_embedding: the ``nn.Embedding`` holding raw omega values,
                shape (vocab_size, n_freq_bands)
        """
        mean_omega = omega_embedding.weight.mean(dim=0)  # (n_freq_bands,)

        n = mean_omega.shape[0]
        diff = mean_omega.unsqueeze(0) - mean_omega.unsqueeze(1)  # (n, n)
        dist_sq = diff ** 2

        repulsion = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        mask = 1.0 - torch.eye(n, device=mean_omega.device)
        return (repulsion * mask).sum() / (n * (n - 1))


class AmplitudeEntropyLoss(nn.Module):
    """
    Encourages amplitude to stay informative (not collapse to a constant) by
    maximizing the entropy of ``Bernoulli(A/2)``.

    Entropy is maximized at A=1.0, minimum at A=0 or A=2.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, amplitude: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            amplitude: (B, T) in (0, 2)
        """
        p = (amplitude / 2.0).clamp(self.epsilon, 1.0 - self.epsilon)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))  # (B, T)

        if mask is not None:
            entropy = entropy * mask.float()
            return -entropy.sum() / (mask.float().sum() + 1e-8)

        return -entropy.mean()  # negative because we maximize entropy


class OPETLoss(nn.Module):
    """
    Combined OPET auxiliary loss.

    L_total = L_task + lambda_coh * L_coh + lambda_sharp * L_sharp
                      + lambda_orth * L_orth + lambda_amp * L_amp

    Usage::

        opet_loss = OPETLoss()
        losses = opet_loss(opet_out, task_loss, omega_embedding=embedding.phase_freq_emb.omega_raw)
        losses["total"].backward()
    """

    def __init__(
        self,
        lambda_coherence: float = 0.10,
        lambda_sharpness: float = 0.05,
        lambda_orthogonality: float = 0.01,
        lambda_amplitude: float = 0.02,
        phase_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_coherence = lambda_coherence
        self.lambda_sharpness = lambda_sharpness
        self.lambda_orthogonality = lambda_orthogonality
        self.lambda_amplitude = lambda_amplitude

        self.coherence_loss = PhaseCoherenceLoss(temperature=phase_temperature)
        self.sharpness_loss = BoundarySharpnessLoss()
        self.orthogonality_loss = PhaseOrthogonalityLoss()
        self.amplitude_loss = AmplitudeEntropyLoss()

    def forward(
        self,
        opet_out: dict,                                  # output dict from OPETEmbedding.forward()
        task_loss: torch.Tensor,                         # scalar loss from downstream task
        omega_embedding: Optional[nn.Embedding] = None,  # for the orthogonality loss
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        phi = opet_out['phase']        # (B, T)
        emb = opet_out['token_emb']    # (B, T, d_model)
        amp = opet_out['amplitude']    # (B, T)

        l_coh = self.coherence_loss(phi, emb, mask)
        l_sharp = self.sharpness_loss(phi, emb, mask)
        l_amp = self.amplitude_loss(amp, mask)

        l_orth = torch.tensor(0.0, device=phi.device)
        if omega_embedding is not None:
            l_orth = self.orthogonality_loss(omega_embedding)

        total = (
            task_loss
            + self.lambda_coherence * l_coh
            + self.lambda_sharpness * l_sharp
            + self.lambda_orthogonality * l_orth
            + self.lambda_amplitude * l_amp
        )

        return {
            'total': total,
            'task': task_loss,
            'coherence': l_coh,
            'sharpness': l_sharp,
            'orthogonality': l_orth,
            'amplitude': l_amp,
        }
