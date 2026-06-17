"""OPET analysis & visualization helpers.

Post-hoc tools to understand what the phase signal has learned:
  - ``PhaseAnalyzer``: extract boundary scores, phase velocity, coherence maps
  - ``format_analysis_table``: pretty-print a per-token phase analysis
  - ``compute_phase_entropy``: diversity of the learned phase distribution

These operate on any module exposing the ``OPETEmbedding`` forward-output
contract (a dict with ``'phase'``, ``'amplitude'``, ``'oscillation'``, etc.).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import torch


class PhaseAnalyzer:
    """Post-hoc analysis of OPET phase outputs."""

    def __init__(self, embedding: torch.nn.Module) -> None:
        self.embedding = embedding

    @torch.no_grad()
    def analyze_sequence(
        self,
        token_ids: torch.Tensor,    # (1, T)
        token_strings: List[str],   # list of T decoded tokens
    ) -> dict:
        """Full analysis of a single sequence."""
        self.embedding.eval()
        out = self.embedding(token_ids)

        phi = out['phase'][0]      # (T,)
        amp = out['amplitude'][0]  # (T,)
        osc = out['oscillation'][0]  # (T, PHASE_DIM)
        omega = out['omega'][0]    # (T,)
        psi = out['psi'][0]        # (T,)

        phase_vel = torch.diff(phi)     # (T-1,)
        abs_vel = phase_vel.abs()

        boundary_score = (abs_vel - abs_vel.min()) / (abs_vel.max() - abs_vel.min() + 1e-8)

        cos_vel = torch.cos(phase_vel)  # (T-1,)
        coherence = (cos_vel + 1) / 2   # map to [0, 1]

        energy = (osc ** 2).sum(dim=-1)  # (T,)

        return {
            'tokens': token_strings,
            'phase': phi.cpu().numpy(),
            'amplitude': amp.cpu().numpy(),
            'omega': omega.cpu().numpy(),
            'psi': psi.cpu().numpy(),
            'phase_velocity': phase_vel.cpu().numpy(),   # length T-1
            'boundary_score': boundary_score.cpu().numpy(),  # length T-1
            'coherence': coherence.cpu().numpy(),        # length T-1
            'osc_energy': energy.cpu().numpy(),          # length T
            'cos_phi': osc[:, 0].cpu().numpy(),
            'sin_phi': osc[:, 1].cpu().numpy(),
        }

    @torch.no_grad()
    def phase_similarity_matrix(self, token_ids: torch.Tensor) -> np.ndarray:
        """
        Pairwise phase similarity matrix for a sequence:
        sim(i, j) = cos(phi_i - phi_j) in [-1, 1]
        """
        self.embedding.eval()
        out = self.embedding(token_ids)
        phi = out['phase'][0]  # (T,)
        T = phi.shape[0]

        phi_i = phi.unsqueeze(1).expand(T, T)
        phi_j = phi.unsqueeze(0).expand(T, T)
        sim = torch.cos(phi_i - phi_j)
        return sim.cpu().numpy()

    @torch.no_grad()
    def compute_boundary_precision_recall(
        self,
        token_ids: torch.Tensor,
        true_boundaries: List[int],  # indices where boundaries exist (0-indexed, in [0, T-2])
        threshold: float = 0.5,
    ) -> dict:
        """Evaluate how well the phase boundary score predicts true boundaries."""
        self.embedding.eval()
        out = self.embedding(token_ids)
        phi = out['phase'][0]

        phase_vel = torch.diff(phi).abs()
        score = (phase_vel - phase_vel.min()) / (phase_vel.max() - phase_vel.min() + 1e-8)
        pred_boundaries = set((score > threshold).nonzero(as_tuple=True)[0].cpu().tolist())
        true_set = set(true_boundaries)

        tp = len(pred_boundaries & true_set)
        fp = len(pred_boundaries - true_set)
        fn = len(true_set - pred_boundaries)

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn,
            'threshold': threshold,
            'n_predicted': len(pred_boundaries),
            'n_true': len(true_set),
        }

    def find_optimal_threshold(
        self,
        token_ids: torch.Tensor,
        true_boundaries: List[int],
        n_thresholds: int = 20,
    ) -> Tuple[float, dict]:
        """Sweep thresholds to find the one maximizing F1."""
        best_f1 = -1.0
        best_t = 0.5
        best_metrics: dict = {}

        for t in np.linspace(0.1, 0.9, n_thresholds):
            m = self.compute_boundary_precision_recall(token_ids, true_boundaries, float(t))
            if m['f1'] > best_f1:
                best_f1 = m['f1']
                best_t = float(t)
                best_metrics = m

        return best_t, best_metrics


def format_analysis_table(analysis: dict) -> str:
    """Pretty-print a phase analysis as a text table."""
    tokens = analysis['tokens']
    phase = analysis['phase']
    amp = analysis['amplitude']
    bscore = analysis['boundary_score']
    T = len(tokens)

    lines = [f"{'Token':<20} {'phi (phase)':>11} {'A (amp)':>10} {'Boundary':>10}", "-" * 56]
    for i in range(T):
        b = f"{bscore[i-1]:.3f}" if i > 0 else "  --  "
        lines.append(f"{tokens[i]:<20} {phase[i]:>11.4f} {amp[i]:>10.4f} {b:>10}")
    return "\n".join(lines)


def compute_phase_entropy(phi: np.ndarray) -> float:
    """
    Entropy of the wrapped phase distribution.
    High entropy = diverse phases (good diversity).
    Low entropy  = phases collapsed (degenerate).
    """
    phi_wrapped = phi % (2 * math.pi) - math.pi
    counts, _ = np.histogram(phi_wrapped, bins=32, range=(-math.pi, math.pi))
    probs = counts / (counts.sum() + 1e-8)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))
