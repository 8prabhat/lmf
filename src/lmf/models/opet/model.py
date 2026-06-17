"""OPET transformer LM -- an example consumer of the reusable OPET embedding.

Wires :class:`~lmf.models.opet.embedding.OPETEmbedding` (phase-enriched token
embeddings) into the same RMSNorm + RoPE + SwiGLU + SDPA transformer blocks
used by the baseline (``lmf.models.transformer``), so the only difference
from the baseline is the embedding layer and the extra phase-coherence
auxiliary losses computed in ``training_step``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from ..transformer.model import Block, RMSNorm
from .embedding import OPETEmbedding, OPETEmbeddingConfig
from .losses import OPETLoss


@dataclass(frozen=True)
class OPETTransformerConfig:
    vocab_size: int
    dim: int = 512
    layers: int = 8
    heads: int = 8
    max_seq_len: int = 4096

    # OPET embedding knobs
    context_window: int = 4
    n_freq_bands: int = 8
    phase_init_scale: float = 0.1
    dropout: float = 0.1

    # OPET auxiliary loss weights
    lambda_coherence: float = 0.10
    lambda_sharpness: float = 0.05
    lambda_orthogonality: float = 0.01
    lambda_amplitude: float = 0.02
    phase_temperature: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def embedding_config(self) -> OPETEmbeddingConfig:
        return OPETEmbeddingConfig(
            vocab_size=self.vocab_size,
            d_model=self.dim,
            context_window=self.context_window,
            n_freq_bands=self.n_freq_bands,
            phase_init_scale=self.phase_init_scale,
            dropout=self.dropout,
        )


class OPETTransformerLM(nn.Module):
    """Transformer LM whose input embedding is OPET phase-enriched.

    ``forward`` returns ``(logits, opet_out)`` -- the same two-element shape
    the baseline transformer returns (``logits, caches``) so generic
    evaluation helpers like ``transformer_bits_per_token`` work unmodified.
    KV-caching is not implemented: the context phase modulator looks at the
    whole sequence, so ``generate`` recomputes the full forward each step.
    """

    def __init__(self, config: OPETTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = OPETEmbedding(config.embedding_config())
        self.input_proj = nn.Linear(self.embedding.cfg.output_dim, config.dim, bias=False)

        self.blocks = nn.ModuleList([Block(config.dim, config.heads) for _ in range(config.layers)])
        self.norm = RMSNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.opet_loss = OPETLoss(
            lambda_coherence=config.lambda_coherence,
            lambda_sharpness=config.lambda_sharpness,
            lambda_orthogonality=config.lambda_orthogonality,
            lambda_amplitude=config.lambda_amplitude,
            phase_temperature=config.phase_temperature,
        )

    @staticmethod
    def _full_attn_mask(attention_mask, n, device):
        if attention_mask is None or bool(attention_mask.all()):
            return None
        causal = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
        key_valid = attention_mask.bool()[:, None, None, :]
        return causal[None, None] & key_valid

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        opet_out = self.embedding(ids)
        x = self.input_proj(opet_out['embeddings'])

        attn_mask = self._full_attn_mask(attention_mask, ids.shape[1], ids.device)
        for block in self.blocks:
            x, _ = block(x, None, False, attn_mask)

        logits = self.head(self.norm(x))
        return logits, opet_out

    def training_step(self, tokens: torch.Tensor, task_metadata: dict[str, Any] | None = None,
                      loss_term_scales: dict[str, float] | None = None) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        attention_mask = meta.get("attention_mask")
        logits, opet_out = self(tokens, attention_mask=attention_mask)

        per_tok = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1),
            reduction="none").reshape(tokens.shape[0], -1)
        loss_mask = meta.get("loss_mask")
        # Intersect with attention_mask, matching evaluation (padding must not
        # contribute to the loss even when loss_mask alone would allow it).
        if loss_mask is None and attention_mask is None:
            task_loss = per_tok.mean()
        else:
            valid = torch.ones_like(per_tok, dtype=torch.bool)
            if loss_mask is not None:
                valid = valid & loss_mask[:, 1:].bool()
            if attention_mask is not None:
                valid = valid & attention_mask[:, 1:].bool()
            valid = valid.to(per_tok.dtype)
            task_loss = (per_tok * valid).sum() / valid.sum().clamp_min(1)

        losses = self.opet_loss(
            opet_out, task_loss,
            omega_embedding=self.embedding.phase_freq_emb.omega_raw,
            mask=meta.get("attention_mask"),
        )
        scales = loss_term_scales or {}
        total = (
            scales.get("language_modeling", 1.0) * losses["task"]
            + self.config.lambda_coherence * scales.get("phase_coherence", 1.0) * losses["coherence"]
            + self.config.lambda_sharpness * scales.get("phase_sharpness", 1.0) * losses["sharpness"]
            + self.config.lambda_orthogonality * scales.get("phase_orthogonality", 1.0) * losses["orthogonality"]
            + self.config.lambda_amplitude * scales.get("phase_amplitude", 1.0) * losses["amplitude"]
        )
        return {
            "total": total,
            "language_modeling": losses["task"],
            "phase_coherence": losses["coherence"],
            "phase_sharpness": losses["sharpness"],
            "phase_orthogonality": losses["orthogonality"],
            "phase_amplitude": losses["amplitude"],
        }

    def _sample_token(self, logits: torch.Tensor, cfg) -> torch.Tensor:
        if cfg is None or cfg.deterministic:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / max(cfg.temperature, 1e-5)
        if cfg.top_k > 0:
            thresh = logits.topk(min(cfg.top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        if cfg.top_p < 1.0:
            sl, idx = logits.sort(dim=-1, descending=True)
            remove = sl.softmax(-1).cumsum(-1) > cfg.top_p
            remove[..., 0] = False
            sl = sl.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sl)
        return torch.multinomial(logits.softmax(-1), 1)

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, sampling_config=None) -> torch.Tensor:
        """Greedy/sampled decoding without KV cache (full re-forward each step)."""
        tokens = prompt_tokens
        out = []
        for _ in range(max_new_tokens):
            logits, _ = self(tokens)
            next_token = self._sample_token(logits[:, -1], sampling_config)
            out.append(next_token)
            tokens = torch.cat([tokens, next_token], dim=1)
        return torch.cat(out, dim=1)

    def architecture_manifest(self) -> dict:
        return {
            "name": "OPETTransformerLM",
            "config": self.config.to_dict(),
            "parameters": {"total": sum(p.numel() for p in self.parameters())},
        }


@MODELS.register("opet")
def build_opet(model_cfg: dict, vocab_size: int | None = None) -> OPETTransformerLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return OPETTransformerLM(OPETTransformerConfig(**cfg))
