"""OPET embedding/model tests."""

from __future__ import annotations

import torch

from lmf.data import ProceduralCorpus
from lmf.evaluation import bits_per_token
from lmf.models.opet import (
    PHASE_DIM,
    OPETEmbedding,
    OPETEmbeddingConfig,
    OPETLoss,
    OPETTrainer,
    OPETTransformerConfig,
    OPETTransformerLM,
    PhaseAnalyzer,
    compute_phase_entropy,
)


def _embedding_cfg(**overrides) -> OPETEmbeddingConfig:
    base = dict(vocab_size=64, d_model=32, context_window=2, n_freq_bands=4, dropout=0.0)
    base.update(overrides)
    return OPETEmbeddingConfig(**base)


def _model(**overrides) -> OPETTransformerLM:
    base = dict(vocab_size=64, dim=32, layers=2, heads=4, context_window=2, n_freq_bands=4, dropout=0.0)
    base.update(overrides)
    return OPETTransformerLM(OPETTransformerConfig(**base))


def test_embedding_forward_shapes():
    cfg = _embedding_cfg()
    emb = OPETEmbedding(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 9))
    out = emb(ids)

    assert out["embeddings"].shape == (3, 9, cfg.output_dim) == (3, 9, cfg.d_model + PHASE_DIM)
    assert out["phase"].shape == (3, 9)
    assert out["amplitude"].shape == (3, 9)
    assert out["oscillation"].shape == (3, 9, PHASE_DIM)


def test_opet_loss_gradients_flow():
    cfg = _embedding_cfg()
    emb = OPETEmbedding(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    out = emb(ids)

    loss_fn = OPETLoss()
    # A real downstream model consumes `embeddings` (which includes the gated
    # oscillation); fold that in here so phase_gate also gets a gradient.
    task_loss = out["embeddings"].sum() + 2.5
    losses = loss_fn(out, task_loss, omega_embedding=emb.phase_freq_emb.omega_raw)

    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    for name, param in emb.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"


def test_phase_analyzer_and_entropy():
    cfg = _embedding_cfg()
    emb = OPETEmbedding(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    analyzer = PhaseAnalyzer(emb)
    analysis = analyzer.analyze_sequence(ids, [f"tok{i}" for i in range(8)])

    assert analysis["phase"].shape == (8,)
    assert analysis["boundary_score"].shape == (7,)
    assert compute_phase_entropy(analysis["phase"]) >= 0.0

    sim = analyzer.phase_similarity_matrix(ids)
    assert sim.shape == (8, 8)


def test_model_forward_and_training_step():
    model = _model()
    tokens = torch.randint(0, 64, (3, 16))
    losses = model.training_step(tokens)

    assert "total" in losses and torch.isfinite(losses["total"])
    losses["total"].backward()


def test_model_generate():
    model = _model()
    prompt = torch.randint(0, 64, (2, 4))
    out = model.generate(prompt, 5)
    assert out.shape == (2, 5)


def test_trainer_and_bpt():
    corpus = ProceduralCorpus(vocab_size=64)
    model = _model()
    trainer = OPETTrainer(model, corpus, device="cpu", precision="fp32",
                          warmup_steps=2, total_steps=10, lr=3e-3)
    trainer.train_steps(3, batch_size=2, seq_len=24, log_every=0)
    bpt = bits_per_token(model, corpus, batch_size=2, seq_len=16, n_batches=1)
    assert bpt > 0
