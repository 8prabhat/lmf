"""End-to-end: trainer loop, evaluation metrics, checkpoint round-trip, registries."""

from __future__ import annotations

import torch

from lmf.core.registry import MODELS, TRAINERS
from lmf.data import ProceduralCorpus, TrainingBatch
from lmf.evaluation import bits_per_token, calibrate_commit_threshold, tokens_per_settle
from lmf.evaluation.benchmarks import long_context_throughput
from lmf.evaluation.metrics import _forward_language_model
from lmf.models.rhca import RHCAConfig, RollingFrontierRHCA, RHCATrainer
from lmf.training.checkpoints import architecture_fingerprint, load_checkpoint, save_checkpoint


def _tiny_model():
    return RollingFrontierRHCA(RHCAConfig(
        vocab_size=64, field_dim=32, latent_dim=16, codebook="lowrank",
        codebook_factor_dim=8, frontier_size=8, max_commit=3, memory_slots=12,
        memory_read_top_k=4, memory_write_top_k=3,
        tail_size=16, local_kernel_size=3, max_hypotheses=1, ssm_macro_steps=2,
        ssm_scan_steps=4))


def test_registries_populated():
    assert "rhca" in MODELS and "transformer" in MODELS
    assert "rhca" in TRAINERS and "transformer" in TRAINERS


def test_trainer_runs_on_cpu():
    corpus = ProceduralCorpus(vocab_size=64)
    model = _tiny_model()
    trainer = RHCATrainer(model, corpus, device="cpu", precision="fp32",
                          warmup_steps=2, total_steps=10, lr=3e-3)
    records = trainer.train_steps(5, batch_size=4, seq_len=48, log_every=0)
    assert len(records) == 5
    assert all("total" in r for r in records)


def test_bits_per_token_dispatch_rhca():
    corpus = ProceduralCorpus(vocab_size=64)
    model = _tiny_model()
    bpt = bits_per_token(model, corpus, batch_size=2, seq_len=32, n_batches=1)
    assert bpt > 0 and bpt == bpt  # finite


def test_calibration_returns_threshold():
    corpus = ProceduralCorpus(vocab_size=64)
    model = _tiny_model()
    result = calibrate_commit_threshold(model, corpus, batch_size=2, seq_len=32, n_batches=2)
    assert "model_accuracy" in result


def test_structural_benchmarks_run():
    model = _tiny_model()
    tps = tokens_per_settle(model, prompt_len=6, gen_len=16, batch_size=2)
    assert tps["tokens_per_settle"] >= 1.0
    ctx = long_context_throughput(model, contexts=(32, 64), batch_size=1)
    assert len(ctx["results"]) == 2 and ctx["generation_state_size_constant"]


def test_checkpoint_roundtrip(tmp_path):
    corpus = ProceduralCorpus(vocab_size=64)
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, step=7)
    fresh = _tiny_model()
    assert architecture_fingerprint(fresh) == architecture_fingerprint(model)
    ckpt = load_checkpoint(path, fresh, strict=True)
    assert ckpt["step"] == 7


def test_generic_lm_evaluation_forwards_boundary_metadata():
    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))
            self.seen = None

        def forward(
            self,
            tokens,
            attention_mask=None,
            segment_ids=None,
            sentence_end_mask=None,
        ):
            self.seen = (
                attention_mask,
                segment_ids,
                sentence_end_mask,
            )
            return torch.zeros(*tokens.shape, 8), None

    tokens = torch.tensor([[1, 2, 3, 4]])
    attention = torch.ones_like(tokens, dtype=torch.bool)
    segments = torch.tensor([[0, 0, 1, 1]])
    boundaries = torch.tensor([[False, True, False, True]])
    batch = TrainingBatch(
        tokens,
        attention,
        attention,
        metadata={
            "segment_ids": segments,
            "sentence_end_mask": boundaries,
        },
    )
    model = Recorder()
    _forward_language_model(model, batch)
    assert model.seen[0] is attention
    assert model.seen[1] is segments
    assert model.seen[2] is boundaries
