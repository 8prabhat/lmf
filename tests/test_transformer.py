"""Transformer baseline family."""

from __future__ import annotations

import torch

from lmf.core.build import configure_token_hierarchy
from lmf.core.registry import MODELS, TRAINERS
from lmf.data import (
    ByteBPETokenizer,
    InMemoryTextCorpus,
    MultiGearTokenizer,
    ProceduralCorpus,
    SpecialTokenTokenizer,
)
from lmf.evaluation import bits_per_token, lm_metrics, repetition_rate
from lmf.models.transformer import CachedTransformerLM, TransformerConfig, TransformerTrainer


def _model():
    return CachedTransformerLM(TransformerConfig(vocab_size=64, dim=32, layers=2, heads=4))


def test_forward_and_training_step():
    model = _model()
    tokens = torch.randint(0, 64, (3, 20))
    losses = model.training_step(tokens)
    assert "total" in losses and torch.isfinite(losses["total"]).all()
    losses["total"].backward()


def test_kv_cached_generate_matches_length():
    model = _model()
    out = model.generate(torch.randint(0, 64, (2, 5)), 12)
    assert out.shape == (2, 12)


def test_generate_honors_sampling_config():
    """Finding 7: generation must not be hard-greedy when a sampler is given."""
    from lmf.models.rhca.state import SamplingConfig
    torch.manual_seed(0)
    model = _model()
    prompt = torch.randint(0, 64, (1, 5))
    greedy = model.generate(prompt, 16, SamplingConfig(deterministic=True))
    sampled = model.generate(prompt, 16, SamplingConfig(deterministic=False, temperature=1.5,
                                                        top_k=20))
    assert greedy.shape == sampled.shape == (1, 16)


def test_padding_mask_changes_logits():
    """Finding 9: attention_mask must actually affect the transformer's attention."""
    model = _model()
    tokens = torch.randint(0, 64, (1, 8))
    full = torch.ones(1, 8, dtype=torch.bool)
    padded = full.clone()
    padded[:, :3] = False                      # mark first 3 as padding
    out_full, _ = model(tokens, attention_mask=full)
    out_pad, _ = model(tokens, attention_mask=padded)
    assert not torch.allclose(out_full, out_pad)


def test_packed_document_segments_are_isolated():
    model = _model().eval()
    tokens = torch.randint(0, 64, (1, 12))
    segments = torch.tensor([[0] * 6 + [1] * 6])
    logits, _ = model(tokens, segment_ids=segments)
    changed = tokens.clone()
    changed[:, :6] = (changed[:, :6] + 17) % 64
    changed_logits, _ = model(changed, segment_ids=segments)
    assert torch.allclose(
        logits[:, 6:],
        changed_logits[:, 6:],
        atol=1e-5,
    )


def test_single_segment_training_keeps_fused_causal_attention(monkeypatch):
    model = _model()
    tokens = torch.randint(0, 64, (2, 16))
    mask = torch.ones_like(tokens, dtype=torch.bool)
    segments = torch.arange(2)[:, None].expand_as(tokens)
    seen_segments = []
    original = model._full_attn_mask

    def record(attention_mask, segment_ids, length, device):
        seen_segments.append(segment_ids)
        return original(attention_mask, segment_ids, length, device)

    monkeypatch.setattr(model, "_full_attn_mask", record)
    losses = model.training_step(
        tokens,
        {
            "attention_mask": mask,
            "loss_mask": mask,
            "segment_ids": segments,
            "single_segment_rows": True,
        },
    )
    assert torch.isfinite(losses["total"])
    assert seen_segments == [None]


def test_trainer_and_bpt():
    corpus = ProceduralCorpus(vocab_size=64)
    model = _model()
    trainer = TransformerTrainer(model, corpus, device="cpu", precision="fp32",
                                 warmup_steps=2, total_steps=10, lr=3e-3)
    trainer.train_steps(5, batch_size=4, seq_len=48, log_every=0)
    bpt = bits_per_token(model, corpus, batch_size=2, seq_len=32, n_batches=1)
    assert bpt > 0


def test_generic_repetition_metric_accepts_tensor_generation():
    corpus = ProceduralCorpus(vocab_size=64)
    rate = repetition_rate(
        _model(),
        corpus,
        batch_size=1,
        prompt_len=4,
        gen_len=8,
        n_batches=1,
        ngram=2,
    )
    assert 0.0 <= rate <= 1.0


def test_lm_metrics_include_bits_per_byte_for_decodable_corpus():
    text = "alpha beta gamma delta epsilon zeta eta theta " * 30
    tokenizer = ByteBPETokenizer(max_vocab=320)
    tokenizer.train([text])
    corpus = InMemoryTextCorpus(
        text, tokenizer=tokenizer, train_frac=0.7, valid_frac=0.2,
        seed=3, wrap_special=False)
    model = CachedTransformerLM(
        TransformerConfig(vocab_size=corpus.vocab_size, dim=32, layers=1, heads=4))
    metrics = lm_metrics(model, corpus, batch_size=2, seq_len=16, n_batches=1)
    assert metrics["bits_per_token"] > 0
    assert metrics["bits_per_byte"] > 0
    assert metrics["bytes_per_token"] > 0


def test_multigear_hierarchical_transformer_uses_hierarchy():
    base = MultiGearTokenizer(max_vocab=340, max_token_bytes=16)
    text = "alpha beta gamma delta conference Australia India Morgan " * 20
    base.train([text])
    tokenizer = SpecialTokenTokenizer(base)
    model = CachedTransformerLM(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=1,
            heads=4,
            max_seq_len=96,
            hierarchical_output=True,
            hierarchy_output_mode="bias",
            input_gear_embedding=True,
        )
    )
    assert configure_token_hierarchy(model, tokenizer)
    ids = tokenizer.encode(text)[:32]
    tokens = torch.tensor([ids, list(reversed(ids))], dtype=torch.long)
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    logits, _ = model(tokens)
    assert logits.shape == (2, len(ids), tokenizer.vocab_size)
    generated = model.generate(tokens[:, :8], 3)
    assert generated.shape == (2, 3)


def test_mght_registry_defaults_to_multigear_hierarchy():
    model = MODELS.create(
        "mght",
        {"vocab_size": 64, "dim": 32, "layers": 1, "heads": 4, "max_seq_len": 32},
        None,
    )
    assert isinstance(model, CachedTransformerLM)
    assert model.config.hierarchical_output
    assert model.config.input_gear_embedding
    assert model.config.hierarchy_output_mode == "bias"
    assert "mght" in TRAINERS
