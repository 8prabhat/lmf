"""MultiGear-aware transformer objectives and segmentation augmentation."""

from __future__ import annotations

import torch

from lmf.core.build import build, configure_token_hierarchy, initialize_token_embeddings
from lmf.core.config import ExperimentConfig
from lmf.data import (
    InMemoryTextCorpus,
    MultiGearTextCorpus,
    MultiGearTokenizer,
    SpecialTokenTokenizer,
)
from lmf.data.batch import TrainingBatch
from lmf.models.transformer import CachedTransformerLM, TransformerConfig, TransformerTrainer


_TEXT = (
    "alpha beta gamma delta epsilon zeta eta theta. "
    "alpha beta gamma delta epsilon zeta eta theta. "
) * 50


def _tokenizer():
    base = MultiGearTokenizer(max_vocab=430, max_token_bytes=16)
    base.train([_TEXT])
    return SpecialTokenTokenizer(base)


def _hierarchical_model(tokenizer, *, auxiliary: bool = False):
    model = CachedTransformerLM(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=2,
            heads=4,
            max_seq_len=64,
            hierarchical_output=True,
            hierarchy_gears=6,
            hierarchy_aux_weight=0.2 if auxiliary else 0.0,
            hierarchy_aux_min_gear=1,
        )
    )
    assert configure_token_hierarchy(model, tokenizer)
    initialize_token_embeddings(model, tokenizer, "merge_compositional")
    return model


def test_special_wrapped_hierarchy_covers_vocab_and_reconstructs_children():
    tokenizer = _tokenizer()
    metadata = tokenizer.token_hierarchy()
    assert len(metadata["token_gears"]) == tokenizer.vocab_size
    assert len(metadata["token_children"]) == tokenizer.vocab_size
    assert metadata["gear_count"] == 6
    assert metadata["token_gears"][-1] == 5
    assert metadata["token_bytes"][-1] == []
    base = tokenizer.base
    token_id = next(
        token_id for token_id, children in enumerate(metadata["token_children"])
        if children[0] >= 0
    )
    left, right = metadata["token_children"][token_id]
    assert base._vocab[token_id] == base._vocab[left] + base._vocab[right]


def test_hierarchical_output_is_normalized_and_backpropagates():
    tokenizer = _tokenizer()
    model = _hierarchical_model(tokenizer)
    tokens = torch.tensor([tokenizer.encode(_TEXT[:100])[:32]], dtype=torch.long)
    scores, _ = model(tokens)
    assert torch.allclose(scores.exp().sum(-1), torch.ones_like(scores[..., 0]), atol=1e-5)
    losses = model.training_step(tokens)
    assert {"gear_prediction", "within_gear", "language_modeling", "total"} <= losses.keys()
    losses["total"].backward()
    assert model.gear_head.weight.grad is not None


def test_hierarchy_auxiliary_loss_backpropagates_to_slots():
    tokenizer = _tokenizer()
    model = _hierarchical_model(tokenizer, auxiliary=True)
    tokens = torch.tensor([tokenizer.encode(_TEXT[:160])[:48]], dtype=torch.long)
    losses = model.training_step(tokens)
    assert losses["hierarchy_aux"] > 0
    losses["total"].backward()
    assert model.decomposition_slots.grad is not None
    assert torch.isfinite(model.decomposition_slots.grad).all()


def test_immediate_child_auxiliary_target_is_supported():
    tokenizer = _tokenizer()
    model = CachedTransformerLM(
        TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=1,
            heads=4,
            hierarchy_aux_weight=0.2,
            hierarchy_aux_min_gear=1,
            hierarchy_aux_target="children",
        )
    )
    configure_token_hierarchy(model, tokenizer)
    tokens = torch.tensor([tokenizer.encode(_TEXT[:160])[:48]], dtype=torch.long)
    losses = model.training_step(tokens)
    assert losses["hierarchy_aux"] > 0


def test_segmentation_dropout_preserves_text_and_loss_masks():
    tokenizer = _tokenizer()
    corpus = InMemoryTextCorpus(_TEXT, tokenizer=tokenizer, wrap_special=False)
    model = CachedTransformerLM(
        TransformerConfig(vocab_size=tokenizer.vocab_size, dim=32, layers=1, heads=4)
    )
    trainer = TransformerTrainer(
        model,
        corpus,
        device="cpu",
        precision="fp32",
        segmentation_dropout_prob=1.0,
        segmentation_dropout_min_gear=1,
        segmentation_dropout_max_depth=1,
    )
    metadata = tokenizer.token_hierarchy()
    token_id = next(
        token_id for token_id, children in enumerate(metadata["token_children"])
        if children[0] >= 0 and metadata["token_gears"][token_id] >= 1
    )
    eos = tokenizer.special_to_id["<|eos|>"]
    pad = tokenizer.special_to_id["<|pad|>"]
    tokens = torch.tensor([[token_id, eos, pad, pad]], dtype=torch.long)
    attention = torch.tensor([[True, True, False, False]])
    loss = torch.tensor([[True, False, False, False]])
    augmented = trainer._apply_segmentation_dropout(
        TrainingBatch(tokens, attention, loss)
    )
    before = tokenizer.decode(tokens[0, attention[0]].tolist())
    after = tokenizer.decode(augmented.tokens[0, augmented.attention_mask[0]].tolist())
    assert after == before
    assert int(augmented.loss_mask.sum()) == 2
    assert augmented.metadata["segmentation_dropout_replacements"] == 1


def test_segmentation_dropout_overflow_keeps_supervised_tail():
    tokenizer = _tokenizer()
    corpus = InMemoryTextCorpus(_TEXT, tokenizer=tokenizer, wrap_special=False)
    model = CachedTransformerLM(
        TransformerConfig(vocab_size=tokenizer.vocab_size, dim=32, layers=1, heads=4)
    )
    trainer = TransformerTrainer(
        model,
        corpus,
        device="cpu",
        precision="fp32",
        segmentation_dropout_prob=1.0,
        segmentation_dropout_min_gear=0,
        segmentation_dropout_max_depth=1,
    )
    metadata = tokenizer.token_hierarchy()
    token_id = next(
        token_id for token_id, children in enumerate(metadata["token_children"])
        if children[0] >= 0
    )
    tokens = torch.tensor([[token_id] * 4], dtype=torch.long)
    attention = torch.ones_like(tokens, dtype=torch.bool)
    loss = torch.tensor([[False, False, False, True]])
    augmented = trainer._apply_segmentation_dropout(TrainingBatch(tokens, attention, loss))
    assert int(augmented.loss_mask.sum()) >= 1
    assert bool((augmented.loss_mask & ~augmented.attention_mask).any()) is False


def test_all_multigear_enhancements_train_together():
    tokenizer = _tokenizer()
    corpus = InMemoryTextCorpus(_TEXT, tokenizer=tokenizer, wrap_special=False)
    model = _hierarchical_model(tokenizer, auxiliary=True)
    trainer = TransformerTrainer(
        model,
        corpus,
        device="cpu",
        precision="fp32",
        lr=3e-3,
        warmup_steps=1,
        total_steps=3,
        segmentation_dropout_prob=0.25,
        segmentation_dropout_min_gear=2,
        segmentation_dropout_max_depth=1,
    )
    records = trainer.train_steps(3, batch_size=2, seq_len=32, log_every=0)
    assert len(records) == 3
    assert all("hierarchy_aux" in record and "gear_prediction" in record for record in records)
    assert all(torch.isfinite(torch.tensor(record["total"])) for record in records)


def test_normal_build_wires_all_multigear_enhancements():
    cfg = _enhanced_config()
    corpus, model, trainer, run = build(cfg)
    assert isinstance(corpus, MultiGearTextCorpus)
    assert bool(model._gear_active.any())
    assert trainer.segmentation_dropout_prob == 0.1
    assert len(trainer.train_steps(1, run["batch_size"], run["seq_len"], log_every=0)) == 1


def test_flat_transformer_does_not_request_token_hierarchy():
    class OrdinaryTokenizer:
        def token_hierarchy(self):
            raise AssertionError("flat model must not request hierarchy")

    model = CachedTransformerLM(
        TransformerConfig(vocab_size=300, dim=16, layers=1, heads=2, max_seq_len=16)
    )
    assert configure_token_hierarchy(model, OrdinaryTokenizer()) is False


def _enhanced_config():
    return ExperimentConfig(
        {
            "seed": 0,
            "device": "cpu",
            "precision": "fp32",
            "data": {
                "name": "multigear_text",
                "text": _TEXT,
                "max_vocab": 430,
            },
            "model": {
                "name": "transformer",
                "dim": 32,
                "layers": 1,
                "heads": 4,
                "max_seq_len": 64,
                "token_embedding_init": "merge_compositional",
                "hierarchical_output": True,
                "hierarchy_gears": 6,
                "hierarchy_aux_weight": 0.1,
                "hierarchy_aux_target": "bytes",
            },
            "trainer": {
                "name": "transformer",
                "segmentation_dropout_prob": 0.1,
                "segmentation_dropout_min_gear": 2,
            },
            "run": {"batch_size": 2, "seq_len": 32, "steps": 1},
        },
        "test",
    )


def test_enhanced_checkpoint_roundtrip(tmp_path):
    _, _, trainer, run = build(_enhanced_config())
    trainer.train_steps(1, run["batch_size"], run["seq_len"], log_every=0)
    path = tmp_path / "enhanced.pt"
    trainer.save_checkpoint(path)
    _, fresh_model, fresh_trainer, _ = build(_enhanced_config())
    fresh_trainer.load_checkpoint(path)
    assert fresh_trainer.step == 1
    assert bool(fresh_model._gear_active.any())
