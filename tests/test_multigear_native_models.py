from __future__ import annotations

import json

import numpy as np
import torch

from lmf.core.config import ExperimentConfig
from lmf.core.build import build
from lmf.core.registry import MODELS, TRAINERS
from lmf.ablation.points import discover_points
from lmf.data import EduCombinedCorpus, MultiGearTokenizer, ProceduralCorpus, SpecialTokenTokenizer
from lmf.evaluation import bits_per_token
from lmf.models.native import (
    MGCFConfig,
    MRWTConfig,
    MultiGearFractalCausalFieldLM,
    MultiGearResidualWorkbenchTransformerLM,
    NativeLMConfig,
)
from lmf.models.native.model import (
    MultiGearConstructiveProgramMachineLM,
    MultiGearElasticCausalMeshLM,
)
from lmf.models.native.trainer import NativeLMTrainer


def _tiny_edu_root(tmp_path):
    root = tmp_path / "edu"
    domain = root / "toy"
    domain.mkdir(parents=True)
    train = (np.arange(512, dtype=np.uint16) % 64).astype(np.uint16)
    train.tofile(domain / "train_bpe32768_v2.bin")
    manifest = {
        "dtype": "uint16",
        "train_tokens": int(train.size),
        "vocab_size": 64,
        "tokenizer_fingerprint": "test",
    }
    (domain / "train_bpe32768_v2.bin.manifest.json").write_text(json.dumps(manifest))
    torch.save(torch.arange(160, dtype=torch.int32) % 64, domain / "valid_bpe32768_v2.pt")
    torch.save((torch.arange(160, dtype=torch.int32) + 1) % 64, domain / "test_bpe32768_v2.pt")
    return root


def test_edu_combined_memmap_sampling_and_sampler_state(tmp_path):
    corpus = EduCombinedCorpus(str(_tiny_edu_root(tmp_path)), load_tokenizer=False, seed=123)
    assert corpus.vocab_size == 64
    batch = corpus.sample_tokenized(4, 32, "train")
    assert batch.shape == (4, 32)
    assert batch.dtype == torch.long
    valid = corpus.sample_tokenized(2, 24, "valid")
    test = corpus.sample_tokenized(2, 24, "test")
    assert valid.shape == test.shape == (2, 24)

    state = corpus.sampler_state()
    first = corpus.sample_tokenized(2, 16, "train")
    corpus.load_sampler_state(state)
    again = corpus.sample_tokenized(2, 16, "train")
    assert torch.equal(first, again)


def test_native_registries_are_populated():
    for name in ("mecm", "mcpm", "mgcf", "mrwt"):
        assert name in MODELS
        assert name in TRAINERS


def _exercise_model(model):
    tokens = torch.randint(0, 64, (3, 20))
    logits, _ = model(tokens)
    assert logits.shape == (3, 20, 64)
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    out = model.generate(tokens[:, :5], 7)
    assert out.shape == (3, 7)


def test_mecm_forward_backward_generate():
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(vocab_size=64, dim=32, layers=2, kernel_size=5, max_seq_len=64)
    )
    _exercise_model(model)


def test_mcpm_forward_backward_generate():
    model = MultiGearConstructiveProgramMachineLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            execution_residual=True,
        )
    )
    _exercise_model(model)


def test_mrwt_forward_backward_generate_and_anchor_fallback():
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(vocab_size=64, dim=32, layers=2, heads=4, max_seq_len=64)
    )
    tokens = torch.randint(0, 64, (2, 16))
    logits, _ = model(tokens)
    anchor = model.anchor_logits(tokens)
    assert torch.allclose(logits, anchor, atol=1e-6)
    _exercise_model(model)


def test_full_mecm_exposes_research_losses_and_ablation_points():
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=2,
            route_aux_weight=0.01,
            draft_aux_weight=0.01,
        )
    )
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert {"route_balance", "draft_tree"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    points = discover_points(model)
    assert "span_atlas.scales.skip[0]" in points
    assert "active_cover.bypass" in points
    assert "reasoning_mesh.layers.skip[0]" in points


def test_mecm_gear_aware_output_uses_token_hierarchy():
    base = MultiGearTokenizer(max_vocab=340, max_token_bytes=16)
    text = "alpha beta gamma delta conference Australia India Morgan " * 20
    base.train([text])
    tokenizer = SpecialTokenTokenizer(base)
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=96,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=1,
            draft_aux_weight=0.0,
            gear_aware_output=True,
            gear_aux_weight=0.01,
        )
    )
    model.configure_token_hierarchy(**tokenizer.token_hierarchy())
    ids = tokenizer.encode(text)[:48]
    tokens = torch.tensor([ids, list(reversed(ids))], dtype=torch.long)
    losses = model.training_step(tokens)
    assert {"gear_prediction", "within_gear", "gear_aux"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    logits, _ = model(tokens)
    assert logits.shape == (2, len(ids), tokenizer.vocab_size)
    generated = model.generate(tokens[:, :8], 3)
    assert generated.shape == (2, 3)


def test_mecm_gear_bias_output_uses_auxiliary_gear_loss():
    base = MultiGearTokenizer(max_vocab=340, max_token_bytes=16)
    text = "alpha beta gamma delta conference Australia India Morgan " * 20
    base.train([text])
    tokenizer = SpecialTokenTokenizer(base)
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=96,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=1,
            draft_aux_weight=0.0,
            gear_aware_output=True,
            gear_output_mode="bias",
            gear_aux_weight=0.01,
        )
    )
    model.configure_token_hierarchy(**tokenizer.token_hierarchy())
    ids = tokenizer.encode(text)[:48]
    tokens = torch.tensor([ids, list(reversed(ids))], dtype=torch.long)
    losses = model.training_step(tokens)
    assert "gear_aux" in losses
    assert torch.isfinite(losses["total"])
    logits, _ = model(tokens)
    assert logits.shape == (2, len(ids), tokenizer.vocab_size)


def _tiny_multigear_tokenizer():
    base = MultiGearTokenizer(max_vocab=340, max_token_bytes=16)
    text = "alpha beta gamma delta conference Australia India Morgan " * 20
    base.train([text])
    return SpecialTokenTokenizer(base), text


def test_mgcf_forward_backward_generate_and_causal_non_leakage():
    tokenizer, text = _tiny_multigear_tokenizer()
    model = MultiGearFractalCausalFieldLM(
        MGCFConfig(
            vocab_size=tokenizer.vocab_size,
            dim=32,
            layers=2,
            kernel_size=3,
            dilations=(1, 2, 4),
            memory_scales=(2, 5, 9),
            max_seq_len=96,
            gear_output_mode="bias",
            gear_aux_weight=0.01,
            composition_aux_weight=0.01,
        )
    )
    model.configure_token_hierarchy(**tokenizer.token_hierarchy())
    ids = tokenizer.encode(text)[:48]
    tokens = torch.tensor([ids, list(reversed(ids))], dtype=torch.long)
    logits, _ = model(tokens)
    assert logits.shape == (2, len(ids), tokenizer.vocab_size)
    losses = model.training_step(tokens)
    assert {"language_modeling", "gear_aux", "composition_aux"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    generated = model.generate(tokens[:, :8], 3)
    assert generated.shape == (2, 3)
    assert model._token_children.shape == (tokenizer.vocab_size, 2)
    assert bool((model._token_byte_lengths > 1).any())

    changed = tokens.clone()
    changed[:, 24:] = torch.flip(changed[:, 24:], dims=[1])
    with torch.no_grad():
        before, _ = model(tokens)
        after, _ = model(changed)
    assert torch.allclose(before[:, :24], after[:, :24], atol=1e-5)

    long_prompt = tokens[:, :40]
    context = model._generation_context()
    assert context < model.config.max_seq_len
    with torch.no_grad():
        full_hidden = model._forward_hidden(long_prompt)
        window = long_prompt[:, -context:]
        offset = long_prompt.shape[1] - window.shape[1]
        cropped_hidden = model._forward_hidden(window, position_offset=offset)
    assert torch.allclose(full_hidden[:, -1], cropped_hidden[:, -1], atol=1e-5)


def test_mgcf_registry_builds_with_multigear_hierarchy():
    tokenizer, text = _tiny_multigear_tokenizer()
    cfg = {
        "name": "mgcf",
        "dim": 32,
        "layers": 1,
        "kernel_size": 3,
        "dilations": [1, 2],
        "memory_scales": [2, 5],
        "max_seq_len": 96,
        "gear_output_mode": "factorized",
    }
    model = MODELS.create("mgcf", {k: v for k, v in cfg.items() if k != "name"}, tokenizer.vocab_size)
    model.configure_token_hierarchy(**tokenizer.token_hierarchy())
    ids = tokenizer.encode(text)[:32]
    tokens = torch.tensor([ids], dtype=torch.long)
    losses = model.training_step(tokens)
    assert {"gear_prediction", "within_gear"}.issubset(losses)
    assert torch.isfinite(losses["total"])


def test_full_mcpm_exposes_program_execution_and_verifier_points():
    model = MultiGearConstructiveProgramMachineLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            execution_residual=True,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=2,
            route_aux_weight=0.01,
            draft_aux_weight=0.01,
            program_aux_weight=0.01,
            verifier_aux_weight=0.01,
        )
    )
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert {"program_controller", "contract_verifier", "draft_tree"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    points = discover_points(model)
    assert "program_controller.bypass" in points
    assert "execution_workbench.rounds.skip[0]" in points
    assert "contract_verifier.bypass" in points


def test_full_mrwt_exposes_workbench_points_and_preserves_anchor_at_zero_gate():
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            heads=4,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            workbench_rounds=2,
            budget_aux_weight=0.01,
            draft_aux_weight=0.01,
        )
    )
    tokens = torch.randint(0, 64, (2, 20))
    logits, _ = model(tokens)
    assert torch.allclose(logits, model.anchor_logits(tokens), atol=1e-6)
    losses = model.training_step(tokens)
    assert {"budget_controller", "draft_tree"}.issubset(losses)
    assert torch.isfinite(losses["total"])
    points = discover_points(model)
    assert "atlas.scales.skip[0]" in points
    assert "budget_controller.bypass" in points
    assert "workbench_rounds.skip[0]" in points


def test_full_native_training_reuses_single_hidden_pass(monkeypatch):
    model = MultiGearConstructiveProgramMachineLM(
        NativeLMConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            kernel_size=5,
            max_seq_len=64,
            execution_residual=True,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            mesh_layers=2,
            route_aux_weight=0.01,
            draft_aux_weight=0.01,
            program_aux_weight=0.01,
            verifier_aux_weight=0.01,
        )
    )
    calls = 0
    original = model._forward_hidden

    def counted_forward(ids):
        nonlocal calls
        calls += 1
        return original(ids)

    monkeypatch.setattr(model, "_forward_hidden", counted_forward)
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    assert calls == 1


def test_full_mrwt_training_reuses_single_hidden_pass(monkeypatch):
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            heads=4,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            workbench_rounds=2,
            budget_aux_weight=0.01,
            draft_aux_weight=0.01,
        )
    )
    calls = 0
    original = model._forward_hidden

    def counted_forward(ids, attention_mask=None):
        nonlocal calls
        calls += 1
        return original(ids, attention_mask=attention_mask)

    monkeypatch.setattr(model, "_forward_hidden", counted_forward)
    tokens = torch.randint(0, 64, (2, 20))
    losses = model.training_step(tokens)
    assert torch.isfinite(losses["total"])
    assert calls == 1


def test_mrwt_zero_gate_generation_uses_anchor_cache(monkeypatch):
    model = MultiGearResidualWorkbenchTransformerLM(
        MRWTConfig(
            vocab_size=64,
            dim=32,
            layers=2,
            heads=4,
            max_seq_len=64,
            full_architecture=True,
            atlas_kernel_sizes=(3, 5, 9),
            workbench_rounds=2,
        )
    )
    called = False

    def fake_generate(prompt_tokens, max_new_tokens, sampling_config=None):
        nonlocal called
        called = True
        return torch.zeros(
            prompt_tokens.shape[0],
            max_new_tokens,
            dtype=torch.long,
            device=prompt_tokens.device,
        )

    monkeypatch.setattr(model.anchor, "generate", fake_generate)
    out = model.generate(torch.randint(0, 64, (2, 5)), 7)
    assert called
    assert out.shape == (2, 7)


def test_native_trainer_and_bpt_smoke():
    corpus = ProceduralCorpus(vocab_size=64)
    model = MultiGearElasticCausalMeshLM(
        NativeLMConfig(vocab_size=64, dim=32, layers=2, kernel_size=5, max_seq_len=64)
    )
    trainer = NativeLMTrainer(
        model,
        corpus,
        device="cpu",
        precision="fp32",
        warmup_steps=1,
        total_steps=3,
        lr=1e-3,
    )
    records = trainer.train_steps(2, batch_size=2, seq_len=24, log_every=0)
    assert len(records) == 2
    assert bits_per_token(model, corpus, batch_size=2, seq_len=24, n_batches=1) > 0


def test_build_from_config_for_all_native_models():
    for name in ("mecm", "mcpm", "mrwt"):
        cfg = ExperimentConfig(
            {
                "seed": 0,
                "device": "cpu",
                "precision": "fp32",
                "data": {"name": "procedural", "vocab_size": 64},
                "model": {
                    "name": name,
                    "vocab_size": 64,
                    "dim": 32,
                    "layers": 2,
                    **({"heads": 4} if name == "mrwt" else {"kernel_size": 5}),
                    "max_seq_len": 64,
                },
                "trainer": {"name": name, "lr": 1e-3, "total_steps": 2, "warmup_steps": 1},
                "run": {"batch_size": 2, "seq_len": 24, "steps": 2},
            },
            "test",
        )
        corpus, model, trainer, run = build(cfg)
        assert corpus.vocab_size == 64
        assert torch.isfinite(model.training_step(corpus.sample_tokenized(2, 24))["total"])
        assert run["steps"] == 2
