from __future__ import annotations

import torch

from lmf.core.registry import MODELS
from lmf.data import MultiGearTokenizer, SpecialTokenTokenizer
from lmf.models.mgcf import MGCFConfig, MultiGearFractalCausalFieldLM


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
