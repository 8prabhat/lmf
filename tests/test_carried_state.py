"""Bounded carried-state training and the minimal loss set."""

from __future__ import annotations

import torch


EXPECTED_LOSSES = {"commit_token", "routing_balance"}


def test_carried_state_step_returns_pruned_losses(tiny_model, tiny_tokens):
    losses = tiny_model.carried_state_training_step(tiny_tokens)
    assert EXPECTED_LOSSES <= set(losses)
    assert "total" in losses
    assert set(losses) == EXPECTED_LOSSES | {"total"}
    for v in losses.values():
        assert torch.isfinite(v).all()


def test_carried_state_is_differentiable(tiny_model, tiny_tokens):
    losses = tiny_model.carried_state_training_step(tiny_tokens)
    losses["total"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in tiny_model.parameters())


def test_training_reduces_loss(tiny_model, tiny_tokens):
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)
    first = None
    last = None
    for i in range(25):
        opt.zero_grad()
        loss = tiny_model.carried_state_training_step(tiny_tokens)["commit_token"]
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < first  # the model can actually fit the toy sequence


def test_training_window_count_is_bounded(tiny_model, tiny_config, monkeypatch):
    calls = 0
    original = tiny_model.settle

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_model, "settle", counted)
    tokens = torch.randint(3, tiny_config.vocab_size, (2, tiny_config.frontier_size * 10))
    tiny_model.carried_state_training_step(tokens, max_train_windows=2)
    assert calls == 2


def test_training_advances_by_max_commit_carrying_settled_frontier(tiny_model, tiny_config):
    """Finding 1: training must use the same state transition as inference —
    commit max_commit tokens and carry the settled frontier (via _advance_state)."""
    h, mc = tiny_config.frontier_size, tiny_config.max_commit
    tokens = torch.randint(3, tiny_config.vocab_size, (2, h * 3))
    state = tiny_model.prefill(tokens[:, :h])
    settled, _ = tiny_model.settle(state, active_only=True)
    advanced = tiny_model._advance_state(state, settled, tokens[:, h:h + mc])
    # committed_count moved by exactly max_commit, not frontier_size.
    assert int(advanced.committed_count[0] - state.committed_count[0]) == mc


def test_trainable_interface_delegates(tiny_model, tiny_tokens):
    via_interface = tiny_model.training_step(tiny_tokens, {"max_train_windows": 2})
    assert "total" in via_interface and torch.isfinite(via_interface["total"]).all()
