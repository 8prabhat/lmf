"""Research-script utilities: family-aware checkpoint loading and SVG charts."""

from __future__ import annotations

import torch

from lmf.models.pure_parallel_gear import PureParallelGearConfig, PureParallelGearLM
from lmf.research_utils import bar_chart_svg, load_model
from lmf.training.checkpoints import save_checkpoint


def _tiny_model() -> PureParallelGearLM:
    torch.manual_seed(0)
    config = PureParallelGearConfig(
        vocab_size=37,
        dim=16,
        layers=1,
        ffn_dim=32,
        num_banks=1,
        gears_per_bank=2,
        rotor_channels=2,
        predictor_gears=2,
        settling_rounds=1,
        max_sentence_tokens=8,
        max_seq_len=64,
    )
    return PureParallelGearLM(config)


def test_load_model_round_trips_pure_parallel_gear_checkpoint(tmp_path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=0)

    loaded = load_model(path, device="cpu")

    assert isinstance(loaded, PureParallelGearLM)
    for original, restored in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(original, restored)


def test_load_model_rejects_unsupported_architecture(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"manifest": {"name": "NotAModel"}, "config": {}, "model": {}}, path)
    try:
        load_model(path, device="cpu")
    except ValueError as exc:
        assert "unsupported checkpoint architecture" in str(exc)
    else:
        raise AssertionError("expected ValueError for unsupported architecture")


def test_bar_chart_svg_ordinary_writes_valid_svg(tmp_path):
    path = tmp_path / "bars.svg"
    bar_chart_svg(path, ["a", "b", "c"], [1.0, 2.0, 3.0], "Title", lower_is_better=True)
    content = path.read_text()
    assert content.startswith("<svg")
    assert content.count("<rect") >= 4  # background + 3 bars
    assert "lower is better" in content


def test_bar_chart_svg_diverging_colors_by_sign(tmp_path):
    path = tmp_path / "diverging.svg"
    bar_chart_svg(path, ["x", "y"], [2.0, -1.0], "Diff", diverging=True, baseline=0.0)
    content = path.read_text()
    assert "#b2182b" in content  # positive delta
    assert "#2166ac" in content  # negative delta
