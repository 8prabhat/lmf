from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from lmf.core.registry import MODELS, TRAINERS
from lmf.data import (
    NumericFallbackTokenizer,
    PairedDocumentManifestCorpus,
    SentenceBoundaryDetector,
    build_document_index,
    build_paired_training_manifest,
)
from lmf.models.pure_parallel_gear import (
    GearState,
    PureGearLayer,
    PureParallelGearConfig,
    PureParallelGearLM,
    PureParallelGearTrainer,
)
from lmf.models.pure_parallel_gear.model import _rotate
from lmf.training.checkpoints import load_checkpoint
from scripts.benchmark_pure_parallel_gear import (
    assert_fair_configs,
    configs as benchmark_configs,
    gear_parameter_count,
    throughput,
)
from lmf.diagnostics import cache_bytes


def config(**overrides) -> PureParallelGearConfig:
    values = {
        "vocab_size": 97,
        "dim": 32,
        "layers": 2,
        "ffn_dim": 64,
        "num_banks": 2,
        "gears_per_bank": 4,
        "rotor_channels": 2,
        "predictor_gears": 3,
        "settling_rounds": 2,
        "max_sentence_tokens": 8,
        "max_seq_len": 256,
    }
    values.update(overrides)
    return PureParallelGearConfig(**values)


def model(**overrides) -> PureParallelGearLM:
    return PureParallelGearLM(config(**overrides))


def boundaries(tokens: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(tokens, dtype=torch.bool)
    result[:, 5] = True
    result[:, 11] = True
    return result


def test_only_canonical_gear_family_is_registered():
    assert "pure_parallel_gear" in MODELS
    assert "pure_parallel_gear" in TRAINERS
    assert "parallel_gear_lm" not in MODELS
    assert "parallel_gear_v2" not in MODELS
    instance = model()
    assert all(isinstance(layer, PureGearLayer) for layer in instance.layers)
    manifest = instance.architecture_manifest()
    assert manifest["name"] == "PureParallelGear"
    for key in (
        "self_attention",
        "qkv_projections",
        "token_similarity",
        "history_retrieval",
        "history_tensor",
        "kv_cache",
        "token_routing",
        "transformer_blocks",
    ):
        assert manifest["invariants"][key] is False
    source = inspect.getsource(PureGearLayer)
    assert "scaled_dot_product_attention" not in source
    assert "softmax" not in source


def test_closed_form_rotor_matches_sequential_output_and_gradient():
    torch.manual_seed(3)
    layer = model(layers=1).layers[0]
    hidden_closed = torch.randn(13, layer.dim, requires_grad=True)
    hidden_loop = hidden_closed.detach().clone().requires_grad_(True)
    initial = layer.initial_state(1, hidden_closed.device)
    initial_loop = layer.initial_state(1, hidden_closed.device)
    closed, _, _ = layer._token_dynamics(
        hidden_closed, initial, fixed_omega=False
    )

    source = layer.input_norm(hidden_loop).float()
    shape = (len(hidden_loop), layer.banks, layer.gears, layer.channels)
    delta = layer.config.theta_limit * torch.tanh(
        layer.angle_projection(source).reshape(shape)
    )
    clutch = torch.sigmoid(layer.clutch_projection(source).reshape(shape))
    torque = layer.config.torque_limit * clutch[..., None] * torch.tanh(
        layer.torque_projection(source).reshape(*shape, 2)
    )
    retention = layer.retention_low + (
        layer.retention_high - layer.retention_low
    ) * torch.sigmoid(
        layer.retention_projection(source).reshape(shape)
    )
    rotor = initial_loop.rotor
    rows = []
    for index in range(len(hidden_loop)):
        rotor = (
            retention[index][None, ..., None]
            * _rotate(rotor, delta[index] + initial_loop.omega)
            + torque[index]
        )
        rows.append(rotor)
    sequential = torch.cat(rows, dim=0)
    closed.square().sum().backward()
    sequential.square().sum().backward()
    assert torch.allclose(closed, sequential, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        hidden_closed.grad, hidden_loop.grad, atol=2e-5, rtol=2e-5
    )


def test_affine_retention_provides_selective_forgetting():
    layer = model(layers=1).layers[0]
    hidden = torch.zeros(6, layer.dim)
    state = layer.initial_state(1, hidden.device)
    with torch.no_grad():
        layer.angle_projection.weight.zero_()
        layer.torque_projection.weight.zero_()
        layer.retention_projection.weight.zero_()
        layer.retention_projection.bias.fill_(-8.0)
        layer.base_omega.zero_()
    rotor, _, retention = layer._token_dynamics(
        hidden, state, fixed_omega=False
    )
    assert bool((retention < 1.0).all())
    norms = rotor.square().sum(dim=-1).sqrt().mean(dim=(1, 2, 3))
    assert bool((norms[1:] < norms[:-1]).all())


def test_bank_retention_ranges_remain_timescale_separated():
    layer = model(layers=1, num_banks=4).layers[0]
    assert bool(
        (layer.retention_high[:-1] <= layer.retention_low[1:]).all()
    )
    hidden = torch.randn(7, layer.dim)
    _, _, retention = layer._token_dynamics(
        hidden,
        layer.initial_state(1, hidden.device),
        fixed_omega=False,
    )
    bank_means = retention.mean(dim=(0, 2, 3))
    assert bool((bank_means[1:] > bank_means[:-1]).all())


def test_readout_preserves_radial_information():
    layer = model(layers=1).layers[0]
    state = layer.initial_state(1, torch.device("cpu"))
    clutch = torch.full_like(state.omega, 0.5)
    unit = layer._readout(
        state.rotor,
        state.omega,
        state.load,
        clutch,
        state.rotor,
    )
    scaled = layer._readout(
        2.0 * state.rotor,
        state.omega,
        state.load,
        clutch,
        state.rotor,
    )
    assert not torch.allclose(unit, scaled)


def test_predictor_residual_has_nonzero_architectural_floor():
    instance = model(layers=1)
    assert instance.predictor is not None
    with torch.no_grad():
        instance.predictor.gear_residual.fill_(-100.0)
    tokens = torch.randint(0, instance.config.vocab_size, (1, 9))
    records = instance.diagnostics(tokens)
    assert float(records[-1]["gear_residual_scale"]) == pytest.approx(
        instance.config.predictor_residual_floor,
        abs=1e-6,
    )


def test_long_sentence_receives_explicit_intra_sentence_clutch():
    layer = model(
        layers=1,
        max_sentence_tokens=128,
        intra_sentence_clutch_tokens=8,
    ).layers[0]
    hidden = torch.randn(1, 19, layer.dim)
    mask = torch.ones(1, 19, dtype=torch.bool)
    segments = torch.zeros(1, 19, dtype=torch.long)
    ends = torch.zeros(1, 19, dtype=torch.bool)
    _, state, record = layer(
        hidden,
        token_mask=mask,
        segment_ids=segments,
        sentence_end_mask=ends,
    )
    assert float(record["coupling_activity"].detach()) > 0.0
    assert int(state.sentence_length.item()) == 19


def test_production_forward_uses_sentence_scan_not_token_loop():
    source = inspect.getsource(PureGearLayer.forward)
    assert "for t in range(length)" not in source
    assert (
        model().architecture_manifest()["invariants"]["sentence_execution"]
        == "parallel_affine_scan"
    )


def test_full_and_streaming_logits_match_with_constant_cache():
    torch.manual_seed(4)
    instance = model().eval()
    tokens = torch.randint(0, 97, (2, 19))
    ends = boundaries(tokens)
    full, _ = instance(tokens, sentence_end_mask=ends)
    cache = None
    pieces = []
    sizes = []
    for position in range(tokens.shape[1]):
        logits, cache = instance(
            tokens[:, position : position + 1],
            cache=cache,
            use_cache=True,
            sentence_end_mask=ends[:, position : position + 1],
        )
        pieces.append(logits)
        sizes.append(cache_bytes(cache))
    assert torch.allclose(
        full, torch.cat(pieces, dim=1), atol=2e-5, rtol=2e-5
    )
    assert len(set(sizes)) == 1
    assert cache.layers[0].rotor.shape == (2, 2, 4, 2, 2)


def test_streaming_matches_full_across_intra_sentence_clutches():
    torch.manual_seed(41)
    instance = model(
        layers=1,
        max_sentence_tokens=64,
        intra_sentence_clutch_tokens=4,
    ).eval()
    tokens = torch.randint(0, 97, (2, 17))
    ends = torch.zeros_like(tokens, dtype=torch.bool)
    full, _ = instance(tokens, sentence_end_mask=ends)
    cache = None
    pieces = []
    for position in range(tokens.shape[1]):
        logits, cache = instance(
            tokens[:, position : position + 1],
            cache=cache,
            use_cache=True,
            sentence_end_mask=ends[:, position : position + 1],
        )
        pieces.append(logits)
    assert torch.allclose(
        full,
        torch.cat(pieces, dim=1),
        atol=2e-5,
        rtol=2e-5,
    )
    assert int(cache.layers[0].sentence_length[0]) == 17


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is required for the split control/data state regression",
)
def test_mps_forward_keeps_control_state_on_cpu():
    instance = model(layers=1).to("mps").eval()
    tokens = torch.randint(0, 97, (2, 19), device="mps")
    ends = boundaries(tokens)
    logits, cache = instance(
        tokens,
        use_cache=True,
        sentence_end_mask=ends,
    )
    torch.mps.synchronize()
    assert logits.device.type == "mps"
    assert cache.layers[0].rotor.device.type == "mps"
    assert cache.layers[0].sentence_length.device.type == "cpu"
    assert cache.layers[0].segment_id.device.type == "cpu"


def _state_index(state: GearState, index: int) -> GearState:
    return GearState(
        state.rotor[index : index + 1],
        state.omega[index : index + 1],
        state.load[index : index + 1],
        state.sentence_length[index : index + 1],
        state.segment_id[index : index + 1],
    )


def _state_cat(states: list[GearState]) -> GearState:
    return GearState(
        torch.cat([state.rotor for state in states], dim=0),
        torch.cat([state.omega for state in states], dim=0),
        torch.cat([state.load for state in states], dim=0),
        torch.cat([state.sentence_length for state in states], dim=0),
        torch.cat([state.segment_id for state in states], dim=0),
    )


def _legacy_forward_row(layer, hidden, token_mask, segment_ids, sentence_end_mask, state, ablations):
    """Verbatim transcription of the pre-vectorization per-row reference loop.

    Kept only as an independent ground truth for
    test_vectorized_forward_matches_legacy_row_loop -- production forward()
    now processes the whole batch per timestep instead of per row per chunk.
    """
    outputs = []
    rotor_energy = []
    clutch_rows = []
    coupling_rows = []
    position = 0
    current_segment = int(state.segment_id.item())
    sentence_length = int(state.sentence_length.item())
    while position < hidden.shape[0]:
        if not bool(token_mask[position]):
            outputs.append(torch.zeros(layer.dim, device=hidden.device))
            position += 1
            continue
        segment = int(segment_ids[position])
        if current_segment != segment:
            state = layer.initial_state(1, hidden.device)
            state.segment_id.fill_(segment)
            current_segment = segment
            sentence_length = 0
        remaining = layer.config.max_sentence_tokens - sentence_length
        stop = min(hidden.shape[0], position + max(1, remaining))
        boundary = False
        for candidate in range(position, stop):
            if not bool(token_mask[candidate]) or int(segment_ids[candidate]) != segment:
                stop = candidate
                break
            if bool(sentence_end_mask[candidate]):
                stop = candidate + 1
                boundary = True
                break
        if stop == position:
            continue
        if stop - position >= remaining:
            boundary = True
        chunk = hidden[position:stop]
        rotor, clutch, retention = layer._token_dynamics(
            chunk,
            state,
            fixed_omega=(
                not layer.config.learned_angular_velocity
                or "fixed_angular_velocities" in ablations
            ),
        )
        omega = state.omega.expand(len(chunk), -1, -1, -1)
        load = state.load.expand(len(chunk), -1, -1, -1)
        next_state = GearState(
            rotor[-1:],
            state.omega,
            state.load,
            torch.full_like(state.sentence_length, sentence_length + len(chunk)),
            state.segment_id,
        )
        coupling = rotor.new_zeros(())
        if boundary and layer.config.boundary_settling and "no_boundary_settling" not in ablations:
            next_state, coupling = layer.settle(
                next_state,
                cross_bank=(
                    layer.config.cross_bank_coupling
                    and "no_cross_bank_coupling" not in ablations
                ),
                commuting_only=(
                    not layer.config.overlapping_coupling
                    or "commuting_coupling_only" in ablations
                ),
                use_load=(
                    layer.config.use_load_state and "no_load_state" not in ablations
                ),
            )
            rotor = torch.cat((rotor[:-1], next_state.rotor), dim=0)
            omega = torch.cat((omega[:-1], next_state.omega), dim=0)
            load = torch.cat((load[:-1], next_state.load), dim=0)
            sentence_length = 0
        else:
            sentence_length += len(chunk)
        previous_rotor = torch.cat((state.rotor, rotor[:-1]), dim=0)
        outputs.extend(
            layer._readout(
                rotor, omega, load, clutch, previous_rotor
            ).unbind(0)
        )
        rotor_energy.append(rotor.square().sum(dim=-1))
        clutch_rows.append(clutch)
        coupling_rows.append(coupling)
        state = next_state
        position = stop
    output = torch.stack(outputs, dim=0)
    energy = torch.cat(rotor_energy, dim=0)
    clutches = torch.cat(clutch_rows, dim=0)
    coupling_activity = (
        torch.stack(coupling_rows).mean() if coupling_rows else output.new_zeros(())
    )
    diagnostics = {
        "rotor_energy": energy,
        "clutch": clutches,
        "coupling_activity": coupling_activity,
        "omega": state.omega,
        "load": state.load,
        "rotor": state.rotor,
    }
    return output, state, diagnostics


def _legacy_forward(layer, hidden, token_mask, segment_ids, sentence_end_mask, state=None, ablations=()):
    hidden_dtype = hidden.dtype
    hidden = hidden.float()
    if state is None:
        state = layer.initial_state(hidden.shape[0], hidden.device)
    rows, states, diagnostics = [], [], []
    disabled = frozenset(ablations)
    for row in range(hidden.shape[0]):
        output, row_state, record = _legacy_forward_row(
            layer,
            hidden[row],
            token_mask[row],
            segment_ids[row],
            sentence_end_mask[row],
            _state_index(state, row),
            disabled,
        )
        rows.append(output)
        states.append(row_state)
        diagnostics.append(record)
    gear_output = torch.stack(rows)
    residual_scale = layer.residual_floor + (
        1.0 - layer.residual_floor
    ) * torch.sigmoid(layer.gear_residual)
    hidden = hidden + residual_scale * layer.dropout(gear_output)
    if layer.use_ffn and "no_local_swiglu" not in disabled:
        value, gate = layer.ffn_in(layer.ffn_norm(hidden)).chunk(2, dim=-1)
        feedforward = layer.ffn_out(F.silu(gate) * value)
        hidden = hidden + torch.tanh(layer.ffn_residual) * layer.dropout(feedforward)
    record = {
        "rotor_energy": torch.cat([item["rotor_energy"] for item in diagnostics], dim=0),
        "clutch": torch.cat([item["clutch"] for item in diagnostics], dim=0),
        "coupling_activity": torch.stack(
            [item["coupling_activity"] for item in diagnostics]
        ).mean(),
        "omega": torch.cat([item["omega"] for item in diagnostics], dim=0),
        "load": torch.cat([item["load"] for item in diagnostics], dim=0),
        "rotor": torch.cat([item["rotor"] for item in diagnostics], dim=0),
    }
    return hidden.to(hidden_dtype), _state_cat(states), record


def test_vectorized_forward_matches_legacy_row_loop():
    torch.manual_seed(11)
    layer = model(layers=1, max_sentence_tokens=5).layers[0]
    batch, length = 3, 23
    hidden = torch.randn(batch, length, layer.dim, requires_grad=True)
    hidden_ref = hidden.detach().clone().requires_grad_(True)

    token_mask = torch.ones(batch, length, dtype=torch.bool)
    token_mask[0, 17:20] = False
    token_mask[2, :3] = False

    segment_ids = torch.zeros(batch, length, dtype=torch.long)
    segment_ids[0, 9:] = 1
    segment_ids[1, 14:] = 1
    segment_ids[1, 20:] = 2

    sentence_end_mask = torch.zeros(batch, length, dtype=torch.bool)
    sentence_end_mask[0, 3] = True
    sentence_end_mask[1, 6] = True
    sentence_end_mask[1, 16] = True
    sentence_end_mask[2, 10] = True

    new_output, new_state, new_record = layer.forward(
        hidden,
        token_mask=token_mask,
        segment_ids=segment_ids,
        sentence_end_mask=sentence_end_mask,
    )
    ref_output, ref_state, ref_record = _legacy_forward(
        layer,
        hidden_ref,
        token_mask,
        segment_ids,
        sentence_end_mask,
    )

    assert torch.allclose(new_output, ref_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_record["rotor_energy"], ref_record["rotor_energy"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_record["clutch"], ref_record["clutch"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(new_state.rotor, ref_state.rotor, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_state.omega, ref_state.omega, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_state.load, ref_state.load, atol=1e-5, rtol=1e-5)
    assert torch.equal(new_state.sentence_length, ref_state.sentence_length)
    assert torch.equal(new_state.segment_id, ref_state.segment_id)

    new_output.square().sum().backward()
    ref_output.square().sum().backward()
    assert torch.allclose(hidden.grad, hidden_ref.grad, atol=2e-4, rtol=2e-4)


def _legacy_chunked_forward_row(
    layer,
    hidden,
    token_mask,
    segment_ids,
    sentence_end_mask,
    delta,
    clutch_controls,
    torque,
    retention_controls,
    state,
    *,
    fixed_omega,
    settling_enabled,
    cross_bank,
    commuting_only,
    use_load,
):
    """Verbatim transcription of the pre-chunk-parallel-rewrite per-row,
    per-chunk reference loop (intra_sentence_clutch_tokens + retention scan
    included). Kept only as ground truth for
    test_chunk_parallel_forward_matches_legacy_chunked_loop -- production
    forward() now batches all rows' chunks per step instead of looping row
    by row, chunk by chunk.
    """
    outputs = []
    rotor_rows = []
    clutch_rows = []
    retention_rows = []
    coupling_rows = []
    position = 0
    current_segment = int(state.segment_id.item())
    sentence_length = int(state.sentence_length.item())

    while position < hidden.shape[0]:
        if not bool(token_mask[position]):
            outputs.append(hidden.new_zeros(layer.dim))
            position += 1
            continue

        segment = int(segment_ids[position])
        if segment != current_segment:
            state = layer.initial_state(1, hidden.device)
            state.segment_id.fill_(segment)
            current_segment = segment
            sentence_length = 0

        remaining = layer.config.max_sentence_tokens - sentence_length
        clutch_interval = layer.config.intra_sentence_clutch_tokens
        clutch_remaining = (
            remaining
            if clutch_interval == 0
            else clutch_interval - sentence_length % clutch_interval
        )
        stop = min(
            hidden.shape[0],
            position + max(1, min(remaining, clutch_remaining)),
        )
        boundary = False
        for candidate in range(position, stop):
            if (
                not bool(token_mask[candidate])
                or int(segment_ids[candidate]) != segment
            ):
                stop = candidate
                break
            if bool(sentence_end_mask[candidate]):
                stop = candidate + 1
                boundary = True
                break
        if stop == position:
            continue
        if stop - position >= remaining:
            boundary = True
        micro_clutch = (
            not boundary
            and clutch_interval > 0
            and stop - position >= clutch_remaining
        )

        chunk = hidden[position:stop]
        clutch = clutch_controls[position:stop]
        retention = retention_controls[position:stop]
        rotor = layer._scan_token_dynamics(
            delta[position:stop],
            torque[position:stop],
            retention,
            state,
            fixed_omega=fixed_omega,
        )
        previous_rotor = torch.cat((state.rotor, rotor[:-1]), dim=0)
        omega = state.omega.expand(len(chunk), -1, -1, -1)
        load = state.load.expand(len(chunk), -1, -1, -1)
        next_state = GearState(
            rotor[-1:],
            state.omega,
            state.load,
            torch.full_like(state.sentence_length, sentence_length + len(chunk)),
            state.segment_id,
        )

        if boundary or micro_clutch:
            if settling_enabled:
                next_state, coupling = layer.settle(
                    next_state,
                    cross_bank=cross_bank,
                    commuting_only=commuting_only,
                    use_load=use_load,
                )
                rotor = torch.cat((rotor[:-1], next_state.rotor), dim=0)
                omega = torch.cat((omega[:-1], next_state.omega), dim=0)
                load = torch.cat((load[:-1], next_state.load), dim=0)
                coupling_rows.append(coupling)
            if boundary:
                next_state = GearState(
                    next_state.rotor,
                    next_state.omega,
                    next_state.load,
                    torch.zeros_like(next_state.sentence_length),
                    next_state.segment_id,
                )
                sentence_length = 0
            else:
                sentence_length += len(chunk)
                next_state = GearState(
                    next_state.rotor,
                    next_state.omega,
                    next_state.load,
                    torch.full_like(next_state.sentence_length, sentence_length),
                    next_state.segment_id,
                )
        else:
            sentence_length += len(chunk)

        outputs.extend(
            layer._readout(rotor, omega, load, clutch, previous_rotor).unbind(0)
        )
        rotor_rows.append(rotor.square().sum(dim=-1))
        clutch_rows.append(clutch)
        retention_rows.append(retention)
        state = next_state
        position = stop

    output = torch.stack(outputs, dim=0)
    empty_state = hidden.new_zeros(layer.banks, layer.gears, layer.channels)
    return output, state, {
        "rotor_energy": (
            torch.cat(rotor_rows, dim=0) if rotor_rows else empty_state[None][:0]
        ),
        "clutch": (
            torch.cat(clutch_rows, dim=0) if clutch_rows else empty_state[None][:0]
        ),
        "retention": (
            torch.cat(retention_rows, dim=0) if retention_rows else empty_state[None][:0]
        ),
        "coupling_activity": (
            torch.stack(coupling_rows).mean() if coupling_rows else hidden.new_zeros(())
        ),
        "omega": state.omega,
        "load": state.load,
        "rotor": state.rotor,
    }


def _legacy_chunked_gear_only(
    layer, hidden, token_mask, segment_ids, sentence_end_mask, state=None, ablations=()
):
    """Row loop only, no residual/FFN tail -- directly comparable to
    PureGearLayer._forward_batched's contract (raw gear output, not the
    full layer output)."""
    hidden = hidden.float()
    batch = hidden.shape[0]
    if state is None:
        state = layer.initial_state(batch, hidden.device)
    disabled = frozenset(ablations)
    fixed_omega = (
        not layer.config.learned_angular_velocity
        or "fixed_angular_velocities" in disabled
    )
    settling_enabled = (
        layer.config.boundary_settling and "no_boundary_settling" not in disabled
    )
    cross_bank = (
        layer.config.cross_bank_coupling and "no_cross_bank_coupling" not in disabled
    )
    commuting_only = (
        not layer.config.overlapping_coupling
        or "commuting_coupling_only" in disabled
    )
    use_load = layer.config.use_load_state and "no_load_state" not in disabled

    control_token_mask = token_mask.detach().to(device="cpu", dtype=torch.bool)
    control_segment_ids = segment_ids.detach().to(device="cpu", dtype=torch.long)
    control_sentence_end_mask = sentence_end_mask.detach().to(
        device="cpu", dtype=torch.bool
    )
    delta, clutch_controls, torque, retention_controls = layer._project_token_controls(
        hidden
    )
    rows, states, diagnostics = [], [], []
    for row in range(batch):
        row_output, row_state, row_record = _legacy_chunked_forward_row(
            layer,
            hidden[row],
            control_token_mask[row],
            control_segment_ids[row],
            control_sentence_end_mask[row],
            delta[row],
            clutch_controls[row],
            torque[row],
            retention_controls[row],
            _state_index(state, row),
            fixed_omega=fixed_omega,
            settling_enabled=settling_enabled,
            cross_bank=cross_bank,
            commuting_only=commuting_only,
            use_load=use_load,
        )
        rows.append(row_output)
        states.append(row_state)
        diagnostics.append(row_record)

    gear_output = torch.stack(rows, dim=0)
    next_state = _state_cat(states)
    record = {
        "rotor_energy": torch.cat([item["rotor_energy"] for item in diagnostics], dim=0),
        "clutch": torch.cat([item["clutch"] for item in diagnostics], dim=0),
        "retention": torch.cat([item["retention"] for item in diagnostics], dim=0),
        "coupling_activity": torch.stack(
            [item["coupling_activity"] for item in diagnostics]
        ).mean(),
        "omega": next_state.omega,
        "load": next_state.load,
        "rotor": next_state.rotor,
    }
    return gear_output, next_state, record, fixed_omega, settling_enabled, cross_bank, commuting_only, use_load


def _legacy_chunked_forward(
    layer, hidden, token_mask, segment_ids, sentence_end_mask, state=None, ablations=()
):
    hidden_dtype = hidden.dtype
    disabled = frozenset(ablations)
    (
        gear_output,
        next_state,
        record,
        _,
        _,
        _,
        _,
        _,
    ) = _legacy_chunked_gear_only(
        layer, hidden, token_mask, segment_ids, sentence_end_mask, state, ablations
    )
    hidden = hidden.float()
    residual_scale = layer.residual_floor + (
        1.0 - layer.residual_floor
    ) * torch.sigmoid(layer.gear_residual)
    record["gear_residual_scale"] = residual_scale
    hidden = hidden + residual_scale * layer.dropout(gear_output)
    if layer.use_ffn and "no_local_swiglu" not in disabled:
        value, gate = layer.ffn_in(layer.ffn_norm(hidden)).chunk(2, dim=-1)
        feedforward = layer.ffn_out(F.silu(gate) * value)
        hidden = hidden + torch.tanh(layer.ffn_residual) * layer.dropout(feedforward)
    return hidden.to(hidden_dtype), next_state, record


def _make_chunk_parallel_scenario(case: str):
    torch.manual_seed(11)
    if case == "basic_multi_segment":
        layer = model(layers=1, max_sentence_tokens=20, intra_sentence_clutch_tokens=5).layers[0]
        batch, length = 4, 47
        token_mask = torch.ones(batch, length, dtype=torch.bool)
        token_mask[0, 30:33] = False
        token_mask[3, :4] = False
        segment_ids = torch.zeros(batch, length, dtype=torch.long)
        segment_ids[1, 25:] = 1
        segment_ids[2, 10:] = 1
        segment_ids[2, 35:] = 2
        sentence_end_mask = torch.zeros(batch, length, dtype=torch.bool)
        sentence_end_mask[0, 7] = True
        sentence_end_mask[1, 12] = True
        sentence_end_mask[1, 40] = True
        sentence_end_mask[2, 22] = True
    elif case == "clutch_disabled":
        layer = model(layers=1, max_sentence_tokens=6, intra_sentence_clutch_tokens=0).layers[0]
        batch, length = 3, 29
        token_mask = torch.ones(batch, length, dtype=torch.bool)
        segment_ids = torch.zeros(batch, length, dtype=torch.long)
        sentence_end_mask = torch.zeros(batch, length, dtype=torch.bool)
        sentence_end_mask[0, 4] = True
        sentence_end_mask[1, 9] = True
        sentence_end_mask[1, 18] = True
    elif case == "single_row":
        layer = model(layers=1, max_sentence_tokens=20, intra_sentence_clutch_tokens=5).layers[0]
        batch, length = 1, 31
        token_mask = torch.ones(batch, length, dtype=torch.bool)
        segment_ids = torch.zeros(batch, length, dtype=torch.long)
        sentence_end_mask = torch.zeros(batch, length, dtype=torch.bool)
        sentence_end_mask[0, 14] = True
    elif case == "segment_change_at_start_and_end":
        layer = model(layers=1, max_sentence_tokens=20, intra_sentence_clutch_tokens=5).layers[0]
        batch, length = 2, 25
        token_mask = torch.ones(batch, length, dtype=torch.bool)
        segment_ids = torch.zeros(batch, length, dtype=torch.long)
        segment_ids[0, :] = 1  # differs from the fresh state's initial segment_id=-1 immediately
        segment_ids[1, -1:] = 2  # segment change at the very last position
        sentence_end_mask = torch.zeros(batch, length, dtype=torch.bool)
        sentence_end_mask[0, 10] = True
    elif case == "all_masked_row":
        layer = model(layers=1, max_sentence_tokens=20, intra_sentence_clutch_tokens=5).layers[0]
        batch, length = 3, 18
        token_mask = torch.ones(batch, length, dtype=torch.bool)
        token_mask[1, :] = False
        segment_ids = torch.zeros(batch, length, dtype=torch.long)
        sentence_end_mask = torch.zeros(batch, length, dtype=torch.bool)
        sentence_end_mask[0, 8] = True
    elif case == "interleaved_masking":
        layer = model(layers=1, max_sentence_tokens=9, intra_sentence_clutch_tokens=3).layers[0]
        batch, length = 2, 22
        token_mask = torch.ones(batch, length, dtype=torch.bool)
        token_mask[0, 5] = False
        token_mask[0, 6] = False
        token_mask[1, 10:13] = False
        segment_ids = torch.zeros(batch, length, dtype=torch.long)
        sentence_end_mask = torch.zeros(batch, length, dtype=torch.bool)
        sentence_end_mask[1, 7] = True
    else:
        raise ValueError(case)
    hidden = torch.randn(batch, length, layer.dim, requires_grad=True)
    return layer, hidden, token_mask, segment_ids, sentence_end_mask


@pytest.mark.parametrize(
    "case",
    [
        "basic_multi_segment",
        "clutch_disabled",
        "single_row",
        "segment_change_at_start_and_end",
        "all_masked_row",
        "interleaved_masking",
    ],
)
def test_chunk_parallel_forward_matches_legacy_chunked_loop(case):
    layer, hidden, token_mask, segment_ids, sentence_end_mask = (
        _make_chunk_parallel_scenario(case)
    )
    hidden_ref = hidden.detach().clone().requires_grad_(True)
    batch = hidden.shape[0]

    state = layer.initial_state(batch, hidden.device)
    (
        ref_gear_output,
        ref_state,
        ref_record,
        fixed_omega,
        settling_enabled,
        cross_bank,
        commuting_only,
        use_load,
    ) = _legacy_chunked_gear_only(
        layer, hidden_ref, token_mask, segment_ids, sentence_end_mask, state
    )

    new_gear_output, new_state, new_record = layer._forward_batched(
        hidden.float(),
        token_mask,
        segment_ids,
        sentence_end_mask,
        layer.initial_state(batch, hidden.device),
        fixed_omega=fixed_omega,
        settling_enabled=settling_enabled,
        cross_bank=cross_bank,
        commuting_only=commuting_only,
        use_load=use_load,
    )

    assert torch.allclose(new_gear_output, ref_gear_output, atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        new_record["rotor_energy"], ref_record["rotor_energy"], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(new_record["clutch"], ref_record["clutch"], atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        new_record["retention"], ref_record["retention"], atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        new_record["coupling_activity"], ref_record["coupling_activity"], atol=1e-6
    )
    assert torch.allclose(new_state.rotor, ref_state.rotor, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_state.omega, ref_state.omega, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_state.load, ref_state.load, atol=1e-5, rtol=1e-5)
    assert torch.equal(new_state.sentence_length, ref_state.sentence_length)
    assert torch.equal(new_state.segment_id, ref_state.segment_id)

    new_gear_output.square().sum().backward()
    ref_gear_output.square().sum().backward()
    assert torch.allclose(hidden.grad, hidden_ref.grad, atol=2e-4, rtol=2e-4)


def _legacy_settle(layer, state, *, cross_bank=True, commuting_only=False, use_load=True):
    """Verbatim transcription of the pre-vectorization sequential pair loop.

    Ground truth for test_vectorized_settle_matches_legacy_pair_loop --
    settle() now mixes each disjoint-pair pass in one batched call via
    _mix_gear_pairs instead of looping _mix_gears once per pair.
    """
    rotor = state.rotor
    activity = rotor.new_zeros(())
    for round_index in range(layer.config.settling_rounds):
        gate_round = min(round_index, layer.intra_gate.shape[0] - 1)
        for left in range(0, layer.gears - 1, 2):
            rotor = layer._mix_gears(
                rotor, state.omega, state.load, left, left + 1,
                layer.intra_gate[gate_round, :, left],
            )
            activity = activity + torch.sigmoid(layer.intra_gate[gate_round, :, left]).mean()
        if not commuting_only:
            for left in range(1, layer.gears - 1, 2):
                rotor = layer._mix_gears(
                    rotor, state.omega, state.load, left, left + 1,
                    layer.intra_gate[gate_round, :, left],
                )
                activity = activity + torch.sigmoid(layer.intra_gate[gate_round, :, left]).mean()
        if cross_bank and layer.banks > 1:
            for left in range(layer.banks):
                right = (left + 1) % layer.banks
                rotor = layer._mix_banks(
                    rotor, state.omega, state.load, left, right,
                    layer.cross_gate[gate_round, left],
                )
                activity = activity + torch.sigmoid(layer.cross_gate[gate_round, left]).mean()

    magnitude = rotor.square().sum(dim=-1).clamp_min(1e-8).sqrt()
    log_energy = magnitude.log().clamp(-4.0, 4.0)
    normalized = rotor / magnitude[..., None]
    if use_load and layer.config.use_load_state:
        orientation = normalized[..., 0] - normalized[..., 1]
        load = torch.tanh(
            state.load
            + layer.load_response[..., 0].float() * log_energy
            + layer.load_response[..., 1].float() * orientation
        )
    else:
        load = torch.zeros_like(state.load)
    omega = (
        layer.config.omega_limit
        * torch.tanh(
            state.omega / layer.config.omega_limit
            + layer.omega_response[..., 0].float() * load
            + layer.omega_response[..., 1].float() * log_energy
        )
        if layer.config.learned_angular_velocity
        else state.omega
    )
    count = max(1, layer.config.settling_rounds)
    return (
        GearState(
            normalized
            * magnitude.clamp_max(layer.config.rotor_radius_limit)[..., None],
            omega,
            load,
            torch.zeros_like(state.sentence_length),
            state.segment_id,
        ),
        activity / count,
    )


@pytest.mark.parametrize("gears,banks", [(8, 4), (5, 3), (2, 1)])
def test_vectorized_settle_matches_legacy_pair_loop(gears, banks):
    torch.manual_seed(17)
    layer = model(layers=1, gears_per_bank=gears, num_banks=banks).layers[0]
    batch = 3
    state = layer.initial_state(batch, torch.device("cpu"))
    rotor = (state.rotor + 0.1 * torch.randn_like(state.rotor)).detach().requires_grad_(True)
    rotor_ref = rotor.detach().clone().requires_grad_(True)
    omega = (state.omega + 0.05 * torch.randn_like(state.omega)).detach().requires_grad_(True)
    omega_ref = omega.detach().clone().requires_grad_(True)
    load = (0.1 * torch.randn_like(state.load)).detach().requires_grad_(True)
    load_ref = load.detach().clone().requires_grad_(True)

    new_state, new_activity = layer.settle(
        GearState(rotor, omega, load, state.sentence_length, state.segment_id)
    )
    ref_state, ref_activity = _legacy_settle(
        layer, GearState(rotor_ref, omega_ref, load_ref, state.sentence_length, state.segment_id)
    )

    assert torch.allclose(new_state.rotor, ref_state.rotor, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_state.omega, ref_state.omega, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_state.load, ref_state.load, atol=1e-5, rtol=1e-5)
    assert torch.allclose(new_activity, ref_activity, atol=1e-5, rtol=1e-5)

    new_state.rotor.square().sum().backward()
    ref_state.rotor.square().sum().backward()
    assert torch.allclose(rotor.grad, rotor_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(omega.grad, omega_ref.grad, atol=2e-4, rtol=2e-4)
    assert torch.allclose(load.grad, load_ref.grad, atol=2e-4, rtol=2e-4)


def test_future_tokens_cannot_change_past_logits():
    instance = model().eval()
    tokens = torch.randint(0, 97, (1, 20))
    ends = boundaries(tokens)
    original, _ = instance(tokens, sentence_end_mask=ends)
    changed = tokens.clone()
    changed[:, 14:] = (changed[:, 14:] + 17) % 97
    altered, _ = instance(changed, sentence_end_mask=ends)
    assert torch.allclose(original[:, :14], altered[:, :14], atol=1e-6)


def test_packed_segments_reset_all_gear_state():
    instance = model().eval()
    tokens = torch.randint(0, 97, (1, 18))
    segments = torch.tensor([[0] * 9 + [1] * 9])
    ends = torch.zeros_like(tokens, dtype=torch.bool)
    ends[:, 8] = True
    logits, _ = instance(
        tokens, segment_ids=segments, sentence_end_mask=ends
    )
    changed = tokens.clone()
    changed[:, :9] = (changed[:, :9] + 23) % 97
    changed_logits, _ = instance(
        changed, segment_ids=segments, sentence_end_mask=ends
    )
    assert torch.allclose(logits[:, 9:], changed_logits[:, 9:], atol=1e-6)


def test_order_changes_rotor_knowledge_state():
    instance = model(layers=1).eval()
    tokens = torch.tensor([[2, 3, 5, 7, 11, 13, 17, 19]])
    reverse = tokens.flip(1)
    end = torch.zeros_like(tokens, dtype=torch.bool)
    _, first = instance(tokens, use_cache=True, sentence_end_mask=end)
    _, second = instance(reverse, use_cache=True, sentence_end_mask=end)
    assert not torch.allclose(
        first.layers[0].rotor, second.layers[0].rotor
    )


def test_overlapping_clutches_are_noncommutative():
    layer = model(layers=1).layers[0]
    state = layer.initial_state(1, torch.device("cpu"))
    with torch.no_grad():
        layer.intra_gate.fill_(4.0)
        layer.pair_kernel.fill_(0.2)
    first = layer._mix_gears(
        state.rotor, state.omega, state.load, 0, 1, layer.intra_gate[0, :, 0]
    )
    first = layer._mix_gears(
        first, state.omega, state.load, 1, 2, layer.intra_gate[0, :, 1]
    )
    second = layer._mix_gears(
        state.rotor, state.omega, state.load, 1, 2, layer.intra_gate[0, :, 1]
    )
    second = layer._mix_gears(
        second, state.omega, state.load, 0, 1, layer.intra_gate[0, :, 0]
    )
    assert not torch.allclose(first, second)


def test_gears_are_independent_before_boundary_clutching():
    layer = model(layers=1).layers[0]
    hidden = torch.randn(7, layer.dim)
    base = layer.initial_state(1, hidden.device)
    changed = GearState(
        base.rotor.clone(),
        base.omega.clone(),
        base.load.clone(),
        base.sentence_length.clone(),
        base.segment_id.clone(),
    )
    changed.rotor[:, 0, 0, 0] = torch.tensor([0.0, 1.0])
    original, _, _ = layer._token_dynamics(hidden, base, fixed_omega=False)
    altered, _, _ = layer._token_dynamics(hidden, changed, fixed_omega=False)
    unaffected = torch.ones_like(original, dtype=torch.bool)
    unaffected[:, 0, 0, 0] = False
    assert torch.allclose(original[unaffected], altered[unaffected])
    assert not torch.allclose(original[:, 0, 0, 0], altered[:, 0, 0, 0])


def test_all_trainable_parameters_have_finite_gradients():
    instance = model()
    tokens = torch.randint(0, 97, (3, 18))
    loss = instance.training_step(
        tokens, {"sentence_end_mask": boundaries(tokens)}
    )["total"]
    loss.backward()
    bad = [
        name
        for name, parameter in instance.named_parameters()
        if parameter.requires_grad
        and (
            parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
            or float(parameter.grad.norm()) == 0.0
        )
    ]
    assert bad == []


def test_boundary_detector_handles_abbreviation_decimal_quote_and_cap():
    tokenizer = NumericFallbackTokenizer(256)
    detector = SentenceBoundaryDetector(tokenizer, max_sentence_tokens=5)
    assert not detector.is_boundary(list(b"Dr."))
    assert not detector.is_boundary(list(b"3.14"))
    assert detector.is_boundary(list(b"Done!\""))
    boundary, forced = detector.classify(list(b"abcde"))
    assert boundary and forced
    ids, ends, forced_mask = detector.scan_tokens(list(b"A. B!"))
    assert ids.shape == ends.shape == forced_mask.shape
    assert bool(ends.any())
    _, open_ends, _ = detector.scan_tokens(
        list(b"unfinished prompt"), close_final=False
    )
    assert not bool(open_ends[-1])
    for value in (
        list(b"Dr."),
        list(b"3.14"),
        list(b"Done!\""),
        list(b"x" * 80 + b"."),
    ):
        assert detector.classify_incremental(value) == detector.classify(value)


def _tiny_manifest(tmp_path: Path):
    root = tmp_path / "corpus"
    domain = root / "demo"
    domain.mkdir(parents=True)
    tokenizer_name = "tiny"
    train = np.asarray(
        [90, 65, 46, 70, 91, 90, 66, 33, 71, 91],
        dtype=np.uint16,
    )
    train.tofile(domain / f"train_{tokenizer_name}.bin")
    (domain / f"train_{tokenizer_name}.bin.manifest.json").write_text(
        json.dumps({"dtype": "uint16", "vocab_size": 96})
    )
    torch.save(torch.tensor([90, 20, 91]), domain / f"valid_{tokenizer_name}.pt")
    torch.save(torch.tensor([90, 21, 91]), domain / f"test_{tokenizer_name}.pt")
    index = tmp_path / "index"
    build_document_index(
        root,
        index,
        tokenizer_name=tokenizer_name,
        domains=("demo",),
        bos_id=90,
        eos_id=91,
        sealed_per_mille=0,
    )
    manifest = tmp_path / "manifest"
    build_paired_training_manifest(
        root,
        index,
        manifest,
        tokenizer_name=tokenizer_name,
        rows_by_length={8: 2},
        seed=4,
        domains=("demo",),
        max_sentence_tokens=3,
    )
    return manifest


def test_manifest_freezes_sentence_metadata(tmp_path):
    manifest = _tiny_manifest(tmp_path)
    metadata = json.loads((manifest / "manifest.json").read_text())
    assert metadata["format"] == "lmf_paired_document_windows_v2"
    assert metadata["boundary_detector_hash"]
    for suffix in (
        "sentence_ids",
        "sentence_end_mask",
        "forced_boundary_mask",
    ):
        assert (manifest / f"length_8_{suffix}.npy").exists()
    corpus = PairedDocumentManifestCorpus(str(manifest), wrap=False)
    batch = corpus.sample_batch(1, 8)
    assert batch.metadata["sentence_ids"].shape == batch.tokens.shape
    assert bool(batch.metadata["sentence_end_mask"].any())
    crossing = (
        batch.metadata["segment_ids"][:, 1:]
        != batch.metadata["segment_ids"][:, :-1]
    )
    assert not bool((batch.loss_mask[:, 1:] & crossing).any())
    sampler_before = corpus.sampler_state()
    selected = corpus.batch_from_indices([1, 0], 8)
    assert selected.metadata["manifest_row_ids"].tolist() == [1, 0]
    assert corpus.sampler_state() == sampler_before


def test_trainer_keeps_parameters_and_optimizer_moments_fp32():
    from lmf.data import ProceduralCorpus

    instance = model(vocab_size=97)
    trainer = PureParallelGearTrainer(
        instance,
        ProceduralCorpus(vocab_size=97),
        device="cpu",
        precision="bf16",
        lr=1e-3,
        total_training_tokens=256,
        warmup_tokens=1,
        context_lengths=(16,),
        context_fractions=(1.0,),
    )
    trainer.train_steps(1, 2, 16, log_every=0)
    assert {parameter.dtype for parameter in trainer.raw_model.parameters()} == {
        torch.float32
    }
    moment_dtypes = {
        value.dtype
        for state in trainer.optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point()
    }
    assert moment_dtypes == {torch.float32}
    assert any(group.get("gear_dynamics") for group in trainer.optimizer.param_groups)
    parameter_groups = {
        id(parameter): group
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    assert parameter_groups[
        id(instance.layers[0].angle_projection.weight)
    ]["lr_multiplier"] == 1.0
    assert parameter_groups[
        id(instance.layers[0].base_omega)
    ]["lr_multiplier"] == trainer.dynamics_lr_multiplier


def test_efficiency_probe_prefills_the_requested_context():
    from lmf.models.transformer import CachedTransformerLM, TransformerConfig

    instance = CachedTransformerLM(
        TransformerConfig(vocab_size=97, dim=32, layers=1, heads=4)
    )
    short = throughput(
        instance,
        vocab_size=97,
        seq_len=16,
        device="cpu",
        repeats=1,
    )
    long = throughput(
        instance,
        vocab_size=97,
        seq_len=32,
        device="cpu",
        repeats=1,
    )
    assert long["cache_bytes"] == 2 * short["cache_bytes"]
    assert short["prefill_p50_seconds"] > 0.0
    assert short["incremental_p50_seconds"] > 0.0


def test_legacy_checkpoints_are_explicitly_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "manifest": {"name": "PureParallelPredictiveGearV2"},
            "schema_version": 1,
        },
        path,
    )
    with pytest.raises(RuntimeError, match="intentionally incompatible"):
        load_checkpoint(path, model())


def test_parameter_count_formula_and_matched_baselines():
    proxy = benchmark_configs("proxy", 4109)
    fairness = assert_fair_configs(proxy)
    assert all(
        abs(value) <= 0.005
        for value in fairness["relative_to_transformer"].values()
    )
    assert gear_parameter_count(proxy["gear"]) == sum(
        parameter.numel()
        for parameter in PureParallelGearLM(proxy["gear"]).parameters()
    )


def test_shape_changing_ablations_remove_inactive_parameters():
    no_predictor = model(use_predictor_gear=False)
    assert no_predictor.predictor is None
    no_ffn = model(use_local_swiglu=False)
    assert all(not hasattr(layer, "ffn_in") for layer in no_ffn.layers)
    no_settling = model(boundary_settling=False)
    names = dict(no_settling.named_parameters())
    assert not any(
        key in name
        for name in names
        for key in ("pair_kernel", "cross_kernel", "intra_gate", "cross_gate")
    )


def memory_model(**overrides) -> PureParallelGearLM:
    values = {
        "use_fast_weight_memory": True,
        "fast_weight_banks": 2,
        "fast_weight_key_dim": 4,
        "fast_weight_value_dim": 4,
        "fast_weight_chunk_tokens": 5,
    }
    values.update(overrides)
    return model(**values)


def test_default_config_has_no_fast_weight_memory():
    instance = model()
    assert instance.memory is None
    tokens = torch.randint(0, 97, (2, 9))
    ends = torch.zeros_like(tokens, dtype=torch.bool)
    _, cache = instance(tokens, sentence_end_mask=ends, use_cache=True)
    assert cache.memory is None
    manifest = instance.architecture_manifest()
    assert manifest["invariants"]["token_similarity"] is False
    assert manifest["invariants"]["history_retrieval"] is False
    assert manifest["invariants"]["fast_weight_memory"] is False


def test_fast_weight_memory_streaming_matches_full():
    torch.manual_seed(41)
    instance = memory_model(
        layers=1, max_sentence_tokens=64, intra_sentence_clutch_tokens=4
    ).eval()
    tokens = torch.randint(0, 97, (2, 17))
    ends = torch.zeros_like(tokens, dtype=torch.bool)
    full, _ = instance(tokens, sentence_end_mask=ends)
    cache = None
    pieces = []
    for position in range(tokens.shape[1]):
        logits, cache = instance(
            tokens[:, position : position + 1],
            cache=cache,
            use_cache=True,
            sentence_end_mask=ends[:, position : position + 1],
        )
        pieces.append(logits)
    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=2e-5, rtol=2e-5)


def test_fast_weight_memory_chunked_carry_matches_naive_recurrence():
    """Differential test: the chunked rescale-cumsum-then-carry scan must
    match a naive token-by-token recurrence with no chunking trick at all
    -- the same methodology used to validate the rotor's chunk-parallel
    rewrite earlier, applied to the new numerically-safer scan."""
    torch.manual_seed(7)
    instance = memory_model(layers=1)
    memory = instance.memory
    tokens = torch.randint(0, 97, (2, 23))
    hidden = torch.randn(2, 23, instance.config.dim)
    token_embeddings = instance.token(tokens)
    token_mask = torch.ones(2, 23, dtype=torch.bool)
    segment_ids = torch.zeros(2, 23, dtype=torch.long)
    segment_ids[1, 13:] = 1

    memory_out, logit_bias, gate, next_state, _ = memory(
        hidden, token_embeddings, token_mask, segment_ids, instance.head, None
    )

    key_full = F.normalize(
        memory.key_proj(hidden).reshape(2, 23, memory.banks, memory.key_dim), dim=-1
    )
    query_full = F.normalize(
        memory.query_proj(hidden).reshape(2, 23, memory.banks, memory.key_dim),
        dim=-1,
    )
    value_full = memory.value_down_proj(token_embeddings)
    ref_read = torch.zeros(2, 23, memory.banks, memory.value_dim)
    for row in range(2):
        accumulator = torch.zeros(memory.banks, memory.key_dim, memory.value_dim)
        current_segment = -1
        for position in range(23):
            segment = int(segment_ids[row, position])
            if segment != current_segment:
                accumulator = torch.zeros(memory.banks, memory.key_dim, memory.value_dim)
                current_segment = segment
            outer = (
                key_full[row, position][..., None]
                * value_full[row, position][None, None, :]
            )
            accumulator = memory.decay * accumulator + outer
            ref_read[row, position] = (
                query_full[row, position][..., None] * accumulator
            ).sum(dim=-2)
    ref_flat = ref_read.reshape(2, 23, memory.banks * memory.value_dim)
    ref_memory_out = memory.memory_out_proj(ref_flat)
    ref_logit_bias = instance.head(memory.value_up_proj(ref_flat))
    ref_gate = torch.sigmoid(memory.gate_proj(torch.cat([hidden, ref_flat], dim=-1)))
    assert torch.allclose(memory_out, ref_memory_out, atol=1e-5, rtol=1e-5)
    assert torch.allclose(logit_bias, ref_logit_bias, atol=1e-4, rtol=1e-4)
    assert torch.allclose(gate, ref_gate, atol=1e-5, rtol=1e-5)


def test_fast_weight_memory_embedding_grounding_recovers_recent_token():
    """With the gate forced open and decay near-instant, the direct
    logit-bias path should put its largest mass on the actual vocab id of
    the immediately preceding token -- a direct test that values are
    grounded in token identity (not an arbitrary hidden-state feature).

    Embeddings/projections are overwritten with a deterministic, orthogonal
    construction (rather than relying on random-init embedding statistics)
    so the outcome doesn't depend on chance dot-product collisions among
    97 vocab entries squeezed into a handful of key/value dimensions.
    """
    torch.manual_seed(5)
    instance = memory_model(
        layers=1,
        fast_weight_banks=1,
        fast_weight_key_dim=4,
        fast_weight_value_dim=4,
        fast_weight_decay=0.999,
        fast_weight_chunk_tokens=4,
    )
    memory = instance.memory
    recent_token, other_token = 42, 10
    with torch.no_grad():
        instance.token.weight.zero_()
        instance.token.weight[recent_token, 0] = 1.0
        instance.token.weight[other_token, 1] = 1.0
        memory.key_proj.weight.zero_()
        memory.key_proj.weight[:, : memory.key_dim] = torch.eye(memory.key_dim)
        memory.query_proj.weight.copy_(memory.key_proj.weight)
        memory.value_down_proj.weight.zero_()
        memory.value_down_proj.weight[:, : memory.value_dim] = torch.eye(
            memory.value_dim
        )
        memory.value_up_proj.weight.zero_()
        memory.value_up_proj.weight[: memory.value_dim, :] = torch.eye(
            memory.value_dim
        )

    tokens = torch.tensor([[recent_token]])
    hidden = instance.token(tokens).clone()
    token_embeddings = instance.token(tokens)
    token_mask = torch.ones(1, 1, dtype=torch.bool)
    segment_ids = torch.zeros(1, 1, dtype=torch.long)
    _, logit_bias, _, _, _ = memory(
        hidden, token_embeddings, token_mask, segment_ids, instance.head, None
    )
    assert int(logit_bias[0, 0].argmax()) == recent_token
    assert float(logit_bias[0, 0, other_token].detach()) == pytest.approx(0.0, abs=1e-6)


def test_fast_weight_memory_gate_is_content_dependent():
    """Two inputs differing only in whether a target token was recently
    seen must produce a measurably different gate -- catches a regression
    to a constant scalar gate (the original, weaker sketch this design
    replaced)."""
    torch.manual_seed(11)
    instance = memory_model(layers=1)
    memory = instance.memory
    # gate_proj.weight is zero-initialized so a fresh model starts at a
    # predictable, calibrated constant (copy_gate_target_mean) -- give it
    # a small explicit perturbation to test the *architecture's capacity*
    # for content-dependence, not the specific at-init behavior.
    with torch.no_grad():
        memory.gate_proj.weight.normal_(std=0.05)
    target_token = 17
    batch_size, length = 4, 12
    tokens_with = torch.randint(0, 97, (batch_size, length))
    tokens_with[:, 3] = target_token
    tokens_without = tokens_with.clone()
    tokens_without[:, 3] = (target_token + 1) % 97
    token_mask = torch.ones(batch_size, length, dtype=torch.bool)
    segment_ids = torch.zeros(batch_size, length, dtype=torch.long)
    hidden = torch.randn(batch_size, length, instance.config.dim)

    _, _, gate_with, _, _ = memory(
        hidden,
        instance.token(tokens_with),
        token_mask,
        segment_ids,
        instance.head,
        None,
    )
    _, _, gate_without, _, _ = memory(
        hidden,
        instance.token(tokens_without),
        token_mask,
        segment_ids,
        instance.head,
        None,
    )
    assert not torch.allclose(gate_with, gate_without, atol=1e-6)


def test_fast_weight_memory_direct_logit_path_changes_predictions():
    """Ablation test: with the same trained weights and input, logits with
    the direct logit_bias term zeroed out vs. included must differ
    non-trivially -- proves the direct path isn't a no-op already achieved
    by the residual injection alone."""
    torch.manual_seed(13)
    instance = memory_model(layers=1)
    # gate_proj is zero-init (predictable constant-mean start) and the
    # value projections use small-std random init, so the term's magnitude
    # at a fresh model's typical init is tiny relative to head(hidden) --
    # force the gate open and the value path to a trained-scale magnitude
    # to test the mechanism's actual effect size, not untrained happenstance.
    with torch.no_grad():
        instance.memory.gate_proj.bias.fill_(10.0)
        instance.memory.value_up_proj.weight.normal_(std=1.0)
    tokens = torch.randint(0, 97, (2, 15))
    ends = torch.zeros_like(tokens, dtype=torch.bool)
    hidden, _, _, memory_extras = instance._forward_hidden(
        tokens, sentence_end_mask=ends
    )
    logits_without_bias = instance.head(hidden)
    logits_with_bias = logits_without_bias + (
        memory_extras["gate"] * memory_extras["logit_bias"]
    )
    assert not torch.allclose(logits_without_bias, logits_with_bias, atol=1e-6)
    assert (logits_with_bias - logits_without_bias).abs().mean() > 1e-4


def test_fast_weight_memory_gate_balance_regularizer_pulls_toward_target():
    """Mirrors how clutch_balance was validated against the dead-bank
    crash: on an adversarial setup that pushes the gate toward saturation,
    the run with copy_gate_balance_weight active should keep the gate
    closer to copy_gate_target_mean than the run with it disabled."""
    torch.manual_seed(17)

    def biased_gate_mean(balance_weight: float) -> float:
        instance = memory_model(
            layers=1, copy_gate_balance_weight=balance_weight, copy_gate_target_mean=0.10
        )
        optimizer = torch.optim.SGD(instance.parameters(), lr=0.5)
        tokens = torch.randint(0, 97, (4, 20))
        for _ in range(20):
            optimizer.zero_grad()
            metrics = instance.training_step(tokens)
            # Adversarial: directly reward a wide-open gate via the
            # language-modeling-free term, independent of whether copying
            # actually helps -- isolates the regularizer's effect.
            adversarial = metrics["total"] - 5.0 * metrics["copy_gate_mean"]
            adversarial.backward()
            optimizer.step()
        with torch.no_grad():
            final = instance.training_step(tokens)
        return float(final["copy_gate_mean"])

    unregularized = biased_gate_mean(0.0)
    regularized = biased_gate_mean(0.02)
    target = 0.10
    assert abs(regularized - target) < abs(unregularized - target)


def test_fast_weight_memory_long_segment_stays_finite():
    """Numerical safety check: the bug class this chunked design avoids
    (rescale-cumsum overflow over a whole long segment) would show up as
    non-finite values in the memory matrix or its gradient on a long,
    single-segment sequence."""
    torch.manual_seed(19)
    instance = memory_model(layers=1, max_sentence_tokens=2048, fast_weight_chunk_tokens=128)
    tokens = torch.randint(0, 97, (1, 2048))
    ends = torch.zeros_like(tokens, dtype=torch.bool)
    metrics = instance.training_step(tokens)
    assert torch.isfinite(metrics["total"])
    metrics["total"].backward()
    for name, parameter in instance.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
