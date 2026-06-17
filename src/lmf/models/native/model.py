"""Runnable baselines for MECM, MCPM, and MRWT.

These are intentionally conservative first implementations. They implement the
training/evaluation/prediction contracts and the non-degradation gates from the
architecture document, while leaving the heavier research mechanisms behind
zero-gated residual paths.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.registry import MODELS
from .components import (
    ActiveCoverRouter,
    BudgetController,
    CausalConvBlock,
    CausalResidualAdapter,
    ContractVerifier,
    ExecutionWorkbench,
    ExecutionTraceAdapter,
    FractalCausalFieldBlock,
    HierarchicalDraftHead,
    MGCFConfig,
    MRWTConfig,
    MultiScaleSpanAtlas,
    NativeLMConfig,
    ProgramController,
    SparseReasoningMesh,
    ZeroGatedCausalSummary,
    init_embedding,
    lm_cross_entropy,
    parameter_count,
    positional_ids,
    sample_from_logits,
    transformer_anchor,
)


class NativeCausalLM(nn.Module):
    """Non-Transformer causal baseline used for MECM and MCPM."""

    family_name = "native_causal"

    def __init__(self, config: NativeLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.position = nn.Embedding(config.max_seq_len, config.dim)
        self.blocks = nn.ModuleList(
            [
                CausalConvBlock(config.dim, config.kernel_size, config.dropout)
                for _ in range(config.layers)
            ]
        )
        self.mesh = ZeroGatedCausalSummary(config.dim) if config.mesh_residual else None
        if config.full_architecture:
            self.span_atlas = MultiScaleSpanAtlas(config.dim, tuple(config.atlas_kernel_sizes))
            self.active_cover = ActiveCoverRouter(config.dim, tuple(config.atlas_kernel_sizes))
            self.reasoning_mesh = SparseReasoningMesh(
                config.dim, config.mesh_layers, max(config.kernel_size, max(config.atlas_kernel_sizes))
            )
            self.draft_tree = HierarchicalDraftHead(
                config.dim,
                config.vocab_size,
                tuple(config.draft_horizons),
                stride=config.draft_aux_stride,
            )
        else:
            self.span_atlas = None
            self.active_cover = None
            self.reasoning_mesh = None
            self.draft_tree = None
        self.execution = (
            ExecutionTraceAdapter(config.dim, config.vocab_size)
            if config.execution_residual
            else None
        )
        if config.full_architecture and config.execution_residual:
            self.program_controller = ProgramController(config.dim)
            self.execution_workbench = ExecutionWorkbench(
                config.dim, max(1, config.mesh_layers), config.kernel_size
            )
            self.contract_verifier = ContractVerifier(config.dim)
        else:
            self.program_controller = None
            self.execution_workbench = None
            self.contract_verifier = None
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.gear_head = (
            nn.Linear(config.dim, config.hierarchy_gears, bias=False)
            if config.gear_aware_output or config.gear_aux_weight > 0.0
            else None
        )
        if self.gear_head is not None:
            self.register_buffer(
                "_token_gears",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_token_to_local",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_gear_active",
                torch.zeros(config.hierarchy_gears, dtype=torch.bool),
            )
        init_embedding(self.token)
        if self.gear_head is not None:
            nn.init.normal_(self.gear_head.weight, mean=0.0, std=0.02)

    def _forward_hidden(self, ids: torch.Tensor) -> torch.Tensor:
        pos = positional_ids(ids.shape[1], self.config.max_seq_len, ids.device)
        hidden = self.token(ids) + self.position(pos)[None]
        for block in self.blocks:
            hidden = block(hidden)
        if self.mesh is not None:
            hidden = self.mesh(hidden)
        if self.span_atlas is not None:
            hidden = self.span_atlas(hidden)
        if self.active_cover is not None:
            hidden = self.active_cover(hidden)
        if self.reasoning_mesh is not None:
            hidden = self.reasoning_mesh(hidden)
        if self.execution is not None:
            hidden = self.execution(hidden, ids)
        if self.program_controller is not None:
            hidden = self.program_controller(hidden)
        if self.execution_workbench is not None:
            hidden = self.execution_workbench(hidden)
        if self.contract_verifier is not None:
            hidden = self.contract_verifier(hidden)
        return self.norm(hidden)

    def _logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.config.gear_aware_output:
            if self.config.gear_output_mode == "bias":
                return self._gear_biased_scores(hidden)
            return self._hierarchical_scores(hidden)
        return self.head(hidden)

    def configure_token_hierarchy(
        self,
        gear_count: int,
        token_gears: list[int],
        token_children: list[list[int]] | None = None,
        token_bytes: list[list[int]] | None = None,
    ) -> None:
        """Install MultiGear token->gear metadata for gear-aware output."""
        if self.gear_head is None:
            return
        if gear_count > self.config.hierarchy_gears:
            raise ValueError(
                f"tokenizer needs {gear_count} gears, model supports "
                f"{self.config.hierarchy_gears}"
            )
        if len(token_gears) != self.config.vocab_size:
            raise ValueError("token_gears length must equal model vocabulary size")
        if hasattr(self, "_token_children"):
            if token_children is None:
                raise ValueError("token_children are required by this model")
            if len(token_children) != self.config.vocab_size:
                raise ValueError("token_children length must equal model vocabulary size")
        if hasattr(self, "_token_byte_lengths"):
            if token_bytes is None:
                raise ValueError("token_bytes are required by this model")
            if len(token_bytes) != self.config.vocab_size:
                raise ValueError("token_bytes length must equal model vocabulary size")
        gears = torch.tensor(token_gears, dtype=torch.long, device=self._token_gears.device)
        if bool(((gears < 0) | (gears >= gear_count)).any()):
            raise ValueError("token gear outside declared gear_count")
        if hasattr(self, "_token_children"):
            children = torch.tensor(
                token_children, dtype=torch.long, device=self._token_children.device
            )
            if bool(((children < -1) | (children >= self.config.vocab_size)).any()):
                raise ValueError("token child outside vocabulary")
            self._token_children.copy_(children)
        if hasattr(self, "_token_byte_lengths"):
            lengths = torch.tensor(
                [
                    min(len(values), getattr(self.config, "max_token_bytes", 64))
                    for values in token_bytes
                ],
                dtype=torch.long,
                device=self._token_byte_lengths.device,
            )
            self._token_byte_lengths.copy_(lengths)
        local = torch.full_like(self._token_to_local, -1)
        active = torch.zeros_like(self._gear_active)
        for gear in range(gear_count):
            token_ids = torch.nonzero(gears == gear, as_tuple=False).flatten()
            active[gear] = bool(len(token_ids))
            local[token_ids] = torch.arange(len(token_ids), device=local.device)
        self._token_gears.copy_(gears)
        self._token_to_local.copy_(local)
        self._gear_active.copy_(active)

    def _require_token_hierarchy(self) -> None:
        if self.gear_head is None:
            raise RuntimeError("gear-aware output is not enabled")
        if not bool(self._gear_active.any()):
            raise RuntimeError(
                "token hierarchy is required; build with a MultiGear tokenizer "
                "or call configure_token_hierarchy()"
            )

    def _gear_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        self._require_token_hierarchy()
        logits = self.gear_head(hidden)
        return logits.masked_fill(~self._gear_active, float("-inf"))

    def _hierarchical_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return full-vocabulary log scores from gear + within-gear factors."""
        self._require_token_hierarchy()
        token_scores = self.head(hidden)
        gear_log_probs = self._gear_logits(hidden).log_softmax(dim=-1)
        normalizers = []
        for gear in range(self.config.hierarchy_gears):
            token_ids = torch.nonzero(self._token_gears == gear, as_tuple=False).flatten()
            if len(token_ids):
                normalizers.append(
                    token_scores.index_select(-1, token_ids).logsumexp(dim=-1)
                )
            else:
                normalizers.append(torch.zeros_like(token_scores[..., 0]))
        within_gear_normalizers = torch.stack(normalizers, dim=-1)
        return (
            token_scores
            - within_gear_normalizers.index_select(-1, self._token_gears)
            + gear_log_probs.index_select(-1, self._token_gears)
        )

    def _gear_biased_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Cheaper gear-aware scores: token logits plus token-gear bias."""
        self._require_token_hierarchy()
        return self.head(hidden) + self._gear_logits(hidden).index_select(-1, self._token_gears)

    @staticmethod
    def _valid_next(tokens: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if meta.get("loss_mask") is not None:
            valid = valid & meta["loss_mask"][:, 1:].bool()
        if meta.get("attention_mask") is not None:
            valid = valid & meta["attention_mask"][:, 1:].bool()
        return valid

    def _hierarchical_language_modeling_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._require_token_hierarchy()
        target_gears = self._token_gears[targets]
        gear_logits = self._gear_logits(hidden)
        gear_losses = F.cross_entropy(
            gear_logits.reshape(-1, gear_logits.shape[-1]),
            target_gears.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid_float = valid.to(gear_losses.dtype)
        count = valid_float.sum().clamp_min(1)
        gear_loss = (gear_losses * valid_float).sum() / count

        token_loss_sum = hidden.sum() * 0.0
        for gear in range(self.config.hierarchy_gears):
            positions = valid & (target_gears == gear)
            if not bool(positions.any()):
                continue
            token_ids = torch.nonzero(self._token_gears == gear, as_tuple=False).flatten()
            local_logits = F.linear(hidden[positions], self.token.weight[token_ids])
            local_targets = self._token_to_local[targets[positions]]
            token_loss_sum = token_loss_sum + F.cross_entropy(
                local_logits, local_targets, reduction="sum"
            )
        token_loss = token_loss_sum / count
        return gear_loss + token_loss, gear_loss, token_loss

    def _gear_auxiliary_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        self._require_token_hierarchy()
        target_gears = self._token_gears[targets]
        gear_logits = self._gear_logits(hidden)
        gear_losses = F.cross_entropy(
            gear_logits.reshape(-1, gear_logits.shape[-1]),
            target_gears.reshape(-1),
            reduction="none",
        ).reshape_as(targets)
        valid_float = valid.to(gear_losses.dtype)
        return (gear_losses * valid_float).sum() / valid_float.sum().clamp_min(1)

    def forward(self, ids: torch.Tensor, attention_mask=None, **_: Any):
        # attention_mask is accepted for trainer/evaluator compatibility. These
        # non-Transformer baselines do not attend to pad tokens; loss masking
        # still prevents pad positions from contributing to training/eval loss.
        hidden = self._forward_hidden(ids)
        return self._logits_from_hidden(hidden), None

    def training_step(
        self,
        tokens: torch.Tensor,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        hidden = self._forward_hidden(tokens)
        valid_next = self._valid_next(tokens, meta)
        prediction_hidden = hidden[:, :-1]
        targets = tokens[:, 1:]
        if (
            self.config.gear_aware_output
            and self.config.gear_output_mode == "factorized"
        ):
            language_modeling, gear_prediction, within_gear = (
                self._hierarchical_language_modeling_loss(
                    prediction_hidden, targets, valid_next
                )
            )
        else:
            logits = self._logits_from_hidden(hidden)
            language_modeling = lm_cross_entropy(
                logits,
                tokens,
                loss_mask=meta.get("loss_mask"),
                attention_mask=meta.get("attention_mask"),
            )
            gear_prediction = None
            within_gear = None
        scale = (loss_term_scales or {}).get("language_modeling", 1.0)
        total = scale * language_modeling
        result = {"language_modeling": language_modeling}
        if gear_prediction is not None:
            result["gear_prediction"] = gear_prediction
            result["within_gear"] = within_gear
        if self.config.gear_aux_weight > 0.0:
            gear_aux = self._gear_auxiliary_loss(prediction_hidden, targets, valid_next)
            total = total + (
                self.config.gear_aux_weight
                * (loss_term_scales or {}).get("gear_aux", 1.0)
                * gear_aux
            )
            result["gear_aux"] = gear_aux
        composition_aux_weight = getattr(self.config, "composition_aux_weight", 0.0)
        if composition_aux_weight > 0.0 and hasattr(self, "_composition_auxiliary_loss"):
            composition_aux = self._composition_auxiliary_loss(
                prediction_hidden, targets, valid_next
            )
            total = total + (
                composition_aux_weight
                * (loss_term_scales or {}).get("composition_aux", 1.0)
                * composition_aux
            )
            result["composition_aux"] = composition_aux
        if self.active_cover is not None and self.config.route_aux_weight > 0.0:
            route_loss = self.active_cover.last_balance_loss
            if route_loss is not None:
                total = total + (
                    self.config.route_aux_weight
                    * (loss_term_scales or {}).get("route_balance", 1.0)
                    * route_loss
                )
                result["route_balance"] = route_loss
        if self.draft_tree is not None and self.config.draft_aux_weight > 0.0:
            draft_loss = self.draft_tree.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.draft_aux_weight
                * (loss_term_scales or {}).get("draft_tree", 1.0)
                * draft_loss
            )
            result["draft_tree"] = draft_loss
        if self.program_controller is not None and self.config.program_aux_weight > 0.0:
            program_loss = self.program_controller.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.program_aux_weight
                * (loss_term_scales or {}).get("program_controller", 1.0)
                * program_loss
            )
            result["program_controller"] = program_loss
        if self.contract_verifier is not None and self.config.verifier_aux_weight > 0.0:
            verifier_loss = self.contract_verifier.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.verifier_aux_weight
                * (loss_term_scales or {}).get("contract_verifier", 1.0)
                * verifier_loss
            )
            result["contract_verifier"] = verifier_loss
        result["total"] = total
        return result

    def _sample_hierarchical_token(self, hidden: torch.Tensor, sampling_config=None) -> torch.Tensor:
        """Sample gear first, then sample only from that gear's token subset."""
        gears = sample_from_logits(self._gear_logits(hidden), sampling_config).squeeze(-1)
        output = torch.empty((hidden.shape[0], 1), dtype=torch.long, device=hidden.device)
        for gear in torch.unique(gears).tolist():
            rows = torch.nonzero(gears == gear, as_tuple=False).flatten()
            token_ids = torch.nonzero(self._token_gears == gear, as_tuple=False).flatten()
            local_logits = F.linear(hidden[rows], self.token.weight[token_ids])
            local_choice = sample_from_logits(local_logits, sampling_config).squeeze(-1)
            output[rows, 0] = token_ids[local_choice]
        return output

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, sampling_config=None):
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must be a rank-2 tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        sequence = prompt_tokens
        out = []
        for _ in range(max_new_tokens):
            if (
                self.config.gear_aware_output
                and self.config.gear_output_mode == "factorized"
            ):
                hidden = self._forward_hidden(sequence)
                token = self._sample_hierarchical_token(hidden[:, -1], sampling_config)
            else:
                logits, _ = self(sequence)
                token = sample_from_logits(logits[:, -1], sampling_config)
            out.append(token)
            sequence = torch.cat([sequence, token], dim=1)
        if not out:
            return torch.empty(
                prompt_tokens.shape[0], 0, dtype=torch.long, device=prompt_tokens.device
            )
        return torch.cat(out, dim=1)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "family": self.family_name,
            "config": self.config.to_dict(),
            "parameters": {"total": parameter_count(self)},
        }


class MultiGearElasticCausalMeshLM(NativeCausalLM):
    """MECM baseline: causal long-convolution trunk plus zero-gated mesh summary."""

    family_name = "mecm"


class MultiGearConstructiveProgramMachineLM(NativeCausalLM):
    """MCPM baseline: surface model plus zero-gated deterministic trace adapter."""

    family_name = "mcpm"


class MultiGearFractalCausalFieldLM(NativeCausalLM):
    """MGCF: non-attention, non-Mamba MultiGear-native causal field model."""

    family_name = "mgcf"

    def __init__(self, config: MGCFConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.dim)
        self.position = nn.Embedding(config.max_seq_len, config.dim)
        self.gear_embedding = (
            nn.Embedding(config.hierarchy_gears, config.dim)
            if config.input_gear_embedding
            else None
        )
        self.byte_length_embedding = (
            nn.Embedding(config.max_token_bytes + 1, config.dim)
            if config.byte_length_embedding
            else None
        )
        self.composition_proj = (
            nn.Linear(2 * config.dim, config.dim, bias=False)
            if config.hierarchy_composition
            else None
        )
        self.composition_gate = (
            nn.Parameter(torch.tensor(float(config.composition_gate_init)))
            if config.hierarchy_composition
            else None
        )
        self.composition_slots = (
            nn.Parameter(torch.empty(2, config.dim))
            if config.composition_aux_weight > 0.0
            else None
        )
        self.blocks = nn.ModuleList(
            [
                FractalCausalFieldBlock(
                    config.dim,
                    config.kernel_size,
                    tuple(config.dilations),
                    tuple(config.memory_scales),
                    config.dropout,
                    memory_type=config.memory_type,
                    memory_gate_init=config.memory_gate_init,
                )
                for _ in range(config.layers)
            ]
        )
        self.mesh = None
        self.span_atlas = None
        self.active_cover = None
        self.reasoning_mesh = None
        self.execution = None
        self.program_controller = None
        self.execution_workbench = None
        self.contract_verifier = None
        self.draft_tree = None
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.head.weight = self.token.weight
        self.gear_head = (
            nn.Linear(config.dim, config.hierarchy_gears, bias=False)
            if (
                config.gear_aware_output
                or config.gear_aux_weight > 0.0
                or config.input_gear_embedding
            )
            else None
        )
        if self.gear_head is not None:
            self.register_buffer(
                "_token_gears",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_token_to_local",
                torch.full((config.vocab_size,), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_gear_active",
                torch.zeros(config.hierarchy_gears, dtype=torch.bool),
            )
        if (
            config.hierarchy_composition
            or config.byte_length_embedding
            or config.composition_aux_weight > 0.0
        ):
            self.register_buffer(
                "_token_children",
                torch.full((config.vocab_size, 2), -1, dtype=torch.long),
            )
            self.register_buffer(
                "_token_byte_lengths",
                torch.zeros(config.vocab_size, dtype=torch.long),
            )
        init_embedding(self.token)
        if self.gear_embedding is not None:
            init_embedding(self.gear_embedding)
        if self.byte_length_embedding is not None:
            init_embedding(self.byte_length_embedding)
        if self.gear_head is not None:
            nn.init.normal_(self.gear_head.weight, mean=0.0, std=0.02)
        if self.composition_slots is not None:
            nn.init.normal_(self.composition_slots, mean=0.0, std=0.02)

    def _hierarchy_input_residual(self, ids: torch.Tensor) -> torch.Tensor:
        self._require_token_hierarchy()
        residual = None
        if self.composition_proj is not None:
            children = self._token_children[ids]
            child_mask = children.ge(0).unsqueeze(-1)
            safe_children = children.clamp_min(0)
            child_embeddings = self.token(safe_children) * child_mask.to(self.token.weight.dtype)
            composed = self.composition_proj(child_embeddings.flatten(start_dim=-2))
            parent_mask = child_mask.any(dim=-2).to(composed.dtype)
            composed = self.composition_gate * composed * parent_mask
            residual = composed if residual is None else residual + composed
        if self.byte_length_embedding is not None:
            lengths = self._token_byte_lengths[ids].clamp_max(self.config.max_token_bytes)
            length_state = self.byte_length_embedding(lengths)
            residual = length_state if residual is None else residual + length_state
        if residual is None:
            return torch.zeros(
                *ids.shape,
                self.config.dim,
                dtype=self.token.weight.dtype,
                device=ids.device,
            )
        return residual

    def _forward_hidden(self, ids: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        pos = (torch.arange(ids.shape[1], device=ids.device) + int(position_offset)) % self.config.max_seq_len
        hidden = self.token(ids) + self.position(pos)[None]
        if self.gear_embedding is not None:
            self._require_token_hierarchy()
            gear_ids = self._token_gears[ids].clamp_min(0)
            hidden = hidden + self.gear_embedding(gear_ids)
        if self.composition_proj is not None or self.byte_length_embedding is not None:
            hidden = hidden + self._hierarchy_input_residual(ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)

    def _composition_auxiliary_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        self._require_token_hierarchy()
        if self.composition_slots is None:
            return hidden.sum() * 0.0
        children = self._token_children[targets]
        total = hidden.sum() * 0.0
        terms = 0
        for slot in range(2):
            child_targets = children[..., slot]
            mask = valid & child_targets.ge(0)
            if not bool(mask.any()):
                continue
            slot_hidden = hidden + self.composition_slots[slot]
            logits = F.linear(slot_hidden[mask], self.token.weight)
            total = total + F.cross_entropy(logits, child_targets[mask], reduction="mean")
            terms += 1
        return total / max(1, terms)

    def _generation_context(self) -> int:
        branch_reach = (self.config.kernel_size - 1) * max(self.config.dilations)
        memory_reach = max(self.config.memory_scales, default=1) - 1
        per_layer = max(branch_reach, memory_reach)
        return min(self.config.max_seq_len, 1 + self.config.layers * per_layer)

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, sampling_config=None):
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must be a rank-2 tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        sequence = prompt_tokens
        out = []
        context = self._generation_context()
        for _ in range(max_new_tokens):
            window = sequence[:, -context:]
            offset = sequence.shape[1] - window.shape[1]
            if (
                self.config.gear_aware_output
                and self.config.gear_output_mode == "factorized"
            ):
                hidden = self._forward_hidden(window, position_offset=offset)
                token = self._sample_hierarchical_token(hidden[:, -1], sampling_config)
            else:
                hidden = self._forward_hidden(window, position_offset=offset)
                token = sample_from_logits(
                    self._logits_from_hidden(hidden)[:, -1], sampling_config
                )
            out.append(token)
            sequence = torch.cat([sequence, token], dim=1)
        if not out:
            return torch.empty(
                prompt_tokens.shape[0], 0, dtype=torch.long, device=prompt_tokens.device
            )
        return torch.cat(out, dim=1)


class MultiGearResidualWorkbenchTransformerLM(nn.Module):
    """MRWT baseline with exact anchor fallback and zero-gated residual adapters."""

    family_name = "mrwt"

    def __init__(self, config: MRWTConfig) -> None:
        super().__init__()
        self.config = config
        self.anchor = transformer_anchor(config)
        if config.full_architecture:
            self.atlas = (
                MultiScaleSpanAtlas(config.dim, tuple(config.atlas_kernel_sizes))
                if config.use_atlas
                else None
            )
            self.budget_controller = BudgetController(config.dim)
            self.workbench_rounds = (
                nn.ModuleList(
                    [
                        CausalResidualAdapter(config.dim, config.workbench_kernel_size)
                        for _ in range(config.workbench_rounds)
                    ]
                )
                if config.use_workbench
                else nn.ModuleList()
            )
            self.draft_tree = HierarchicalDraftHead(
                config.dim,
                config.vocab_size,
                tuple(config.draft_horizons),
                stride=config.draft_aux_stride,
            )
            self.workbench = None
        else:
            self.atlas = (
                CausalResidualAdapter(config.dim, config.atlas_kernel_size)
                if config.use_atlas
                else None
            )
            self.workbench = (
                CausalResidualAdapter(config.dim, config.workbench_kernel_size)
                if config.use_workbench
                else None
            )
            self.budget_controller = None
            self.workbench_rounds = nn.ModuleList()
            self.draft_tree = None

    def _forward_hidden(self, ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        hidden, _ = self.anchor._forward_hidden(ids, attention_mask=attention_mask)
        if self.atlas is not None:
            hidden = self.atlas(hidden)
        if self.budget_controller is not None:
            hidden = self.budget_controller(hidden)
        if self.workbench is not None:
            hidden = self.workbench(hidden)
        for workbench_round in self.workbench_rounds:
            hidden = workbench_round(hidden)
        return hidden

    def _logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.anchor._output_scores(hidden)

    @staticmethod
    def _valid_next(tokens: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
        valid = torch.ones_like(tokens[:, 1:], dtype=torch.bool)
        if meta.get("loss_mask") is not None:
            valid = valid & meta["loss_mask"][:, 1:].bool()
        if meta.get("attention_mask") is not None:
            valid = valid & meta["attention_mask"][:, 1:].bool()
        return valid

    def forward(self, ids: torch.Tensor, attention_mask=None, **_: Any):
        hidden = self._forward_hidden(ids, attention_mask=attention_mask)
        return self._logits_from_hidden(hidden), None

    def anchor_logits(self, ids: torch.Tensor, attention_mask=None) -> torch.Tensor:
        logits, _ = self.anchor(ids, attention_mask=attention_mask)
        return logits

    def _residual_paths_disabled(self) -> bool:
        for name, parameter in self.named_parameters():
            if name.startswith("anchor."):
                continue
            if name.endswith("gate") and bool((parameter.detach().abs() != 0).any()):
                return False
        return True

    def training_step(
        self,
        tokens: torch.Tensor,
        task_metadata: dict[str, Any] | None = None,
        loss_term_scales: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        meta = task_metadata or {}
        hidden = self._forward_hidden(tokens, attention_mask=meta.get("attention_mask"))
        logits = self._logits_from_hidden(hidden)
        language_modeling = lm_cross_entropy(
            logits,
            tokens,
            loss_mask=meta.get("loss_mask"),
            attention_mask=meta.get("attention_mask"),
        )
        scale = (loss_term_scales or {}).get("language_modeling", 1.0)
        total = scale * language_modeling
        result = {"language_modeling": language_modeling}
        valid_next = self._valid_next(tokens, meta)
        if self.budget_controller is not None and self.config.budget_aux_weight > 0.0:
            budget_loss = self.budget_controller.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.budget_aux_weight
                * (loss_term_scales or {}).get("budget_controller", 1.0)
                * budget_loss
            )
            result["budget_controller"] = budget_loss
        if self.draft_tree is not None and self.config.draft_aux_weight > 0.0:
            draft_loss = self.draft_tree.loss(hidden, tokens, valid_next)
            total = total + (
                self.config.draft_aux_weight
                * (loss_term_scales or {}).get("draft_tree", 1.0)
                * draft_loss
            )
            result["draft_tree"] = draft_loss
        result["total"] = total
        return result

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int, sampling_config=None):
        # Full-prefix decoding keeps residual atlas/workbench semantics exact.
        if prompt_tokens.ndim != 2:
            raise ValueError("prompt_tokens must be a rank-2 tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return torch.empty(
                prompt_tokens.shape[0], 0, dtype=torch.long, device=prompt_tokens.device
            )
        if self._residual_paths_disabled():
            return self.anchor.generate(prompt_tokens, max_new_tokens, sampling_config)
        sequence = prompt_tokens
        out = []
        for _ in range(max_new_tokens):
            logits, _ = self(sequence)
            token = sample_from_logits(logits[:, -1], sampling_config)
            out.append(token)
            sequence = torch.cat([sequence, token], dim=1)
        if not out:
            return torch.empty(
                prompt_tokens.shape[0], 0, dtype=torch.long, device=prompt_tokens.device
            )
        return torch.cat(out, dim=1)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "name": type(self).__name__,
            "family": self.family_name,
            "config": self.config.to_dict(),
            "parameters": {"total": parameter_count(self)},
        }


@MODELS.register("mecm")
def build_mecm(model_cfg: dict, vocab_size: int | None = None) -> MultiGearElasticCausalMeshLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    cfg["mesh_residual"] = True
    cfg["execution_residual"] = False
    return MultiGearElasticCausalMeshLM(NativeLMConfig(**cfg))


@MODELS.register("mcpm")
def build_mcpm(
    model_cfg: dict, vocab_size: int | None = None
) -> MultiGearConstructiveProgramMachineLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    cfg["mesh_residual"] = True
    cfg["execution_residual"] = True
    return MultiGearConstructiveProgramMachineLM(NativeLMConfig(**cfg))


@MODELS.register("mgcf")
def build_mgcf(model_cfg: dict, vocab_size: int | None = None) -> MultiGearFractalCausalFieldLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return MultiGearFractalCausalFieldLM(MGCFConfig(**cfg))


@MODELS.register("mrwt")
def build_mrwt(
    model_cfg: dict, vocab_size: int | None = None
) -> MultiGearResidualWorkbenchTransformerLM:
    cfg = dict(model_cfg)
    if vocab_size is not None:
        cfg["vocab_size"] = vocab_size
    return MultiGearResidualWorkbenchTransformerLM(MRWTConfig(**cfg))
