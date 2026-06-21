"""RHCA v4 configuration.

This is a deliberately slimmed-down descendant of the v3 config. The
architecture review found that the resonance-routing and order-aware-memory
machinery had no measurable effect (RFK17/RFK18 inside the noise floor), so those
levers are removed entirely rather than carried as dead config. The derived plan
state and learned energy head were also removed because they duplicated memory /
token-confidence signals while materially increasing training cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RHCAConfig:
    vocab_size: int

    # --- capacity ---
    field_dim: int = 1024            # D — review Q2.4: depth beats width; keep at 1024
    latent_dim: int = 512            # r
    codebook: str = "lowrank"        # "lowrank" (review Q2.1) or "geometric" (flat tied)
    codebook_factor_dim: int = 256   # e — V*e + e*D ≈ 8.8M vs 54.6M flat

    # --- rolling state geometry ---
    frontier_size: int = 32          # H — review Q1: keep settle cheap
    max_commit: int = 8
    memory_slots: int = 128
    memory_read_top_k: int = 16
    memory_write_top_k: int = 8      # tokens write to this many slots (sparse)
    memory_write_temperature: float = 3.0
    tail_size: int = 512             # review Q6: full-seq exact recall, SDPA-backed
    local_kernel_size: int = 7
    # Rotary relative position encoding on the tail-attention path (query phase =
    # frontier draft offset, key phase = tail-slot distance into the past).
    # Without it, tail attention is pure content addressing, which cannot reliably
    # learn a fixed-distance copy task (e.g. "the token N back") — confirmed via
    # direct ablation: the model failed to learn the procedural corpus's echo task
    # even after thousands of steps. Ablatable like the codebook strategy switch.
    tail_rope: bool = True
    # Hypotheses are reserved for future adaptive-expansion work; the current LM
    # decoder runs single-hypothesis (review Q5 + external finding 3 — keeping
    # width>1 only created untrained, dead parameter rows). >1 is allowed but unused.
    max_hypotheses: int = 1

    # --- settle (unshared deep macro steps — review Q2.2) ---
    ssm_macro_steps: int = 4         # K UNSHARED macro steps
    ssm_scan_steps: int = 12         # L parallel scan steps per macro step

    # --- entropy-based commit mechanism ---
    commit_entropy_threshold: float = 0.80
    routing_balance_weight: float = 0.05

    special_token_ids: dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size, "field_dim": self.field_dim,
            "latent_dim": self.latent_dim, "frontier_size": self.frontier_size,
            "max_commit": self.max_commit, "memory_slots": self.memory_slots,
            "memory_read_top_k": self.memory_read_top_k,
            "max_hypotheses": self.max_hypotheses, "tail_size": self.tail_size,
            "ssm_macro_steps": self.ssm_macro_steps, "ssm_scan_steps": self.ssm_scan_steps,
        }
        bad = [k for k, v in positive.items() if v <= 0]
        if bad:
            raise ValueError(f"config values must be positive: {bad}")
        if self.max_commit > self.frontier_size:
            raise ValueError("max_commit cannot exceed frontier_size")
        if self.memory_read_top_k > self.memory_slots:
            raise ValueError("memory_read_top_k cannot exceed memory_slots")
        if self.memory_write_top_k > self.memory_slots:
            raise ValueError("memory_write_top_k cannot exceed memory_slots")
        if self.codebook not in {"lowrank", "geometric"}:
            raise ValueError("codebook must be 'lowrank' or 'geometric'")
        if self.local_kernel_size < 2:
            raise ValueError("local_kernel_size must be >= 2")

    def to_dict(self) -> dict:
        return asdict(self)
