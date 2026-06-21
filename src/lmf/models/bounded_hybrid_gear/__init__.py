"""Bounded Hybrid Gear public API: bounded local-attention trunk + scan-based
Gear memory, with token-rate and block-rate (switchable fusion) variants."""

from .attention import BoundedLocalAttention, LocalKVCache
from .model import (
    BoundedTransformerCache,
    BoundedTransformerConfig,
    BoundedTransformerLM,
    GearScanState,
    HybridGearCache,
    HybridParallelGearConfig,
    HybridParallelGearLM,
    PureGearV3Cache,
    PureGearV3Layer,
    PureParallelGearV3Config,
    PureParallelGearV3LM,
    build_bounded_transformer,
    build_hybrid_parallel_gear,
    build_pure_parallel_gear_v3,
)
from .mps_scan import mps_affine_scan
from .scan import (
    chunked_affine_scan,
    chunked_rotor_scan,
    complex_mul,
    compose_affine,
    hillis_steele_affine_scan,
)
from .trainer import (
    PureParallelGearV3Trainer,
    build_bounded_transformer_trainer,
    build_hybrid_parallel_gear_trainer,
    build_pure_parallel_gear_v3_trainer,
)
from .block_memory import (
    BlockGearMemoryCache,
    BlockHybridGearV4Cache,
    BlockHybridGearV4Config,
    BlockHybridGearV4LM,
    build_bounded_hybrid_gear_block_additive,
    build_bounded_hybrid_gear_block_selective_film,
    build_bounded_hybrid_gear_block_bank_router,
)

__all__ = [
    "BoundedLocalAttention",
    "BoundedTransformerCache",
    "BoundedTransformerConfig",
    "BoundedTransformerLM",
    "BlockGearMemoryCache",
    "BlockHybridGearV4Cache",
    "BlockHybridGearV4Config",
    "BlockHybridGearV4LM",
    "GearScanState",
    "HybridGearCache",
    "HybridParallelGearConfig",
    "HybridParallelGearLM",
    "LocalKVCache",
    "PureGearV3Cache",
    "PureGearV3Layer",
    "PureParallelGearV3Config",
    "PureParallelGearV3LM",
    "PureParallelGearV3Trainer",
    "build_bounded_transformer",
    "build_bounded_hybrid_gear_block_additive",
    "build_bounded_hybrid_gear_block_selective_film",
    "build_bounded_hybrid_gear_block_bank_router",
    "build_hybrid_parallel_gear",
    "build_pure_parallel_gear_v3",
    "build_bounded_transformer_trainer",
    "build_hybrid_parallel_gear_trainer",
    "build_pure_parallel_gear_v3_trainer",
    "chunked_affine_scan",
    "chunked_rotor_scan",
    "complex_mul",
    "compose_affine",
    "hillis_steele_affine_scan",
    "mps_affine_scan",
]
