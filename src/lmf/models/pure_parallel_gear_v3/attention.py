"""Bounded grouped-query local attention shared by hybrid and control models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LocalKVCache:
    key: torch.Tensor
    value: torch.Tensor
    segment_id: torch.Tensor
    valid: torch.Tensor

    def detach(self) -> "LocalKVCache":
        return LocalKVCache(
            self.key.detach(),
            self.value.detach(),
            self.segment_id.detach(),
            self.valid.detach(),
        )

    def to(self, *args, **kwargs) -> "LocalKVCache":
        key = self.key.to(*args, **kwargs)
        device = key.device
        return LocalKVCache(
            key,
            self.value.to(*args, **kwargs),
            self.segment_id.to(device=device),
            self.valid.to(device=device),
        )


class BoundedLocalAttention(nn.Module):
    """Causal GQA over a fixed right-aligned window.

    Training materializes ``[batch, sequence, heads, window]`` scores, never a
    sequence-square matrix. Generation carries a constant-size KV ring.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        window: int,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if heads % kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if window < 2:
            raise ValueError("local attention window must be at least two")
        self.dim = int(dim)
        self.heads = int(heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = dim // heads
        self.window = int(window)
        self.group_size = heads // kv_heads
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(
            dim, 2 * kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(heads, window))

    def initial_cache(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LocalKVCache:
        shape = (batch, self.window, self.kv_heads, self.head_dim)
        return LocalKVCache(
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
            torch.full((batch, self.window), -1, device=device, dtype=torch.long),
            torch.zeros((batch, self.window), device=device, dtype=torch.bool),
        )

    @staticmethod
    def _windows(value: torch.Tensor, window: int) -> torch.Tensor:
        padded = F.pad(value, (0, 0, 0, 0, window - 1, 0))
        unfolded = padded.unfold(1, window, 1)
        return unfolded.permute(0, 1, 4, 2, 3)

    @staticmethod
    def _scalar_windows(value: torch.Tensor, window: int, pad: int) -> torch.Tensor:
        padded = F.pad(value, (window - 1, 0), value=pad)
        return padded.unfold(1, window, 1)

    def _fixed_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        segment_id: torch.Tensor,
        valid: torch.Tensor,
    ) -> LocalKVCache:
        take = min(self.window, key.shape[1])
        key = key[:, -take:]
        value = value[:, -take:]
        segment_id = segment_id[:, -take:]
        valid = valid[:, -take:]
        padding = self.window - take
        if padding:
            key = F.pad(key, (0, 0, 0, 0, padding, 0))
            value = F.pad(value, (0, 0, 0, 0, padding, 0))
            segment_id = F.pad(segment_id, (padding, 0), value=-1)
            valid = F.pad(valid, (padding, 0), value=False)
        return LocalKVCache(key, value, segment_id, valid)

    def _blocked_training_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Use one fused SDPA call over a batch of overlapping local blocks."""
        batch, length = query.shape[:2]
        chunks = (length + self.window - 1) // self.window
        padded_length = chunks * self.window
        padding = padded_length - length
        if padding:
            query = F.pad(query, (0, 0, 0, 0, 0, padding))
            key = F.pad(key, (0, 0, 0, 0, 0, padding))
            value = F.pad(value, (0, 0, 0, 0, 0, padding))
            token_mask = F.pad(token_mask, (0, padding), value=False)
            segment_ids = F.pad(segment_ids, (0, padding), value=-1)
        key = key.repeat_interleave(self.group_size, dim=2)
        value = value.repeat_interleave(self.group_size, dim=2)
        query_block = query.reshape(
            batch, chunks, self.window, self.heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        key_source = F.pad(key, (0, 0, 0, 0, self.window, 0))
        value_source = F.pad(value, (0, 0, 0, 0, self.window, 0))
        key_block = key_source.unfold(1, 2 * self.window, self.window).permute(
            0, 1, 2, 4, 3
        )
        value_block = value_source.unfold(
            1, 2 * self.window, self.window
        ).permute(0, 1, 2, 4, 3)

        query_segments = segment_ids.reshape(batch, chunks, self.window)
        key_segments = F.pad(
            segment_ids, (self.window, 0), value=-1
        ).unfold(1, 2 * self.window, self.window)
        key_valid = F.pad(
            token_mask, (self.window, 0), value=False
        ).unfold(1, 2 * self.window, self.window)
        query_position = torch.arange(
            self.window, device=query.device
        )[:, None]
        key_position = torch.arange(
            -self.window, self.window, device=query.device
        )[None, :]
        distance = query_position - key_position
        causal_window = (distance >= 0) & (distance < self.window)
        allowed = (
            causal_window[None, None]
            & key_valid[:, :, None]
            & (query_segments[:, :, :, None] == key_segments[:, :, None])
        )
        relative_index = (self.window - 1 - distance).clamp(
            0, self.window - 1
        )
        bias = self.relative_bias[:, relative_index]
        attention_mask = bias[None, None].expand(
            batch, chunks, -1, -1, -1
        ).clone()
        attention_mask = attention_mask.masked_fill(
            ~allowed[:, :, None],
            torch.finfo(attention_mask.dtype).min,
        )

        flat_batch = batch * chunks
        attended = F.scaled_dot_product_attention(
            query_block.reshape(
                flat_batch, self.heads, self.window, self.head_dim
            ),
            key_block.reshape(
                flat_batch, self.heads, 2 * self.window, self.head_dim
            ),
            value_block.reshape(
                flat_batch, self.heads, 2 * self.window, self.head_dim
            ),
            attn_mask=attention_mask.reshape(
                flat_batch, self.heads, self.window, 2 * self.window
            ),
        )
        attended = attended.reshape(
            batch, chunks, self.heads, self.window, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        return attended.reshape(batch, padded_length, self.dim)[:, :length]

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        token_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        cache: LocalKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, LocalKVCache | None]:
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).reshape(
            batch, length, self.heads, self.head_dim
        )
        key, value = self.kv_proj(hidden).reshape(
            batch, length, 2, self.kv_heads, self.head_dim
        ).unbind(dim=2)

        if cache is None and (length > 1 or not use_cache):
            attended = self._blocked_training_attention(
                query,
                key,
                value,
                token_mask,
                segment_ids,
            )
            output = self.out_proj(attended)
            output = output * token_mask[:, :, None].to(output.dtype)
            next_cache = (
                self._fixed_cache(key, value, segment_ids, token_mask)
                if use_cache
                else None
            )
            return output, next_cache

        if cache is None:
            combined_key = key
            combined_value = value
            combined_segments = segment_ids
            combined_valid = token_mask
        else:
            combined_key = torch.cat((cache.key, key), dim=1)
            combined_value = torch.cat((cache.value, value), dim=1)
            combined_segments = torch.cat(
                (cache.segment_id, segment_ids), dim=1
            )
            combined_valid = torch.cat((cache.valid, token_mask), dim=1)

        if cache is not None and length > 1:
            prefix_query = torch.zeros(
                (
                    batch,
                    cache.key.shape[1],
                    self.heads,
                    self.head_dim,
                ),
                device=query.device,
                dtype=query.dtype,
            )
            attended = self._blocked_training_attention(
                torch.cat((prefix_query, query), dim=1),
                combined_key,
                combined_value,
                combined_valid,
                combined_segments,
            )[:, -length:]
            output = self.out_proj(attended)
            output = output * token_mask[:, :, None].to(output.dtype)
            next_cache = (
                self._fixed_cache(
                    combined_key,
                    combined_value,
                    combined_segments,
                    combined_valid,
                )
                if use_cache
                else None
            )
            return output, next_cache

        key_windows = self._windows(combined_key, self.window)[:, -length:]
        value_windows = self._windows(combined_value, self.window)[:, -length:]
        segment_windows = self._scalar_windows(
            combined_segments, self.window, -1
        )[:, -length:]
        valid_windows = self._scalar_windows(
            combined_valid, self.window, 0
        )[:, -length:].bool()

        key_windows = key_windows.repeat_interleave(self.group_size, dim=3)
        value_windows = value_windows.repeat_interleave(self.group_size, dim=3)
        scores = (
            query.float()[:, :, None] * key_windows.float()
        ).sum(dim=-1)
        scores = scores.permute(0, 1, 3, 2)
        scores = scores * (self.head_dim ** -0.5)
        scores = scores + self.relative_bias[None, None]
        allowed = (
            valid_windows
            & (segment_windows == segment_ids[:, :, None])
        )
        scores = scores.masked_fill(
            ~allowed[:, :, None],
            torch.finfo(scores.dtype).min,
        )
        probability = scores.softmax(dim=-1).to(value_windows.dtype)
        attended = (
            probability.permute(0, 1, 3, 2)[..., None] * value_windows
        ).sum(dim=2)
        attended = attended.reshape(batch, length, self.dim)
        output = self.out_proj(attended)
        output = output * token_mask[:, :, None].to(output.dtype)

        next_cache = (
            self._fixed_cache(
                combined_key,
                combined_value,
                combined_segments,
                combined_valid,
            )
            if use_cache
            else None
        )
        return output, next_cache
