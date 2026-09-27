"""Pre-normalized masked attention blocks used by ProposalJEPA."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


def _clear_invalid(value: Tensor, valid: Tensor) -> Tensor:
    return value.masked_fill(~valid[..., None], 0.0)


def _attention_context(backend: str):
    if backend == "math":
        return sdpa_kernel(SDPBackend.MATH)
    if backend == "auto":
        return nullcontext()
    raise ValueError("attention_backend must be math or auto")


def run_block(block: nn.Module, *inputs: Tensor, activation_checkpointing: bool) -> Tensor:
    if (activation_checkpointing and not getattr(block, "retain_activations", False)
            and block.training and torch.is_grad_enabled()):
        return checkpoint(block, *inputs, use_reentrant=False, preserve_rng_state=True)
    return block(*inputs)


def validate_retained_activation_blocks(policy, *, activation_checkpointing, depths):
    """Resolve exact block names without parameters, RNG use, or fixed depths."""
    if not isinstance(policy, (list, tuple)) or any(type(name) is not str for name in policy):
        raise TypeError("retain_activation_blocks must be a list or tuple of exact block names")
    names = tuple(policy)
    if len(names) != len(set(names)):
        raise ValueError("retain_activation_blocks must not contain duplicates")
    if names and activation_checkpointing is not True:
        raise ValueError("retained blocks require activation_checkpointing=True")
    allowed = {f"{prefix}.{index}" for prefix, depth in depths.items() for index in range(depth)}
    if not set(names) <= allowed:
        raise ValueError("retain_activation_blocks contains an unknown or non-block name")
    return names


class FeedForward(nn.Module):
    """The exact GELU/dropout feed-forward branch shared by both block kinds."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            hidden_dim,
            ffn_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.gelu = nn.GELU(approximate="none")
        self.hidden_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(
            ffn_dim,
            hidden_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        value = self.fc1(value)
        value = self.gelu(value)
        value = self.hidden_dropout(value)
        value = self.fc2(value)
        return self.output_dropout(value)


class EncoderBlock(nn.Module):
    """Bidirectional masked factual-record encoder block."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        *,
        attention_backend: str = "math",
    ) -> None:
        super().__init__()
        if attention_backend not in {"math", "auto"}:
            raise ValueError("attention_backend must be math or auto")
        self.attention_backend = attention_backend
        self.self_norm = nn.LayerNorm(
            hidden_dim,
            eps=1e-5,
            elementwise_affine=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.self_output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(
            hidden_dim,
            eps=1e-5,
            elementwise_affine=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(self, value: Tensor, valid: Tensor) -> Tensor:
        value = _clear_invalid(value, valid)
        normalized = self.self_norm(value)
        with _attention_context(self.attention_backend):
            attended, _ = self.self_attn(
                query=normalized,
                key=normalized,
                value=normalized,
                key_padding_mask=None if bool(valid.all()) else ~valid,
                need_weights=False,
                is_causal=False,
            )
        value = _clear_invalid(
            value + self.self_output_dropout(attended), valid
        )
        normalized = self.ffn_norm(value)
        return _clear_invalid(value + self.ffn(normalized), valid)


class DecoderBlock(nn.Module):
    """Masked query decoder with optional tokenwise memory normalization."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        *,
        attention_backend: str = "math",
        memory_norm: bool = False,
    ) -> None:
        super().__init__()
        if attention_backend not in {"math", "auto"}:
            raise ValueError("attention_backend must be math or auto")
        if type(memory_norm) is not bool:
            raise ValueError("memory_norm must be bool")
        self.attention_backend = attention_backend
        self.memory_norm = memory_norm
        self.residual_scale = 1.0
        self.memory_norm_epsilon = 1e-5
        self.self_norm = nn.LayerNorm(
            hidden_dim,
            eps=1e-5,
            elementwise_affine=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.self_output_dropout = nn.Dropout(dropout)
        self.cross_norm = nn.LayerNorm(
            hidden_dim,
            eps=1e-5,
            elementwise_affine=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            batch_first=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.cross_output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(
            hidden_dim,
            eps=1e-5,
            elementwise_affine=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout)

    def forward(
        self,
        query: Tensor,
        query_valid: Tensor,
        memory: Tensor,
        memory_valid: Tensor,
    ) -> Tensor:
        query = _clear_invalid(query, query_valid)
        normalized = self.self_norm(query)
        with _attention_context(self.attention_backend):
            attended, _ = self.self_attn(
                query=normalized,
                key=normalized,
                value=normalized,
                key_padding_mask=None if bool(query_valid.all()) else ~query_valid,
                need_weights=False,
                is_causal=False,
            )
        query = _clear_invalid(
            query + self.residual_scale * self.self_output_dropout(attended), query_valid
        )

        normalized = self.cross_norm(query)
        if self.memory_norm:
            memory = F.layer_norm(
                _clear_invalid(memory, memory_valid), (memory.shape[-1],), eps=self.memory_norm_epsilon
            )
        with _attention_context(self.attention_backend):
            attended, _ = self.cross_attn(
                query=normalized,
                key=memory,
                value=memory,
                key_padding_mask=None if bool(memory_valid.all()) else ~memory_valid,
                need_weights=False,
                is_causal=False,
            )
        query = _clear_invalid(
            query + self.residual_scale * self.cross_output_dropout(attended), query_valid
        )

        normalized = self.ffn_norm(query)
        return _clear_invalid(query + self.residual_scale * self.ffn(normalized), query_valid)


class DividedDecoderBlock(DecoderBlock):
    """Causal temporal -> spatial -> factual cross attention -> FFN.

    Inherited ``self_*`` modules are the spatial stage. Temporal and spatial
    projections are independent (TimeSformer); this is not a dense-attention
    equivalent. No maximum time/spatial size is a learned parameter.
    """

    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float,
                 *, attention_backend: str = "math", memory_norm: bool = False):
        super().__init__(hidden_dim, heads, ffn_dim, dropout,
                         attention_backend=attention_backend, memory_norm=memory_norm)
        self.temporal_norm = nn.LayerNorm(hidden_dim, eps=1e-5, device="cpu", dtype=torch.float32)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True,
            device="cpu", dtype=torch.float32,
        )
        self.temporal_output_dropout = nn.Dropout(dropout)
        self.residual_scale = 1.0
        self.memory_norm_epsilon = 1e-5

    @staticmethod
    def _safe_padding(valid: Tensor) -> Tensor | None:
        # Unused subgroups get a finite zero sentinel; their outputs are cleared.
        # Avoid a redundant all-zero mask so auto SDPA can use its fast kernels.
        if bool(valid.all()):
            return None
        safe = valid.clone()
        safe[:, 0] |= ~safe.any(dim=1)
        return ~safe

    def forward(self, query: Tensor, query_valid: Tensor,
                memory: Tensor, memory_valid: Tensor) -> Tensor:
        if query.ndim != 4 or query_valid.shape != query.shape[:-1]:
            raise ValueError("divided query requires [B,H,M,d] and [B,H,M] validity")
        batch, horizon, spatial, hidden = query.shape
        if min(query.shape) < 1 or memory.ndim != 3 or memory.shape[0] != batch:
            raise ValueError("divided query/memory axes must be nonempty and agree")
        if memory.shape[-1] != hidden or memory_valid.shape != memory.shape[:-1]:
            raise ValueError("divided memory requires [B,S,d] and [B,S] validity")
        if query_valid.dtype != torch.bool or memory_valid.dtype != torch.bool:
            raise TypeError("attention validity must be boolean")
        if memory.shape[1] < 1 or not bool(memory_valid.any(dim=1).all()):
            raise ValueError("every decoder row requires factual/goal memory")
        query = _clear_invalid(query, query_valid)
        temporal = query.permute(0, 2, 1, 3).reshape(batch * spatial, horizon, hidden)
        temporal_valid = query_valid.permute(0, 2, 1).reshape(batch * spatial, horizon)
        causal = torch.ones(horizon, horizon, dtype=torch.bool, device=query.device).tril()
        if bool(temporal_valid.all()):
            mask = ~causal
        else:
            allowed = causal[None] & temporal_valid[:, None, :]
            # Invalid early queries and entirely padded locations must not produce
            # all-masked softmax NaNs, including during backward/checkpoint replay.
            empty = ~allowed.any(dim=-1)
            allowed[:, :, 0] |= empty
            mask = (~allowed)[:, None].expand(-1, self.temporal_attn.num_heads, -1, -1)
            mask = mask.reshape(batch * spatial * self.temporal_attn.num_heads, horizon, horizon)
        normalized = self.temporal_norm(temporal)
        with _attention_context(self.attention_backend):
            attended, _ = self.temporal_attn(normalized, normalized, normalized,
                                            attn_mask=mask, need_weights=False)
        temporal = _clear_invalid(temporal + self.residual_scale * self.temporal_output_dropout(attended), temporal_valid)
        query = temporal.reshape(batch, spatial, horizon, hidden).permute(0, 2, 1, 3)
        spatial_query = query.reshape(batch * horizon, spatial, hidden)
        spatial_valid = query_valid.reshape(batch * horizon, spatial)
        normalized = self.self_norm(spatial_query)
        with _attention_context(self.attention_backend):
            attended, _ = self.self_attn(normalized, normalized, normalized,
                key_padding_mask=self._safe_padding(spatial_valid), need_weights=False)
        query = _clear_invalid(spatial_query + self.residual_scale * self.self_output_dropout(attended), spatial_valid)
        query = query.reshape(batch, horizon * spatial, hidden)
        flat_valid = query_valid.reshape(batch, horizon * spatial)
        memory = _clear_invalid(memory, memory_valid)
        if self.memory_norm:
            memory = F.layer_norm(memory, (hidden,), eps=self.memory_norm_epsilon)
        with _attention_context(self.attention_backend):
            attended, _ = self.cross_attn(self.cross_norm(query), memory, memory,
                key_padding_mask=self._safe_padding(memory_valid), need_weights=False)
        query = _clear_invalid(query + self.residual_scale * self.cross_output_dropout(attended), flat_valid)
        query = _clear_invalid(query + self.residual_scale * self.ffn(self.ffn_norm(query)), flat_valid)
        return query.reshape(batch, horizon, spatial, hidden)
