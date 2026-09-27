"""Goal-free factual record selector."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import EncoderBlock
from .positions import age_sincos


class FactualSelector(nn.Module):
    """Score factual records without reading a goal or a future target."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.type_embedding = nn.Embedding(
            3,
            hidden_dim,
            device="cpu",
            dtype=torch.float32,
        )
        self.blocks = nn.ModuleList(
            EncoderBlock(hidden_dim, heads, ffn_dim, dropout)
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(
            hidden_dim,
            eps=1e-5,
            elementwise_affine=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.score = nn.Linear(
            hidden_dim,
            1,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )

    def forward(
        self,
        projected_pooled_grid: Tensor,
        projected_action: Tensor,
        age: Tensor,
        record_type: Tensor,
        valid: Tensor,
    ) -> Tensor:
        positional = age_sincos(
            age,
            dim=projected_pooled_grid.shape[-1],
        ).to(
            device=projected_pooled_grid.device,
            dtype=projected_pooled_grid.dtype,
        )
        value = (
            projected_pooled_grid
            + projected_action
            + positional
            + self.type_embedding(record_type)
        )
        for block in self.blocks:
            value = block(value, valid)
        value = self.final_norm(value)
        score = self.score(value).squeeze(-1).float()
        return score.masked_fill(~valid, 0.0)
