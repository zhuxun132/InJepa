"""Goal-conditioned masked-future decoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .blocks import DecoderBlock, run_block
from .positions import age_sincos, fixed_2d_sincos


class MaskedFutureDecoder(nn.Module):
    """Decode independent mode tapes from factual records and a goal grid."""

    def __init__(
        self,
        latent_dim: int,
        grid_side: int,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        depth: int,
        dropout: float,
        modes: int,
        horizon: int,
        *,
        attention_backend: str = "math",
        memory_norm: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        self.activation_checkpointing = activation_checkpointing
        self.type_embedding = nn.Embedding(
            6,
            hidden_dim,
            device="cpu",
            dtype=torch.float32,
        )
        self.mode_embedding = nn.Embedding(
            modes,
            hidden_dim,
            device="cpu",
            dtype=torch.float32,
        )
        self.horizon_embedding = nn.Embedding(
            horizon,
            hidden_dim,
            device="cpu",
            dtype=torch.float32,
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(hidden_dim, heads, ffn_dim, dropout,
                         attention_backend=attention_backend, memory_norm=memory_norm)
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(
            hidden_dim,
            eps=1e-5,
            elementwise_affine=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.output_proj = nn.Linear(
            hidden_dim,
            latent_dim,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.mass_head = nn.Linear(
            hidden_dim,
            1,
            bias=True,
            device="cpu",
            dtype=torch.float32,
        )
        self.register_buffer(
            "spatial_position",
            fixed_2d_sincos(dim=hidden_dim, grid_side=grid_side),
            persistent=True,
        )

    def forward(
        self,
        projected_record_visual: Tensor,
        projected_record_action: Tensor,
        record_age: Tensor,
        record_type: Tensor,
        record_valid: Tensor,
        projected_goal: Tensor,
        active_h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, records, spatial, hidden = projected_record_visual.shape
        modes = self.mode_embedding.num_embeddings
        horizon = self.horizon_embedding.num_embeddings
        position = self.spatial_position.to(
            device=projected_record_visual.device,
            dtype=projected_record_visual.dtype,
        )
        age = age_sincos(record_age, dim=hidden).to(
            device=projected_record_visual.device,
            dtype=projected_record_visual.dtype,
        )

        current = record_type == 2
        past_visual = torch.zeros_like(record_type)
        past_action = torch.ones_like(record_type)
        visual_type = torch.where(
            current,
            torch.full_like(record_type, 2),
            past_visual,
        )
        action_type = torch.where(
            current,
            torch.full_like(record_type, 3),
            past_action,
        )
        visual = (
            projected_record_visual
            + age[:, :, None, :]
            + position[None, None, :, :]
            + self.type_embedding(visual_type)[:, :, None, :]
        )
        action = (
            projected_record_action
            + age
            + self.type_embedding(action_type)
        )
        visual = visual.masked_fill(
            ~record_valid[:, :, None, None], 0.0
        )
        action = action.masked_fill(~record_valid[:, :, None], 0.0)
        record_memory = torch.cat((visual, action[:, :, None, :]), dim=2)
        record_memory = record_memory.reshape(
            batch, records * (spatial + 1), hidden
        )

        goal_index = torch.full(
            (batch, spatial),
            4,
            dtype=torch.int64,
            device=record_type.device,
        )
        goal = (
            projected_goal
            + position[None, :, :]
            + self.type_embedding(goal_index)
        )
        memory = torch.cat((record_memory, goal), dim=1)
        record_memory_valid = record_valid[:, :, None].expand(
            batch, records, spatial + 1
        ).reshape(batch, records * (spatial + 1))
        memory_valid = torch.cat(
            (
                record_memory_valid,
                torch.ones(
                    batch,
                    spatial,
                    dtype=torch.bool,
                    device=record_valid.device,
                ),
            ),
            dim=1,
        )
        memory = memory[:, None, :, :].expand(
            batch, modes, memory.shape[1], hidden
        ).reshape(batch * modes, memory.shape[1], hidden)
        memory_valid = memory_valid[:, None, :].expand(
            batch, modes, memory_valid.shape[1]
        ).reshape(batch * modes, memory_valid.shape[1])

        query_type_index = torch.full(
            (modes, horizon, spatial),
            5,
            dtype=torch.int64,
            device=record_type.device,
        )
        per_mode_query = (
            self.mode_embedding.weight[:, None, None, :]
            + self.horizon_embedding.weight[None, :, None, :]
            + position[None, None, :, :]
            + self.type_embedding(query_type_index)
        )
        query = per_mode_query[None, :, :, :, :].expand(
            batch, modes, horizon, spatial, hidden
        ).reshape(batch * modes, horizon * spatial, hidden)
        query_valid = active_h[:, None, :, None].expand(
            batch, modes, horizon, spatial
        ).reshape(batch * modes, horizon * spatial)
        query = query.masked_fill(~query_valid[..., None], 0.0)

        for block in self.blocks:
            query = run_block(block, query, query_valid, memory, memory_valid,
                              activation_checkpointing=self.activation_checkpointing)

        normalized = self.final_norm(query)
        normalized = normalized.masked_fill(~query_valid[..., None], 0.0)
        projected = self.output_proj(normalized)
        tape = projected.reshape(
            batch, modes, horizon, spatial, projected.shape[-1]
        )
        tape = tape.masked_fill(
            ~active_h[:, None, :, None, None], 0.0
        )

        normalized_tape = normalized.reshape(
            batch, modes, horizon, spatial, hidden
        )
        active_mask = active_h[:, None, :, None, None]
        denominator = (
            active_h.sum(dim=1).to(dtype=normalized_tape.dtype)
            * spatial
        )[:, None, None]
        pooled = normalized_tape.masked_fill(~active_mask, 0.0).sum(
            dim=(2, 3)
        ) / denominator
        mode_logits = self.mass_head(pooled).squeeze(-1)
        log_mass = torch.log_softmax(mode_logits.float(), dim=1)
        return tape, log_mass
