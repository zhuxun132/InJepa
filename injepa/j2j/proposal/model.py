"""The single parameterized ProposalJEPA top-level object."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .future import MaskedFutureDecoder
from .initialization import initialize_proposal_
from .selector import FactualSelector


@dataclass(frozen=True)
class ProposalOutput:
    tape: Tensor
    log_mass: Tensor


def _require_positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_constructor(
    latent_dim: object,
    grid_side: object,
    hidden_dim: object,
    heads: object,
    ffn_dim: object,
    selector_depth: object,
    future_depth: object,
    dropout: object,
    modes: object,
    horizon: object,
    global_seed: object,
) -> None:
    latent = _require_positive_integer("latent_dim", latent_dim)
    _require_positive_integer("grid_side", grid_side)
    hidden = _require_positive_integer("hidden_dim", hidden_dim)
    attention_heads = _require_positive_integer("heads", heads)
    _require_positive_integer("ffn_dim", ffn_dim)
    _require_positive_integer("selector_depth", selector_depth)
    _require_positive_integer("future_depth", future_depth)
    _require_positive_integer("modes", modes)
    _require_positive_integer("horizon", horizon)
    if hidden % 4 != 0:
        raise ValueError("hidden_dim must be divisible by four")
    if hidden % attention_heads != 0:
        raise ValueError("hidden_dim must be divisible by heads")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise TypeError("dropout must be a real scalar")
    if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    if global_seed < 0 or global_seed >= 2**64:
        raise ValueError("global_seed must be an unsigned 64-bit integer")
    if latent <= 0:
        raise ValueError("latent_dim must be positive")


class ProposalJEPA(nn.Module):
    """Factual selector plus goal-conditioned future proposal network."""

    def __init__(
        self,
        *,
        latent_dim: int = 768,
        grid_side: int = 6,
        hidden_dim: int = 384,
        heads: int = 6,
        ffn_dim: int = 1536,
        selector_depth: int = 3,
        future_depth: int = 3,
        dropout: float = 0.1,
        modes: int = 2,
        horizon: int = 4,
        global_seed: int,
    ) -> None:
        _validate_constructor(
            latent_dim,
            grid_side,
            hidden_dim,
            heads,
            ffn_dim,
            selector_depth,
            future_depth,
            dropout,
            modes,
            horizon,
            global_seed,
        )
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.input_proj = nn.Linear(
                latent_dim,
                hidden_dim,
                bias=True,
                device="cpu",
                dtype=torch.float32,
            )
            self.selector = FactualSelector(
                hidden_dim,
                heads,
                ffn_dim,
                selector_depth,
                float(dropout),
            )
            self.future = MaskedFutureDecoder(
                latent_dim,
                grid_side,
                hidden_dim,
                heads,
                ffn_dim,
                future_depth,
                float(dropout),
                modes,
                horizon,
            )
            initialize_proposal_(self, global_seed=global_seed)

    def _validate_records(
        self,
        record_grid: Tensor,
        incoming_action_embedding: Tensor,
        record_age: Tensor,
        record_type: Tensor,
        record_valid: Tensor,
    ) -> tuple[int, int, int, int]:
        values = (
            record_grid,
            incoming_action_embedding,
            record_age,
            record_type,
            record_valid,
        )
        if not all(isinstance(value, Tensor) for value in values):
            raise TypeError("all record fields must be tensors")
        if record_grid.ndim != 4:
            raise ValueError("record_grid must have rank four")
        batch, records, spatial, latent = record_grid.shape
        if batch < 1 or records < 1:
            raise ValueError("batch and record axes must be nonempty")
        if spatial != self.future.spatial_position.shape[0]:
            raise ValueError("record spatial width does not match the model")
        if latent != self.input_proj.in_features:
            raise ValueError("record latent width does not match the model")
        if incoming_action_embedding.shape != (batch, records, latent):
            raise ValueError("incoming action shape is inconsistent")
        if record_age.shape != (batch, records):
            raise ValueError("record age shape is inconsistent")
        if record_type.shape != (batch, records):
            raise ValueError("record type shape is inconsistent")
        if record_valid.shape != (batch, records):
            raise ValueError("record valid shape is inconsistent")
        if record_grid.dtype != torch.float32:
            raise TypeError("record_grid must have dtype float32")
        if incoming_action_embedding.dtype != torch.float32:
            raise TypeError("incoming_action_embedding must have dtype float32")
        if record_age.dtype != torch.int64:
            raise TypeError("record_age must have dtype int64")
        if record_type.dtype != torch.int64:
            raise TypeError("record_type must have dtype int64")
        if record_valid.dtype != torch.bool:
            raise TypeError("record_valid must have dtype bool")

        model_device = self.input_proj.weight.device
        if any(value.device != model_device for value in values):
            raise ValueError("record fields and model must share one device")
        if bool((record_valid.sum(dim=1) < 1).any()):
            raise ValueError("each row must contain a valid factual record")
        if not bool(torch.isfinite(record_grid[record_valid]).all()):
            raise ValueError("valid record grids must be finite")
        if not bool(
            torch.isfinite(incoming_action_embedding[record_valid]).all()
        ):
            raise ValueError("valid incoming actions must be finite")

        current = record_valid & (record_type == 2)
        if not bool((current.sum(dim=1) == 1).all()):
            raise ValueError("each row must contain exactly one current record")
        if not bool((record_age[current] == 0).all()):
            raise ValueError("the current record must have age zero")
        noncurrent = record_valid & ~current
        if not bool((record_age[noncurrent] >= 1).all()):
            raise ValueError("non-current factual ages must be at least one")
        if not bool(
            ((record_type[noncurrent] == 0) | (record_type[noncurrent] == 1)).all()
        ):
            raise ValueError("non-current factual record types must be bank or recent")
        return batch, records, spatial, latent

    def _validate_goal_active(
        self,
        goal_grid: Tensor,
        active_h: Tensor,
        batch: int,
        spatial: int,
        latent: int,
    ) -> None:
        if not isinstance(goal_grid, Tensor) or not isinstance(active_h, Tensor):
            raise TypeError("goal_grid and active_h must be tensors")
        horizon = self.future.horizon_embedding.num_embeddings
        if goal_grid.shape != (batch, spatial, latent):
            raise ValueError("goal grid shape is inconsistent")
        if active_h.shape != (batch, horizon):
            raise ValueError("active_h shape is inconsistent")
        if goal_grid.dtype != torch.float32:
            raise TypeError("goal_grid must have dtype float32")
        if active_h.dtype != torch.bool:
            raise TypeError("active_h must have dtype bool")
        model_device = self.input_proj.weight.device
        if goal_grid.device != model_device or active_h.device != model_device:
            raise ValueError("goal, horizon mask, and model must share one device")
        if not bool(torch.isfinite(goal_grid).all()):
            raise ValueError("goal grid must be finite")
        if not bool(active_h[:, 0].all()):
            raise ValueError("each active horizon must be a nonempty prefix")
        if horizon > 1 and bool(((~active_h[:, :-1]) & active_h[:, 1:]).any()):
            raise ValueError("active_h must be a contiguous true prefix")

    @staticmethod
    def _sanitize_records(
        record_grid: Tensor,
        incoming_action_embedding: Tensor,
        record_age: Tensor,
        record_type: Tensor,
        record_valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        grid = record_grid.detach().masked_fill(
            ~record_valid[:, :, None, None], 0.0
        )
        action = incoming_action_embedding.detach().masked_fill(
            ~record_valid[:, :, None], 0.0
        )
        age = record_age.masked_fill(~record_valid, 0)
        kind = record_type.masked_fill(~record_valid, 0)
        return grid, action, age, kind

    def score_records(
        self,
        record_grid,
        incoming_action_embedding,
        record_age,
        record_type,
        record_valid,
    ) -> Tensor:
        batch, records, _spatial, latent = self._validate_records(
            record_grid,
            incoming_action_embedding,
            record_age,
            record_type,
            record_valid,
        )
        grid, action, age, kind = self._sanitize_records(
            record_grid,
            incoming_action_embedding,
            record_age,
            record_type,
            record_valid,
        )
        pooled = grid.mean(dim=2)
        projected_grid = self.input_proj(pooled.reshape(batch * records, latent))
        projected_action = self.input_proj(
            action.reshape(batch * records, latent)
        )
        hidden = projected_grid.shape[-1]
        return self.selector(
            projected_grid.reshape(batch, records, hidden).detach(),
            projected_action.reshape(batch, records, hidden).detach(),
            age,
            kind,
            record_valid,
        )

    def forward(
        self,
        record_grid,
        incoming_action_embedding,
        record_age,
        record_type,
        record_valid,
        goal_grid,
        active_h,
    ) -> ProposalOutput:
        batch, records, spatial, latent = self._validate_records(
            record_grid,
            incoming_action_embedding,
            record_age,
            record_type,
            record_valid,
        )
        self._validate_goal_active(
            goal_grid,
            active_h,
            batch,
            spatial,
            latent,
        )
        grid, action, age, kind = self._sanitize_records(
            record_grid,
            incoming_action_embedding,
            record_age,
            record_type,
            record_valid,
        )
        goal = goal_grid.detach()
        projected_visual = self.input_proj(
            grid.reshape(batch * records * spatial, latent)
        )
        projected_action = self.input_proj(
            action.reshape(batch * records, latent)
        )
        projected_goal = self.input_proj(goal.reshape(batch * spatial, latent))
        hidden = projected_visual.shape[-1]
        tape, log_mass = self.future(
            projected_visual.reshape(batch, records, spatial, hidden),
            projected_action.reshape(batch, records, hidden),
            age,
            kind,
            record_valid,
            projected_goal.reshape(batch, spatial, hidden),
            active_h,
        )
        return ProposalOutput(tape=tape, log_mass=log_mass)
