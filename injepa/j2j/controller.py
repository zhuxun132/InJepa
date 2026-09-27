"""Deterministic one-step control over disposable H4 candidate rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
from torch import Tensor, nn

from j2j.adapter import ActionId, Raw4Adapter
from j2j.memory_objective import FactualModelBatch
from j2j.rollout import CandidateRollout, rollout_candidates


class ControlStatus(Enum):
    """The three executable outcomes of one closed-loop decision."""

    ACTION = "ACTION"
    STOP = "STOP"
    NO_VERIFIED_PROPOSAL = "NO_VERIFIED_PROPOSAL"


@dataclass(frozen=True)
class ControlThresholds:
    """Horizon-specific consistency thresholds and the STOP threshold."""

    path: Tensor
    endpoint: Tensor
    stop: float


@dataclass(frozen=True)
class ControlStep:
    """One action decision plus disposable current-cycle diagnostics."""

    action_id: Tensor
    status: tuple[ControlStatus, ...]
    diagnostics: object


@dataclass(frozen=True)
class _ControlDiagnostics:
    selected_k: Tensor
    selected_h: Tensor
    consistent: Tensor
    goal_distance: Tensor


def _threshold_horizon(thresholds: ControlThresholds) -> int:
    if not isinstance(thresholds, ControlThresholds):
        raise TypeError("thresholds must be a ControlThresholds")
    for name, value in (
        ("path", thresholds.path),
        ("endpoint", thresholds.endpoint),
    ):
        if not isinstance(value, Tensor) or value.ndim != 1:
            raise ValueError(f"{name} threshold must have shape [H]")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} threshold must have dtype float32")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} threshold must be finite")
    if thresholds.path.shape != thresholds.endpoint.shape:
        raise ValueError("path and endpoint thresholds must share one horizon")
    if thresholds.path.numel() < 1:
        raise ValueError("threshold horizon must be nonempty")
    if isinstance(thresholds.stop, bool) or not isinstance(
        thresholds.stop, (int, float)
    ):
        raise TypeError("stop threshold must be a real scalar")
    if not math.isfinite(float(thresholds.stop)):
        raise ValueError("stop threshold must be finite")
    return int(thresholds.path.numel())


def _require_frozen_eval_owner(owner: object, *, name: str) -> nn.Module:
    if not isinstance(owner, nn.Module):
        raise TypeError(f"{name} must be an nn.Module")
    if any(module.training for module in owner.modules()):
        raise RuntimeError(f"{name} must be entirely in eval mode")
    if any(parameter.requires_grad for parameter in owner.parameters()):
        raise RuntimeError(f"{name} must not contain trainable parameters")
    return owner


def _motion_action(value: int) -> bool:
    return value in (
        int(ActionId.FWD),
        int(ActionId.LEFT),
        int(ActionId.RIGHT),
    )


@torch.inference_mode()
def select_verified_candidate(
    rollout: CandidateRollout,
    log_mass: Tensor,
    goal_global: Tensor,
    thresholds: ControlThresholds,
) -> ControlStep:
    """Select each row's verified prefix by the frozen lexicographic rule."""

    horizon = _threshold_horizon(thresholds)
    predicted = rollout.predicted.float()
    batch_size, candidate_count, rollout_horizon, latent_dim = predicted.shape
    if rollout_horizon != horizon:
        raise ValueError("rollout and threshold horizons disagree")
    if rollout.action_id.shape != (batch_size, candidate_count, horizon):
        raise ValueError("rollout action_id shape is inconsistent")
    if rollout.valid_prefix.shape != (batch_size, candidate_count, horizon):
        raise ValueError("rollout valid_prefix shape is inconsistent")
    if rollout.path_error.shape != (batch_size, candidate_count, horizon):
        raise ValueError("rollout path_error shape is inconsistent")
    if rollout.endpoint_error.shape != (batch_size, candidate_count, horizon):
        raise ValueError("rollout endpoint_error shape is inconsistent")
    if log_mass.shape != (batch_size, candidate_count):
        raise ValueError("log_mass shape is inconsistent")
    if goal_global.shape != (batch_size, latent_dim):
        raise ValueError("goal_global shape is inconsistent")

    device = predicted.device
    path_threshold = thresholds.path.to(device=device, dtype=torch.float32)
    endpoint_threshold = thresholds.endpoint.to(
        device=device, dtype=torch.float32
    )
    consistent = (
        rollout.valid_prefix.to(device=device, dtype=torch.bool)
        & (
            rollout.path_error.float()
            <= path_threshold[None, None, :]
        )
        & (
            rollout.endpoint_error.float()
            <= endpoint_threshold[None, None, :]
        )
    )
    goal_distance = (
        predicted - goal_global.float()[:, None, None, :]
    ).abs().mean(dim=-1)
    mass = torch.exp(log_mass.float())

    selected_k = torch.full(
        (batch_size,), -1, dtype=torch.int64, device=device
    )
    selected_h = torch.full_like(selected_k, -1)
    action_id = torch.full_like(selected_k, -1)
    statuses: list[ControlStatus] = [
        ControlStatus.NO_VERIFIED_PROPOSAL
    ] * batch_size

    for batch_index in range(batch_size):
        candidates = torch.nonzero(
            consistent[batch_index], as_tuple=False
        )
        if candidates.numel() == 0:
            continue
        winner = min(
            (
                (
                    int(pair[0].item()),
                    int(pair[1].item()),
                )
                for pair in candidates
            ),
            key=lambda pair: (
                float(goal_distance[batch_index, pair[0], pair[1]].item()),
                -float(mass[batch_index, pair[0]].item()),
                pair[0],
                pair[1],
            ),
        )
        winner_k, winner_h_zero = winner
        first_action = int(
            rollout.action_id[batch_index, winner_k, 0].item()
        )
        if not _motion_action(first_action):
            continue
        selected_k[batch_index] = winner_k
        selected_h[batch_index] = winner_h_zero + 1
        action_id[batch_index] = first_action
        statuses[batch_index] = ControlStatus.ACTION

    diagnostics = _ControlDiagnostics(
        selected_k=selected_k,
        selected_h=selected_h,
        consistent=consistent,
        goal_distance=goal_distance,
    )
    return ControlStep(
        action_id=action_id,
        status=tuple(statuses),
        diagnostics=diagnostics,
    )


def _subset_facts(
    facts: FactualModelBatch,
    indices: Tensor,
) -> FactualModelBatch:
    batch_size = facts.record_grid.shape[0]
    full_indices = torch.arange(
        batch_size, dtype=torch.int64, device=indices.device
    )
    if torch.equal(indices, full_indices):
        return facts
    caller_rows = tuple(int(value) for value in indices.tolist())
    return FactualModelBatch(
        record_grid=facts.record_grid.index_select(0, indices),
        incoming_action_embedding=(
            facts.incoming_action_embedding.index_select(0, indices)
        ),
        record_age=facts.record_age.index_select(0, indices),
        record_type=facts.record_type.index_select(0, indices),
        record_valid=facts.record_valid.index_select(0, indices),
        pool_mask=facts.pool_mask.index_select(0, indices),
        view_keys=tuple(facts.view_keys[row] for row in caller_rows),
        padded_record_frame_keys=tuple(
            facts.padded_record_frame_keys[row] for row in caller_rows
        ),
    )


def _subset_or_identity(value: Tensor, indices: Tensor) -> Tensor:
    full_indices = torch.arange(
        value.shape[0], dtype=torch.int64, device=indices.device
    )
    if torch.equal(indices, full_indices):
        return value
    return value.index_select(0, indices)


class J2JController:
    """Borrow the supplied Proposal and INTACT objects for one-step control."""

    def __init__(
        self,
        *,
        intact_model: object,
        proposal: object,
        thresholds: ControlThresholds,
    ) -> None:
        _require_frozen_eval_owner(intact_model, name="intact_model")
        _require_frozen_eval_owner(proposal, name="proposal")
        _threshold_horizon(thresholds)
        self.intact_model = intact_model
        self.proposal = proposal
        self.thresholds = thresholds

    def _require_ready(self) -> None:
        _require_frozen_eval_owner(self.intact_model, name="intact_model")
        _require_frozen_eval_owner(self.proposal, name="proposal")

    @torch.inference_mode()
    def step(
        self,
        *,
        facts: FactualModelBatch,
        goal_grid: Tensor,
        embedding_history: Tensor,
        raw4_history: Tensor,
    ) -> ControlStep:
        """Return one primitive without retaining or updating cycle state."""

        self._require_ready()
        horizon = _threshold_horizon(self.thresholds)
        batch_size = embedding_history.shape[0]
        device = embedding_history.device
        current = embedding_history[:, -1].float()
        goal_global = goal_grid.float().mean(dim=1)
        stop_distance = (current - goal_global).abs().mean(dim=-1)
        gate_rows = torch.nonzero(
            stop_distance <= float(self.thresholds.stop),
            as_tuple=False,
        ).flatten()

        action_id = torch.full(
            (batch_size,), -1, dtype=torch.int64, device=device
        )
        statuses: list[ControlStatus] = [
            ControlStatus.NO_VERIFIED_PROPOSAL
        ] * batch_size
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if gate_rows.numel() > 0:
            gate_current = current.index_select(0, gate_rows)
            gate_previous = raw4_history.index_select(0, gate_rows)[:, -1]
            gate_logits = self.intact_model.action_logits(
                gate_current,
                torch.zeros_like(gate_current),
                gate_previous,
            )
            gate_decoded = Raw4Adapter.decode_logits(gate_logits)
            # The gate action is deliberately discarded, but it still crosses
            # the same real scalar ActionId -> raw4 adapter seam as rollout.
            gate_raw4 = torch.stack(
                [
                    Raw4Adapter.encode(ActionId(int(value.item())))
                    for value in gate_decoded
                ],
                dim=0,
            ).to(device=gate_logits.device, dtype=torch.float32)
            del gate_raw4
            gate_stop = gate_decoded == int(ActionId.STOP)
            if bool(gate_stop.any().item()):
                stop_rows = gate_rows.index_select(
                    0,
                    torch.nonzero(gate_stop, as_tuple=False).flatten(),
                )
                stopped[stop_rows] = True
                action_id[stop_rows] = int(ActionId.STOP)
                for row in stop_rows.tolist():
                    statuses[int(row)] = ControlStatus.STOP

        ordinary_rows = torch.nonzero(~stopped, as_tuple=False).flatten()
        if ordinary_rows.numel() == 0:
            empty_consistent = torch.zeros(
                batch_size,
                0,
                horizon,
                dtype=torch.bool,
                device=device,
            )
            empty_goal_distance = torch.full(
                (batch_size, 0, horizon),
                fill_value=float("inf"),
                dtype=torch.float32,
                device=device,
            )
            diagnostics = _ControlDiagnostics(
                selected_k=torch.full_like(action_id, -1),
                selected_h=torch.full_like(action_id, -1),
                consistent=empty_consistent,
                goal_distance=empty_goal_distance,
            )
            return ControlStep(action_id, tuple(statuses), diagnostics)

        ordinary_facts = _subset_facts(facts, ordinary_rows)
        ordinary_goal = _subset_or_identity(goal_grid, ordinary_rows)
        ordinary_embeddings = _subset_or_identity(
            embedding_history, ordinary_rows
        )
        ordinary_raw4 = _subset_or_identity(raw4_history, ordinary_rows)
        active_h = torch.ones(
            ordinary_rows.numel(),
            horizon,
            dtype=torch.bool,
            device=ordinary_goal.device,
        )
        proposed = self.proposal(
            ordinary_facts.record_grid,
            ordinary_facts.incoming_action_embedding,
            ordinary_facts.record_age,
            ordinary_facts.record_type,
            ordinary_facts.record_valid,
            ordinary_goal,
            active_h,
        )
        history_size = int(
            self.intact_model.predictor.pos_embedding.size(1)
        )
        rollout = rollout_candidates(
            self.intact_model,
            proposed.tape,
            ordinary_embeddings,
            ordinary_raw4,
            history_size=history_size,
        )
        selected = select_verified_candidate(
            rollout,
            proposed.log_mass,
            ordinary_goal.float().mean(dim=1),
            self.thresholds,
        )

        action_id[ordinary_rows] = selected.action_id.to(
            device=device, dtype=torch.int64
        )
        for local_row, caller_row in enumerate(ordinary_rows.tolist()):
            statuses[int(caller_row)] = selected.status[local_row]

        local_diagnostics = selected.diagnostics
        candidate_count = rollout.action_id.shape[1]
        selected_k = torch.full_like(action_id, -1)
        selected_h = torch.full_like(action_id, -1)
        selected_k[ordinary_rows] = local_diagnostics.selected_k.to(device)
        selected_h[ordinary_rows] = local_diagnostics.selected_h.to(device)
        consistent = torch.zeros(
            batch_size,
            candidate_count,
            horizon,
            dtype=torch.bool,
            device=device,
        )
        goal_distance = torch.full(
            (batch_size, candidate_count, horizon),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )
        consistent[ordinary_rows] = local_diagnostics.consistent.to(device)
        goal_distance[ordinary_rows] = local_diagnostics.goal_distance.to(device)
        diagnostics = _ControlDiagnostics(
            selected_k=selected_k,
            selected_h=selected_h,
            consistent=consistent,
            goal_distance=goal_distance,
        )
        return ControlStep(action_id, tuple(statuses), diagnostics)
