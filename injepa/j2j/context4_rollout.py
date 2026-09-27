"""Folded all-mode context-four rollout with final-only candidate ranking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from j2j.context4 import CONTEXT_SIZE, append_context_latent


@dataclass(frozen=True)
class CandidateRanking:
    goal_distance: Tensor
    ordered_indices: Tensor
    winner_k: Tensor
    winner_h: Tensor


@dataclass(frozen=True)
class CandidateRollout:
    endpoints: Tensor
    actions: Tensor
    consistency: Tensor
    goal_distance: Tensor
    ordered_indices: Tensor
    winner_k: Tensor
    winner_h: Tensor
    first_action: Tensor


def _require_finite(name: str, value: Tensor) -> None:
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _raw4(action_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
    result = torch.zeros(
        (*action_ids.shape, 4),
        device=action_ids.device,
        dtype=dtype,
    )
    return result.scatter_(-1, action_ids[..., None], 1.0)


def rank_all_candidates(
    endpoints: Tensor,
    goal_grid: Tensor,
    log_mass: Tensor,
) -> CandidateRanking:
    """Rank every finite ``(mode,prefix)`` by the frozen lexicographic key."""

    if endpoints.ndim != 5:
        raise ValueError("endpoints must have shape [batch,modes,horizon,spatial,latent]")
    batch, modes, horizon, spatial, latent = endpoints.shape
    if batch < 1 or modes < 1 or horizon < 1:
        raise ValueError("candidate axes must be nonempty")
    if goal_grid.shape != (batch, spatial, latent):
        raise ValueError("goal_grid shape is inconsistent with endpoints")
    if log_mass.shape != (batch, modes):
        raise ValueError("log_mass shape is inconsistent with endpoints")
    if len({endpoints.device, goal_grid.device, log_mass.device}) != 1:
        raise ValueError("candidate tensors must share one device")
    _require_finite("endpoints", endpoints)
    _require_finite("goal_grid", goal_grid)
    _require_finite("log_mass", log_mass)

    goal_distance = (
        endpoints.float() - goal_grid[:, None, None].float()
    ).abs().mean(dim=(-1, -2))
    flat_distance = goal_distance.reshape(batch, modes * horizon)
    flat_mass_key = (-log_mass.float())[:, :, None].expand(
        batch, modes, horizon
    ).reshape(batch, modes * horizon)

    # Initial flat order is (k,h), so two stable sorts implement exactly
    # (goal_distance, -log_mass, k, h) without dropping or CPU-looping rows.
    secondary = torch.argsort(flat_mass_key, dim=1, stable=True)
    distance_in_secondary_order = flat_distance.gather(1, secondary)
    primary = torch.argsort(distance_in_secondary_order, dim=1, stable=True)
    flat_order = secondary.gather(1, primary)
    ordered_indices = torch.stack(
        (flat_order // horizon, flat_order % horizon),
        dim=-1,
    )
    winner_k = ordered_indices[:, 0, 0]
    winner_h = ordered_indices[:, 0, 1]
    return CandidateRanking(
        goal_distance=goal_distance,
        ordered_indices=ordered_indices,
        winner_k=winner_k,
        winner_h=winner_h,
    )


def rollout_candidates(
    *,
    proposal_tape: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    factual_context: Tensor,
    outgoing_raw4: Tensor,
    context_valid: Tensor,
    actor_step,
    forward_step,
    context_size: int = CONTEXT_SIZE,
) -> CandidateRollout:
    """Roll all K modes together, once per horizon step, then rank all KxH."""

    if context_size != CONTEXT_SIZE:
        raise ValueError("context_size must equal the RAE-NWM source value 4")
    if proposal_tape.ndim != 5 or factual_context.ndim != 4:
        raise ValueError("proposal/context ranks are inconsistent")
    batch, modes, horizon, spatial, latent = proposal_tape.shape
    if factual_context.shape[0] != batch or factual_context.shape[2:] != (spatial, latent):
        raise ValueError("factual_context shape is inconsistent with proposal_tape")
    context_length = factual_context.shape[1]
    if not 1 <= context_length <= context_size:
        raise ValueError("factual context length must be in [1,4]")
    if outgoing_raw4.shape != (batch, context_length, 4):
        raise ValueError("outgoing_raw4 shape is inconsistent")
    if context_valid.shape != (batch, context_length) or context_valid.dtype != torch.bool:
        raise ValueError("context_valid must be bool [batch,context]")
    if not bool(context_valid[:, -1].all()):
        raise ValueError("the current factual slot must be valid")
    if log_mass.shape != (batch, modes) or goal_grid.shape != (batch, spatial, latent):
        raise ValueError("proposal mass or goal shape is inconsistent")
    if len(
        {
            proposal_tape.device,
            log_mass.device,
            goal_grid.device,
            factual_context.device,
            outgoing_raw4.device,
            context_valid.device,
        }
    ) != 1:
        raise ValueError("rollout tensors must share one device")
    for name, value in (
        ("proposal_tape", proposal_tape),
        ("log_mass", log_mass),
        ("goal_grid", goal_grid),
        ("factual_context", factual_context),
        ("outgoing_raw4", outgoing_raw4),
    ):
        _require_finite(name, value)

    grids = factual_context[:, None].expand(
        batch, modes, context_length, spatial, latent
    ).reshape(batch * modes, context_length, spatial, latent).clone()
    action_context = outgoing_raw4[:, None].expand(
        batch, modes, context_length, 4
    ).reshape(batch * modes, context_length, 4).clone()
    valid = context_valid[:, None].expand(
        batch, modes, context_length
    ).reshape(batch * modes, context_length).clone()
    tape = proposal_tape.reshape(batch * modes, horizon, spatial, latent)
    current = grids[:, -1]
    previous = action_context[:, -1]
    endpoint_rows: list[Tensor] = []
    action_rows: list[Tensor] = []
    consistency_rows: list[Tensor] = []

    for step in range(horizon):
        intent = tape[:, step] - current
        logits = actor_step(current, intent, previous)
        if logits.shape != (batch * modes, 4):
            raise ValueError("actor_step must return [batch*modes,4]")
        _require_finite("actor logits", logits)
        motion_logits = logits.clone()
        motion_logits[:, 0] = -torch.inf
        action = motion_logits.argmax(dim=-1)
        if not bool(((action >= 1) & (action <= 3)).all()):
            raise RuntimeError("ordinary proposals may only choose motion actions")
        action_raw4 = _raw4(action, dtype=action_context.dtype)

        conditioned_actions = action_context.clone()
        conditioned_actions[:, -1] = action_raw4
        predicted = forward_step(grids, conditioned_actions, valid)
        if predicted.shape != (batch * modes, spatial, latent):
            raise ValueError("forward_step must return [batch*modes,spatial,latent]")
        _require_finite("forward prediction", predicted)

        endpoint_rows.append(predicted)
        action_rows.append(action)
        consistency_rows.append(
            (predicted.float() - tape[:, step].float()).abs().mean(dim=(-1, -2))
        )
        grids = append_context_latent(grids, predicted, context_size=context_size)
        action_context = torch.cat(
            (conditioned_actions[:, -context_size:], action_raw4[:, None]),
            dim=1,
        )[:, -context_size:]
        valid = torch.cat(
            (
                valid[:, -context_size:],
                torch.ones((batch * modes, 1), dtype=torch.bool, device=valid.device),
            ),
            dim=1,
        )[:, -context_size:]
        current = predicted
        previous = action_raw4

    endpoints = torch.stack(endpoint_rows, dim=1).reshape(
        batch, modes, horizon, spatial, latent
    )
    actions = torch.stack(action_rows, dim=1).reshape(batch, modes, horizon)
    consistency = torch.stack(consistency_rows, dim=1).reshape(batch, modes, horizon)
    ranking = rank_all_candidates(endpoints, goal_grid, log_mass)
    rows = torch.arange(batch, device=actions.device)
    first_action = actions[rows, ranking.winner_k, 0]
    return CandidateRollout(
        endpoints=endpoints,
        actions=actions,
        consistency=consistency,
        goal_distance=ranking.goal_distance,
        ordered_indices=ranking.ordered_indices,
        winner_k=ranking.winner_k,
        winner_h=ranking.winner_h,
        first_action=first_action,
    )


def goal_match_action(
    *,
    actor_step,
    current_grid: Tensor,
    goal_grid: Tensor,
    previous_raw4: Tensor,
) -> Tensor:
    """Use the same G without masking STOP for the explicit goal-match branch."""

    if current_grid.shape != goal_grid.shape or current_grid.ndim != 3:
        raise ValueError("current and goal grids must share [batch,spatial,latent]")
    if previous_raw4.shape != (current_grid.shape[0], 4):
        raise ValueError("previous_raw4 must be [batch,4]")
    for name, value in (
        ("current_grid", current_grid),
        ("goal_grid", goal_grid),
        ("previous_raw4", previous_raw4),
    ):
        _require_finite(name, value)
    logits = actor_step(current_grid, goal_grid - current_grid, previous_raw4)
    if logits.shape != (current_grid.shape[0], 4):
        raise ValueError("actor_step must return [batch,4]")
    _require_finite("goal-match logits", logits)
    return logits.argmax(dim=-1)
