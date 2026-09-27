"""Pure Context4 Full, NoF and Q-preserving NoG deployment rankers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor

from j2j import context4_rollout as _production_rollout
from j2j.context4 import CONTEXT_SIZE, append_context_latent


@dataclass(frozen=True)
class RankerResult:
    proposal_tape: Tensor | None
    log_mass: Tensor
    mode_indices: Tensor
    endpoints: Tensor
    actions: Tensor
    action_prefixes: Tensor | None
    consistency: Tensor | None
    goal_distance: Tensor
    ordered_indices: Tensor
    winner_k: Tensor
    winner_h: Tensor
    first_action: Tensor
    counters: Mapping[str, int]


def _require_finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _positive_active(value: Any, maximum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    if value > maximum:
        raise ValueError(f"{name} cannot exceed the checkpoint model identity")
    return value


def _validate_proposal(
    proposal_tape: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    *,
    active_k: int,
    active_h: int,
) -> tuple[int, int, int, int, int]:
    if not isinstance(proposal_tape, Tensor) or proposal_tape.ndim != 5:
        raise ValueError("proposal_tape must have shape [batch,modes,horizon,spatial,latent]")
    batch, modes, horizon, spatial, latent = proposal_tape.shape
    if min(batch, modes, horizon, spatial, latent) <= 0:
        raise ValueError("proposal_tape axes must be nonempty")
    if log_mass.shape != (batch, modes):
        raise ValueError("log_mass shape is inconsistent with proposal_tape")
    if goal_grid.shape != (batch, spatial, latent):
        raise ValueError("goal_grid shape is inconsistent with proposal_tape")
    _positive_active(active_k, modes, "active_k")
    _positive_active(active_h, horizon, "active_h")
    for name, value in (
        ("proposal_tape", proposal_tape),
        ("log_mass", log_mass),
        ("goal_grid", goal_grid),
    ):
        _require_finite(name, value)
    if len({proposal_tape.device, log_mass.device, goal_grid.device}) != 1:
        raise ValueError("proposal tensors must share one device")
    return batch, modes, horizon, spatial, latent


def _select_modes(
    proposal_tape: Tensor,
    log_mass: Tensor,
    *,
    active_k: int,
    active_h: int,
) -> tuple[Tensor, Tensor, Tensor]:
    order = torch.argsort(-log_mass.float(), dim=1, stable=True)
    mode_indices = order[:, :active_k]
    batch_index = torch.arange(proposal_tape.shape[0], device=proposal_tape.device)[:, None]
    tape = proposal_tape[batch_index, mode_indices, :active_h].clone()
    mass = log_mass.gather(1, mode_indices).clone()
    return tape, mass, mode_indices


def _raw4(action_ids: Tensor, *, dtype: torch.dtype) -> Tensor:
    result = torch.zeros((*action_ids.shape, 4), dtype=dtype, device=action_ids.device)
    return result.scatter_(-1, action_ids[..., None], 1.0)


def _globalize_ranking(ranking, mode_indices: Tensor) -> tuple[Tensor, Tensor]:
    local_modes = ranking.ordered_indices[..., 0]
    global_modes = mode_indices.gather(1, local_modes)
    ordered = torch.stack((global_modes, ranking.ordered_indices[..., 1]), dim=-1)
    winner = mode_indices.gather(1, ranking.winner_k[:, None]).squeeze(1)
    return ordered, winner


def _validate_context(
    factual_context: Tensor,
    factual_outgoing_raw4: Tensor,
    previous_raw4: Tensor,
    context_valid: Tensor,
    *,
    batch: int,
    spatial: int,
    latent: int,
) -> int:
    if factual_context.ndim != 4 or factual_context.shape[0] != batch:
        raise ValueError("factual_context must be [batch,context,spatial,latent]")
    if factual_context.shape[2:] != (spatial, latent):
        raise ValueError("factual_context spatial coordinates do not match proposal")
    context = factual_context.shape[1]
    if not 1 <= context <= CONTEXT_SIZE:
        raise ValueError("factual context length must lie in [1,4]")
    if factual_outgoing_raw4.shape != (batch, context, 4):
        raise ValueError("factual_outgoing_raw4 must align with factual context")
    if previous_raw4.shape != (batch, 4):
        raise ValueError("previous_raw4 must be [batch,4]")
    if context_valid.shape != (batch, context) or context_valid.dtype != torch.bool:
        raise ValueError("context_valid must be bool [batch,context]")
    if not bool(context_valid[:, -1].all()):
        raise ValueError("current factual slot must be valid")
    for name, value in (
        ("factual_context", factual_context),
        ("factual_outgoing_raw4", factual_outgoing_raw4),
        ("previous_raw4", previous_raw4),
    ):
        _require_finite(name, value)
    devices = {
        factual_context.device,
        factual_outgoing_raw4.device,
        previous_raw4.device,
        context_valid.device,
    }
    if len(devices) != 1:
        raise ValueError("context tensors must share one device")
    return context


def rank_goal_gf(
    *, goal_grid, factual_context, factual_outgoing_raw4, previous_raw4,
    context_valid, actor_step, forward_step, active_k, active_h, generator,
    epsilon=1e-6,
):
    """Sample goal-conditioned G/F branches without a Q proposal."""
    for name, value in (("active_k", active_k), ("active_h", active_h)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError("epsilon must be positive and finite")
    if not isinstance(generator, torch.Generator):
        raise TypeError("goal_gf requires a private torch.Generator")
    if not isinstance(goal_grid, Tensor) or goal_grid.ndim != 3 or min(goal_grid.shape) < 1:
        raise ValueError("goal_grid must be nonempty [batch,spatial,latent]")
    _require_finite("goal_grid", goal_grid)
    batch, spatial, latent = goal_grid.shape
    length = _validate_context(factual_context, factual_outgoing_raw4, previous_raw4,
        context_valid, batch=batch, spatial=spatial, latent=latent)
    if goal_grid.device != factual_context.device:
        raise ValueError("goal and context must share one device")
    grids = factual_context[:, None].expand(-1, active_k, -1, -1, -1).reshape(
        batch * active_k, length, spatial, latent).clone()
    outgoing = factual_outgoing_raw4[:, None].expand(-1, active_k, -1, -1).reshape(
        batch * active_k, length, 4).clone()
    valid = context_valid[:, None].expand(-1, active_k, -1).reshape(
        batch * active_k, length).clone()
    previous = previous_raw4[:, None].expand(-1, active_k, -1).reshape(
        batch * active_k, 4).clone()
    goal = goal_grid[:, None].expand(-1, active_k, -1, -1).reshape(
        batch * active_k, spatial, latent)
    endpoint_rows, action_rows = [], []
    for _ in range(active_h):
        current = grids[:, -1]
        logits = actor_step(current, goal - current, previous)
        if not isinstance(logits, Tensor) or logits.shape != (batch * active_k, 4):
            raise ValueError("actor_step must return [batch*active_k,4]")
        _require_finite("actor logits", logits)
        if logits.device != grids.device:
            raise ValueError("actor logits must share the context device")
        motion_logits = logits.float().clone()
        motion_logits[:, 0] = -torch.inf
        action = torch.multinomial(motion_logits.softmax(-1).to(generator.device),
            1, generator=generator).squeeze(-1).to(grids.device)
        raw = _raw4(action, dtype=outgoing.dtype)
        conditioned = outgoing.clone()
        conditioned[:, -1] = raw
        predicted = forward_step(grids, conditioned, valid)
        if not isinstance(predicted, Tensor) or predicted.shape != (batch * active_k, spatial, latent):
            raise ValueError("forward_step must return [batch*active_k,spatial,latent]")
        _require_finite("forward prediction", predicted)
        if predicted.device != grids.device:
            raise ValueError("forward prediction must share the context device")
        endpoint_rows.append(predicted)
        action_rows.append(action)
        grids = append_context_latent(grids, predicted, context_size=CONTEXT_SIZE)
        outgoing = torch.cat((conditioned, raw[:, None]), dim=1)[:, -CONTEXT_SIZE:]
        valid = torch.cat((valid, torch.ones((batch * active_k, 1),
            dtype=torch.bool, device=valid.device)), dim=1)[:, -CONTEXT_SIZE:]
        previous = raw
    endpoints = torch.stack(endpoint_rows, dim=1).reshape(batch, active_k, active_h, spatial, latent)
    actions = torch.stack(action_rows, dim=1).reshape(batch, active_k, active_h)
    distance = (endpoints.float() - goal_grid[:, None, None].float()).abs().mean((-1, -2))
    initial = (factual_context[:, -1].float() - goal_grid.float()).abs().mean((-1, -2))
    score = distance[:, :, -1] / initial[:, None].clamp_min(epsilon)
    _require_finite("goal distance", distance)
    _require_finite("branch score", score)
    order = torch.argsort(score, dim=1, stable=True)
    modes = torch.arange(active_k, device=goal_grid.device).expand(batch, -1)
    winner = order[:, 0]
    return RankerResult(proposal_tape=None,
        log_mass=goal_grid.new_full((batch, active_k), -math.log(active_k)),
        mode_indices=modes, endpoints=endpoints, actions=actions, action_prefixes=None,
        consistency=None, goal_distance=distance,
        ordered_indices=torch.stack((order, torch.full_like(order, active_h - 1)), dim=-1),
        winner_k=winner, winner_h=torch.full_like(winner, active_h - 1),
        first_action=actions[torch.arange(batch, device=goal_grid.device), winner, 0],
        counters=dict(q_calls=0, g_calls=active_h, f_calls=active_h,
            g_rows=batch * active_k * active_h, f_rows=batch * active_k * active_h,
            candidate_rows=batch * active_k))


def rank_full(
    *,
    proposal_tape: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    factual_context: Tensor,
    factual_outgoing_raw4: Tensor,
    previous_raw4: Tensor,
    context_valid: Tensor,
    actor_step,
    forward_step,
    active_k: int,
    active_h: int,
) -> RankerResult:
    """Call the production rollout/ranker after the disposable action bridge."""

    batch, _modes, _horizon, spatial, latent = _validate_proposal(
        proposal_tape, log_mass, goal_grid, active_k=active_k, active_h=active_h
    )
    _validate_context(
        factual_context,
        factual_outgoing_raw4,
        previous_raw4,
        context_valid,
        batch=batch,
        spatial=spatial,
        latent=latent,
    )
    tape, mass, mode_indices = _select_modes(
        proposal_tape, log_mass, active_k=active_k, active_h=active_h
    )
    rollout_outgoing = factual_outgoing_raw4.clone()
    rollout_outgoing[:, -1] = previous_raw4
    rollout = _production_rollout.rollout_candidates(
        proposal_tape=tape,
        log_mass=mass,
        goal_grid=goal_grid,
        factual_context=factual_context,
        outgoing_raw4=rollout_outgoing,
        context_valid=context_valid,
        actor_step=actor_step,
        forward_step=forward_step,
        context_size=CONTEXT_SIZE,
    )
    ordered, winner_k = _globalize_ranking(rollout, mode_indices)
    return RankerResult(
        proposal_tape=tape,
        log_mass=mass,
        mode_indices=mode_indices,
        endpoints=rollout.endpoints,
        actions=rollout.actions,
        action_prefixes=None,
        consistency=rollout.consistency,
        goal_distance=rollout.goal_distance,
        ordered_indices=ordered,
        winner_k=winner_k,
        winner_h=rollout.winner_h,
        first_action=rollout.first_action,
        counters={
            "q_calls": 0,
            "g_calls": active_h,
            "g_rows": batch * active_k * active_h,
            "f_calls": active_h,
            "f_rows": batch * active_k * active_h,
            "candidate_rows": batch * active_k * active_h,
        },
    )


def rank_no_f(
    *,
    proposal_tape: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    current_grid: Tensor,
    previous_raw4: Tensor,
    actor_step,
    active_k: int,
    active_h: int,
) -> RankerResult:
    """Decode Q with G while scoring Q endpoints directly and never calling F."""

    batch, _modes, _horizon, spatial, latent = _validate_proposal(
        proposal_tape, log_mass, goal_grid, active_k=active_k, active_h=active_h
    )
    if current_grid.shape != (batch, spatial, latent):
        raise ValueError("current_grid shape is inconsistent with proposal_tape")
    if previous_raw4.shape != (batch, 4):
        raise ValueError("previous_raw4 must be [batch,4]")
    _require_finite("current_grid", current_grid)
    _require_finite("previous_raw4", previous_raw4)
    if len({proposal_tape.device, current_grid.device, previous_raw4.device}) != 1:
        raise ValueError("NoF tensors must share one device")
    tape, mass, mode_indices = _select_modes(
        proposal_tape, log_mass, active_k=active_k, active_h=active_h
    )
    current = current_grid[:, None].expand(-1, active_k, -1, -1).reshape(
        batch * active_k, spatial, latent
    )
    previous = previous_raw4[:, None].expand(-1, active_k, -1).reshape(
        batch * active_k, 4
    ).clone()
    flat_tape = tape.reshape(batch * active_k, active_h, spatial, latent)
    actions: list[Tensor] = []
    for step in range(active_h):
        target = flat_tape[:, step]
        logits = actor_step(current, target - current, previous)
        if not isinstance(logits, Tensor) or logits.shape != (batch * active_k, 4):
            raise ValueError("actor_step must return [batch*active_k,4]")
        _require_finite("actor logits", logits)
        motion_logits = logits.clone()
        motion_logits[:, 0] = -torch.inf
        action = motion_logits.argmax(dim=-1)
        if not bool(((action >= 1) & (action <= 3)).all()):
            raise RuntimeError("ordinary NoF candidates must choose motion actions")
        actions.append(action)
        current = target
        previous = _raw4(action, dtype=previous_raw4.dtype)
    action_tensor = torch.stack(actions, dim=1).reshape(batch, active_k, active_h)
    ranking = _production_rollout.rank_all_candidates(tape, goal_grid, mass)
    ordered, winner_k = _globalize_ranking(ranking, mode_indices)
    rows = torch.arange(batch, device=tape.device)
    first_action = action_tensor[rows, ranking.winner_k, 0]
    return RankerResult(
        proposal_tape=tape,
        log_mass=mass,
        mode_indices=mode_indices,
        endpoints=tape,
        actions=action_tensor,
        action_prefixes=None,
        consistency=None,
        goal_distance=ranking.goal_distance,
        ordered_indices=ordered,
        winner_k=winner_k,
        winner_h=ranking.winner_h,
        first_action=first_action,
        counters={
            "q_calls": 0,
            "g_calls": active_h,
            "g_rows": batch * active_k * active_h,
            "g_decision_calls": 1,
            "g_logging_calls": active_h - 1,
            "f_calls": 0,
            "f_rows": 0,
            "candidate_rows": batch * active_k * active_h,
        },
    )


def rank_proposal_only(
    *,
    proposal_tape: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    current_grid: Tensor,
    previous_raw4: Tensor,
    actor_step,
    active_k: int,
    active_h: int,
) -> RankerResult:
    """Form complete actions from Q transitions and rank final Q endpoints only."""

    batch, _modes, _horizon, spatial, latent = _validate_proposal(
        proposal_tape, log_mass, goal_grid, active_k=active_k, active_h=active_h
    )
    if current_grid.shape != (batch, spatial, latent):
        raise ValueError("current_grid shape is inconsistent with proposal_tape")
    if previous_raw4.shape != (batch, 4):
        raise ValueError("previous_raw4 must be [batch,4]")
    _require_finite("current_grid", current_grid)
    _require_finite("previous_raw4", previous_raw4)
    if len({proposal_tape.device, current_grid.device, previous_raw4.device}) != 1:
        raise ValueError("Proposal-only tensors must share one device")
    tape, mass, mode_indices = _select_modes(
        proposal_tape, log_mass, active_k=active_k, active_h=active_h
    )
    current = current_grid[:, None].expand(-1, active_k, -1, -1).reshape(
        batch * active_k, spatial, latent
    )
    previous = previous_raw4[:, None].expand(-1, active_k, -1).reshape(
        batch * active_k, 4
    ).clone()
    flat_tape = tape.reshape(batch * active_k, active_h, spatial, latent)
    actions: list[Tensor] = []
    for step in range(active_h):
        target = flat_tape[:, step]
        logits = actor_step(current, target - current, previous)
        if not isinstance(logits, Tensor) or logits.shape != (batch * active_k, 4):
            raise ValueError("actor_step must return [batch*active_k,4]")
        _require_finite("actor logits", logits)
        motion_logits = logits.clone()
        motion_logits[:, 0] = -torch.inf
        action = motion_logits.argmax(dim=-1)
        if not bool(((action >= 1) & (action <= 3)).all()):
            raise RuntimeError("ordinary Proposal-only candidates must choose motion actions")
        actions.append(action)
        current = target
        previous = _raw4(action, dtype=previous_raw4.dtype)
    action_tensor = torch.stack(actions, dim=1).reshape(batch, active_k, active_h)
    terminal = _production_rollout.rank_all_candidates(tape[:, :, -1:], goal_grid, mass)
    ordered, winner_k = _globalize_ranking(terminal, mode_indices)
    ordered = torch.stack(
        (ordered[..., 0], torch.full_like(ordered[..., 1], active_h - 1)), dim=-1
    )
    rows = torch.arange(batch, device=tape.device)
    first_action = action_tensor[rows, terminal.winner_k, 0]
    goal_distance = (
        tape.float() - goal_grid[:, None, None].float()
    ).abs().mean(dim=(-1, -2))
    return RankerResult(
        proposal_tape=tape,
        log_mass=mass,
        mode_indices=mode_indices,
        endpoints=tape,
        actions=action_tensor,
        action_prefixes=None,
        consistency=None,
        goal_distance=goal_distance,
        ordered_indices=ordered,
        winner_k=winner_k,
        winner_h=torch.full(
            (batch,), active_h - 1, dtype=torch.long, device=tape.device
        ),
        first_action=first_action,
        counters={
            "q_calls": 0,
            "g_calls": active_h,
            "g_rows": batch * active_k * active_h,
            "f_calls": 0,
            "f_rows": 0,
            "candidate_rows": batch * active_k,
        },
    )


def rank_posthoc_f(
    *,
    proposal_tape: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    factual_context: Tensor,
    factual_outgoing_raw4: Tensor,
    previous_raw4: Tensor,
    context_valid: Tensor,
    actor_step,
    forward_step,
    active_k: int,
    active_h: int,
) -> RankerResult:
    """Freeze Q-to-G actions first, then let F evaluate that fixed sequence."""

    batch, _modes, _horizon, spatial, latent = _validate_proposal(
        proposal_tape, log_mass, goal_grid, active_k=active_k, active_h=active_h
    )
    _validate_context(
        factual_context,
        factual_outgoing_raw4,
        previous_raw4,
        context_valid,
        batch=batch,
        spatial=spatial,
        latent=latent,
    )
    tape, mass, mode_indices = _select_modes(
        proposal_tape, log_mass, active_k=active_k, active_h=active_h
    )

    # Phase 1: form the entire action sequence from Q transitions only.
    q_current = factual_context[:, -1][:, None].expand(
        -1, active_k, -1, -1
    ).reshape(batch * active_k, spatial, latent)
    previous = previous_raw4[:, None].expand(-1, active_k, -1).reshape(
        batch * active_k, 4
    ).clone()
    flat_tape = tape.reshape(batch * active_k, active_h, spatial, latent)
    action_rows: list[Tensor] = []
    for step in range(active_h):
        target = flat_tape[:, step]
        logits = actor_step(q_current, target - q_current, previous)
        if not isinstance(logits, Tensor) or logits.shape != (batch * active_k, 4):
            raise ValueError("actor_step must return [batch*active_k,4]")
        _require_finite("actor logits", logits)
        motion_logits = logits.clone()
        motion_logits[:, 0] = -torch.inf
        action = motion_logits.argmax(dim=-1)
        if not bool(((action >= 1) & (action <= 3)).all()):
            raise RuntimeError("ordinary Post-hoc candidates must choose motion actions")
        action_rows.append(action)
        q_current = target
        previous = _raw4(action, dtype=previous_raw4.dtype)
    flat_actions = torch.stack(action_rows, dim=1)

    # Phase 2: F sees the frozen action sequence. Its outputs never return to G.
    context_length = factual_context.shape[1]
    grids = factual_context[:, None].expand(
        batch, active_k, context_length, spatial, latent
    ).reshape(batch * active_k, context_length, spatial, latent).clone()
    action_context = factual_outgoing_raw4[:, None].expand(
        batch, active_k, context_length, 4
    ).reshape(batch * active_k, context_length, 4).clone()
    action_context[:, -1] = previous_raw4[:, None].expand(
        -1, active_k, -1
    ).reshape(batch * active_k, 4)
    valid = context_valid[:, None].expand(
        batch, active_k, context_length
    ).reshape(batch * active_k, context_length).clone()
    endpoint_rows: list[Tensor] = []
    consistency_rows: list[Tensor] = []
    for step in range(active_h):
        action = flat_actions[:, step]
        action_raw4 = _raw4(action, dtype=action_context.dtype)
        conditioned_actions = action_context.clone()
        conditioned_actions[:, -1] = action_raw4
        predicted = forward_step(grids, conditioned_actions, valid)
        if not isinstance(predicted, Tensor) or predicted.shape != (
            batch * active_k, spatial, latent
        ):
            raise ValueError("forward_step must return [batch*active_k,spatial,latent]")
        _require_finite("forward prediction", predicted)
        endpoint_rows.append(predicted)
        consistency_rows.append(
            (predicted.float() - flat_tape[:, step].float()).abs().mean(dim=(-1, -2))
        )
        grids = append_context_latent(grids, predicted, context_size=CONTEXT_SIZE)
        action_context = torch.cat(
            (conditioned_actions[:, -CONTEXT_SIZE:], action_raw4[:, None]), dim=1
        )[:, -CONTEXT_SIZE:]
        valid = torch.cat(
            (
                valid[:, -CONTEXT_SIZE:],
                torch.ones(
                    (batch * active_k, 1), dtype=torch.bool, device=valid.device
                ),
            ),
            dim=1,
        )[:, -CONTEXT_SIZE:]

    endpoints = torch.stack(endpoint_rows, dim=1).reshape(
        batch, active_k, active_h, spatial, latent
    )
    actions = flat_actions.reshape(batch, active_k, active_h)
    consistency = torch.stack(consistency_rows, dim=1).reshape(
        batch, active_k, active_h
    )
    ranking = _production_rollout.rank_all_candidates(endpoints, goal_grid, mass)
    ordered, winner_k = _globalize_ranking(ranking, mode_indices)
    rows = torch.arange(batch, device=tape.device)
    first_action = actions[rows, ranking.winner_k, 0]
    return RankerResult(
        proposal_tape=tape,
        log_mass=mass,
        mode_indices=mode_indices,
        endpoints=endpoints,
        actions=actions,
        action_prefixes=None,
        consistency=consistency,
        goal_distance=ranking.goal_distance,
        ordered_indices=ordered,
        winner_k=winner_k,
        winner_h=ranking.winner_h,
        first_action=first_action,
        counters={
            "q_calls": 0,
            "g_calls": active_h,
            "g_rows": batch * active_k * active_h,
            "f_calls": active_h,
            "f_rows": batch * active_k * active_h,
            "candidate_rows": batch * active_k * active_h,
        },
    )


def rank_f_feedback_q(
    *,
    initial_intent: Tensor,
    trajectory_latent: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    factual_context: Tensor,
    factual_outgoing_raw4: Tensor,
    previous_raw4: Tensor,
    context_valid: Tensor,
    q_step,
    actor_step,
    forward_step,
    active_k: int,
    active_h: int,
) -> RankerResult:
    """Feed every F consequence to both the next Q cell and the next G call."""

    if initial_intent.ndim != 4:
        raise ValueError("initial_intent must be [batch,modes,spatial,latent]")
    batch, modes, spatial, latent = initial_intent.shape
    if trajectory_latent.ndim != 3 or trajectory_latent.shape[:2] != (batch, modes):
        raise ValueError("trajectory_latent must align with initial intent modes")
    if log_mass.shape != (batch, modes) or goal_grid.shape != (batch, spatial, latent):
        raise ValueError("feedback-Q proposal axes disagree")
    _positive_active(active_k, modes, "active_k")
    if type(active_h) is not int or active_h <= 0:
        raise ValueError("active_h must be a positive integer")
    _validate_context(
        factual_context,
        factual_outgoing_raw4,
        previous_raw4,
        context_valid,
        batch=batch,
        spatial=spatial,
        latent=latent,
    )
    for name, value in (
        ("initial_intent", initial_intent),
        ("trajectory_latent", trajectory_latent),
        ("log_mass", log_mass),
        ("goal_grid", goal_grid),
    ):
        _require_finite(name, value)

    order = torch.argsort(-log_mass.float(), dim=1, stable=True)[:, :active_k]
    rows = torch.arange(batch, device=initial_intent.device)[:, None]
    intent = initial_intent[rows, order]
    z = trajectory_latent[rows, order]
    mass = log_mass.gather(1, order)
    current = factual_context[:, -1][:, None].expand(-1, active_k, -1, -1).clone()
    previous = previous_raw4[:, None].expand(-1, active_k, -1).clone()

    context_length = factual_context.shape[1]
    grids = factual_context[:, None].expand(
        batch, active_k, context_length, spatial, latent
    ).reshape(batch * active_k, context_length, spatial, latent).clone()
    action_context = factual_outgoing_raw4[:, None].expand(
        batch, active_k, context_length, 4
    ).reshape(batch * active_k, context_length, 4).clone()
    valid = context_valid[:, None].expand(
        batch, active_k, context_length
    ).reshape(batch * active_k, context_length).clone()

    intents: list[Tensor] = []
    endpoints: list[Tensor] = []
    actions: list[Tensor] = []
    consistency: list[Tensor] = []
    for step in range(active_h):
        if step:
            intent = q_step(current, z)
            if not isinstance(intent, Tensor) or intent.shape != current.shape:
                raise ValueError("q_step must return [batch,active_k,spatial,latent]")
            _require_finite("feedback Q intent", intent)
        intents.append(intent)
        flat_current = current.reshape(batch * active_k, spatial, latent)
        flat_intent = intent.reshape(batch * active_k, spatial, latent)
        flat_previous = previous.reshape(batch * active_k, 4)
        logits = actor_step(flat_current, flat_intent - flat_current, flat_previous)
        if not isinstance(logits, Tensor) or logits.shape != (batch * active_k, 4):
            raise ValueError("actor_step must return [batch*active_k,4]")
        _require_finite("actor logits", logits)
        motion_logits = logits.clone()
        motion_logits[:, 0] = -torch.inf
        action = motion_logits.argmax(dim=-1)
        if not bool(((action >= 1) & (action <= 3)).all()):
            raise RuntimeError("ordinary feedback-Q candidates must choose motion actions")
        action_raw4 = _raw4(action, dtype=action_context.dtype)
        conditioned = action_context.clone()
        conditioned[:, -1] = action_raw4
        predicted = forward_step(grids, conditioned, valid)
        if not isinstance(predicted, Tensor) or predicted.shape != (
            batch * active_k, spatial, latent
        ):
            raise ValueError("forward_step must return [batch*active_k,spatial,latent]")
        _require_finite("forward prediction", predicted)
        predicted_shaped = predicted.reshape(batch, active_k, spatial, latent)
        endpoints.append(predicted_shaped)
        actions.append(action.reshape(batch, active_k))
        consistency.append(
            (predicted_shaped.float() - intent.float()).abs().mean(dim=(-1, -2))
        )
        current = predicted_shaped
        previous = action_raw4.reshape(batch, active_k, 4)
        grids = append_context_latent(grids, predicted, context_size=CONTEXT_SIZE)
        action_context = torch.cat(
            (conditioned[:, -CONTEXT_SIZE:], action_raw4[:, None]), dim=1
        )[:, -CONTEXT_SIZE:]
        valid = torch.cat(
            (valid[:, -CONTEXT_SIZE:], torch.ones(
                (batch * active_k, 1), dtype=torch.bool, device=valid.device)), dim=1
        )[:, -CONTEXT_SIZE:]

    intent_tape = torch.stack(intents, dim=2)
    endpoint_tape = torch.stack(endpoints, dim=2)
    action_tape = torch.stack(actions, dim=2)
    consistency_tape = torch.stack(consistency, dim=2)
    ranking = _production_rollout.rank_all_candidates(endpoint_tape, goal_grid, mass)
    ordered, winner_k = _globalize_ranking(ranking, order)
    first_action = action_tape[
        torch.arange(batch, device=action_tape.device), ranking.winner_k, 0
    ]
    return RankerResult(
        proposal_tape=intent_tape,
        log_mass=mass,
        mode_indices=order,
        endpoints=endpoint_tape,
        actions=action_tape,
        action_prefixes=None,
        consistency=consistency_tape,
        goal_distance=ranking.goal_distance,
        ordered_indices=ordered,
        winner_k=winner_k,
        winner_h=ranking.winner_h,
        first_action=first_action,
        counters={
            "q_calls": active_h,
            "g_calls": active_h,
            "g_rows": batch * active_k * active_h,
            "f_calls": active_h,
            "f_rows": batch * active_k * active_h,
            "candidate_rows": batch * active_k * active_h,
        },
    )


def rank_no_g(
    *,
    proposal_tape: Tensor,
    log_mass: Tensor,
    goal_grid: Tensor,
    factual_context: Tensor,
    factual_outgoing_raw4: Tensor,
    previous_raw4: Tensor,
    context_valid: Tensor,
    forward_step,
    active_k: int,
    active_h: int,
) -> RankerResult:
    """Match each Q prefix against one shared exhaustive motion-only F tree."""

    batch, _modes, _horizon, spatial, latent = _validate_proposal(
        proposal_tape, log_mass, goal_grid, active_k=active_k, active_h=active_h
    )
    context_length = _validate_context(
        factual_context,
        factual_outgoing_raw4,
        previous_raw4,
        context_valid,
        batch=batch,
        spatial=spatial,
        latent=latent,
    )
    tape, mass, mode_indices = _select_modes(
        proposal_tape, log_mass, active_k=active_k, active_h=active_h
    )

    parent_grids = factual_context[:, None].clone()
    parent_actions = factual_outgoing_raw4[:, None].clone()
    parent_valid = context_valid[:, None].clone()
    parent_prefixes = torch.empty((batch, 1, 0), dtype=torch.int64, device=tape.device)
    parent_paths = tape.new_empty((batch, 1, 0, spatial, latent))
    level_prefixes: list[Tensor] = []
    level_paths: list[Tensor] = []
    f_rows = 0

    for _depth in range(active_h):
        parents = parent_grids.shape[1]
        action_ids = torch.arange(1, 4, dtype=torch.int64, device=tape.device)
        action_ids = action_ids.view(1, 1, 3).expand(batch, parents, 3)
        flat_actions = action_ids.reshape(-1)
        grids = parent_grids[:, :, None].expand(
            batch, parents, 3, parent_grids.shape[2], spatial, latent
        ).reshape(batch * parents * 3, parent_grids.shape[2], spatial, latent).clone()
        outgoing = parent_actions[:, :, None].expand(
            batch, parents, 3, parent_actions.shape[2], 4
        ).reshape(batch * parents * 3, parent_actions.shape[2], 4).clone()
        valid = parent_valid[:, :, None].expand(
            batch, parents, 3, parent_valid.shape[2]
        ).reshape(batch * parents * 3, parent_valid.shape[2]).clone()
        raw = _raw4(flat_actions, dtype=factual_outgoing_raw4.dtype)
        outgoing[:, -1] = raw
        prediction = forward_step(grids, outgoing, valid)
        if not isinstance(prediction, Tensor) or prediction.shape != (
            batch * parents * 3,
            spatial,
            latent,
        ):
            raise ValueError("forward_step must return [tree_rows,spatial,latent]")
        _require_finite("forward prediction", prediction)
        f_rows += int(prediction.shape[0])
        endpoint = prediction.reshape(batch, parents * 3, spatial, latent)

        prefixes = parent_prefixes[:, :, None].expand(
            batch, parents, 3, parent_prefixes.shape[-1]
        ).reshape(batch, parents * 3, parent_prefixes.shape[-1])
        prefixes = torch.cat((prefixes, action_ids.reshape(batch, parents * 3, 1)), dim=-1)
        paths = parent_paths[:, :, None].expand(
            batch, parents, 3, parent_paths.shape[2], spatial, latent
        ).reshape(batch, parents * 3, parent_paths.shape[2], spatial, latent)
        paths = torch.cat((paths, endpoint[:, :, None]), dim=2)
        level_prefixes.append(prefixes)
        level_paths.append(paths)

        next_grids = append_context_latent(grids, prediction, context_size=CONTEXT_SIZE)
        next_actions = torch.cat((outgoing[:, -CONTEXT_SIZE:], raw[:, None]), dim=1)[
            :, -CONTEXT_SIZE:
        ]
        next_valid = torch.cat(
            (
                valid[:, -CONTEXT_SIZE:],
                torch.ones((valid.shape[0], 1), dtype=torch.bool, device=valid.device),
            ),
            dim=1,
        )[:, -CONTEXT_SIZE:]
        parent_grids = next_grids.reshape(
            batch, parents * 3, next_grids.shape[1], spatial, latent
        )
        parent_actions = next_actions.reshape(batch, parents * 3, next_actions.shape[1], 4)
        parent_valid = next_valid.reshape(batch, parents * 3, next_valid.shape[1])
        parent_prefixes = prefixes
        parent_paths = paths
        context_length = min(CONTEXT_SIZE, context_length + 1)

    endpoints = tape.new_empty((batch, active_k, active_h, spatial, latent))
    action_prefixes = torch.zeros(
        (batch, active_k, active_h, active_h), dtype=torch.int64, device=tape.device
    )
    for horizon_index in range(active_h):
        paths = level_paths[horizon_index]
        prefixes = level_prefixes[horizon_index]
        target = tape[:, :, None, : horizon_index + 1]
        error = (
            paths[:, None].float() - target.float()
        ).abs().mean(dim=(-1, -2, -3))
        winner = error.argmin(dim=-1)
        batch_index = torch.arange(batch, device=tape.device)[:, None]
        selected_paths = paths[batch_index, winner]
        selected_prefixes = prefixes[batch_index, winner]
        endpoints[:, :, horizon_index] = selected_paths[:, :, -1]
        action_prefixes[:, :, horizon_index, : horizon_index + 1] = selected_prefixes

    ranking = _production_rollout.rank_all_candidates(endpoints, goal_grid, mass)
    ordered, winner_k = _globalize_ranking(ranking, mode_indices)
    rows = torch.arange(batch, device=tape.device)
    first_action = action_prefixes[rows, ranking.winner_k, ranking.winner_h, 0]
    return RankerResult(
        proposal_tape=tape,
        log_mass=mass,
        mode_indices=mode_indices,
        endpoints=endpoints,
        actions=action_prefixes[..., 0],
        action_prefixes=action_prefixes,
        consistency=None,
        goal_distance=ranking.goal_distance,
        ordered_indices=ordered,
        winner_k=winner_k,
        winner_h=ranking.winner_h,
        first_action=first_action,
        counters={
            "q_calls": 0,
            "g_calls": 0,
            "g_rows": 0,
            "f_calls": active_h,
            "f_rows": f_rows,
            "tree_nodes_per_batch": sum(3**depth for depth in range(1, active_h + 1)),
            "candidate_rows": batch * active_k * active_h,
        },
    )


__all__ = [
    "RankerResult",
    "rank_full",
    "rank_no_f",
    "rank_proposal_only",
    "rank_posthoc_f",
    "rank_no_g",
]
