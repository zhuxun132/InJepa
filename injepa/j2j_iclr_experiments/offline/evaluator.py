"""Concrete one-pass evaluator over one materialized released trajectory."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import json
from typing import Any

import torch
from torch import Tensor, nn

from j2j.context4 import CONTEXT_SIZE, append_context_latent
from j2j.context4_data import (
    JointBatch,
    PreparedJointTrajectory,
    TrajectoryDescriptor,
    materialize_joint_batch,
    prepare_joint_trajectory,
)
from j2j.context4_objective import q_to_g_logits, qg_marginal_nll
from j2j.context4_training import build_q_goal_views
from j2j.proposal.losses import proper_mixture_terms

from .metrics import cosine_distance
from .transforms import apply_context_suffix, fixed_token_permutation, mean_repeat_grid


@dataclass(frozen=True)
class MaterializedOfflineTrajectory:
    prepared: PreparedJointTrajectory
    batch: JointBatch
    building: str
    source: str
    trajectory: str
    goal_indices: tuple[int, ...]


def _batch_to(batch: JointBatch, device: torch.device) -> JointBatch:
    return JointBatch(
        **{
            field.name: (getattr(batch, field.name).to(device=device)
                         if isinstance(getattr(batch, field.name), Tensor)
                         else getattr(batch, field.name))
            for field in fields(JointBatch)
        }
    )


def materialize_offline_trajectory(
    item: object,
    *,
    horizon: int,
    device: str | torch.device,
    variant_contract: Any | None = None,
) -> MaterializedOfflineTrajectory:
    """Prepare and fully materialize one trajectory exactly once."""

    prepared = prepare_joint_trajectory(item)
    if variant_contract is not None:
        prepared = variant_contract.prepared_trajectory_transform(prepared)
    source_item = getattr(item, "source_item", None)
    canonical = getattr(source_item, "canonical_trajectory", None)
    trajectory_key = getattr(canonical, "canonical_trajectory_key", None)
    building = getattr(canonical, "scan_id", None)
    source = getattr(canonical, "source_id", None)
    if type(trajectory_key) is not bytes or not trajectory_key:
        raise TypeError("offline item lacks a canonical trajectory key")
    if not isinstance(building, str) or not building:
        raise TypeError("offline item lacks a canonical building/scan id")
    if not isinstance(source, str) or not source:
        raise TypeError("offline item lacks a canonical source id")
    transitions = int(prepared.action_ids.numel())
    descriptor = TrajectoryDescriptor(
        dataset_index=0,
        trajectory_key=trajectory_key,
        origin_count=transitions,
    )
    goal_plan, _counts = build_q_goal_views((descriptor,), horizon=horizon)
    pairs = goal_plan[0]
    origins = torch.arange(transitions, dtype=torch.int64)
    q_origins = torch.tensor([origin for origin, _goal in pairs], dtype=torch.int64)
    goals = torch.tensor([goal for _origin, goal in pairs], dtype=torch.int64)
    batch = materialize_joint_batch(
        prepared,
        origin_indices=origins,
        q_origin_indices=q_origins,
        goal_indices=goals,
        horizon=horizon,
        context_size=CONTEXT_SIZE,
        include_terminal_stop=True,
    )
    batch = replace(batch, q_sample_keys=tuple(
        json.dumps({"trajectory_key": trajectory_key.hex(), "origin": origin, "goal": goal},
                   sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        for origin, goal in pairs
    ))
    if variant_contract is not None:
        context = variant_contract.context_transform(
            {
                "context_grid": batch.context_grid,
                "context_incoming_raw4": batch.context_incoming_raw4,
                "context_outgoing_raw4": batch.context_outgoing_raw4,
                "context_age": batch.context_age,
                "context_type": batch.context_type,
                "context_valid": batch.context_valid,
            }
        )
        batch = replace(batch, **dict(context))
    return MaterializedOfflineTrajectory(
        prepared=prepared,
        batch=_batch_to(batch, torch.device(device)),
        building=building,
        source=source,
        trajectory=trajectory_key.hex(),
        goal_indices=tuple(int(goal) for goal in goals.tolist()),
    )


def _project_batch(batch: JointBatch, projection) -> JointBatch:
    context = projection(batch.context_grid)
    current = projection(batch.current_grid)
    next_grid = projection(batch.next_grid)
    q_goal = projection(batch.q_goal_grid)
    terminal = projection(batch.terminal_grid)
    terminal_goal = terminal[-1:].expand_as(current) if current.shape[0] else terminal[:0]
    return replace(
        batch,
        context_grid=context,
        current_grid=current,
        local_intent=next_grid - current,
        q_goal_grid=q_goal,
        goal_intent=terminal_goal - current,
        terminal_grid=terminal,
        terminal_intent=torch.zeros_like(terminal),
    )


def _goal_donor_rows(goal_indices: tuple[int, ...]) -> tuple[int, ...] | None:
    if len(set(goal_indices)) < 2:
        return None
    result: list[int] = []
    for index, goal in enumerate(goal_indices):
        for offset in range(1, len(goal_indices) + 1):
            candidate = (index + offset) % len(goal_indices)
            if goal_indices[candidate] != goal:
                result.append(candidate)
                break
        else:  # pragma: no cover - guarded by the distinct-goal check
            raise RuntimeError("distinct goal donor search failed")
    return tuple(result)


def _component_batch(
    batch: JointBatch,
    *,
    arm: str,
    component: str,
    permutation: Tensor,
    goal_indices: tuple[int, ...],
) -> tuple[JointBatch | None, str | None]:
    if arm == "baseline":
        return batch, None
    if arm.startswith("context_suffix_"):
        if component == "g":
            return batch, None
        length = int(arm.rsplit("_", 1)[1])
        context = apply_context_suffix(
            {
                "record_grid": batch.context_grid,
                "incoming_raw4": batch.context_incoming_raw4,
                "outgoing_raw4": batch.context_outgoing_raw4,
                "record_age": batch.context_age,
                "record_type": batch.context_type,
                "record_valid": batch.context_valid,
            },
            length=length,
        )
        return replace(
            batch,
            context_grid=context["record_grid"],
            context_incoming_raw4=context["incoming_raw4"],
            context_outgoing_raw4=context["outgoing_raw4"],
            context_age=context["record_age"],
            context_type=context["record_type"],
            context_valid=context["record_valid"],
        ), None
    if arm == "mean_repeat_input":
        return _project_batch(batch, mean_repeat_grid), None
    if arm == "fixed_permutation_input":
        return _project_batch(
            batch,
            lambda value: fixed_token_permutation(value, permutation),
        ), None
    if arm == "goal_permutation":
        if component != "q":
            return batch, None
        donors = _goal_donor_rows(goal_indices)
        if donors is None:
            return None, "fewer than two distinct legal goal views"
        donor_rows = torch.tensor(donors, dtype=torch.int64, device=batch.q_goal_grid.device)
        return replace(batch, q_goal_grid=batch.q_goal_grid.index_select(0, donor_rows)), None
    if arm == "intent_permutation":
        if component != "g":
            return batch, None
        if batch.current_grid.shape[0] < 2:
            return None, "fewer than two legal factual intents"
        return replace(
            batch,
            local_intent=batch.local_intent.roll(1, dims=0),
            goal_intent=batch.goal_intent.roll(1, dims=0),
        ), None
    if arm == "previous_action_permutation":
        if component != "g":
            return batch, None
        factual_previous = torch.cat(
            (batch.previous_raw4, batch.terminal_previous_raw4),
            dim=0,
        )
        if factual_previous.shape[0] < 2:
            return None, "fewer than two legal factual G states"
        permuted = factual_previous.roll(1, dims=0)
        transition_count = batch.previous_raw4.shape[0]
        return replace(
            batch,
            previous_raw4=permuted[:transition_count],
            terminal_previous_raw4=permuted[transition_count:],
        ), None
    if arm == "action_permutation":
        if component != "f":
            return batch, None
        if batch.current_grid.shape[0] < 2:
            return None, "fewer than two legal recorded actions"
        outgoing = batch.context_outgoing_raw4.clone()
        outgoing[:, -1] = batch.outgoing_raw4.roll(1, dims=0)
        return replace(batch, context_outgoing_raw4=outgoing), None
    raise ValueError(f"unknown offline arm {arm!r}")


def _metadata(
    trajectory: MaterializedOfflineTrajectory,
    *,
    arm: str,
    component: str,
    origin: int | None,
    view: str,
    horizon: int,
    donor_key: str | None = None,
) -> dict[str, Any]:
    return {
        "building": trajectory.building,
        "source": trajectory.source,
        "trajectory": trajectory.trajectory,
        "origin": origin,
        "view": view,
        "horizon": horizon,
        "arm": arm,
        "component": component,
        "donor_key": donor_key,
        "context_fill": (
            min(origin + 1, CONTEXT_SIZE) if type(origin) is int and origin >= 0 else None
        ),
        "available": True,
    }


def _proposal_forward(model: nn.Module, batch: JointBatch, *, samples: int | None = None,
                      horizon: int | None = None, sampling_namespace: str = "offline/v1"):
    q_rows = batch.q_origin_row
    incoming = model.embed_actions(batch.context_incoming_raw4.index_select(0, q_rows))
    facts = (
        batch.context_grid.index_select(0, q_rows),
        incoming.detach().float(),
        batch.context_age.index_select(0, q_rows),
        batch.context_type.index_select(0, q_rows),
        batch.context_valid.index_select(0, q_rows),
        batch.q_goal_grid,
    )
    if bool(getattr(model.proposal, "is_stochastic", False)):
        if len(batch.q_sample_keys) != batch.q_goal_grid.shape[0]:
            raise ValueError("offline stochastic Q requires one key per factual occurrence")
        return model.proposal.sample_prior(
            *facts, samples=model.proposal.default_samples if samples is None else samples,
            horizon=getattr(model.proposal, "default_horizon", model.proposal.trained_horizon) if horizon is None else horizon,
            sample_keys=batch.q_sample_keys, seed=model.proposal.global_seed,
            namespace=sampling_namespace)
    if samples is not None or horizon is not None:
        raise ValueError("fixed-mixture Q uses its checkpoint budgets before ranking")
    return model.proposal(*facts, batch.q_active_h)


def _q_rows(
    model: nn.Module,
    trajectory: MaterializedOfflineTrajectory,
    batch: JointBatch,
    *,
    arm: str,
    proposal,
    q_called: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if batch.q_goal_grid.shape[0] == 0:
        return [], {"q_calls": 0, "g_calls": 0, "f_calls": 0, "f_rows": 0}
    q_rows = batch.q_origin_row
    stochastic = bool(getattr(getattr(model, "proposal", None), "is_stochastic", False))
    observed_horizon = min(proposal.tape.shape[2], batch.q_target_grid.shape[1])
    label_valid = batch.q_active_h[:, :observed_horizon]
    target = batch.q_target_grid[:, :observed_horizon]
    scored_tape = proposal.tape[:, :, :observed_horizon]
    terms = proper_mixture_terms(
        scored_tape,
        proposal.log_mass,
        target,
        label_valid,
    )
    mask = label_valid[:, None, :, None, None]
    errors = (scored_tape.float() - target[:, None].float()).abs()
    denominator = (
        label_valid.sum(dim=1).float()
        * proposal.tape.shape[-2]
        * proposal.tape.shape[-1]
    )
    mode_error = errors.masked_fill(~mask, 0.0).sum(dim=(2, 3, 4)) / denominator[:, None]
    mass = proposal.log_mass.exp()
    top_mode = torch.argsort(-proposal.log_mass, dim=1, stable=True)[:, 0]
    qg_logits = q_to_g_logits(
        model.actor,
        model,
        batch.current_grid.index_select(0, q_rows),
        proposal.tape[:, :, 0],
        batch.previous_raw4.index_select(0, q_rows),
    )
    qg_nll = qg_marginal_nll(qg_logits, batch.action_ids.index_select(0, q_rows))
    q_count, modes = proposal.tape.shape[:2]
    horizon = observed_horizon
    plans = torch.zeros(
        (q_count, modes, horizon), dtype=torch.int64, device=proposal.tape.device
    )
    current = batch.current_grid.index_select(0, q_rows)[:, None].expand(
        -1, modes, -1, -1
    ).clone()
    previous = batch.previous_raw4.index_select(0, q_rows)[:, None].expand(
        -1, modes, -1
    ).clone()
    plan_g_calls = 1
    plan_g_rows = q_count * modes
    for step in range(horizon):
        active = label_valid[:, step]
        if not bool(active.any()):
            continue
        if step == 0:
            logits = qg_logits[active]
        else:
            active_count = int(active.sum())
            target = proposal.tape[active, :, step]
            selected_current = current[active]
            logits = model.actor_logits(
                selected_current.reshape(
                    active_count * modes, *selected_current.shape[2:]
                ),
                (target - selected_current).reshape(
                    active_count * modes, *selected_current.shape[2:]
                ),
                previous[active].reshape(active_count * modes, 4),
            ).reshape(active_count, modes, 4)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError("Q-plan G logits contain non-finite values")
            plan_g_calls += 1
            plan_g_rows += active_count * modes
        motion_logits = logits.clone()
        motion_logits[..., 0] = -torch.inf
        action = motion_logits.argmax(dim=-1)
        plans[active, :, step] = action
        current[active] = proposal.tape[active, :, step]
        next_previous = torch.zeros(
            (*action.shape, 4), dtype=previous.dtype, device=previous.device
        ).scatter_(-1, action[..., None], 1.0)
        previous[active] = next_previous
    output: list[dict[str, Any]] = []
    donor_rows = _goal_donor_rows(trajectory.goal_indices) if arm == "goal_permutation" else None
    for index in range(proposal.tape.shape[0]):
        responsibility = terms.responsibility[index]
        safe_log = torch.where(
            responsibility > 0,
            responsibility.log(),
            torch.zeros_like(responsibility),
        )
        entropy = -(responsibility * safe_log).sum()
        origin_row = int(batch.q_origin_row[index])
        origin = int(batch.origin_indices[origin_row])
        goal = trajectory.goal_indices[index]
        active_h = int(label_valid[index].sum())
        active_plan = plans[index, :, :active_h]
        distinct_plans = len({tuple(int(v) for v in row) for row in active_plan.tolist()})
        mode_actions = active_plan[:, 0]
        row = _metadata(
            trajectory,
            arm=arm,
            component="q",
            origin=origin,
            view="terminal" if goal == len(trajectory.prepared.action_ids) else "hashed",
            horizon=active_h,
            donor_key=(
                f"goal:{trajectory.goal_indices[donor_rows[index]]}"
                if donor_rows is not None
                else None
            ),
        )
        q_donor_key = row["donor_key"]
        row.update(
            {
                "goal_index": goal,
                "proper_mixture_nll": float(terms.occurrence_nll[index]),
                "oracle_mode_fit_mae": float(mode_error[index].min()),
                "mass_weighted_mae": float((mode_error[index] * mass[index]).sum()),
                "top_mass_mode_mae": float(mode_error[index, top_mode[index]]),
                "responsibility_entropy": float(entropy),
                "responsibility_effective_k": float(entropy.exp()),
                "qg_marginal_nll": float(qg_nll[index]),
                "distinct_first_action_count": len(set(int(v) for v in mode_actions.tolist())),
                "distinct_full_plan_count": distinct_plans,
            }
        )
        if stochastic:
            row["predictive_sample_nll"] = row.pop("proper_mixture_nll")
            row["prior_action_marginal_nll"] = row.pop("qg_marginal_nll")
        output.append(row)
        endpoint_element_denominator = int(
            proposal.tape.shape[-2] * proposal.tape.shape[-1]
        )
        for step in range(active_h):
            endpoint_mode_error = errors[index, :, step].mean(dim=(-2, -1))
            endpoint_row = _metadata(
                trajectory,
                arm=arm,
                component="q",
                origin=origin,
                view="endpoint",
                horizon=step + 1,
                donor_key=q_donor_key,
            )
            endpoint_row.update(
                {
                    "goal_index": goal,
                    "mode_count": modes,
                    "endpoint_element_denominator": endpoint_element_denominator,
                    "oracle_mode_endpoint_mae": float(endpoint_mode_error.min()),
                    "mass_weighted_endpoint_mae": float(
                        (endpoint_mode_error * mass[index]).sum()
                    ),
                    "top_mass_mode_endpoint_mae": float(
                        endpoint_mode_error[top_mode[index]]
                    ),
                }
            )
            output.append(endpoint_row)
        for mode in range(modes):
            mode_row = _metadata(
                trajectory,
                arm=arm,
                component="q",
                origin=origin,
                view="mode",
                horizon=active_h,
                donor_key=q_donor_key,
            )
            mode_row.update(
                {
                    "goal_index": goal,
                    "mode": mode,
                    "log_mass": float(proposal.log_mass[index, mode]),
                    "mass": float(mass[index, mode]),
                    "responsibility": float(responsibility[mode]),
                    "tape_mae": float(mode_error[index, mode]),
                    "first_action": int(active_plan[mode, 0]),
                    "plan_code": "-".join(str(int(v)) for v in active_plan[mode].tolist()),
                }
            )
            output.append(mode_row)
        for left in range(modes):
            for right in range(left + 1, modes):
                pair_distance = (
                    proposal.tape[index, left, :active_h].float()
                    - proposal.tape[index, right, :active_h].float()
                ).abs().mean()
                pair_row = _metadata(
                    trajectory,
                    arm=arm,
                    component="q",
                    origin=origin,
                    view="mode_pair",
                    horizon=active_h,
                    donor_key=q_donor_key,
                )
                pair_row.update(
                    {
                        "goal_index": goal,
                        "left_mode": left,
                        "right_mode": right,
                        "mode_pair_full_grid_distance": float(pair_distance),
                    }
                )
                output.append(pair_row)
    if stochastic:
        for row in output:
            row.update(sampling_kind="prior_mc", samples=modes,
                       requested_horizon=proposal.tape.shape[2],
                       sampling_namespace=proposal.sampling_namespace)
    return output, {
        "q_calls": int(q_called),
        "g_calls": plan_g_calls,
        "g_rows": plan_g_rows,
        "f_calls": 0,
        "f_rows": 0,
    }


def _ranking_rows(
    model: nn.Module,
    trajectory: MaterializedOfflineTrajectory,
    batch: JointBatch,
    *,
    proposal,
    k_values: tuple[int, ...],
    h_values: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from .rankers import rank_full, rank_no_f

    output: list[dict[str, Any]] = []
    counters = {"q_calls": 0, "g_calls": 0, "g_rows": 0, "f_calls": 0, "f_rows": 0}
    for active_k in k_values:
        for active_h in h_values:
            selected = torch.nonzero(batch.q_active_h.sum(dim=1) >= active_h).reshape(-1)
            if selected.numel() == 0:
                continue
            origins = batch.q_origin_row.index_select(0, selected)
            kwargs = {
                "proposal_tape": proposal.tape.index_select(0, selected),
                "log_mass": proposal.log_mass.index_select(0, selected),
                "goal_grid": batch.q_goal_grid.index_select(0, selected),
                "active_k": active_k,
                "active_h": active_h,
            }
            results = {
                "full": rank_full(
                    **kwargs,
                    factual_context=batch.context_grid.index_select(0, origins),
                    factual_outgoing_raw4=batch.context_outgoing_raw4.index_select(0, origins),
                    previous_raw4=batch.previous_raw4.index_select(0, origins),
                    context_valid=batch.context_valid.index_select(0, origins),
                    actor_step=model.actor_logits,
                    forward_step=model.predict_next_grid,
                ),
                "no_f_deploy": rank_no_f(
                    **kwargs,
                    current_grid=batch.current_grid.index_select(0, origins),
                    previous_raw4=batch.previous_raw4.index_select(0, origins),
                    actor_step=model.actor_logits,
                ),
            }
            factual_targets = batch.q_target_grid.index_select(0, selected)
            for deployment, result in results.items():
                for name in counters:
                    counters[name] += int(result.counters.get(name, 0))
                for row_index, q_index in enumerate(selected.tolist()):
                    origin = int(batch.origin_indices[int(batch.q_origin_row[q_index])])
                    for local_mode in range(active_k):
                        global_mode = int(result.mode_indices[row_index, local_mode])
                        for horizon_index in range(active_h):
                            factual_difference = (
                                result.endpoints[row_index, local_mode, horizon_index].float()
                                - factual_targets[row_index, horizon_index].float()
                            )
                            row = _metadata(
                                trajectory,
                                arm="baseline",
                                component="ranking",
                                origin=origin,
                                view=deployment,
                                horizon=horizon_index + 1,
                            )
                            row.update(
                                {
                                    "k_active": active_k,
                                    "h_active": active_h,
                                    "mode": global_mode,
                                    "log_mass": float(result.log_mass[row_index, local_mode]),
                                    "goal_distance": float(
                                        result.goal_distance[
                                            row_index, local_mode, horizon_index
                                        ]
                                    ),
                                    "factual_endpoint_mse": float(
                                        factual_difference.square().mean()
                                    ),
                                    "factual_endpoint_mae": float(
                                        factual_difference.abs().mean()
                                    ),
                                    "first_action": int(
                                        result.actions[row_index, local_mode, 0]
                                    ),
                                    "consistency": (
                                        float(
                                            result.consistency[
                                                row_index, local_mode, horizon_index
                                            ]
                                        )
                                        if result.consistency is not None
                                        else None
                                    ),
                                    "winner": bool(
                                        global_mode == int(result.winner_k[row_index])
                                        and horizon_index == int(result.winner_h[row_index])
                                    ),
                                }
                            )
                            output.append(row)
    return output, counters


def _g_rows(
    model: nn.Module,
    trajectory: MaterializedOfflineTrajectory,
    batch: JointBatch,
    *,
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    call_count = 0
    for view, current, intent, previous, labels, origins in (
        (
            "local",
            batch.current_grid,
            batch.local_intent,
            batch.previous_raw4,
            batch.action_ids,
            batch.origin_indices,
        ),
        (
            "goal",
            batch.current_grid,
            batch.goal_intent,
            batch.previous_raw4,
            batch.action_ids,
            batch.origin_indices,
        ),
        (
            "terminal",
            batch.terminal_grid,
            batch.terminal_intent,
            batch.terminal_previous_raw4,
            batch.terminal_action_ids,
            torch.tensor(
                [len(trajectory.prepared.action_ids)],
                dtype=torch.int64,
                device=batch.current_grid.device,
            ),
        ),
    ):
        # Terminal has no same-view factual intent donor: its only valid intent
        # is the zero STOP intent, so it is excluded from this corruption arm.
        if arm == "intent_permutation" and view == "terminal":
            continue
        if current.shape[0] == 0:
            continue
        logits = model.actor_logits(current, intent, previous).float()
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("G logits contain non-finite values")
        log_probability = torch.log_softmax(logits, dim=-1)
        probability = log_probability.exp()
        confidence, predicted = probability.max(dim=-1)
        nll = -log_probability.gather(1, labels[:, None]).squeeze(1)
        call_count += 1
        for index in range(current.shape[0]):
            donor_key = None
            if arm == "intent_permutation":
                donor_index = (index - 1) % len(origins)
                donor_key = f"origin:{int(origins[donor_index])}"
            elif arm == "previous_action_permutation":
                transition_count = int(batch.current_grid.shape[0])
                target_index = transition_count if view == "terminal" else index
                donor_index = (target_index - 1) % (transition_count + 1)
                donor_key = (
                    f"origin:{int(batch.origin_indices[donor_index])}"
                    if donor_index < transition_count
                    else f"terminal:{len(trajectory.prepared.action_ids)}"
                )
            row = _metadata(
                trajectory,
                arm=arm,
                component="g",
                origin=int(origins[index]),
                view=view,
                horizon=1,
                donor_key=donor_key,
            )
            row.update(
                {
                    "label": int(labels[index]),
                    "prediction": int(predicted[index]),
                    "confidence": float(confidence[index]),
                    "correct": bool(predicted[index] == labels[index]),
                    "nll": float(nll[index]),
                }
            )
            output.append(row)
    return output, {
        "q_calls": 0,
        "g_calls": call_count,
        "g_rows": len(output),
        "f_calls": 0,
        "f_rows": 0,
    }


def _f_rows(
    model: nn.Module,
    trajectory: MaterializedOfflineTrajectory,
    batch: JointBatch,
    *,
    arm: str,
    horizon: int,
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output: list[dict[str, Any]] = []
    f_calls = 0
    origin_rows = int(batch.current_grid.shape[0])
    prepared = trajectory.prepared
    transitions = int(prepared.action_ids.numel())
    for row_index in range(origin_rows):
        origin = int(batch.origin_indices[row_index])
        grids = batch.context_grid[row_index : row_index + 1].clone()
        outgoing = batch.context_outgoing_raw4[row_index : row_index + 1].clone()
        valid = batch.context_valid[row_index : row_index + 1].clone()
        factual_origin = prepared.grids[origin].to(device=grids.device)
        path_squared_error = 0.0
        path_absolute_error = 0.0
        path_element_count = 0
        for step in range(min(horizon, transitions - origin)):
            factual_action_index = origin + step
            donor_index = (
                (factual_action_index - 1) % transitions
                if arm == "action_permutation"
                else factual_action_index
            )
            action_raw4 = prepared.outgoing_raw4[donor_index : donor_index + 1].to(
                device=outgoing.device
            )
            conditioned = outgoing.clone()
            conditioned[:, -1] = action_raw4
            prediction = model.predict_next_grid(grids, conditioned, valid)
            f_calls += 1
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("F prediction contains non-finite values")
            target = prepared.grids[factual_action_index + 1 : factual_action_index + 2].to(
                device=prediction.device
            )
            difference = prediction.float() - target.float()
            prediction_squared_error = float(difference.square().sum())
            prediction_absolute_error = float(difference.abs().sum())
            path_squared_error += prediction_squared_error
            path_absolute_error += prediction_absolute_error
            path_element_count += difference.numel()
            persistence = factual_origin[None].float() - target.float()
            zero_difference: Tensor | None = None
            if step == 0:
                zero_conditioned = outgoing.clone()
                zero_conditioned[:, -1] = 0.0
                zero_prediction = model.predict_next_grid(grids, zero_conditioned, valid)
                f_calls += 1
                if not bool(torch.isfinite(zero_prediction).all()):
                    raise FloatingPointError("zero-raw4 F prediction contains non-finite values")
                zero_difference = zero_prediction.float() - target.float()
            row = _metadata(
                trajectory,
                arm=arm,
                component="f",
                origin=origin,
                view="recursive" if step else "one_step",
                horizon=step + 1,
                donor_key=(f"action:{donor_index}" if arm == "action_permutation" else None),
            )
            row.update(
                {
                    "mse": float(difference.square().mean()),
                    "mae": float(difference.abs().mean()),
                    "endpoint_mse": prediction_squared_error / difference.numel(),
                    "endpoint_mae": prediction_absolute_error / difference.numel(),
                    "path_mse": path_squared_error / path_element_count,
                    "path_mae": path_absolute_error / path_element_count,
                    "cosine_distance": float(
                        cosine_distance(prediction, target, epsilon=epsilon)[0]
                    ),
                    "persistence_mse": float(persistence.square().mean()),
                    "persistence_improvement_mse": float(
                        persistence.square().mean() - difference.square().mean()
                    ),
                    "zero_raw4_is_oos_bos_negative": bool(step == 0),
                    "zero_raw4_mse": (
                        float(zero_difference.square().mean())
                        if zero_difference is not None
                        else None
                    ),
                }
            )
            output.append(row)
            grids = append_context_latent(grids, prediction, context_size=CONTEXT_SIZE)
            outgoing = torch.cat((conditioned[:, -CONTEXT_SIZE:], action_raw4[:, None]), dim=1)[
                :, -CONTEXT_SIZE:
            ]
            valid = torch.cat(
                (
                    valid[:, -CONTEXT_SIZE:],
                    torch.ones((1, 1), dtype=torch.bool, device=valid.device),
                ),
                dim=1,
            )[:, -CONTEXT_SIZE:]
    return output, {
        "q_calls": 0,
        "g_calls": 0,
        "g_rows": 0,
        "f_calls": f_calls,
        "f_rows": len(output),
    }


class OfflineTrajectoryEvaluator:
    """Evaluate baseline and single-factor arms while reusing unaffected outputs."""

    _AFFECTS = {
        "q": frozenset(
            {
                "context_suffix_1",
                "context_suffix_2",
                "context_suffix_4",
                "mean_repeat_input",
                "fixed_permutation_input",
                "goal_permutation",
            }
        ),
        "g": frozenset(
            {
                "mean_repeat_input",
                "fixed_permutation_input",
                "intent_permutation",
                "previous_action_permutation",
            }
        ),
        "f": frozenset(
            {
                "context_suffix_1",
                "context_suffix_2",
                "context_suffix_4",
                "mean_repeat_input",
                "fixed_permutation_input",
                "action_permutation",
            }
        ),
        "ranking": frozenset(),
    }

    def __init__(
        self,
        model: nn.Module,
        *,
        horizon: int,
        permutation: Tensor,
        cosine_epsilon: float,
        active_k: int,
        active_h: int,
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.permutation = permutation
        self.cosine_epsilon = cosine_epsilon
        self.active_k = active_k
        self.active_h = active_h
        self._trajectory_id: str | None = None
        self._cache: dict[tuple[str, str], tuple[list[dict[str, Any]], dict[str, int]]] = {}
        self._proposal_cache: dict[str, Any] = {}

    def _generate_proposal(self, batch):
        if bool(getattr(self.model.proposal, "is_stochastic", False)):
            # One bank covers both the explicit requested budget and the
            # diagnostic prefix sweep; every arm reuses its addressed bank.
            return _proposal_forward(self.model, batch,
                samples=max(self.active_k, self.model.proposal.default_samples),
                horizon=max(self.active_h, self.horizon, getattr(self.model.proposal, "default_horizon", self.model.proposal.trained_horizon)),
                sampling_namespace="offline/v1")
        return _proposal_forward(self.model, batch)

    def __call__(
        self,
        trajectory: MaterializedOfflineTrajectory,
        arm: str,
    ) -> tuple[dict[str, Any], ...]:
        if self._trajectory_id != trajectory.trajectory:
            self._trajectory_id = trajectory.trajectory
            self._cache.clear()
            self._proposal_cache.clear()
        emitted: list[dict[str, Any]] = []
        actual_counts = {"q_calls": 0, "g_calls": 0, "g_rows": 0, "f_calls": 0, "f_rows": 0}
        components = ("q", "g", "f", "ranking") if arm == "baseline" else ("q", "g", "f")
        for component in components:
            effective_arm = arm if arm in self._AFFECTS[component] else "baseline"
            cache_key = (component, effective_arm)
            if cache_key not in self._cache:
                component_batch, unavailable = _component_batch(
                    trajectory.batch,
                    arm=effective_arm,
                    component=component,
                    permutation=self.permutation,
                    goal_indices=trajectory.goal_indices,
                )
                if component_batch is None:
                    rows = [
                        {
                            **_metadata(
                                trajectory,
                                arm=effective_arm,
                                component=component,
                                origin=None,
                                view="unavailable",
                                horizon=self.horizon,
                            ),
                            "available": False,
                            "status": "UNAVAILABLE",
                            "reason": unavailable,
                        }
                    ]
                    counts = dict(actual_counts)
                elif component == "q":
                    proposal = self._proposal_cache.get(effective_arm)
                    q_called = proposal is None
                    if proposal is None:
                        proposal = self._generate_proposal(component_batch)
                        self._proposal_cache[effective_arm] = proposal
                    rows, counts = _q_rows(
                        self.model,
                        trajectory,
                        component_batch,
                        arm=effective_arm,
                        proposal=proposal,
                        q_called=q_called,
                    )
                elif component == "g":
                    rows, counts = _g_rows(self.model, trajectory, component_batch, arm=effective_arm)
                elif component == "f":
                    rows, counts = _f_rows(
                        self.model,
                        trajectory,
                        component_batch,
                        arm=effective_arm,
                        horizon=self.horizon,
                        epsilon=self.cosine_epsilon,
                    )
                else:
                    proposal = self._proposal_cache.get("baseline")
                    q_called = proposal is None
                    if proposal is None:
                        proposal = self._generate_proposal(component_batch)
                        self._proposal_cache["baseline"] = proposal
                    k_values = tuple(
                        value
                        for value in sorted({1, 2, 4, self.active_k})
                        if value <= proposal.tape.shape[1]
                    )
                    h_values = tuple(
                        value
                        for value in sorted({1, 2, 4, self.active_h})
                        if value <= proposal.tape.shape[2]
                    )
                    rows, counts = _ranking_rows(
                        self.model,
                        trajectory,
                        component_batch,
                        proposal=proposal,
                        k_values=k_values,
                        h_values=h_values,
                    )
                    counts["q_calls"] += int(q_called)
                self._cache[cache_key] = (rows, counts)
                for name in actual_counts:
                    actual_counts[name] += counts.get(name, 0)
            rows, _counts = self._cache[cache_key]
            for row in rows:
                copied = dict(row)
                copied["arm"] = arm
                emitted.append(copied)
        emitted.append(
            {
                **_metadata(
                    trajectory,
                    arm=arm,
                    component="invocations",
                    origin=None,
                    view="actual_incremental",
                    horizon=self.horizon,
                ),
                **actual_counts,
            }
        )
        return tuple(emitted)


__all__ = [
    "MaterializedOfflineTrajectory",
    "OfflineTrajectoryEvaluator",
    "materialize_offline_trajectory",
]
