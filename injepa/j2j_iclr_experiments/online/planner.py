"""Inference-only Context4 Q/G/F planner for one factual decision cycle."""

from __future__ import annotations

import math
import json
import hashlib

import torch
from torch import Tensor

from j2j.adapter import ActionId
from j2j.context4_variants import VariantContract

from .adapter import DecisionState, PolicyDecision, _tensor_sha256
from .stop import canonical_stop_distance, zero_intent_stop_action


def _positive_at_most(name: str, value: object, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0 or value > maximum:
        raise ValueError(f"active {name} is outside the checkpoint model identity")
    return value


def _finite_grid(name: str, value: Tensor, shape: torch.Size) -> Tensor:
    if not isinstance(value, Tensor) or value.shape != shape:
        raise ValueError(f"{name} transform changed the grid shape")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} grid must be float32")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} grid must be finite")
    return value


class Context4Planner:
    """Consume a disposable factual view and return one canonical primitive."""

    def __init__(
        self,
        *,
        model,
        deployment: str,
        k_model: int,
        h_model: int,
        active_k: int,
        active_h: int,
        stop_mode: str = "never_stop",
        stop_threshold: float | None = None,
        grid_transform=None,
        variant_contract: VariantContract | None = None,
        sampling_seed: int | None = None,
        sampling_namespace: str = "online/v1",
        whole_branch_ranking=None,
        stagnation_guard=None,
    ) -> None:
        if type(k_model) is not int or k_model <= 0:
            raise ValueError("checkpoint model K must be a positive integer")
        if type(h_model) is not int or h_model <= 0:
            raise ValueError("checkpoint model H must be a positive integer")
        proposal = getattr(model, "proposal", None)
        self.stochastic = bool(getattr(proposal, "is_stochastic", False))
        self.active_k = _positive_at_most("K", active_k, active_k if self.stochastic or deployment == "goal_gf" else k_model)
        self.active_h = _positive_at_most("H", active_h, active_h if self.stochastic or deployment == "goal_gf" else h_model)
        if sampling_seed is None:
            sampling_seed = getattr(proposal, "global_seed", 0)
        if type(sampling_seed) is not int or sampling_seed < 0:
            raise ValueError("sampling_seed must be a nonnegative integer")
        if not isinstance(sampling_namespace, str) or not sampling_namespace:
            raise ValueError("sampling_namespace must be nonempty text")
        self.sampling_seed = sampling_seed
        self.sampling_namespace = sampling_namespace
        if deployment not in {
            "full", "no_f", "no_f_deploy", "proposal_only", "posthoc_f",
            "f_feedback_q", "goal_gf",
        }:
            raise ValueError(
                "deployment must be full, no_f, proposal_only, posthoc_f, f_feedback_q, or goal_gf"
            )
        if stop_mode not in {"never_stop", "calibrated"}:
            raise ValueError("stop_mode must be never_stop or calibrated")
        if stop_mode == "calibrated":
            if isinstance(stop_threshold, bool) or not isinstance(
                stop_threshold, (int, float)
            ):
                raise TypeError("calibrated STOP requires a numeric threshold")
            if not math.isfinite(float(stop_threshold)) or float(stop_threshold) < 0.0:
                raise ValueError("STOP threshold must be finite and non-negative")
        elif stop_threshold is not None:
            raise ValueError("never_stop may not carry a STOP threshold")
        if grid_transform is not None and not callable(grid_transform):
            raise TypeError("grid_transform must be callable")
        if variant_contract is not None and not isinstance(
            variant_contract, VariantContract
        ):
            raise TypeError("variant_contract must be a resolved VariantContract")
        self.model = model
        self.deployment = "no_f" if deployment == "no_f_deploy" else deployment
        self.k_model = k_model
        self.h_model = h_model
        self.stop_mode = stop_mode
        self.stop_threshold = None if stop_threshold is None else float(stop_threshold)
        self.variant_contract = variant_contract
        self.whole_branch_ranking = None
        if whole_branch_ranking is not None:
            from j2j_iclr_experiments.offline.whole_branch_rank import validate_whole_branch_options
            if deployment not in {"full", "posthoc_f", "f_feedback_q", "goal_gf"}:
                raise ValueError(
                    "whole branch ranking requires full, posthoc_f, or f_feedback_q deployment"
                )
            self.whole_branch_ranking = validate_whole_branch_options(whole_branch_ranking)
        if self.deployment == "goal_gf":
            self.whole_branch_ranking = self.whole_branch_ranking or dict(risk_weight=0.0, epsilon=1e-6)
            if (self.whole_branch_ranking["risk_weight"] != 0
                    or self.whole_branch_ranking.get("score_mode", "l1_consistency") != "l1_consistency"):
                raise ValueError("goal_gf requires matched normalized L1 endpoint-only risk0")
        if self.deployment == "posthoc_f" and self.whole_branch_ranking is None:
            raise ValueError("posthoc_f requires the matched whole branch evaluator")
        self.stagnation_guard = None
        if stagnation_guard is not None:
            from j2j_iclr_experiments.offline.stagnation_guard import validate_stagnation_options
            if self.deployment != "full" or self.whole_branch_ranking is None:
                raise ValueError("stagnation guard requires full whole-branch deployment")
            self.stagnation_guard = validate_stagnation_options(stagnation_guard)
            if self.stagnation_guard["frame_count"] > 4:
                raise ValueError("stagnation frame_count exceeds this planner's factual Context4")
        self.grid_transform = (
            grid_transform
            or (
                variant_contract.runtime_grid_transform
                if variant_contract is not None
                else None
            )
            or (lambda grid: grid.clone())
        )

    def _model_device(self, fallback: torch.device) -> torch.device:
        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                return next(parameters()).device
            except StopIteration:
                pass
        return fallback

    def _prepare_state(
        self, state: DecisionState, *, compute_stop_distance: bool = True
    ) -> dict[str, Tensor | float]:
        """Build the sole factual tensor view shared by online and local paths."""

        records = tuple(state.records)
        if not 1 <= len(records) <= 4:
            raise ValueError("factual context must contain one to four real records")
        if not all(record.valid for record in records):
            raise ValueError("online records supplied to the planner must be factual")
        device = self._model_device(records[-1].grid.device)
        canonical_grids = torch.stack([record.grid for record in records]).to(
            device=device, dtype=torch.float32
        )
        canonical_goal = state.goal_grid.to(device=device, dtype=torch.float32)
        if canonical_goal.shape != canonical_grids.shape[1:]:
            raise ValueError("goal grid shape does not match factual grids")
        if not bool(
            torch.isfinite(canonical_grids).all()
            and torch.isfinite(canonical_goal).all()
        ):
            raise FloatingPointError("canonical online grids must be finite")
        canonical_distance = (
            canonical_stop_distance(canonical_grids[-1], canonical_goal)
            if compute_stop_distance and self.stop_mode == "calibrated"
            else 0.0
        )
        grids = _finite_grid(
            "factual",
            self.grid_transform(canonical_grids.clone()),
            canonical_grids.shape,
        ).to(device=device)
        goal = _finite_grid(
            "goal", self.grid_transform(canonical_goal.clone()), canonical_goal.shape
        ).to(device=device)
        previous = state.previous_raw4.to(device=device, dtype=torch.float32)
        if previous.shape != (4,) or not bool(torch.isfinite(previous).all()):
            raise ValueError("previous factual action must be a finite raw4 vector")
        incoming = torch.stack([record.incoming_raw4 for record in records]).to(
            device=device, dtype=torch.float32
        )
        outgoing = torch.stack(
            [
                torch.zeros(4, dtype=torch.float32, device=device)
                if record.outgoing_raw4_or_none is None
                else record.outgoing_raw4_or_none.to(
                    device=device, dtype=torch.float32
                )
                for record in records
            ]
        )
        if not bool(torch.isfinite(incoming).all() and torch.isfinite(outgoing).all()):
            raise FloatingPointError("factual action axes must be finite")
        length = len(records)
        ages = torch.arange(
            length - 1, -1, -1, device=device, dtype=torch.int64
        )[None]
        kinds = torch.ones((1, length), device=device, dtype=torch.int64)
        kinds[:, -1] = 2
        valid = torch.ones((1, length), device=device, dtype=torch.bool)
        if (
            self.variant_contract is not None
            and self.variant_contract.identity.variant_id == "context1"
        ):
            pad = 4 - length
            grids = torch.cat(
                (
                    torch.zeros(
                        (pad, *grids.shape[1:]), dtype=grids.dtype, device=device
                    ),
                    grids,
                ),
                dim=0,
            )
            incoming = torch.cat(
                (torch.zeros((pad, 4), dtype=incoming.dtype, device=device), incoming),
                dim=0,
            )
            outgoing = torch.cat(
                (torch.zeros((pad, 4), dtype=outgoing.dtype, device=device), outgoing),
                dim=0,
            )
            ages = torch.cat(
                (torch.zeros((1, pad), dtype=ages.dtype, device=device), ages), dim=1
            )
            kinds = torch.cat(
                (torch.zeros((1, pad), dtype=kinds.dtype, device=device), kinds), dim=1
            )
            valid = torch.cat(
                (torch.zeros((1, pad), dtype=valid.dtype, device=device), valid), dim=1
            )
        if self.variant_contract is not None:
            context = self.variant_contract.context_transform(
                {
                    "context_grid": grids[None],
                    "context_incoming_raw4": incoming[None],
                    "context_outgoing_raw4": outgoing[None],
                    "context_age": ages,
                    "context_type": kinds,
                    "context_valid": valid,
                }
            )
            grids = context["context_grid"][0]
            incoming = context["context_incoming_raw4"][0]
            outgoing = context["context_outgoing_raw4"][0]
            ages = context["context_age"]
            kinds = context["context_type"]
            valid = context["context_valid"]
        future_mask = torch.ones(
            (1, self.h_model), device=device, dtype=torch.bool
        )
        return {
            "canonical_distance": canonical_distance,
            "grids": grids,
            "goal": goal,
            "previous": previous,
            "incoming": incoming,
            "outgoing": outgoing,
            "ages": ages,
            "kinds": kinds,
            "valid": valid,
            "future_mask": future_mask,
            **({"sampling_key": json.dumps({
                "step": state.step,
                "goal": state.goal_observation_key or _tensor_sha256(state.goal_grid),
                "records": [{"observation": record.observation_key,
                             "incoming": record.incoming_raw4.detach().cpu().tolist(),
                             "outgoing": (None if record.outgoing_raw4_or_none is None else
                                          record.outgoing_raw4_or_none.detach().cpu().tolist())}
                            for record in records],
            }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")}
              if self.stochastic or self.deployment == "goal_gf" else {}),
        }

    def _proposal(self, prepared: dict[str, Tensor | float]):
        facts = self._proposal_facts(prepared)
        future_mask = prepared["future_mask"]
        assert isinstance(future_mask, Tensor)
        if self.stochastic:
            proposal = self.model.proposal.sample_prior(
                *facts, samples=self.active_k, horizon=self.active_h,
                sample_keys=(prepared["sampling_key"],), seed=self.sampling_seed,
                namespace=self.sampling_namespace)
            expected_k, expected_h = self.active_k, self.active_h
        else:
            proposal = self.model.proposal(*facts, future_mask)
            expected_k, expected_h = self.k_model, self.h_model
        if proposal.tape.shape[:3] != (1, expected_k, expected_h):
            raise ValueError("Q output K/H axes disagree with checkpoint identity")
        if proposal.log_mass.shape != (1, expected_k):
            raise ValueError("Q mass axis disagrees with checkpoint identity")
        return proposal

    def _proposal_facts(self, prepared: dict[str, Tensor | float]):
        grids = prepared["grids"]
        incoming = prepared["incoming"]
        goal = prepared["goal"]
        ages = prepared["ages"]
        kinds = prepared["kinds"]
        valid = prepared["valid"]
        assert all(
            isinstance(value, Tensor)
            for value in (grids, incoming, goal, ages, kinds, valid)
        )
        return (
            grids[None],
            self.model.embed_actions(incoming[None]),
            ages,
            kinds,
            valid,
            goal[None],
        )

    def plan(self, state: DecisionState) -> PolicyDecision:
        from j2j_iclr_experiments.offline.rankers import (
            rank_full,
            rank_no_f,
            rank_f_feedback_q,
            rank_posthoc_f,
            rank_proposal_only,
        )

        prepared = self._prepare_state(state)
        grids = prepared["grids"]
        goal = prepared["goal"]
        previous = prepared["previous"]
        outgoing = prepared["outgoing"]
        valid = prepared["valid"]
        canonical_distance = float(prepared["canonical_distance"])
        assert all(
            isinstance(value, Tensor)
            for value in (grids, goal, previous, outgoing, valid)
        )

        stop_entered = (
            self.stop_mode == "calibrated"
            and canonical_distance <= float(self.stop_threshold)
        )
        stop_probe_calls = 0
        whole_branch_diagnostics = None
        with torch.no_grad():
            if stop_entered:
                stop_probe_calls = 1
                stop_action = zero_intent_stop_action(
                    self.model.actor_logits, grids[-1:], previous[None]
                )[0]
                if int(stop_action.item()) == int(ActionId.STOP):
                    return PolicyDecision(
                        action_id=int(ActionId.STOP),
                        diagnostics={
                            "canonical_stop_distance": canonical_distance,
                            "stop_category": "executed",
                            "counters": {
                                "q_calls": 0,
                                "g_calls": 1,
                                "g_rows": 1,
                                "f_calls": 0,
                                "f_rows": 0,
                                "candidate_rows": 0,
                            },
                        },
                    )

            if self.deployment == "goal_gf":
                return self._plan_goal_gf(prepared, stop_entered, stop_probe_calls)
            proposal = self._proposal(prepared)
            if self.deployment in {"full", "posthoc_f", "f_feedback_q"}:
                if self.deployment == "f_feedback_q":
                    if not self.stochastic or not hasattr(
                        self.model.proposal, "recurrent_step_from_state"
                    ):
                        raise RuntimeError("f_feedback_q requires stochastic recurrent Q")
                    facts = self._proposal_facts(prepared)
                    ranking = rank_f_feedback_q(
                        initial_intent=proposal.tape[:, :, 0],
                        trajectory_latent=proposal.trajectory_latent,
                        log_mass=proposal.log_mass,
                        goal_grid=goal[None],
                        factual_context=grids[None],
                        factual_outgoing_raw4=outgoing[None],
                        previous_raw4=previous[None],
                        context_valid=valid,
                        q_step=lambda current, z: self.model.proposal.recurrent_step_from_state(
                            *facts, previous_grid=current, trajectory_latent=z
                        ),
                        actor_step=self.model.actor_logits,
                        forward_step=self.model.predict_next_grid,
                        active_k=self.active_k,
                        active_h=self.active_h,
                    )
                else:
                    ranker = rank_full if self.deployment == "full" else rank_posthoc_f
                    ranking = ranker(
                        proposal_tape=proposal.tape,
                        log_mass=proposal.log_mass,
                        goal_grid=goal[None],
                        factual_context=grids[None],
                        factual_outgoing_raw4=outgoing[None],
                        previous_raw4=previous[None],
                        context_valid=valid,
                        actor_step=self.model.actor_logits,
                        forward_step=self.model.predict_next_grid,
                        active_k=self.active_k,
                        active_h=self.active_h,
                    )
                if self.whole_branch_ranking is not None:
                    from j2j_iclr_experiments.offline.whole_branch_rank import rank_whole_branches
                    ranking, scores = rank_whole_branches(ranking,
                        current_grid=grids[-1:], goal_grid=goal[None], **self.whole_branch_ranking)
                    whole_branch_diagnostics = {
                        name: value.detach().cpu().tolist() for name, value in scores.items()}
                    whole_branch_diagnostics.update(
                        actions=ranking.actions.detach().cpu().tolist(),
                        mode_indices=ranking.mode_indices.detach().cpu().tolist(),
                        f_goal_distance=ranking.goal_distance.detach().cpu().tolist(),
                        consistency=ranking.consistency.detach().cpu().tolist(),
                        winner_k=ranking.winner_k.detach().cpu().tolist(),
                        winner_h=ranking.winner_h.detach().cpu().tolist())
                    if self.stagnation_guard is not None:
                        from j2j_iclr_experiments.offline.stagnation_guard import apply_stagnation_guard
                        primary_counters = dict(ranking.counters)
                        ranking, guard = apply_stagnation_guard(ranking, scores,
                            factual_grids=grids, factual_valid=valid,
                            outgoing_raw4=outgoing, goal_grid=goal,
                            forward_step=self.model.predict_next_grid,
                            options=self.stagnation_guard)
                        whole_branch_diagnostics["stagnation_guard"] = guard
                        if ranking is None:
                            # Extra action-only turn probes have no Q branch or R.
                            # Never attribute their action to an old FWD winner.
                            whole_branch_diagnostics.update(winner_k=None, winner_h=None)
                            counters = primary_counters
                            counters["q_calls"] = 1
                            counters["g_calls"] = int(counters.get("g_calls", 0)) + stop_probe_calls
                            counters["g_rows"] = int(counters.get("g_rows", 0)) + stop_probe_calls
                            counters["f_calls"] = int(counters.get("f_calls", 0)) + guard["extra_f_calls"]
                            counters["f_rows"] = int(counters.get("f_rows", 0)) + guard["extra_f_rows"]
                            counters["candidate_rows"] = int(counters.get("candidate_rows", 0)) + guard["extra_f_rows"]
                            return PolicyDecision(action_id=guard["selected_action"], diagnostics={
                                "canonical_stop_distance": canonical_distance,
                                "stop_category": "entered-motion" if stop_entered else "not-entered",
                                "winner_source": "fallback_turns", "winner_k": None, "winner_h": 1,
                                "counters": counters, "whole_branch_ranking": whole_branch_diagnostics})
                        whole_branch_diagnostics.update(
                            winner_k=ranking.winner_k.detach().cpu().tolist(),
                            winner_h=ranking.winner_h.detach().cpu().tolist(),
                            ordered_mode_indices=ranking.ordered_indices[..., 0].detach().cpu().tolist())
            elif self.deployment == "proposal_only":
                ranking = rank_proposal_only(
                    proposal_tape=proposal.tape,
                    log_mass=proposal.log_mass,
                    goal_grid=goal[None],
                    current_grid=grids[-1:],
                    previous_raw4=previous[None],
                    actor_step=self.model.actor_logits,
                    active_k=self.active_k,
                    active_h=self.active_h,
                )
            else:
                ranking = rank_no_f(
                    proposal_tape=proposal.tape,
                    log_mass=proposal.log_mass,
                    goal_grid=goal[None],
                    current_grid=grids[-1:],
                    previous_raw4=previous[None],
                    actor_step=self.model.actor_logits,
                    active_k=self.active_k,
                    active_h=self.active_h,
                )

        action = int(ranking.first_action[0].item())
        if action not in {
            int(ActionId.FWD),
            int(ActionId.LEFT),
            int(ActionId.RIGHT),
        }:
            raise RuntimeError("ordinary planning returned a non-motion action")
        counters = dict(ranking.counters)
        if self.deployment != "f_feedback_q":
            counters["q_calls"] = 1
        counters["g_calls"] = int(counters.get("g_calls", 0)) + stop_probe_calls
        counters["g_rows"] = int(counters.get("g_rows", 0)) + stop_probe_calls
        return PolicyDecision(
            action_id=action,
            diagnostics={
                "canonical_stop_distance": canonical_distance,
                "stop_category": "entered-motion" if stop_entered else "not-entered",
                "winner_k": int(ranking.winner_k[0].item()),
                "winner_h": int(ranking.winner_h[0].item()) + 1,
                "counters": counters,
                **({"whole_branch_ranking": whole_branch_diagnostics}
                   if whole_branch_diagnostics is not None else {}),
            },
        )

    def _plan_goal_gf(self, prepared, stop_entered, stop_probe_calls):
        """Use a fresh private categorical stream; never prepare or sample Q."""
        from j2j_iclr_experiments.offline.rankers import rank_goal_gf

        grids, goal = prepared["grids"], prepared["goal"]
        address = json.dumps(["goal_gf/categorical/v1", self.sampling_seed,
            self.sampling_namespace], separators=(",", ":")).encode() + prepared["sampling_key"]
        seed = int.from_bytes(hashlib.sha256(address).digest()[:8], "big")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        epsilon = self.whole_branch_ranking["epsilon"]
        ranking = rank_goal_gf(goal_grid=goal[None], factual_context=grids[None],
            factual_outgoing_raw4=prepared["outgoing"][None],
            previous_raw4=prepared["previous"][None], context_valid=prepared["valid"],
            actor_step=self.model.actor_logits, forward_step=self.model.predict_next_grid,
            active_k=self.active_k, active_h=self.active_h, generator=generator, epsilon=epsilon)
        initial = (grids[-1:].float() - goal[None].float()).abs().mean((-1, -2))
        score = ranking.goal_distance[:, :, -1] / initial[:, None].clamp_min(epsilon)
        details = {"branch_score": score.detach().cpu().tolist(),
            "terminal_normalized": score.detach().cpu().tolist(),
            "goal_normalization_inconclusive": (initial <= epsilon).cpu().tolist(),
            "score_mode": "l1_consistency", "risk_weight": 0.0, "epsilon": epsilon,
            "candidate_source": "goal_gf_categorical", "action_temperature": 1.0,
            "proposal_tape": None, "consistency": None, "q_metrics_status": "not_applicable"}
        for name, value in dict(actions=ranking.actions, mode_indices=ranking.mode_indices,
                f_goal_distance=ranking.goal_distance, winner_k=ranking.winner_k,
                winner_h=ranking.winner_h,
                ordered_mode_indices=ranking.ordered_indices[..., 0]).items():
            details[name] = value.detach().cpu().tolist()
        counters = dict(ranking.counters)
        counters["g_calls"] += stop_probe_calls
        counters["g_rows"] += stop_probe_calls
        return PolicyDecision(action_id=int(ranking.first_action[0].item()), diagnostics={
            "canonical_stop_distance": float(prepared["canonical_distance"]),
            "stop_category": "entered-motion" if stop_entered else "not-entered",
            "winner_k": int(ranking.winner_k[0].item()),
            "winner_h": int(ranking.winner_h[0].item()) + 1,
            "q_metrics_status": "not_applicable", "counters": counters,
            "whole_branch_ranking": details})


def _plan_local_candidates(
    planner: Context4Planner,
    state: DecisionState,
    *,
    stop_category: str = "not-entered",
) -> dict[str, object]:
    """Run Q once and feed that exact proposal to current Full and NoG rankers."""

    from j2j_iclr_experiments.offline import rankers

    if type(planner) is not Context4Planner:
        raise TypeError("local planner must be the canonical Context4Planner")
    if stop_category not in {"not-entered", "entered-motion"}:
        raise ValueError("local source state has an invalid STOP category")
    # Stage 1 already froze the canonical STOP category/distance. Stage 2a
    # must regenerate Q/rankers, not run a second STOP-distance decision.
    prepared = planner._prepare_state(state, compute_stop_distance=False)
    grids = prepared["grids"]
    goal = prepared["goal"]
    previous = prepared["previous"]
    outgoing = prepared["outgoing"]
    valid = prepared["valid"]
    assert all(
        isinstance(value, Tensor)
        for value in (grids, goal, previous, outgoing, valid)
    )
    stop_probe_calls = 0
    with torch.no_grad():
        if stop_category == "entered-motion":
            stop_probe_calls = 1
            action = zero_intent_stop_action(
                planner.model.actor_logits, grids[-1:], previous[None]
            )[0]
            if int(action.item()) == int(ActionId.STOP):
                raise RuntimeError("replayed entered-motion STOP probe changed to STOP")
        proposal = planner._proposal(prepared)
        common = {
            "proposal_tape": proposal.tape,
            "log_mass": proposal.log_mass,
            "goal_grid": goal[None],
            "factual_context": grids[None],
            "factual_outgoing_raw4": outgoing[None],
            "previous_raw4": previous[None],
            "context_valid": valid,
            "active_k": planner.active_k,
            "active_h": planner.active_h,
        }
        full = rankers.rank_full(
            **common,
            actor_step=planner.model.actor_logits,
            forward_step=planner.model.predict_next_grid,
        )
        no_g = rankers.rank_no_g(
            **common,
            forward_step=planner.model.predict_next_grid,
        )
    return {
        "q_calls": 1,
        "stop_probe_g_calls": stop_probe_calls,
        "proposal": proposal,
        "full": full,
        "no_g_enumerate": no_g,
    }


__all__ = ["Context4Planner", "_plan_local_candidates"]
