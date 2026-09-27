"""Factual-only online state owner for the Context4 ImageGoal policy.

The adapter deliberately owns encoded state.  The Habitat runner history is
used only as an independently supplied lifecycle receipt; it never causes a
cached observation to be encoded twice and imagined rollout states never enter
the records below.
"""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

from collections import deque
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from j2j.adapter import ActionId, Raw4Adapter
from j2j.receipts import canonical_json_bytes
from j2j_iclr_experiments.local_habitat.capability import _array_bytes
from j2j_iclr_experiments.common.artifacts import to_plain_json


_HISTORY_FIELDS = frozenset({"rgb", "action", "order", "mask"})
_ACTION_NAMES = {
    "STOP": ActionId.STOP,
    "FWD": ActionId.FWD,
    "MOVE_FORWARD": ActionId.FWD,
    "LEFT": ActionId.LEFT,
    "TURN_LEFT": ActionId.LEFT,
    "RIGHT": ActionId.RIGHT,
    "TURN_RIGHT": ActionId.RIGHT,
}


def _observation_key(value: Any) -> str:
    """Return a process-local deterministic key without retaining image bytes."""

    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        payload = (
            str(tensor.dtype).encode("ascii")
            + repr(tuple(tensor.shape)).encode("ascii")
            + tensor.numpy().tobytes()
        )
    elif isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        # The key is a lifecycle guard, not an artifact identity.  Sensor
        # arrays bind their logical dtype and shape as well as canonical
        # row-major bytes, so byte-identical views cannot alias one another.
        tobytes = getattr(value, "tobytes", None)
        if (
            callable(tobytes)
            and getattr(value, "dtype", None) is not None
            and getattr(value, "shape", None) is not None
        ):
            array = np.asarray(value)
            canonical = np.ascontiguousarray(array)
            dtype = canonical.dtype.str.encode("ascii")
            shape = repr(tuple(canonical.shape)).encode("ascii")
            payload = (
                b"J2J_OBSERVATION_ARRAY_V1\x00"
                + len(dtype).to_bytes(8, "little")
                + dtype
                + len(shape).to_bytes(8, "little")
                + shape
                + canonical.tobytes(order="C")
            )
        elif callable(tobytes):
            payload = bytes(tobytes())
        else:
            payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_action(value: Any) -> ActionId:
    if isinstance(value, ActionId):
        return value
    if isinstance(value, bool):
        raise TypeError("history action cannot be boolean")
    if isinstance(value, str):
        try:
            return _ACTION_NAMES[value.upper()]
        except KeyError as exc:
            raise ValueError("history action is not canonical") from exc
    if isinstance(value, int):
        try:
            return ActionId(value)
        except ValueError as exc:
            raise ValueError("history action is not canonical") from exc
    raise TypeError("history action must be a canonical action")


def _validate_grid(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} encoder output must be [spatial,latent]")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} encoder output must be float32")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} encoder output must be finite")
    return value.detach().clone()


def _tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    descriptor = canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    return hashlib.sha256(descriptor + tensor.numpy().tobytes(order="C")).hexdigest()


def _axes_sha256(incoming: Tensor, outgoing: Tensor | None) -> str:
    payload = {
        "incoming_raw4_sha256": _tensor_sha256(incoming),
        "outgoing_raw4_sha256": (
            None if outgoing is None else _tensor_sha256(outgoing)
        ),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _formal_rgb_sha256(value: Any) -> str:
    try:
        payload = _array_bytes(value, name="RGB")
    except (TypeError, ValueError):
        # Opaque strings/objects are retained only by established unit fakes.
        # Formal local consumers separately require numeric Habitat arrays.
        return _observation_key(value)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PolicyDecision:
    action_id: int
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class FactualRecord:
    grid: Tensor
    incoming_raw4: Tensor
    outgoing_raw4_or_none: Tensor | None
    valid: bool
    observation_key: str


@dataclass(frozen=True)
class DecisionState:
    records: tuple[FactualRecord, ...]
    goal_grid: Tensor
    previous_raw4: Tensor
    step: int
    goal_observation_key: str = ""


def _clone_record(record: FactualRecord) -> FactualRecord:
    return FactualRecord(
        grid=record.grid.detach().clone(),
        incoming_raw4=record.incoming_raw4.detach().clone(),
        outgoing_raw4_or_none=(
            None
            if record.outgoing_raw4_or_none is None
            else record.outgoing_raw4_or_none.detach().clone()
        ),
        valid=record.valid,
        observation_key=record.observation_key,
    )


def build_factual_decision_state(
    records: Sequence[FactualRecord], *, goal_grid: Tensor, step: int,
    goal_observation_key: str = "",
) -> DecisionState:
    """Construct the sole current DecisionState ABI from factual records."""

    values = tuple(records)
    if not 1 <= len(values) <= 4 or any(type(row) is not FactualRecord for row in values):
        raise ValueError("decision state requires one to four factual records")
    if type(step) is not int or step < 0:
        raise ValueError("decision state step must be a non-negative integer")
    goal = _validate_grid(goal_grid, name="goal")
    cloned = tuple(_clone_record(row) for row in values)
    if goal.shape != cloned[-1].grid.shape:
        raise ValueError("decision state goal/factual grid shapes differ")
    return DecisionState(
        records=cloned,
        goal_grid=goal,
        previous_raw4=cloned[-1].incoming_raw4.detach().clone(),
        step=step,
        goal_observation_key=goal_observation_key,
    )


class Context4PolicyAdapter:
    """Own the last four real encoded observations and their two action axes."""

    def __init__(
        self,
        *,
        encode_grid,
        planner,
        provenance: Mapping[str, object],
        stop_mode: str = "never_stop",
        stop_receipt: Mapping[str, object] | None = None,
        stop_admission: object = None,
        max_diagnostics: int = 128,
        execution_interval: int = 1,
    ) -> None:
        if type(execution_interval) is not int or execution_interval < 1:
            raise ValueError("execution_interval must be a positive integer")
        if execution_interval > 1 and stop_mode != "never_stop":
            raise ValueError("two-step execution requires reach-only motion")
        self._execution_interval = execution_interval
        self._queued_execution = None
        self._planning_count = 0
        if not callable(encode_grid):
            raise TypeError("encode_grid must be callable")
        if not callable(getattr(planner, "plan", None)):
            raise TypeError("planner must expose plan(state)")
        if isinstance(max_diagnostics, bool) or not isinstance(max_diagnostics, int):
            raise TypeError("max_diagnostics must be an integer")
        if max_diagnostics <= 0:
            raise ValueError("max_diagnostics must be positive")
        if stop_mode not in {"never_stop", "calibrated"}:
            raise ValueError("stop_mode must be never_stop or calibrated")
        if stop_mode == "calibrated" and stop_receipt is None:
            raise RuntimeError("calibrated formal STOP requires a receipt")
        if stop_mode == "calibrated":
            from .stop import validate_stop_calibration_receipt

            if stop_admission is None:
                validate_stop_calibration_receipt(stop_receipt)  # type: ignore[arg-type]
            else:
                from .stop_admission import validate_stop_token
                validate_stop_token(stop_admission, stop_receipt)
        self._encode_grid = encode_grid
        self._planner = planner
        self._base_provenance = to_plain_json(provenance)
        self._stop_mode = stop_mode
        self._stop_receipt = (
            None if stop_receipt is None else to_plain_json(stop_receipt)
        )
        self._diagnostics: deque[Mapping[str, object]] = deque(maxlen=max_diagnostics)
        self._diagnostics_dropped = 0
        self._records: list[FactualRecord] = []
        self._verified_history: list[tuple[str, ActionId, int, bool]] = []
        self._factual_identities: list[dict[str, object]] = []
        self._goal_grid: Tensor | None = None
        self._goal_grid_key = ""
        self._goal_key: str | None = None
        self._goal_formal_sha256: str | None = None
        self._pending: ActionId | None = None
        self._terminal = False
        self._closed = False

    def reset(self, goal_rgb: Any) -> None:
        if self._closed:
            raise RuntimeError("adapter is closed")
        goal_key = _observation_key(goal_rgb)
        goal_grid = _validate_grid(self._encode_grid(goal_rgb), name="goal")
        self._queued_execution = None
        self._planning_count = 0
        self._records.clear()
        self._verified_history.clear()
        self._factual_identities.clear()
        self._diagnostics.clear()
        self._diagnostics_dropped = 0
        self._goal_key = goal_key
        self._goal_formal_sha256 = _formal_rgb_sha256(goal_rgb)
        self._goal_grid = goal_grid
        self._goal_grid_key = _tensor_sha256(goal_grid)
        self._pending = None
        self._terminal = False

    def _validate_history(
        self, current_rgb: Any, history: Sequence[Mapping[str, Any]]
    ) -> tuple[str, list[tuple[str, ActionId, int, bool]]]:
        if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
            raise TypeError("history must be a sequence of factual rows")
        parsed: list[tuple[str, ActionId, int, bool]] = []
        for row in history:
            if not isinstance(row, Mapping):
                raise TypeError("history row must be a mapping")
            extra = set(row) - _HISTORY_FIELDS
            missing = _HISTORY_FIELDS - set(row)
            if extra or missing:
                raise KeyError("history contains a privileged/unknown field or is incomplete")
            if row["mask"] is not True:
                raise ValueError("history factual mask must be true")
            order = row["order"]
            if isinstance(order, bool) or not isinstance(order, int):
                raise TypeError("history order must be an integer")
            parsed.append(
                (_observation_key(row["rgb"]), _as_action(row["action"]), order, True)
            )
        if parsed[: len(self._verified_history)] != self._verified_history:
            raise RuntimeError("history prefix changed")
        expected_length = len(self._verified_history) + (1 if self._pending is not None else 0)
        if len(parsed) != expected_length:
            raise RuntimeError("history does not contain exactly one pending successor")
        current_key = _observation_key(current_rgb)
        if self._pending is not None:
            key, action, order, _mask = parsed[-1]
            if key != current_key:
                raise RuntimeError("history successor RGB does not match current observation")
            if action is not self._pending:
                raise RuntimeError("history action does not match pending executed action")
            if order != len(self._verified_history):
                raise RuntimeError("history order is not contiguous")
        elif self._records:
            raise RuntimeError("duplicate decision without a pending successor")
        return current_key, parsed

    def act(self, current_rgb: Any, goal_rgb: Any, history: Sequence[Mapping[str, Any]]):
        if self._closed:
            raise RuntimeError("adapter is closed")
        if self._goal_grid is None or self._goal_key is None:
            raise RuntimeError("reset(goal_rgb) must be called before act")
        if self._terminal:
            raise RuntimeError("episode is terminal after STOP")
        if _observation_key(goal_rgb) != self._goal_key:
            raise RuntimeError("goal changed within the episode")

        # Validate the independently supplied lifecycle receipt before encoding.
        current_key, parsed_history = self._validate_history(current_rgb, history)
        current_grid = _validate_grid(self._encode_grid(current_rgb), name="current")

        # Construct the entire next state off to the side.  Encoding and every
        # planner branch may fail; none of those failures is allowed to consume
        # the pending action/history or partially append a factual record.
        next_records = [_clone_record(record) for record in self._records]
        next_verified_history = list(self._verified_history)
        next_factual_identities = copy.deepcopy(self._factual_identities)
        if self._pending is None:
            incoming = Raw4Adapter.encode_bos().to(current_grid.device)
        else:
            outgoing = Raw4Adapter.encode(self._pending).to(current_grid.device)
            previous = next_records[-1]
            next_records[-1] = FactualRecord(
                grid=previous.grid,
                incoming_raw4=previous.incoming_raw4,
                outgoing_raw4_or_none=outgoing.detach().clone(),
                valid=True,
                observation_key=previous.observation_key,
            )
            incoming = outgoing.detach().clone()
            next_verified_history = list(parsed_history)
            if not next_factual_identities:
                raise RuntimeError("factual identity owner lost its pending predecessor")
            prior = dict(next_factual_identities[-1])
            prior["executed_action"] = self._pending.name
            prior["outgoing_raw4_sha256"] = _tensor_sha256(outgoing)
            prior["action_axes_sha256"] = _axes_sha256(
                previous.incoming_raw4, outgoing
            )
            next_factual_identities[-1] = prior

        next_records.append(
            FactualRecord(
                grid=current_grid,
                incoming_raw4=incoming.detach().clone(),
                outgoing_raw4_or_none=None,
                valid=True,
                observation_key=current_key,
            )
        )
        next_records = next_records[-4:]
        formal_rgb_sha = _formal_rgb_sha256(current_rgb)
        next_factual_identities.append(
            {
                "order": len(parsed_history),
                "rgb_sha256": formal_rgb_sha,
                "grid_sha256": _tensor_sha256(current_grid),
                "incoming_raw4_sha256": _tensor_sha256(incoming),
                "outgoing_raw4_sha256": None,
                "action_axes_sha256": _axes_sha256(incoming, None),
                "executed_action": None,
            }
        )

        state = build_factual_decision_state(
            next_records,
            goal_grid=self._goal_grid,
            step=len(parsed_history),
            goal_observation_key=self._goal_grid_key,
        )
        queued = self._queued_execution
        result = self._planner.plan(state) if queued is None else queued[0]
        if isinstance(result, PolicyDecision):
            raw_action = result.action_id
            diagnostics = result.diagnostics
        elif isinstance(result, Mapping):
            raw_action = result.get("action_id", result.get("action"))
            diagnostics = result.get("diagnostics", {})
        else:
            raw_action = result
            diagnostics = {}
        action = _as_action(raw_action)
        if not isinstance(diagnostics, Mapping):
            raise TypeError("planner diagnostics must be a mapping")
        diagnostic_row = copy.deepcopy(dict(diagnostics))
        diagnostic_row.setdefault("step", len(parsed_history))
        diagnostic_row.setdefault("action_id", int(action))
        next_queue = (queued[1:] or None) if queued is not None else None
        if self._execution_interval > 1:
            if queued is None:
                branch = diagnostics.get("whole_branch_ranking")
                try:
                    guard = branch.get("stagnation_guard") or {}
                    if guard.get("selection_source") == "fallback_turns":
                        fallback = guard["fallback"]
                        if (branch["winner_k"] is not None or not guard["triggered"]
                                or fallback["horizon"] != 1
                                or fallback["actions"][fallback["selected_index"]] != int(action)
                                or guard["selected_action"] != int(action)
                                or action not in (ActionId.LEFT, ActionId.RIGHT)):
                            raise ValueError("invalid single-turn fallback")
                        winner, sequence = None, [action]
                    else:
                        modes = branch["mode_indices"][0]
                        winner = branch["winner_k"][0]
                        if modes.count(winner) != 1:
                            raise ValueError("winner must identify exactly one branch")
                        sequence = branch["actions"][0][modes.index(winner)]
                        if len(sequence) < self._execution_interval:
                            raise ValueError("winner shorter than execution interval")
                        sequence = sequence[:self._execution_interval]
                    sequence = [_as_action(value) for value in sequence]
                    if sequence[0] != action:
                        raise ValueError("winner sequence does not match first action")
                    if any(value is ActionId.STOP for value in sequence):
                        raise ValueError("committed winner must contain motion primitives")
                except (KeyError, IndexError, TypeError, AttributeError) as exc:
                    raise ValueError("sequence execution requires a complete whole-branch winner") from exc
                metadata = dict(execution_interval=self._execution_interval,
                    plan_execution_length=len(sequence), planning_step_index=self._planning_count,
                    plan_origin_real_step=len(parsed_history))
                diagnostic_row.update(metadata, real_step_index=len(parsed_history),
                                      execution_offset=0, replanned=True)
                next_queue = [PolicyDecision(int(value), dict(metadata,
                    real_step_index=len(parsed_history)+offset, execution_offset=offset, replanned=False,
                    canonical_stop_distance=diagnostics.get("canonical_stop_distance"),
                    whole_branch_ranking=None, whole_branch_ranking_status="reused_plan",
                    winner_k=winner, stop_category="not-entered"))
                    for offset, value in enumerate(sequence[1:], 1)] or None
            # The queued object contains only action and scalar provenance, never imagined grids.


        # The planner result has now passed every fallible validation.  Commit
        # all adapter-owned episode fields as one transaction.
        self._queued_execution = next_queue
        if queued is None:
            self._planning_count += 1
        self._records = next_records
        self._verified_history = next_verified_history
        self._factual_identities = next_factual_identities
        if len(self._diagnostics) == self._diagnostics.maxlen:
            self._diagnostics_dropped += 1
        self._diagnostics.append(MappingProxyType(diagnostic_row))
        if action is ActionId.STOP:
            self._terminal = True
            self._pending = None
        else:
            self._pending = action
        return action

    def diagnostic_goal_stop_scores(self) -> dict[str, object]:
        """Score existing shared G after a fixed action; never change the policy."""
        if self._closed or not self._records or self._goal_grid is None:
            raise RuntimeError("goal STOP probe requires a committed factual state")
        model = self._planner.model
        if any(module.training for module in model.modules()):
            raise ValueError("goal STOP probe requires all modules in eval mode")
        devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            state = build_factual_decision_state(
                self._records, goal_grid=self._goal_grid,
                step=len(self._verified_history), goal_observation_key=self._goal_grid_key)
            prepared = self._planner._prepare_state(state, compute_stop_distance=False)
            current = prepared["grids"][-1:].detach().clone()
            intent = prepared["goal"][None].detach().clone() - current
            previous = prepared["previous"][None].detach().clone()
            result = {}
            for label, delta in (("goal_intent", intent), ("zero_intent", torch.zeros_like(intent))):
                logits = model.actor_logits(current.clone(), delta, previous.clone())
                if not isinstance(logits, Tensor) or logits.shape != (1, 4):
                    raise ValueError("goal STOP probe requires four actor logits")
                logits = logits.detach().float()
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError("goal STOP probe logits must be finite")
                result[label] = {
                    "logits": logits[0].cpu().tolist(),
                    "probabilities": logits.softmax(-1)[0].cpu().tolist(),
                    "stop_margin": float((logits[0, 0] - logits[0, 1:].max()).item()),
                }
        return result

    def latest_step_diagnostics(self) -> Mapping[str, object]:
        if not self._diagnostics:
            raise RuntimeError("adapter has no completed decision diagnostics")
        return MappingProxyType(copy.deepcopy(dict(self._diagnostics[-1])))

    def factual_state_receipt(self) -> Mapping[str, object]:
        """Return hash/scalar-only evidence for the last committed real decision."""

        if not self._factual_identities or self._goal_formal_sha256 is None:
            raise RuntimeError("adapter has no committed factual decision state")
        context = copy.deepcopy(self._factual_identities[-len(self._records) :])
        verified = copy.deepcopy(self._factual_identities[:-1])
        receipt = {
            "schema": "J2J_CONTEXT4_FACTUAL_STATE_RECEIPT_V1",
            "step": len(self._verified_history),
            "goal_rgb_sha256": self._goal_formal_sha256,
            "verified_prefix": verified,
            "context_record_count": len(context),
            "context_records": context,
        }
        # A strict JSON round-trip both proves the no-tensor ABI and returns a
        # detached object graph that observers may mutate without affecting us.
        return copy.deepcopy(
            json.loads(
                json.dumps(
                    receipt, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            )
        )

    def build_replayed_decision_state(
        self,
        factual_rgbs: Sequence[Any],
        goal_rgb: Any,
        executed_actions: Sequence[Any],
    ) -> DecisionState:
        """Rebuild a transient state with the same encoder/action ABI as ``act``."""

        if (
            isinstance(factual_rgbs, (str, bytes))
            or not isinstance(factual_rgbs, Sequence)
            or not factual_rgbs
        ):
            raise ValueError("replayed factual RGB sequence must be nonempty")
        if isinstance(executed_actions, (str, bytes)) or not isinstance(
            executed_actions, Sequence
        ):
            raise TypeError("replayed action prefix must be a sequence")
        actions = tuple(_as_action(action) for action in executed_actions)
        if len(factual_rgbs) != len(actions) + 1:
            raise ValueError("replayed RGB/action prefix lengths disagree")
        grids = tuple(
            _validate_grid(self._encode_grid(rgb), name="replayed factual")
            for rgb in factual_rgbs
        )
        goal = _validate_grid(self._encode_grid(goal_rgb), name="replayed goal")
        records: list[FactualRecord] = []
        for index, (rgb, grid) in enumerate(zip(factual_rgbs, grids, strict=True)):
            incoming = (
                Raw4Adapter.encode_bos().to(grid.device)
                if index == 0
                else Raw4Adapter.encode(actions[index - 1]).to(grid.device)
            )
            outgoing = (
                None
                if index == len(actions)
                else Raw4Adapter.encode(actions[index]).to(grid.device)
            )
            records.append(
                FactualRecord(
                    grid=grid,
                    incoming_raw4=incoming,
                    outgoing_raw4_or_none=outgoing,
                    valid=True,
                    observation_key=_observation_key(rgb),
                )
            )
        return build_factual_decision_state(
            records[-4:], goal_grid=goal, step=len(actions)
        )

    def encode_grid_for_evidence(self, rgb: Any) -> Tensor:
        """Encode one real endpoint without mutating the policy state owner."""

        return _validate_grid(self._encode_grid(rgb), name="evidence endpoint")

    def plan_replayed_candidates(
        self, state: DecisionState, *, stop_category: str
    ) -> Mapping[str, object]:
        from .planner import _plan_local_candidates

        return _plan_local_candidates(
            self._planner, state, stop_category=stop_category
        )

    def drain_step_diagnostics(self) -> tuple[Mapping[str, object], ...]:
        rows = tuple(dict(row) for row in self._diagnostics)
        self._diagnostics.clear()
        self._diagnostics_dropped = 0
        return rows

    def diagnostic_buffer_status(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "capacity": int(self._diagnostics.maxlen or 0),
                "retained": len(self._diagnostics),
                "dropped": self._diagnostics_dropped,
            }
        )

    def provenance_record(self) -> Mapping[str, object]:
        value = copy.deepcopy(self._base_provenance)
        value.update(
            {
                "stop_mode": self._stop_mode,
                "stop_receipt_present": self._stop_receipt is not None,
                "status": (
                    "DIAGNOSTIC"
                    if self._base_provenance.get("stop_threshold_policy", {}).get("kind") in {"diagnostic_override", "diagnostic_reach_only"}
                    else "FORMAL"
                    if self._stop_mode == "calibrated" and self._stop_receipt is not None
                    else "PARTIAL"
                ),
            }
        )
        return MappingProxyType(value)

    def close(self) -> None:
        self._closed = True
        self._queued_execution = None
        self._planning_count = 0
        self._records.clear()
        self._factual_identities.clear()
        self._pending = None
        self._goal_grid = None
        self._goal_formal_sha256 = None


__all__ = [
    "Context4PolicyAdapter",
    "DecisionState",
    "FactualRecord",
    "PolicyDecision",
    "build_factual_decision_state",
]
