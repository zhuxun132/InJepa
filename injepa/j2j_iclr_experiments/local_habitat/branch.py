"""Pure branch-executor contract used before any Habitat-specific adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import math
import re
from typing import Any

import torch

from j2j.receipts import canonical_json_bytes
from j2j_iclr_experiments.common.artifacts import (
    deep_freeze,
    to_plain_json,
    validate_file_identity,
)
from j2j_iclr_experiments.offline.metrics import grid_error_metrics
from .capability import _array_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_MOTION_ACTIONS = frozenset({"FWD", "LEFT", "RIGHT"})
_IDENTITY_FIELDS = {
    "schema",
    "scene",
    "episode",
    "step",
    "source_checkpoint_sha256",
    "source_policy_sha256",
    "source_code_sha256",
    "executed_action_prefix",
    "goal_identity",
    "factual_steps",
    "simulator_sha256",
    "runtime_sha256",
    "sensor_sha256",
    "episode_ledger_sha256",
    "stop_receipt_sha256",
}
_FACTUAL_STEP_FIELDS = {
    "order",
    "rgb_sha256",
    "context_sha256",
    "action_axes_sha256",
    "position",
    "rotation",
}


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    descriptor = canonical_json_bytes(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    )
    return hashlib.sha256(descriptor + tensor.numpy().tobytes(order="C")).hexdigest()


def _rgb_sha256(value: object) -> str:
    return hashlib.sha256(_array_bytes(value, name="RGB")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"local state {name} SHA-256 identity is invalid")
    return value


def _finite_vector(value: object, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"local state factual {name} must have length {length}")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"local state factual {name} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"local state factual {name} must be finite")
        result.append(number)
    return tuple(result)


def _motion_prefix(value: object, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"local state {name} action prefix must be a sequence")
    result = tuple(value)
    if not allow_empty and not result:
        raise ValueError(f"local state {name} action prefix must be nonempty")
    if any(type(action) is not str or action not in _MOTION_ACTIONS for action in result):
        raise ValueError(
            f"local state {name} action prefix must contain only canonical motion actions"
        )
    return result


def _validate_state_identity(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_FIELDS:
        raise ValueError("local state ledger identity fields are incomplete or noncanonical")
    if value.get("schema") != "J2J_CONTEXT4_LOCAL_STATE_V1":
        raise ValueError("local state ledger schema is invalid")
    for name in ("scene", "episode"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError(f"local state {name} identity is invalid")
    step = value.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("local state step must be a non-negative integer")
    for name in (
        "source_checkpoint_sha256",
        "source_policy_sha256",
        "source_code_sha256",
        "simulator_sha256",
        "runtime_sha256",
        "sensor_sha256",
        "episode_ledger_sha256",
        "stop_receipt_sha256",
    ):
        _sha256(value.get(name), name=name)

    executed = _motion_prefix(
        value.get("executed_action_prefix"),
        name="executed",
        allow_empty=True,
    )
    if len(executed) != step:
        raise ValueError("local state executed action prefix is incomplete for its step")

    goal = value.get("goal_identity")
    if not isinstance(goal, Mapping) or set(goal) != {"rgb_sha256"}:
        raise ValueError("local state goal identity fields are incomplete")
    _sha256(goal.get("rgb_sha256"), name="goal RGB")

    factual = value.get("factual_steps")
    if not isinstance(factual, (list, tuple)) or len(factual) != step + 1:
        raise ValueError("local state factual steps are incomplete")
    for expected_order, row in enumerate(factual):
        if not isinstance(row, Mapping) or set(row) != _FACTUAL_STEP_FIELDS:
            raise ValueError("local state factual step fields are incomplete")
        if row.get("order") != expected_order:
            raise ValueError("local state factual step order is not contiguous")
        for name in ("rgb_sha256", "context_sha256", "action_axes_sha256"):
            _sha256(row.get(name), name=f"factual {name}")
        _finite_vector(row.get("position"), length=3, name="position")
        _finite_vector(row.get("rotation"), length=4, name="rotation")
    return value


def evaluate_branches(
    executor,
    state: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    selected_key: str,
) -> Mapping[str, object]:
    """Reset/replay every candidate independently and compute exact regret."""

    reset_and_replay = getattr(executor, "reset_and_replay", None)
    execute_prefix = getattr(executor, "execute_prefix", None)
    if not callable(reset_and_replay) or not callable(execute_prefix):
        raise TypeError("branch executor must expose reset_and_replay and execute_prefix")
    if not isinstance(state, Mapping):
        raise ValueError("branch state must be a mapping")
    identity = _validate_state_identity(state.get("identity"))
    if not isinstance(candidates, Sequence) or not candidates:
        raise ValueError("candidate sequence must be nonempty")
    normalized: list[tuple[str, tuple[Any, ...]]] = []
    seen: set[str] = set()
    for row in candidates:
        if not isinstance(row, Mapping) or "key" not in row or "actions" not in row:
            raise ValueError("candidate row is incomplete")
        key = str(row["key"])
        if key in seen:
            raise ValueError("candidate key is duplicated")
        seen.add(key)
        actions = _motion_prefix(
            row["actions"], name=f"candidate {key}", allow_empty=False
        )
        normalized.append((key, actions))
    if selected_key not in seen:
        raise KeyError("selected candidate is absent")

    expected_identity = copy.deepcopy(dict(identity))
    progress: dict[str, float] = {}
    endpoint_keys: dict[str, object] = {}
    for key, actions in normalized:
        branch_state = copy.deepcopy(dict(state))
        observed_identity = reset_and_replay(branch_state)
        if not isinstance(observed_identity, Mapping) or dict(observed_identity) != expected_identity:
            raise RuntimeError("reset/replay identity mismatch")
        result = execute_prefix(actions)
        if not isinstance(result, Mapping) or "actual_progress" not in result:
            raise ValueError("branch result has no actual progress")
        value = result["actual_progress"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("branch progress must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise FloatingPointError("branch progress must be finite")
        progress[key] = value
        endpoint_keys[key] = result.get("endpoint_key")

    selected_progress = progress[selected_key]
    maximum = max(progress.values())
    return {
        "progress_by_candidate": progress,
        "endpoint_key_by_candidate": endpoint_keys,
        "selected_progress": selected_progress,
        "max_progress": maximum,
        "top1_regret": maximum - selected_progress,
    }


def _branch_key(value: object) -> tuple[str, int, int]:
    if not isinstance(value, Mapping) or set(value) != {
        "method", "global_k", "one_based_h"
    }:
        raise ValueError("formal branch candidate key is incomplete")
    method = value.get("method")
    global_k = value.get("global_k")
    horizon = value.get("one_based_h")
    if method not in {"Full", "NoG-enumerate"}:
        raise ValueError("formal branch candidate method is invalid")
    if type(global_k) is not int or global_k < 0:
        raise ValueError("formal branch candidate global_k is invalid")
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("formal branch candidate horizon is invalid")
    return str(method), global_k, horizon


def _invalid_branch_result(
    candidate: Mapping[str, Any], *, selected_key: Mapping[str, Any]
) -> dict[str, Any]:
    key = copy.deepcopy(dict(candidate["candidate_key"]))
    return {
        "candidate_key": key,
        "actions": copy.deepcopy(list(candidate["actions"])),
        "status": "INVALID",
        "start_distance": None,
        "end_distance": None,
        "actual_progress": None,
        "endpoint_rgb_sha256": None,
        "endpoint_grid_sha256": None,
        "q_error": None,
        "f_error": None,
        "consistency": None,
        "selected": key == dict(selected_key),
    }


def evaluate_candidate_plan_state(
    executor: object,
    adapter: object,
    state: Mapping[str, Any],
    candidate_plan_state: Mapping[str, Any],
    transient_endpoints: Mapping[tuple[str, int, int], Mapping[str, torch.Tensor]],
    *,
    cosine_epsilon: float,
) -> dict[str, Any]:
    """Execute a complete frozen two-method plan and invalidate atomically.

    Q/G/F tensors are supplied from Stage 2a and never regenerated here.  If
    any replay or branch fails, every scientific row for that source state is
    nulled while all frozen candidate keys/actions remain present.
    """

    identity = _validate_state_identity(state.get("identity"))
    if not isinstance(candidate_plan_state, Mapping) or set(candidate_plan_state) != {
        "state_key", "source_full_action", "stop_category", "shared_counters",
        "methods",
    }:
        raise ValueError("formal candidate-plan state is incomplete")
    if dict(candidate_plan_state["state_key"]) != dict(state.get("state_key", {})):
        raise ValueError("formal branch state/candidate-plan key mismatch")
    methods = candidate_plan_state.get("methods")
    if not isinstance(methods, list) or [row.get("method") for row in methods] != [
        "Full", "NoG-enumerate"
    ]:
        raise ValueError("formal branch methods are incomplete or out of order")
    if not isinstance(cosine_epsilon, (int, float)) or isinstance(cosine_epsilon, bool):
        raise TypeError("formal branch cosine epsilon must be numeric")
    epsilon = float(cosine_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("formal branch cosine epsilon must be finite and positive")

    reset_and_replay = getattr(executor, "reset_and_replay", None)
    execute_prefix = getattr(executor, "execute_prefix", None)
    encode_endpoint = getattr(adapter, "encode_grid_for_evidence", None)
    if not all(callable(value) for value in (reset_and_replay, execute_prefix, encode_endpoint)):
        raise TypeError("formal branch path requires canonical replay and encoder methods")

    normalized: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for method_row in methods:
        if not isinstance(method_row, Mapping):
            raise TypeError("formal branch method row must be a mapping")
        selected_key = method_row.get("selected_key")
        _branch_key(selected_key)
        candidates = method_row.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("formal branch candidate set must be complete and nonempty")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise TypeError("formal branch candidate must be a mapping")
            key = _branch_key(candidate.get("candidate_key"))
            if key not in transient_endpoints:
                raise ValueError("formal branch transient Q/F endpoint is absent")
            endpoints = transient_endpoints[key]
            if not isinstance(endpoints, Mapping) or set(endpoints) != {
                "q_endpoint", "f_endpoint"
            }:
                raise ValueError("formal branch transient endpoint pair is incomplete")
            normalized.append((method_row, candidate))

    valid_by_method: dict[str, list[dict[str, Any]]] = {
        "Full": [],
        "NoG-enumerate": [],
    }
    invalid_reason: str | None = None
    for method_row, candidate in normalized:
        method, global_k, horizon = _branch_key(candidate["candidate_key"])
        actions = _motion_prefix(
            candidate.get("actions"),
            name=f"formal {method}/{global_k}/{horizon}",
            allow_empty=False,
        )
        selected_key = method_row["selected_key"]
        try:
            observed = reset_and_replay(copy.deepcopy(dict(state)))
            if not isinstance(observed, Mapping) or dict(observed) != dict(identity):
                raise RuntimeError("formal reset/replay identity mismatch")
            result = execute_prefix(actions)
            if not isinstance(result, Mapping):
                raise TypeError("formal branch executor returned no result mapping")
            start = _finite_branch_scalar(result.get("start_distance"), "start distance")
            end = _finite_branch_scalar(result.get("end_distance"), "end distance")
            progress = _finite_branch_scalar(result.get("actual_progress"), "actual progress")
            if not math.isclose(start - end, progress, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("formal branch progress disagrees with official distances")
            endpoint_rgb = result.get("endpoint_rgb")
            rgb_sha = _rgb_sha256(endpoint_rgb)
            if result.get("endpoint_rgb_sha256") != rgb_sha:
                raise RuntimeError("formal branch endpoint RGB identity mismatch")
            endpoint_grid = encode_endpoint(endpoint_rgb)
            if not isinstance(endpoint_grid, torch.Tensor):
                raise TypeError("formal branch endpoint encoder returned no tensor")
            endpoints = transient_endpoints[(method, global_k, horizon)]
            q_error = _grid_error_json(
                endpoint_grid, endpoints["q_endpoint"], epsilon=epsilon
            )
            f_error = _grid_error_json(
                endpoint_grid, endpoints["f_endpoint"], epsilon=epsilon
            )
            consistency = candidate.get("consistency")
            if consistency is not None:
                consistency = _finite_branch_scalar(consistency, "consistency")
            row = {
                "candidate_key": copy.deepcopy(dict(candidate["candidate_key"])),
                "actions": list(actions),
                "status": "VALID",
                "start_distance": start,
                "end_distance": end,
                "actual_progress": progress,
                "endpoint_rgb_sha256": rgb_sha,
                "endpoint_grid_sha256": _tensor_sha256(endpoint_grid),
                "q_error": q_error,
                "f_error": f_error,
                "consistency": consistency,
                "selected": dict(candidate["candidate_key"]) == dict(selected_key),
            }
            valid_by_method[method].append(row)
        except Exception as exc:
            invalid_reason = f"{type(exc).__name__}: {exc}"
            break

    if invalid_reason is not None:
        return {
            "state_key": copy.deepcopy(dict(state["state_key"])),
            "status": "INVALID",
            "invalid_reason": invalid_reason,
            "methods": [
                {
                    "method": str(method_row["method"]),
                    "complete_candidate_set": False,
                    "selected_key": copy.deepcopy(dict(method_row["selected_key"])),
                    "selected_progress": None,
                    "max_progress": None,
                    "top1_regret": None,
                    "results": [
                        _invalid_branch_result(
                            candidate, selected_key=method_row["selected_key"]
                        )
                        for candidate in method_row["candidates"]
                    ],
                }
                for method_row in methods
            ],
        }

    output_methods: list[dict[str, Any]] = []
    for method_row in methods:
        method = str(method_row["method"])
        rows = valid_by_method[method]
        if len(rows) != len(method_row["candidates"]):
            raise RuntimeError("formal branch candidate set completed incompletely")
        selected_rows = [row for row in rows if row["selected"]]
        if len(selected_rows) != 1:
            raise RuntimeError("formal branch selected candidate is not unique")
        selected_progress = float(selected_rows[0]["actual_progress"])
        maximum = max(float(row["actual_progress"]) for row in rows)
        output_methods.append(
            {
                "method": method,
                "complete_candidate_set": True,
                "selected_key": copy.deepcopy(dict(method_row["selected_key"])),
                "selected_progress": selected_progress,
                "max_progress": maximum,
                "top1_regret": maximum - selected_progress,
                "results": rows,
            }
        )
    return {
        "state_key": copy.deepcopy(dict(state["state_key"])),
        "status": "VALID",
        "invalid_reason": None,
        "methods": output_methods,
    }


def _finite_branch_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"formal branch {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError(f"formal branch {name} must be finite")
    return number


def _grid_error_json(
    observed: torch.Tensor, predicted: torch.Tensor, *, epsilon: float
) -> dict[str, float]:
    if not isinstance(predicted, torch.Tensor):
        raise TypeError("formal branch predicted endpoint must be a tensor")
    values = grid_error_metrics(observed, predicted, epsilon=epsilon)
    result: dict[str, float] = {}
    for name in ("mse", "mae", "cosine_distance"):
        tensor = values[name]
        if tensor.numel() != 1:
            raise ValueError("formal branch endpoint metric must be scalar")
        result[name] = _finite_branch_scalar(float(tensor.item()), name)
    return result


def _file_identity(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"formal branch {name} FileIdentity is incomplete")
    path = value.get("path")
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"formal branch {name} path must be absolute")
    if type(value.get("bytes")) is not int or value["bytes"] <= 0:
        raise ValueError(f"formal branch {name} byte count is invalid")
    _sha256(value.get("sha256"), name=f"branch {name}")


def _validate_upstream_identities(
    value: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]] | None,
    *,
    fields: Sequence[str],
) -> None:
    if expected is None:
        return
    if not isinstance(expected, Mapping) or set(expected) != set(fields):
        raise ValueError("formal branch upstream identity set is incomplete or unknown")
    for name in fields:
        embedded = value.get(name)
        _file_identity(embedded, name=name)
        live = validate_file_identity(expected[name], name=f"{name} upstream")
        if to_plain_json(embedded) != to_plain_json(live):
            raise ValueError(
                f"formal branch {name} upstream identity differs from live bytes"
            )


def _validate_error(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "mse", "mae", "cosine_distance"
    }:
        raise ValueError(f"formal branch {name} error mapping is incomplete")
    for field in ("mse", "mae", "cosine_distance"):
        _finite_branch_scalar(value.get(field), f"{name} {field}")


def validate_branch_result_ledger(
    value: Mapping[str, Any],
    *,
    candidate_plan: Mapping[str, Any] | None = None,
    expected_upstream_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Validate exact branch rows and, when supplied, their frozen plan."""

    required = {
        "schema", "status", "state_ledger", "candidate_plan", "decision",
        "analysis_manifest", "checkpoint", "stop_receipt", "capability_receipt",
        "episode_ledger", "visual_coordinate_sha256", "counts", "states",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("formal branch-result ledger fields are incomplete")
    if value.get("schema") != "J2J_CONTEXT4_LOCAL_BRANCH_RESULTS_V1":
        raise ValueError("formal branch-result schema is not current")
    if value.get("status") not in {"COMPLETE", "PARTIAL"}:
        raise ValueError("formal branch-result status is invalid")
    upstream_fields = (
        "state_ledger", "candidate_plan", "decision", "analysis_manifest",
        "checkpoint", "stop_receipt", "capability_receipt", "episode_ledger",
    )
    for name in upstream_fields:
        _file_identity(value.get(name), name=name)
    _validate_upstream_identities(
        value,
        expected_upstream_identities,
        fields=upstream_fields,
    )
    _sha256(value.get("visual_coordinate_sha256"), name="visual coordinate")
    states = value.get("states")
    if not isinstance(states, list):
        raise TypeError("formal branch-result states must be a list")
    planned_states: list[Any] | None = None
    if candidate_plan is not None:
        if not isinstance(candidate_plan, Mapping) or not isinstance(
            candidate_plan.get("states"), list
        ):
            raise ValueError("formal branch candidate plan is incomplete")
        planned_states = candidate_plan["states"]
        if len(planned_states) != len(states):
            raise ValueError("formal branch state count differs from candidate plan")

    valid_states = valid_candidates = total_candidates = 0
    for state_index, state in enumerate(states):
        if not isinstance(state, Mapping) or set(state) != {
            "state_key", "status", "invalid_reason", "methods"
        }:
            raise ValueError("formal branch state fields are incomplete")
        state_status = state.get("status")
        if state_status not in {"VALID", "INVALID"}:
            raise ValueError("formal branch state status is invalid")
        if state_status == "VALID":
            if state.get("invalid_reason") is not None:
                raise ValueError("valid branch state cannot carry an invalid reason")
            valid_states += 1
        elif not isinstance(state.get("invalid_reason"), str) or not state["invalid_reason"]:
            raise ValueError("invalid branch state requires a reason")
        methods = state.get("methods")
        if not isinstance(methods, list) or [row.get("method") for row in methods] != [
            "Full", "NoG-enumerate"
        ]:
            raise ValueError("formal branch methods are incomplete or out of order")
        planned_methods = None
        if planned_states is not None:
            planned = planned_states[state_index]
            if (
                not isinstance(planned, Mapping)
                or dict(planned.get("state_key", {})) != dict(state["state_key"])
                or not isinstance(planned.get("methods"), list)
            ):
                raise ValueError("formal branch state key differs from candidate plan")
            planned_methods = planned["methods"]
        for method_index, method in enumerate(methods):
            if not isinstance(method, Mapping) or set(method) != {
                "method", "complete_candidate_set", "selected_key",
                "selected_progress", "max_progress", "top1_regret", "results",
            }:
                raise ValueError("formal branch method fields are incomplete")
            results = method.get("results")
            if not isinstance(results, list) or not results:
                raise ValueError("formal branch method has no frozen candidate rows")
            planned_candidates = None
            if planned_methods is not None:
                planned_method = planned_methods[method_index]
                if (
                    not isinstance(planned_method, Mapping)
                    or planned_method.get("method") != method.get("method")
                    or dict(planned_method.get("selected_key", {}))
                    != dict(method.get("selected_key", {}))
                    or not isinstance(planned_method.get("candidates"), list)
                ):
                    raise ValueError("formal branch method differs from candidate plan")
                planned_candidates = planned_method["candidates"]
                if len(planned_candidates) != len(results):
                    raise ValueError("formal branch candidate count differs from plan")
            selected_rows: list[Mapping[str, Any]] = []
            for result_index, result in enumerate(results):
                if not isinstance(result, Mapping) or set(result) != {
                    "candidate_key", "actions", "status", "start_distance",
                    "end_distance", "actual_progress", "endpoint_rgb_sha256",
                    "endpoint_grid_sha256", "q_error", "f_error", "consistency",
                    "selected",
                }:
                    raise ValueError("formal branch result fields are incomplete")
                _branch_key(result.get("candidate_key"))
                _motion_prefix(
                    result.get("actions"), name="formal branch", allow_empty=False
                )
                if type(result.get("selected")) is not bool:
                    raise TypeError("formal branch selected flag must be boolean")
                if result["selected"]:
                    selected_rows.append(result)
                if planned_candidates is not None:
                    frozen = planned_candidates[result_index]
                    if (
                        not isinstance(frozen, Mapping)
                        or dict(result["candidate_key"])
                        != dict(frozen.get("candidate_key", {}))
                        or list(result["actions"]) != list(frozen.get("actions", []))
                    ):
                        raise ValueError("formal branch result changed frozen candidate")
                    frozen_consistency = frozen.get("consistency")
                    if state_status == "VALID" and result.get(
                        "consistency"
                    ) != frozen_consistency:
                        raise ValueError(
                            "formal branch consistency changed frozen candidate plan"
                        )
                if state_status == "VALID":
                    if result.get("status") != "VALID":
                        raise ValueError("valid branch state contains an invalid candidate")
                    start = _finite_branch_scalar(result.get("start_distance"), "start distance")
                    end = _finite_branch_scalar(result.get("end_distance"), "end distance")
                    progress = _finite_branch_scalar(result.get("actual_progress"), "actual progress")
                    if not math.isclose(start - end, progress, rel_tol=1e-6, abs_tol=1e-6):
                        raise ValueError("formal branch result progress drifted")
                    _sha256(result.get("endpoint_rgb_sha256"), name="endpoint RGB")
                    _sha256(result.get("endpoint_grid_sha256"), name="endpoint grid")
                    _validate_error(result.get("q_error"), name="Q")
                    _validate_error(result.get("f_error"), name="F")
                    if method["method"] == "Full":
                        _finite_branch_scalar(result.get("consistency"), "consistency")
                    elif result.get("consistency") is not None:
                        raise ValueError("NoG branch consistency must be null")
                    valid_candidates += 1
                else:
                    if result.get("status") != "INVALID":
                        raise ValueError("invalid branch state contains a valid candidate")
                    for field in (
                        "start_distance", "end_distance", "actual_progress",
                        "endpoint_rgb_sha256", "endpoint_grid_sha256", "q_error",
                        "f_error", "consistency",
                    ):
                        if result.get(field) is not None:
                            raise ValueError("invalid branch result retained scientific output")
            total_candidates += len(results)
            if len(selected_rows) != 1 or dict(selected_rows[0]["candidate_key"]) != dict(
                method.get("selected_key", {})
            ):
                raise ValueError("formal branch selected candidate identity drifted")
            if state_status == "VALID":
                if method.get("complete_candidate_set") is not True:
                    raise ValueError("valid branch method must be complete")
                progresses = [float(row["actual_progress"]) for row in results]
                selected_progress = float(selected_rows[0]["actual_progress"])
                maximum = max(progresses)
                if not all(
                    math.isclose(float(method[name]), expected, rel_tol=1e-6, abs_tol=1e-6)
                    for name, expected in (
                        ("selected_progress", selected_progress),
                        ("max_progress", maximum),
                        ("top1_regret", maximum - selected_progress),
                    )
                ):
                    raise ValueError("formal branch regret summary drifted")
            elif (
                method.get("complete_candidate_set") is not False
                or any(
                    method.get(name) is not None
                    for name in ("selected_progress", "max_progress", "top1_regret")
                )
            ):
                raise ValueError("invalid branch method retained a scientific summary")
    counts = value.get("counts")
    expected_counts = {
        "states": len(states),
        "valid_states": valid_states,
        "invalid_states": len(states) - valid_states,
        "candidates": total_candidates,
        "valid_candidates": valid_candidates,
    }
    if not isinstance(counts, Mapping) or dict(counts) != expected_counts:
        raise ValueError("formal branch-result counts drifted")
    expected_status = "COMPLETE" if valid_states == len(states) else "PARTIAL"
    if value.get("status") != expected_status:
        raise ValueError("formal branch-result root status drifted")
    frozen = deep_freeze(to_plain_json(value))
    assert isinstance(frozen, Mapping)
    return frozen


__all__ = [
    "evaluate_branches",
    "evaluate_candidate_plan_state",
    "validate_branch_result_ledger",
]
