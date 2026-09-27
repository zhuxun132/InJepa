"""Exact scalar contracts for current Context4 local state evidence."""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

from collections.abc import Mapping, Sequence
import copy
import hashlib
import math
import re
from types import MappingProxyType
from typing import Any

from j2j.receipts import canonical_json_bytes
from j2j_iclr_experiments.common.artifacts import (
    deep_freeze,
    to_plain_json,
    validate_file_identity,
)
from j2j_iclr_experiments.common.identity import FULL_V1_VARIANT_ID

from .branch import _validate_state_identity
from .capability import _array_bytes


SELECTION_DOMAIN = "J2J_CONTEXT4_LOCAL_STATE_SELECTION_V1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METHODS = ("Full", "NoG-enumerate")
_MOTION = ("FWD", "LEFT", "RIGHT")
_TITLE = "conditional on Full/seed3072/epoch30 visited-state distribution"
_STATE_ROOT_FIELDS = {
    "schema",
    "status",
    "title",
    "source",
    "budgets",
    "selection",
    "decision",
    "analysis_manifest",
    "evaluation_overlay",
    "sensor_config",
    "preflight_receipt",
    "checkpoint",
    "training_sidecar",
    "stop_receipt",
    "capability_receipt",
    "episode_ledger",
    "ranker_identity",
    "candidate_contract",
    "states",
}


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _state_key(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"scene", "episode", "step"}:
        raise ValueError("state_key must contain exactly scene, episode and step")
    for field in ("scene", "episode"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"state_key {field} must be a nonempty string")
    _integer(value.get("step"), name="state_key step")
    return {"scene": value["scene"], "episode": value["episode"], "step": value["step"]}


def selection_sha256(state_key: Mapping[str, object], *, domain: str = SELECTION_DOMAIN) -> str:
    if not isinstance(domain, str) or not domain:
        raise ValueError("selection domain must be nonempty")
    canonical = _state_key(state_key)
    return hashlib.sha256(domain.encode("utf-8") + canonical_json_bytes(canonical)).hexdigest()


def select_state_rows(
    eligible: Sequence[Mapping[str, Any]], *, budget: int, domain: str
) -> tuple[dict[str, Any], ...]:
    """Choose global minimum state hashes only after receiving the full universe."""

    if isinstance(eligible, (str, bytes)) or not isinstance(eligible, Sequence):
        raise TypeError("eligible states must be a sequence")
    if type(budget) is not int or budget <= 0:
        raise ValueError("state budget must be a positive integer")
    rows: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw in eligible:
        if not isinstance(raw, Mapping):
            raise TypeError("eligible state row must be a mapping")
        row = copy.deepcopy(dict(raw))
        key = _state_key(row.get("state_key"))
        token = (str(key["scene"]), str(key["episode"]), int(key["step"]))
        if token in seen:
            raise ValueError("eligible universe contains a duplicate state key")
        seen.add(token)
        rows.append((selection_sha256(key, domain=domain), row))
    rows.sort(key=lambda item: item[0])
    return tuple(row for _digest, row in rows[:budget])


def canonical_rgb_sha256(value: object) -> str:
    """Use the exact formal Habitat numeric-array framing."""

    return hashlib.sha256(_array_bytes(value, name="RGB")).hexdigest()


def validate_source_decision(
    *,
    action: object,
    diagnostics: Mapping[str, Any],
    k_active: int,
    h_active: int,
) -> dict[str, Any]:
    """Validate the three existing STOP/control-flow counter cases."""

    if type(k_active) is not int or k_active <= 0 or type(h_active) is not int or h_active <= 0:
        raise ValueError("active K/H must be positive integers")
    if action not in {*_MOTION, "STOP"}:
        raise ValueError("source action is not canonical")
    if not isinstance(diagnostics, Mapping):
        raise TypeError("source diagnostics must be a mapping")
    category = diagnostics.get("stop_category")
    cases = {
        "not-entered": {
            "action": "motion",
            "q_calls": 1,
            "g_calls": h_active,
            "g_rows": k_active * h_active,
            "f_calls": h_active,
            "f_rows": k_active * h_active,
            "candidate_rows": k_active * h_active,
        },
        "entered-motion": {
            "action": "motion",
            "q_calls": 1,
            "g_calls": h_active + 1,
            "g_rows": k_active * h_active + 1,
            "f_calls": h_active,
            "f_rows": k_active * h_active,
            "candidate_rows": k_active * h_active,
        },
        "executed": {
            "action": "STOP",
            "q_calls": 0,
            "g_calls": 1,
            "g_rows": 1,
            "f_calls": 0,
            "f_rows": 0,
            "candidate_rows": 0,
        },
    }
    if category not in cases:
        raise ValueError("source STOP category is invalid")
    expected = dict(cases[str(category)])
    action_kind = expected.pop("action")
    if (action == "STOP") != (action_kind == "STOP"):
        raise ValueError("source action conflicts with STOP category")
    counters = diagnostics.get("counters")
    if not isinstance(counters, Mapping) or dict(counters) != expected:
        raise ValueError("source Q/G/F counters drifted from the current planner")
    distance = _finite(
        diagnostics.get("canonical_stop_distance"), name="canonical STOP distance"
    )
    eligible = action != "STOP"
    winner_k = diagnostics.get("winner_k")
    winner_h = diagnostics.get("winner_h")
    if eligible:
        if type(winner_k) is not int or not 0 <= winner_k < k_active:
            raise ValueError("source winner_k is outside active K")
        if type(winner_h) is not int or not 1 <= winner_h <= h_active:
            raise ValueError("source winner_h is outside active H")
    elif winner_k is not None or winner_h is not None:
        raise ValueError("executed STOP cannot claim a motion winner")
    return {
        "eligible": eligible,
        "action": str(action),
        "stop_category": str(category),
        "canonical_stop_distance": distance,
        "winner_k": winner_k,
        "winner_h": winner_h,
        "counters": copy.deepcopy(expected),
    }


def validate_stage2a_full_counter_reconciliation(
    *,
    source_full_diagnostics: Mapping[str, Any],
    candidate_plan_state: Mapping[str, Any],
) -> Mapping[str, int]:
    """Prove Stage 2a regenerated the exact Stage-1 Full call graph."""

    if not isinstance(source_full_diagnostics, Mapping) or set(
        source_full_diagnostics
    ) != {
        "stop_category",
        "counters",
    }:
        raise ValueError("Stage-1 Full counter evidence is incomplete")
    if not isinstance(candidate_plan_state, Mapping):
        raise TypeError("Stage-2a candidate-plan state must be a mapping")
    category = source_full_diagnostics.get("stop_category")
    if category not in {"not-entered", "entered-motion"} or (
        candidate_plan_state.get("stop_category") != category
    ):
        raise ValueError("Stage-1/Stage-2a STOP category drifted")
    shared = candidate_plan_state.get("shared_counters")
    methods = candidate_plan_state.get("methods")
    if (
        not isinstance(shared, Mapping)
        or set(shared)
        != {"q_calls", "stop_probe_g_calls", "total_g_calls", "total_f_calls"}
        or not isinstance(methods, list)
    ):
        raise ValueError("Stage-2a counter evidence is incomplete")
    full_rows = [
        row
        for row in methods
        if isinstance(row, Mapping) and row.get("method") == "Full"
    ]
    if len(full_rows) != 1 or not isinstance(full_rows[0].get("counters"), Mapping):
        raise ValueError("Stage-2a Full counter evidence is not unique")
    full = full_rows[0]["counters"]
    if set(full) != {
        "q_calls",
        "g_calls",
        "g_rows",
        "f_calls",
        "f_rows",
        "candidate_rows",
        "tree_nodes_per_batch",
    }:
        raise ValueError("Stage-2a Full counter fields are incomplete")
    components = {
        "q_calls": shared["q_calls"],
        "stop_probe_g_calls": shared["stop_probe_g_calls"],
        "g_calls": full["g_calls"],
        "g_rows": full["g_rows"],
        "f_calls": full["f_calls"],
        "f_rows": full["f_rows"],
        "candidate_rows": full["candidate_rows"],
    }
    if any(type(value) is not int or value < 0 for value in components.values()):
        raise ValueError("Stage-2a Full counters are not non-negative integers")
    recomposed = {
        "q_calls": components["q_calls"],
        "g_calls": components["g_calls"] + components["stop_probe_g_calls"],
        "g_rows": components["g_rows"] + components["stop_probe_g_calls"],
        "f_calls": components["f_calls"],
        "f_rows": components["f_rows"],
        "candidate_rows": components["candidate_rows"],
    }
    source = source_full_diagnostics.get("counters")
    if not isinstance(source, Mapping) or dict(source) != recomposed:
        raise ValueError("Stage-1 and Stage-2a Full counters do not reconcile")
    frozen = deep_freeze(recomposed)
    assert isinstance(frozen, Mapping)
    return frozen


def _file_identity_shape(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} FileIdentity is incomplete")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ValueError(f"{name} path is invalid")
    if not value["path"].startswith("/"):
        raise ValueError(f"{name} path must be absolute")
    _integer(value.get("bytes"), name=f"{name} bytes", minimum=1)
    if not isinstance(value.get("sha256"), str) or _SHA256.fullmatch(value["sha256"]) is None:
        raise ValueError(f"{name} SHA is invalid")


def _validate_upstream_identities(
    value: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]] | None,
    *,
    fields: Sequence[str],
) -> None:
    """Bind a formal consumer to the actual create-once upstream bytes.

    Pure mechanics callers may omit ``expected``.  The formal orchestration
    path always supplies the exact upstream set, and every expected identity
    is rehashed from live bytes at this consumption boundary.
    """

    if expected is None:
        return
    if not isinstance(expected, Mapping) or set(expected) != set(fields):
        raise ValueError("formal upstream identity set is incomplete or unknown")
    for name in fields:
        embedded = value.get(name)
        _file_identity_shape(embedded, name=name)
        live = validate_file_identity(expected[name], name=f"{name} upstream")
        if to_plain_json(embedded) != to_plain_json(live):
            raise ValueError(f"{name} upstream identity differs from live bytes")


def _candidate_key(value: object, *, method: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"method", "global_k", "one_based_h"}:
        raise ValueError("candidate key fields are incomplete")
    if value.get("method") != method:
        raise ValueError("candidate method identity drifted")
    _integer(value.get("global_k"), name="candidate global_k")
    _integer(value.get("one_based_h"), name="candidate one_based_h", minimum=1)
    return dict(value)


def validate_candidate_plan_ledger(
    value: Mapping[str, Any],
    *,
    k_active: int,
    h_active: int,
    expected_state_keys: Sequence[Mapping[str, Any]],
    expected_upstream_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Validate complete Full/NoG plans before any branch outcome exists."""

    if type(k_active) is not int or k_active <= 0 or type(h_active) is not int or h_active <= 0:
        raise ValueError("active K/H must be positive integers")
    required = {
        "schema", "status", "state_ledger", "decision", "analysis_manifest",
        "checkpoint", "stop_receipt", "capability_receipt", "episode_ledger",
        "visual_coordinate_sha256", "counts", "states",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("candidate-plan ledger fields are incomplete or noncanonical")
    if value.get("schema") != "J2J_CONTEXT4_LOCAL_CANDIDATE_PLAN_V1":
        raise ValueError("candidate-plan schema is not current")
    if value.get("status") not in {"COMPLETE", "PARTIAL"}:
        raise ValueError("candidate-plan status is invalid")
    upstream_fields = (
        "state_ledger", "decision", "analysis_manifest", "checkpoint", "stop_receipt",
        "capability_receipt", "episode_ledger",
    )
    for name in upstream_fields:
        _file_identity_shape(value.get(name), name=name)
    _validate_upstream_identities(
        value,
        expected_upstream_identities,
        fields=upstream_fields,
    )
    if not isinstance(value.get("visual_coordinate_sha256"), str) or _SHA256.fullmatch(value["visual_coordinate_sha256"]) is None:
        raise ValueError("candidate-plan visual coordinate SHA is invalid")
    states = value.get("states")
    if not isinstance(states, list):
        raise TypeError("candidate-plan states must be a list")
    expected = [_state_key(key) for key in expected_state_keys]
    if [dict(row.get("state_key", {})) for row in states if isinstance(row, Mapping)] != expected:
        raise ValueError("candidate-plan state keys/order drifted")
    total_candidates = 0
    tree_nodes = sum(3**depth for depth in range(1, h_active + 1))
    for state in states:
        if not isinstance(state, Mapping) or set(state) != {
            "state_key", "source_full_action", "stop_category", "shared_counters", "methods"
        }:
            raise ValueError("candidate-plan state fields are incomplete")
        if state.get("source_full_action") not in _MOTION:
            raise ValueError("candidate-plan source action must be motion")
        category = state.get("stop_category")
        if category not in {"not-entered", "entered-motion"}:
            raise ValueError("candidate-plan STOP category is invalid")
        shared = state.get("shared_counters")
        expected_shared = {
            "q_calls": 1,
            "stop_probe_g_calls": 1 if category == "entered-motion" else 0,
            "total_g_calls": h_active + (1 if category == "entered-motion" else 0),
            "total_f_calls": h_active * 2,
        }
        if not isinstance(shared, Mapping) or dict(shared) != expected_shared:
            raise ValueError("candidate-plan shared Q/G/F counters drifted")
        methods = state.get("methods")
        if not isinstance(methods, list) or [row.get("method") for row in methods if isinstance(row, Mapping)] != list(_METHODS):
            raise ValueError("candidate-plan methods must be Full then NoG-enumerate")
        method_mode_sets: list[tuple[int, ...]] = []
        for method_row, method in zip(methods, _METHODS, strict=True):
            if not isinstance(method_row, Mapping) or set(method_row) != {
                "method", "selected_key", "counters", "candidates"
            }:
                raise ValueError("candidate-plan method fields are incomplete")
            counters = method_row.get("counters")
            expected_counters = (
                {
                    "q_calls": 0, "g_calls": h_active, "g_rows": k_active * h_active,
                    "f_calls": h_active, "f_rows": k_active * h_active,
                    "candidate_rows": k_active * h_active, "tree_nodes_per_batch": 0,
                }
                if method == "Full"
                else {
                    "q_calls": 0, "g_calls": 0, "g_rows": 0,
                    "f_calls": h_active, "f_rows": tree_nodes,
                    "candidate_rows": k_active * h_active,
                    "tree_nodes_per_batch": tree_nodes,
                }
            )
            if not isinstance(counters, Mapping) or dict(counters) != expected_counters:
                raise ValueError("candidate-plan method counters drifted")
            candidates = method_row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != k_active * h_active:
                raise ValueError(f"candidate set must contain complete K*H={k_active*h_active}")
            seen: list[dict[str, Any]] = []
            observed_pairs: list[tuple[int, int]] = []
            for index, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping) or set(candidate) != {
                    "candidate_key", "actions", "negative_log_mass", "goal_distance", "consistency"
                }:
                    raise ValueError("candidate fields are incomplete")
                key = _candidate_key(candidate.get("candidate_key"), method=method)
                k = int(key["global_k"])
                h = int(key["one_based_h"])
                observed_pairs.append((k, h))
                actions = candidate.get("actions")
                if not isinstance(actions, list) or len(actions) != h or any(action not in _MOTION for action in actions):
                    raise ValueError("candidate action prefix is incomplete")
                _finite(candidate.get("negative_log_mass"), name="negative log mass")
                _finite(candidate.get("goal_distance"), name="goal distance")
                if method == "Full":
                    _finite(candidate.get("consistency"), name="consistency")
                elif candidate.get("consistency") is not None:
                    raise ValueError("NoG consistency must be null")
                seen.append(key)
            mode_set = tuple(sorted({pair[0] for pair in observed_pairs}))
            if len(mode_set) != k_active:
                raise ValueError("candidate set does not contain exactly K active global modes")
            expected_pairs = [
                (global_k, horizon)
                for global_k in mode_set
                for horizon in range(1, h_active + 1)
            ]
            if observed_pairs != expected_pairs:
                raise ValueError(
                    "candidate order/key differs from the complete sorted active K/H set"
                )
            method_mode_sets.append(mode_set)
            if dict(method_row.get("selected_key", {})) not in seen:
                raise ValueError("selected candidate key is absent")
            total_candidates += len(candidates)
        if method_mode_sets[0] != method_mode_sets[1]:
            raise ValueError("Full and NoG must share the exact Q global-mode set")
    counts = value.get("counts")
    expected_counts = {
        "states": len(states),
        "complete_states": len(states),
        "candidates": total_candidates,
    }
    if not isinstance(counts, Mapping) or dict(counts) != expected_counts:
        raise ValueError("candidate-plan counts drifted")
    return MappingProxyType(copy.deepcopy(dict(value)))


def _ordered_key_digest(keys: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes([_state_key(key) for key in keys])
    ).hexdigest()


def _ranker_identity(value: object, *, k_active: int, h_active: int) -> dict[str, Any]:
    required = {
        "source", "full_deployment", "no_f_deployment", "k_active",
        "h_active", "tie_break",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("ranker identity fields are incomplete")
    _file_identity_shape(value.get("source"), name="ranker source")
    if value.get("full_deployment") != "full" or value.get("no_f_deployment") != "no_f":
        raise ValueError("ranker deployment identities drifted")
    if value.get("k_active") != k_active or value.get("h_active") != h_active:
        raise ValueError("ranker active K/H drifted")
    tie_break = value.get("tie_break")
    expected = {
        "full": [
            "F_endpoint_goal_distance", "negative_log_mass", "global_k",
            "one_based_h",
        ],
        "no_f": [
            "Q_endpoint_goal_distance", "negative_log_mass", "global_k",
            "one_based_h",
        ],
    }
    if not isinstance(tie_break, Mapping) or to_plain_json(tie_break) != expected:
        raise ValueError("ranker tie-break identity drifted")
    return to_plain_json(value)


def _candidate_contract(value: object, *, k_active: int, h_active: int) -> dict[str, Any]:
    required = {
        "motion_actions", "ordinary_stop_forbidden", "retain_all_finite",
        "full_count", "no_f_count", "local_methods", "no_g_tree_nodes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("candidate contract fields are incomplete")
    count = k_active * h_active
    expected = {
        "motion_actions": list(_MOTION),
        "ordinary_stop_forbidden": True,
        "retain_all_finite": True,
        "full_count": count,
        "no_f_count": count,
        "local_methods": list(_METHODS),
        "no_g_tree_nodes": sum(3**depth for depth in range(1, h_active + 1)),
    }
    if to_plain_json(value) != expected:
        raise ValueError("candidate contract drifted from complete current K/H semantics")
    return expected


def validate_state_ledger(
    value: Mapping[str, Any],
    *,
    expected_upstream_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Validate the exact current local-state root and source-state rows."""

    if not isinstance(value, Mapping) or set(value) != _STATE_ROOT_FIELDS:
        raise ValueError("local state ledger fields are incomplete or noncanonical")
    if value.get("schema") != "J2J_CONTEXT4_LOCAL_STATE_LEDGER_V1":
        raise ValueError("local state ledger schema is not current")
    if value.get("status") not in {"COMPLETE", "PARTIAL"}:
        raise ValueError("local state ledger status is invalid")
    if value.get("title") != _TITLE:
        raise ValueError("local state ledger title is not current")
    source = value.get("source")
    if not isinstance(source, Mapping) or to_plain_json(source) != {
        "variant_id": FULL_V1_VARIANT_ID,
        "deployment": "full",
        "training_seed": 3072,
        "epoch": 30,
    }:
        raise ValueError("local source must be exact-final Full seed3072 epoch30")
    budgets = value.get("budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != {
        "episode_budget", "step_budget", "state_budget",
        "candidate_budget_per_method",
    }:
        raise ValueError("local state budgets are incomplete")
    for name in ("episode_budget", "step_budget", "state_budget"):
        _integer(budgets.get(name), name=f"local {name}", minimum=1)
    candidate_budgets = budgets.get("candidate_budget_per_method")
    if not isinstance(candidate_budgets, Mapping) or list(candidate_budgets) != list(_METHODS):
        raise ValueError("local candidate budgets must be ordered Full then NoG-enumerate")
    values = tuple(candidate_budgets.values())
    if any(type(item) is not int or item <= 0 for item in values) or values[0] != values[1]:
        raise ValueError("local per-method candidate budgets must be equal positive integers")
    candidate_count = int(values[0])
    ranker = value.get("ranker_identity")
    if not isinstance(ranker, Mapping):
        raise ValueError("local ranker identity is absent")
    k_active = ranker.get("k_active")
    h_active = ranker.get("h_active")
    if type(k_active) is not int or k_active <= 0 or type(h_active) is not int or h_active <= 0:
        raise ValueError("local ranker K/H is invalid")
    if candidate_count != k_active * h_active:
        raise ValueError("local candidate budget must equal the complete active K*H set")
    _ranker_identity(ranker, k_active=k_active, h_active=h_active)
    _candidate_contract(value.get("candidate_contract"), k_active=k_active, h_active=h_active)
    upstream_fields = (
        "decision", "analysis_manifest", "evaluation_overlay", "sensor_config",
        "preflight_receipt", "checkpoint", "training_sidecar", "stop_receipt",
        "capability_receipt", "episode_ledger",
    )
    for name in upstream_fields:
        _file_identity_shape(value.get(name), name=name)
    _validate_upstream_identities(
        value,
        expected_upstream_identities,
        fields=upstream_fields,
    )
    states = value.get("states")
    if not isinstance(states, list):
        raise TypeError("local state ledger states must be a list")
    selection = value.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "method", "domain", "eligible_count", "eligible_key_digest",
        "selected_count", "selected_key_digest",
    }:
        raise ValueError("local state selection fields are incomplete")
    if selection.get("method") != "global_min_sha256_after_complete_eligible_universe":
        raise ValueError("local state selection method is not outcome-blind global minimum")
    if selection.get("domain") != SELECTION_DOMAIN:
        raise ValueError("local state selection domain drifted")
    eligible_count = _integer(selection.get("eligible_count"), name="eligible_count")
    selected_count = _integer(selection.get("selected_count"), name="selected_count")
    for name in ("eligible_key_digest", "selected_key_digest"):
        if not isinstance(selection.get(name), str) or _SHA256.fullmatch(selection[name]) is None:
            raise ValueError(f"local selection {name} is invalid")
    if selected_count != len(states) or selected_count != min(eligible_count, budgets["state_budget"]):
        raise ValueError("local state selected count disagrees with eligible universe/budget")
    previous = ""
    selected_keys: list[dict[str, Any]] = []
    for state in states:
        if not isinstance(state, Mapping) or set(state) != {
            "state_key", "selection_sha256", "identity", "full_action",
            "full_diagnostics",
        }:
            raise ValueError("local state row fields are incomplete")
        digest = state.get("selection_sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None or digest < previous:
            raise ValueError("local states are not in canonical selection order")
        previous = digest
        key = _state_key(state.get("state_key"))
        if digest != selection_sha256(key):
            raise ValueError("local state selection SHA does not match its state key")
        identity = _validate_state_identity(state.get("identity"))
        if [identity["scene"], identity["episode"], identity["step"]] != [
            key["scene"], key["episode"], key["step"],
        ]:
            raise ValueError("local state key and identity disagree")
        normalized = validate_source_decision(
            action=state.get("full_action"),
            diagnostics=state.get("full_diagnostics"),
            k_active=k_active,
            h_active=h_active,
        )
        if not normalized["eligible"]:
            raise ValueError("executed STOP is not eligible for the local state ledger")
        selected_keys.append(key)
    if selection["selected_key_digest"] != _ordered_key_digest(selected_keys):
        raise ValueError("local selected-key digest drifted")
    frozen = deep_freeze(to_plain_json(value))
    assert isinstance(frozen, Mapping)
    return frozen


def build_state_ledger(
    eligible: Sequence[Mapping[str, Any]],
    *,
    episode_budget: int,
    step_budget: int,
    state_budget: int,
    candidate_budget_per_method: Mapping[str, int],
    artifacts: Mapping[str, Mapping[str, Any]],
    ranker_identity: Mapping[str, Any],
    candidate_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build a state ledger only after the complete eligible universe exists."""

    for name, raw in (
        ("episode_budget", episode_budget),
        ("step_budget", step_budget),
        ("state_budget", state_budget),
    ):
        _integer(raw, name=name, minimum=1)
    if not isinstance(artifacts, Mapping):
        raise TypeError("local state artifact identities must be a mapping")
    required_artifacts = {
        "decision", "analysis_manifest", "evaluation_overlay", "sensor_config",
        "preflight_receipt", "checkpoint", "training_sidecar", "stop_receipt",
        "capability_receipt", "episode_ledger",
    }
    if set(artifacts) != required_artifacts:
        raise ValueError("local state artifact identities are incomplete")
    normalized_eligible: list[dict[str, Any]] = []
    for raw in eligible:
        if not isinstance(raw, Mapping) or set(raw) != {
            "state_key", "identity", "full_action", "full_diagnostics",
        }:
            raise ValueError("eligible local state fields are incomplete")
        row = to_plain_json(raw)
        key = _state_key(row["state_key"])
        normalized_eligible.append(
            {
                **row,
                "state_key": key,
                "selection_sha256": selection_sha256(key),
            }
        )
    ordered_eligible = sorted(
        normalized_eligible, key=lambda row: str(row["selection_sha256"])
    )
    selected = ordered_eligible[:state_budget]
    ledger = {
        "schema": "J2J_CONTEXT4_LOCAL_STATE_LEDGER_V1",
        "status": "COMPLETE",
        "title": _TITLE,
        "source": {
            "variant_id": FULL_V1_VARIANT_ID,
            "deployment": "full",
            "training_seed": 3072,
            "epoch": 30,
        },
        "budgets": {
            "episode_budget": episode_budget,
            "step_budget": step_budget,
            "state_budget": state_budget,
            "candidate_budget_per_method": to_plain_json(candidate_budget_per_method),
        },
        "selection": {
            "method": "global_min_sha256_after_complete_eligible_universe",
            "domain": SELECTION_DOMAIN,
            "eligible_count": len(ordered_eligible),
            "eligible_key_digest": _ordered_key_digest(
                [row["state_key"] for row in ordered_eligible]
            ),
            "selected_count": len(selected),
            "selected_key_digest": _ordered_key_digest(
                [row["state_key"] for row in selected]
            ),
        },
        **{name: to_plain_json(artifacts[name]) for name in sorted(required_artifacts)},
        "ranker_identity": to_plain_json(ranker_identity),
        "candidate_contract": to_plain_json(candidate_contract),
        "states": selected,
    }
    return validate_state_ledger(ledger)


__all__ = [
    "SELECTION_DOMAIN",
    "canonical_rgb_sha256",
    "build_state_ledger",
    "select_state_rows",
    "selection_sha256",
    "validate_candidate_plan_ledger",
    "validate_source_decision",
    "validate_stage2a_full_counter_reconciliation",
    "validate_state_ledger",
]
