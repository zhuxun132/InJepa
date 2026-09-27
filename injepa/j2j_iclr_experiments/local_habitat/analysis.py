"""Pure supporting analysis for immutable Context4 local branch ledgers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from j2j_iclr_experiments.common.artifacts import (
    deep_freeze,
    load_json_file_identity,
    to_plain_json,
    validate_file_identity,
)


TITLE = "conditional on Full/seed3072/epoch30 visited-state distribution"
_METHODS = ("Full", "NoG-enumerate")
_ERRORS = ("mse", "mae", "cosine_distance")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _summary(values: Sequence[float], *, unavailable: bool = False) -> dict[str, object]:
    if unavailable:
        return {"status": "UNAVAILABLE", "n": 0, "value": None}
    if not values:
        return {"status": "DEGENERATE", "n": 0, "value": None}
    return {
        "status": "OK",
        "n": len(values),
        "value": float(np.mean(np.asarray(values, dtype=np.float64))),
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    order = np.argsort(data, kind="stable")
    ranks = np.empty(data.size, dtype=np.float64)
    start = 0
    while start < data.size:
        stop = start + 1
        while stop < data.size and data[order[stop]] == data[order[start]]:
            stop += 1
        # Ranks are one-based; ties receive the arithmetic mean rank.
        ranks[order[start:stop]] = ((start + 1) + stop) / 2.0
        start = stop
    return ranks


def _correlation_summary(
    consistency: Sequence[float], errors: Sequence[float]
) -> dict[str, dict[str, object]]:
    if not consistency:
        unavailable = {"status": "UNAVAILABLE", "n": 0, "value": None}
        return {"spearman": dict(unavailable), "kendall_tau_b": dict(unavailable)}
    count = len(consistency)
    if count != len(errors):
        raise ValueError("correlation axes must align")
    x = np.asarray(consistency, dtype=np.float64)
    y = np.asarray(errors, dtype=np.float64)
    if count < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        degenerate = {"status": "DEGENERATE", "n": count, "value": None}
        return {"spearman": dict(degenerate), "kendall_tau_b": dict(degenerate)}

    xr = _average_ranks(x)
    yr = _average_ranks(y)
    spearman = float(np.corrcoef(xr, yr)[0, 1])

    concordant = discordant = ties_x = ties_y = 0
    for left in range(count):
        for right in range(left + 1, count):
            dx = float(x[left] - x[right])
            dy = float(y[left] - y[right])
            if dx == 0.0 and dy == 0.0:
                # Joint ties contribute to neither denominator term.
                continue
            if dx == 0.0:
                ties_x += 1
            elif dy == 0.0:
                ties_y += 1
            elif dx * dy > 0.0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_x)
        * (concordant + discordant + ties_y)
    )
    if denominator == 0.0:
        tau = {"status": "DEGENERATE", "n": count, "value": None}
    else:
        tau = {
            "status": "OK",
            "n": count,
            "value": (concordant - discordant) / denominator,
        }
    return {
        "spearman": {"status": "OK", "n": count, "value": spearman},
        "kendall_tau_b": tau,
    }


def _stable_key(row: Mapping[str, Any]) -> tuple[float, str, str, str]:
    return (
        float(row["consistency"]),
        json.dumps(row["state_key"], sort_keys=True, separators=(",", ":")),
        str(row["method"]),
        json.dumps(row["candidate_key"], sort_keys=True, separators=(",", ":")),
    )


def _calibration(
    rows: Sequence[Mapping[str, Any]], *, error_key: str, bins: int
) -> dict[str, object]:
    available = [row for row in rows if row.get("consistency") is not None]
    if not available:
        return {"status": "UNAVAILABLE", "bins": []}
    ordered = sorted(available, key=_stable_key)
    bin_count = min(bins, len(ordered))
    quotient, remainder = divmod(len(ordered), bin_count)
    output: list[dict[str, object]] = []
    cursor = 0
    for index in range(bin_count):
        size = quotient + (1 if index < remainder else 0)
        group = ordered[cursor : cursor + size]
        cursor += size
        output.append(
            {
                "index": index,
                "count": len(group),
                "mean_consistency": float(
                    np.mean([float(row["consistency"]) for row in group])
                ),
                "mean_endpoint_error": float(
                    np.mean([float(row[error_key]) for row in group])
                ),
                "mean_actual_progress": float(
                    np.mean([float(row["actual_progress"]) for row in group])
                ),
            }
        )
    return {"status": "OK", "bins": output}


def _validate_manifest(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("analysis_manifest must be a mapping")
    required = {"schema", "status", "protocol_sha256", "files", "operations", "runtime"}
    if set(value) != required:
        raise ValueError("analysis manifest fields are incomplete or noncanonical")
    if value.get("schema") != "J2J_CONTEXT4_ANALYSIS_SOURCE_MANIFEST_V1":
        raise ValueError("analysis manifest schema is not current")
    if value.get("status") != "FROZEN_BEFORE_THIS_RUN_OUTCOMES":
        raise ValueError("analysis manifest was not frozen before this run outcomes")
    return value


def _file_identity(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} FileIdentity is incomplete")
    path = value.get("path")
    byte_count = value.get("bytes")
    digest = value.get("sha256")
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError(f"{name} FileIdentity path must be absolute")
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError(f"{name} FileIdentity byte count must be positive")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{name} FileIdentity SHA-256 is invalid")
    return {"path": path, "bytes": byte_count, "sha256": digest}


def _validate_scalar_summary(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"status", "n", "value"}:
        raise ValueError(f"local analysis {name} scalar summary is incomplete")
    status = value.get("status")
    count = value.get("n")
    if status not in {"OK", "DEGENERATE", "UNAVAILABLE"}:
        raise ValueError(f"local analysis {name} status is invalid")
    if type(count) is not int or count < 0:
        raise ValueError(f"local analysis {name} count is invalid")
    if status == "OK":
        _finite(value.get("value"), name=f"local analysis {name} value")
        if count <= 0:
            raise ValueError(f"local analysis {name} OK count must be positive")
    elif value.get("value") is not None:
        raise ValueError(f"local analysis {name} non-OK value must be null")


def _validate_error_summaries(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(_ERRORS):
        raise ValueError(f"local analysis {name} error summaries are incomplete")
    for metric in _ERRORS:
        _validate_scalar_summary(value[metric], name=f"{name}.{metric}")


def _validate_correlation(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "spearman",
        "kendall_tau_b",
    }:
        raise ValueError(f"local analysis {name} correlation is incomplete")
    for metric in ("spearman", "kendall_tau_b"):
        _validate_scalar_summary(value[metric], name=f"{name}.{metric}")


def _validate_calibration(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"status", "bins"}:
        raise ValueError(f"local analysis {name} calibration is incomplete")
    status = value.get("status")
    bins = value.get("bins")
    if status not in {"OK", "DEGENERATE", "UNAVAILABLE"} or not isinstance(
        bins, list
    ):
        raise ValueError(f"local analysis {name} calibration status is invalid")
    if status != "OK" and bins:
        raise ValueError(f"local analysis {name} unavailable bins must be empty")
    for index, row in enumerate(bins):
        if not isinstance(row, Mapping) or set(row) != {
            "index",
            "count",
            "mean_consistency",
            "mean_endpoint_error",
            "mean_actual_progress",
        }:
            raise ValueError(f"local analysis {name} bin is incomplete")
        if row.get("index") != index or type(row.get("count")) is not int or row[
            "count"
        ] <= 0:
            raise ValueError(f"local analysis {name} bin index/count is invalid")
        for field in (
            "mean_consistency",
            "mean_endpoint_error",
            "mean_actual_progress",
        ):
            _finite(row.get(field), name=f"local analysis {name}.{field}")


def validate_local_analysis(
    value: Mapping[str, Any],
    *,
    branch_ledger: Mapping[str, Any],
    branch_results_identity: Mapping[str, Any],
    analysis_manifest_identity: Mapping[str, Any],
    calibration_bins: int | None = None,
) -> Mapping[str, Any]:
    """Validate the exact supporting-only analysis before and after publish."""

    from .branch import validate_branch_result_ledger

    if calibration_bins is not None and (
        type(calibration_bins) is not int or calibration_bins <= 0
    ):
        raise ValueError("local analysis calibration_bins must be positive")
    expected_branch = to_plain_json(
        validate_file_identity(branch_results_identity, name="branch results")
    )
    expected_manifest = to_plain_json(
        validate_file_identity(analysis_manifest_identity, name="analysis manifest")
    )
    live_branch, observed_branch = load_json_file_identity(
        expected_branch, name="branch results"
    )
    live_manifest, observed_manifest = load_json_file_identity(
        expected_manifest, name="analysis manifest"
    )
    if to_plain_json(observed_branch) != expected_branch:
        raise ValueError("local analysis branch live identity drifted")
    if to_plain_json(observed_manifest) != expected_manifest:
        raise ValueError("local analysis manifest live identity drifted")
    if to_plain_json(branch_ledger) != live_branch:
        raise ValueError("local analysis branch ledger differs from live bytes")
    _validate_manifest(live_manifest)
    branch_ledger = validate_branch_result_ledger(live_branch)
    required = {
        "schema",
        "status",
        "title",
        "claim_scope",
        "branch_results",
        "analysis_manifest",
        "counts",
        "methods",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("local analysis root fields are incomplete")
    if value.get("schema") != "J2J_CONTEXT4_LOCAL_ANALYSIS_V1":
        raise ValueError("local analysis schema is not current")
    if value.get("title") != TITLE or value.get("claim_scope") != (
        "NOT_NAVIGATION_PERFORMANCE"
    ):
        raise ValueError("local analysis scope/title drifted")
    if _file_identity(value.get("branch_results"), name="branch results") != (
        expected_branch
    ):
        raise ValueError("local analysis branch-results identity drifted")
    if _file_identity(value.get("analysis_manifest"), name="analysis manifest") != (
        expected_manifest
    ):
        raise ValueError("local analysis manifest identity drifted")
    if not isinstance(branch_ledger, Mapping) or branch_ledger.get("schema") != (
        "J2J_CONTEXT4_LOCAL_BRANCH_RESULTS_V1"
    ):
        raise ValueError("local analysis branch-result ledger is not current")
    branch_counts = branch_ledger.get("counts")
    if not isinstance(branch_counts, Mapping):
        raise ValueError("local analysis branch-result counts are absent")
    expected_counts = {
        "states": branch_counts.get("states"),
        "valid_states": branch_counts.get("valid_states"),
        "invalid_states": branch_counts.get("invalid_states"),
        "candidate_rows": branch_counts.get("valid_candidates"),
    }
    if not isinstance(value.get("counts"), Mapping) or dict(value["counts"]) != (
        expected_counts
    ):
        raise ValueError("local analysis counts drifted from branch results")
    expected_status = (
        "SUPPORTING_DESCRIPTIVE"
        if branch_ledger.get("status") == "COMPLETE"
        else "PARTIAL"
    )
    if value.get("status") != expected_status:
        raise ValueError("local analysis status drifted from branch results")
    methods = value.get("methods")
    if not isinstance(methods, list) or [row.get("method") for row in methods] != list(
        _METHODS
    ):
        raise ValueError("local analysis methods are incomplete or out of order")
    for row in methods:
        if not isinstance(row, Mapping) or set(row) != {
            "method",
            "status",
            "q_error",
            "f_error",
            "progress",
            "top1_regret",
            "correlations",
            "calibration",
        }:
            raise ValueError("local analysis method fields are incomplete")
        if row.get("status") not in {"COMPLETE", "DEGENERATE", "PARTIAL"}:
            raise ValueError("local analysis method status is invalid")
        _validate_error_summaries(row.get("q_error"), name="Q")
        _validate_error_summaries(row.get("f_error"), name="F")
        _validate_scalar_summary(row.get("progress"), name="progress")
        _validate_scalar_summary(row.get("top1_regret"), name="top1 regret")
        correlations = row.get("correlations")
        if not isinstance(correlations, Mapping) or set(correlations) != {
            "consistency_vs_q_endpoint_mae",
            "consistency_vs_f_endpoint_mae",
        }:
            raise ValueError("local analysis correlations are incomplete")
        for name, correlation in correlations.items():
            _validate_correlation(correlation, name=name)
        calibration = row.get("calibration")
        if not isinstance(calibration, Mapping) or set(calibration) != {
            "q_endpoint_mae",
            "f_endpoint_mae",
        }:
            raise ValueError("local analysis calibration is incomplete")
        for name, bins in calibration.items():
            _validate_calibration(bins, name=name)
        if row["method"] == "NoG-enumerate":
            for correlation in correlations.values():
                if any(summary.get("status") != "UNAVAILABLE" for summary in correlation.values()):
                    raise ValueError("NoG local analysis correlation must be unavailable")
            if any(item.get("status") != "UNAVAILABLE" for item in calibration.values()):
                raise ValueError("NoG local analysis calibration must be unavailable")
    if calibration_bins is None:
        full = methods[0]
        full_calibration = full.get("calibration")
        q_calibration = (
            full_calibration.get("q_endpoint_mae")
            if isinstance(full_calibration, Mapping)
            else None
        )
        observed_bins = (
            q_calibration.get("bins") if isinstance(q_calibration, Mapping) else None
        )
        calibration_bins = (
            len(observed_bins)
            if isinstance(observed_bins, list) and observed_bins
            else 10
        )
    rederived = analyze_branch_results(
        to_plain_json(branch_ledger),
        analysis_manifest=live_manifest,
        branch_results_identity=expected_branch,
        analysis_manifest_identity=expected_manifest,
        calibration_bins=calibration_bins,
    )
    if to_plain_json(value) != rederived:
        raise ValueError("local analysis values differ from branch-derived analysis")
    frozen = deep_freeze(to_plain_json(value))
    assert isinstance(frozen, Mapping)
    return frozen


def analyze_branch_results(
    branch_ledger: Mapping[str, Any],
    *,
    analysis_manifest: Mapping[str, Any],
    branch_results_identity: Mapping[str, Any],
    analysis_manifest_identity: Mapping[str, Any],
    calibration_bins: int = 10,
) -> dict[str, Any]:
    """Describe Q/F endpoint error versus real branch progress without feedback."""

    ledger = copy.deepcopy(branch_ledger)
    _validate_manifest(analysis_manifest)
    branch_identity = _file_identity(
        branch_results_identity, name="branch results"
    )
    manifest_identity = _file_identity(
        analysis_manifest_identity, name="analysis manifest"
    )
    candidate_identity = ledger.get("candidate_plan")
    if isinstance(candidate_identity, Mapping) and branch_identity == dict(
        candidate_identity
    ):
        raise ValueError(
            "branch results FileIdentity cannot alias its candidate-plan input"
        )
    ledger_manifest_identity = ledger.get("analysis_manifest")
    if not isinstance(ledger_manifest_identity, Mapping) or manifest_identity != dict(
        ledger_manifest_identity
    ):
        raise ValueError("analysis manifest FileIdentity drifted from branch results")
    if type(calibration_bins) is not int or calibration_bins <= 0:
        raise ValueError("calibration_bins must be a positive integer")
    if not isinstance(ledger, Mapping) or ledger.get("schema") != (
        "J2J_CONTEXT4_LOCAL_BRANCH_RESULTS_V1"
    ):
        raise ValueError("branch-result ledger schema is not current")
    states = ledger.get("states")
    if not isinstance(states, list):
        raise TypeError("branch-result states must be a list")

    rows_by_method: dict[str, list[dict[str, Any]]] = {name: [] for name in _METHODS}
    regrets: dict[str, list[float]] = {name: [] for name in _METHODS}
    valid_states = invalid_states = 0
    for state in states:
        if not isinstance(state, Mapping):
            raise TypeError("branch state must be a mapping")
        if state.get("status") != "VALID":
            invalid_states += 1
            continue
        valid_states += 1
        methods = state.get("methods")
        if not isinstance(methods, list) or [row.get("method") for row in methods] != list(_METHODS):
            raise ValueError("branch methods must be Full then NoG-enumerate")
        for method in methods:
            name = str(method["method"])
            if method.get("complete_candidate_set") is True:
                regrets[name].append(_finite(method.get("top1_regret"), name="top1 regret"))
            results = method.get("results")
            if not isinstance(results, list):
                raise TypeError("branch results must be a list")
            for result in results:
                if not isinstance(result, Mapping) or result.get("status") != "VALID":
                    continue
                q_error = result.get("q_error")
                f_error = result.get("f_error")
                if not isinstance(q_error, Mapping) or not isinstance(f_error, Mapping):
                    raise ValueError("valid branch row requires Q/F errors")
                normalized = {
                    "state_key": copy.deepcopy(state.get("state_key")),
                    "method": name,
                    "candidate_key": copy.deepcopy(result.get("candidate_key")),
                    "actual_progress": _finite(
                        result.get("actual_progress"), name="actual progress"
                    ),
                    "consistency": (
                        None
                        if result.get("consistency") is None
                        else _finite(result.get("consistency"), name="consistency")
                    ),
                }
                for owner, values in (("q", q_error), ("f", f_error)):
                    for metric in _ERRORS:
                        normalized[f"{owner}_{metric}"] = _finite(
                            values.get(metric), name=f"{owner} {metric}"
                        )
                rows_by_method[name].append(normalized)

    methods_output: list[dict[str, Any]] = []
    for name in _METHODS:
        rows = rows_by_method[name]
        q = {
            metric: _summary([float(row[f"q_{metric}"]) for row in rows])
            for metric in _ERRORS
        }
        f = {
            metric: _summary([float(row[f"f_{metric}"]) for row in rows])
            for metric in _ERRORS
        }
        consistency = [
            float(row["consistency"])
            for row in rows
            if row.get("consistency") is not None
        ]
        q_mae = [float(row["q_mae"]) for row in rows if row.get("consistency") is not None]
        f_mae = [float(row["f_mae"]) for row in rows if row.get("consistency") is not None]
        methods_output.append(
            {
                "method": name,
                "status": (
                    "PARTIAL"
                    if ledger.get("status") != "COMPLETE"
                    else ("COMPLETE" if rows else "DEGENERATE")
                ),
                "q_error": q,
                "f_error": f,
                "progress": _summary([float(row["actual_progress"]) for row in rows]),
                "top1_regret": _summary(regrets[name]),
                "correlations": {
                    "consistency_vs_q_endpoint_mae": _correlation_summary(
                        consistency, q_mae
                    ),
                    "consistency_vs_f_endpoint_mae": _correlation_summary(
                        consistency, f_mae
                    ),
                },
                "calibration": {
                    "q_endpoint_mae": _calibration(
                        rows, error_key="q_mae", bins=calibration_bins
                    ),
                    "f_endpoint_mae": _calibration(
                        rows, error_key="f_mae", bins=calibration_bins
                    ),
                },
            }
        )

    return {
        "schema": "J2J_CONTEXT4_LOCAL_ANALYSIS_V1",
        "status": (
            "SUPPORTING_DESCRIPTIVE"
            if ledger.get("status") == "COMPLETE"
            else "PARTIAL"
        ),
        "title": TITLE,
        "claim_scope": "NOT_NAVIGATION_PERFORMANCE",
        "branch_results": copy.deepcopy(branch_identity),
        "analysis_manifest": copy.deepcopy(manifest_identity),
        "counts": {
            "states": len(states),
            "valid_states": valid_states,
            "invalid_states": invalid_states,
            "candidate_rows": sum(len(rows) for rows in rows_by_method.values()),
        },
        "methods": methods_output,
    }


__all__ = ["TITLE", "analyze_branch_results", "validate_local_analysis"]
