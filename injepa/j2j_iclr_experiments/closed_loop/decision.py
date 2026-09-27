"""Frozen pre-outcome decision and analysis identities for current Context4."""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

from collections.abc import Mapping, Sequence
import json
from pathlib import Path, PurePosixPath
import platform
from typing import Any

import numpy as np

from j2j.context4_variants import resolve_variant_contract
from j2j_iclr_experiments.common.artifacts import (
    canonical_mapping_sha256,
    create_once_json,
    deep_freeze,
    file_identity,
    load_json_file_identity,
    require_sha256,
    to_plain_json,
    validate_file_identity,
)
from j2j_iclr_experiments.common.identity import (
    FULL_V1_VARIANT_ID,
    derive_variant_identity,
)


ANALYSIS_SCHEMA = "J2J_CONTEXT4_ANALYSIS_SOURCE_MANIFEST_V1"
DECISION_SCHEMA = "J2J_CONTEXT4_FIVE_ROW_RUN_DECISION_V1"
DECISION_INTENT_SCHEMA = "J2J_CONTEXT4_FIVE_ROW_DECISION_INTENT_V1"
FINAL_RECEIPT_SCHEMA = "J2J_CONTEXT4_CLOSED_LOOP_EVALUATION_V1"
FROZEN_STATUS = "FROZEN_BEFORE_THIS_RUN_OUTCOMES"
ROW_IDS = ("Full", "Context1", "MeanRepeat", "NoQG", "NoF-deploy")
DEPLOYMENTS = ("full", "full", "full", "full", "no_f")
_ROW_VARIANT_IDS = {
    "Full": FULL_V1_VARIANT_ID,
    "Context1": "context1",
    "MeanRepeat": "mean_repeat",
    "NoQG": "no_qg",
    "NoF-deploy": FULL_V1_VARIANT_ID,
}
SEEDS = (0, 42, 3072)
ANALYSIS_FILES = (
    "j2j_iclr_experiments/common/statistics.py",
    "j2j_iclr_experiments/local_habitat/analysis.py",
    "j2j_iclr_experiments/offline/metrics.py",
    "scripts/evaluate_imagegoal.py",
    "j2j/evaluation/episode_sharding.py",
    "j2j/evaluation/habitat_runner.py",
    "j2j/evaluation/video.py",
)
ANALYSIS_OPERATIONS = {
    "row_unit": ["state_key", "method", "candidate_key"],
    "quantile_method": "numpy_linear",
    "spearman_rank_ties": "average",
    "kendall_variant": "tau_b",
    "calibration_order": [
        "consistency",
        "state_key",
        "method",
        "candidate_key",
    ],
    "degenerate_status": "DEGENERATE",
    "closed_loop_estimand": "episode_then_seed_equal_weight",
}
CLAIM_SCOPE = {
    "freeze_scope": "before_this_run_outcomes",
    "navigation_metrics": ["SR", "SPL"],
    "missing_run_rows": "DELETE_ASSOCIATED_CLAIM_AND_KEEP_FIXED_FAMILY_MISSING",
}
_FULL_TIE_BREAK = (
    "F_endpoint_goal_distance",
    "negative_log_mass",
    "global_k",
    "one_based_h",
)
_NO_F_TIE_BREAK = (
    "Q_endpoint_goal_distance",
    "negative_log_mass",
    "global_k",
    "one_based_h",
)


def resolve_variant_decision_row(
    variant_identity: Mapping[str, object],
    *,
    deployment: str,
    claimed_row: str | None = None,
) -> str:
    """Resolve one and only one frozen closed-loop row from native identity."""

    contract = resolve_variant_contract(variant_identity)
    variant_id = contract.identity.variant_id
    if deployment in {"no_f", "no_f_deploy"}:
        if variant_id != FULL_V1_VARIANT_ID:
            raise ValueError("no_f deployment is admitted only for the Full variant")
        expected = "NoF-deploy"
    elif deployment == "full":
        expected = next(
            row
            for row in ROW_IDS[:-1]
            if _ROW_VARIANT_IDS[row] == variant_id
        )
    else:
        raise ValueError("deployment is outside the frozen full/no_f identities")
    if claimed_row is not None and claimed_row != expected:
        raise ValueError(
            f"claimed row {claimed_row!r} does not match variant/deployment identity"
        )
    return expected


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} field names must be strings")
    return value


def _exact_fields(
    value: object, expected: set[str], *, name: str
) -> Mapping[str, Any]:
    raw = _mapping(value, name=name)
    if set(raw) != expected:
        missing = sorted(expected - set(raw))
        extra = sorted(set(raw) - expected)
        raise ValueError(
            f"{name} fields are not exact (missing={missing}, extra={extra})"
        )
    return raw


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an ordered list")
    return value


def _nonempty(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical string")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative integer, not bool")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    observed = _nonnegative_integer(value, name=name)
    if observed <= 0:
        raise ValueError(f"{name} must be positive")
    return observed


def _same(left: object, right: object, *, name: str) -> None:
    if to_plain_json(left) != to_plain_json(right):
        raise RuntimeError(f"{name} identity drift")


def _validated_analysis_file(
    value: object, *, project_root: Path, expected_relative_path: str
) -> dict[str, Any]:
    row = _exact_fields(
        value, {"relative_path", "bytes", "sha256"}, name="analysis source"
    )
    relative = _nonempty(row["relative_path"], name="analysis source relative path")
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or relative != posix.as_posix()
        or any(part in {".", ".."} for part in posix.parts)
        or relative != expected_relative_path
    ):
        raise ValueError("analysis source path/order is not the exact allowlist")
    source = (project_root / Path(*posix.parts)).resolve()
    try:
        source.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("analysis source path escapes project root") from exc
    observed = file_identity(source, name=f"analysis source {relative}")
    expected_bytes = _positive_integer(row["bytes"], name="analysis source bytes")
    expected_sha = require_sha256(row["sha256"], name="analysis source SHA")
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha:
        raise ValueError(f"analysis source live bytes/SHA drift: {relative}")
    return {
        "relative_path": relative,
        "bytes": expected_bytes,
        "sha256": expected_sha,
    }


def validate_analysis_source_manifest(
    manifest: object,
    *,
    expected_protocol_sha256: str,
    project_root: str | Path,
) -> Mapping[str, object]:
    """Validate exact live source bytes and return a recursively frozen copy."""

    expected_protocol = require_sha256(
        expected_protocol_sha256, name="expected protocol SHA"
    )
    root = _exact_fields(
        manifest,
        {"schema", "status", "protocol_sha256", "files", "operations", "runtime"},
        name="analysis manifest",
    )
    if root["schema"] != ANALYSIS_SCHEMA:
        raise ValueError("analysis manifest schema is not current Context4")
    if root["status"] != FROZEN_STATUS:
        raise ValueError("analysis manifest status was not frozen before outcomes")
    if root["protocol_sha256"] != expected_protocol:
        raise ValueError("analysis manifest protocol SHA identity drift")
    resolved_root = Path(project_root).expanduser().resolve()
    files = _sequence(root["files"], name="analysis manifest files")
    if len(files) != len(ANALYSIS_FILES):
        raise ValueError("analysis source field set is incomplete")
    validated_files = [
        _validated_analysis_file(
            row,
            project_root=resolved_root,
            expected_relative_path=expected_relative,
        )
        for row, expected_relative in zip(files, ANALYSIS_FILES, strict=True)
    ]
    operations = _exact_fields(
        root["operations"], set(ANALYSIS_OPERATIONS), name="analysis operations"
    )
    if to_plain_json(operations) != ANALYSIS_OPERATIONS:
        raise ValueError("analysis operation identity is not the frozen contract")
    runtime = _exact_fields(
        root["runtime"], {"python_version", "numpy_version"}, name="analysis runtime"
    )
    expected_runtime = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if to_plain_json(runtime) != expected_runtime:
        raise ValueError("analysis runtime identity drift")
    frozen = deep_freeze(
        {
            "schema": ANALYSIS_SCHEMA,
            "status": FROZEN_STATUS,
            "protocol_sha256": expected_protocol,
            "files": validated_files,
            "operations": ANALYSIS_OPERATIONS,
            "runtime": expected_runtime,
        }
    )
    assert isinstance(frozen, Mapping)
    return frozen


def _validate_checkpoint_entry(value: object) -> dict[str, Any]:
    row = _exact_fields(
        value,
        {
            "checkpoint_id",
            "variant_id",
            "training_seed",
            "checkpoint",
            "training_sidecar",
            "training_code_sha256",
            "epoch",
            "exact_final",
        },
        name="checkpoint entry",
    )
    checkpoint_id = _nonempty(row["checkpoint_id"], name="checkpoint ID")
    variant_id = _nonempty(row["variant_id"], name="checkpoint variant ID")
    seed = _nonnegative_integer(row["training_seed"], name="checkpoint training seed")
    if seed not in SEEDS:
        raise ValueError("checkpoint training seed is outside the frozen seed set")
    checkpoint = validate_file_identity(row["checkpoint"], name="checkpoint")
    sidecar_payload, sidecar = load_json_file_identity(
        row["training_sidecar"], name="training sidecar"
    )
    native_variant = derive_variant_identity(sidecar_payload)
    resolve_variant_contract(native_variant)
    native_model = _mapping(sidecar_payload.get("model"), name="training sidecar model")
    native_seed = _nonnegative_integer(
        native_model.get("global_seed"), name="native checkpoint training seed"
    )
    if variant_id != native_variant.get("variant_id"):
        raise ValueError("checkpoint variant ID differs from its native sidecar")
    if seed != native_seed:
        raise ValueError("checkpoint training seed differs from its native sidecar")
    code_sha = require_sha256(
        row["training_code_sha256"], name="training code SHA"
    )
    if row["epoch"] != 30 or type(row["epoch"]) is not int:
        raise ValueError("checkpoint epoch must be exact-final epoch 30")
    if row["exact_final"] is not True:
        raise ValueError("checkpoint exact_final must be true")
    return {
        "checkpoint_id": checkpoint_id,
        "variant_id": variant_id,
        "training_seed": seed,
        "checkpoint": to_plain_json(checkpoint),
        "training_sidecar": to_plain_json(sidecar),
        "training_code_sha256": code_sha,
        "epoch": 30,
        "exact_final": True,
    }


def _validate_ranker_identity(value: object) -> dict[str, Any]:
    row = _exact_fields(
        value,
        {
            "source",
            "full_deployment",
            "no_f_deployment",
            "k_active",
            "h_active",
            "tie_break",
        },
        name="ranker identity",
    )
    if row["full_deployment"] != "full" or row["no_f_deployment"] != "no_f":
        raise ValueError("ranker deployment identities are invalid")
    k_active = _positive_integer(row["k_active"], name="ranker K")
    h_active = _positive_integer(row["h_active"], name="ranker H")
    tie_break = _exact_fields(
        row["tie_break"], {"full", "no_f"}, name="ranker tie-break"
    )
    if tuple(_sequence(tie_break["full"], name="Full tie-break")) != _FULL_TIE_BREAK:
        raise ValueError("Full ranker tie-break identity drift")
    if tuple(_sequence(tie_break["no_f"], name="NoF tie-break")) != _NO_F_TIE_BREAK:
        raise ValueError("NoF ranker tie-break identity drift")
    return {
        "source": to_plain_json(validate_file_identity(row["source"], name="ranker source")),
        "full_deployment": "full",
        "no_f_deployment": "no_f",
        "k_active": k_active,
        "h_active": h_active,
        "tie_break": {"full": list(_FULL_TIE_BREAK), "no_f": list(_NO_F_TIE_BREAK)},
    }


def _validate_candidate_contract(
    value: object, *, k_active: int, h_active: int
) -> dict[str, Any]:
    row = _exact_fields(
        value,
        {
            "motion_actions",
            "ordinary_stop_forbidden",
            "retain_all_finite",
            "full_count",
            "no_f_count",
            "local_methods",
            "no_g_tree_nodes",
        },
        name="candidate contract",
    )
    if tuple(_sequence(row["motion_actions"], name="candidate motion actions")) != (
        "FWD",
        "LEFT",
        "RIGHT",
    ):
        raise ValueError("candidate motion action identity drift")
    if row["ordinary_stop_forbidden"] is not True or row["retain_all_finite"] is not True:
        raise ValueError("candidate survival/STOP contract is invalid")
    expected_count = k_active * h_active
    full_count = _nonnegative_integer(
        row["full_count"], name="Full candidate count"
    )
    no_f_count = _nonnegative_integer(
        row["no_f_count"], name="NoF candidate count"
    )
    if full_count != expected_count or no_f_count != expected_count:
        raise ValueError("candidate K/H count identity drift")
    if tuple(_sequence(row["local_methods"], name="local methods")) != (
        "Full",
        "NoG-enumerate",
    ):
        raise ValueError("candidate local method identity drift")
    expected_tree = sum(3**h for h in range(1, h_active + 1))
    tree_nodes = _nonnegative_integer(
        row["no_g_tree_nodes"], name="NoG H-tree-node count"
    )
    if tree_nodes != expected_tree:
        raise ValueError("NoG H-tree node count identity drift")
    return {
        "motion_actions": ["FWD", "LEFT", "RIGHT"],
        "ordinary_stop_forbidden": True,
        "retain_all_finite": True,
        "full_count": expected_count,
        "no_f_count": expected_count,
        "local_methods": ["Full", "NoG-enumerate"],
        "no_g_tree_nodes": expected_tree,
    }


def _validate_seed_run(
    value: object, *, expected_seed: int, checkpoints: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    row = _exact_fields(
        value,
        {
            "training_seed",
            "disposition",
            "checkpoint_id",
            "outcome_independent_reason",
        },
        name="decision seed run",
    )
    if row["training_seed"] != expected_seed or type(row["training_seed"]) is not int:
        raise ValueError("decision seed order/identity is invalid")
    disposition = row["disposition"]
    if disposition not in {"RUN", "NOT_RUN"}:
        raise ValueError("decision seed disposition must be RUN or NOT_RUN")
    checkpoint_id = row["checkpoint_id"]
    reason = row["outcome_independent_reason"]
    if disposition == "RUN":
        checkpoint_id = _nonempty(checkpoint_id, name="RUN checkpoint ID")
        if reason is not None:
            raise ValueError("RUN outcome-independent reason must be null")
        checkpoint = checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise ValueError("RUN checkpoint reference is absent from checkpoint_set")
        if checkpoint["training_seed"] != expected_seed:
            raise ValueError("cross-seed checkpoint reuse is forbidden")
    else:
        if checkpoint_id is not None:
            raise ValueError("NOT_RUN checkpoint ID must be null")
        _nonempty(reason, name="NOT_RUN outcome-independent reason")
    return {
        "training_seed": expected_seed,
        "disposition": disposition,
        "checkpoint_id": checkpoint_id,
        "outcome_independent_reason": reason,
    }


def validate_five_row_run_decision(
    decision: object,
    *,
    expected_protocol_sha256: str,
    expected_deployment: str,
    live_identity: Mapping[str, Any],
) -> Mapping[str, object]:
    """Validate all 15 frozen slots and the one exact live launch tuple."""

    expected_protocol = require_sha256(
        expected_protocol_sha256, name="expected protocol SHA"
    )
    root = _exact_fields(
        decision,
        {
            "schema",
            "status",
            "protocol_sha256",
            "rows",
            "seeds",
            "checkpoint_set",
            "episode_ledger",
            "stop_receipt",
            "preflight_receipt",
            "habitat_scientific_identity",
            "ranker_identity",
            "candidate_contract",
            "analysis_manifest",
            "claim_scope",
        },
        name="five-row decision",
    )
    if root["schema"] != DECISION_SCHEMA:
        raise ValueError("decision schema is not current Context4")
    if root["status"] != FROZEN_STATUS:
        raise ValueError("decision status was not frozen before outcomes")
    if root["protocol_sha256"] != expected_protocol:
        raise ValueError("decision protocol SHA identity drift")
    if tuple(_sequence(root["seeds"], name="decision seeds")) != SEEDS:
        raise ValueError("decision seed order is not [0,42,3072]")

    checkpoint_rows = _sequence(root["checkpoint_set"], name="checkpoint set")
    checkpoints_list = [_validate_checkpoint_entry(row) for row in checkpoint_rows]
    checkpoint_ids = [row["checkpoint_id"] for row in checkpoints_list]
    if checkpoint_ids != sorted(checkpoint_ids) or len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("checkpoint_set must be uniquely sorted by checkpoint ID")
    checkpoints = {row["checkpoint_id"]: row for row in checkpoints_list}

    rows_raw = _sequence(root["rows"], name="decision rows")
    if len(rows_raw) != len(ROW_IDS):
        raise ValueError("decision must contain exactly five rows")
    rows: list[dict[str, Any]] = []
    references: list[tuple[str, int, str]] = []
    for raw, row_id, deployment in zip(rows_raw, ROW_IDS, DEPLOYMENTS, strict=True):
        row = _exact_fields(
            raw, {"row_id", "deployment", "seed_runs"}, name="decision row"
        )
        if row["row_id"] != row_id:
            raise ValueError("decision row order/ID is invalid")
        if row["deployment"] != deployment:
            raise ValueError("decision row deployment relabel is forbidden")
        seed_rows = _sequence(row["seed_runs"], name=f"{row_id} seed runs")
        if len(seed_rows) != len(SEEDS):
            raise ValueError("every decision row must contain all three seed slots")
        validated_seed_rows = [
            _validate_seed_run(seed_row, expected_seed=seed, checkpoints=checkpoints)
            for seed_row, seed in zip(seed_rows, SEEDS, strict=True)
        ]
        for seed_row in validated_seed_rows:
            if seed_row["disposition"] == "RUN":
                references.append(
                    (row_id, seed_row["training_seed"], str(seed_row["checkpoint_id"]))
                )
        rows.append(
            {
                "row_id": row_id,
                "deployment": deployment,
                "seed_runs": validated_seed_rows,
            }
        )

    referenced_ids = {checkpoint_id for _row, _seed, checkpoint_id in references}
    if referenced_ids != set(checkpoints):
        raise ValueError("checkpoint_set contains missing or unreferenced checkpoint IDs")
    uses_by_checkpoint: dict[str, list[tuple[str, int]]] = {}
    for row_id, seed, checkpoint_id in references:
        checkpoint_variant = checkpoints[checkpoint_id]["variant_id"]
        if checkpoint_variant != _ROW_VARIANT_IDS[row_id]:
            raise ValueError(
                "decision row requires its exact native trained variant checkpoint"
            )
        uses_by_checkpoint.setdefault(checkpoint_id, []).append((row_id, seed))
    for uses in uses_by_checkpoint.values():
        if len(uses) > 1:
            if {row_id for row_id, _seed in uses} != {"Full", "NoF-deploy"}:
                raise ValueError("checkpoint sharing is limited to Full/NoF-deploy")
            if len({seed for _row_id, seed in uses}) != 1:
                raise ValueError("Full/NoF checkpoint sharing must use the same seed")
    for seed_index, seed in enumerate(SEEDS):
        full = rows[0]["seed_runs"][seed_index]
        no_f = rows[4]["seed_runs"][seed_index]
        if full["disposition"] == no_f["disposition"] == "RUN":
            if full["checkpoint_id"] != no_f["checkpoint_id"]:
                raise ValueError("same-seed Full/NoF must share the admitted Full checkpoint")

    episode_ledger = validate_file_identity(root["episode_ledger"], name="episode ledger")
    stop_receipt = validate_file_identity(root["stop_receipt"], name="STOP receipt")
    preflight_receipt = validate_file_identity(root["preflight_receipt"], name="preflight receipt")
    analysis_manifest = validate_file_identity(root["analysis_manifest"], name="analysis manifest")
    habitat = _exact_fields(
        root["habitat_scientific_identity"], {"sha256"}, name="Habitat scientific identity"
    )
    habitat_sha = require_sha256(
        habitat["sha256"], name="Habitat scientific identity SHA"
    )
    ranker = _validate_ranker_identity(root["ranker_identity"])
    candidate = _validate_candidate_contract(
        root["candidate_contract"],
        k_active=ranker["k_active"],
        h_active=ranker["h_active"],
    )
    claim = _exact_fields(root["claim_scope"], set(CLAIM_SCOPE), name="claim scope")
    if to_plain_json(claim) != CLAIM_SCOPE:
        raise ValueError("decision claim scope identity drift")

    live = _mapping(live_identity, name="live identity")
    required_live_fields = {
        "protocol_sha256",
        "row_id",
        "deployment",
        "training_seed",
        "checkpoint_id",
        "checkpoint_entry",
        "episode_ledger",
        "stop_receipt",
        "preflight_receipt",
        "habitat_scientific_identity",
        "ranker_identity",
        "candidate_contract",
        "analysis_manifest",
    }
    if not required_live_fields <= set(live) or set(live) - required_live_fields - {
        "checkpoint_set"
    }:
        raise ValueError("live identity fields are incomplete or unknown")
    if live.get("protocol_sha256") != expected_protocol:
        raise RuntimeError("live protocol SHA identity drift")
    live_deployment = _nonempty(live.get("deployment"), name="live deployment")
    if live_deployment != expected_deployment:
        raise RuntimeError("live deployment differs from expected deployment")
    live_row_id = _nonempty(live.get("row_id"), name="live row ID")
    if live_row_id not in ROW_IDS:
        raise ValueError("live row ID is not in the five-row decision")
    row_index = ROW_IDS.index(live_row_id)
    if DEPLOYMENTS[row_index] != live_deployment:
        raise RuntimeError("live row/deployment identity drift")
    live_seed = _nonnegative_integer(live.get("training_seed"), name="live training seed")
    if live_seed not in SEEDS:
        raise ValueError("live training seed is outside the frozen seed set")
    live_checkpoint_id = _nonempty(live.get("checkpoint_id"), name="live checkpoint ID")
    matching = [
        seed_row
        for row in rows
        if row["row_id"] == live_row_id and row["deployment"] == live_deployment
        for seed_row in row["seed_runs"]
        if seed_row["training_seed"] == live_seed
        and seed_row["disposition"] == "RUN"
    ]
    if len(matching) != 1:
        raise RuntimeError("current live row/seed/deployment must match exactly one RUN slot")
    if matching[0]["checkpoint_id"] != live_checkpoint_id:
        raise RuntimeError("live RUN checkpoint ID drift")
    live_checkpoint = _validate_checkpoint_entry(live.get("checkpoint_entry"))
    if live_checkpoint["checkpoint_id"] != live_checkpoint_id:
        raise RuntimeError("live checkpoint entry ID drift")
    _same(checkpoints[live_checkpoint_id], live_checkpoint, name="checkpoint")

    live_episode = validate_file_identity(live.get("episode_ledger"), name="live episode ledger")
    live_stop = validate_file_identity(live.get("stop_receipt"), name="live STOP receipt")
    live_preflight = validate_file_identity(
        live.get("preflight_receipt"), name="live preflight receipt"
    )
    live_analysis = validate_file_identity(
        live.get("analysis_manifest"), name="live analysis manifest"
    )
    _same(episode_ledger, live_episode, name="episode ledger")
    _same(stop_receipt, live_stop, name="STOP receipt")
    _same(preflight_receipt, live_preflight, name="preflight receipt")
    _same(analysis_manifest, live_analysis, name="analysis manifest")
    live_habitat = _exact_fields(
        live.get("habitat_scientific_identity"),
        {"sha256"},
        name="live Habitat scientific identity",
    )
    require_sha256(live_habitat["sha256"], name="live Habitat SHA")
    _same({"sha256": habitat_sha}, live_habitat, name="Habitat scientific")
    live_ranker = _validate_ranker_identity(live.get("ranker_identity"))
    live_candidate = _validate_candidate_contract(
        live.get("candidate_contract"),
        k_active=live_ranker["k_active"],
        h_active=live_ranker["h_active"],
    )
    _same(ranker, live_ranker, name="ranker K/H")
    _same(candidate, live_candidate, name="candidate K/H")

    frozen = deep_freeze(
        {
            "schema": DECISION_SCHEMA,
            "status": FROZEN_STATUS,
            "protocol_sha256": expected_protocol,
            "rows": rows,
            "seeds": list(SEEDS),
            "checkpoint_set": checkpoints_list,
            "episode_ledger": to_plain_json(episode_ledger),
            "stop_receipt": to_plain_json(stop_receipt),
            "preflight_receipt": to_plain_json(preflight_receipt),
            "habitat_scientific_identity": {"sha256": habitat_sha},
            "ranker_identity": ranker,
            "candidate_contract": candidate,
            "analysis_manifest": to_plain_json(analysis_manifest),
            "claim_scope": CLAIM_SCOPE,
        }
    )
    assert isinstance(frozen, Mapping)
    return frozen


def freeze_analysis_source_manifest(
    project_root: str | Path,
    protocol_sha256: str,
    output_path: str | Path,
) -> Mapping[str, object]:
    root = Path(project_root).expanduser().resolve()
    protocol = require_sha256(protocol_sha256, name="protocol SHA")
    files = []
    for relative in ANALYSIS_FILES:
        identity = file_identity(root / relative, name=f"analysis source {relative}")
        files.append(
            {
                "relative_path": relative,
                "bytes": identity["bytes"],
                "sha256": identity["sha256"],
            }
        )
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "status": FROZEN_STATUS,
        "protocol_sha256": protocol,
        "files": files,
        "operations": ANALYSIS_OPERATIONS,
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
    }
    validated = validate_analysis_source_manifest(
        payload, expected_protocol_sha256=protocol, project_root=root
    )
    create_once_json(output_path, validated)
    return validated


def _validate_intent(intent: object) -> list[dict[str, Any]]:
    root = _exact_fields(intent, {"schema", "rows"}, name="decision intent")
    if root["schema"] != DECISION_INTENT_SCHEMA:
        raise ValueError("decision intent schema is not current Context4")
    rows_raw = _sequence(root["rows"], name="decision intent rows")
    if len(rows_raw) != len(ROW_IDS):
        raise ValueError("decision intent must freeze all five rows")
    rows: list[dict[str, Any]] = []
    for raw, row_id in zip(rows_raw, ROW_IDS, strict=True):
        row = _exact_fields(raw, {"row_id", "seed_runs"}, name="decision intent row")
        if row["row_id"] != row_id:
            raise ValueError("decision intent row order/ID is invalid")
        seed_rows = _sequence(row["seed_runs"], name="decision intent seed runs")
        if len(seed_rows) != len(SEEDS):
            raise ValueError("decision intent must freeze all three seeds per row")
        validated: list[dict[str, Any]] = []
        for seed_row, seed in zip(seed_rows, SEEDS, strict=True):
            item = _exact_fields(
                seed_row,
                {"training_seed", "disposition", "outcome_independent_reason"},
                name="decision intent seed run",
            )
            if item["training_seed"] != seed or type(item["training_seed"]) is not int:
                raise ValueError("decision intent seed order/identity is invalid")
            disposition = item["disposition"]
            reason = item["outcome_independent_reason"]
            if disposition == "RUN":
                if reason is not None:
                    raise ValueError("RUN intent outcome-independent reason must be null")
            elif disposition == "NOT_RUN":
                _nonempty(reason, name="NOT_RUN intent outcome-independent reason")
            else:
                raise ValueError("decision intent disposition must be RUN or NOT_RUN")
            validated.append(
                {
                    "training_seed": seed,
                    "disposition": disposition,
                    "outcome_independent_reason": reason,
                }
            )
        rows.append({"row_id": row_id, "seed_runs": validated})
    return rows


def freeze_five_row_run_decision(
    intent: object,
    live_identity: Mapping[str, Any],
    output_path: str | Path,
) -> Mapping[str, object]:
    """Derive and atomically freeze one complete five-row/three-seed matrix."""

    rows = _validate_intent(intent)
    live = _mapping(live_identity, name="live identity")
    protocol = require_sha256(live.get("protocol_sha256"), name="live protocol SHA")
    current_checkpoint = _validate_checkpoint_entry(live.get("checkpoint_entry"))
    available_raw = live.get("checkpoint_set", [current_checkpoint])
    available = [
        _validate_checkpoint_entry(row)
        for row in _sequence(available_raw, name="live checkpoint set")
    ]
    by_seed_and_variant = {
        (row["training_seed"], row["variant_id"]): row for row in available
    }
    if len(by_seed_and_variant) != len(available):
        raise ValueError("live checkpoint set has duplicate native seed/variant entries")
    current_row = _nonempty(live.get("row_id"), name="live row ID")
    current_seed = _nonnegative_integer(live.get("training_seed"), name="live training seed")
    current_deployment = _nonempty(live.get("deployment"), name="live deployment")
    if current_row not in ROW_IDS or DEPLOYMENTS[ROW_IDS.index(current_row)] != current_deployment:
        raise ValueError("live row/deployment is outside the frozen matrix")
    if current_checkpoint["training_seed"] != current_seed:
        raise ValueError("live checkpoint native seed identity drift")
    expected_current_variant = _ROW_VARIANT_IDS[current_row]
    if current_checkpoint["variant_id"] != expected_current_variant:
        raise ValueError("live row and checkpoint native variant identity disagree")

    decision_rows: list[dict[str, Any]] = []
    selected_checkpoints: dict[str, dict[str, Any]] = {}
    for intent_row, deployment in zip(rows, DEPLOYMENTS, strict=True):
        row_id = intent_row["row_id"]
        produced_seed_runs: list[dict[str, Any]] = []
        for seed_run in intent_row["seed_runs"]:
            checkpoint_id: str | None = None
            if seed_run["disposition"] == "RUN":
                seed = seed_run["training_seed"]
                checkpoint = by_seed_and_variant.get(
                    (seed, _ROW_VARIANT_IDS[row_id])
                )
                if checkpoint is None:
                    raise ValueError(
                        f"RUN slot {row_id}/seed{seed} has no native-seed checkpoint"
                    )
                checkpoint_id = checkpoint["checkpoint_id"]
                selected_checkpoints[checkpoint_id] = checkpoint
            produced_seed_runs.append(
                {
                    **seed_run,
                    "checkpoint_id": checkpoint_id,
                }
            )
        decision_rows.append(
            {
                "row_id": row_id,
                "deployment": deployment,
                "seed_runs": produced_seed_runs,
            }
        )
    payload = {
        "schema": DECISION_SCHEMA,
        "status": FROZEN_STATUS,
        "protocol_sha256": protocol,
        "rows": decision_rows,
        "seeds": list(SEEDS),
        "checkpoint_set": [selected_checkpoints[key] for key in sorted(selected_checkpoints)],
        "episode_ledger": to_plain_json(live.get("episode_ledger")),
        "stop_receipt": to_plain_json(live.get("stop_receipt")),
        "preflight_receipt": to_plain_json(live.get("preflight_receipt")),
        "habitat_scientific_identity": to_plain_json(live.get("habitat_scientific_identity")),
        "ranker_identity": to_plain_json(live.get("ranker_identity")),
        "candidate_contract": to_plain_json(live.get("candidate_contract")),
        "analysis_manifest": to_plain_json(live.get("analysis_manifest")),
        "claim_scope": CLAIM_SCOPE,
    }
    validated = validate_five_row_run_decision(
        payload,
        expected_protocol_sha256=protocol,
        expected_deployment=current_deployment,
        live_identity=live,
    )
    create_once_json(output_path, validated)
    return validated


def bind_context4_final_receipt(
    payload: Mapping[str, Any],
    *,
    adapter_provenance: Mapping[str, Any],
    live_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a final mature-runner payload to both immutable current artifacts."""

    raw = _mapping(payload, name="Context4 final receipt")
    existing_schema = raw.get("schema")
    if existing_schema is not None and existing_schema != FINAL_RECEIPT_SCHEMA:
        raise ValueError(
            "Context4 final receipt identity/schema cannot relabel generic evidence"
        )
    provenance = _mapping(adapter_provenance, name="adapter provenance")
    runtime = provenance.get("runtime_identity")
    owner = runtime if isinstance(runtime, Mapping) else provenance
    live = _mapping(live_identity, name="live final identity")
    bound: dict[str, Any] = {}
    for field, label in (
        ("decision", "decision"),
        ("analysis_manifest", "analysis manifest"),
    ):
        admitted = validate_file_identity(owner.get(field), name=f"adapter {label}")
        observed = validate_file_identity(live.get(field), name=f"live {label}")
        _same(admitted, observed, name=label)
        if field in raw:
            _same(raw[field], admitted, name=f"receipt {label}")
        bound[field] = to_plain_json(admitted)
    result = to_plain_json(raw)
    result["schema"] = FINAL_RECEIPT_SCHEMA
    result.update(bound)
    # Reject NaN/Inf and any non-JSON residue before returning publication bytes.
    json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return result


def load_bound_json(identity: object, *, name: str) -> tuple[dict[str, Any], Mapping[str, object]]:
    """Public consumer helper used by the canonical factory/launcher."""

    return load_json_file_identity(identity, name=name)


__all__ = [
    "ANALYSIS_FILES",
    "ANALYSIS_SCHEMA",
    "DECISION_INTENT_SCHEMA",
    "DECISION_SCHEMA",
    "FINAL_RECEIPT_SCHEMA",
    "bind_context4_final_receipt",
    "canonical_mapping_sha256",
    "freeze_analysis_source_manifest",
    "freeze_five_row_run_decision",
    "load_bound_json",
    "resolve_variant_decision_row",
    "validate_analysis_source_manifest",
    "validate_five_row_run_decision",
]
