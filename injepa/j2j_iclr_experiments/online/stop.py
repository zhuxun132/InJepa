"""Source-bound full-grid STOP calibration and zero-intent G probe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch import Tensor

from j2j.adapter import ActionId, Raw4Adapter
from j2j.authority import sha256_file
from j2j.data.annotations import AnnotationRow
from j2j.data.keys import frame_key as canonical_frame_key
from j2j.data.keys import trajectory_key as canonical_trajectory_key
from j2j.data.source import (
    load_canonical_manifest_partition,
    open_released_streamvln_source,
)
from j2j.encoding.cache import open_cache_store
from j2j.encoding.vjepa2 import OFFICIAL_SOURCE_COMMIT, OFFICIAL_SOURCE_TREE
from j2j.evaluation.vjepa_grid_encoder import (
    VJEPA_POOL_IDENTITY_SHA256,
    VJEPA_PREPROCESS_IDENTITY_SHA256,
)
from j2j.receipts import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_IDENTITY = re.compile(r"[0-9a-f]{40}")
_FRAME_KEY = re.compile(r"[0-9a-f]{72}")
_PROTOCOL_SHA256 = "82600ee13fee62646182257c8c56472a09e8916adc9d513f18abb887b0dc72fc"
_LEDGER_SCHEMA = "J2J_STOP_CACHE_BOUND_LEDGER_V2"
_PAIR_CONTRACTS = {
    "key": "J2J_STOP_PAIR_KEY_V1",
    "positive": "J2J_STOP_POS_V1",
    "negative": "J2J_STOP_NEG_V1",
    "source_binding": _LEDGER_SCHEMA,
}
_LEDGER_FIELDS = {
    "pair_key", "positive_pair_key", "pair_type", "label",
    "source_id", "goal_source_id", "partition", "goal_partition",
    "building", "goal_building", "current_scan", "goal_scan",
    "trajectory_key", "goal_trajectory_key", "current_step", "goal_step",
    "terminal_step", "transition_action_id", "current_frame_key",
    "goal_frame_key", "current_rgb_sha256", "goal_rgb_sha256", "distance",
    "source_manifest_sha256",
}
_PROVENANCE_FIELDS = {
    "schema", "protocol_sha256", "partition", "pair_contracts", "ledger",
    "artifacts", "visual_coordinate_identity", "z32_parent_manifest_sha256",
    "numpy_version", "bootstrap",
}
_ARTIFACT_FIELDS = {
    "canonical_source_manifest", "source_catalog", "cache_manifest",
    "vjepa_checkpoint", "whitening", "producer_entrypoints_code",
    "validator_stop_code", "source_loader_code", "cache_loader_code",
}
_COORDINATE_FIELDS = {
    "vjepa_source_commit", "vjepa_source_tree", "checkpoint_sha256",
    "preprocess_sha256", "pool_sha256", "whitening_sha256",
}
_BOOTSTRAP_FIELDS = {
    "cluster_unit", "replicates", "bit_generator", "seed", "ci_level",
    "quantile_method",
}
_FORMAL_BOOTSTRAP = {
    "cluster_unit": "building",
    "replicates": 10_000,
    "bit_generator": "PCG64",
    "seed": 0,
    "ci_level": 0.95,
    "quantile_method": "linear",
}
_CACHE_IDENTITY_FIELDS = {
    "vjepa_source_commit",
    "checkpoint_sha256",
    "preprocess_sha256",
    "pool_sha256",
    "whitening_sha256",
}
_THRESHOLD_RULE = {
    "objective": "balanced_error_rate",
    "formula": "0.5*(FNR+FPR)",
    "comparison": "distance<=threshold",
    "tie_break": "smallest_threshold",
}


def _finite_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} must be finite")


def canonical_stop_distance(current_grid: object, goal_grid: object) -> float:
    """Return the single CPU/C-order/float32 STOP-distance reduction."""

    def _canonical_grid(value: object, *, name: str) -> np.ndarray:
        if isinstance(value, Tensor):
            value = value.detach().cpu().numpy()
        try:
            grid = np.asarray(value, dtype=np.dtype("<f4"), order="C")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"STOP {name} grid is not a numeric spatial array") from exc
        if grid.ndim != 2 or grid.shape[0] < 1 or grid.shape[1] != 768:
            raise ValueError(f"STOP {name} grid must be nonempty [spatial,768]")
        if not grid.flags.c_contiguous:
            grid = np.ascontiguousarray(grid, dtype=np.dtype("<f4"))
        if not np.isfinite(grid).all():
            raise FloatingPointError(f"STOP {name} grid must be finite")
        return grid

    current = _canonical_grid(current_grid, name="current")
    goal = _canonical_grid(goal_grid, name="goal")
    if current.shape != goal.shape:
        raise ValueError("STOP current/goal grids must have identical shapes and coordinates")
    difference = np.subtract(current, goal, dtype=np.float32)
    distance = np.mean(np.abs(difference), dtype=np.float32)
    if not np.isfinite(distance) or distance < 0:
        raise FloatingPointError("STOP canonical distance must be finite and non-negative")
    return float(distance)


def zero_intent_stop_action(
    actor_step, current_grid: Tensor, previous_raw4: Tensor
) -> Tensor:
    """Call the shared G in its trained terminal, zero-intent coordinate."""

    if not callable(actor_step):
        raise TypeError("actor_step must be callable")
    if current_grid.ndim != 3:
        raise ValueError("current_grid must be [batch,spatial,latent]")
    if previous_raw4.shape != (current_grid.shape[0], 4):
        raise ValueError("previous_raw4 must be [batch,4]")
    _finite_tensor("current_grid", current_grid)
    _finite_tensor("previous_raw4", previous_raw4)
    logits = actor_step(current_grid, torch.zeros_like(current_grid), previous_raw4)
    if not isinstance(logits, Tensor) or logits.shape != (current_grid.shape[0], 4):
        raise ValueError("actor_step must return [batch,4]")
    _finite_tensor("actor logits", logits)
    return Raw4Adapter.decode_logits(logits)


def _domain(label: str) -> bytes:
    return label.encode("ascii") + b"\x00"


def _length_prefix(payload: bytes) -> bytes:
    return len(payload).to_bytes(8, "little") + payload


def _hex_bytes(value: object, *, name: str, pattern: re.Pattern[str]) -> bytes:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"STOP {name} has an invalid hash identity")
    return bytes.fromhex(value)


def _nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"STOP {name} must be a non-negative integer")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"STOP {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"STOP {name} must be finite")
    return number


def _frame_key(trajectory_key: str, step: int) -> str:
    trajectory = _hex_bytes(trajectory_key, name="trajectory", pattern=_SHA256)
    index = _nonnegative_int(step, name="frame step")
    if index >= 2**32:
        raise ValueError("STOP frame step exceeds uint32")
    return (trajectory + index.to_bytes(4, "little")).hex()


def _pair_key(pair_type: str, current_frame_key: str, goal_frame_key: str) -> str:
    current = _hex_bytes(current_frame_key, name="current frame", pattern=_FRAME_KEY)
    goal = _hex_bytes(goal_frame_key, name="goal frame", pattern=_FRAME_KEY)
    return hashlib.sha256(
        _domain("J2J_STOP_PAIR_KEY_V1")
        + _length_prefix(pair_type.encode("ascii"))
        + current + goal
    ).hexdigest()


def _positive_selection_key(trajectory_key: str, step: int) -> bytes:
    return hashlib.sha256(
        _domain("J2J_STOP_POS_V1")
        + bytes.fromhex(trajectory_key)
        + step.to_bytes(4, "little")
    ).digest()


def _source_derived_population(
    *,
    canonical_source_manifest_path: str | Path,
    canonical_source_manifest_sha256: str,
) -> tuple[dict[str, Any], ...]:
    """Rebuild every project-dev STOP pair from the canonical source."""

    items = load_canonical_manifest_partition(
        canonical_source_manifest_path,
        expected_manifest_sha256=canonical_source_manifest_sha256,
        partition="project-dev",
        require_production_eligible=True,
    )
    if not items:
        raise ValueError("STOP source project-dev trajectory population is empty")
    frames: list[dict[str, Any]] = []
    positives: list[dict[str, Any]] = []
    for item in items:
        trajectory = item.canonical_trajectory
        key = trajectory.canonical_trajectory_key.hex()
        actions = trajectory.actions
        if any(
            type(action) is not int or action not in (1, 2, 3)
            for action in actions[1:]
        ):
            raise ValueError(
                "STOP source trajectory actions must be factual motion actions"
            )
        recomputed_key = canonical_trajectory_key(
            AnnotationRow(
                source_id=trajectory.source_id,
                row_index=trajectory.annotation_row_index,
                episode_id=trajectory.annotation_row_index,
                scan_id=trajectory.scan_id,
                video_prefix=trajectory.canonical_video_prefix,
                actions=actions,
            ),
            annotation_revision_sha256=item.annotation_sha256,
        )
        if recomputed_key != trajectory.canonical_trajectory_key:
            raise ValueError(
                "STOP source trajectory key differs from its source/action identity"
            )
        terminal = len(actions) - 1
        for step, rgb_sha in enumerate(trajectory.decoded_rgb_sha256s):
            frames.append({
                "source_id": trajectory.source_id,
                "partition": item.projection_partition,
                "building": trajectory.scan_id,
                "scan": trajectory.scan_id,
                "trajectory_key": key,
                "step": step,
                "frame_key": _frame_key(key, step),
                "rgb_sha256": rgb_sha,
            })
        selected: list[tuple[str, int, int | None]] = [("SELF", terminal, None)]
        turn_steps = [
            step for step in range(terminal)
            if actions[step + 1] in (int(ActionId.LEFT), int(ActionId.RIGHT))
        ]
        if turn_steps:
            step = min(
                turn_steps,
                key=lambda value: (_positive_selection_key(key, value), value),
            )
            selected.append(("PURE_TURN", step, int(actions[step + 1])))
        for pair_type, current_step, transition_action in selected:
            goal_step = current_step if pair_type == "SELF" else current_step + 1
            current_frame = _frame_key(key, current_step)
            goal_frame = _frame_key(key, goal_step)
            pair = _pair_key(pair_type, current_frame, goal_frame)
            positives.append({
                "pair_key": pair,
                "positive_pair_key": pair,
                "pair_type": pair_type,
                "label": True,
                "source_id": trajectory.source_id,
                "goal_source_id": trajectory.source_id,
                "partition": item.projection_partition,
                "goal_partition": item.projection_partition,
                "building": trajectory.scan_id,
                "goal_building": trajectory.scan_id,
                "current_scan": trajectory.scan_id,
                "goal_scan": trajectory.scan_id,
                "trajectory_key": key,
                "goal_trajectory_key": key,
                "current_step": current_step,
                "goal_step": goal_step,
                "terminal_step": terminal,
                "transition_action_id": transition_action,
                "current_frame_key": current_frame,
                "goal_frame_key": goal_frame,
                "current_rgb_sha256": trajectory.decoded_rgb_sha256s[current_step],
                "goal_rgb_sha256": trajectory.decoded_rgb_sha256s[goal_step],
                "source_manifest_sha256": canonical_source_manifest_sha256,
            })
    frames.sort(key=lambda row: bytes.fromhex(row["frame_key"]))
    positives.sort(key=lambda row: bytes.fromhex(row["pair_key"]))
    result: list[dict[str, Any]] = []
    for positive in positives:
        result.append(positive)
        candidates = [
            frame for frame in frames
            if frame["source_id"] == positive["source_id"]
            and frame["partition"] == positive["partition"]
            and frame["scan"] != positive["current_scan"]
        ]
        if not candidates:
            raise ValueError(
                "STOP source has an empty different-scan negative population"
            )
        choice = int.from_bytes(
            hashlib.sha256(
                _domain("J2J_STOP_NEG_V1") + bytes.fromhex(positive["pair_key"])
            ).digest()[:8],
            "big",
        ) % len(candidates)
        goal = candidates[choice]
        result.append({
            "pair_key": _pair_key(
                "NEGATIVE", positive["current_frame_key"], goal["frame_key"]
            ),
            "positive_pair_key": positive["pair_key"],
            "pair_type": "NEGATIVE",
            "label": False,
            "source_id": positive["source_id"],
            "goal_source_id": goal["source_id"],
            "partition": positive["partition"],
            "goal_partition": goal["partition"],
            "building": positive["building"],
            "goal_building": goal["building"],
            "current_scan": positive["current_scan"],
            "goal_scan": goal["scan"],
            "trajectory_key": positive["trajectory_key"],
            "goal_trajectory_key": goal["trajectory_key"],
            "current_step": positive["current_step"],
            "goal_step": goal["step"],
            "terminal_step": positive["terminal_step"],
            "transition_action_id": None,
            "current_frame_key": positive["current_frame_key"],
            "goal_frame_key": goal["frame_key"],
            "current_rgb_sha256": positive["current_rgb_sha256"],
            "goal_rgb_sha256": goal["rgb_sha256"],
            "source_manifest_sha256": canonical_source_manifest_sha256,
        })
    return tuple(result)


def validate_stop_pair_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    canonical_source_manifest_path: str | Path,
    canonical_source_manifest_sha256: str,
) -> Mapping[str, object]:
    """Validate an exact source-derived pair population and its distances."""

    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or not rows
    ):
        raise ValueError("STOP pair ledger must be a nonempty sequence")
    expected_rows = _source_derived_population(
        canonical_source_manifest_path=canonical_source_manifest_path,
        canonical_source_manifest_sha256=canonical_source_manifest_sha256,
    )
    expected = {str(row["pair_key"]): row for row in expected_rows}
    observed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("STOP ledger rows must be mappings")
        if set(row) != _LEDGER_FIELDS:
            raise ValueError("STOP ledger row fields are incomplete or noncanonical")
        pair_key = row.get("pair_key")
        _hex_bytes(pair_key, name="pair key", pattern=_SHA256)
        if pair_key in observed:
            raise ValueError("STOP pair ledger contains a duplicate pair key")
        distance = _finite_number(row.get("distance"), name="distance")
        if distance < 0.0:
            raise ValueError("STOP distance must be non-negative")
        observed[str(pair_key)] = row
    if set(observed) != set(expected):
        raise ValueError(
            "STOP ledger is not the complete source-derived project-dev population"
        )
    for pair_key, expected_row in expected.items():
        row = observed[pair_key]
        for field, expected_value in expected_row.items():
            if row.get(field) != expected_value:
                raise ValueError(
                    f"STOP ledger {field} differs from the source-derived population"
                )
    positive_by_type: Counter[str] = Counter(
        str(row["pair_type"]) for row in rows if bool(row["label"])
    )
    pure_turn_rows = [row for row in rows if row["pair_type"] == "PURE_TURN"]
    pure_turn_buildings = {str(row["building"]) for row in pure_turn_rows}
    if not pure_turn_rows:
        raise ValueError("STOP source population has no PURE_TURN positive")
    if len(pure_turn_buildings) < 2:
        raise ValueError("PURE_TURN positives must cover at least two buildings")
    if not any(float(row["distance"]) > 0.0 for row in pure_turn_rows):
        raise ValueError("PURE_TURN distances may not all be zero")
    for row in rows:
        if row["pair_type"] == "SELF" and float(row["distance"]) != 0.0:
            raise ValueError("STOP SELF source pair distance must be exactly zero")
    return {
        "positive_by_type": dict(positive_by_type),
        "negative_count": sum(row["pair_type"] == "NEGATIVE" for row in rows),
        "pure_turn_building_count": len(pure_turn_buildings),
        "partition": "project-dev",
        "formal_eligible": True,
    }


def recompute_cache_bound_stop_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, object]]:
    """Reopen admitted source/cache artifacts and recompute every distance."""

    artifacts = _receipt_mapping(
        provenance.get("artifacts"), name="provenance.artifacts"
    )
    coordinate = _receipt_mapping(
        provenance.get("visual_coordinate_identity"),
        name="provenance.visual_coordinate_identity",
    )
    source_identity = _receipt_mapping(
        artifacts.get("canonical_source_manifest"),
        name="canonical source manifest",
    )
    catalog_identity = _receipt_mapping(
        artifacts.get("source_catalog"), name="source catalog"
    )
    cache_identity = _receipt_mapping(
        artifacts.get("cache_manifest"), name="Z32 cache manifest"
    )
    cache_manifest_path = Path(str(cache_identity.get("path", ""))).resolve()
    if cache_manifest_path.name != "manifest.json":
        raise ValueError("STOP Z32 cache artifact must be the live manifest.json")
    expected_identities = {
        field: coordinate.get(field) for field in _CACHE_IDENTITY_FIELDS
    }
    try:
        cache_store = open_cache_store(
            cache_manifest_path.parent,
            expected_parent_manifest_sha256=str(
                provenance.get("z32_parent_manifest_sha256", "")
            ),
            expected_identities=expected_identities,
            require_stage="Z32",
            require_training_eligible=True,
            require_production_eligible=True,
        )
        source = open_released_streamvln_source(
            canonical_manifest=str(source_identity.get("path", "")),
            source_catalog=str(catalog_identity.get("path", "")),
            cache_store=cache_store,
            expected_manifest_sha256=str(source_identity.get("sha256", "")),
            partition="project-dev",
            require_production_eligible=True,
        )
    except Exception as exc:
        raise ValueError(
            f"STOP source catalog/Z32 cache authority could not be admitted: {exc}"
        ) from exc

    census = validate_stop_pair_ledger(
        rows,
        canonical_source_manifest_path=str(source_identity.get("path", "")),
        canonical_source_manifest_sha256=str(source_identity.get("sha256", "")),
    )

    locators: dict[str, tuple[int, dict[str, Any]]] = {}
    used_global_rows: set[int] = set()
    for item in source:
        trajectory = item.canonical_trajectory
        trajectory_key = trajectory.canonical_trajectory_key
        for step, global_row in enumerate(item.cache_global_rows):
            frame = canonical_frame_key(trajectory_key, step).hex()
            if frame in locators:
                raise ValueError("STOP source catalog duplicates a canonical frame key")
            if global_row in used_global_rows:
                raise ValueError("STOP source catalog reuses a Z32 cache row")
            used_global_rows.add(global_row)
            locators[frame] = (
                global_row,
                {
                    "frame_key": frame,
                    "source_dataset": trajectory.source_id,
                    "partition": item.projection_partition,
                    "building": trajectory.scan_id,
                    "compressed_jpeg_sha256": trajectory.compressed_jpeg_sha256s[
                        step
                    ],
                    "decoded_rgb_sha256": trajectory.decoded_rgb_sha256s[step],
                },
            )

    recomputed_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            current_locator, current_expected = locators[
                str(row["current_frame_key"])
            ]
            goal_locator, goal_expected = locators[str(row["goal_frame_key"])]
        except KeyError as exc:
            raise ValueError(
                "STOP ledger frame is absent from the admitted source catalog"
            ) from exc
        try:
            current_record, goal_record = cache_store.read_global_rows(
                (current_locator, goal_locator)
            )
        except Exception as exc:
            raise ValueError("STOP Z32 cache pair read failed") from exc
        for role, record, expected in (
            ("current", current_record, current_expected),
            ("goal", goal_record, goal_expected),
        ):
            observed = {
                "frame_key": getattr(record, "frame_key", b"").hex(),
                "source_dataset": getattr(record, "source_dataset", None),
                "partition": getattr(record, "partition", None),
                "building": getattr(record, "building", None),
                "compressed_jpeg_sha256": getattr(
                    record, "compressed_jpeg_sha256", None
                ),
                "decoded_rgb_sha256": getattr(record, "decoded_rgb_sha256", None),
            }
            if observed != expected:
                raise ValueError(
                    f"STOP {role} source catalog/Z32 cache frame identity mismatch"
                )
        distance = canonical_stop_distance(current_record.grid, goal_record.grid)
        if float(row["distance"]) != distance:
            raise ValueError("STOP ledger distance differs from the live Z32 cache")
        recomputed = dict(row)
        recomputed["distance"] = distance
        recomputed_rows.append(recomputed)

    pure_turn = [row for row in recomputed_rows if row["pair_type"] == "PURE_TURN"]
    if not any(row["distance"] > 0.0 for row in pure_turn):
        raise ValueError("STOP live Z32 PURE_TURN distances may not all be zero")
    for row in recomputed_rows:
        if row["pair_type"] == "SELF" and row["distance"] != 0.0:
            raise ValueError("STOP live Z32 SELF distance must be exactly zero")
    return tuple(recomputed_rows), census


def load_stop_pair_ledger(path: str | Path) -> tuple[Mapping[str, Any], ...]:
    """Read exact canonical JSONL; blank, partial, or noncanonical rows fail."""

    ledger_path = Path(path).expanduser().resolve()
    try:
        payload = ledger_path.read_bytes()
    except OSError as exc:
        raise ValueError("STOP calibration ledger is unavailable") from exc
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("STOP calibration ledger is not canonical JSONL")
    rows: list[Mapping[str, Any]] = []
    for line in payload.splitlines(keepends=True):
        if line == b"\n" or not line.endswith(b"\n"):
            raise ValueError("STOP calibration ledger is not canonical JSONL")
        try:
            row = json.loads(line[:-1].decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("STOP calibration ledger is not valid JSONL") from exc
        if not isinstance(row, Mapping):
            raise ValueError("STOP calibration ledger row is not a JSON object")
        try:
            canonical = canonical_json_bytes(dict(row))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "STOP calibration ledger contains noncanonical values"
            ) from exc
        if canonical != line:
            raise ValueError("STOP calibration ledger is not canonical JSONL")
        rows.append(dict(row))
    if not rows:
        raise ValueError("STOP calibration ledger is empty")
    return tuple(rows)


def select_stop_threshold(distances: Tensor, labels: Tensor) -> float:
    """Minimize balanced error for the inclusive ``distance <= tau`` gate."""

    if not isinstance(distances, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("distances and labels must be tensors")
    if distances.ndim != 1 or labels.shape != distances.shape:
        raise ValueError("distances and labels must be aligned vectors")
    if labels.dtype != torch.bool:
        raise TypeError("STOP labels must be bool")
    if distances.numel() < 2 or not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError("STOP threshold requires both classes")
    if not bool(torch.isfinite(distances).all()) or bool((distances < 0).any()):
        raise ValueError("STOP distances must be finite and non-negative")
    values = distances.detach().to(device="cpu", dtype=torch.float64).numpy()
    truth = labels.detach().cpu().numpy()
    thresholds, true_positive, false_positive = _sorted_binary_counts(values, truth)
    positives = float(truth.sum())
    negatives = float((~truth).sum())
    error = ((positives - true_positive) / positives + false_positive / negatives) / 2.0
    # Counts are sampled only at the final element of each equal-distance
    # group. This preserves the inclusive gate and smallest-threshold tie rule.
    return float(thresholds[int(np.argmin(error))])


def _rates(
    distances: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float]:
    predicted = distances <= threshold
    false_negative_rate = float(np.mean(~predicted[labels]))
    false_positive_rate = float(np.mean(predicted[~labels]))
    balanced_error = 0.5 * (false_negative_rate + false_positive_rate)
    return balanced_error, 1.0 - balanced_error


def _sorted_binary_counts(distances: np.ndarray, labels: np.ndarray):
    order = np.argsort(distances, kind="stable")
    values = distances[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    true_positive = np.cumsum(labels[order], dtype=np.int64)[ends]
    false_positive = ends + 1 - true_positive
    return values[ends], true_positive, false_positive


def _roc(distances: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("STOP ROC requires both classes")
    _, tp, fp = _sorted_binary_counts(distances, labels)
    false_positive = np.r_[0.0, fp / negatives].tolist()
    true_positive = np.r_[0.0, tp / positives].tolist()
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return {
        "positive_direction": "smaller_distance",
        "auc": float(integrate(true_positive, false_positive)),
        "false_positive_rate": false_positive,
        "true_positive_rate": true_positive,
    }


def _receipt_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"STOP receipt {name} must be a mapping")
    return value


def _validate_bootstrap_parameters(value: object) -> dict[str, Any]:
    parameters = _receipt_mapping(value, name="bootstrap parameters")
    if set(parameters) != _BOOTSTRAP_FIELDS:
        raise ValueError("STOP bootstrap parameter fields are not exact")
    if dict(parameters) != _FORMAL_BOOTSTRAP:
        raise ValueError(
            "STOP formal bootstrap must be exactly 10000/building/PCG64/seed0/"
            "95%-linear"
        )
    return dict(parameters)


def _validate_configurable_bootstrap(value: object) -> dict[str, Any]:
    parameters = _receipt_mapping(value, name="bootstrap parameters")
    if set(parameters) != _BOOTSTRAP_FIELDS:
        raise ValueError("STOP bootstrap parameter fields are not exact")
    if (parameters["cluster_unit"] != "building"
            or parameters["bit_generator"] != "PCG64"
            or parameters["quantile_method"] != "linear"
            or type(parameters["replicates"]) is not int or parameters["replicates"] < 1
            or type(parameters["seed"]) is not int or parameters["seed"] < 0
            or isinstance(parameters["ci_level"], bool)
            or not isinstance(parameters["ci_level"], (int, float))
            or not 0 < parameters["ci_level"] < 1):
        raise ValueError("STOP bootstrap parameters are invalid")
    return dict(parameters)


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *, threshold: float,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    buildings = sorted({str(row["building"]) for row in rows})
    grouped = {
        building: tuple(row for row in rows if str(row["building"]) == building)
        for building in buildings
    }
    rng = np.random.Generator(np.random.PCG64(int(parameters["seed"])))
    replicates = int(parameters["replicates"])
    balanced_accuracy = np.empty(replicates, dtype=np.float64)
    auc = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(buildings), size=len(buildings))
        sample_rows = [
            row for building_index in sampled
            for row in grouped[buildings[int(building_index)]]
        ]
        distances = np.asarray(
            [float(row["distance"]) for row in sample_rows], dtype=np.float64
        )
        labels = np.asarray(
            [bool(row["label"]) for row in sample_rows], dtype=np.bool_
        )
        _balanced_error, balanced_accuracy[index] = _rates(
            distances, labels, threshold
        )
        auc[index] = float(_roc(distances, labels)["auc"])
    tail = (1.0 - float(parameters["ci_level"])) / 2.0
    interval = (tail, 1.0 - tail)
    return {
        "parameters": dict(parameters),
        "balanced_accuracy": np.quantile(
            balanced_accuracy, interval, method="linear"
        ).tolist(),
        "roc_auc": np.quantile(auc, interval, method="linear").tolist(),
        "degenerate_replicates": bool(
            np.all(balanced_accuracy == balanced_accuracy[0])
            and np.all(auc == auc[0])
        ),
    }


def derive_stop_calibration(
    rows: Sequence[Mapping[str, Any]],
    *, census: Mapping[str, object],
    bootstrap_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute every scientific receipt field from one validated ledger."""

    parameters = _validate_configurable_bootstrap(bootstrap_parameters)
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.bool_)
    distances = np.asarray(
        [float(row["distance"]) for row in rows], dtype=np.float64
    )
    threshold = select_stop_threshold(
        torch.from_numpy(distances), torch.from_numpy(labels)
    )
    balanced_error, balanced_accuracy = _rates(distances, labels, threshold)
    derived_census = dict(census)
    derived_census["class_counts"] = {
        "positive": int(labels.sum()),
        "negative": int((~labels).sum()),
    }
    derived_census["building_count"] = len(
        {str(row["building"]) for row in rows}
    )
    return {
        "census": derived_census,
        "threshold": threshold,
        "balanced_error_rate": balanced_error,
        "balanced_accuracy": balanced_accuracy,
        "threshold_rule": dict(_THRESHOLD_RULE),
        "roc": _roc(distances, labels),
        "bootstrap": _cluster_bootstrap(
            rows, threshold=threshold, parameters=parameters
        ),
    }


def _receipt_sha(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(
            f"STOP receipt {name} identity must be a lowercase SHA-256"
        )
    return value


def _verified_receipt_file(value: object, *, name: str) -> tuple[Path, bytes]:
    identity = _receipt_mapping(value, name=name)
    if set(identity) != {"path", "bytes", "sha256"}:
        raise ValueError(f"STOP receipt {name} identity fields are invalid")
    path = Path(str(identity.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"STOP receipt {name} file is unavailable")
    expected_bytes = identity.get("bytes")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError(f"STOP receipt {name} bytes must be positive")
    expected_sha = _receipt_sha(identity.get("sha256"), name=name)
    payload = path.read_bytes()
    if len(payload) != expected_bytes or sha256_file(path) != expected_sha:
        raise ValueError(f"STOP receipt {name} live bytes/SHA identity mismatch")
    return path, payload


def _current_code_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_stop_calibration_provenance(
    provenance: Mapping[str, object],
    *,
    ledger_identity: Mapping[str, object],
    expected_bootstrap: Mapping[str, object] | None = None,
    expected_numpy_version: str | None = None,
) -> Mapping[str, object]:
    """Validate the sole source-bound provenance schema and its live bytes."""

    value = _receipt_mapping(provenance, name="provenance")
    if set(value) != _PROVENANCE_FIELDS:
        raise ValueError("STOP provenance fields are incomplete or noncanonical")
    if value.get("schema") != "J2J_STOP_CALIBRATION_PROVENANCE_V3":
        raise ValueError("STOP provenance schema is invalid")
    if value.get("protocol_sha256") != _PROTOCOL_SHA256:
        raise ValueError("STOP provenance protocol identity is invalid")
    if value.get("partition") != "project-dev":
        raise ValueError("STOP provenance partition must be project-dev")
    pair_contracts = _receipt_mapping(
        value.get("pair_contracts"), name="provenance.pair_contracts"
    )
    if dict(pair_contracts) != _PAIR_CONTRACTS:
        raise ValueError(
            "STOP provenance pair/source construction identity is invalid"
        )
    ledger = _receipt_mapping(value.get("ledger"), name="provenance.ledger")
    if dict(ledger) != dict(ledger_identity):
        raise ValueError("STOP provenance ledger identity differs from the receipt")
    _verified_receipt_file(ledger, name="provenance ledger")

    artifacts = _receipt_mapping(
        value.get("artifacts"), name="provenance.artifacts"
    )
    if set(artifacts) != _ARTIFACT_FIELDS:
        raise ValueError("STOP provenance artifact fields are not exact")
    verified: dict[str, Mapping[str, object]] = {}
    for name in sorted(_ARTIFACT_FIELDS):
        identity = _receipt_mapping(artifacts[name], name=f"artifacts.{name}")
        _verified_receipt_file(identity, name=f"artifact {name}")
        verified[name] = identity
    expected_entrypoints = Path(__file__).with_name("entrypoints.py").resolve()
    expected_stop = Path(__file__).resolve()
    if dict(verified["producer_entrypoints_code"]) != _current_code_identity(
        expected_entrypoints
    ):
        raise ValueError("STOP producer entrypoint code identity drifted")
    if dict(verified["validator_stop_code"]) != _current_code_identity(expected_stop):
        raise ValueError("STOP provenance validator stop.py code identity drifted")
    source_loader = Path(open_released_streamvln_source.__code__.co_filename).resolve()
    cache_loader = Path(open_cache_store.__code__.co_filename).resolve()
    if dict(verified["source_loader_code"]) != _current_code_identity(source_loader):
        raise ValueError("STOP provenance source loader code identity drifted")
    if dict(verified["cache_loader_code"]) != _current_code_identity(cache_loader):
        raise ValueError("STOP provenance cache loader code identity drifted")

    _receipt_sha(
        value.get("z32_parent_manifest_sha256"),
        name="Z32 parent manifest",
    )

    coordinate = _receipt_mapping(
        value.get("visual_coordinate_identity"),
        name="provenance.visual_coordinate_identity",
    )
    if set(coordinate) != _COORDINATE_FIELDS:
        raise ValueError("STOP visual coordinate identity fields are not exact")
    if coordinate.get("vjepa_source_commit") != OFFICIAL_SOURCE_COMMIT:
        raise ValueError("STOP visual coordinate source commit is invalid")
    if coordinate.get("vjepa_source_tree") != OFFICIAL_SOURCE_TREE:
        raise ValueError("STOP visual coordinate source tree is invalid")
    if coordinate.get("preprocess_sha256") != VJEPA_PREPROCESS_IDENTITY_SHA256:
        raise ValueError("STOP visual coordinate preprocess identity is invalid")
    if coordinate.get("pool_sha256") != VJEPA_POOL_IDENTITY_SHA256:
        raise ValueError("STOP visual coordinate pool identity is invalid")
    if coordinate.get("checkpoint_sha256") != verified["vjepa_checkpoint"].get(
        "sha256"
    ):
        raise ValueError("STOP visual coordinate checkpoint identity drifted")
    if coordinate.get("whitening_sha256") != verified["whitening"].get("sha256"):
        raise ValueError("STOP visual coordinate whitening identity drifted")
    for field in _COORDINATE_FIELDS - {"vjepa_source_commit", "vjepa_source_tree"}:
        _receipt_sha(coordinate.get(field), name=f"visual coordinate {field}")
    if _GIT_IDENTITY.fullmatch(str(coordinate["vjepa_source_commit"])) is None or (
        _GIT_IDENTITY.fullmatch(str(coordinate["vjepa_source_tree"])) is None
    ):
        raise ValueError("STOP visual coordinate source identity is malformed")
    numpy_version = value.get("numpy_version")
    if not isinstance(numpy_version, str) or not numpy_version:
        raise ValueError("STOP provenance NumPy version is missing")
    if expected_numpy_version is not None and numpy_version != expected_numpy_version:
        raise ValueError("STOP provenance NumPy version differs from the producer")
    bootstrap = _validate_bootstrap_parameters(value.get("bootstrap"))
    if expected_bootstrap is not None and bootstrap != dict(expected_bootstrap):
        raise ValueError("STOP provenance bootstrap parameters differ from the run")
    return value


def _scientific_fields_equal(observed: object, expected: object) -> bool:
    try:
        return canonical_json_bytes({"value": observed}) == canonical_json_bytes(
            {"value": expected}
        )
    except (TypeError, ValueError):
        return False


def validate_stop_calibration_receipt(
    receipt: Mapping[str, object],
) -> Mapping[str, object]:
    """Rebuild source population and statistics before admitting FORMAL STOP."""

    root = _receipt_mapping(receipt, name="root")
    if root.get("schema") == "J2J_RAW_STOP_CALIBRATION_V1":
        from j2j_recurrent_experiments.closed_loop.raw_stop import validate_raw_stop_calibration_receipt
        return validate_raw_stop_calibration_receipt(root)
    required = {
        "schema", "status", "interpretation", "threshold", "threshold_rule",
        "ledger", "census", "balanced_error_rate", "balanced_accuracy", "roc",
        "bootstrap", "calibration_provenance", "provenance",
    }
    if set(root) != required:
        raise ValueError("STOP receipt fields are incomplete or noncanonical")
    if root.get("schema") != "STOP_CALIBRATION_V1" or root.get("status") != "FORMAL":
        raise ValueError("STOP receipt schema/status is not formal")
    if root.get("interpretation") != "released-data location-match transfer proxy":
        raise ValueError("STOP receipt interpretation is not the frozen proxy claim")

    ledger = _receipt_mapping(root["ledger"], name="ledger")
    ledger_path, _ledger_bytes = _verified_receipt_file(ledger, name="ledger")
    provenance_identity = _receipt_mapping(
        root["calibration_provenance"], name="calibration_provenance"
    )
    _provenance_path, provenance_bytes = _verified_receipt_file(
        provenance_identity, name="calibration provenance"
    )
    provenance = _receipt_mapping(root["provenance"], name="provenance")
    try:
        parsed_provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("STOP calibration provenance bytes are not valid JSON") from exc
    if parsed_provenance != dict(provenance):
        raise ValueError("STOP embedded provenance differs from companion bytes")
    if provenance_bytes != canonical_json_bytes(dict(provenance)):
        raise ValueError("STOP calibration provenance is not canonical JSON bytes")
    validate_stop_calibration_provenance(
        provenance,
        ledger_identity=ledger,
        expected_numpy_version=np.__version__,
    )
    artifacts = _receipt_mapping(
        provenance["artifacts"], name="provenance.artifacts"
    )
    rows = load_stop_pair_ledger(ledger_path)
    rows, census = recompute_cache_bound_stop_rows(rows, provenance=provenance)
    derived = derive_stop_calibration(
        rows,
        census=census,
        bootstrap_parameters=_receipt_mapping(
            provenance["bootstrap"], name="provenance.bootstrap"
        ),
    )
    for field, expected in derived.items():
        if not _scientific_fields_equal(root.get(field), expected):
            raise ValueError(f"STOP receipt {field} differs from recomputed value")
    return root


__all__ = [
    "canonical_stop_distance",
    "derive_stop_calibration",
    "load_stop_pair_ledger",
    "recompute_cache_bound_stop_rows",
    "select_stop_threshold",
    "validate_stop_calibration_receipt",
    "validate_stop_calibration_provenance",
    "validate_stop_pair_ledger",
    "zero_intent_stop_action",
]
