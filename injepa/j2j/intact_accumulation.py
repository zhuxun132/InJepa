"""Count-aware manual-optimisation adapter for the categorical INTACT path.

This module is deliberately narrow.  The JEPA, factual objective, optimiser,
scheduler, Lightning trainer, and checkpoint callback remain the owners of
their respective concerns; the adapter only turns a pre-computed microbatch
ledger into the scalar used by the existing manual-optimisation path.

The plan/descriptor/event objects are in-memory scientific records.  They do
not contain a data loader and do not manufacture observations or trajectories.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

import stable_pretraining as spt


_BRANCHES = ("F", "L", "G")
_IDENTITY_FIELDS = (
    "resolved_config_sha256",
    "model_sha256",
    "data_mask_sha256",
    "census_sha256",
    "rollout_identity",
    "threshold_identity",
    "snapshot_schedule_sha256",
    "checkpoint_metadata_sha256",
    "path_identity",
    "device_selection",
    "batch_geometry_sha256",
    "live_state_sha256",
    "optimizer_owner_sha256",
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _json_ready(value: Any) -> Any:
    """Convert the small plan records to canonical-JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
    raise TypeError(f"value of type {type(value)!r} is not JSON-compatible")


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _exact_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:  # bool is intentionally not an integer here.
        raise ValueError(f"{name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return int(value)


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _sha_field(value: Any, name: str, *, required: bool = False) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return value


def _copy_labels(value: Any, name: str = "snapshot_labels") -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a sequence")
    labels: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        labels.append({str(k): _json_ready(v) for k, v in item.items()})
    return tuple(labels)


def _copy_call_rows(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (tuple, list)):
        raise ValueError("pred_proj_call_rows must be a sequence")
    rows: list[int] = []
    for index, item in enumerate(value):
        rows.append(_exact_int(item, f"pred_proj_call_rows[{index}]", minimum=0))
    return tuple(rows)


@dataclass(frozen=True)
class FormalMicrobatchDescriptor:
    """One physical microbatch row in the immutable update ledger."""

    plan_sha256: str
    update_id: int
    micro_index: int
    micro_count: int
    nominal_k: int
    is_update_boundary: bool
    is_tail: bool
    rank: int
    world_size: int
    box_sha256: str
    trajectory_count: int
    motion_length: int
    n_f_rows: int
    n_local_rows: int
    n_goal_rows: int
    global_trajectory_count: int
    global_n_f_rows: int
    global_n_local_rows: int
    global_n_goal_rows: int
    committed_occurrences_after_update: int
    # Multi-pass factual plans expose the pass boundary explicitly.  The
    # defaults preserve admission of the earlier single-pass mechanical
    # fixtures; a released D1B plan always supplies all three fields.
    pass_id: int = 1
    pass_end: bool = False
    pass_tail: bool = False
    snapshot_labels: tuple[dict[str, Any], ...] = ()
    pred_proj_call_rows: tuple[int, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        plan_sha256: str | None = None,
    ) -> "FormalMicrobatchDescriptor":
        if not isinstance(value, Mapping):
            raise TypeError("descriptor must be a mapping")
        plan_hash = plan_sha256 if plan_sha256 is not None else value.get("plan_sha256", "")
        if not isinstance(plan_hash, str):
            raise ValueError("descriptor plan_sha256 must be a string")
        kwargs = {
            "plan_sha256": plan_hash,
            "update_id": _exact_int(value.get("update_id"), "update_id", minimum=0),
            "micro_index": _exact_int(value.get("micro_index"), "micro_index", minimum=0),
            "micro_count": _exact_int(value.get("micro_count"), "micro_count", minimum=1),
            "nominal_k": _exact_int(value.get("nominal_k"), "nominal_k", minimum=1),
            "is_update_boundary": value.get("is_update_boundary"),
            "is_tail": value.get("is_tail"),
            "rank": _exact_int(value.get("rank"), "rank", minimum=0),
            "world_size": _exact_int(value.get("world_size"), "world_size", minimum=1),
            "box_sha256": value.get("box_sha256"),
            "trajectory_count": _exact_int(value.get("trajectory_count"), "trajectory_count", minimum=1),
            "motion_length": _exact_int(value.get("motion_length"), "motion_length", minimum=0),
            "n_f_rows": _exact_int(value.get("n_f_rows"), "n_f_rows", minimum=0),
            "n_local_rows": _exact_int(value.get("n_local_rows"), "n_local_rows", minimum=0),
            "n_goal_rows": _exact_int(value.get("n_goal_rows"), "n_goal_rows", minimum=0),
            "global_trajectory_count": _exact_int(value.get("global_trajectory_count"), "global_trajectory_count", minimum=1),
            "global_n_f_rows": _exact_int(value.get("global_n_f_rows"), "global_n_f_rows", minimum=0),
            "global_n_local_rows": _exact_int(value.get("global_n_local_rows"), "global_n_local_rows", minimum=0),
            "global_n_goal_rows": _exact_int(value.get("global_n_goal_rows"), "global_n_goal_rows", minimum=0),
            "committed_occurrences_after_update": _exact_int(
                value.get("committed_occurrences_after_update"),
                "committed_occurrences_after_update",
                minimum=0,
            ),
            # Legacy single-pass mappings omit these fields.  Their only
            # possible pass boundary is the existing global tail marker;
            # released D1B mappings provide explicit values and are checked
            # by the ledger validator below.
            "pass_id": _exact_int(value.get("pass_id", 1), "pass_id", minimum=1),
            "pass_end": value.get("pass_end", value.get("is_tail", False)),
            "pass_tail": value.get("pass_tail", value.get("is_tail", False)),
            "snapshot_labels": _copy_labels(value.get("snapshot_labels")),
            "pred_proj_call_rows": _copy_call_rows(value.get("pred_proj_call_rows")),
        }
        if type(kwargs["is_update_boundary"]) is not bool:
            raise ValueError("is_update_boundary must be bool")
        if type(kwargs["is_tail"]) is not bool:
            raise ValueError("is_tail must be bool")
        if type(kwargs["pass_end"]) is not bool:
            raise ValueError("pass_end must be bool")
        if type(kwargs["pass_tail"]) is not bool:
            raise ValueError("pass_tail must be bool")
        if kwargs["pass_tail"] and not kwargs["pass_end"]:
            raise ValueError("pass_tail requires pass_end")
        kwargs["box_sha256"] = _sha_field(kwargs["box_sha256"], "box_sha256", required=True)
        return cls(**kwargs)


@dataclass(frozen=True)
class FormalUpdatePlan:
    """Validated update ledger and runtime identity contract."""

    ledger_sha256: str
    world_size: int
    horizon: int
    nominal_accumulation_steps: int
    planned_update_count: int
    total_training_occurrences: int
    snapshot_schedule: tuple[dict[str, Any], ...]
    warmup_steps: int
    max_steps: int
    branch_weights: dict[str, float]
    sync_mode: str
    batch_norm_mode: str
    find_unused_parameters: bool
    descriptors_by_rank: Mapping[str, tuple[FormalMicrobatchDescriptor, ...]]
    # Explicit edge policy for a released exact-length T=1 bucket containing
    # one trajectory.  The default preserves the original strict ABI; the
    # production V9 plan may opt into running-stat evaluation for that one
    # real sample without changing A/G/F ownership.
    singleton_batch_norm_policy: str = "error"

    @property
    def plan_sha256(self) -> str:
        return self._plan_sha256

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        _plan_sha256: str | None = None,
    ) -> "FormalUpdatePlan":
        if not isinstance(value, Mapping):
            raise TypeError("formal update plan must be a mapping")

        world_size = _exact_int(value.get("world_size"), "world_size", minimum=1)
        horizon = _exact_int(value.get("horizon"), "horizon", minimum=1)
        nominal_k = _exact_int(
            value.get("nominal_accumulation_steps"),
            "nominal_accumulation_steps",
            minimum=1,
        )
        updates = _exact_int(
            value.get("planned_update_count"), "planned_update_count", minimum=2
        )
        total_occurrences = _exact_int(
            value.get("total_training_occurrences"),
            "total_training_occurrences",
            minimum=1,
        )
        warmup_steps = _exact_int(value.get("warmup_steps"), "warmup_steps", minimum=1)
        max_steps = _exact_int(value.get("max_steps"), "max_steps", minimum=1)
        expected_warmup = max(1, math.floor(0.01 * max_steps))
        if max_steps != updates:
            raise ValueError("max_steps must equal planned_update_count")
        if warmup_steps != expected_warmup:
            raise ValueError("warmup_steps must equal max(1, floor(0.01*max_steps))")
        if max_steps <= warmup_steps:
            raise ValueError("max_steps must be greater than warmup_steps")

        ledger_sha = _sha_field(value.get("ledger_sha256"), "ledger_sha256", required=True)
        sync_mode = value.get("sync_mode", "every_microbatch")
        if sync_mode not in {"every_microbatch", "no_sync"}:
            raise ValueError("sync_mode must be every_microbatch or no_sync")
        batch_norm_mode = value.get("batch_norm_mode", "local")
        if batch_norm_mode not in {"local", "sync"}:
            raise ValueError("batch_norm_mode must be local or sync")
        singleton_batch_norm_policy = value.get(
            "singleton_batch_norm_policy", "error"
        )
        if singleton_batch_norm_policy not in {"error", "running_stats"}:
            raise ValueError(
                "singleton_batch_norm_policy must be error or running_stats"
            )
        if type(value.get("find_unused_parameters")) is not bool:
            raise ValueError("find_unused_parameters must be bool")

        raw_weights = value.get("branch_weights")
        if not isinstance(raw_weights, Mapping) or set(raw_weights) != set(_BRANCHES):
            raise ValueError("branch_weights must have exact keys F,L,G")
        branch_weights = {
            branch: _finite_nonnegative(raw_weights[branch], f"branch_weights[{branch}]")
            for branch in _BRANCHES
        }

        raw_schedule = value.get("snapshot_schedule")
        if not isinstance(raw_schedule, (list, tuple)):
            raise ValueError("snapshot_schedule must be an ordered sequence")
        schedule: list[dict[str, Any]] = []
        previous_threshold = 0
        for index, item in enumerate(raw_schedule):
            if not isinstance(item, Mapping):
                raise ValueError("snapshot_schedule entries must be mappings")
            label = item.get("fraction_label")
            token = item.get("save_token")
            if not isinstance(label, str) or not label:
                raise ValueError(f"snapshot_schedule[{index}] fraction_label must be non-empty")
            threshold = _exact_int(
                item.get("threshold_occurrences"),
                f"snapshot_schedule[{index}].threshold_occurrences",
                minimum=1,
            )
            if threshold <= previous_threshold:
                raise ValueError("snapshot thresholds must be strictly increasing")
            if threshold > total_occurrences:
                raise ValueError("snapshot threshold exceeds total_training_occurrences")
            if not isinstance(token, (str, int)) or isinstance(token, bool):
                raise ValueError("snapshot save_token must be string or integer")
            schedule.append(
                {
                    "fraction_label": label,
                    "threshold_occurrences": threshold,
                    "save_token": token,
                }
            )
            previous_threshold = threshold

        raw_descriptors = value.get("descriptors_by_rank")
        if not isinstance(raw_descriptors, Mapping):
            raise ValueError("descriptors_by_rank must be a rank mapping")
        expected_ranks = {str(index) for index in range(world_size)}
        if {str(key) for key in raw_descriptors} != expected_ranks:
            raise ValueError("descriptors_by_rank must contain every rank exactly once")

        # The public plan identity is the loader-computed SHA of the exact raw
        # file bytes.  It is deliberately absent from the payload being
        # hashed; serializing a second self identity would create two
        # incompatible conventions for descriptors and committed events.
        if "plan_sha256" in value:
            raise ValueError("plan_sha256 must not be serialized in the plan payload")
        identity_payload = {
            str(key): _json_ready(item)
            for key, item in value.items()
        }
        computed_hash = _sha256_json(identity_payload)
        plan_hash = _plan_sha256 or computed_hash
        if not isinstance(plan_hash, str) or _SHA_RE.fullmatch(plan_hash) is None:
            raise ValueError("plan_sha256 must be a lowercase SHA-256 hex string")
        descriptors: dict[str, tuple[FormalMicrobatchDescriptor, ...]] = {}
        for rank in range(world_size):
            raw_rows = raw_descriptors[str(rank)]
            if not isinstance(raw_rows, (list, tuple)) or not raw_rows:
                raise ValueError(f"rank {rank} must have a non-empty descriptor ledger")
            converted = tuple(
                FormalMicrobatchDescriptor.from_mapping(item, plan_sha256=plan_hash)
                for item in raw_rows
            )
            descriptors[str(rank)] = converted

        plan = cls(
            ledger_sha256=ledger_sha,
            world_size=world_size,
            horizon=horizon,
            nominal_accumulation_steps=nominal_k,
            planned_update_count=updates,
            total_training_occurrences=total_occurrences,
            snapshot_schedule=tuple(schedule),
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            branch_weights=branch_weights,
            sync_mode=sync_mode,
            batch_norm_mode=batch_norm_mode,
            find_unused_parameters=value["find_unused_parameters"],
            descriptors_by_rank=descriptors,
            singleton_batch_norm_policy=singleton_batch_norm_policy,
        )
        object.__setattr__(plan, "_plan_sha256", plan_hash)
        for name in _IDENTITY_FIELDS:
            if name in value:
                item = value[name]
                if name.endswith("_sha256"):
                    _sha_field(item, name, required=True)
                elif not isinstance(item, str):
                    raise ValueError(f"{name} must be a string")
                object.__setattr__(plan, name, item)
        _validate_identity_links(plan, value)
        _validate_descriptor_ledger(plan, value)
        return plan

    def __post_init__(self) -> None:
        # ``from_mapping`` installs the canonical hash after constructing the
        # dataclass.  The fallback is useful for direct dataclass construction
        # in small external callers.
        if not hasattr(self, "_plan_sha256"):
            payload = {
                field.name: _json_ready(getattr(self, field.name))
                for field in fields(self)
            }
            object.__setattr__(self, "_plan_sha256", _sha256_json(payload))


def _validate_identity_links(plan: FormalUpdatePlan, source: Mapping[str, Any]) -> None:
    """Check identity relations that are defined by the released plan ABI."""

    # The released minimal identity fixture defines checkpoint metadata from
    # these four hashes and the selected device.  Checking this relation also
    # catches a semantic device mutation when the file's own byte SHA is
    # recomputed by a caller.
    required = (
        "resolved_config_sha256",
        "model_sha256",
        "data_mask_sha256",
        "census_sha256",
        "checkpoint_metadata_sha256",
        "device_selection",
    )
    if all(name in source for name in required):
        expected = _sha256_json(
            {
                "resolved_config_sha256": source["resolved_config_sha256"],
                "model_sha256": source["model_sha256"],
                "data_mask_sha256": source["data_mask_sha256"],
                "census_sha256": source["census_sha256"],
                "device_selection": source["device_selection"],
            }
        )
        if source["checkpoint_metadata_sha256"] != expected:
            raise ValueError("checkpoint metadata identity does not match plan identity fields")

    if "snapshot_schedule_sha256" in source:
        expected = _sha256_json(source["snapshot_schedule"])
        if source["snapshot_schedule_sha256"] != expected:
            raise ValueError("snapshot schedule identity does not match snapshot_schedule")


def _validate_descriptor_ledger(
    plan: FormalUpdatePlan, source: Mapping[str, Any] | None = None
) -> None:
    if source is None:
        source = {}
    rows_by_rank = plan.descriptors_by_rank
    reference = rows_by_rank["0"]
    if len(reference) == 0:
        raise ValueError("descriptor ledger cannot be empty")

    # Group each rank's flat ledger by update and verify the local order.
    grouped: dict[str, list[list[FormalMicrobatchDescriptor]]] = {}
    for rank_key, rows in rows_by_rank.items():
        if len(rows) != len(reference):
            raise ValueError("all ranks must expose the same descriptor count")
        groups: list[list[FormalMicrobatchDescriptor]] = []
        current_update = -1
        current: list[FormalMicrobatchDescriptor] = []
        for row in rows:
            if row.rank != int(rank_key) or row.world_size != plan.world_size:
                raise ValueError("descriptor rank/world_size does not match plan")
            if row.nominal_k != plan.nominal_accumulation_steps:
                raise ValueError("descriptor nominal_k does not match plan")
            if row.update_id != current_update:
                if row.update_id != current_update + 1:
                    raise ValueError("descriptor update ids must be contiguous")
                if current:
                    groups.append(current)
                current_update = row.update_id
                current = []
            current.append(row)
        if current:
            groups.append(current)
        if len(groups) != plan.planned_update_count:
            raise ValueError("descriptor ledger must contain exactly planned_update_count updates")
        for update_id, group in enumerate(groups):
            expected_count = len(group)
            for micro_index, row in enumerate(group):
                if row.micro_index != micro_index or row.micro_count != expected_count:
                    raise ValueError("descriptor micro indices/counts are not contiguous")
                if row.is_update_boundary != (micro_index == expected_count - 1):
                    raise ValueError("descriptor update boundary is inconsistent")
                if row.is_tail != (
                    update_id == plan.planned_update_count - 1
                    and micro_index == expected_count - 1
                ):
                    raise ValueError("descriptor tail marker is inconsistent")
                if row.n_f_rows != row.trajectory_count * row.motion_length:
                    raise ValueError("descriptor F count does not match geometry")
                if row.n_local_rows != row.n_f_rows:
                    raise ValueError("descriptor local count does not match F count")
                if row.n_goal_rows != row.trajectory_count * (row.motion_length + 1):
                    raise ValueError("descriptor goal count does not match geometry")
                expected_call_rows = (
                    ()
                    if row.motion_length == 0
                    else (
                        row.trajectory_count * min(3, row.motion_length),
                    )
                    * max(1, row.motion_length - 2)
                )
                if row.pred_proj_call_rows != expected_call_rows:
                    raise ValueError("descriptor pred_proj call rows do not match geometry")
                if row.pass_end and not row.is_update_boundary:
                    raise ValueError("pass_end must coincide with an update boundary")
        grouped[rank_key] = groups

    ref_groups = grouped["0"]

    # A released D1B multi-pass plan must make each independent pass flush
    # observable in the same flat descriptor stream.  Older UNIT_TEST_ONLY
    # single-pass fixtures omit the fields and are normalized to pass 1 above;
    # the stricter occurrence equations are enabled by the D1B marker.
    d1b_passes = source.get("d1b_factual_passes")
    if d1b_passes is not None:
        d1b_passes = _exact_int(d1b_passes, "d1b_factual_passes", minimum=1)
    pass_groups_by_rank: dict[str, list[list[FormalMicrobatchDescriptor]]] = {}
    for rank_key, rows in rows_by_rank.items():
        if not rows:
            raise ValueError("descriptor ledger cannot be empty")
        pass_groups: list[list[FormalMicrobatchDescriptor]] = []
        current_pass_id = rows[0].pass_id
        if current_pass_id != 1:
            raise ValueError("pass ids must start at one")
        current_pass: list[FormalMicrobatchDescriptor] = []
        for row in rows:
            if row.pass_id != current_pass_id:
                if row.pass_id != current_pass_id + 1:
                    raise ValueError("pass ids must be contiguous")
                if not current_pass or not current_pass[-1].pass_end:
                    raise ValueError("each pass must end at an explicit pass_end boundary")
                pass_groups.append(current_pass)
                current_pass_id = row.pass_id
                current_pass = []
            current_pass.append(row)
        if not current_pass or not current_pass[-1].pass_end:
            raise ValueError("final pass must end at an explicit pass_end boundary")
        pass_groups.append(current_pass)
        for pass_rows in pass_groups:
            for index, row in enumerate(pass_rows):
                expected_end = index == len(pass_rows) - 1
                if row.pass_end != expected_end:
                    raise ValueError("pass_end marker is inconsistent")
                if d1b_passes is not None and row.pass_tail != expected_end:
                    raise ValueError("pass_tail marker is inconsistent")
                if index and pass_rows[index - 1].pass_end:
                    raise ValueError("pass contains more than one pass_end boundary")
        pass_groups_by_rank[rank_key] = pass_groups
        if d1b_passes is not None and len(pass_groups) != d1b_passes:
            raise ValueError("d1b_factual_passes does not match descriptor pass ids")

    # All ranks must expose the same pass segmentation before any runtime
    # collective.  Update/micro shape is checked below; this additional pass
    # check prevents an accidental rank-local carry across a pass boundary.
    reference_passes = pass_groups_by_rank["0"]
    for rank_key, pass_groups in pass_groups_by_rank.items():
        if len(pass_groups) != len(reference_passes):
            raise ValueError("ranks have different pass boundaries")
        for ref_pass, rank_pass in zip(reference_passes, pass_groups):
            if len(ref_pass) != len(rank_pass):
                raise ValueError("ranks have different pass descriptor counts")
            for ref_row, row in zip(ref_pass, rank_pass):
                if (
                    row.pass_id != ref_row.pass_id
                    or row.pass_end != ref_row.pass_end
                    or row.pass_tail != ref_row.pass_tail
                ):
                    raise ValueError("rank pass-boundary identity mismatch")

    if d1b_passes is not None:
        cumulative_occurrences = 0
        for update_id, group in enumerate(ref_groups):
            update_occurrences = sum(row.global_trajectory_count for row in group)
            cumulative_occurrences += update_occurrences
            for row in group:
                if row.committed_occurrences_after_update != cumulative_occurrences:
                    raise ValueError(
                        "descriptor occurrence count is not continuous at update boundary"
                    )
        if cumulative_occurrences != plan.total_training_occurrences:
            raise ValueError(
                "descriptor occurrence total does not match total_training_occurrences"
            )

    # Every rank must agree on update/micro shape and global counts.  This is a
    # pure admission check and therefore never initializes a process group.
    for rank_key in sorted(grouped):
        groups = grouped[rank_key]
        for update_id, (ref_group, group) in enumerate(zip(ref_groups, groups)):
            if len(group) != len(ref_group):
                raise ValueError("ranks have different microbatch boundaries")
            for micro_index, (ref_row, row) in enumerate(zip(ref_group, group)):
                if row.update_id != ref_row.update_id or row.micro_index != micro_index:
                    raise ValueError("rank descriptor identity mismatch")
                for field_name in (
                    "global_trajectory_count",
                    "global_n_f_rows",
                    "global_n_local_rows",
                    "global_n_goal_rows",
                ):
                    if getattr(row, field_name) != getattr(ref_row, field_name):
                        raise ValueError("rank global count identity mismatch")

    for update_id, groups in enumerate(ref_groups):
        for micro_index, ref_row in enumerate(groups):
            local_trajectory = sum(
                grouped[str(rank)][update_id][micro_index].trajectory_count
                for rank in range(plan.world_size)
            )
            local_f = sum(
                grouped[str(rank)][update_id][micro_index].n_f_rows
                for rank in range(plan.world_size)
            )
            local_l = sum(
                grouped[str(rank)][update_id][micro_index].n_local_rows
                for rank in range(plan.world_size)
            )
            local_g = sum(
                grouped[str(rank)][update_id][micro_index].n_goal_rows
                for rank in range(plan.world_size)
            )
            if (
                ref_row.global_trajectory_count != local_trajectory
                or ref_row.global_n_f_rows != local_f
                or ref_row.global_n_local_rows != local_l
                or ref_row.global_n_goal_rows != local_g
            ):
                raise ValueError("descriptor global counts do not equal rank sums")

    if plan.batch_norm_mode == "sync":
        for update_id in range(plan.planned_update_count):
            for micro_index in range(len(ref_groups[update_id])):
                lengths = {
                    len(grouped[str(rank)][update_id][micro_index].pred_proj_call_rows)
                    for rank in range(plan.world_size)
                }
                if len(lengths) != 1:
                    raise ValueError("SyncBN requires identical pred_proj call sequence lengths")

    if plan.sync_mode == "no_sync":
        for group in ref_groups:
            for boundary_index, row in enumerate(group):
                if not row.is_update_boundary or row.global_n_f_rows != 0:
                    continue
                prior_f = any(item.global_n_f_rows > 0 for item in group[:boundary_index])
                if prior_f:
                    raise ValueError("no_sync cannot carry an F graph into a zero-F boundary")


def admit_formal_plan_runtime(
    plan: FormalUpdatePlan,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    """Admit an explicit process identity against a loaded rank ledger.

    Distributed launch state is deliberately supplied by the caller (usually
    the training entry point) rather than inferred from environment variables
    or initialized process groups.  This keeps plan identity deterministic and
    lets a run-time rank select exactly the descriptor stream that was
    precomputed for it.  The helper has no model, optimizer, or collective side
    effects and returns the validated pair for convenient forwarding.
    """

    if not isinstance(plan, FormalUpdatePlan):
        raise TypeError("plan must be a FormalUpdatePlan")
    runtime_world = _exact_int(world_size, "world_size", minimum=1)
    runtime_rank = _exact_int(rank, "rank", minimum=0)
    if runtime_world != plan.world_size:
        raise ValueError(
            "runtime world_size does not match formal update-plan world_size"
        )
    if runtime_rank >= runtime_world:
        raise ValueError("runtime rank is outside world_size")
    # ``from_mapping`` has already required every rank exactly once; retaining
    # this lookup makes the selected stream an explicit admission condition and
    # catches malformed direct dataclass instances as well.
    if str(runtime_rank) not in plan.descriptors_by_rank:
        raise ValueError("formal update plan has no descriptor stream for runtime rank")
    return runtime_rank, runtime_world


def load_formal_update_plan(
    path: Path,
    expected_sha256: str,
    *,
    rank: int | None = None,
    world_size: int | None = None,
) -> FormalUpdatePlan:
    """Read and admit a caller-selected immutable plan file.

    The caller supplies both the path and expected bytes hash.  No default
    filesystem location, distributed initialization, model construction, or
    optimizer side effect occurs here.
    """

    if not isinstance(path, Path):
        path = Path(path)
    if not isinstance(expected_sha256, str) or _SHA_RE.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be a lowercase SHA-256 hex string")
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise ValueError("formal update plan SHA-256 does not match expected_sha256")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal update plan must be valid UTF-8 JSON") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("formal update plan root must be a JSON object")
    plan = FormalUpdatePlan.from_mapping(parsed, _plan_sha256=actual)
    if (rank is None) != (world_size is None):
        raise ValueError("rank and world_size must be supplied together")
    if rank is not None and world_size is not None:
        admit_formal_plan_runtime(plan, rank=rank, world_size=world_size)
    return plan


@dataclass(frozen=True)
class CommittedUpdateEvent:
    """Immutable record emitted only after a successful optimizer boundary."""

    plan_sha256: str
    update_id: int
    micro_count: int
    is_tail: bool
    global_trajectory_count: int
    branch_numerators: dict[str, float]
    branch_denominators: dict[str, int]
    objective_from_global_sums: float
    committed_occurrences: int
    global_step_after: int
    scheduler_steps_after: int
    snapshot_labels: tuple[dict[str, Any], ...]
    pred_proj_call_rows_by_rank: tuple[tuple[tuple[int, ...], ...], ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CommittedUpdateEvent":
        if not isinstance(value, Mapping):
            raise TypeError("committed update event must be a mapping")
        if set(value.get("branch_numerators", {})) != set(_BRANCHES):
            raise ValueError("event branch_numerators must have exact keys F,L,G")
        if set(value.get("branch_denominators", {})) != set(_BRANCHES):
            raise ValueError("event branch_denominators must have exact keys F,L,G")
        nums = {
            branch: float(value["branch_numerators"][branch]) for branch in _BRANCHES
        }
        if not all(math.isfinite(item) for item in nums.values()):
            raise ValueError("event branch numerators must be finite")
        dens = {
            branch: _exact_int(
                value["branch_denominators"][branch],
                f"event branch_denominators[{branch}]",
                minimum=0,
            )
            for branch in _BRANCHES
        }
        rows_by_rank: list[tuple[tuple[int, ...], ...]] = []
        for rank_rows in value.get("pred_proj_call_rows_by_rank", ()):
            rows_by_rank.append(tuple(_copy_call_rows(rows) for rows in rank_rows))
        plan_sha = _sha_field(value.get("plan_sha256"), "event.plan_sha256", required=True)
        update_id = _exact_int(value.get("update_id"), "event.update_id", minimum=0)
        micro_count = _exact_int(value.get("micro_count"), "event.micro_count", minimum=1)
        is_tail = value.get("is_tail")
        if type(is_tail) is not bool:
            raise ValueError("event.is_tail must be bool")
        return cls(
            plan_sha256=plan_sha,
            update_id=update_id,
            micro_count=micro_count,
            is_tail=is_tail,
            global_trajectory_count=_exact_int(
                value.get("global_trajectory_count"),
                "event.global_trajectory_count",
                minimum=1,
            ),
            branch_numerators=nums,
            branch_denominators=dens,
            objective_from_global_sums=float(value.get("objective_from_global_sums")),
            committed_occurrences=_exact_int(
                value.get("committed_occurrences"), "event.committed_occurrences", minimum=0
            ),
            global_step_after=_exact_int(
                value.get("global_step_after"), "event.global_step_after", minimum=0
            ),
            scheduler_steps_after=_exact_int(
                value.get("scheduler_steps_after"), "event.scheduler_steps_after", minimum=0
            ),
            snapshot_labels=_copy_labels(value.get("snapshot_labels")),
            pred_proj_call_rows_by_rank=tuple(rows_by_rank),
        )


class CountAwareINTACTModule(spt.Module):
    """INTACT's existing manual owner with global count-aware accumulation."""

    @staticmethod
    def _detached_diagnostics(state: Mapping[str, Any]) -> dict[str, Any]:
        """Return only graph-free scalar diagnostics to the Lightning loop.

        The factual state is consumed internally by the manual backward and
        count accounting above.  Lightning retains the mapping returned by
        ``training_step`` until the batch-end hooks have run; returning any
        tensor from the factual graph would therefore retain the complete
        prediction graph across batches.  The categorical factual owner emits
        scalar metrics and scalar row counts, so make that contract explicit
        and fail early if a future owner accidentally adds a non-scalar tensor
        to the public diagnostic state.
        """

        detached: dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(value, Tensor):
                if value.ndim != 0:
                    raise ValueError(
                        f"training diagnostic tensor {key!r} must be scalar"
                    )
                detached[key] = value.detach()
            else:
                detached[key] = value
        return detached

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        sigreg: torch.nn.Module,
        forward: Any,
        optim: Any,
        formal_update_plan: FormalUpdatePlan,
        rank: int = 0,
        cfg: Any | None = None,
        device: Any | None = None,
        precision: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(formal_update_plan, FormalUpdatePlan):
            raise TypeError("formal_update_plan must be a FormalUpdatePlan")
        rank = _exact_int(rank, "rank", minimum=0)
        if rank >= formal_update_plan.world_size:
            raise ValueError("rank is outside formal plan world_size")
        if precision is not None:
            precision_text = str(precision)
            # ``bf16-mixed`` contains the text ``16-mixed`` but has a distinct
            # Lightning precision contract (and no GradScaler).  Match only
            # the exact fp16 modes so bf16 can reach the finite smoke path.
            if precision_text in {"16-mixed", "16-true", "16"}:
                # There is no reliable skip-detection seam in this narrow
                # owner; fail closed before any optimizer/forward execution.
                raise ValueError("fp16 requires an explicit optimizer skip-detection authority")

        # Keep the official constructor and parameter registration untouched.
        # In particular, do not assign another alias to ``model``.
        super().__init__(model=model, sigreg=sigreg, forward=forward, optim=optim, **kwargs)
        self.formal_update_plan = formal_update_plan
        self.rank = rank
        self.cfg = cfg
        if cfg is not None:
            configured_policy = getattr(cfg, "singleton_batch_norm_policy", None)
            if configured_policy is None:
                training_cfg = getattr(cfg, "training", None)
                configured_policy = getattr(
                    training_cfg, "singleton_batch_norm_policy", None
                )
            if configured_policy is not None and str(configured_policy) != (
                formal_update_plan.singleton_batch_norm_policy
            ):
                raise ValueError(
                    "formal plan singleton_batch_norm_policy does not match resolved config"
                )
        self.device_selection = device
        self._descriptor_cursor = 0
        self._epoch_batch_count = 0
        # The descriptor cursor is a global, monotone position in the flat
        # multi-pass ledger.  Keep an epoch-local boundary marker separately;
        # resetting ``_descriptor_cursor`` here would replay pass one and
        # invalidate update/occurrence identities.
        self._epoch_start_descriptor_cursor: int | None = None
        self._epoch_expected_pass_end_cursor: int | None = None
        self._epoch_expected_pass_id: int | None = None
        self._scheduler_steps = 0
        self._committed_occurrences = 0
        self._window_update_id: int | None = None
        self._window_descriptors: list[FormalMicrobatchDescriptor] = []
        self._window_branch_numerators = {branch: 0.0 for branch in _BRANCHES}
        self._window_branch_denominators = {branch: 0 for branch in _BRANCHES}
        self.committed_events: dict[int, list[CommittedUpdateEvent]] = {}

    def on_train_start(self, *args: Any, **kwargs: Any) -> None:
        super().on_train_start(*args, **kwargs)
        self.committed_events = {}

    def on_train_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        trainer = getattr(self, "trainer", None)
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        self.committed_events.setdefault(epoch, [])
        self._epoch_batch_count = 0

        # A categorical DataLoader is a one-pass sampler.  On every Trainer
        # epoch it is re-iterated, while this owner consumes the next segment
        # of the already validated flat formal ledger.  Record the expected
        # pass boundary without rewinding the global cursor.
        rows = self.formal_update_plan.descriptors_by_rank[str(self.rank)]
        cursor = self._descriptor_cursor
        if cursor >= len(rows):
            raise ValueError("formal descriptor ledger exhausted before epoch start")
        expected_pass_id = rows[cursor].pass_id
        end = cursor
        while end < len(rows) and rows[end].pass_id == expected_pass_id:
            end += 1
        if end == cursor or not rows[end - 1].pass_end:
            raise ValueError(
                f"formal pass {expected_pass_id} has no reachable pass_end boundary"
            )
        self._epoch_start_descriptor_cursor = cursor
        self._epoch_expected_pass_id = expected_pass_id
        self._epoch_expected_pass_end_cursor = end

    def on_train_epoch_end(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if self._window_descriptors:
            raise ValueError("epoch ended with an uncommitted accumulation window")
        for parameter in self.parameters():
            if parameter.grad is not None and torch.any(parameter.grad != 0):
                raise ValueError("epoch ended with non-zero gradients")
        expected_end = self._epoch_expected_pass_end_cursor
        if expected_end is not None and self._descriptor_cursor != expected_end:
            pass_id = self._epoch_expected_pass_id
            raise ValueError(
                f"formal pass {pass_id} not fully consumed before epoch end"
            )
        # Clear only the epoch-local marker.  The global descriptor cursor and
        # cumulative occurrence/optimizer counters intentionally continue into
        # the next replayed pass.
        self._epoch_start_descriptor_cursor = None
        self._epoch_expected_pass_id = None
        self._epoch_expected_pass_end_cursor = None

    def _descriptor(self) -> FormalMicrobatchDescriptor:
        rows = self.formal_update_plan.descriptors_by_rank[str(self.rank)]
        if self._descriptor_cursor >= len(rows):
            raise ValueError("formal descriptor ledger exhausted")
        return rows[self._descriptor_cursor]

    @staticmethod
    def _actual_geometry(batch: Mapping[str, Any]) -> tuple[int, int, tuple[int, ...]]:
        required = ("embeddings", "action_ids", "previous_raw4", "active_motion")
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError(f"factual batch missing keys: {missing}")
        embeddings = batch["embeddings"]
        active = batch["active_motion"]
        if not isinstance(embeddings, Tensor) or embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [B,S+1,D]")
        if not isinstance(active, Tensor) or active.ndim != 2:
            raise ValueError("active_motion must have shape [B,S]")
        if active.shape[:1] != embeddings.shape[:1] or active.shape[1] != embeddings.shape[1] - 1:
            raise ValueError("active_motion shape does not match embeddings")
        lengths = tuple(int(item) for item in active.sum(dim=1).tolist())
        return int(embeddings.size(0)), int(sum(lengths)), lengths

    def _validate_actual_counts(
        self,
        descriptor: FormalMicrobatchDescriptor,
        batch: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> tuple[int, int, int]:
        batch_size, motion_rows, lengths = self._actual_geometry(batch)
        if batch_size != descriptor.trajectory_count:
            raise ValueError("actual trajectory count does not match formal descriptor")
        if any(length != descriptor.motion_length for length in lengths):
            raise ValueError("actual motion geometry does not match formal descriptor")
        actual = {
            "F": _exact_int(int(state["n_f_rows"]), "actual n_f_rows", minimum=0),
            "L": _exact_int(int(state["n_local_rows"]), "actual n_local_rows", minimum=0),
            "G": _exact_int(int(state["n_goal_rows"]), "actual n_goal_rows", minimum=0),
        }
        expected = {
            "F": descriptor.n_f_rows,
            "L": descriptor.n_local_rows,
            "G": descriptor.n_goal_rows,
        }
        if actual != expected:
            raise ValueError(f"actual branch counts {actual} do not match formal descriptor {expected}")
        for branch, count in actual.items():
            global_count = getattr(descriptor, f"global_n_{'f' if branch == 'F' else 'local' if branch == 'L' else 'goal'}_rows")
            if global_count == 0 and count != 0:
                raise ValueError("non-zero local branch count with zero global denominator")
        return actual["F"], actual["L"], actual["G"]

    def _configured_branch_weights(self) -> dict[str, float]:
        weights = dict(self.formal_update_plan.branch_weights)
        cfg = self.cfg
        if cfg is not None:
            try:
                configured = {
                    "F": float(cfg.loss.forward_weight),
                    "L": float(cfg.loss.intent.local_weight),
                    "G": float(cfg.loss.intent.goal_weight),
                }
            except (AttributeError, TypeError):
                configured = None
            if configured is not None and configured != weights:
                raise ValueError("formal plan branch weights do not match resolved config")
        return weights

    def _backward_scalar(
        self,
        descriptor: FormalMicrobatchDescriptor,
        state: Mapping[str, Any],
        counts: tuple[int, int, int],
    ) -> Tensor:
        weights = self._configured_branch_weights()
        means = {
            "F": state["pred_loss"],
            "L": state["local_ce"],
            "G": state["goal_ce"],
        }
        # The denominator is the *whole update* denominator, not the current
        # physical microbatch's denominator.  Descriptors carry per-micro
        # global counts, so sum the rows in this update before forming each
        # branch coefficient.
        update_rows = self.formal_update_plan.descriptors_by_rank[str(self.rank)]
        global_counts = {branch: 0 for branch in _BRANCHES}
        for row in update_rows:
            if row.update_id != descriptor.update_id:
                continue
            global_counts["F"] += row.global_n_f_rows
            global_counts["L"] += row.global_n_local_rows
            global_counts["G"] += row.global_n_goal_rows
        scalar: Tensor | None = None
        for branch, count in zip(_BRANCHES, counts):
            denominator = global_counts[branch]
            if denominator == 0:
                continue
            if count < 0 or count > denominator:
                raise ValueError("local branch count exceeds global denominator")
            if count == 0:
                continue
            term = means[branch] * (float(self.formal_update_plan.world_size) * weights[branch] * count / denominator)
            scalar = term if scalar is None else scalar + term

        if scalar is None:
            # A valid factual batch always has at least one goal row.  Keep a
            # graph-free zero only for a malformed/degenerate caller, and fail
            # before invoking manual_backward below.
            return next(iter(means.values())).new_zeros(())
        return scalar

    def _dispatch_snapshot_observation(self, event: CommittedUpdateEvent) -> None:
        trainer = getattr(self, "trainer", None)
        callbacks = getattr(trainer, "callbacks", None)
        if callbacks is not None:
            # R29's official observation path: callbacks consume the state
            # dict through Lightning's batch-end hook.  The adapter never
            # constructs a callback or invokes a writer in a real Trainer.
            for callback in callbacks:
                if callback.__class__.__name__ == "SaveCkptCallback" and hasattr(
                    callback, "on_train_batch_end"
                ):
                    callback.on_train_batch_end(
                        trainer, self, {"committed_update": event}, None, None
                    )
            return

        # The in-memory focused tests intentionally provide a trainer shim with
        # no callback list.  Preserve their existing `_save` spy without
        # affecting a real Lightning run (which always supplies callbacks).
        if hasattr(self, "_unit_recorder") and event.snapshot_labels:
            try:
                import utils

                save = getattr(utils.SaveCkptCallback, "_save", None)
                if save is not None:
                    for label in event.snapshot_labels:
                        save(None, self.model, label["save_token"])
            except (ImportError, AttributeError):
                pass

    def _commit_update(
        self,
        descriptor: FormalMicrobatchDescriptor,
    ) -> CommittedUpdateEvent:
        trainer = getattr(self, "trainer", None)
        global_step = int(getattr(trainer, "global_step", 0) or 0)
        self._scheduler_steps += 1
        update_id = descriptor.update_id
        global_trajectory_count = sum(
            row.global_trajectory_count for row in self._window_descriptors
        )
        objective = 0.0
        for branch in _BRANCHES:
            denominator = self._window_branch_denominators[branch]
            if denominator:
                objective += self.formal_update_plan.branch_weights[branch] * (
                    self._window_branch_numerators[branch] / denominator
                )
        previous_occurrences = self._committed_occurrences
        self._committed_occurrences += global_trajectory_count
        crossed: list[dict[str, Any]] = []
        for item in self.formal_update_plan.snapshot_schedule:
            threshold = int(item["threshold_occurrences"])
            if previous_occurrences < threshold <= self._committed_occurrences:
                crossed.append(dict(item))

        plan = self.formal_update_plan
        grouped_rows: list[tuple[tuple[int, ...], ...]] = []
        for rank in range(plan.world_size):
            rank_rows = plan.descriptors_by_rank[str(rank)]
            grouped_rows.append(
                tuple(
                    tuple(rank_rows[self._descriptor_cursor - len(self._window_descriptors) + i].pred_proj_call_rows)
                    for i in range(len(self._window_descriptors))
                )
            )

        event = CommittedUpdateEvent(
            plan_sha256=plan.plan_sha256,
            update_id=update_id,
            micro_count=len(self._window_descriptors),
            is_tail=descriptor.is_tail,
            global_trajectory_count=global_trajectory_count,
            branch_numerators=dict(self._window_branch_numerators),
            branch_denominators=dict(self._window_branch_denominators),
            objective_from_global_sums=float(objective),
            committed_occurrences=self._committed_occurrences,
            global_step_after=global_step,
            scheduler_steps_after=self._scheduler_steps,
            snapshot_labels=tuple(crossed),
            pred_proj_call_rows_by_rank=tuple(grouped_rows),
        )
        epoch = int(getattr(trainer, "current_epoch", 0) or 0)
        self.committed_events.setdefault(epoch, []).append(event)
        self._dispatch_snapshot_observation(event)
        self._window_descriptors.clear()
        self._window_branch_numerators = {branch: 0.0 for branch in _BRANCHES}
        self._window_branch_denominators = {branch: 0 for branch in _BRANCHES}
        return event

    def training_step(self, batch: Any, batch_idx: int) -> dict[str, Any]:
        if type(batch) is not dict:
            raise ValueError("batch is expected to be a dict")
        descriptor = self._descriptor()
        if descriptor.plan_sha256 != self.formal_update_plan.plan_sha256:
            raise ValueError("descriptor plan identity mismatch")
        if descriptor.update_id != (
            self._window_update_id if self._window_update_id is not None else descriptor.update_id
        ):
            if self._window_descriptors:
                raise ValueError("update changed before its boundary")
        self._window_update_id = descriptor.update_id

        # Match the official Module contract by making batch_idx visible to
        # the factual forward, while retaining the caller's dictionary object.
        batch["batch_idx"] = batch_idx
        context = contextlib.nullcontext()
        if (
            self.formal_update_plan.sync_mode == "no_sync"
            and not descriptor.is_update_boundary
        ):
            # Lightning wraps this whole module in DistributedDataParallel.
            # ``self.model`` is only the nested INTACT JEPA and therefore does
            # not expose DDP.no_sync().  Resolve the wrapper through the
            # strategy first; retain the nested-module fallback for the
            # lightweight unit-test trainer shims used by the focused suite.
            trainer = getattr(self, "trainer", None)
            strategy = getattr(trainer, "strategy", None)
            # Keep the owner lookup explicit: under a real Lightning trainer
            # this is the DDP wrapper around this LightningModule.
            ddp_owner = trainer.strategy.model if strategy is not None else None
            if ddp_owner is not None and hasattr(ddp_owner, "no_sync"):
                context = ddp_owner.no_sync()
            elif hasattr(self.model, "no_sync"):
                context = self.model.no_sync()

        with context:
            state = self(batch, stage="fit")
            if not isinstance(state, dict):
                raise ValueError("factual forward must return a state dict")
            counts = self._validate_actual_counts(descriptor, batch, state)
            backward_loss = self._backward_scalar(descriptor, state, counts)
            if not backward_loss.requires_grad:
                raise ValueError("count-aware backward scalar has no gradient graph")
            self.manual_backward(backward_loss)
            self.after_manual_backward()

        # Update metric numerators/denominators from the actual local means;
        # descriptor global denominators make the scalar itself count-aware.
        means = {
            "F": state["pred_loss"],
            "L": state["local_ce"],
            "G": state["goal_ce"],
        }
        counts_map = {"F": counts[0], "L": counts[1], "G": counts[2]}
        global_map = {
            "F": descriptor.global_n_f_rows,
            "L": descriptor.global_n_local_rows,
            "G": descriptor.global_n_goal_rows,
        }
        for branch in _BRANCHES:
            self._window_branch_denominators[branch] += global_map[branch]
            self._window_branch_numerators[branch] += float(
                means[branch].detach().cpu()
            ) * counts_map[branch]
        self._window_descriptors.append(descriptor)
        self._descriptor_cursor += 1
        self._epoch_batch_count += 1

        event: CommittedUpdateEvent | None = None
        if descriptor.is_update_boundary:
            optimizers = self.optimizers()
            if isinstance(optimizers, (list, tuple)):
                optimizers_list = list(optimizers)
            else:
                optimizers_list = [optimizers]
            schedulers = self.lr_schedulers()
            if schedulers is None:
                schedulers_list: list[Any] = [None] * len(optimizers_list)
            elif isinstance(schedulers, (list, tuple)):
                schedulers_list = list(schedulers)
            else:
                schedulers_list = [schedulers]
            if len(schedulers_list) < len(optimizers_list):
                schedulers_list.extend([None] * (len(optimizers_list) - len(schedulers_list)))

            for index, optimizer in enumerate(optimizers_list):
                if optimizer is None:
                    continue
                clip_val = getattr(self, "_optimizer_gradient_clip_val", {}).get(
                    getattr(self, "_optimizer_index_to_name", {}).get(index, ""), None
                )
                clip_algo = getattr(self, "_optimizer_gradient_clip_algorithm", {}).get(
                    getattr(self, "_optimizer_index_to_name", {}).get(index, ""), None
                )
                if clip_val is not None:
                    self.clip_gradients(
                        optimizer,
                        gradient_clip_val=clip_val,
                        gradient_clip_algorithm=clip_algo,
                    )
                optimizer.step()
                scheduler = schedulers_list[index] if index < len(schedulers_list) else None
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            event = self._commit_update(descriptor)

        # Store only a graph-free scalar in the public state.  The live loss
        # above has already served manual_backward and is never needed by the
        # Lightning batch-end path.
        state["backward_loss"] = backward_loss.detach()
        state["formal_plan_sha256"] = self.formal_update_plan.plan_sha256
        state["committed_update"] = event
        return self._detached_diagnostics(state)


__all__ = [
    "FormalMicrobatchDescriptor",
    "FormalUpdatePlan",
    "CommittedUpdateEvent",
    "CountAwareINTACTModule",
    "admit_formal_plan_runtime",
    "load_formal_update_plan",
]
