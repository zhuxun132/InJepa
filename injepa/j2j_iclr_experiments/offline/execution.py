"""Rank-exclusive control plane for current Context4 offline evaluation.

This module owns launcher admission, payload-free population planning and
rank-shard validation.  It does not load checkpoints, cache tensors, or wrap
the evaluation model in DDP.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
from typing import Any

import torch
from torch.utils.data import DataLoader

from j2j.context4_data import TrajectoryDescriptor, build_rank_origin_plan
from j2j.receipts import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_LAUNCH_FIELDS = ("WORLD_SIZE", "RANK", "LOCAL_RANK")
_SHARD_SCHEMA = "J2J_CONTEXT4_OFFLINE_RANK_SHARD_V1"
_CONTROL_FIELDS = frozenset({"population_ordinal", "emission_ordinal"})


def _exact_positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive non-boolean integer")
    return value


def _exact_nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative non-boolean integer")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_integer(name: str) -> int:
    raw = os.environ[name]
    if not raw or not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"launcher {name} must be a non-negative integer")
    return int(raw)


@dataclass(frozen=True)
class ExecutionIdentity:
    """One already-admitted launcher rank and its requested device class."""

    world_size: int
    rank: int
    local_rank: int
    device: str
    _cuda_claim: object | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_environment(
        cls,
        configured_world_size: int,
        device: str,
    ) -> "ExecutionIdentity":
        world_size = _exact_positive_integer(
            configured_world_size, "configured world_size"
        )
        if device not in {"cpu", "cuda"}:
            raise ValueError("runtime device must be cpu or cuda")

        present = {name for name in _LAUNCH_FIELDS if name in os.environ}
        if present and present != set(_LAUNCH_FIELDS):
            missing = sorted(set(_LAUNCH_FIELDS) - present)
            raise ValueError(
                "launcher environment must supply WORLD_SIZE, RANK and LOCAL_RANK "
                f"together; missing {missing}"
            )
        if present:
            actual_world_size = _environment_integer("WORLD_SIZE")
            rank = _environment_integer("RANK")
            local_rank = _environment_integer("LOCAL_RANK")
        else:
            if world_size != 1:
                raise ValueError(
                    "configured distributed world_size requires a complete launcher "
                    "environment"
                )
            actual_world_size, rank, local_rank = 1, 0, 0

        if actual_world_size != world_size:
            raise ValueError(
                "configured world_size does not match launcher WORLD_SIZE"
            )
        if not 0 <= rank < actual_world_size:
            raise ValueError("launcher RANK is outside WORLD_SIZE")
        if local_rank < 0:
            raise ValueError("launcher LOCAL_RANK must be non-negative")

        claim: object | None = None
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("requested CUDA launcher device is unavailable")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError("launcher LOCAL_RANK has no visible CUDA device")
            torch.cuda.set_device(local_rank)
            # Keep the one-byte admission allocation alive with the identity so
            # checkpoint loading cannot silently drift to another current device.
            claim = torch.empty(
                (1,), dtype=torch.uint8, device=torch.device("cuda", local_rank)
            )
        return cls(
            world_size=actual_world_size,
            rank=rank,
            local_rank=local_rank,
            device=device,
            _cuda_claim=claim,
        )


@dataclass(frozen=True)
class _ProcessGroup:
    identity: ExecutionIdentity

    def barrier(self) -> None:
        if self.identity.world_size > 1:
            torch.distributed.barrier()


@contextmanager
def managed_process_group(
    identity: ExecutionIdentity,
    *,
    timeout_seconds: int,
) -> Iterator[_ProcessGroup]:
    """Own an evaluation-only process group for barriers/publication."""

    if type(identity) is not ExecutionIdentity:
        raise TypeError("identity must be an ExecutionIdentity")
    timeout = _exact_positive_integer(timeout_seconds, "distributed timeout")
    owns_initialization_attempt = False
    try:
        if identity.world_size > 1:
            if not torch.distributed.is_available():
                raise RuntimeError("torch distributed runtime is unavailable")
            if torch.distributed.is_initialized():
                raise RuntimeError(
                    "offline command refuses a process group it did not create"
                )
            backend = "nccl" if identity.device == "cuda" else "gloo"
            owns_initialization_attempt = True
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                rank=identity.rank,
                world_size=identity.world_size,
                timeout=timedelta(seconds=timeout),
            )
        yield _ProcessGroup(identity)
    finally:
        if owns_initialization_attempt and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _descriptor_value(raw: object, name: str) -> object:
    if isinstance(raw, Mapping):
        if name not in raw:
            raise ValueError(f"population descriptor is missing {name}")
        return raw[name]
    if not hasattr(raw, name):
        raise ValueError(f"population descriptor is missing {name}")
    return getattr(raw, name)


def _trajectory_key(value: object) -> str:
    if type(value) is bytes:
        if len(value) != 32:
            raise ValueError("trajectory_key bytes must have length 32")
        return value.hex()
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError("trajectory_key must be lowercase 64-hex")
    return value


def canonicalize_population_descriptors(
    descriptors: Sequence[object],
) -> tuple[dict[str, Any], ...]:
    """Project metadata-only descriptors without touching dataset payloads."""

    if isinstance(descriptors, (str, bytes)) or not isinstance(descriptors, Sequence):
        raise TypeError("population descriptors must be a sequence")
    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_keys: set[str] = set()
    for raw in descriptors:
        index = _exact_nonnegative_integer(
            _descriptor_value(raw, "dataset_index"), "dataset_index"
        )
        key = _trajectory_key(_descriptor_value(raw, "trajectory_key"))
        building = _descriptor_value(raw, "building")
        source_id = _descriptor_value(raw, "source_id")
        origin_count = _exact_nonnegative_integer(
            _descriptor_value(raw, "origin_count"), "origin_count"
        )
        if not isinstance(building, str) or not building:
            raise ValueError("population building must be a non-empty string")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("population source_id must be a non-empty string")
        if index in seen_indices:
            raise ValueError("population dataset indices must be unique")
        if key in seen_keys:
            raise ValueError("population trajectory keys must be unique")
        seen_indices.add(index)
        seen_keys.add(key)
        records.append(
            {
                "dataset_index": index,
                "trajectory_key": key,
                "building": building,
                "source_id": source_id,
                "origin_count": origin_count,
            }
        )
    if not records:
        raise ValueError("project-dev population must not be empty")
    return tuple(records)


def project_dataset_descriptors(dataset: object) -> tuple[dict[str, Any], ...]:
    """Read only held source metadata from a categorical dataset."""

    source_items = getattr(dataset, "_source_items", None)
    if type(source_items) is not tuple or len(source_items) != len(dataset):
        raise TypeError("categorical dataset does not expose its immutable source metadata")
    records: list[dict[str, Any]] = []
    for index, source_item in enumerate(source_items):
        if getattr(source_item, "projection_partition", None) != "project-dev":
            raise ValueError("offline population contains a non-project-dev trajectory")
        canonical = getattr(source_item, "canonical_trajectory", None)
        actions = getattr(canonical, "actions", None)
        if type(actions) is not tuple or not actions:
            raise ValueError("source metadata lacks the canonical factual action sequence")
        records.append(
            {
                "dataset_index": index,
                "trajectory_key": getattr(canonical, "canonical_trajectory_key", None),
                "building": getattr(canonical, "scan_id", None),
                "source_id": getattr(canonical, "source_id", None),
                "origin_count": len(actions) - 1,
            }
        )
    return canonicalize_population_descriptors(tuple(records))


@dataclass(frozen=True)
class PopulationPlan:
    partition: str
    source_manifest_sha256: str
    coverage: str
    status: str
    population_records: tuple[dict[str, Any], ...]
    selected_records: tuple[dict[str, Any], ...]
    owner_keys_by_rank: tuple[tuple[str, ...], ...]
    owner_indices_by_rank: tuple[tuple[int, ...], ...]
    population_sha256: str
    selection_sha256: str
    owner_plan_sha256: str

    @property
    def world_size(self) -> int:
        return len(self.owner_keys_by_rank)


def build_population_plan(
    descriptors: Sequence[object],
    *,
    coverage: str,
    budget: int | None,
    world_size: int,
    partition: str,
    source_manifest_sha256: str,
) -> PopulationPlan:
    """Select globally, then reuse the established whole-trajectory owner plan."""

    ranks = _exact_positive_integer(world_size, "world_size")
    if partition != "project-dev":
        raise ValueError("offline population partition must be project-dev")
    manifest_sha = _sha256(source_manifest_sha256, "source manifest SHA")
    records = canonicalize_population_descriptors(descriptors)
    if coverage == "full_project_dev":
        if budget is not None:
            raise ValueError("full_project_dev forbids a trajectory budget")
        selected = records
        status = "COMPLETE"
    elif coverage == "bounded":
        limit = _exact_positive_integer(budget, "bounded trajectory budget")
        selected = records[:limit]
        status = "PARTIAL"
    else:
        raise ValueError("coverage must be full_project_dev or bounded")
    if not selected:
        raise ValueError("offline selected population must not be empty")

    population_identity = {
        "partition": partition,
        "source_manifest_sha256": manifest_sha,
        "trajectory_count": len(records),
        "building_count": len({row["building"] for row in records}),
        "origin_count": sum(row["origin_count"] for row in records),
        "canonical_key_digest": _digest(
            [row["trajectory_key"] for row in records]
        ),
        "records": list(records),
    }
    population_sha = _digest(population_identity)
    selection_sha = _digest(
        {
            "population_sha256": population_sha,
            "coverage": coverage,
            "trajectory_budget": budget,
            "selected_keys": [row["trajectory_key"] for row in selected],
        }
    )
    trajectory_descriptors = tuple(
        TrajectoryDescriptor(
            dataset_index=int(row["dataset_index"]),
            trajectory_key=bytes.fromhex(str(row["trajectory_key"])),
            origin_count=int(row["origin_count"]),
        )
        for row in selected
    )
    origin_total = sum(row.origin_count for row in trajectory_descriptors)
    if origin_total <= 0:
        raise ValueError("selected population has no factual origins")
    owner_plan = build_rank_origin_plan(
        trajectory_descriptors,
        world_size=ranks,
        effective_batch=origin_total,
    )
    owner_keys = tuple(
        tuple(
            str(row["trajectory_key"])
            for selected_index, row in enumerate(selected)
            if owner_plan.owner_for(selected_index) == rank
        )
        for rank in range(ranks)
    )
    owner_indices = tuple(
        tuple(
            int(row["dataset_index"])
            for selected_index, row in enumerate(selected)
            if owner_plan.owner_for(selected_index) == rank
        )
        for rank in range(ranks)
    )
    owner_sha = _digest(
        {
            "selection_sha256": selection_sha,
            "world_size": ranks,
            "owner_keys_by_rank": [list(values) for values in owner_keys],
            "owner_indices_by_rank": [list(values) for values in owner_indices],
        }
    )
    return PopulationPlan(
        partition=partition,
        source_manifest_sha256=manifest_sha,
        coverage=coverage,
        status=status,
        population_records=records,
        selected_records=selected,
        owner_keys_by_rank=owner_keys,
        owner_indices_by_rank=owner_indices,
        population_sha256=population_sha,
        selection_sha256=selection_sha,
        owner_plan_sha256=owner_sha,
    )


def iter_rank_owned_trajectories(
    dataset: object,
    owned_indices: Sequence[int],
    *,
    workers: int,
) -> Iterator[Any]:
    """Materialize only this rank's exact indices, with no sampler padding."""

    worker_count = _exact_nonnegative_integer(workers, "workers")
    indices = tuple(
        _exact_nonnegative_integer(value, "owned dataset index")
        for value in owned_indices
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        sampler=indices,
        num_workers=worker_count,
        shuffle=False,
    )
    yield from loader


@dataclass(frozen=True)
class ValidatedRankUnion:
    receipts: tuple[dict[str, Any], ...]
    population_plan: PopulationPlan


def _key_list(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"rank {name} must be a trajectory-key sequence")
    keys = tuple(_trajectory_key(item) for item in value)
    if len(keys) != len(set(keys)):
        raise ValueError(f"rank {name} contains duplicate trajectory keys")
    return keys


def validate_rank_union(
    receipts: Sequence[Mapping[str, Any]],
    population_plan: PopulationPlan,
) -> ValidatedRankUnion:
    """Fail closed unless rank evidence is an exact disjoint owner union."""

    if type(population_plan) is not PopulationPlan:
        raise TypeError("population_plan must be a PopulationPlan")
    if isinstance(receipts, (str, bytes)) or not isinstance(receipts, Sequence):
        raise TypeError("rank receipts must be a sequence")
    if len(receipts) != population_plan.world_size:
        raise RuntimeError("rank receipt set is missing one or more ranks")
    by_rank: dict[int, dict[str, Any]] = {}
    run_sha: str | None = None
    all_observed: list[str] = []
    shard_roots: set[Path] = set()
    shard_paths: set[Path] = set()
    for raw in receipts:
        if not isinstance(raw, Mapping):
            raise ValueError("rank receipt must be a mapping")
        receipt = dict(raw)
        if receipt.get("schema") != _SHARD_SCHEMA or receipt.get("status") != "COMPLETE":
            raise RuntimeError("rank receipt schema/status is not COMPLETE")
        rank = _exact_nonnegative_integer(receipt.get("rank"), "rank")
        if rank >= population_plan.world_size or rank in by_rank:
            raise RuntimeError("rank receipt set has a duplicate or foreign rank")
        for field_name, expected in (
            ("population_sha256", population_plan.population_sha256),
            ("selection_sha256", population_plan.selection_sha256),
            ("owner_plan_sha256", population_plan.owner_plan_sha256),
        ):
            if receipt.get(field_name) != expected:
                raise RuntimeError(f"rank {rank} owner/population plan drift in {field_name}")
        if receipt.get("configured_world_size") != population_plan.world_size:
            raise RuntimeError("rank configured world size drift")
        if receipt.get("actual_world_size") != population_plan.world_size:
            raise RuntimeError("rank actual world size drift")
        _exact_nonnegative_integer(receipt.get("local_rank"), "rank local_rank")
        _exact_nonnegative_integer(receipt.get("workers"), "rank workers")
        if receipt.get("device") not in {"cpu", "cuda"}:
            raise RuntimeError("rank device identity is invalid")
        candidate_run_sha = _sha256(receipt.get("run_sha256"), "rank run SHA")
        if run_sha is None:
            run_sha = candidate_run_sha
        elif candidate_run_sha != run_sha:
            raise RuntimeError("rank receipts bind different run identities")
        expected_keys = _key_list(
            receipt.get("expected_trajectory_keys"), "expected keys"
        )
        observed_keys = _key_list(
            receipt.get("observed_trajectory_keys"), "observed keys"
        )
        owner_keys = population_plan.owner_keys_by_rank[rank]
        if expected_keys != owner_keys:
            raise RuntimeError(f"rank {rank} expected keys drift from owner plan")
        if observed_keys != owner_keys:
            raise RuntimeError(
                f"rank {rank} observed owner union is missing, duplicate or foreign"
            )
        expected_digest = _digest(list(expected_keys))
        observed_digest = _digest(list(observed_keys))
        if receipt.get("expected_key_digest") != expected_digest:
            raise RuntimeError(f"rank {rank} expected key digest drift")
        if receipt.get("observed_key_digest") != observed_digest:
            raise RuntimeError(f"rank {rank} observed key digest drift")
        counts = receipt.get("counts")
        if not isinstance(counts, Mapping):
            raise RuntimeError(f"rank {rank} counts are missing")
        requested = _exact_nonnegative_integer(
            counts.get("requested_trajectory_rows"),
            "rank requested trajectory rows",
        )
        materializations = _exact_nonnegative_integer(
            counts.get("unique_owner_materializations"),
            "rank unique owner materializations",
        )
        _exact_nonnegative_integer(
            counts.get("emitted_scalar_rows"), "rank emitted scalar rows"
        )
        if requested != len(owner_keys) or materializations != len(observed_keys):
            raise RuntimeError(f"rank {rank} owner materialization counts drift")
        shards = receipt.get("shards")
        if not isinstance(shards, Mapping):
            raise RuntimeError(f"rank {rank} shard identities are missing")
        if not set(shards).issubset(
            {"q", "g", "f", "reliance", "ranking", "invocations"}
        ):
            raise RuntimeError(f"rank {rank} has a foreign scalar shard")
        for shard_name, identity in shards.items():
            if not isinstance(identity, Mapping):
                raise RuntimeError(f"rank {rank} shard identity is malformed")
            path_raw = identity.get("path")
            if not isinstance(path_raw, str) or not Path(path_raw).is_absolute():
                raise RuntimeError(f"rank {rank} shard path must be absolute")
            path = Path(path_raw)
            if path.parent.name != f"rank_{rank:05d}" or path.parent.parent.name != "rank_shards":
                raise RuntimeError(f"rank {rank} shard path drifts from its owned directory")
            if path.name != f"{shard_name}.jsonl":
                raise RuntimeError(f"rank {rank} shard filename drifts from its component")
            resolved_path = path.resolve()
            if resolved_path in shard_paths:
                raise RuntimeError("rank receipts alias one scalar shard path")
            shard_paths.add(resolved_path)
            shard_roots.add(path.parent.parent.resolve())
        by_rank[rank] = receipt
        all_observed.extend(observed_keys)
    if set(by_rank) != set(range(population_plan.world_size)):
        raise RuntimeError("rank receipt union is missing one or more ranks")
    if len(shard_roots) > 1:
        raise RuntimeError("rank shards do not share one run-owned root")
    selected = tuple(row["trajectory_key"] for row in population_plan.selected_records)
    if len(all_observed) != len(set(all_observed)):
        raise RuntimeError("rank owner union contains duplicate trajectories")
    if set(all_observed) != set(selected):
        raise RuntimeError("rank owner union is missing or contains foreign trajectories")
    return ValidatedRankUnion(
        receipts=tuple(by_rank[rank] for rank in range(population_plan.world_size)),
        population_plan=population_plan,
    )


def _iter_shard_rows(identity: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    path_raw = identity.get("path")
    if not isinstance(path_raw, str) or not path_raw:
        raise ValueError("rank shard path must be non-empty")
    path = Path(path_raw)
    if not path.is_file():
        raise RuntimeError(f"rank shard is missing: {path}")
    byte_count = identity.get("bytes")
    if type(byte_count) is not int or byte_count < 0 or path.stat().st_size != byte_count:
        raise RuntimeError("rank shard byte count drift")
    if _file_sha256(path) != _sha256(identity.get("sha256"), "rank shard SHA"):
        raise RuntimeError("rank shard SHA drift")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"rank shard contains malformed JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError("rank shard rows must be JSON objects")
            yield row


def iter_validated_scalar_shards(
    validated: ValidatedRankUnion,
) -> Iterator[dict[str, Any]]:
    """Stream all scalar shards in canonical global emission order."""

    if type(validated) is not ValidatedRankUnion:
        raise TypeError("validated rank union is required")
    iterators: list[Iterator[dict[str, Any]]] = []
    for receipt in validated.receipts:
        shards = receipt.get("shards")
        if not isinstance(shards, Mapping):
            raise RuntimeError("rank shard identities must be a mapping")
        for name in sorted(shards):
            identity = shards[name]
            if not isinstance(name, str) or not isinstance(identity, Mapping):
                raise RuntimeError("rank shard identity is malformed")
            iterators.append(_iter_shard_rows(identity))

    heap: list[tuple[int, int, int, dict[str, Any]]] = []

    def push(iterator_index: int) -> None:
        try:
            row = next(iterators[iterator_index])
        except StopIteration:
            return
        population_ordinal = _exact_nonnegative_integer(
            row.get("population_ordinal"), "population_ordinal"
        )
        emission_ordinal = _exact_nonnegative_integer(
            row.get("emission_ordinal"), "emission_ordinal"
        )
        if population_ordinal >= len(validated.population_plan.selected_records):
            raise RuntimeError("rank shard population ordinal is foreign")
        heapq.heappush(
            heap,
            (population_ordinal, emission_ordinal, iterator_index, row),
        )

    for index in range(len(iterators)):
        push(index)
    prior: tuple[int, int] | None = None
    while heap:
        population_ordinal, emission_ordinal, iterator_index, row = heapq.heappop(heap)
        order = (population_ordinal, emission_ordinal)
        expected = (
            (0, 0)
            if prior is None
            else (
                (prior[0], prior[1] + 1)
                if population_ordinal == prior[0]
                else (prior[0] + 1, 0)
            )
        )
        if order != expected:
            raise RuntimeError(
                "rank shards contain a missing, duplicate or out-of-order global "
                "emission ordinal"
            )
        prior = order
        yield {key: value for key, value in row.items() if key not in _CONTROL_FIELDS}
        push(iterator_index)
    if prior is None or prior[0] != len(validated.population_plan.selected_records) - 1:
        raise RuntimeError("rank shards do not cover every selected trajectory")


__all__ = [
    "ExecutionIdentity",
    "PopulationPlan",
    "ValidatedRankUnion",
    "build_population_plan",
    "canonicalize_population_descriptors",
    "iter_rank_owned_trajectories",
    "iter_validated_scalar_shards",
    "managed_process_group",
    "project_dataset_descriptors",
    "validate_rank_union",
]
