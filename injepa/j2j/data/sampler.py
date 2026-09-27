"""Lossless exact-length batch plans for categorical INTACT trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import random
import re
from typing import Any

import torch

from j2j.receipts import canonical_json_bytes


_MAX_UINT32 = 2**32
_MAX_UINT64 = 2**64
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CategoricalBatchPlanError(ValueError):
    """Raised when a categorical plan descriptor or capacity is invalid."""


@dataclass(frozen=True)
class CategoricalBatchBox:
    motion_length: int
    dataset_indices: tuple[int, ...]
    trajectory_keys: tuple[bytes, ...]
    box_sha256: str

    @property
    def size(self) -> int:
        return len(self.dataset_indices)


@dataclass(frozen=True)
class CategoricalMultiplicity:
    dataset_index: int
    trajectory_key: bytes
    count: int


@dataclass(frozen=True)
class CategoricalBatchPlan:
    trajectory_batch_capacity: int
    ordered_boxes: tuple[CategoricalBatchBox, ...]
    num_length_one_trajectories: int
    t1_atom_sizes: tuple[int, ...]
    t1_pair_count: int
    t1_triple_count: int
    multiplicity: tuple[CategoricalMultiplicity, ...]
    total_trajectory_count: int
    ordered_box_count: int
    plan_sha256: str
    singleton_batch_norm_policy: str = "error"

    @property
    def ledger_sha256(self) -> str:
        return self.plan_sha256


@dataclass(frozen=True)
class _Descriptor:
    dataset_index: int
    trajectory_key: bytes
    motion_length: int
    projection_partition: str | None = None


def _exact_uint(value: object, *, name: str, upper: int) -> int:
    if type(value) is not int:
        raise CategoricalBatchPlanError(f"{name} must be an exact Python integer")
    if value < 0 or value >= upper:
        raise CategoricalBatchPlanError(f"{name} is outside its unsigned range")
    return value


def _key_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        key = value
    elif isinstance(value, str) and len(value) == 64:
        try:
            key = bytes.fromhex(value)
        except ValueError as exc:
            raise CategoricalBatchPlanError("trajectory key is not hexadecimal") from exc
    else:
        raise CategoricalBatchPlanError("trajectory key must be 32 raw bytes")
    if len(key) != 32:
        raise CategoricalBatchPlanError("trajectory key must be exactly 32 raw bytes")
    return key


def _descriptor_values(descriptor: object) -> tuple[object, object, object, object | None]:
    if isinstance(descriptor, Mapping):
        def first(names: tuple[str, ...]) -> object | None:
            for name in names:
                if name in descriptor:
                    return descriptor[name]
            return None
        return (
            first(("dataset_index", "index")),
            first(("canonical_trajectory_key", "trajectory_key", "key")),
            first(("motion_length", "T", "length")),
            first(("projection_partition", "partition")),
        )
    if isinstance(descriptor, Sequence) and not isinstance(descriptor, (str, bytes, bytearray)):
        if len(descriptor) not in (3, 4):
            raise CategoricalBatchPlanError("descriptor sequence must have three or four fields")
        values = tuple(descriptor)
        return values[0], values[1], values[2], values[3] if len(values) == 4 else None
    def attr(names: tuple[str, ...]) -> object | None:
        for name in names:
            if hasattr(descriptor, name):
                return getattr(descriptor, name)
        return None
    return (
        attr(("dataset_index", "index")),
        attr(("canonical_trajectory_key", "trajectory_key", "key")),
        attr(("motion_length", "T", "length")),
        attr(("projection_partition", "partition")),
    )


def _normalize_descriptors(descriptors: Sequence[object]) -> tuple[_Descriptor, ...]:
    if isinstance(descriptors, (str, bytes, bytearray)) or not isinstance(descriptors, Sequence):
        raise TypeError("descriptors must be a sequence")
    normalized: list[_Descriptor] = []
    seen_indices: set[int] = set()
    seen_keys: set[bytes] = set()
    for raw in descriptors:
        index_raw, key_raw, length_raw, partition_raw = _descriptor_values(raw)
        index = _exact_uint(index_raw, name="dataset_index", upper=_MAX_UINT64)
        length = _exact_uint(length_raw, name="motion_length", upper=_MAX_UINT32)
        key = _key_bytes(key_raw)
        if index in seen_indices:
            raise CategoricalBatchPlanError("dataset indices must be unique")
        if key in seen_keys:
            raise CategoricalBatchPlanError("trajectory keys must be unique")
        if partition_raw is not None and partition_raw not in {"project-train", "project-dev"}:
            raise CategoricalBatchPlanError("descriptor projection partition is invalid")
        seen_indices.add(index)
        seen_keys.add(key)
        normalized.append(_Descriptor(index, key, length, partition_raw))
    return tuple(sorted(normalized, key=lambda item: (item.motion_length, item.trajectory_key, item.dataset_index)))


def _validate_capacity(value: object) -> int:
    if type(value) is not int or value < 1:
        raise CategoricalBatchPlanError("trajectory_batch_capacity must be a positive exact Python integer")
    return value


def _validate_singleton_batch_norm_policy(value: object) -> str:
    if value not in {"error", "running_stats"}:
        raise CategoricalBatchPlanError(
            "singleton_batch_norm_policy must be 'error' or 'running_stats'"
        )
    return str(value)


def _t1_atoms(
    items: Sequence[_Descriptor], *, allow_singleton: bool = False
) -> tuple[tuple[_Descriptor, ...], ...]:
    count = len(items)
    if count == 0:
        return ()
    if count == 1:
        if allow_singleton:
            return (tuple(items),)
        raise CategoricalBatchPlanError("one T=1 trajectory cannot satisfy the BN batch rule")
    atoms: list[tuple[_Descriptor, ...]] = []
    offset = 0
    if count % 2:
        atoms.append(tuple(items[:3]))
        offset = 3
    while offset < count:
        atoms.append(tuple(items[offset : offset + 2]))
        offset += 2
    return tuple(atoms)


def _check_t1_feasibility(
    count: int, capacity: int, *, allow_singleton: bool = False
) -> None:
    if count == 0:
        return
    if capacity == 1 and not (allow_singleton and count == 1):
        raise CategoricalBatchPlanError("T=1 bucket is infeasible for the requested trajectory capacity")
    if count == 1 and not allow_singleton:
        raise CategoricalBatchPlanError("T=1 bucket is infeasible for the requested trajectory capacity")
    if capacity == 2 and count % 2 and not (allow_singleton and count == 1):
        raise CategoricalBatchPlanError("odd T=1 bucket requires capacity at least three")


def _box_sha(motion_length: int, indices: tuple[int, ...], keys: tuple[bytes, ...]) -> str:
    preimage = {
        "domain": "J2J_CATEGORICAL_BATCH_V1",
        "schema_version": 1,
        "fields": [
            {"name": "motion_length", "type": "uint32", "value": motion_length},
            {"name": "dataset_indices", "type": "uint64[]", "value": list(indices)},
            {"name": "trajectory_keys", "type": "bytes32_hex[]", "value": [key.hex() for key in keys]},
        ],
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def _plan_sha(capacity: int, boxes: tuple[CategoricalBatchBox, ...], n1: int,
              atom_sizes: tuple[int, ...], pair_count: int, triple_count: int,
              multiplicity: tuple[CategoricalMultiplicity, ...], total_count: int,
              singleton_batch_norm_policy: str) -> str:
    box_values = [
        {"motion_length": box.motion_length, "dataset_indices": list(box.dataset_indices),
         "trajectory_keys": [key.hex() for key in box.trajectory_keys], "box_sha256": box.box_sha256}
        for box in boxes
    ]
    multiplicity_values = [
        {"dataset_index": entry.dataset_index, "trajectory_key": entry.trajectory_key.hex(), "count": entry.count}
        for entry in multiplicity
    ]
    preimage = {
        "domain": "J2J_CATEGORICAL_PLAN_V1", "schema_version": 1,
        "fields": [
            {"name": "trajectory_batch_capacity", "type": "positive_int", "value": capacity},
            {"name": "ordered_boxes", "type": "box[]", "value": box_values},
            {"name": "num_length_one_trajectories", "type": "uint64", "value": n1},
            {"name": "t1_atom_sizes", "type": "uint8[]", "value": list(atom_sizes)},
            {"name": "t1_pair_count", "type": "uint64", "value": pair_count},
            {"name": "t1_triple_count", "type": "uint64", "value": triple_count},
            {"name": "singleton_batch_norm_policy", "type": "string", "value": singleton_batch_norm_policy},
            {"name": "multiplicity", "type": "multiplicity[]", "value": multiplicity_values},
            {"name": "total_trajectory_count", "type": "uint64", "value": total_count},
            {"name": "ordered_box_count", "type": "uint64", "value": len(boxes)},
        ],
    }
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def build_categorical_batch_plan(
    descriptors: Sequence[object],
    trajectory_batch_capacity: int,
    *,
    generator: torch.Generator | None = None,
    singleton_batch_norm_policy: str = "error",
) -> CategoricalBatchPlan:
    """Materialize one exact-length, no-drop categorical epoch plan."""
    capacity = _validate_capacity(trajectory_batch_capacity)
    singleton_policy = _validate_singleton_batch_norm_policy(
        singleton_batch_norm_policy
    )
    allow_singleton = singleton_policy == "running_stats"
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a caller-owned torch.Generator")
    normalized = _normalize_descriptors(descriptors)
    t1 = tuple(item for item in normalized if item.motion_length == 1)
    _check_t1_feasibility(len(t1), capacity, allow_singleton=allow_singleton)
    t1_atoms = _t1_atoms(t1, allow_singleton=allow_singleton)
    buckets: dict[int, list[_Descriptor]] = {}
    for item in normalized:
        buckets.setdefault(item.motion_length, []).append(item)
    base_boxes: list[CategoricalBatchBox] = []
    for motion_length in sorted(buckets):
        bucket = buckets[motion_length]
        atoms = t1_atoms if motion_length == 1 else tuple((item,) for item in bucket)
        current: list[_Descriptor] = []
        current_size = 0
        for atom in atoms:
            atom_size = len(atom)
            if atom_size > capacity:
                raise CategoricalBatchPlanError("an indivisible atom exceeds trajectory capacity")
            if current and current_size + atom_size > capacity:
                indices = tuple(item.dataset_index for item in current)
                keys = tuple(item.trajectory_key for item in current)
                base_boxes.append(CategoricalBatchBox(motion_length, indices, keys, _box_sha(motion_length, indices, keys)))
                current, current_size = [], 0
            current.extend(atom)
            current_size += atom_size
        if current:
            indices = tuple(item.dataset_index for item in current)
            keys = tuple(item.trajectory_key for item in current)
            base_boxes.append(CategoricalBatchBox(motion_length, indices, keys, _box_sha(motion_length, indices, keys)))
    if generator is None or len(base_boxes) <= 1:
        ordered_boxes = tuple(base_boxes)
    else:
        try:
            permutation = torch.randperm(len(base_boxes), generator=generator, device=generator.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CategoricalBatchPlanError("box permutation failed") from exc
        ordered_boxes = tuple(base_boxes[int(index)] for index in permutation.cpu().tolist())
    multiplicity = tuple(
        CategoricalMultiplicity(item.dataset_index, item.trajectory_key, 1)
        for item in sorted(normalized, key=lambda value: (value.dataset_index, value.trajectory_key))
    )
    atom_sizes = tuple(len(atom) for atom in t1_atoms)
    pair_count = sum(size == 2 for size in atom_sizes)
    triple_count = sum(size == 3 for size in atom_sizes)
    plan_sha = _plan_sha(capacity, ordered_boxes, len(t1), atom_sizes, pair_count,
                         triple_count, multiplicity, len(normalized), singleton_policy)
    return CategoricalBatchPlan(capacity, ordered_boxes, len(t1), atom_sizes,
                                pair_count, triple_count, multiplicity,
                                len(normalized), len(ordered_boxes), plan_sha,
                                singleton_policy)


def build_rank_sharded_batch_plans(
    descriptors: Sequence[object],
    *,
    capacities: Sequence[int],
    seed: int,
    singleton_batch_norm_policy: str = "error",
    max_attempts: int = 64,
) -> dict[int, CategoricalBatchPlan]:
    """Build deterministic, lossless rank-local plans with equal box counts.

    This planning-only helper consumes the same descriptor projection as
    :func:`build_categorical_batch_plan`; it never reads payloads, duplicates a
    trajectory, or pads a rank.  T=1 pair/triple atoms remain on one rank.
    """

    if isinstance(capacities, (str, bytes, bytearray)) or not isinstance(capacities, Sequence):
        raise CategoricalBatchPlanError("capacities must be a non-empty sequence")
    caps = tuple(capacities)
    if not caps or any(type(value) is not int or value < 1 for value in caps):
        raise CategoricalBatchPlanError("capacities must contain positive exact integers")
    if type(seed) is not int or seed < 0:
        raise CategoricalBatchPlanError("seed must be a non-negative exact integer")
    if type(max_attempts) is not int or max_attempts < 1:
        raise CategoricalBatchPlanError("max_attempts must be a positive exact integer")
    policy = _validate_singleton_batch_norm_policy(singleton_batch_norm_policy)
    normalized = _normalize_descriptors(descriptors)
    if len(normalized) < len(caps):
        raise CategoricalBatchPlanError("descriptor count is smaller than rank count")

    by_length: dict[int, list[_Descriptor]] = {}
    for item in normalized:
        by_length.setdefault(item.motion_length, []).append(item)
    atoms: list[tuple[_Descriptor, ...]] = []
    for length in sorted(by_length):
        if length == 1:
            atoms.extend(_t1_atoms(by_length[length], allow_singleton=policy == "running_stats"))
        else:
            atoms.extend((item,) for item in by_length[length])

    def box_counts(assignments: Sequence[Sequence[tuple[_Descriptor, ...]]]) -> list[int]:
        result: list[int] = []
        for rank, rank_atoms in enumerate(assignments):
            counts: dict[int, int] = {}
            t1_boxes = 0
            for atom in rank_atoms:
                length = atom[0].motion_length
                if length == 1:
                    t1_boxes += 1
                else:
                    counts[length] = counts.get(length, 0) + len(atom)
            result.append(t1_boxes + sum((n + caps[rank] - 1) // caps[rank] for n in counts.values()))
        return result

    def assignment_for(attempt: int) -> list[list[tuple[_Descriptor, ...]]]:
        rng = random.Random(seed + attempt)
        shuffled = list(atoms)
        rng.shuffle(shuffled)
        assigned: list[list[tuple[_Descriptor, ...]]] = [[] for _ in caps]
        counts: list[dict[int, int]] = [{} for _ in caps]
        boxes = [0 for _ in caps]

        def apply(rank: int, atom: tuple[_Descriptor, ...], sign: int) -> None:
            length = atom[0].motion_length
            size = len(atom)
            if length == 1:
                boxes[rank] += sign
                return
            old = counts[rank].get(length, 0)
            before = (old + caps[rank] - 1) // caps[rank]
            new = old + sign * size
            if new < 0:
                raise CategoricalBatchPlanError("rank sharding produced a negative length count")
            after = (new + caps[rank] - 1) // caps[rank]
            boxes[rank] += after - before
            if new:
                counts[rank][length] = new
            else:
                counts[rank].pop(length, None)

        for atom in shuffled:
            # Keep the complete T=1 atom sequence on one rank.  Rebuilding a
            # rank plan normalizes the length-one bucket, so splitting atoms
            # across ranks would allow the downstream owner to regroup them.
            if atom[0].motion_length == 1:
                assigned[0].append(atom)
                apply(0, atom, +1)
                continue
            scores = []
            for rank in range(len(caps)):
                length = atom[0].motion_length
                if length == 1:
                    projected = boxes[rank] + 1
                else:
                    old = counts[rank].get(length, 0)
                    projected = boxes[rank] + int(old % caps[rank] == 0)
                trajectory_count = sum(len(item) for item in assigned[rank])
                scores.append((projected, trajectory_count / caps[rank], rank))
            rank = min(scores)[2]
            assigned[rank].append(atom)
            apply(rank, atom, +1)

        # Repair capacity-residue differences with deterministic moves/swaps.
        for _ in range(len(atoms) * 2):
            current = list(boxes)
            if len(set(current)) == 1:
                break
            hi = max(range(len(caps)), key=lambda r: current[r])
            lo = min(range(len(caps)), key=lambda r: current[r])
            improved = False
            for index, atom in enumerate(tuple(assigned[hi])):
                if atom[0].motion_length == 1:
                    continue
                if len(assigned[hi]) <= 1:
                    break
                assigned[hi].pop(index)
                assigned[lo].append(atom)
                apply(hi, atom, -1)
                apply(lo, atom, +1)
                candidate = list(boxes)
                apply(lo, atom, -1)
                apply(hi, atom, +1)
                assigned[lo].pop()
                assigned[hi].insert(index, atom)
                if max(candidate) - min(candidate) < max(current) - min(current):
                    assigned[hi].pop(index)
                    assigned[lo].append(atom)
                    apply(hi, atom, -1)
                    apply(lo, atom, +1)
                    improved = True
                    break
            if improved:
                continue
            for hi_index, hi_atom in enumerate(tuple(assigned[hi])):
                if hi_atom[0].motion_length == 1:
                    continue
                swapped = False
                for lo_index, lo_atom in enumerate(tuple(assigned[lo])):
                    if lo_atom[0].motion_length == 1:
                        continue
                    assigned[hi][hi_index], assigned[lo][lo_index] = lo_atom, hi_atom
                    apply(hi, hi_atom, -1)
                    apply(lo, lo_atom, -1)
                    apply(hi, lo_atom, +1)
                    apply(lo, hi_atom, +1)
                    candidate = list(boxes)
                    apply(lo, hi_atom, -1)
                    apply(hi, lo_atom, -1)
                    apply(hi, hi_atom, +1)
                    apply(lo, lo_atom, +1)
                    assigned[hi][hi_index], assigned[lo][lo_index] = hi_atom, lo_atom
                    if max(candidate) - min(candidate) < max(current) - min(current):
                        assigned[hi][hi_index], assigned[lo][lo_index] = lo_atom, hi_atom
                        swapped = improved = True
                        break
                if swapped:
                    break
            if not improved:
                break
        return assigned

    chosen = None
    for attempt in range(max_attempts):
        candidate = assignment_for(attempt)
        if all(candidate) and len(set(box_counts(candidate))) == 1:
            chosen = candidate
            break
    if chosen is None:
        raise CategoricalBatchPlanError(
            "deterministic rank sharding could not equalize ordered box counts; adjust capacities or seed"
        )

    plans: dict[int, CategoricalBatchPlan] = {}
    for rank, rank_atoms in enumerate(chosen):
        rank_descriptors = [item for atom in rank_atoms for item in atom]
        generator = torch.Generator().manual_seed(seed + rank)
        plans[rank] = build_categorical_batch_plan(
            rank_descriptors,
            trajectory_batch_capacity=caps[rank],
            generator=generator,
            singleton_batch_norm_policy=policy,
        )
    if len({plan.ordered_box_count for plan in plans.values()}) != 1:
        raise CategoricalBatchPlanError("rank plans have unequal ordered box counts")
    return plans


class CategoricalBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Standard PyTorch adapter over an already materialized batch plan."""

    def __init__(self, plan: CategoricalBatchPlan) -> None:
        if type(plan) is not CategoricalBatchPlan:
            raise TypeError("CategoricalBatchSampler expects a CategoricalBatchPlan")
        self._plan = plan

    @property
    def plan(self) -> CategoricalBatchPlan:
        return self._plan

    def __iter__(self):
        for box in self._plan.ordered_boxes:
            yield list(box.dataset_indices)

    def __len__(self) -> int:
        return self._plan.ordered_box_count


def _formal_exact_positive(value: object, name: str) -> int:
    """Validate a schedule scalar without coercing configuration values."""

    if type(value) is not int or value < 1:
        raise CategoricalBatchPlanError(f"{name} must be a positive exact Python integer")
    return value


def _formal_sha(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise CategoricalBatchPlanError(f"{name} must be a lowercase SHA-256 string")
    return value


def _formal_plan_json(value: object) -> bytes:
    """Use the same canonical JSON owner as the formal-plan loader."""

    return canonical_json_bytes(value)


def _pred_proj_call_rows(batch_size: int, motion_length: int) -> list[int]:
    """Project exact ``(B,T)`` geometry to INTACT's real call partition."""

    if motion_length == 0:
        return []
    row_count = batch_size * min(3, motion_length)
    return [row_count] * max(1, motion_length - 2)


def _normalise_rank_plans(
    batch_plan: CategoricalBatchPlan | Mapping[object, CategoricalBatchPlan],
    *,
    world_size: int | None,
) -> tuple[int, dict[int, CategoricalBatchPlan]]:
    """Normalize one or explicitly sharded D1B plan per rank.

    The single-plan form is the normal current path.  A mapping is accepted
    only when the caller has already built one exact D1B plan for every rank;
    this helper never shards or duplicates trajectories itself.
    """

    if isinstance(batch_plan, CategoricalBatchPlan):
        effective_world = 1 if world_size is None else _formal_exact_positive(world_size, "world_size")
        if effective_world != 1:
            raise CategoricalBatchPlanError(
                "a single categorical batch plan can only describe world_size=1; provide one plan per rank"
            )
        return 1, {0: batch_plan}
    if not isinstance(batch_plan, Mapping) or not batch_plan:
        raise CategoricalBatchPlanError(
            "batch_plan must be a CategoricalBatchPlan or non-empty rank mapping"
        )
    rank_plans: dict[int, CategoricalBatchPlan] = {}
    for raw_rank, plan in batch_plan.items():
        if type(raw_rank) is int:
            rank = raw_rank
        elif isinstance(raw_rank, str) and raw_rank.isdigit():
            rank = int(raw_rank)
        else:
            raise CategoricalBatchPlanError("rank-plan keys must be non-negative integers")
        if rank < 0 or not isinstance(plan, CategoricalBatchPlan):
            raise CategoricalBatchPlanError("rank-plan values must be CategoricalBatchPlan instances")
        if rank in rank_plans:
            raise CategoricalBatchPlanError("rank-plan keys must be unique")
        rank_plans[rank] = plan
    inferred = max(rank_plans) + 1
    effective_world = inferred if world_size is None else _formal_exact_positive(world_size, "world_size")
    if set(rank_plans) != set(range(effective_world)):
        raise CategoricalBatchPlanError("rank plans must contain every rank exactly once")
    return effective_world, rank_plans


def build_formal_update_plan_payload(
    batch_plan: CategoricalBatchPlan | Mapping[object, CategoricalBatchPlan],
    *,
    passes: int,
    nominal_accumulation_steps: int,
    horizon: int,
    ledger_sha256: str,
    branch_weights: Mapping[str, float],
    snapshot_schedule: Sequence[Mapping[str, Any]],
    world_size: int | None = None,
    sync_mode: str = "every_microbatch",
    batch_norm_mode: str = "local",
    singleton_batch_norm_policy: str = "error",
    find_unused_parameters: bool = True,
    identity_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the formal A/G/F update ledger from an existing D1B plan.

    This is deliberately a pure, metadata-only bridge.  ``batch_plan`` is
    already the owner of trajectory ordering, exact-length buckets, and
    box hashes.  We repeat that ordered plan once per configured factual pass,
    group boxes *within each pass* for gradient accumulation, and derive all
    branch row counts from each box's ``(B,T)`` geometry.  No RGB, cache grid,
    action, target, model, or optimizer is created or read here.

    A rank mapping is an explicit caller-owned sharding contract.  The helper
    does not split a plan, pad it, or carry an accumulation window across pass
    boundaries.  The returned mapping is accepted directly by
    ``FormalUpdatePlan.from_mapping`` and may be serialized by a thin CLI.
    """

    effective_passes = _formal_exact_positive(passes, "passes")
    nominal_k = _formal_exact_positive(
        nominal_accumulation_steps, "nominal_accumulation_steps"
    )
    effective_horizon = _formal_exact_positive(horizon, "horizon")
    ledger = _formal_sha(ledger_sha256, "ledger_sha256")
    effective_world, rank_plans = _normalise_rank_plans(
        batch_plan, world_size=world_size
    )
    singleton_policy = _validate_singleton_batch_norm_policy(
        singleton_batch_norm_policy
    )

    if not isinstance(branch_weights, Mapping) or set(branch_weights) != {"F", "L", "G"}:
        raise CategoricalBatchPlanError("branch_weights must have exact keys F,L,G")
    weights: dict[str, float] = {}
    for branch in ("F", "L", "G"):
        value = branch_weights[branch]
        if isinstance(value, bool):
            raise CategoricalBatchPlanError(f"branch_weights[{branch}] must be finite and non-negative")
        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise CategoricalBatchPlanError(
                f"branch_weights[{branch}] must be finite and non-negative"
            ) from exc
        if not math.isfinite(converted) or converted < 0:
            raise CategoricalBatchPlanError(
                f"branch_weights[{branch}] must be finite and non-negative"
            )
        weights[branch] = converted

    if not isinstance(snapshot_schedule, Sequence) or isinstance(
        snapshot_schedule, (str, bytes, bytearray)
    ):
        raise CategoricalBatchPlanError("snapshot_schedule must be an ordered sequence")
    schedule: list[dict[str, Any]] = []
    for item in snapshot_schedule:
        if not isinstance(item, Mapping):
            raise CategoricalBatchPlanError("snapshot_schedule entries must be mappings")
        # Copy only JSON-compatible scalar/sequence values; the formal owner
        # performs the detailed threshold/token admission on the same bytes.
        schedule.append(dict(item))

    box_counts = {rank: plan.ordered_box_count for rank, plan in rank_plans.items()}
    if len(set(box_counts.values())) != 1:
        raise CategoricalBatchPlanError(
            "all rank plans must contain the same number of ordered boxes"
        )
    boxes_per_pass = next(iter(box_counts.values()))
    if boxes_per_pass < 1:
        raise CategoricalBatchPlanError("categorical batch plan must contain at least one box")

    # Chunk each rank's ordered boxes independently per pass.  Equal box
    # counts and one fixed K imply identical update/micro boundaries across
    # ranks; local geometries may still differ when an explicit sharding
    # caller supplies them.
    groups_per_pass = (boxes_per_pass + nominal_k - 1) // nominal_k
    planned_updates = effective_passes * groups_per_pass
    if planned_updates < 2:
        raise CategoricalBatchPlanError(
            "formal update plan requires at least two optimizer updates"
        )

    # The public descriptor ABI is flat by rank.  Emit a separate row for
    # every rank/pass/group/micro while retaining a single global cumulative
    # occurrence counter at each optimizer boundary.
    rows_by_rank: dict[str, list[dict[str, Any]]] = {
        str(rank): [] for rank in range(effective_world)
    }
    committed_occurrences = 0
    update_cursor = 0
    for pass_index in range(effective_passes):
        for group_start in range(0, boxes_per_pass, nominal_k):
            group_size = min(nominal_k, boxes_per_pass - group_start)
            previous_occurrences = committed_occurrences
            # Compute global geometry for every corresponding micro first.
            global_geometry: list[tuple[int, int, int, int]] = []
            for micro_index in range(group_size):
                boxes = tuple(
                    rank_plans[rank].ordered_boxes[group_start + micro_index]
                    for rank in range(effective_world)
                )
                global_geometry.append(
                    (
                        sum(box.size for box in boxes),
                        sum(box.size * box.motion_length for box in boxes),
                        sum(box.size * box.motion_length for box in boxes),
                        sum(box.size * (box.motion_length + 1) for box in boxes),
                    )
                )
            committed_occurrences += sum(item[0] for item in global_geometry)
            crossed_labels = tuple(
                dict(item)
                for item in schedule
                if previous_occurrences < int(item["threshold_occurrences"]) <= committed_occurrences
            )
            for rank in range(effective_world):
                plan = rank_plans[rank]
                for micro_index in range(group_size):
                    box = plan.ordered_boxes[group_start + micro_index]
                    global_b, global_f, global_l, global_g = global_geometry[micro_index]
                    local_b = box.size
                    local_f = local_b * box.motion_length
                    local_g = local_b * (box.motion_length + 1)
                    rows_by_rank[str(rank)].append(
                        {
                            "update_id": update_cursor,
                            "micro_index": micro_index,
                            "micro_count": group_size,
                            "nominal_k": nominal_k,
                            "is_update_boundary": micro_index == group_size - 1,
                            "is_tail": (
                                pass_index == effective_passes - 1
                                and group_start + group_size == boxes_per_pass
                                and micro_index == group_size - 1
                            ),
                            "pass_id": pass_index + 1,
                            "pass_end": (
                                group_start + group_size == boxes_per_pass
                                and micro_index == group_size - 1
                            ),
                            "pass_tail": (
                                group_start + group_size == boxes_per_pass
                                and micro_index == group_size - 1
                            ),
                            "rank": rank,
                            "world_size": effective_world,
                            "box_sha256": box.box_sha256,
                            "trajectory_count": local_b,
                            "motion_length": box.motion_length,
                            "n_f_rows": local_f,
                            "n_local_rows": local_f,
                            "n_goal_rows": local_g,
                            "global_trajectory_count": global_b,
                            "global_n_f_rows": global_f,
                            "global_n_local_rows": global_l,
                            "global_n_goal_rows": global_g,
                            "committed_occurrences_after_update": committed_occurrences,
                            "snapshot_labels": (
                                list(crossed_labels)
                                if micro_index == group_size - 1
                                else []
                            ),
                            "pred_proj_call_rows": _pred_proj_call_rows(
                                local_b, box.motion_length
                            ),
                        }
                    )
            update_cursor += 1

    total_occurrences = committed_occurrences
    warmup_steps = max(1, math.floor(0.01 * planned_updates))
    if warmup_steps >= planned_updates:
        # This is normally reachable only for a malformed tiny algebraic
        # fixture; fail before handing bytes to the formal owner.
        raise CategoricalBatchPlanError(
            "formal update plan must have more updates than warmup steps"
        )
    payload: dict[str, Any] = {
        "ledger_sha256": ledger,
        "world_size": effective_world,
        "horizon": effective_horizon,
        "nominal_accumulation_steps": nominal_k,
        "planned_update_count": planned_updates,
        "total_training_occurrences": total_occurrences,
        "snapshot_schedule": schedule,
        "warmup_steps": warmup_steps,
        "max_steps": planned_updates,
        "branch_weights": weights,
        "sync_mode": sync_mode,
        "batch_norm_mode": batch_norm_mode,
        "singleton_batch_norm_policy": singleton_policy,
        "find_unused_parameters": find_unused_parameters,
        "descriptors_by_rank": rows_by_rank,
        "d1b_batch_plan_sha256_by_rank": {
            str(rank): plan.plan_sha256 for rank, plan in rank_plans.items()
        },
        "d1b_factual_passes": effective_passes,
    }
    if identity_fields is not None:
        if not isinstance(identity_fields, Mapping):
            raise CategoricalBatchPlanError("identity_fields must be a mapping")
        for key, value in identity_fields.items():
            if key in payload:
                raise CategoricalBatchPlanError(
                    f"identity_fields cannot overwrite reserved plan field {key!r}"
                )
            payload[str(key)] = value

    # Validate through the sole formal-plan owner now, before a caller writes
    # a plan file.  Import is local so normal D1B dataset/sampler users do not
    # pay the training-owner import cost.
    try:
        from j2j.intact_accumulation import FormalUpdatePlan

        FormalUpdatePlan.from_mapping(payload)
    except CategoricalBatchPlanError:
        raise
    except Exception as exc:
        raise CategoricalBatchPlanError(
            "formal update-plan payload failed the existing admission owner"
        ) from exc
    return payload


__all__ = [
    "CategoricalBatchBox",
    "CategoricalBatchPlan",
    "CategoricalBatchPlanError",
    "CategoricalBatchSampler",
    "CategoricalMultiplicity",
    "build_categorical_batch_plan",
    "build_rank_sharded_batch_plans",
    "build_formal_update_plan_payload",
]
