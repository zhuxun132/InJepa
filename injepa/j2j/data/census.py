"""Exact H4 factual, goal-view, and branch-count census primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .goals import hashed_goal_index


class CensusError(ValueError):
    """Raised when census inputs violate the frozen factual-data contract."""


@dataclass(frozen=True)
class GoalViewOrigin:
    trajectory_key: bytes
    t: int
    terminal: int
    pre_eviction_support: bool


@dataclass(frozen=True)
class CensusTrajectory:
    partition: str
    canonical_trajectory_key: bytes
    dedup_key: bytes
    scan_id: str
    actions: tuple[int, ...]
    dedup_alias_count: int
    zero_motion_terminal: bool
    goal_view_origins: tuple[GoalViewOrigin, ...]


@dataclass(frozen=True)
class PartitionCensus:
    partition: str
    trajectories: int
    frames: int
    N1: int
    N2: int
    N3: int
    N4: int
    M4: int
    N_hash_eligible: int
    N_hash_eq_terminal: int
    N_hash_unique: int
    N_Q: int
    M_Q: int
    N_INT: int
    N_F: int
    N_Glocal: int
    N_Ggoal: int
    dedup_alias_count: int
    zero_motion_terminal_rows: int


@dataclass(frozen=True)
class ProductionCensus:
    audit_kind: str
    max_horizon: int
    partitions: tuple[PartitionCensus, ...]


@dataclass(frozen=True)
class GoalViewCensus:
    N_hash_eligible: int
    N_hash_eq_terminal: int
    N_hash_unique: int
    N_Q: int
    M_4_unique: int
    M_Q: int
    M_Q_plus: int


@dataclass
class _MutablePartitionCensus:
    trajectories: int = 0
    frames: int = 0
    N1: int = 0
    N2: int = 0
    N3: int = 0
    N4: int = 0
    M4: int = 0
    N_hash_eligible: int = 0
    N_hash_eq_terminal: int = 0
    N_hash_unique: int = 0
    N_Q: int = 0
    M_Q: int = 0
    N_INT: int = 0
    N_F: int = 0
    N_Glocal: int = 0
    N_Ggoal: int = 0
    dedup_alias_count: int = 0
    zero_motion_terminal_rows: int = 0


class CompactCensusAccumulator:
    """Accumulate the complete factual-origin census without origin objects.

    The ordinary :func:`census` API remains the exhaustive audit surface used by
    bounded fixtures.  Production already knows that every canonical trajectory
    contributes exactly one factual goal origin for each ``t < T``; this
    accumulator applies the same equations directly and retains only uniqueness
    gates plus one integer counter record per partition.
    """

    def __init__(self, *, audit_kind: str, max_horizon: int = 4) -> None:
        if audit_kind not in ("annotation_pre_audit", "production_post_rgb"):
            raise CensusError("unsupported census audit kind")
        if type(max_horizon) is not int or max_horizon != 4:
            raise CensusError("the frozen production horizon is exactly four")
        self._audit_kind = audit_kind
        self._max_horizon = max_horizon
        self._partitions: dict[str, _MutablePartitionCensus] = {}
        self._trajectory_keys: set[bytes] = set()
        self._dedup_keys: set[bytes] = set()
        self._scan_partitions: dict[str, str] = {}

    def add_complete_trajectory(
        self,
        *,
        partition: str,
        canonical_trajectory_key: bytes,
        dedup_key: bytes,
        scan_id: str,
        actions: tuple[int, ...],
        dedup_alias_count: int,
        zero_motion_terminal: bool,
    ) -> None:
        """Add one trajectory whose factual goal ledger is exactly ``range(T)``."""
        if type(partition) is not str or not partition:
            raise CensusError("partition must be a non-empty string")
        if (
            type(canonical_trajectory_key) is not bytes
            or len(canonical_trajectory_key) != 32
            or type(dedup_key) is not bytes
            or len(dedup_key) != 32
        ):
            raise CensusError("trajectory and dedup keys must be 32 raw bytes")
        if type(scan_id) is not str or not scan_id:
            raise CensusError("scan_id must be a non-empty string")
        if type(actions) is not tuple or not actions:
            raise CensusError("actions must be a full non-empty tuple")
        if type(actions[0]) is not int or actions[0] != -1:
            raise CensusError("actions must start with exact BOS -1")
        if any(
            type(action) is not int or action not in (1, 2, 3)
            for action in actions[1:]
        ):
            raise CensusError(
                "STOP is analytic; factual successor actions must be 1, 2, or 3"
            )
        if type(dedup_alias_count) is not int or dedup_alias_count < 0:
            raise CensusError("dedup_alias_count must be a non-negative exact integer")
        if type(zero_motion_terminal) is not bool:
            raise CensusError("zero_motion_terminal must be exact bool")
        if zero_motion_terminal != (actions == (-1,)):
            raise CensusError("zero-motion terminal flag disagrees with actions")
        if canonical_trajectory_key in self._trajectory_keys:
            raise CensusError("duplicate canonical trajectory key")
        if dedup_key in self._dedup_keys:
            raise CensusError("duplicate dedup key")
        self._trajectory_keys.add(canonical_trajectory_key)
        self._dedup_keys.add(dedup_key)
        prior_partition = self._scan_partitions.setdefault(scan_id, partition)
        if prior_partition != partition:
            raise CensusError("one scan cannot cross census partitions")

        terminal = len(actions) - 1
        horizons = tuple(
            max(terminal - horizon + 1, 0)
            for horizon in range(1, self._max_horizon + 1)
        )
        hash_eligible = horizons[3]
        hash_eq_terminal = 0
        hash_unique = 0
        for t in range(hash_eligible):
            try:
                goal = hashed_goal_index(
                    canonical_trajectory_key,
                    t=t,
                    terminal=terminal,
                    horizon=self._max_horizon,
                )
            except ValueError as exc:
                raise CensusError("invalid hashed goal origin") from exc
            if goal == terminal:
                hash_eq_terminal += 1
            else:
                hash_unique += 1

        values = self._partitions.setdefault(partition, _MutablePartitionCensus())
        values.trajectories += 1
        values.frames += terminal + 1
        values.N1 += horizons[0]
        values.N2 += horizons[1]
        values.N3 += horizons[2]
        values.N4 += horizons[3]
        values.M4 += sum(horizons)
        values.N_hash_eligible += hash_eligible
        values.N_hash_eq_terminal += hash_eq_terminal
        values.N_hash_unique += hash_unique
        values.N_Q += terminal + hash_unique
        values.M_Q += sum(horizons) + self._max_horizon * hash_unique
        values.N_INT += terminal + 1
        values.N_F += terminal
        values.N_Glocal += terminal
        values.N_Ggoal += terminal + 1
        values.dedup_alias_count += dedup_alias_count
        values.zero_motion_terminal_rows += int(zero_motion_terminal)

    def finish(self) -> ProductionCensus:
        partitions = tuple(
            PartitionCensus(partition=partition, **vars(self._partitions[partition]))
            for partition in sorted(self._partitions)
        )
        return ProductionCensus(
            audit_kind=self._audit_kind,
            max_horizon=self._max_horizon,
            partitions=partitions,
        )


def count_horizons(
    lengths: Iterable[int], *, max_horizon: int = 4
) -> dict[int, int]:
    """Count active factual suffixes from transition counts ``T``."""
    if type(max_horizon) is not int or max_horizon < 1:
        raise ValueError("max_horizon must be a positive integer")
    try:
        transition_counts = tuple(lengths)
    except TypeError as exc:
        raise ValueError("lengths must be iterable") from exc
    if any(type(length) is not int or length < 0 for length in transition_counts):
        raise ValueError("transition lengths must be non-negative exact integers")
    return {
        horizon: sum(max(length - horizon + 1, 0) for length in transition_counts)
        for horizon in range(1, max_horizon + 1)
    }


def unique_active_blocks(counts: Mapping[int, int]) -> int:
    """Return ``M4`` from the exact four horizon counts."""
    if not isinstance(counts, Mapping) or set(counts) != {1, 2, 3, 4}:
        raise ValueError("counts must contain exactly horizons 1 through 4")
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("horizon counts must be non-negative exact integers")
    return sum(counts[horizon] for horizon in range(1, 5))


def _validate_origin(origin: object) -> GoalViewOrigin:
    if not isinstance(origin, GoalViewOrigin):
        raise CensusError("goal origins must be GoalViewOrigin values")
    if type(origin.trajectory_key) is not bytes or len(origin.trajectory_key) != 32:
        raise CensusError("goal origin trajectory_key must be 32 raw bytes")
    if (
        type(origin.t) is not int
        or type(origin.terminal) is not int
        or not 0 <= origin.t < origin.terminal < 2**32
    ):
        raise CensusError("goal origin requires 0 <= t < terminal < 2**32")
    if type(origin.pre_eviction_support) is not bool:
        raise CensusError("pre_eviction_support must be exact bool")
    return origin


def goal_view_census(
    origins: Iterable[GoalViewOrigin], *, max_horizon: int = 4
) -> GoalViewCensus:
    """Count terminal and frozen-hash factual goal presentations."""
    if type(max_horizon) is not int or max_horizon != 4:
        raise CensusError("the frozen census horizon is exactly four")
    try:
        materialized = tuple(_validate_origin(origin) for origin in origins)
    except TypeError as exc:
        raise CensusError("origins must be iterable") from exc
    identities = {
        (origin.trajectory_key, origin.t, origin.terminal)
        for origin in materialized
    }
    if len(identities) != len(materialized):
        raise CensusError("goal origin ledger contains duplicates")

    eligible = 0
    eq_terminal = 0
    unique = 0
    m_4_unique = 0
    m_q_plus = 0
    for origin in sorted(
        materialized,
        key=lambda value: (value.trajectory_key, value.t, value.terminal),
    ):
        active_horizon = min(max_horizon, origin.terminal - origin.t)
        m_4_unique += active_horizon
        goal_views = 1
        if origin.terminal - origin.t >= max_horizon:
            eligible += 1
            try:
                goal = hashed_goal_index(
                    origin.trajectory_key,
                    t=origin.t,
                    terminal=origin.terminal,
                    horizon=max_horizon,
                )
            except ValueError as exc:
                raise CensusError("invalid hashed goal origin") from exc
            if goal == origin.terminal:
                eq_terminal += 1
            else:
                unique += 1
                goal_views += 1
        if origin.pre_eviction_support:
            m_q_plus += goal_views * active_horizon

    return GoalViewCensus(
        N_hash_eligible=eligible,
        N_hash_eq_terminal=eq_terminal,
        N_hash_unique=unique,
        N_Q=len(materialized) + unique,
        M_4_unique=m_4_unique,
        M_Q=m_4_unique + max_horizon * unique,
        M_Q_plus=m_q_plus,
    )


def _validate_trajectory(trajectory: object) -> tuple[CensusTrajectory, int]:
    if not isinstance(trajectory, CensusTrajectory):
        raise CensusError("canonical_trajectories must contain CensusTrajectory values")
    if type(trajectory.partition) is not str or not trajectory.partition:
        raise CensusError("partition must be a non-empty string")
    if (
        type(trajectory.canonical_trajectory_key) is not bytes
        or len(trajectory.canonical_trajectory_key) != 32
        or type(trajectory.dedup_key) is not bytes
        or len(trajectory.dedup_key) != 32
    ):
        raise CensusError("trajectory and dedup keys must be 32 raw bytes")
    if type(trajectory.scan_id) is not str or not trajectory.scan_id:
        raise CensusError("scan_id must be a non-empty string")
    if type(trajectory.actions) is not tuple or not trajectory.actions:
        raise CensusError("actions must be a full non-empty tuple")
    if type(trajectory.actions[0]) is not int or trajectory.actions[0] != -1:
        raise CensusError("actions must start with exact BOS -1")
    if any(
        type(action) is not int or action not in (1, 2, 3)
        for action in trajectory.actions[1:]
    ):
        raise CensusError("STOP is analytic; factual successor actions must be 1, 2, or 3")
    if (
        type(trajectory.dedup_alias_count) is not int
        or trajectory.dedup_alias_count < 0
    ):
        raise CensusError("dedup_alias_count must be a non-negative exact integer")
    if type(trajectory.zero_motion_terminal) is not bool:
        raise CensusError("zero_motion_terminal must be exact bool")
    is_zero_motion = trajectory.actions == (-1,)
    if trajectory.zero_motion_terminal != is_zero_motion:
        raise CensusError("zero-motion terminal flag disagrees with actions")
    if type(trajectory.goal_view_origins) is not tuple:
        raise CensusError("goal_view_origins must be a tuple")

    terminal = len(trajectory.actions) - 1
    seen_t: set[int] = set()
    for raw_origin in trajectory.goal_view_origins:
        origin = _validate_origin(raw_origin)
        if origin.trajectory_key != trajectory.canonical_trajectory_key:
            raise CensusError("goal origin trajectory key disagrees with its trajectory")
        if origin.terminal != terminal:
            raise CensusError("goal origin terminal disagrees with transition count")
        if origin.t in seen_t:
            raise CensusError("trajectory goal ledger contains duplicate origins")
        seen_t.add(origin.t)
    return trajectory, terminal


def census(
    canonical_trajectories: Iterable[CensusTrajectory],
    *,
    audit_kind: str,
    max_horizon: int = 4,
) -> ProductionCensus:
    """Compute exact per-partition H4 and factual branch counts."""
    if audit_kind not in ("annotation_pre_audit", "production_post_rgb"):
        raise CensusError("unsupported census audit kind")
    if type(max_horizon) is not int or max_horizon != 4:
        raise CensusError("the frozen production horizon is exactly four")
    try:
        materialized = tuple(canonical_trajectories)
    except TypeError as exc:
        raise CensusError("canonical_trajectories must be iterable") from exc

    by_partition: dict[str, list[tuple[CensusTrajectory, int]]] = {}
    trajectory_keys: set[bytes] = set()
    dedup_keys: set[bytes] = set()
    scan_partitions: dict[str, str] = {}
    for raw_trajectory in materialized:
        trajectory, terminal = _validate_trajectory(raw_trajectory)
        if trajectory.canonical_trajectory_key in trajectory_keys:
            raise CensusError("duplicate canonical trajectory key")
        if trajectory.dedup_key in dedup_keys:
            raise CensusError("duplicate dedup key")
        trajectory_keys.add(trajectory.canonical_trajectory_key)
        dedup_keys.add(trajectory.dedup_key)
        prior_partition = scan_partitions.setdefault(
            trajectory.scan_id, trajectory.partition
        )
        if prior_partition != trajectory.partition:
            raise CensusError("one scan cannot cross census partitions")
        expected_origins = set(range(terminal))
        actual_origins = {origin.t for origin in trajectory.goal_view_origins}
        if audit_kind == "production_post_rgb" and actual_origins != expected_origins:
            raise CensusError(
                "production goal ledger must contain exactly one origin for every t < T"
            )
        by_partition.setdefault(trajectory.partition, []).append((trajectory, terminal))

    partitions: list[PartitionCensus] = []
    for partition in sorted(by_partition):
        rows = sorted(
            by_partition[partition],
            key=lambda item: item[0].canonical_trajectory_key,
        )
        lengths = [terminal for _, terminal in rows]
        horizon_counts = count_horizons(lengths, max_horizon=max_horizon)
        origins = tuple(
            origin
            for trajectory, _ in rows
            for origin in trajectory.goal_view_origins
        )
        goals = goal_view_census(origins, max_horizon=max_horizon)
        n1 = horizon_counts[1]
        trajectories = len(rows)
        partitions.append(
            PartitionCensus(
                partition=partition,
                trajectories=trajectories,
                frames=sum(terminal + 1 for _, terminal in rows),
                N1=n1,
                N2=horizon_counts[2],
                N3=horizon_counts[3],
                N4=horizon_counts[4],
                M4=unique_active_blocks(horizon_counts),
                N_hash_eligible=goals.N_hash_eligible,
                N_hash_eq_terminal=goals.N_hash_eq_terminal,
                N_hash_unique=goals.N_hash_unique,
                N_Q=goals.N_Q,
                M_Q=goals.M_Q,
                N_INT=n1 + trajectories,
                N_F=n1,
                N_Glocal=n1,
                N_Ggoal=n1 + trajectories,
                dedup_alias_count=sum(
                    trajectory.dedup_alias_count for trajectory, _ in rows
                ),
                zero_motion_terminal_rows=sum(
                    int(trajectory.zero_motion_terminal) for trajectory, _ in rows
                ),
            )
        )
    return ProductionCensus(
        audit_kind=audit_kind,
        max_horizon=max_horizon,
        partitions=tuple(partitions),
    )
