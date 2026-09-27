"""One-owner factual trajectory planning for context-four joint training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from j2j.context4 import CONTEXT_SIZE
from j2j.data.horizon import build_factual_tape


@dataclass(frozen=True)
class TrajectoryDescriptor:
    dataset_index: int
    trajectory_key: bytes
    origin_count: int

    def __post_init__(self) -> None:
        if isinstance(self.dataset_index, bool) or not isinstance(self.dataset_index, int):
            raise TypeError("dataset_index must be an integer")
        if self.dataset_index < 0:
            raise ValueError("dataset_index must be non-negative")
        if type(self.trajectory_key) is not bytes or not self.trajectory_key:
            raise TypeError("trajectory_key must be nonempty bytes")
        if isinstance(self.origin_count, bool) or not isinstance(self.origin_count, int):
            raise TypeError("origin_count must be an integer")
        if self.origin_count < 0:
            raise ValueError("origin_count must be non-negative")


@dataclass(frozen=True)
class OriginSlice:
    trajectory_index: int
    start: int
    stop: int


@dataclass(frozen=True)
class TerminalOwner:
    trajectory_index: int
    rank: int
    update_index: int
    origin: int | None
    side_row: bool


@dataclass(frozen=True)
class MaterializedOriginSlice:
    trajectory_index: int
    start: int
    stop: int
    trajectory_item: object


@dataclass(frozen=True)
class RankOriginPlan:
    descriptors: tuple[TrajectoryDescriptor, ...]
    world_size: int
    effective_batch: int
    total_origins: int
    updates: tuple[tuple[tuple[OriginSlice, ...], ...], ...]
    owners: tuple[int, ...]
    terminals: tuple[TerminalOwner, ...]

    @property
    def update_count(self) -> int:
        return len(self.updates)

    def slices_for(self, rank: int, update_index: int) -> tuple[OriginSlice, ...]:
        if not 0 <= rank < self.world_size:
            raise IndexError("rank is outside the plan")
        if not 0 <= update_index < self.update_count:
            raise IndexError("update_index is outside the plan")
        return self.updates[update_index][rank]

    def owner_for(self, trajectory_index: int) -> int:
        return self.owners[trajectory_index]

    def terminal_for(self, trajectory_index: int) -> TerminalOwner:
        return self.terminals[trajectory_index]


def _validate_plan_inputs(
    descriptors: Sequence[TrajectoryDescriptor],
    world_size: object,
    effective_batch: object,
) -> tuple[tuple[TrajectoryDescriptor, ...], int, int]:
    if isinstance(descriptors, (str, bytes)) or not isinstance(descriptors, Sequence):
        raise TypeError("descriptors must be a sequence")
    values = tuple(descriptors)
    if not values or any(type(value) is not TrajectoryDescriptor for value in values):
        raise ValueError("descriptors must contain TrajectoryDescriptor values")
    if len({value.dataset_index for value in values}) != len(values):
        raise ValueError("dataset indices must be unique")
    if len({value.trajectory_key for value in values}) != len(values):
        raise ValueError("trajectory keys must be unique")
    for name, value in (("world_size", world_size), ("effective_batch", effective_batch)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    return values, int(world_size), int(effective_batch)


def build_rank_origin_plan(
    descriptors: Sequence[TrajectoryDescriptor],
    *,
    world_size: int,
    effective_batch: int,
) -> RankOriginPlan:
    """Assign whole trajectories once, then make balanced global update slices."""

    values, ranks, batch_size = _validate_plan_inputs(
        descriptors, world_size, effective_batch
    )
    total_origins = sum(value.origin_count for value in values)
    if total_origins <= 0:
        raise ValueError("an epoch must contain at least one factual origin")

    loads = [0] * ranks
    owned_counts = [0] * ranks
    owners = [-1] * len(values)
    placement_order = sorted(
        range(len(values)),
        key=lambda index: (-values[index].origin_count, values[index].trajectory_key, index),
    )
    for trajectory_index in placement_order:
        rank = min(range(ranks), key=lambda item: (loads[item], owned_counts[item], item))
        owners[trajectory_index] = rank
        loads[rank] += values[trajectory_index].origin_count
        owned_counts[rank] += 1

    rank_trajectories = [
        sorted(
            (index for index, owner in enumerate(owners) if owner == rank and values[index].origin_count),
            key=lambda index: (values[index].trajectory_key, index),
        )
        for rank in range(ranks)
    ]
    trajectory_cursor = [0] * ranks
    origin_cursor = [0] * ranks
    rank_remaining = [sum(values[index].origin_count for index in stream) for stream in rank_trajectories]
    last_origin_update: dict[int, int] = {}
    updates: list[tuple[tuple[OriginSlice, ...], ...]] = []
    remaining_global = total_origins

    def consume(rank: int, count: int, destination: list[OriginSlice], update_index: int) -> None:
        while count:
            stream = rank_trajectories[rank]
            pointer = trajectory_cursor[rank]
            trajectory_index = stream[pointer]
            descriptor = values[trajectory_index]
            start = origin_cursor[rank]
            take = min(count, descriptor.origin_count - start)
            stop = start + take
            destination.append(OriginSlice(trajectory_index, start, stop))
            if stop == descriptor.origin_count:
                last_origin_update[trajectory_index] = update_index
                trajectory_cursor[rank] += 1
                origin_cursor[rank] = 0
            else:
                origin_cursor[rank] = stop
            rank_remaining[rank] -= take
            count -= take

    while remaining_global:
        update_index = len(updates)
        target = min(batch_size, remaining_global)
        per_rank: list[list[OriginSlice]] = [[] for _ in range(ranks)]
        needed = target
        rotation = update_index % ranks
        while needed:
            active = [rank for rank in range(ranks) if rank_remaining[rank] > 0]
            if not active:
                raise RuntimeError("origin planner exhausted ranks before its global count")
            ordered = sorted(active, key=lambda rank: ((rank - rotation) % ranks, rank))
            share, extra = divmod(needed, len(ordered))
            progressed = 0
            for offset, rank in enumerate(ordered):
                requested = share + (1 if offset < extra else 0)
                if requested == 0:
                    requested = 1
                take = min(requested, rank_remaining[rank], needed)
                if take:
                    consume(rank, take, per_rank[rank], update_index)
                    needed -= take
                    progressed += take
                if not needed:
                    break
            if not progressed:
                raise RuntimeError("origin planner made no progress")
        updates.append(tuple(tuple(items) for items in per_rank))
        remaining_global -= target

    terminals = tuple(
        TerminalOwner(
            trajectory_index=index,
            rank=owners[index],
            update_index=(last_origin_update[index] if descriptor.origin_count else 0),
            origin=(descriptor.origin_count - 1 if descriptor.origin_count else None),
            side_row=descriptor.origin_count == 0,
        )
        for index, descriptor in enumerate(values)
    )
    plan = RankOriginPlan(
        descriptors=values,
        world_size=ranks,
        effective_batch=batch_size,
        total_origins=total_origins,
        updates=tuple(updates),
        owners=tuple(owners),
        terminals=terminals,
    )
    if plan.update_count != math.ceil(total_origins / batch_size):
        raise RuntimeError("origin update count is inconsistent")
    return plan


def iter_materialized_rank_slices(
    dataset: object,
    plan: RankOriginPlan,
    rank: int,
    *,
    prepare_trajectory: Callable[[object], object] | None = None,
) -> Iterator[tuple[int, tuple[MaterializedOriginSlice, ...]]]:
    """Yield rank slices while reading each owned trajectory at most once."""

    if type(plan) is not RankOriginPlan:
        raise TypeError("plan must be a RankOriginPlan")
    if not 0 <= rank < plan.world_size:
        raise ValueError("rank is outside the plan")
    getitem = getattr(dataset, "__getitem__", None)
    if not callable(getitem):
        raise TypeError("dataset must support indexed materialization")
    remaining_uses: dict[int, int] = {}
    for update_index in range(plan.update_count):
        for item in plan.slices_for(rank, update_index):
            remaining_uses[item.trajectory_index] = remaining_uses.get(item.trajectory_index, 0) + 1
    materialized: dict[int, object] = {}
    for update_index in range(plan.update_count):
        rows: list[MaterializedOriginSlice] = []
        for item in plan.slices_for(rank, update_index):
            trajectory_index = item.trajectory_index
            if trajectory_index not in materialized:
                dataset_index = plan.descriptors[trajectory_index].dataset_index
                value = dataset[dataset_index]
                materialized[trajectory_index] = (
                    prepare_trajectory(value)
                    if prepare_trajectory is not None
                    else value
                )
            rows.append(
                MaterializedOriginSlice(
                    trajectory_index=trajectory_index,
                    start=item.start,
                    stop=item.stop,
                    trajectory_item=materialized[trajectory_index],
                )
            )
            remaining_uses[trajectory_index] -= 1
            if remaining_uses[trajectory_index] == 0:
                # ``rows`` retains this exact object through the yielded update;
                # the iterator itself does not keep the completed trajectory.
                del materialized[trajectory_index]
        yield update_index, tuple(rows)


@dataclass(frozen=True)
class JointBatch:
    origin_indices: Tensor
    current_grid: Tensor
    next_grid: Tensor
    local_intent: Tensor
    previous_raw4: Tensor
    outgoing_raw4: Tensor
    action_ids: Tensor
    context_grid: Tensor
    context_incoming_raw4: Tensor
    context_outgoing_raw4: Tensor
    context_age: Tensor
    context_type: Tensor
    context_valid: Tensor
    q_goal_grid: Tensor
    q_target_grid: Tensor
    q_active_h: Tensor
    q_origin_row: Tensor
    goal_intent: Tensor
    terminal_grid: Tensor
    terminal_intent: Tensor
    terminal_previous_raw4: Tensor
    terminal_action_ids: Tensor
    q_sample_keys: tuple[bytes, ...] = ()


@dataclass(frozen=True)
class PreparedJointTrajectory:
    """One admitted trajectory with aligned actions computed exactly once."""

    grids: Tensor
    action_ids: Tensor
    incoming_raw4: Tensor
    outgoing_raw4: Tensor
    transform_spec_sha: str | None = None

    def pin_memory(self) -> "PreparedJointTrajectory":
        """Let ``DataLoader(pin_memory=True)`` pin this custom payload."""

        return PreparedJointTrajectory(
            grids=self.grids.pin_memory(),
            action_ids=self.action_ids.pin_memory(),
            incoming_raw4=self.incoming_raw4.pin_memory(),
            outgoing_raw4=self.outgoing_raw4.pin_memory(),
            transform_spec_sha=self.transform_spec_sha,
        )


def _raw4_from_action_ids(action_ids: Tensor, frame_count: int) -> tuple[Tensor, Tensor]:
    outgoing = torch.zeros((frame_count, 4), dtype=torch.float32, device=action_ids.device)
    if action_ids.numel():
        outgoing[:-1].scatter_(1, action_ids[:, None], 1.0)
    incoming = torch.zeros_like(outgoing)
    if action_ids.numel():
        incoming[1:] = outgoing[:-1]
    return incoming, outgoing


def prepare_joint_trajectory(trajectory_item: object) -> PreparedJointTrajectory:
    """Validate and align one trajectory once before any origin slicing."""

    if type(trajectory_item) is PreparedJointTrajectory:
        return trajectory_item
    grids = getattr(trajectory_item, "grids", None)
    action_ids = getattr(trajectory_item, "action_ids", None)
    if not isinstance(grids, Tensor) or grids.ndim != 3 or grids.dtype != torch.float32:
        raise TypeError("trajectory grids must be float32 [T+1,spatial,latent]")
    if not isinstance(action_ids, Tensor) or action_ids.dtype != torch.int64 or action_ids.ndim != 1:
        raise TypeError("trajectory action_ids must be int64 [T]")
    if grids.shape[0] != action_ids.shape[0] + 1:
        raise ValueError("trajectory frame/action lengths are inconsistent")
    if grids.device != action_ids.device:
        raise ValueError("trajectory grids and actions must share one device")
    if grids.requires_grad or action_ids.requires_grad:
        raise ValueError("trajectory facts must be detached")
    if not bool(torch.isfinite(grids).all()):
        raise ValueError("trajectory grids must be finite")
    if action_ids.numel() and not bool(((action_ids >= 1) & (action_ids <= 3)).all()):
        raise ValueError("trajectory actions must be FWD, LEFT, or RIGHT")
    grids = grids.detach()
    action_ids = action_ids.detach()
    incoming, outgoing = _raw4_from_action_ids(action_ids, grids.shape[0])
    return PreparedJointTrajectory(
        grids=grids,
        action_ids=action_ids,
        incoming_raw4=incoming.detach(),
        outgoing_raw4=outgoing.detach(),
    )


def _materialize_context_windows(
    prepared: PreparedJointTrajectory,
    origin_indices: Tensor,
    *,
    context_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Vectorize the masked causal last-four gather from admitted facts."""

    grids = prepared.grids
    batch = int(origin_indices.numel())
    spatial_shape = tuple(grids.shape[1:])
    if batch == 0:
        return (
            grids.new_empty((0, context_size, *spatial_shape)),
            grids.new_empty((0, context_size, 4)),
            grids.new_empty((0, context_size, 4)),
            torch.empty((0, context_size), dtype=torch.int64, device=grids.device),
            torch.empty((0, context_size), dtype=torch.int64, device=grids.device),
            torch.empty((0, context_size), dtype=torch.bool, device=grids.device),
        )

    offsets = torch.arange(
        1 - context_size,
        1,
        dtype=torch.int64,
        device=origin_indices.device,
    )
    indices = origin_indices[:, None] + offsets[None]
    valid = indices >= 0
    safe_indices = indices.clamp_min(0).reshape(-1)
    context_grid = grids.index_select(0, safe_indices).reshape(
        batch, context_size, *spatial_shape
    )
    incoming = prepared.incoming_raw4.index_select(0, safe_indices).reshape(
        batch, context_size, 4
    )
    outgoing = prepared.outgoing_raw4.index_select(0, safe_indices).reshape(
        batch, context_size, 4
    )
    context_grid.masked_fill_(~valid[:, :, None, None], 0.0)
    incoming = incoming.masked_fill(~valid[:, :, None], 0.0)
    outgoing = outgoing.masked_fill(~valid[:, :, None], 0.0)

    ages = (-offsets)[None].expand(batch, -1).masked_fill(~valid, 0)
    kinds = torch.ones_like(ages).masked_fill(~valid, 0)
    kinds[:, -1] = 2
    return context_grid, incoming, outgoing, ages, kinds, valid


def _gather_factual_tapes(
    grids: Tensor, q_origin_indices: Tensor, goal_indices: Tensor, *, horizon: int
) -> tuple[Tensor, Tensor]:
    """Gather validated factual occurrences once; never mutate source grids."""
    count = q_origin_indices.numel()
    if not count:
        return (
            grids.new_empty((0, horizon, *grids.shape[1:])),
            torch.empty((0, horizon), dtype=torch.bool, device=grids.device),
        )
    steps = torch.arange(1, horizon + 1, device=grids.device)
    indices = q_origin_indices.to(device=grids.device)[:, None] + steps[None]
    goals = goal_indices.to(device=grids.device)[:, None]
    active = indices <= goals
    safe_indices = torch.minimum(indices, goals)
    targets = grids.index_select(0, safe_indices.reshape(-1)).reshape(
        count, horizon, *grids.shape[1:]
    ).detach()
    targets.masked_fill_(~active[:, :, None, None], 0.0)
    return targets, active


def materialize_joint_batch(
    trajectory_item: object,
    *,
    origin_indices: Tensor,
    goal_indices: Tensor,
    q_origin_indices: Tensor | None = None,
    horizon: int,
    context_size: int = CONTEXT_SIZE,
    include_terminal_stop: bool = False,
    loss_weights: Mapping[str, float] | None = None,
) -> JointBatch:
    """Derive all factual rows from an already materialized trajectory object."""

    if loss_weights is None:
        local_enabled = goal_enabled = True
    else:
        from j2j.context4_objective import _loss_weight_values
        _, _, local_weight, goal_weight, _ = _loss_weight_values(loss_weights)
        local_enabled, goal_enabled = bool(local_weight), bool(goal_weight)
    prepared = prepare_joint_trajectory(trajectory_item)
    grids = prepared.grids
    action_ids = prepared.action_ids
    spatial_shape = tuple(grids.shape[1:])
    if q_origin_indices is None:
        q_origin_indices = origin_indices
    if any(
        not isinstance(value, Tensor) or value.dtype != torch.int64
        for value in (origin_indices, q_origin_indices, goal_indices)
    ):
        raise TypeError("origin_indices, q_origin_indices and goal_indices must be int64")
    if origin_indices.ndim != 1 or q_origin_indices.ndim != 1 or goal_indices.ndim != 1:
        raise ValueError("origin and goal indices must be vectors")
    if q_origin_indices.shape != goal_indices.shape:
        raise ValueError("Q origin occurrences and goals must align")
    if len(set(origin_indices.tolist())) != origin_indices.numel():
        raise ValueError("factual origin rows must be unique")
    if not isinstance(include_terminal_stop, bool):
        raise TypeError("include_terminal_stop must be a bool")
    if not origin_indices.numel() and (q_origin_indices.numel() or not include_terminal_stop):
        raise ValueError("an empty factual batch is allowed only for a terminal side row")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be positive")
    if context_size != CONTEXT_SIZE:
        raise ValueError("context_size must equal four")
    if origin_indices.numel() and (
        bool((origin_indices < 0).any()) or bool((origin_indices >= action_ids.shape[0]).any())
    ):
        raise ValueError("origin indices must identify factual transitions")
    origin_to_row = {origin: row for row, origin in enumerate(origin_indices.tolist())}
    if any(origin not in origin_to_row for origin in q_origin_indices.tolist()):
        raise ValueError("each Q occurrence must refer to one factual origin row")
    if goal_indices.numel() and (
        bool((goal_indices <= q_origin_indices).any())
        or bool((goal_indices >= grids.shape[0]).any())
    ):
        raise ValueError("each goal must be a later factual frame")

    incoming = prepared.incoming_raw4
    outgoing = prepared.outgoing_raw4
    q_targets, q_active = _gather_factual_tapes(
        grids, q_origin_indices, goal_indices, horizon=horizon
    )

    current = grids.index_select(0, origin_indices).detach()
    next_grid = grids.index_select(0, origin_indices + 1).detach()
    selected_actions = action_ids.index_select(0, origin_indices).detach()
    previous = incoming.index_select(0, origin_indices).detach()
    outgoing_current = outgoing.index_select(0, origin_indices).detach()
    goals = grids.index_select(0, goal_indices).detach()
    (
        context_grid,
        context_incoming,
        context_outgoing,
        context_age,
        context_type,
        context_valid,
    ) = _materialize_context_windows(
        prepared,
        origin_indices,
        context_size=context_size,
    )
    q_origin_row = torch.tensor(
        [origin_to_row[origin] for origin in q_origin_indices.tolist()],
        dtype=torch.int64,
        device=grids.device,
    )
    if include_terminal_stop:
        terminal_grid = grids[-1:].detach()
        terminal_intent = torch.zeros_like(terminal_grid)
        terminal_previous = incoming[-1:].detach()
        terminal_actions = torch.zeros(1, dtype=torch.int64, device=grids.device)
    else:
        terminal_grid = grids.new_empty((0, *spatial_shape))
        terminal_intent = grids.new_empty((0, *spatial_shape))
        terminal_previous = grids.new_empty((0, 4))
        terminal_actions = torch.empty(0, dtype=torch.int64, device=grids.device)
    return JointBatch(
        origin_indices=origin_indices.detach().clone(),
        current_grid=current,
        next_grid=next_grid,
        local_intent=(next_grid - current).detach() if local_enabled else current[:0],
        previous_raw4=previous,
        outgoing_raw4=outgoing_current,
        action_ids=selected_actions,
        context_grid=context_grid,
        context_incoming_raw4=context_incoming,
        context_outgoing_raw4=context_outgoing,
        context_age=context_age,
        context_type=context_type,
        context_valid=context_valid,
        q_goal_grid=goals,
        q_target_grid=q_targets,
        q_active_h=q_active,
        q_origin_row=q_origin_row,
        goal_intent=(grids[-1:].expand_as(current) - current).detach() if goal_enabled else current[:0],
        terminal_grid=terminal_grid,
        terminal_intent=terminal_intent,
        terminal_previous_raw4=terminal_previous,
        terminal_action_ids=terminal_actions,
    )
