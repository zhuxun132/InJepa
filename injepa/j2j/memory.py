"""Causal factual memory for JEPA-to-JEPA ImageNav."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .adapter import ActionId, Raw4Adapter
from .data.keys import frame_key as canonical_frame_key


_WARM_BANK_DOMAIN = b"J2J_WARM_BANK_V1\x00"
_MAX_EXACT_FLOAT32_INTEGER = 2**24
_MOTION_ACTIONS = (ActionId.FWD, ActionId.LEFT, ActionId.RIGHT)


def _require_nonnegative_int(value: object, *, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_finite_float32_tensor(
    value: object,
    *,
    name: str,
    rank: int,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype float32")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")
    return value


def _raw4_matches(raw4: Tensor, expected: Tensor) -> bool:
    return torch.equal(raw4, expected.to(device=raw4.device))


@dataclass(frozen=True)
class MemoryConfig:
    """Capacity and protected-recent-window settings for factual memory."""

    capacity: int
    recent_window: int = 2

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.capacity, name="capacity")
        _require_nonnegative_int(self.recent_window, name="recent_window")


@dataclass(frozen=True)
class FactualRecord:
    """One real observation and the real action that produced it."""

    grid: Tensor
    incoming_raw4: Tensor
    time: int
    frame_key: bytes
    trajectory_key: bytes

    def __post_init__(self) -> None:
        grid = _require_finite_float32_tensor(
            self.grid,
            name="grid",
            rank=2,
        )
        incoming_raw4 = _require_finite_float32_tensor(
            self.incoming_raw4,
            name="incoming_raw4",
            rank=1,
            shape=(4,),
        )
        _require_nonnegative_int(self.time, name="time")

        if type(self.frame_key) is not bytes:
            raise TypeError("frame_key must be exact bytes")
        if len(self.frame_key) != 36:
            raise ValueError("frame_key must contain exactly 36 bytes")
        if type(self.trajectory_key) is not bytes:
            raise TypeError("trajectory_key must be exact bytes")
        if len(self.trajectory_key) != 32:
            raise ValueError("trajectory_key must contain exactly 32 bytes")
        if self.frame_key != canonical_frame_key(self.trajectory_key, self.time):
            raise ValueError("frame_key is not canonical for trajectory_key and time")

        if self.time == 0:
            if not _raw4_matches(incoming_raw4, Raw4Adapter.encode_bos()):
                raise ValueError("time zero requires the BOS raw4 encoding")
        elif not any(
            _raw4_matches(incoming_raw4, Raw4Adapter.encode(action))
            for action in _MOTION_ACTIONS
        ):
            raise ValueError("positive-time records require a motion raw4 encoding")

        object.__setattr__(self, "grid", grid.detach().clone())
        object.__setattr__(
            self,
            "incoming_raw4",
            incoming_raw4.detach().clone(),
        )


def _validate_record(record: object, *, name: str) -> FactualRecord:
    if not isinstance(record, FactualRecord):
        raise TypeError(f"{name} must be a FactualRecord")

    _require_finite_float32_tensor(record.grid, name=f"{name}.grid", rank=2)
    incoming_raw4 = _require_finite_float32_tensor(
        record.incoming_raw4,
        name=f"{name}.incoming_raw4",
        rank=1,
        shape=(4,),
    )
    _require_nonnegative_int(record.time, name=f"{name}.time")
    if type(record.frame_key) is not bytes:
        raise TypeError(f"{name}.frame_key must be exact bytes")
    if len(record.frame_key) != 36:
        raise ValueError(f"{name}.frame_key must contain exactly 36 bytes")
    if type(record.trajectory_key) is not bytes:
        raise TypeError(f"{name}.trajectory_key must be exact bytes")
    if len(record.trajectory_key) != 32:
        raise ValueError(
            f"{name}.trajectory_key must contain exactly 32 bytes"
        )
    if record.frame_key != canonical_frame_key(
        record.trajectory_key,
        record.time,
    ):
        raise ValueError(f"{name}.frame_key is not canonical")
    if record.time == 0:
        valid_action = _raw4_matches(
            incoming_raw4,
            Raw4Adapter.encode_bos(),
        )
    else:
        valid_action = any(
            _raw4_matches(incoming_raw4, Raw4Adapter.encode(action))
            for action in _MOTION_ACTIONS
        )
    if not valid_action:
        raise ValueError(f"{name}.incoming_raw4 is not causally valid")
    return record


def _require_record_tuple(value: object, *, name: str) -> tuple[FactualRecord, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    for index, record in enumerate(value):
        _validate_record(record, name=f"{name}[{index}]")
    return value


def _validate_ordered_records(
    groups: tuple[tuple[FactualRecord, ...], ...],
    current: FactualRecord,
) -> None:
    records = tuple(record for group in groups for record in group) + (current,)
    trajectory_key = current.trajectory_key
    if any(record.trajectory_key != trajectory_key for record in records):
        raise ValueError("all memory records must belong to one trajectory")

    frame_keys = tuple(record.frame_key for record in records)
    if len(set(frame_keys)) != len(frame_keys):
        raise ValueError("memory record frame keys must be unique")
    if any(
        previous.time >= following.time
        for previous, following in zip(records, records[1:])
    ):
        raise ValueError("memory record times must be strictly increasing")


@dataclass(frozen=True)
class FactualMemory:
    """Steady factual state split into bank, protected recent, and current."""

    config: MemoryConfig
    bank: tuple[FactualRecord, ...]
    recent: tuple[FactualRecord, ...]
    current: FactualRecord

    def __post_init__(self) -> None:
        if not isinstance(self.config, MemoryConfig):
            raise TypeError("config must be a MemoryConfig")
        bank = _require_record_tuple(self.bank, name="bank")
        recent = _require_record_tuple(self.recent, name="recent")
        current = _validate_record(self.current, name="current")
        if len(bank) > self.config.capacity:
            raise ValueError("bank exceeds configured capacity")
        if len(recent) > self.config.recent_window:
            raise ValueError("recent exceeds configured recent_window")
        _validate_ordered_records((bank, recent), current)


@dataclass(frozen=True)
class PreEvictionView:
    """Transient factual view presented to an eviction scorer."""

    pool: tuple[FactualRecord, ...]
    recent: tuple[FactualRecord, ...]
    current: FactualRecord

    def __post_init__(self) -> None:
        pool = _require_record_tuple(self.pool, name="pool")
        recent = _require_record_tuple(self.recent, name="recent")
        current = _validate_record(self.current, name="current")
        _validate_ordered_records((pool, recent), current)


@dataclass(frozen=True)
class MemoryUpdate:
    """One causal memory update and its optional transient eviction evidence."""

    memory: FactualMemory
    eviction_view: PreEvictionView | None
    evicted_frame_key: bytes | None

    def __post_init__(self) -> None:
        if not isinstance(self.memory, FactualMemory):
            raise TypeError("memory must be a FactualMemory")
        if self.eviction_view is not None and not isinstance(
            self.eviction_view,
            PreEvictionView,
        ):
            raise TypeError("eviction_view must be a PreEvictionView or None")
        if self.evicted_frame_key is not None:
            if type(self.evicted_frame_key) is not bytes:
                raise TypeError("evicted_frame_key must be exact bytes or None")
            if len(self.evicted_frame_key) != 36:
                raise ValueError(
                    "evicted_frame_key must contain exactly 36 bytes"
                )


def start_memory(record: FactualRecord, config: MemoryConfig) -> FactualMemory:
    """Start a new trajectory-local memory from its real BOS observation."""
    record = _validate_record(record, name="record")
    if not isinstance(config, MemoryConfig):
        raise TypeError("config must be a MemoryConfig")
    if record.time != 0:
        raise ValueError("memory must start at trajectory time zero")
    return FactualMemory(config=config, bank=(), recent=(), current=record)


def update_after_observation(
    memory: FactualMemory,
    observed: FactualRecord,
    *,
    score_fn: Callable[[PreEvictionView], Tensor] | None = None,
) -> MemoryUpdate:
    """Advance memory by one real observation and evict at most one old fact."""
    if not isinstance(memory, FactualMemory):
        raise TypeError("memory must be a FactualMemory")
    memory.__post_init__()
    observed = _validate_record(observed, name="observed")
    if observed.trajectory_key != memory.current.trajectory_key:
        raise ValueError("observed record belongs to a different trajectory")
    if observed.time != memory.current.time + 1:
        raise ValueError("observed record must be the next trajectory time")

    recent = memory.recent + (memory.current,)
    departed: FactualRecord | None = None
    if len(recent) > memory.config.recent_window:
        departed = recent[0]
        recent = recent[1:]

    if departed is None:
        updated = FactualMemory(
            config=memory.config,
            bank=memory.bank,
            recent=recent,
            current=observed,
        )
        return MemoryUpdate(updated, None, None)

    pool = tuple(sorted(memory.bank + (departed,), key=lambda item: item.time))
    if memory.config.capacity == 0:
        updated = FactualMemory(
            config=memory.config,
            bank=(),
            recent=recent,
            current=observed,
        )
        return MemoryUpdate(updated, None, departed.frame_key)

    if len(pool) <= memory.config.capacity:
        updated = FactualMemory(
            config=memory.config,
            bank=pool,
            recent=recent,
            current=observed,
        )
        return MemoryUpdate(updated, None, None)

    if len(pool) != memory.config.capacity + 1:
        raise ValueError("causal update produced an invalid eviction pool size")
    if score_fn is None or not callable(score_fn):
        raise TypeError("score_fn is required when the factual pool is full")

    eviction_view = PreEvictionView(
        pool=pool,
        recent=recent,
        current=observed,
    )
    scores = score_fn(eviction_view)
    scores = _require_finite_float32_tensor(
        scores,
        name="scores",
        rank=1,
        shape=(len(pool),),
    )
    evicted_index = min(
        range(len(pool)),
        key=lambda index: (
            scores[index].item(),
            pool[index].time,
            pool[index].frame_key,
        ),
    )
    evicted = pool[evicted_index]
    retained = pool[:evicted_index] + pool[evicted_index + 1 :]
    updated = FactualMemory(
        config=memory.config,
        bank=retained,
        recent=recent,
        current=observed,
    )
    return MemoryUpdate(updated, eviction_view, evicted.frame_key)


def warm_hash_scores(view: PreEvictionView) -> Tensor:
    """Return exact float32 reverse ranks of the frozen warm-bank digests."""
    if not isinstance(view, PreEvictionView):
        raise TypeError("view must be a PreEvictionView")
    view.__post_init__()

    digests = tuple(
        hashlib.sha256(
            _WARM_BANK_DOMAIN
            + record.trajectory_key
            + struct.pack("<I", record.time)
        ).digest()
        for record in view.pool
    )
    if len(set(digests)) != len(digests):
        raise ValueError("warm-bank digest collision within eviction pool")

    maximum_reverse_rank = len(digests) - 1
    if maximum_reverse_rank > _MAX_EXACT_FLOAT32_INTEGER:
        raise ValueError("warm-bank reverse rank is not exact in float32")

    order = sorted(range(len(digests)), key=digests.__getitem__)
    scores = torch.empty(len(digests), dtype=torch.float32)
    for rank, record_index in enumerate(order):
        scores[record_index] = maximum_reverse_rank - rank
    return scores
