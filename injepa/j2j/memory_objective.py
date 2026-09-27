"""Factual-memory objectives for JEPA-to-JEPA ImageNav."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch
from torch import Tensor, nn

from .adapter import ActionId, Raw4Adapter
from .data.annotations import AnnotationRow
from .data.goals import hashed_goal_index
from .data.keys import frame_key, trajectory_key
from .memory import FactualMemory, FactualRecord, PreEvictionView
from .proposal import ProperMixtureTerms, ProposalJEPA, proper_mixture_terms


_DELETE_SAMPLE_DOMAIN = b"J2J_DELETE_SAMPLE_V1\x00"
_VIEW_KINDS = frozenset(("steady", "full", "delete"))
_PRECISIONS = frozenset(("32-true", "bf16-mixed"))
_UINT32_LIMIT = 2**32
_BOS_RAW4 = (0.0, 0.0, 0.0, 0.0)
_MOTION_RAW4 = frozenset(
    (
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)


@dataclass(frozen=True)
class FactualViewKey:
    trajectory_key: bytes
    origin_time: int
    current_frame_key: bytes
    ordered_record_frame_keys: tuple[bytes, ...]
    view_kind: str
    deleted_frame_key: bytes | None


@dataclass(frozen=True)
class FactualObjectiveKey:
    view: FactualViewKey
    terminal_time: int
    goal_time: int
    goal_frame_key: bytes
    target_frame_keys: tuple[bytes, ...]


@dataclass(frozen=True)
class FactualModelBatch:
    record_grid: Tensor
    incoming_action_embedding: Tensor
    record_age: Tensor
    record_type: Tensor
    record_valid: Tensor
    pool_mask: Tensor
    view_keys: tuple[FactualViewKey, ...]
    padded_record_frame_keys: tuple[tuple[bytes | None, ...], ...]


@dataclass(frozen=True)
class FactualObjectiveBatch:
    facts: FactualModelBatch
    keys: tuple[FactualObjectiveKey, ...]
    goal_grid: Tensor
    target: Tensor
    active_h: Tensor


@dataclass(frozen=True)
class DeletionUtilities:
    origin_keys: tuple[tuple[bytes, int], ...]
    full_view_keys: tuple[FactualViewKey, ...]
    deleted_frame_keys: tuple[tuple[bytes, ...], ...]
    values: tuple[Tensor, ...]


class ProposalTrainMode(Enum):
    Q0 = "q0"
    Q_CONTINUE = "q-continue"
    SELECTOR = "selector"
    READ_ONLY = "read-only"


def _require_uint32(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if not 0 <= value < _UINT32_LIMIT:
        raise ValueError(f"{name} must be an unsigned 32-bit integer")
    return value


def _require_bytes(
    value: object,
    *,
    name: str,
    length: int,
) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{name} must be exact bytes")
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} bytes")
    return value


def _frame_time(value: bytes) -> int:
    return struct.unpack("<I", value[32:])[0]


def _require_canonical_frame(
    value: object,
    *,
    trajectory: bytes,
    time: int,
    name: str,
) -> bytes:
    raw = _require_bytes(value, name=name, length=36)
    if raw != frame_key(trajectory, time):
        raise ValueError(f"{name} is not canonical for its trajectory and time")
    return raw


def _require_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    rank: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if value.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}")
    return value


def _require_finite(value: Tensor, *, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must be finite")


def _require_zero(value: Tensor, *, name: str) -> None:
    if torch.count_nonzero(value).item() != 0:
        raise ValueError(f"{name} must be exactly zero")


def _validate_view_key(view: object) -> FactualViewKey:
    if not isinstance(view, FactualViewKey):
        raise TypeError("view key must be a FactualViewKey")
    trajectory = _require_bytes(
        view.trajectory_key,
        name="view trajectory_key",
        length=32,
    )
    origin = _require_uint32(view.origin_time, name="view origin_time")
    _require_canonical_frame(
        view.current_frame_key,
        trajectory=trajectory,
        time=origin,
        name="view current_frame_key",
    )
    if type(view.ordered_record_frame_keys) is not tuple:
        raise TypeError("ordered_record_frame_keys must be a tuple")
    if not view.ordered_record_frame_keys:
        raise ValueError("a factual view must contain its current record")

    times: list[int] = []
    for index, key in enumerate(view.ordered_record_frame_keys):
        raw = _require_bytes(
            key,
            name=f"ordered_record_frame_keys[{index}]",
            length=36,
        )
        if raw[:32] != trajectory:
            raise ValueError("ordered factual keys must share one trajectory")
        time = _frame_time(raw)
        if raw != frame_key(trajectory, time):
            raise ValueError("ordered factual frame key is not canonical")
        times.append(time)
    if len(set(view.ordered_record_frame_keys)) != len(
        view.ordered_record_frame_keys
    ):
        raise ValueError("ordered factual frame keys must be unique")
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("ordered factual frame keys must increase in time")
    if view.ordered_record_frame_keys[-1] != view.current_frame_key:
        raise ValueError("the current frame must be the last factual key")

    if type(view.view_kind) is not str or view.view_kind not in _VIEW_KINDS:
        raise ValueError("view_kind must be steady, full, or delete")
    if view.view_kind in {"steady", "full"}:
        if view.deleted_frame_key is not None:
            raise ValueError("steady and full views cannot declare a deletion")
    else:
        deleted = _require_bytes(
            view.deleted_frame_key,
            name="deleted_frame_key",
            length=36,
        )
        if deleted[:32] != trajectory:
            raise ValueError("deleted record must share the view trajectory")
        deleted_time = _frame_time(deleted)
        if deleted != frame_key(trajectory, deleted_time):
            raise ValueError("deleted_frame_key is not canonical")
        if deleted_time >= origin:
            raise ValueError("a deleted record must be strictly in the past")
        if deleted in view.ordered_record_frame_keys:
            raise ValueError("a delete view still contains its deleted record")
    return view


def _records_for_state(
    state: object,
) -> tuple[
    tuple[FactualRecord, ...],
    tuple[FactualRecord, ...],
    FactualRecord,
    str,
]:
    if isinstance(state, FactualMemory):
        return state.bank, state.recent, state.current, "steady"
    if isinstance(state, PreEvictionView):
        return state.pool, state.recent, state.current, "pre-eviction"
    raise TypeError("factual state must be FactualMemory or PreEvictionView")


def _validate_state_identity(
    view: FactualViewKey,
    state: FactualMemory | PreEvictionView,
) -> tuple[tuple[FactualRecord, ...], tuple[int, ...]]:
    pool, recent, current, state_kind = _records_for_state(state)
    if state_kind == "steady" and view.view_kind != "steady":
        raise ValueError("FactualMemory requires a steady view key")
    if state_kind == "pre-eviction" and view.view_kind not in {"full", "delete"}:
        raise ValueError("PreEvictionView requires a full or delete view key")
    if view.trajectory_key != current.trajectory_key:
        raise ValueError("view and factual state trajectories do not match")
    if view.origin_time != current.time:
        raise ValueError("view origin does not match the current fact")
    if view.current_frame_key != current.frame_key:
        raise ValueError("view current key does not match the factual state")

    records = pool + recent + (current,)
    expected_keys = tuple(record.frame_key for record in records)
    if view.ordered_record_frame_keys != expected_keys:
        raise ValueError("view key does not bind the complete factual state")
    kinds = (0,) * len(pool) + (1,) * len(recent) + (2,)
    return records, kinds


def _validate_record_for_collation(record: object, *, name: str) -> FactualRecord:
    if not isinstance(record, FactualRecord):
        raise TypeError(f"{name} must be a FactualRecord")
    grid = _require_tensor(
        record.grid,
        name=f"{name}.grid",
        dtype=torch.float32,
        rank=2,
    )
    raw4 = _require_tensor(
        record.incoming_raw4,
        name=f"{name}.incoming_raw4",
        dtype=torch.float32,
        rank=1,
    )
    if tuple(raw4.shape) != (4,):
        raise ValueError(f"{name}.incoming_raw4 must have shape [4]")
    _require_finite(grid, name=f"{name}.grid")
    _require_finite(raw4, name=f"{name}.incoming_raw4")
    if grid.requires_grad or raw4.requires_grad:
        raise ValueError(f"{name} factual tensors must be detached")
    _require_uint32(record.time, name=f"{name}.time")
    trajectory = _require_bytes(
        record.trajectory_key,
        name=f"{name}.trajectory_key",
        length=32,
    )
    _require_canonical_frame(
        record.frame_key,
        trajectory=trajectory,
        time=record.time,
        name=f"{name}.frame_key",
    )
    raw_values = tuple(float(value) for value in raw4.detach().cpu().tolist())
    expected = _BOS_RAW4 if record.time == 0 else _MOTION_RAW4
    if record.time == 0:
        valid_raw = raw_values == expected
    else:
        valid_raw = raw_values in expected
    if not valid_raw:
        raise ValueError(f"{name}.incoming_raw4 is not a causal factual action")
    return record


def _resolve_action_encoder_device(
    action_encoder: object,
    requested: torch.device | str,
) -> tuple[nn.Module, torch.device]:
    if not isinstance(action_encoder, nn.Module):
        raise TypeError("action_encoder must be the loaded nn.Module")
    parameters = tuple(action_encoder.parameters())
    if not parameters:
        raise ValueError("action_encoder must expose its loaded parameters")
    if action_encoder.training:
        raise ValueError("action_encoder must already be in eval mode")
    if any(parameter.requires_grad for parameter in parameters):
        raise ValueError("action_encoder parameters must already be frozen")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("action_encoder parameters must share one actual device")
    actual = next(iter(devices))
    try:
        requested_device = torch.device(requested)
    except (TypeError, RuntimeError) as exc:
        raise TypeError("device must be a valid torch device request") from exc
    if requested_device.type != actual.type:
        raise ValueError("requested device type does not match action_encoder")
    if (
        requested_device.index is not None
        and actual.index is not None
        and requested_device.index != actual.index
    ):
        raise ValueError("explicit device index does not match action_encoder")
    return action_encoder, actual


def build_factual_records(
    row: AnnotationRow,
    *,
    annotation_revision_sha256: str,
    grids_by_frame_key: Mapping[bytes, Tensor | object],
) -> tuple[FactualRecord, ...]:
    """Construct the unique causal factual records for one annotation row."""
    if not isinstance(row, AnnotationRow):
        raise TypeError("row must be an AnnotationRow")
    if not isinstance(grids_by_frame_key, Mapping):
        raise TypeError("grids_by_frame_key must be a mapping")
    key = trajectory_key(
        row,
        annotation_revision_sha256=annotation_revision_sha256,
    )
    expected_keys = tuple(
        frame_key(key, time) for time in range(len(row.actions))
    )
    supplied_keys = tuple(grids_by_frame_key.keys())
    for index, supplied in enumerate(supplied_keys):
        _require_bytes(
            supplied,
            name=f"grids_by_frame_key key {index}",
            length=36,
        )
    if len(supplied_keys) != len(expected_keys) or set(supplied_keys) != set(
        expected_keys
    ):
        raise ValueError("cache frame keys must exactly match the trajectory")

    records: list[FactualRecord] = []
    for time, expected_key in enumerate(expected_keys):
        if time == 0:
            incoming = Raw4Adapter.encode_bos()
        else:
            action = ActionId(row.actions[time])
            if action is ActionId.STOP:
                raise ValueError("STOP does not produce a visual successor")
            incoming = Raw4Adapter.encode(action)
        grid = torch.as_tensor(grids_by_frame_key[expected_key])
        records.append(
            FactualRecord(
                grid=grid,
                incoming_raw4=incoming,
                time=time,
                frame_key=expected_key,
                trajectory_key=key,
            )
        )
    return tuple(records)


def sample_deletion_records(
    pool: Sequence[FactualRecord],
    *,
    origin_time: int,
    q: int,
) -> tuple[FactualRecord, ...]:
    """Select the frozen nested prefix of factual deletion records."""
    origin = _require_uint32(origin_time, name="origin_time")
    if type(q) is not int:
        raise TypeError("q must be an exact integer")
    if q < 1:
        raise ValueError("q must be positive")
    try:
        records = tuple(pool)
    except TypeError as exc:
        raise TypeError("pool must be a finite sequence") from exc
    if not records:
        raise ValueError("pool must be nonempty")
    if any(not isinstance(record, FactualRecord) for record in records):
        raise TypeError("pool must contain only FactualRecord values")
    trajectory = records[0].trajectory_key
    if any(record.trajectory_key != trajectory for record in records):
        raise ValueError("pool records must share one trajectory")
    if any(record.time >= origin for record in records):
        raise ValueError("pool records must be strictly before the origin")
    if len({record.frame_key for record in records}) != len(records):
        raise ValueError("pool records must be unique")

    ordered = sorted(
        records,
        key=lambda record: (
            hashlib.sha256(
                _DELETE_SAMPLE_DOMAIN
                + trajectory
                + struct.pack("<I", origin)
                + struct.pack("<I", record.time)
            ).digest(),
            record.time,
        ),
    )
    return tuple(ordered[: min(q, len(ordered))])


def collate_factual_views(
    views: Sequence[
        tuple[FactualViewKey, FactualMemory | PreEvictionView]
    ],
    action_encoder: nn.Module,
    *,
    device: torch.device | str,
    precision: Literal["32-true", "bf16-mixed"],
) -> FactualModelBatch:
    """Collate factual views and embed raw actions with the supplied frozen A."""
    if precision not in _PRECISIONS:
        raise ValueError("precision must be 32-true or bf16-mixed")
    try:
        keyed_states = tuple(views)
    except TypeError as exc:
        raise TypeError("views must be a finite sequence") from exc
    if not keyed_states:
        raise ValueError("views must be nonempty")

    rows: list[
        tuple[FactualViewKey, tuple[FactualRecord, ...], tuple[int, ...]]
    ] = []
    for row, item in enumerate(keyed_states):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("each view entry must be a (key, state) tuple")
        view = _validate_view_key(item[0])
        records, kinds = _validate_state_identity(view, item[1])
        rows.append((view, records, kinds))

    action_encoder, concrete_device = _resolve_action_encoder_device(
        action_encoder,
        device,
    )

    spatial_latent: tuple[int, int] | None = None
    for row, (_view, records, _kinds) in enumerate(rows):
        for column, record in enumerate(records):
            validated = _validate_record_for_collation(
                record,
                name=f"views[{row}].records[{column}]",
            )
            current_shape = tuple(validated.grid.shape)
            if spatial_latent is None:
                spatial_latent = current_shape
            elif current_shape != spatial_latent:
                raise ValueError("all factual grids must have one [M,D] shape")
    if spatial_latent is None:
        raise ValueError("views contain no factual records")

    batch_size = len(rows)
    record_count = max(len(records) for _view, records, _kinds in rows)
    spatial, latent = spatial_latent
    with torch.no_grad():
        record_grid = torch.zeros(
            batch_size,
            record_count,
            spatial,
            latent,
            dtype=torch.float32,
            device=concrete_device,
        )
        raw4 = torch.zeros(
            batch_size,
            record_count,
            4,
            dtype=torch.float32,
            device=concrete_device,
        )
        record_age = torch.zeros(
            batch_size,
            record_count,
            dtype=torch.int64,
            device=concrete_device,
        )
        record_type = torch.zeros_like(record_age)
        record_valid = torch.zeros(
            batch_size,
            record_count,
            dtype=torch.bool,
            device=concrete_device,
        )
        pool_mask = torch.zeros_like(record_valid)

        padded_keys: list[tuple[bytes | None, ...]] = []
        for row, (_view, records, kinds) in enumerate(rows):
            for column, (record, kind) in enumerate(zip(records, kinds)):
                record_grid[row, column].copy_(
                    record.grid.to(
                        device=concrete_device,
                        dtype=torch.float32,
                    )
                )
                raw4[row, column].copy_(
                    record.incoming_raw4.to(
                        device=concrete_device,
                        dtype=torch.float32,
                    )
                )
                record_age[row, column] = records[-1].time - record.time
                record_type[row, column] = kind
                record_valid[row, column] = True
                pool_mask[row, column] = kind == 0
            padded_keys.append(
                tuple(record.frame_key for record in records)
                + (None,) * (record_count - len(records))
            )

        with torch.autocast(
            device_type=concrete_device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16-mixed",
        ):
            embedded = action_encoder(raw4)

        if not isinstance(embedded, Tensor):
            raise TypeError("action_encoder output must be a Tensor")
        if tuple(embedded.shape) != (batch_size, record_count, latent):
            raise ValueError("action_encoder output shape does not match [B,R,D]")
        expected_dtype = (
            torch.float32 if precision == "32-true" else torch.bfloat16
        )
        if embedded.dtype != expected_dtype:
            raise TypeError("action_encoder output dtype does not match precision")
        _require_finite(embedded, name="action_encoder output")
        embedded = embedded.to(
            device=concrete_device,
            dtype=torch.float32,
        ).detach()
        embedded = embedded.masked_fill(~record_valid[:, :, None], 0.0)
        _require_finite(embedded, name="float32 action embedding")

    return FactualModelBatch(
        record_grid=record_grid.detach(),
        incoming_action_embedding=embedded,
        record_age=record_age,
        record_type=record_type,
        record_valid=record_valid,
        pool_mask=pool_mask,
        view_keys=tuple(view for view, _records, _kinds in rows),
        padded_record_frame_keys=tuple(padded_keys),
    )


def _validate_factual_model_batch(facts: object) -> FactualModelBatch:
    if not isinstance(facts, FactualModelBatch):
        raise TypeError("facts must be a FactualModelBatch")
    record_grid = _require_tensor(
        facts.record_grid,
        name="record_grid",
        dtype=torch.float32,
        rank=4,
    )
    action = _require_tensor(
        facts.incoming_action_embedding,
        name="incoming_action_embedding",
        dtype=torch.float32,
        rank=3,
    )
    age = _require_tensor(
        facts.record_age,
        name="record_age",
        dtype=torch.int64,
        rank=2,
    )
    kind = _require_tensor(
        facts.record_type,
        name="record_type",
        dtype=torch.int64,
        rank=2,
    )
    valid = _require_tensor(
        facts.record_valid,
        name="record_valid",
        dtype=torch.bool,
        rank=2,
    )
    pool = _require_tensor(
        facts.pool_mask,
        name="pool_mask",
        dtype=torch.bool,
        rank=2,
    )
    batch, records, _spatial, latent = record_grid.shape
    if batch < 1 or records < 1:
        raise ValueError("factual batch and record axes must be nonempty")
    if tuple(action.shape) != (batch, records, latent):
        raise ValueError("incoming action embedding shape is inconsistent")
    if any(tuple(value.shape) != (batch, records) for value in (age, kind, valid, pool)):
        raise ValueError("factual metadata shapes are inconsistent")
    if len({value.device for value in (record_grid, action, age, kind, valid, pool)}) != 1:
        raise ValueError("all factual tensors must share one device")
    if record_grid.requires_grad or action.requires_grad:
        raise ValueError("factual model inputs must be detached")
    _require_finite(record_grid, name="record_grid")
    _require_finite(action, name="incoming_action_embedding")
    if type(facts.view_keys) is not tuple or len(facts.view_keys) != batch:
        raise ValueError("view_keys must align with the factual batch")
    if (
        type(facts.padded_record_frame_keys) is not tuple
        or len(facts.padded_record_frame_keys) != batch
    ):
        raise ValueError("padded_record_frame_keys must align with the batch")

    for row in range(batch):
        view = _validate_view_key(facts.view_keys[row])
        row_valid = tuple(bool(value) for value in valid[row].tolist())
        seen_padding = False
        for is_valid in row_valid:
            if not is_valid:
                seen_padding = True
            elif seen_padding:
                raise ValueError("factual padding must be trailing")
        valid_count = sum(row_valid)
        if valid_count < 1:
            raise ValueError("each factual row must contain its current fact")
        padded = facts.padded_record_frame_keys[row]
        if type(padded) is not tuple or len(padded) != records:
            raise ValueError("padded frame keys must match the record axis")
        for column, is_valid in enumerate(row_valid):
            if is_valid:
                key = _require_bytes(
                    padded[column],
                    name=f"padded_record_frame_keys[{row}][{column}]",
                    length=36,
                )
                if key[:32] != view.trajectory_key:
                    raise ValueError("padded factual key crosses a trajectory")
            elif padded[column] is not None:
                raise ValueError("padding frame keys must be None")
        active_keys = tuple(padded[:valid_count])
        if active_keys != view.ordered_record_frame_keys:
            raise ValueError("padded keys do not bind the factual view")
        if active_keys[-1] != view.current_frame_key:
            raise ValueError("the last valid factual key must be current")

        times = tuple(_frame_time(key) for key in active_keys)
        expected_age = torch.tensor(
            tuple(view.origin_time - time for time in times),
            dtype=torch.int64,
            device=age.device,
        )
        if not torch.equal(age[row, :valid_count], expected_age):
            raise ValueError("record_age does not match canonical frame times")
        current_mask = kind[row, :valid_count] == 2
        if current_mask.sum().item() != 1 or not bool(current_mask[-1]):
            raise ValueError("each factual row must end in one current record")
        if age[row, valid_count - 1].item() != 0:
            raise ValueError("the current factual record must have age zero")
        if valid_count > 1:
            noncurrent_kind = kind[row, : valid_count - 1]
            if not bool(
                ((noncurrent_kind == 0) | (noncurrent_kind == 1)).all()
            ):
                raise ValueError("noncurrent record types must be pool or recent")
            if not bool((age[row, : valid_count - 1] >= 1).all()):
                raise ValueError("noncurrent facts must have positive age")
        expected_pool = valid[row] & (kind[row] == 0)
        if not torch.equal(pool[row], expected_pool):
            raise ValueError("pool_mask must exactly mark valid pool records")

        if valid_count < records:
            _require_zero(
                record_grid[row, valid_count:],
                name="padded record grids",
            )
            _require_zero(
                action[row, valid_count:],
                name="padded action embeddings",
            )
            _require_zero(age[row, valid_count:], name="padded record ages")
            _require_zero(kind[row, valid_count:], name="padded record types")
            if bool(pool[row, valid_count:].any()):
                raise ValueError("padding cannot be a deletion pool record")
    return facts


def _objective_signature(key: FactualObjectiveKey) -> tuple[object, ...]:
    return (
        key.view.trajectory_key,
        key.view.origin_time,
        key.terminal_time,
        key.goal_time,
        key.goal_frame_key,
        key.target_frame_keys,
    )


def _validate_delete_row_against_full(
    batch: FactualObjectiveBatch,
    *,
    full_row: int,
    delete_row: int,
) -> None:
    facts = batch.facts
    full_view = batch.keys[full_row].view
    delete_view = batch.keys[delete_row].view
    deleted = delete_view.deleted_frame_key
    if deleted is None or deleted not in full_view.ordered_record_frame_keys:
        raise ValueError("delete view does not remove a fact from its full view")
    expected_order = tuple(
        key for key in full_view.ordered_record_frame_keys if key != deleted
    )
    if delete_view.ordered_record_frame_keys != expected_order:
        raise ValueError("delete view changes a nondeleted factual identity")

    full_keys = facts.padded_record_frame_keys[full_row]
    delete_keys = facts.padded_record_frame_keys[delete_row]
    deleted_column = full_keys.index(deleted)
    if not bool(facts.pool_mask[full_row, deleted_column]):
        raise ValueError("declared deletion must remove a full-view pool record")
    if facts.record_valid[delete_row].sum().item() != (
        facts.record_valid[full_row].sum().item() - 1
    ):
        raise ValueError("delete view must remove exactly one factual record")

    for key in expected_order:
        full_column = full_keys.index(key)
        delete_column = delete_keys.index(key)
        for field in (
            "record_grid",
            "incoming_action_embedding",
            "record_age",
            "record_type",
            "record_valid",
            "pool_mask",
        ):
            value = getattr(facts, field)
            if not torch.equal(
                value[full_row, full_column],
                value[delete_row, delete_column],
            ):
                raise ValueError("delete view changes a nondeleted factual value")


def _validate_full_delete_pairs(batch: FactualObjectiveBatch) -> None:
    full_rows: dict[tuple[object, ...], list[int]] = {}
    for row, key in enumerate(batch.keys):
        if key.view.view_kind == "full":
            full_rows.setdefault(_objective_signature(key), []).append(row)

    for row, key in enumerate(batch.keys):
        if key.view.view_kind != "delete":
            continue
        candidates = full_rows.get(_objective_signature(key), ())
        matching: list[int] = []
        for full_row in candidates:
            if not torch.equal(batch.active_h[full_row], batch.active_h[row]):
                continue
            if not torch.equal(batch.target[full_row], batch.target[row]):
                continue
            if not torch.equal(batch.goal_grid[full_row], batch.goal_grid[row]):
                continue
            matching.append(full_row)
        if not matching:
            raise ValueError("delete view has no identical full objective pair")
        _validate_delete_row_against_full(
            batch,
            full_row=matching[0],
            delete_row=row,
        )


def _validate_objective_batch_structure(
    objective_batch: object,
    *,
    proposal: ProposalJEPA | None,
) -> FactualObjectiveBatch:
    if not isinstance(objective_batch, FactualObjectiveBatch):
        raise TypeError("objective_batch must be a FactualObjectiveBatch")
    facts = _validate_factual_model_batch(objective_batch.facts)
    batch, _records, spatial, latent = facts.record_grid.shape
    if type(objective_batch.keys) is not tuple or len(objective_batch.keys) != batch:
        raise ValueError("objective keys must align with factual rows")
    goal = _require_tensor(
        objective_batch.goal_grid,
        name="goal_grid",
        dtype=torch.float32,
        rank=3,
    )
    target = _require_tensor(
        objective_batch.target,
        name="target",
        dtype=torch.float32,
        rank=4,
    )
    active = _require_tensor(
        objective_batch.active_h,
        name="active_h",
        dtype=torch.bool,
        rank=2,
    )
    if active.shape[0] != batch or active.shape[1] < 1:
        raise ValueError("active_h must have shape [B,H] with nonempty H")
    horizon = active.shape[1]
    if tuple(goal.shape) != (batch, spatial, latent):
        raise ValueError("goal_grid shape is inconsistent with factual grids")
    if tuple(target.shape) != (batch, horizon, spatial, latent):
        raise ValueError("target shape is inconsistent with factual grids")
    factual_device = facts.record_grid.device
    if goal.device != factual_device or target.device != factual_device or active.device != factual_device:
        raise ValueError("facts, goal, target, and active_h must share one device")
    _require_finite(goal, name="goal_grid")
    _require_finite(target, name="target")
    if target.requires_grad or target.grad_fn is not None:
        raise ValueError("factual target must be detached")

    if proposal is not None:
        if not isinstance(proposal, ProposalJEPA):
            raise TypeError("proposal must be the existing ProposalJEPA")
        model_device = proposal.input_proj.weight.device
        if factual_device != model_device:
            raise ValueError("objective tensors and ProposalJEPA must share a device")
        if latent != proposal.input_proj.in_features:
            raise ValueError("objective latent width does not match ProposalJEPA")
        if spatial != proposal.future.spatial_position.shape[0]:
            raise ValueError("objective spatial width does not match ProposalJEPA")
        model_horizon = proposal.future.horizon_embedding.num_embeddings
        if horizon != model_horizon:
            raise ValueError("objective horizon does not match ProposalJEPA")

    for row, key in enumerate(objective_batch.keys):
        if not isinstance(key, FactualObjectiveKey):
            raise TypeError("objective keys must be FactualObjectiveKey values")
        view = _validate_view_key(key.view)
        if facts.view_keys[row] != view:
            raise ValueError("facts.view_keys and objective keys do not match")
        origin = view.origin_time
        terminal = _require_uint32(key.terminal_time, name="terminal_time")
        if terminal <= origin:
            raise ValueError("terminal_time must follow the factual origin")
        goal_time = _require_uint32(key.goal_time, name="goal_time")
        expected_active = min(horizon, terminal - origin)
        row_active = active[row]
        if not bool(row_active[0]):
            raise ValueError("active_h must be a nonempty true prefix")
        if horizon > 1 and bool(((~row_active[:-1]) & row_active[1:]).any()):
            raise ValueError("active_h must be a contiguous true prefix")
        if row_active.sum().item() != expected_active:
            raise ValueError("active_h does not cover every factual successor")

        if type(key.target_frame_keys) is not tuple:
            raise TypeError("target_frame_keys must be a tuple")
        expected_targets = tuple(
            frame_key(view.trajectory_key, time)
            for time in range(origin + 1, origin + expected_active + 1)
        )
        if key.target_frame_keys != expected_targets:
            raise ValueError("target_frame_keys are not the canonical successors")

        allowed_goal_times = {terminal}
        if terminal - origin >= horizon:
            allowed_goal_times.add(
                hashed_goal_index(
                    view.trajectory_key,
                    t=origin,
                    terminal=terminal,
                    horizon=horizon,
                )
            )
        if goal_time not in allowed_goal_times:
            raise ValueError("goal_time is not terminal or the canonical hash goal")
        _require_canonical_frame(
            key.goal_frame_key,
            trajectory=view.trajectory_key,
            time=goal_time,
            name="goal_frame_key",
        )

        inactive = ~row_active[:, None, None].expand_as(target[row])
        if torch.count_nonzero(target[row].masked_select(inactive)).item() != 0:
            raise ValueError("inactive factual target slots must be exactly zero")

    _validate_full_delete_pairs(objective_batch)
    return objective_batch


def proposal_future_terms(
    proposal: ProposalJEPA,
    objective_batch: FactualObjectiveBatch,
) -> ProperMixtureTerms:
    """Run the supplied ProposalJEPA once and apply its public proper loss."""
    batch = _validate_objective_batch_structure(
        objective_batch,
        proposal=proposal,
    )
    output = proposal(
        batch.facts.record_grid,
        batch.facts.incoming_action_embedding,
        batch.facts.record_age,
        batch.facts.record_type,
        batch.facts.record_valid,
        batch.goal_grid,
        batch.active_h,
    )
    return proper_mixture_terms(
        output.tape,
        output.log_mass,
        batch.target.detach(),
        batch.active_h,
    )


def deletion_utilities_from_nll(
    occurrence_nll: Tensor,
    objective_batch: FactualObjectiveBatch,
) -> DeletionUtilities:
    """Reduce keyed full/delete occurrence NLLs to factual utilities."""
    nll = _require_tensor(
        occurrence_nll,
        name="occurrence_nll",
        dtype=torch.float32,
        rank=1,
    )
    _require_finite(nll, name="occurrence_nll")
    batch = _validate_objective_batch_structure(
        objective_batch,
        proposal=None,
    )
    if tuple(nll.shape) != (len(batch.keys),):
        raise ValueError("occurrence_nll must contain one value per objective row")

    origin_order: list[tuple[bytes, int]] = []
    rows_by_origin: dict[tuple[bytes, int], list[int]] = {}
    for row, key in enumerate(batch.keys):
        origin = (key.view.trajectory_key, key.view.origin_time)
        if origin not in rows_by_origin:
            origin_order.append(origin)
            rows_by_origin[origin] = []
        rows_by_origin[origin].append(row)

    horizon = batch.active_h.shape[1]
    output_full_views: list[FactualViewKey] = []
    output_deleted_keys: list[tuple[bytes, ...]] = []
    output_values: list[Tensor] = []
    detached_nll = nll.detach()

    for origin in origin_order:
        origin_rows = rows_by_origin[origin]
        terminals = {batch.keys[row].terminal_time for row in origin_rows}
        if len(terminals) != 1:
            raise ValueError("one utility origin must have exactly one terminal")
        terminal = next(iter(terminals))
        expected_goals = {
            (terminal, frame_key(origin[0], terminal)),
        }
        if terminal - origin[1] >= horizon:
            hashed_time = hashed_goal_index(
                origin[0],
                t=origin[1],
                terminal=terminal,
                horizon=horizon,
            )
            expected_goals.add((hashed_time, frame_key(origin[0], hashed_time)))
        actual_goals = {
            (batch.keys[row].goal_time, batch.keys[row].goal_frame_key)
            for row in origin_rows
        }
        if actual_goals != expected_goals:
            raise ValueError("utility requires the complete terminal/hash goal set")

        reference_row = origin_rows[0]
        for row in origin_rows[1:]:
            if not torch.equal(batch.target[row], batch.target[reference_row]):
                raise ValueError("utility rows must share one factual target")
            if not torch.equal(batch.active_h[row], batch.active_h[reference_row]):
                raise ValueError("utility rows must share one active horizon")
            if batch.keys[row].target_frame_keys != batch.keys[
                reference_row
            ].target_frame_keys:
                raise ValueError("utility rows must share canonical target keys")

        ordered_goals = tuple(sorted(expected_goals))
        full_row_by_goal: dict[tuple[int, bytes], int] = {}
        delete_rows_by_goal: dict[tuple[int, bytes], tuple[int, ...]] = {}
        for goal in ordered_goals:
            goal_rows = tuple(
                row
                for row in origin_rows
                if (
                    batch.keys[row].goal_time,
                    batch.keys[row].goal_frame_key,
                )
                == goal
            )
            full_rows = tuple(
                row
                for row in goal_rows
                if batch.keys[row].view.view_kind == "full"
            )
            delete_rows = tuple(
                row
                for row in goal_rows
                if batch.keys[row].view.view_kind == "delete"
            )
            if len(full_rows) != 1:
                raise ValueError("each utility goal requires exactly one full row")
            if not delete_rows or len(full_rows) + len(delete_rows) != len(
                goal_rows
            ):
                raise ValueError("utility goals require a nonempty full/delete set")
            deleted_keys = tuple(
                batch.keys[row].view.deleted_frame_key for row in delete_rows
            )
            if any(key is None for key in deleted_keys) or len(
                set(deleted_keys)
            ) != len(deleted_keys):
                raise ValueError("utility delete rows must have unique deleted keys")
            full_row = full_rows[0]
            if any(
                not torch.equal(batch.goal_grid[row], batch.goal_grid[full_row])
                for row in goal_rows
            ):
                raise ValueError("one utility goal must use one exact goal grid")
            full_row_by_goal[goal] = full_row
            delete_rows_by_goal[goal] = delete_rows

        first_goal = ordered_goals[0]
        first_delete_rows = delete_rows_by_goal[first_goal]
        deleted_order = tuple(
            batch.keys[row].view.deleted_frame_key for row in first_delete_rows
        )
        if any(key is None for key in deleted_order):
            raise ValueError("utility delete keys cannot be None")
        canonical_deleted_order = tuple(
            key for key in deleted_order if key is not None
        )
        full_view = batch.keys[full_row_by_goal[first_goal]].view
        if full_view.view_kind != "full" or full_view.deleted_frame_key is not None:
            raise ValueError("utility requires a canonical full view")
        if (full_view.trajectory_key, full_view.origin_time) != origin:
            raise ValueError("utility full view does not match its origin")

        delete_row_lookup: dict[tuple[int, bytes], dict[bytes, int]] = {}
        for goal in ordered_goals:
            candidate_full = batch.keys[full_row_by_goal[goal]].view
            if candidate_full != full_view:
                raise ValueError("all utility goals must share the exact full view")
            delete_rows = delete_rows_by_goal[goal]
            goal_deleted = tuple(
                batch.keys[row].view.deleted_frame_key for row in delete_rows
            )
            if goal_deleted != canonical_deleted_order:
                raise ValueError("all utility goals must share delete-key order")
            per_key: dict[bytes, int] = {}
            for row in delete_rows:
                deleted_key = batch.keys[row].view.deleted_frame_key
                if deleted_key is None:
                    raise ValueError("utility delete key cannot be None")
                per_key[deleted_key] = row
                first_row = first_delete_rows[
                    canonical_deleted_order.index(deleted_key)
                ]
                if batch.keys[row].view != batch.keys[first_row].view:
                    raise ValueError("delete factual views must match across goals")
            delete_row_lookup[goal] = per_key

        values: list[Tensor] = []
        for deleted_key in canonical_deleted_order:
            differences = torch.stack(
                tuple(
                    detached_nll[delete_row_lookup[goal][deleted_key]]
                    - detached_nll[full_row_by_goal[goal]]
                    for goal in ordered_goals
                )
            )
            value = differences.float().mean().detach()
            if not torch.isfinite(value).item():
                raise ValueError("deletion utility must remain finite")
            values.append(value)
        utility_values = torch.stack(tuple(values)).to(dtype=torch.float32).detach()
        if not torch.isfinite(utility_values).all().item():
            raise ValueError("deletion utilities must be finite")
        output_full_views.append(full_view)
        output_deleted_keys.append(canonical_deleted_order)
        output_values.append(utility_values)

    return DeletionUtilities(
        origin_keys=tuple(origin_order),
        full_view_keys=tuple(output_full_views),
        deleted_frame_keys=tuple(output_deleted_keys),
        values=tuple(output_values),
    )


def deletion_utilities(
    proposal: ProposalJEPA,
    objective_batch: FactualObjectiveBatch,
) -> DeletionUtilities:
    """Compute detached utilities with an already read-only ProposalJEPA."""
    if not isinstance(proposal, ProposalJEPA):
        raise TypeError("proposal must be the existing ProposalJEPA")
    if proposal.training or any(
        module.training
        for module in (proposal.input_proj, proposal.selector, proposal.future)
    ):
        raise ValueError("deletion utility requires ProposalJEPA in eval mode")
    if any(parameter.requires_grad for parameter in proposal.parameters()):
        raise ValueError("deletion utility requires frozen ProposalJEPA parameters")
    with torch.no_grad():
        terms = proposal_future_terms(proposal, objective_batch)
    return deletion_utilities_from_nll(
        terms.occurrence_nll.detach(),
        objective_batch,
    )


def _validate_selector_targets(
    proposal: object,
    factual_batch: object,
    selected_positions: object,
    utilities: object,
) -> tuple[
    ProposalJEPA,
    FactualModelBatch,
    tuple[Tensor, ...],
    DeletionUtilities,
]:
    if not isinstance(proposal, ProposalJEPA):
        raise TypeError("proposal must be the existing ProposalJEPA")
    facts = _validate_factual_model_batch(factual_batch)
    batch_size, _record_count, spatial, latent = facts.record_grid.shape
    if facts.record_grid.device != proposal.input_proj.weight.device:
        raise ValueError("selector facts and ProposalJEPA must share one device")
    if latent != proposal.input_proj.in_features:
        raise ValueError("selector fact width does not match ProposalJEPA")
    if spatial != proposal.future.spatial_position.shape[0]:
        raise ValueError("selector spatial width does not match ProposalJEPA")
    try:
        positions_by_row = tuple(selected_positions)
    except TypeError as exc:
        raise TypeError("selected_positions must be a finite sequence") from exc
    if len(positions_by_row) != batch_size:
        raise ValueError("selector requires one position vector per factual row")
    if not isinstance(utilities, DeletionUtilities):
        raise TypeError("utilities must be DeletionUtilities")
    utility_lengths = (
        len(utilities.origin_keys),
        len(utilities.full_view_keys),
        len(utilities.deleted_frame_keys),
        len(utilities.values),
    )
    if utility_lengths != (batch_size,) * 4:
        raise ValueError("selector requires one utility row per factual origin")

    factual_origins: list[tuple[bytes, int]] = []
    for row, view in enumerate(facts.view_keys):
        if view.view_kind != "full" or view.deleted_frame_key is not None:
            raise ValueError("selector factual rows must be canonical full views")
        if not facts.pool_mask[row].any().item():
            raise ValueError("selector full views require a nonempty pool")
        factual_origins.append((view.trajectory_key, view.origin_time))
    if len(set(factual_origins)) != batch_size:
        raise ValueError("selector factual origins must be unique")

    utility_origins: list[tuple[bytes, int]] = []
    full_view_by_origin: dict[tuple[bytes, int], FactualViewKey] = {}
    target_by_origin: dict[tuple[bytes, int], tuple[tuple[bytes, ...], Tensor]] = {}
    for row in range(batch_size):
        origin_value = utilities.origin_keys[row]
        if type(origin_value) is not tuple or len(origin_value) != 2:
            raise TypeError("utility origin keys must be (trajectory, time) tuples")
        trajectory = _require_bytes(
            origin_value[0],
            name="utility origin trajectory",
            length=32,
        )
        origin_time = _require_uint32(
            origin_value[1],
            name="utility origin time",
        )
        origin = (trajectory, origin_time)
        full_view = _validate_view_key(utilities.full_view_keys[row])
        if full_view.view_kind != "full" or full_view.deleted_frame_key is not None:
            raise ValueError("utility full views must be canonical full views")
        if (full_view.trajectory_key, full_view.origin_time) != origin:
            raise ValueError("utility full view does not match its origin")

        deleted_keys = utilities.deleted_frame_keys[row]
        values = utilities.values[row]
        if type(deleted_keys) is not tuple:
            raise TypeError("utility deleted_frame_keys rows must be tuples")
        values = _require_tensor(
            values,
            name="utility values",
            dtype=torch.float32,
            rank=1,
        )
        if values.numel() < 1 or len(deleted_keys) != values.numel():
            raise ValueError("utility keys and values must be nonempty and aligned")
        if values.requires_grad or values.grad_fn is not None:
            raise ValueError("utility values must be detached")
        _require_finite(values, name="utility values")
        for index, deleted_key in enumerate(deleted_keys):
            raw = _require_bytes(
                deleted_key,
                name=f"utility deleted_frame_keys[{row}][{index}]",
                length=36,
            )
            if raw[:32] != trajectory or raw not in full_view.ordered_record_frame_keys:
                raise ValueError("utility deletion must identify a full-view fact")
        if len(set(deleted_keys)) != len(deleted_keys):
            raise ValueError("utility deleted keys must be unique")
        utility_origins.append(origin)
        full_view_by_origin[origin] = full_view
        target_by_origin[origin] = (deleted_keys, values)

    if len(set(utility_origins)) != batch_size:
        raise ValueError("utility origins must be unique")
    if set(factual_origins) != set(utility_origins):
        raise ValueError("factual and utility origins must exactly cover each other")
    for view in facts.view_keys:
        origin = (view.trajectory_key, view.origin_time)
        if view != full_view_by_origin[origin]:
            raise ValueError("selector facts do not match the utility full view")

    record_count = facts.record_valid.shape[1]
    for row, positions in enumerate(positions_by_row):
        positions = _require_tensor(
            positions,
            name="selected positions",
            dtype=torch.int64,
            rank=1,
        )
        if positions.numel() < 1:
            raise ValueError("selected positions must be nonempty")
        position_values = tuple(
            int(value) for value in positions.detach().cpu().tolist()
        )
        if len(set(position_values)) != len(position_values):
            raise ValueError("selected positions must be unique")
        if any(index < 0 or index >= record_count for index in position_values):
            raise ValueError("selected position is outside the padded record axis")
        if any(
            not bool(
                facts.record_valid[row, index]
                & facts.pool_mask[row, index]
            )
            for index in position_values
        ):
            raise ValueError("selected positions must refer to valid pool records")

        view = facts.view_keys[row]
        origin = (view.trajectory_key, view.origin_time)
        deleted_keys, values = target_by_origin[origin]
        if len(position_values) != len(deleted_keys):
            raise ValueError("selected positions and utility values must align")
        padded = facts.padded_record_frame_keys[row]
        observed_keys = tuple(padded[index] for index in position_values)
        if observed_keys != deleted_keys:
            raise ValueError("selected positions must match deleted-key order")

    return proposal, facts, positions_by_row, utilities


def selector_regression_loss(
    proposal: ProposalJEPA,
    factual_batch: FactualModelBatch,
    selected_positions: Sequence[Tensor],
    utilities: DeletionUtilities,
) -> Tensor:
    """Regress selector scores to detached utilities with origin-equal weight."""
    proposal, facts, positions_by_row, utilities = _validate_selector_targets(
        proposal,
        factual_batch,
        selected_positions,
        utilities,
    )
    scores = proposal.score_records(
        facts.record_grid,
        facts.incoming_action_embedding,
        facts.record_age,
        facts.record_type,
        facts.record_valid,
    )
    targets = {
        origin: values
        for origin, values in zip(utilities.origin_keys, utilities.values)
    }
    origin_losses: list[Tensor] = []
    for row, positions in enumerate(positions_by_row):
        view = facts.view_keys[row]
        origin = (view.trajectory_key, view.origin_time)
        prediction = scores[row].index_select(
            0,
            positions.to(device=scores.device),
        )
        target = targets[origin].detach().to(
            device=scores.device,
            dtype=torch.float32,
        )
        origin_losses.append((prediction - target).square().mean())
    return torch.stack(tuple(origin_losses)).mean()


def configure_proposal_train_mode(
    proposal: ProposalJEPA,
    mode: ProposalTrainMode,
) -> None:
    """Set the frozen Q/S gradient-owner modes on one ProposalJEPA."""
    if not isinstance(proposal, ProposalJEPA):
        raise TypeError("proposal must be the existing ProposalJEPA")
    if not isinstance(mode, ProposalTrainMode):
        raise TypeError("mode must be a ProposalTrainMode")

    proposal.eval()
    for parameter in proposal.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    if mode is ProposalTrainMode.Q0:
        proposal.train()
        proposal.selector.eval()
        for parameter in proposal.input_proj.parameters():
            parameter.requires_grad_(True)
        for parameter in proposal.future.parameters():
            parameter.requires_grad_(True)
    elif mode is ProposalTrainMode.Q_CONTINUE:
        proposal.train()
        proposal.input_proj.eval()
        proposal.selector.eval()
        for parameter in proposal.future.parameters():
            parameter.requires_grad_(True)
    elif mode is ProposalTrainMode.SELECTOR:
        proposal.train()
        proposal.input_proj.eval()
        proposal.future.eval()
        for parameter in proposal.selector.parameters():
            parameter.requires_grad_(True)
    else:
        proposal.eval()
