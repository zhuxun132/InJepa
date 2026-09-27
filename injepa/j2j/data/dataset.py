"""Map-style categorical trajectory views for the released StreamVLN cache.

The implementation in this module is intentionally a thin data seam.  It
does not own a parser, encoder, objective, or training state: source
authority and frame locators belong to :mod:`j2j.data.source`, while cache
validation and payload reads belong to :mod:`j2j.encoding.cache`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from j2j.adapter import ActionId, Raw4Adapter
from j2j.data.keys import frame_key
from j2j.data.source import ReleasedSourceItem, SourceError


_EMBED_DIM = 768
_SPATIAL_TOKENS = 36
_STREAMVLN_REVISION = "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9"


class CategoricalDatasetError(ValueError):
    """Raised when a released trajectory cannot be joined to the cache."""


def _as_cpu_float32_grid(value: object, *, step: int, spatial_shape: tuple[int, int] = (36, 768)) -> torch.Tensor:
    """Validate one public cache grid and copy only the requested row."""

    if not isinstance(value, np.ndarray):
        raise CategoricalDatasetError("cache grid must be a NumPy array")
    expected_dtype = np.dtype("<f4")
    if value.dtype != expected_dtype:
        raise CategoricalDatasetError("cache grid must have little-endian float32 dtype")
    if tuple(value.shape) != spatial_shape:
        raise CategoricalDatasetError(f"cache grid must have shape {spatial_shape}")
    if not value.flags.c_contiguous:
        raise CategoricalDatasetError("cache grid must be C-contiguous")
    if not bool(np.isfinite(value).all()):
        raise CategoricalDatasetError("cache grid must be finite")
    # ``clone`` is deliberately bounded to this trajectory row.  In
    # particular, it never turns the cache's complete memmap into a list.
    try:
        # A read-only memmap cannot be handed directly to torch without a
        # writability warning; make the bounded row copy before conversion.
        row_copy = np.array(value, dtype=expected_dtype, order="C", copy=True)
        tensor = torch.from_numpy(row_copy).detach()
    except (TypeError, ValueError) as exc:
        raise CategoricalDatasetError(f"cache grid conversion failed at step {step}") from exc
    if tensor.dtype != torch.float32 or tensor.shape != spatial_shape:
        raise CategoricalDatasetError("cache grid conversion changed its ABI")
    if not bool(torch.isfinite(tensor).all()):  # also catches an unusual conversion backend
        raise CategoricalDatasetError("cache grid conversion produced non-finite values")
    tensor.requires_grad_(False)
    return tensor


def _validate_source_item(item: object) -> ReleasedSourceItem:
    # Exact type is intentional: D1B must consume the sole D1A source wrapper,
    # not a Dataset-local look-alike or a second authority object.
    if type(item) is not ReleasedSourceItem:
        raise CategoricalDatasetError(
            "source_items must contain exact j2j.data.source.ReleasedSourceItem values"
        )
    if item.projection_partition not in {"project-train", "project-dev"}:
        raise CategoricalDatasetError("categorical source partition must be project-train or project-dev")
    trajectory = item.canonical_trajectory
    actions = trajectory.actions
    if type(actions) is not tuple or not actions or actions[0] != -1:
        raise CategoricalDatasetError("canonical actions must start with BOS -1")
    # STOP is a terminal analytic label, never an outgoing visual transition.
    if any(type(action) is not int or action not in (1, 2, 3) for action in actions[1:]):
        raise CategoricalDatasetError("canonical motion actions must be FWD, LEFT, or RIGHT")
    if len(item.cache_global_rows) != len(actions):
        raise CategoricalDatasetError("cache locator count must equal frame count")
    return item


@dataclass(frozen=True)
class _CategoricalTrajectoryItem:
    """Runtime view holding the direct D1A source object and bounded payload."""

    source_item: ReleasedSourceItem
    grids: torch.Tensor
    action_ids: torch.Tensor
    frame_keys: tuple[bytes, ...]


class CategoricalTrajectoryDataset(torch.utils.data.Dataset):
    """A map-style view over admitted canonical released trajectories.

    ``source_items`` is retained as the same immutable object sequence; each
    item keeps a direct ``source_item`` reference.  Cache payload is read only
    for the requested trajectory in ``__getitem__``.
    """

    def __init__(self, source_items: Sequence[ReleasedSourceItem], cache_store: object, *,
                 spatial_shape: tuple[int, int] = (36, 768)) -> None:
        if (not isinstance(spatial_shape, (tuple, list)) or len(spatial_shape) != 2
                or any(type(value) is not int or value <= 0 for value in spatial_shape)):
            raise CategoricalDatasetError("spatial_shape must contain two positive integers")
        self._spatial_shape = tuple(spatial_shape)
        if isinstance(source_items, (str, bytes)) or not isinstance(source_items, Sequence):
            raise TypeError("source_items must be a sequence")
        reader = getattr(cache_store, "read_global_rows", None)
        if not callable(reader):
            raise TypeError("cache_store must expose read_global_rows")
        try:
            materialized = tuple(source_items)
        except TypeError as exc:
            raise TypeError("source_items must be a sequence") from exc
        validated = tuple(_validate_source_item(item) for item in materialized)
        self._source_items = validated
        self._cache_store = cache_store

    def __len__(self) -> int:
        return len(self._source_items)

    def batch_descriptors(self) -> tuple[tuple[int, bytes, int], ...]:
        """Return the plan-only projection consumed by the categorical sampler.

        The projection contains no grid payload and does not create another
        source authority; keys and lengths are read directly from the held
        D1A source objects.  A tuple is returned so callers cannot mutate the
        dataset's ordering or identity view.
        """

        return tuple(
            (
                index,
                item.canonical_trajectory.canonical_trajectory_key,
                len(item.canonical_trajectory.actions) - 1,
            )
            for index, item in enumerate(self._source_items)
        )

    def __getitem__(self, index: int) -> _CategoricalTrajectoryItem:
        if type(index) is not int:
            raise TypeError("dataset index must be an exact integer")
        if index < 0 or index >= len(self._source_items):
            raise IndexError("dataset index is out of range")
        source_item = self._source_items[index]
        trajectory = source_item.canonical_trajectory
        actions = trajectory.actions
        rows = source_item.cache_global_rows
        try:
            records = list(self._cache_store.read_global_rows(rows))
        except Exception as exc:
            if isinstance(exc, (CategoricalDatasetError, SourceError)):
                raise
            raise CategoricalDatasetError("cache rows could not be read") from exc
        if len(records) != len(rows):
            raise CategoricalDatasetError("cache returned the wrong number of rows")

        grids: list[torch.Tensor] = []
        frame_keys: list[bytes] = []
        for step, record in enumerate(records):
            expected_frame = frame_key(trajectory.canonical_trajectory_key, step)
            actual_frame = getattr(record, "frame_key", None)
            if actual_frame != expected_frame:
                raise CategoricalDatasetError("cache frame key does not match trajectory/time")
            if getattr(record, "source_dataset", None) != trajectory.source_id:
                raise CategoricalDatasetError("cache source dataset does not match source authority")
            if getattr(record, "partition", None) != source_item.projection_partition:
                raise CategoricalDatasetError("cache partition does not match source authority")
            if getattr(record, "revision", None) != _STREAMVLN_REVISION:
                raise CategoricalDatasetError("cache revision does not match StreamVLN authority")
            expected_compressed = trajectory.compressed_jpeg_sha256s[step]
            expected_decoded = trajectory.decoded_rgb_sha256s[step]
            if getattr(record, "compressed_jpeg_sha256", None) != expected_compressed:
                raise CategoricalDatasetError("cache compressed JPEG identity does not match source")
            if getattr(record, "decoded_rgb_sha256", None) != expected_decoded:
                raise CategoricalDatasetError("cache decoded RGB identity does not match source")
            grids.append(_as_cpu_float32_grid(getattr(record, "grid", None), step=step,
                                               spatial_shape=self._spatial_shape))
            frame_keys.append(expected_frame)

        action_ids = torch.tensor(actions[1:], dtype=torch.int64)
        action_ids = action_ids.detach().clone()
        action_ids.requires_grad_(False)
        grid_tensor = torch.stack(grids, dim=0) if grids else torch.empty(
            (0, *self._spatial_shape), dtype=torch.float32
        )
        grid_tensor = grid_tensor.detach().clone()
        grid_tensor.requires_grad_(False)
        return _CategoricalTrajectoryItem(
            source_item=source_item,
            grids=grid_tensor,
            action_ids=action_ids,
            frame_keys=tuple(frame_keys),
        )


def _validate_item(item: object) -> _CategoricalTrajectoryItem:
    if type(item) is not _CategoricalTrajectoryItem:
        raise TypeError("collator expects CategoricalTrajectoryDataset items")
    source_item = _validate_source_item(item.source_item)
    grids = item.grids
    actions = item.action_ids
    if not isinstance(grids, torch.Tensor) or grids.dtype != torch.float32:
        raise TypeError("item grids must be float32 tensors")
    if grids.ndim != 3 or grids.shape[1:] != (_SPATIAL_TOKENS, _EMBED_DIM):
        raise ValueError("item grids must have shape [T+1,36,768]")
    if not grids.is_floating_point() or not bool(torch.isfinite(grids).all()):
        raise ValueError("item grids must be finite floating tensors")
    if grids.requires_grad:
        raise ValueError("item grids must be detached")
    if not isinstance(actions, torch.Tensor) or actions.dtype != torch.int64 or actions.ndim != 1:
        raise TypeError("item action_ids must be int64 [T]")
    if actions.shape[0] + 1 != grids.shape[0]:
        raise ValueError("item action/frame lengths disagree")
    if actions.numel() and not bool(((actions >= 1) & (actions <= 3)).all()):
        raise ValueError("item action_ids must be FWD, LEFT, or RIGHT")
    keys = item.frame_keys
    if type(keys) is not tuple or len(keys) != grids.shape[0]:
        raise ValueError("item frame_keys must align with frames")
    expected_keys = tuple(
        frame_key(source_item.canonical_trajectory.canonical_trajectory_key, step)
        for step in range(grids.shape[0])
    )
    if keys != expected_keys:
        raise ValueError("item frame_keys do not match canonical trajectory")
    if grids.device != actions.device:
        raise ValueError("item tensors must share a device")
    return item


def collate_categorical_trajectories(items: Sequence[_CategoricalTrajectoryItem]) -> dict[str, torch.Tensor]:
    """Collate trajectories into the four tensors consumed by INTACT.

    Spatial tokens are pooled with the fixed, parameter-free mean over 36
    tokens.  Incoming action vectors are shifted one position relative to the
    outgoing ``action_ids``; the first position is the existing all-zero BOS.
    """

    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or len(items) == 0:
        raise ValueError("collator requires a non-empty sequence of items")
    validated = tuple(_validate_item(item) for item in items)
    first = validated[0]
    device = first.grids.device
    dtype = first.grids.dtype
    if dtype != torch.float32:
        raise TypeError("categorical grids must be float32")
    if any(item.grids.device != device for item in validated):
        raise ValueError("all categorical items must share a device")
    batch_size = len(validated)
    max_motion = max(int(item.action_ids.shape[0]) for item in validated)
    embeddings = torch.zeros(
        (batch_size, max_motion + 1, _EMBED_DIM), dtype=dtype, device=device
    )
    action_ids = torch.zeros(
        (batch_size, max_motion), dtype=torch.int64, device=device
    )
    previous_raw4 = torch.zeros(
        (batch_size, max_motion + 1, 4), dtype=dtype, device=device
    )
    active_motion = torch.zeros(
        (batch_size, max_motion), dtype=torch.bool, device=device
    )
    for batch_index, item in enumerate(validated):
        length = int(item.action_ids.shape[0])
        embeddings[batch_index, : length + 1].copy_(item.grids.mean(dim=1))
        if not length:
            continue
        action_ids[batch_index, :length].copy_(item.action_ids)
        active_motion[batch_index, :length] = True
        # Use the frozen adapter's exact one-hot convention.  A single
        # stack keeps the operation deterministic and avoids any RNG path.
        encoded = torch.stack(
            [Raw4Adapter.encode(ActionId(int(action))) for action in item.action_ids.tolist()],
            dim=0,
        ).to(device=device, dtype=dtype)
        previous_raw4[batch_index, 1 : length + 1].copy_(encoded)
    return {
        "embeddings": embeddings,
        "action_ids": action_ids,
        "previous_raw4": previous_raw4,
        "active_motion": active_motion,
    }


__all__ = [
    "CategoricalDatasetError",
    "CategoricalTrajectoryDataset",
    "collate_categorical_trajectories",
]
