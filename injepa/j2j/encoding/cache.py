"""Research cache and whitening primitives for frozen V-JEPA spatial tokens.

The cache deliberately uses only portable NumPy ``.npy``/``.npz`` files and
canonical JSON/JSONL metadata.  It protects the scientific identities needed by
the experiment without implementing a product object store or adversarial file
system protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import operator
import os
from pathlib import Path
import platform
import resource
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_info, threadpool_limits


_FEATURES = 768
_TOKENS = 36
_FRAME_KEY_BYTES = 36
_STREAMVLN_REVISION = "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9"
_VJEPA_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
_SKLEARN_VERSION = "1.7.2"
_PILOT_RTOL = 1e-5
_PILOT_ATOL = 1e-6
_IDENTITY_FIELDS = {
    "vjepa_source_commit",
    "checkpoint_sha256",
    "preprocess_sha256",
    "pool_sha256",
    "whitening_sha256",
}
_SHARED_ENCODING_IDENTITIES = (
    "vjepa_source_commit",
    "checkpoint_sha256",
    "preprocess_sha256",
    "pool_sha256",
)
_HASH_CHARS = frozenset("0123456789abcdef")
_STAGE_DTYPES = {
    "U": np.dtype("<f4"),
    "RAW32": np.dtype("<f4"),
    "Z32": np.dtype("<f4"),
    "Z16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
}


class CacheError(ValueError):
    """A cache artifact violates the frozen scientific contract."""


@dataclass
class CacheRecord:
    """One canonical frame occurrence and its spatial token row."""

    frame_key: bytes
    grid: np.ndarray
    compressed_jpeg_sha256: str
    decoded_rgb_sha256: str
    source_dataset: str
    revision: str
    building: str
    partition: str
    frame_preimage: Mapping[str, Any]
    canonical: bool
    trajectory_alias: bool


@dataclass(frozen=True)
class WhiteningTransform:
    """The fixed float64 Ledoit-Wolf whitening coordinate system."""

    mu: np.ndarray
    whitener: np.ndarray
    eigenvalues: np.ndarray
    shrinkage: float
    artifact_sha256: str | None = None


def _memmap_data_offset(mapped: np.memmap) -> int:
    """Return the ``.npy`` payload offset for a memmap or a view of one."""

    candidate: object | None = mapped
    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        offset = getattr(candidate, "offset", None)
        if isinstance(offset, (int, np.integer)) and not isinstance(offset, bool):
            if int(offset) >= 0:
                return int(offset)
        candidate = getattr(candidate, "base", None)
    raise CacheError("cache grid memmap has no discoverable payload offset")


@dataclass
class _CacheShardState:
    """Validated state for one lazily opened cache shard.

    The state intentionally keeps only the shard's bounded index metadata and
    its read-only memmap.  A store does not eagerly open or copy any other
    shard.
    """

    descriptor: Mapping[str, Any]
    grid: np.memmap
    index_rows: tuple[dict[str, Any], ...]


_INDEX_FIELDS = frozenset(
    {
        "frame_key",
        "compressed_jpeg_sha256",
        "decoded_rgb_sha256",
        "source_dataset",
        "revision",
        "building",
        "partition",
        "frame_preimage",
        "canonical",
        "trajectory_alias",
        "tensor_row_sha256",
    }
)


def _read_index_metadata(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    previous_frame_key: bytes | None,
    ledger: Any,
) -> tuple[tuple[dict[str, Any], ...], bytes | None]:
    """Read and validate one index without touching its grid payload.

    The low-level read is intentional: opening a store must consume every
    shard's bounded index metadata in order, while the public access ledger
    still observes payload opens only when a grid shard is requested.
    """

    try:
        file_descriptor = os.open(path, os.O_RDONLY)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            os.close(file_descriptor)
        index_bytes = b"".join(chunks)
    except OSError as exc:
        raise CacheError("cache index file is missing or unreadable") from exc

    if len(index_bytes) != descriptor["index_bytes"]:
        raise CacheError("cache index byte count mismatch")
    if _sha256_bytes(index_bytes) != descriptor["index_sha256"]:
        raise CacheError("cache index SHA-256 mismatch")

    lines = index_bytes.splitlines(keepends=True)
    expected_rows = int(descriptor["rows"])
    if len(lines) != expected_rows:
        raise CacheError("cache index row count mismatch")
    index_rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.endswith(b"\n"):
            raise CacheError("cache index is truncated")
        raw = line[:-1]
        try:
            row = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CacheError("cache index row is invalid JSON") from exc
        if (
            not isinstance(row, dict)
            or raw != _canonical_json_bytes(row)
            or set(row) != _INDEX_FIELDS
        ):
            raise CacheError("cache index row is not canonical JSON")
        try:
            frame_key = bytes.fromhex(row["frame_key"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheError("cache index frame key is invalid") from exc
        if len(frame_key) != _FRAME_KEY_BYTES:
            raise CacheError("cache index frame key has the wrong length")
        if previous_frame_key is not None and frame_key <= previous_frame_key:
            raise CacheError("cache frame keys are duplicated or not strictly sorted")
        previous_frame_key = frame_key

        if not isinstance(row["canonical"], bool) or not isinstance(
            row["trajectory_alias"], bool
        ):
            raise CacheError("cache index canonical flags are invalid")
        if not _is_sha256(row["compressed_jpeg_sha256"]):
            raise CacheError("compressed JPEG identity is invalid")
        if not _is_sha256(row["decoded_rgb_sha256"]):
            raise CacheError("decoded RGB identity is invalid")
        if row["source_dataset"] not in {"R2R", "RxR"}:
            raise CacheError("only released StreamVLN rows may enter the cache")
        if row["revision"] != _STREAMVLN_REVISION:
            raise CacheError("StreamVLN revision does not match the frozen revision")
        if row["partition"] not in {"project-train", "project-dev"}:
            raise CacheError("cache partition must be project-train or project-dev")
        if not isinstance(row["building"], str) or not row["building"]:
            raise CacheError("building identity is required")
        if not isinstance(row["frame_preimage"], Mapping):
            raise CacheError("frame preimage must be a mapping")
        _canonical_json_bytes(dict(row["frame_preimage"]))
        ledger.update(line)
        index_rows.append(row)
    return tuple(index_rows), previous_frame_key


@dataclass
class _CacheStore:
    """Small random-access view over one already-admitted cache generation."""

    output: Path
    manifest: Mapping[str, Any]
    stage: str
    dtype: np.dtype[Any]
    shards: tuple[Mapping[str, Any], ...]
    starts: tuple[int, ...]
    index_rows: tuple[tuple[dict[str, Any], ...], ...]
    opened_shards: dict[int, _CacheShardState]

    def _open_shard(self, shard_number: int) -> _CacheShardState:
        cached = self.opened_shards.get(shard_number)
        if cached is not None:
            return cached

        descriptor = self.shards[shard_number]
        grid_path = self.output / str(descriptor["grid_path"])

        # The public cache contract requires a read-only memmap.  The call is
        # deliberately delayed until this shard is actually requested.
        try:
            mapped = np.load(grid_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise CacheError("cache grid is not a readable NumPy array") from exc
        if not isinstance(mapped, np.memmap):
            raise CacheError("cache reader must use a memory map")
        expected_shape = tuple(int(size) for size in descriptor["shape"])
        if tuple(mapped.shape) != expected_shape or mapped.dtype != self.dtype:
            raise CacheError("cache grid payload shape or dtype mismatch")
        if not mapped.flags.c_contiguous:
            raise CacheError("cache grid payload must be C-contiguous")

        try:
            if grid_path.stat().st_size != descriptor["grid_bytes"]:
                raise CacheError("cache grid byte count mismatch")
            if _sha256_path(grid_path) != descriptor["grid_sha256"]:
                raise CacheError("cache grid SHA-256 mismatch")
        except OSError as exc:
            raise CacheError("cache grid file is missing or unreadable") from exc

        # Validate every payload row without indexing the memmap.  This keeps
        # the bounded random-access promise (only requested rows are exposed
        # as NumPy objects) while retaining the existing row-SHA/finite gates.
        try:
            row_count = expected_shape[0]
            row_nbytes = int(np.prod(expected_shape[1:], dtype=np.int64)) * int(
                self.dtype.itemsize
            )
            data_offset = _memmap_data_offset(mapped)
            with grid_path.open("rb") as grid_handle:
                for local_index in range(row_count):
                    grid_handle.seek(data_offset + local_index * row_nbytes)
                    payload = grid_handle.read(row_nbytes)
                    if len(payload) != row_nbytes:
                        raise CacheError("cache grid payload is truncated")
                    if self.dtype.kind == "f":
                        values = np.frombuffer(payload, dtype=self.dtype)
                        if not np.isfinite(values).all():
                            raise CacheError("cache grid payload is non-finite")
        except OSError as exc:
            raise CacheError("cache grid payload could not be scanned") from exc

        # Index metadata was consumed and globally checked once by
        # ``open_cache_store``.  Reuse that bounded tuple here; only the grid
        # shard payload is opened lazily at first request.
        index_rows = self.index_rows[shard_number]
        try:
            data_offset = _memmap_data_offset(mapped)
            row_nbytes = int(np.prod(expected_shape[1:], dtype=np.int64)) * int(
                self.dtype.itemsize
            )
            with grid_path.open("rb") as grid_handle:
                for local_index, row in enumerate(index_rows):
                    grid_handle.seek(data_offset + local_index * row_nbytes)
                    payload = grid_handle.read(row_nbytes)
                    if _sha256_bytes(payload) != row["tensor_row_sha256"]:
                        raise CacheError("cache tensor row SHA-256 mismatch")
        except OSError as exc:
            raise CacheError("cache index or tensor row could not be validated") from exc

        state = _CacheShardState(
            descriptor=descriptor,
            grid=mapped,
            index_rows=tuple(index_rows),
        )
        self.opened_shards[shard_number] = state
        return state

    def read_global_rows(self, global_rows: Iterable[int]) -> list[CacheRecord]:
        """Read exactly the requested global rows in caller order."""

        try:
            requested_raw = list(global_rows)
        except (TypeError, ValueError) as exc:
            raise CacheError("global rows must be an iterable of integers") from exc

        requested: list[int] = []
        total_rows = int(self.manifest["rows"])
        for raw_index in requested_raw:
            if isinstance(raw_index, (bool, np.bool_)):
                raise CacheError("global row must be a non-bool integer")
            try:
                index = operator.index(raw_index)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CacheError("global row must be a non-bool integer") from exc
            if index < 0 or index >= total_rows:
                raise CacheError("global row is out of range")
            requested.append(int(index))

        if not requested:
            return []

        # Open each requested shard once, in first-request order.  The final
        # materialization loop below restores the exact caller order and keeps
        # duplicate occurrences.
        shard_for_row: dict[int, int] = {}
        for index in requested:
            if index in shard_for_row:
                continue
            shard_number = int(np.searchsorted(self.starts, index, side="right") - 1)
            shard_for_row[index] = shard_number
            self._open_shard(shard_number)

        result: list[CacheRecord] = []
        for index in requested:
            shard_number = shard_for_row[index]
            state = self.opened_shards[shard_number]
            local_index = index - self.starts[shard_number]
            row = state.index_rows[local_index]
            # Indexing is intentionally performed only for caller-requested
            # rows; returning this C-contiguous memmap view avoids whole-shard
            # copies and preserves the existing CacheRecord semantics.
            grid = state.grid[local_index]
            _validate_grid(grid, self.dtype, _spatial_shape(self.manifest.get("spatial_shape", (_TOKENS, _FEATURES))))
            try:
                frame_key = bytes.fromhex(row["frame_key"])
                record = CacheRecord(
                    frame_key=frame_key,
                    grid=grid,
                    compressed_jpeg_sha256=row["compressed_jpeg_sha256"],
                    decoded_rgb_sha256=row["decoded_rgb_sha256"],
                    source_dataset=row["source_dataset"],
                    revision=row["revision"],
                    building=row["building"],
                    partition=row["partition"],
                    frame_preimage=row["frame_preimage"],
                    canonical=row["canonical"],
                    trajectory_alias=row["trajectory_alias"],
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise CacheError("cache index row is missing provenance") from exc
            _validate_record_metadata(record, whitening_fit=False)
            result.append(record)
        return result

    def iter_index_metadata(self) -> Iterator[tuple[int, Mapping[str, Any]]]:
        """Yield validated index rows in global canonical order.

        ``open_cache_store`` has already consumed and authenticated every
        shard index before returning this store.  This view reuses those
        bounded metadata tuples and deliberately never opens or materializes
        a grid payload; callers can build trajectory/cache locators without
        paying for tensor reads.
        """

        for shard_number, rows in enumerate(self.index_rows):
            start = self.starts[shard_number]
            for local_index, row in enumerate(rows):
                yield start + local_index, row


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CacheError("metadata is not canonical-JSON serializable") from exc


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json_bytes(value) + b"\n")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HASH_CHARS for character in value)
    )


def _require_sha256(value: object, name: str) -> str:
    if not _is_sha256(value):
        raise CacheError(f"{name} must be a lowercase SHA-256")
    return value


def _little_c_array(value: object, dtype: str) -> np.ndarray:
    return np.asarray(value, dtype=dtype, order="C")


def _array_sha256(value: object, dtype: str) -> str:
    array = _little_c_array(value, dtype)
    return _sha256_bytes(array.tobytes(order="C"))


def _absolute_peak_rss_bytes() -> int:
    """Return the Linux process-lifetime RSS high-water mark in bytes."""

    if os.name != "posix" or not Path("/proc/self").is_dir():
        raise CacheError("absolute peak RSS accounting requires the Linux training host")
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _validate_transform(transform: WhiteningTransform) -> WhiteningTransform:
    if not isinstance(transform, WhiteningTransform):
        raise CacheError("whitening transform has the wrong type")
    mu = _little_c_array(transform.mu, "<f8")
    whitener = _little_c_array(transform.whitener, "<f8")
    eigenvalues = _little_c_array(transform.eigenvalues, "<f8")
    shrinkage = float(transform.shrinkage)
    if mu.shape != (_FEATURES,):
        raise CacheError("whitening mean must have shape [768]")
    if whitener.shape != (_FEATURES, _FEATURES):
        raise CacheError("whitener must have shape [768,768]")
    if eigenvalues.shape != (_FEATURES,):
        raise CacheError("whitening eigenvalues must have shape [768]")
    if not (
        np.isfinite(mu).all()
        and np.isfinite(whitener).all()
        and np.isfinite(eigenvalues).all()
        and np.isfinite(shrinkage)
    ):
        raise CacheError("whitening transform must be finite")
    if np.min(eigenvalues) <= 0.0:
        raise CacheError("whitening covariance must be strictly positive definite")
    if not 0.0 <= shrinkage <= 1.0:
        raise CacheError("Ledoit-Wolf shrinkage must lie in [0,1]")
    return WhiteningTransform(
        mu=mu,
        whitener=whitener,
        eigenvalues=eigenvalues,
        shrinkage=shrinkage,
        artifact_sha256=transform.artifact_sha256,
    )


def _validate_record_metadata(record: CacheRecord, *, whitening_fit: bool) -> None:
    if not isinstance(record, CacheRecord):
        raise CacheError("cache rows must be CacheRecord instances")
    if not isinstance(record.frame_key, bytes) or len(record.frame_key) != _FRAME_KEY_BYTES:
        raise CacheError("frame_key must contain raw32 trajectory identity plus uint32 step")
    _require_sha256(record.compressed_jpeg_sha256, "compressed JPEG identity")
    _require_sha256(record.decoded_rgb_sha256, "decoded RGB identity")
    if record.source_dataset not in {"R2R", "RxR"}:
        raise CacheError("only released StreamVLN R2R/RxR rows may enter the cache")
    if record.revision != _STREAMVLN_REVISION:
        raise CacheError("StreamVLN revision does not match the frozen revision")
    if record.partition not in {"project-train", "project-dev"}:
        raise CacheError("cache partition must be project-train or project-dev")
    if whitening_fit and record.partition != "project-train":
        raise CacheError("whitening may only fit project-train frame occurrences")
    if not record.canonical or record.trajectory_alias:
        raise CacheError("trajectory aliases are provenance only, not cache occurrences")
    if not isinstance(record.building, str) or not record.building:
        raise CacheError("building identity is required")
    if not isinstance(record.frame_preimage, Mapping):
        raise CacheError("frame preimage must be a mapping")
    _canonical_json_bytes(dict(record.frame_preimage))


def _spatial_shape(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
        raise CacheError("spatial shape must contain two positive integers")
    return tuple(value)


def _validate_grid(grid: object, dtype: np.dtype[Any], spatial_shape: tuple[int, int] = (_TOKENS, _FEATURES)) -> np.ndarray:
    if not isinstance(grid, np.ndarray):
        raise CacheError("spatial grid must be a NumPy array")
    if grid.shape != spatial_shape:
        raise CacheError(f"spatial grid must have shape {spatial_shape}")
    if grid.dtype != dtype:
        raise CacheError(f"spatial grid must have dtype {dtype.str}")
    if not grid.flags.c_contiguous:
        raise CacheError("spatial grid must be C-contiguous")
    if dtype.kind == "f" and not np.isfinite(grid).all():
        raise CacheError("spatial grid must be finite")
    return grid


def _ordered_records(
    records: Iterable[CacheRecord],
    *,
    dtype: np.dtype[Any],
    whitening_fit: bool,
    spatial_shape: tuple[int, int] = (_TOKENS, _FEATURES),
) -> list[CacheRecord]:
    materialized = list(records)
    if not materialized:
        raise CacheError("at least one frame occurrence is required")
    for record in materialized:
        _validate_record_metadata(record, whitening_fit=whitening_fit)
        _validate_grid(record.grid, dtype, spatial_shape)
    materialized.sort(key=lambda record: record.frame_key)
    for previous, current in zip(materialized, materialized[1:]):
        if previous.frame_key == current.frame_key:
            raise CacheError("frame keys must be unique")
    return materialized


def fit_whitening(
    records: Iterable[CacheRecord],
    *,
    workspace_root: str | os.PathLike[str],
    resource_limits: Mapping[str, int],
) -> tuple[WhiteningTransform, dict[str, Any]]:
    """Fit the frozen Ledoit-Wolf transform from canonical train occurrences."""

    if sklearn.__version__ != _SKLEARN_VERSION:
        raise CacheError(
            f"scikit-learn {_SKLEARN_VERSION} is required for whitening"
        )
    ordered = _ordered_records(
        records,
        dtype=np.dtype("<f4"),
        whitening_fit=True,
    )
    n_train = len(ordered)
    try:
        configured_temp_bytes = int(resource_limits["configured_temp_bytes"])
        host_available = int(resource_limits["host_available_bytes"])
        workspace_available = int(resource_limits["workspace_available_bytes"])
        peak_rss_ceiling = int(resource_limits["peak_rss_ceiling_bytes"])
        peak_disk_ceiling = int(resource_limits["peak_disk_ceiling_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheError("resource_limits is incomplete") from exc
    if min(
        configured_temp_bytes,
        host_available,
        workspace_available,
        peak_rss_ceiling,
        peak_disk_ceiling,
    ) < 0:
        raise CacheError("resource limits must be non-negative")

    x_bytes = 8 * n_train * _FEATURES
    known_host = 3 * x_bytes + 6 * 8 * _FEATURES * _FEATURES
    known_disk = x_bytes + configured_temp_bytes
    if host_available < known_host:
        raise CacheError("available host memory is below the declared scientific bound")
    if workspace_available < known_disk:
        raise CacheError("available workspace is below the declared scientific bound")
    if peak_rss_ceiling < known_host or peak_disk_ceiling < known_disk:
        raise CacheError("configured peak ceiling is below the known lower bound")
    if _absolute_peak_rss_bytes() > peak_rss_ceiling:
        raise CacheError("absolute process peak RSS already exceeds the research ceiling")

    workspace = Path(workspace_root).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    x_path = workspace / "whitening-X.npy"
    if x_path.exists():
        raise CacheError("whitening workspace already contains whitening-X.npy")

    started = time.perf_counter()
    x = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype="<f8",
        shape=(n_train, _FEATURES),
        fortran_order=False,
    )
    for row_index, record in enumerate(ordered):
        grid64 = np.asarray(record.grid, dtype="<f8", order="C")
        x[row_index] = grid64.mean(axis=0, dtype=np.float64)
    x.flush()

    with threadpool_limits(limits=1):
        estimator = LedoitWolf(
            store_precision=False,
            assume_centered=False,
            block_size=1000,
        ).fit(x)
        covariance = np.asarray(estimator.covariance_, dtype="<f8", order="C")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance, UPLO="L")
        if not (
            np.isfinite(estimator.location_).all()
            and np.isfinite(covariance).all()
            and np.isfinite(eigenvalues).all()
            and np.isfinite(eigenvectors).all()
            and np.isfinite(estimator.shrinkage_)
        ):
            raise CacheError("Ledoit-Wolf whitening produced non-finite values")
        if not 0.0 <= float(estimator.shrinkage_) <= 1.0:
            raise CacheError("Ledoit-Wolf shrinkage lies outside [0,1]")
        if float(np.min(eigenvalues)) <= 0.0:
            raise CacheError("whitening covariance is not strictly positive definite")
        inverse_sqrt = (eigenvalues.astype(np.float64) ** -0.5)[None, :]
        whitener = (eigenvectors * inverse_sqrt) @ eigenvectors.T
        pools = threadpool_info()
    if not pools or any(int(pool.get("num_threads", -1)) != 1 for pool in pools):
        raise CacheError("numeric thread pools were not restricted to one thread")

    transform = _validate_transform(
        WhiteningTransform(
            mu=np.asarray(estimator.location_, dtype="<f8", order="C"),
            whitener=np.asarray(whitener, dtype="<f8", order="C"),
            eigenvalues=np.asarray(eigenvalues, dtype="<f8", order="C"),
            shrinkage=float(estimator.shrinkage_),
        )
    )
    elapsed = time.perf_counter() - started
    observed_disk = x_path.stat().st_size + configured_temp_bytes
    if observed_disk > peak_disk_ceiling:
        raise CacheError("observed workspace bytes exceeded the configured research ceiling")

    stat = os.statvfs(workspace)
    receipt: dict[str, Any] = {
        "schema": "J2J_WHITENING_FIT_V1",
        "workspace_root": str(workspace),
        "workspace_filesystem": {
            "block_size": int(stat.f_frsize),
            "available_bytes_at_receipt": int(stat.f_bavail * stat.f_frsize),
        },
        "x_path": x_path.name,
        "x_shape": [n_train, _FEATURES],
        "x_bytes": x_bytes,
        "known_host_bytes": known_host,
        "known_disk_bytes": known_disk,
        "observed_disk_bytes": observed_disk,
        "elapsed_seconds": elapsed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "ledoit_wolf": {
            "version": sklearn.__version__,
            "store_precision": False,
            "assume_centered": False,
            "block_size": 1000,
            "covariance_denominator": "N",
        },
        "threadpools": pools,
    }
    observed_rss = _absolute_peak_rss_bytes()
    if observed_rss > peak_rss_ceiling:
        raise CacheError("absolute process peak RSS exceeded the research ceiling")
    receipt["observed_rss_bytes"] = observed_rss
    return transform, receipt


def apply_whitening(grid: np.ndarray, transform: WhiteningTransform) -> np.ndarray:
    """Apply the fixed CPU float64 whitening equation and cast once to Z32."""

    source = _validate_grid(grid, np.dtype("<f4"))
    frozen = _validate_transform(transform)
    u64 = np.asarray(source, dtype="<f8", order="C")
    y64 = np.einsum(
        "pd,ed->pe",
        u64 - frozen.mu,
        frozen.whitener,
        dtype=np.float64,
        order="C",
        optimize=False,
    )
    z32 = np.asarray(y64, dtype="<f4", order="C")
    if not np.isfinite(z32).all():
        raise CacheError("whitened Z32 grid is non-finite")
    return z32


def validate_reencode_pilot(
    reference: np.ndarray,
    replay: np.ndarray,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Validate one run's repeated float32 pilot within the frozen tolerance."""

    if rtol != _PILOT_RTOL or atol != _PILOT_ATOL:
        raise CacheError("re-encoding pilot tolerance is fixed by the R6 contract")
    left = np.asarray(reference)
    right = np.asarray(replay)
    if left.shape != right.shape or left.ndim != 3 or left.shape[1:] != (_TOKENS, _FEATURES):
        raise CacheError("pilot tensors must share shape [B,36,768]")
    if left.dtype != np.dtype("<f4") or right.dtype != np.dtype("<f4"):
        raise CacheError("pilot tensors must be float32")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise CacheError("pilot tensors must be finite")
    try:
        np.testing.assert_allclose(
            left,
            right,
            rtol=_PILOT_RTOL,
            atol=_PILOT_ATOL,
        )
    except AssertionError as exc:
        raise CacheError("re-encoding pilot exceeded the frozen tolerance") from exc


def save_whitening(
    path: str | os.PathLike[str],
    transform: WhiteningTransform,
) -> dict[str, Any]:
    """Save a pickle-free whitening artifact and return its scientific hashes."""

    frozen = _validate_transform(transform)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            schema=np.asarray("J2J_WHITENING_V1"),
            mu=frozen.mu,
            W=frozen.whitener,
            eigenvalues=frozen.eigenvalues,
            shrinkage=np.asarray(frozen.shrinkage, dtype="<f8"),
        )
    os.replace(temporary, target)
    return {
        "schema": "J2J_WHITENING_ARTIFACT_V1",
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": _sha256_path(target),
        "array_sha256": {
            "mu": _array_sha256(frozen.mu, "<f8"),
            "W": _array_sha256(frozen.whitener, "<f8"),
            "eigenvalues": _array_sha256(frozen.eigenvalues, "<f8"),
            "shrinkage": _array_sha256(frozen.shrinkage, "<f8"),
        },
    }


def load_whitening(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> WhiteningTransform:
    """Load and validate a pickle-free whitening artifact."""

    source = Path(path)
    if not source.is_file():
        raise CacheError("whitening artifact is missing")
    actual_sha256 = _sha256_path(source)
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected whitening identity")
        if actual_sha256 != expected_sha256:
            raise CacheError("whitening artifact SHA-256 mismatch")
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"schema", "mu", "W", "eigenvalues", "shrinkage"}:
                raise CacheError("whitening artifact has an unexpected field ledger")
            if archive["schema"].shape != () or archive["schema"].item() != "J2J_WHITENING_V1":
                raise CacheError("whitening schema mismatch")
            raw_arrays = {
                "mu": (archive["mu"], (_FEATURES,)),
                "W": (archive["W"], (_FEATURES, _FEATURES)),
                "eigenvalues": (archive["eigenvalues"], (_FEATURES,)),
                "shrinkage": (archive["shrinkage"], ()),
            }
            for name, (raw, expected_shape) in raw_arrays.items():
                if raw.shape != expected_shape:
                    raise CacheError(f"stored whitening {name} has the wrong shape")
                if raw.dtype != np.dtype("<f8"):
                    raise CacheError(
                        f"stored whitening {name} must be native little-endian float64"
                    )
                if not raw.flags.c_contiguous:
                    raise CacheError(f"stored whitening {name} must be C-contiguous")
                if not np.isfinite(raw).all():
                    raise CacheError(f"stored whitening {name} must be finite")
            transform = WhiteningTransform(
                mu=raw_arrays["mu"][0].copy(order="C"),
                whitener=raw_arrays["W"][0].copy(order="C"),
                eigenvalues=raw_arrays["eigenvalues"][0].copy(order="C"),
                shrinkage=float(raw_arrays["shrinkage"][0].item()),
                artifact_sha256=actual_sha256 if expected_sha256 is not None else None,
            )
    except (OSError, ValueError, KeyError) as exc:
        raise CacheError("whitening artifact could not be read") from exc
    return _validate_transform(transform)


def _index_row(record: CacheRecord, grid: np.ndarray) -> dict[str, Any]:
    return {
        "frame_key": record.frame_key.hex(),
        "compressed_jpeg_sha256": record.compressed_jpeg_sha256,
        "decoded_rgb_sha256": record.decoded_rgb_sha256,
        "source_dataset": record.source_dataset,
        "revision": record.revision,
        "building": record.building,
        "partition": record.partition,
        "frame_preimage": dict(record.frame_preimage),
        "canonical": record.canonical,
        "trajectory_alias": record.trajectory_alias,
        "tensor_row_sha256": _sha256_bytes(grid.tobytes(order="C")),
    }


def _frame_ledger_sha256(
    records: Sequence[CacheRecord],
    grids: Iterable[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    count = 0
    for record, grid in zip(records, grids):
        digest.update(_canonical_json_bytes(_index_row(record, grid)) + b"\n")
        count += 1
    if count != len(records):
        raise CacheError("frame ledger row count mismatch")
    return digest.hexdigest()


def _validate_run_config(run_config: Mapping[str, Any]) -> dict[str, Any]:
    required = {"cache_batch", "devices", "workers", "workspace", "output"}
    if not isinstance(run_config, Mapping) or set(run_config) != required:
        raise CacheError("run_config must parameterize batch, devices, workers and paths")
    if type(run_config["cache_batch"]) is not int or run_config["cache_batch"] <= 0:
        raise CacheError("cache_batch must be positive")
    if type(run_config["workers"]) is not int or run_config["workers"] < 0:
        raise CacheError("workers must be non-negative")
    if not isinstance(run_config["devices"], list) or not run_config["devices"]:
        raise CacheError("devices must be a non-empty configured list")
    if not all(isinstance(device, str) and device for device in run_config["devices"]):
        raise CacheError("configured devices must be non-empty strings")
    if not all(isinstance(run_config[name], str) and run_config[name] for name in ("workspace", "output")):
        raise CacheError("workspace and output must be configured paths")
    result = dict(run_config)
    _canonical_json_bytes(result)
    return result


def _validate_reconstruction_command(command: Sequence[str]) -> list[str]:
    if (
        not isinstance(command, Sequence)
        or isinstance(command, (str, bytes))
        or not command
        or not all(isinstance(part, str) and part for part in command)
    ):
        raise CacheError("reconstruction command must be a non-empty argv list")
    return list(command)


def _validate_identity_ledger(
    identities: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    if not isinstance(identities, Mapping) or set(identities) != _IDENTITY_FIELDS:
        raise CacheError("cache identities have an unexpected ledger")
    if identities["vjepa_source_commit"] != _VJEPA_SOURCE_COMMIT:
        raise CacheError("V-JEPA source identity mismatch")
    for name in ("checkpoint_sha256", "preprocess_sha256", "pool_sha256"):
        _require_sha256(identities[name], name)
    whitening_sha = identities["whitening_sha256"]
    if stage in {"U", "RAW32"}:
        if whitening_sha is not None:
            raise CacheError("raw U cache must not carry a whitening identity")
    else:
        _require_sha256(whitening_sha, "whitening identity")
    result = dict(identities)
    _canonical_json_bytes(result)
    return result


def _validate_identities(
    identities: Mapping[str, Any],
    *,
    stage: str,
    whitening_transform: WhiteningTransform | None,
) -> dict[str, Any]:
    result = _validate_identity_ledger(identities, stage=stage)
    whitening_sha = result["whitening_sha256"]
    if stage in {"U", "RAW32"}:
        if whitening_transform is not None:
            raise CacheError("raw U cache must not carry a whitening transform")
    elif stage == "Z32":
        if whitening_transform is None:
            raise CacheError("Z32 cache requires a frozen whitening transform")
        frozen = _validate_transform(whitening_transform)
        if frozen.artifact_sha256 is None or frozen.artifact_sha256 != whitening_sha:
            raise CacheError("Z32 whitening artifact does not match its declared identity")
    else:
        if whitening_transform is not None:
            raise CacheError("low-precision derivatives do not refit whitening")
    return result


def _validate_parent(
    parent: Mapping[str, Any],
    *,
    stage: str,
    records: Sequence[CacheRecord],
    identities: Mapping[str, Any],
    production_eligible: bool,
) -> str:
    if not isinstance(parent, Mapping):
        raise CacheError("parent manifest must be a mapping")
    parent_dict = dict(parent)
    parent_sha256 = _sha256_bytes(_canonical_json_bytes(parent_dict))
    if stage in {"U", "RAW32"}:
        if parent_dict.get("schema") != "J2J_TASK4_CACHE_PARENT_V1":
            raise CacheError("U cache requires the Task 4 factual parent")
        if parent_dict.get("streamvln_revision") != _STREAMVLN_REVISION:
            raise CacheError("Task 4 parent revision mismatch")
        for name in (
            "asset_manifest_sha256",
            "annotation_ledger_sha256",
            "split_ledger_sha256",
            "canonical_trajectory_ledger_sha256",
        ):
            _require_sha256(parent_dict.get(name), f"parent {name}")
        if production_eligible and not (
            parent_dict.get("production_eligible") is True
            and parent_dict.get("asset_scope") == "official"
            and parent_dict.get("audit_kind") == "production_post_rgb"
        ):
            raise CacheError("synthetic parents cannot produce a production cache")
    elif stage == "Z32":
        if parent_dict.get("schema") != "J2J_LATENT_CACHE_V1" or parent_dict.get("stage") != "U":
            raise CacheError("Z32 cache requires a U-cache parent")
        parent_identities = _validate_identity_ledger(
            parent_dict.get("identities"),
            stage="U",
        )
        if any(
            parent_identities[name] != identities[name]
            for name in _SHARED_ENCODING_IDENTITIES
        ):
            raise CacheError("Z32 encoding identity differs from its parent U cache")
        expected_ledger = parent_dict.get("frame_ledger_sha256")
        _require_sha256(expected_ledger, "parent U frame ledger")
        actual_ledger = _frame_ledger_sha256(records, (record.grid for record in records))
        if actual_ledger != expected_ledger:
            raise CacheError("Z32 input rows do not match the parent U cache")
        if production_eligible and parent_dict.get("production_eligible") is not True:
            raise CacheError("a non-production U parent cannot produce production Z32")
    else:
        if parent_dict.get("schema") != "J2J_LATENT_CACHE_V1" or parent_dict.get("stage") != "Z32":
            raise CacheError("low-precision cache requires a Z32 parent")
        parent_identities = _validate_identity_ledger(
            parent_dict.get("identities"),
            stage="Z32",
        )
        if parent_identities != dict(identities):
            raise CacheError("low-precision identity differs from its parent Z32 cache")
        expected_ledger = parent_dict.get("frame_ledger_sha256")
        _require_sha256(expected_ledger, "parent Z32 frame ledger")
        actual_ledger = _frame_ledger_sha256(
            records,
            (record.grid for record in records),
        )
        if actual_ledger != expected_ledger:
            raise CacheError("low-precision input rows do not match the parent Z32 cache")
        if parent_dict.get("training_eligible") is not True:
            raise CacheError("low-precision parent must be an eligible Z32 cache")
        if production_eligible and parent_dict.get("production_eligible") is not True:
            raise CacheError("a non-production Z32 parent cannot produce a production derivative")
    return parent_sha256


def write_cache(
    output_dir: str | os.PathLike[str],
    records: Iterable[CacheRecord],
    *,
    stage: str,
    shard_rows: int,
    run_config: Mapping[str, Any],
    identities: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    whitening_transform: WhiteningTransform | None,
    trajectory_alias_ledger_sha256: str,
    reconstruction_command: Sequence[str],
    production_eligible: bool,
    spatial_shape: tuple[int, int] = (_TOKENS, _FEATURES),
) -> dict[str, Any]:
    """Write one portable cache generation, making the manifest visible last."""

    if stage not in _STAGE_DTYPES:
        raise CacheError("cache stage must be U, Z32, Z16 or BF16")
    if not isinstance(shard_rows, int) or shard_rows <= 0:
        raise CacheError("shard_rows must be a positive configured integer")
    if not isinstance(production_eligible, bool):
        raise CacheError("production_eligible must be boolean")
    spatial_shape = _spatial_shape(spatial_shape)
    dtype = _STAGE_DTYPES[stage]
    ordered = _ordered_records(
        records,
        dtype=np.dtype("<f4"),
        whitening_fit=False,
        spatial_shape=spatial_shape,
    )
    configured_run = _validate_run_config(run_config)
    configured_identities = _validate_identities(
        identities,
        stage=stage,
        whitening_transform=whitening_transform,
    )
    _require_sha256(trajectory_alias_ledger_sha256, "trajectory alias ledger")
    configured_command = _validate_reconstruction_command(reconstruction_command)
    parent_sha256 = _validate_parent(
        parent_manifest,
        stage=stage,
        records=ordered,
        identities=configured_identities,
        production_eligible=production_eligible,
    )

    frozen_transform = (
        _validate_transform(whitening_transform)
        if whitening_transform is not None
        else None
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise CacheError("cache generation already has a visible manifest")

    shards: list[dict[str, Any]] = []
    cache_ledger = hashlib.sha256()
    for shard_number, start in enumerate(range(0, len(ordered), shard_rows)):
        shard_records = ordered[start : start + shard_rows]
        shard_id = f"shard-{shard_number:06d}"
        grid_name = f"{shard_id}.grid.npy"
        index_name = f"{shard_id}.index.jsonl"
        grid_path = output / grid_name
        index_path = output / index_name
        grid_tmp = output / f"{grid_name}.tmp"
        index_tmp = output / f"{index_name}.tmp"

        mapped = np.lib.format.open_memmap(
            grid_tmp,
            mode="w+",
            dtype=dtype,
            shape=(len(shard_records), *spatial_shape),
            fortran_order=False,
        )
        index_rows: list[dict[str, Any]] = []
        for local_index, record in enumerate(shard_records):
            if stage == "Z32":
                assert frozen_transform is not None
                result_grid = apply_whitening(record.grid, frozen_transform)
            elif stage == "Z16":
                result_grid = np.asarray(record.grid, dtype="<f2", order="C")
            elif stage == "BF16":
                import torch

                result_grid = (
                    torch.from_numpy(record.grid)
                    .to(torch.bfloat16)
                    .view(torch.uint16)
                    .numpy()
                    .astype("<u2", copy=False)
                )
            else:
                result_grid = record.grid
            mapped[local_index] = result_grid
            stored_grid = np.asarray(mapped[local_index], dtype=dtype, order="C")
            row = _index_row(record, stored_grid)
            index_rows.append(row)
            cache_ledger.update(_canonical_json_bytes(row) + b"\n")
        mapped.flush()
        del mapped

        with index_tmp.open("wb") as handle:
            for row in index_rows:
                handle.write(_canonical_json_bytes(row) + b"\n")
        os.replace(grid_tmp, grid_path)
        os.replace(index_tmp, index_path)
        shards.append(
            {
                "shard_id": shard_id,
                "grid_path": grid_name,
                "index_path": index_name,
                "rows": len(shard_records),
                "shape": [len(shard_records), *spatial_shape],
                "dtype": dtype.str,
                "grid_bytes": grid_path.stat().st_size,
                "grid_sha256": _sha256_path(grid_path),
                "index_bytes": index_path.stat().st_size,
                "index_sha256": _sha256_path(index_path),
            }
        )

    manifest: dict[str, Any] = {
        "schema": "J2J_LATENT_CACHE_V1",
        "stage": stage,
        "dtype": dtype.str,
        "rows": len(ordered),
        "shard_rows": shard_rows,
        "shards": shards,
        "parent_manifest_sha256": parent_sha256,
        "frame_ledger_sha256": cache_ledger.hexdigest(),
        "trajectory_alias_ledger_sha256": trajectory_alias_ledger_sha256,
        "identities": configured_identities,
        "run_config": configured_run,
        "production_eligible": production_eligible,
        "training_eligible": stage in {"Z32", "RAW32"},
        "reconstruction_command": configured_command,
    }
    if spatial_shape != (_TOKENS, _FEATURES):
        manifest["spatial_shape"] = list(spatial_shape)
    if stage == "BF16":
        manifest["logical_dtype"] = "bfloat16"
    manifest_tmp = output / "manifest.json.tmp"
    _write_canonical_json(manifest_tmp, manifest)
    os.replace(manifest_tmp, output / "manifest.json")
    return manifest


def _validate_stream_parent_static(
    parent: Mapping[str, Any],
    *,
    stage: str,
    identities: Mapping[str, Any],
    production_eligible: bool,
) -> tuple[dict[str, Any], str]:
    """Validate parent fields that do not require materializing input rows.

    ``write_cache`` keeps its historical sequence-based validation.  The
    bounded writer validates the same parent schema in two pieces: static
    fields here, then the input frame ledger while records are streamed.
    """

    if not isinstance(parent, Mapping):
        raise CacheError("parent manifest must be a mapping")

    if stage in {"U", "RAW32"}:
        parent_dict = dict(parent)
        return parent_dict, _validate_parent(
            parent_dict,
            stage=stage,
            records=(),
            identities=identities,
            production_eligible=production_eligible,
        )

    parent_dict = dict(parent)
    parent_sha256 = _sha256_bytes(_canonical_json_bytes(parent_dict))
    expected_parent_stage = "U" if stage == "Z32" else "Z32"
    if (
        parent_dict.get("schema") != "J2J_LATENT_CACHE_V1"
        or parent_dict.get("stage") != expected_parent_stage
    ):
        if stage == "Z32":
            raise CacheError("Z32 cache requires a U-cache parent")
        raise CacheError("low-precision cache requires a Z32 parent")

    parent_identities = _validate_identity_ledger(
        parent_dict.get("identities"),
        stage=expected_parent_stage,
    )
    if any(
        parent_identities[name] != identities[name]
        for name in (
            _SHARED_ENCODING_IDENTITIES
            if stage == "Z32"
            else _IDENTITY_FIELDS
        )
    ):
        if stage == "Z32":
            raise CacheError("Z32 encoding identity differs from its parent U cache")
        raise CacheError("low-precision identity differs from its parent Z32 cache")

    _require_sha256(
        parent_dict.get("frame_ledger_sha256"),
        f"parent {expected_parent_stage} frame ledger",
    )
    if stage != "Z32":
        if parent_dict.get("training_eligible") is not True:
            raise CacheError("low-precision parent must be an eligible Z32 cache")
    if production_eligible and parent_dict.get("production_eligible") is not True:
        if stage == "Z32":
            raise CacheError("a non-production U parent cannot produce production Z32")
        raise CacheError("a non-production Z32 parent cannot produce a production derivative")
    return parent_dict, parent_sha256


def _stream_result_grid(
    record: CacheRecord,
    *,
    stage: str,
    dtype: np.dtype[Any],
    whitening_transform: WhiteningTransform | None,
) -> np.ndarray:
    """Convert one float32 input grid using the existing cache equations."""

    if stage == "Z32":
        if whitening_transform is None:  # pragma: no cover - guarded by validation
            raise CacheError("Z32 cache requires a frozen whitening transform")
        return apply_whitening(record.grid, whitening_transform)
    if stage == "Z16":
        return np.asarray(record.grid, dtype="<f2", order="C")
    if stage == "BF16":
        import torch

        return (
            torch.from_numpy(record.grid)
            .to(torch.bfloat16)
            .view(torch.uint16)
            .numpy()
            .astype("<u2", copy=False)
        )
    if stage in {"U", "RAW32"}:
        return record.grid
    raise CacheError(f"unsupported streaming cache stage: {stage}")


def write_cache_streaming(
    output_dir: str | os.PathLike[str],
    records: Iterable[CacheRecord],
    *,
    expected_rows: int,
    stage: str,
    shard_rows: int,
    run_config: Mapping[str, Any],
    identities: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    whitening_transform: WhiteningTransform | None,
    trajectory_alias_ledger_sha256: str,
    reconstruction_command: Sequence[str],
    production_eligible: bool,
    spatial_shape: tuple[int, int] = (_TOKENS, _FEATURES),
    parallel_workers: int = 1,
    parallel_window_rows: int | None = None,
    shard_subdirs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write a cache from an already canonical, frame-key-sorted iterable.

    Unlike :func:`write_cache`, this entry point never sorts or materializes
    the input iterable.  ``expected_rows`` comes from the admitted census and
    each shard's NumPy payload is preallocated before its rows are consumed.
    The on-disk schema, row bytes, ledger formula, and parent identities are
    identical to ``write_cache`` for the same sorted input.
    """

    if stage not in _STAGE_DTYPES:
        raise CacheError("cache stage must be U, Z32, Z16 or BF16")
    if type(expected_rows) is not int or expected_rows <= 0:
        raise CacheError("expected_rows must be a positive configured integer")
    if type(shard_rows) is not int or shard_rows <= 0:
        raise CacheError("shard_rows must be a positive configured integer")
    if not isinstance(production_eligible, bool):
        raise CacheError("production_eligible must be boolean")
    if type(parallel_workers) is not int or parallel_workers <= 0:
        raise CacheError("parallel_workers must be a positive integer")
    if parallel_window_rows is None:
        parallel_window_rows = shard_rows
    if type(parallel_window_rows) is not int or parallel_window_rows <= 0:
        raise CacheError("parallel_window_rows must be a positive integer")
    if parallel_window_rows % shard_rows != 0:
        raise CacheError("parallel_window_rows must be divisible by shard_rows")
    if parallel_workers > 1 and stage not in {"U", "RAW32"}:
        raise CacheError("parallel cache writing currently supports only U and RAW32")

    spatial_shape = _spatial_shape(spatial_shape)
    input_dtype = np.dtype("<f4")
    dtype = _STAGE_DTYPES[stage]
    configured_run = _validate_run_config(run_config)
    configured_identities = _validate_identities(
        identities,
        stage=stage,
        whitening_transform=whitening_transform,
    )
    _require_sha256(trajectory_alias_ledger_sha256, "trajectory alias ledger")
    configured_command = _validate_reconstruction_command(reconstruction_command)
    parent_dict, parent_sha256 = _validate_stream_parent_static(
        parent_manifest,
        stage=stage,
        identities=configured_identities,
        production_eligible=production_eligible,
    )
    frozen_transform = (
        _validate_transform(whitening_transform)
        if whitening_transform is not None
        else None
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise CacheError("cache generation already has a visible manifest")

    if shard_subdirs is not None:
        if not isinstance(shard_subdirs, (list, tuple)) or not shard_subdirs:
            raise CacheError("shard_subdirs must be a nonempty sequence")
        for name in shard_subdirs:
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise CacheError("shard_subdirs must be relative output directories")
            (output / name).mkdir(parents=True, exist_ok=True)
    def shard_prefix(number):
        name = f"shard-{number:06d}"
        return str(Path(shard_subdirs[number % len(shard_subdirs)]) / name) if shard_subdirs else name

    generated_paths: list[Path] = []
    shards: list[dict[str, Any]] = []
    cache_ledger = hashlib.sha256()
    input_ledger = hashlib.sha256()
    previous_frame_key: bytes | None = None
    iterator = iter(records)
    consumed = 0

    try:
        shard_count = (expected_rows + shard_rows - 1) // shard_rows
        if parallel_workers == 1:
            for shard_number in range(shard_count):
                rows_in_shard = min(shard_rows, expected_rows - consumed)
                shard_id = f"shard-{shard_number:06d}"
                grid_name = f"{shard_prefix(shard_number)}.grid.npy"
                index_name = f"{shard_prefix(shard_number)}.index.jsonl"
                grid_path = output / grid_name
                index_path = output / index_name
                grid_tmp = output / f"{grid_name}.tmp"
                index_tmp = output / f"{index_name}.tmp"
                if any(path.exists() for path in (grid_path, index_path, grid_tmp, index_tmp)):
                    raise CacheError("cache generation output contains a stale shard")

                mapped = np.lib.format.open_memmap(
                    grid_tmp,
                    mode="w+",
                    dtype=dtype,
                    shape=(rows_in_shard, *spatial_shape),
                    fortran_order=False,
                )
                generated_paths.extend((grid_tmp, index_tmp, grid_path, index_path))
                try:
                    with index_tmp.open("wb") as index_handle:
                        for local_index in range(rows_in_shard):
                            try:
                                record = next(iterator)
                            except StopIteration as exc:
                                raise CacheError("cache input ended before expected_rows") from exc
                            if not isinstance(record, CacheRecord):
                                raise CacheError("cache rows must be CacheRecord instances")
                            _validate_record_metadata(record, whitening_fit=False)
                            _validate_grid(record.grid, input_dtype, spatial_shape)
                            if (
                                previous_frame_key is not None
                                and record.frame_key <= previous_frame_key
                            ):
                                raise CacheError(
                                    "streaming cache input must be strictly sorted by frame_key"
                                )
                            previous_frame_key = record.frame_key

                            # For derived stages this is the exact input-parent
                            # ledger used by the sequence writer, computed before
                            # any stage cast.  Raw U has no record-dependent
                            # parent ledger check, so avoid a second per-row hash
                            # pass there.
                            if stage not in {"U", "RAW32"}:
                                input_row = _index_row(record, record.grid)
                                input_ledger.update(
                                    _canonical_json_bytes(input_row) + b"\n"
                                )

                            result_grid = _stream_result_grid(
                                record,
                                stage=stage,
                                dtype=dtype,
                                whitening_transform=frozen_transform,
                            )
                            mapped[local_index] = result_grid
                            stored_grid = np.asarray(
                                mapped[local_index], dtype=dtype, order="C"
                            )
                            row = _index_row(record, stored_grid)
                            line = _canonical_json_bytes(row) + b"\n"
                            index_handle.write(line)
                            cache_ledger.update(line)
                            consumed += 1
                    mapped.flush()
                finally:
                    del mapped

                os.replace(grid_tmp, grid_path)
                os.replace(index_tmp, index_path)
                shards.append(
                    {
                        "shard_id": shard_id,
                        "grid_path": grid_name,
                        "index_path": index_name,
                        "rows": rows_in_shard,
                        "shape": [rows_in_shard, *spatial_shape],
                        "dtype": dtype.str,
                        "grid_bytes": grid_path.stat().st_size,
                        "grid_sha256": _sha256_path(grid_path),
                        "index_bytes": index_path.stat().st_size,
                        "index_sha256": _sha256_path(index_path),
                    }
                )
        else:
            def prepare_record(record: CacheRecord) -> tuple[CacheRecord, np.ndarray, bytes]:
                _validate_record_metadata(record, whitening_fit=False)
                _validate_grid(record.grid, input_dtype, spatial_shape)
                result_grid = _stream_result_grid(
                    record,
                    stage=stage,
                    dtype=dtype,
                    whitening_transform=frozen_transform,
                )
                line = _canonical_json_bytes(_index_row(record, result_grid)) + b"\n"
                return record, result_grid, line

            def write_shard(
                shard_number: int,
                prepared: Sequence[tuple[CacheRecord, np.ndarray, bytes]],
            ) -> tuple[dict[str, Any], bytes]:
                shard_id = f"shard-{shard_number:06d}"
                grid_name = f"{shard_prefix(shard_number)}.grid.npy"
                index_name = f"{shard_prefix(shard_number)}.index.jsonl"
                grid_path = output / grid_name
                index_path = output / index_name
                grid_tmp = output / f"{grid_name}.tmp"
                index_tmp = output / f"{index_name}.tmp"
                mapped = np.lib.format.open_memmap(
                    grid_tmp,
                    mode="w+",
                    dtype=dtype,
                    shape=(len(prepared), *spatial_shape),
                    fortran_order=False,
                )
                try:
                    for local_index, (_record, result_grid, _line) in enumerate(prepared):
                        mapped[local_index] = result_grid
                    mapped.flush()
                finally:
                    del mapped
                lines = b"".join(item[2] for item in prepared)
                index_tmp.write_bytes(lines)
                os.replace(grid_tmp, grid_path)
                os.replace(index_tmp, index_path)
                return (
                    {
                        "shard_id": shard_id,
                        "grid_path": grid_name,
                        "index_path": index_name,
                        "rows": len(prepared),
                        "shape": [len(prepared), *spatial_shape],
                        "dtype": dtype.str,
                        "grid_bytes": grid_path.stat().st_size,
                        "grid_sha256": _sha256_path(grid_path),
                        "index_bytes": index_path.stat().st_size,
                        "index_sha256": _sha256_path(index_path),
                    },
                    lines,
                )

            def collect(
                futures: Sequence[Future[tuple[dict[str, Any], bytes]]],
            ) -> None:
                for future in futures:
                    descriptor, lines = future.result()
                    shards.append(descriptor)
                    cache_ledger.update(lines)

            pending_writes: list[Future[tuple[dict[str, Any], bytes]]] = []
            next_shard_number = 0
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                while consumed < expected_rows:
                    window_count = min(parallel_window_rows, expected_rows - consumed)
                    records_window: list[CacheRecord] = []
                    for _ in range(window_count):
                        try:
                            record = next(iterator)
                        except StopIteration as exc:
                            raise CacheError("cache input ended before expected_rows") from exc
                        if not isinstance(record, CacheRecord):
                            raise CacheError("cache rows must be CacheRecord instances")
                        if previous_frame_key is not None and record.frame_key <= previous_frame_key:
                            raise CacheError(
                                "streaming cache input must be strictly sorted by frame_key"
                            )
                        previous_frame_key = record.frame_key
                        records_window.append(record)
                    consumed += window_count

                    prepared_window = list(executor.map(prepare_record, records_window))
                    current_writes: list[Future[tuple[dict[str, Any], bytes]]] = []
                    for start in range(0, window_count, shard_rows):
                        shard_prepared = prepared_window[start : start + shard_rows]
                        shard_id = shard_prefix(next_shard_number)
                        paths = (
                            output / f"{shard_id}.grid.npy.tmp",
                            output / f"{shard_id}.index.jsonl.tmp",
                            output / f"{shard_id}.grid.npy",
                            output / f"{shard_id}.index.jsonl",
                        )
                        if any(path.exists() for path in paths):
                            raise CacheError("cache generation output contains a stale shard")
                        generated_paths.extend(paths)
                        current_writes.append(
                            executor.submit(write_shard, next_shard_number, shard_prepared)
                        )
                        next_shard_number += 1

                    # Current GPU decoding/encoding and row hashing occurred
                    # while the previous window's distinct shard files were
                    # being flushed and hashed.  Commit descriptors in strict
                    # canonical order only after the next window is queued.
                    collect(pending_writes)
                    pending_writes = current_writes
                collect(pending_writes)
            if next_shard_number != shard_count:
                raise CacheError("parallel cache shard count mismatch")

        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise CacheError("cache input contains more rows than expected_rows")

        if stage not in {"U", "RAW32"}:
            expected_input_ledger = parent_dict.get("frame_ledger_sha256")
            if input_ledger.hexdigest() != expected_input_ledger:
                raise CacheError("streaming input rows do not match the parent cache ledger")

        manifest: dict[str, Any] = {
            "schema": "J2J_LATENT_CACHE_V1",
            "stage": stage,
            "dtype": dtype.str,
            "rows": expected_rows,
            "shard_rows": shard_rows,
            "shards": shards,
            "parent_manifest_sha256": parent_sha256,
            "frame_ledger_sha256": cache_ledger.hexdigest(),
            "trajectory_alias_ledger_sha256": trajectory_alias_ledger_sha256,
            "identities": configured_identities,
            "run_config": configured_run,
            "production_eligible": production_eligible,
            "training_eligible": stage in {"Z32", "RAW32"},
            "reconstruction_command": configured_command,
        }
        if spatial_shape != (_TOKENS, _FEATURES):
            manifest["spatial_shape"] = list(spatial_shape)
        if stage == "BF16":
            manifest["logical_dtype"] = "bfloat16"
        manifest_tmp = output / "manifest.json.tmp"
        _write_canonical_json(manifest_tmp, manifest)
        os.replace(manifest_tmp, output / "manifest.json")
        return manifest

    except Exception:
        # Keep a failed generation non-visible and make a retry explicit.  We
        # only remove paths this invocation created; no broad cleanup occurs.
        for path in reversed(generated_paths + [output / "manifest.json.tmp"]):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        raise


# Descriptive aliases for callers that prefer the bounded terminology.  They
# intentionally point to the single implementation above.
write_cache_bounded = write_cache_streaming


def _load_manifest(output: Path) -> dict[str, Any]:
    path = output / "manifest.json"
    if not path.is_file():
        raise CacheError("cache manifest is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheError("cache manifest is unreadable") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "J2J_LATENT_CACHE_V1":
        raise CacheError("cache manifest schema mismatch")
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[str, np.dtype[Any]]:
    stage = manifest.get("stage")
    if stage not in _STAGE_DTYPES:
        raise CacheError("cache manifest stage is invalid")
    dtype = _STAGE_DTYPES[stage]
    if manifest.get("dtype") != dtype.str:
        raise CacheError("cache manifest dtype is invalid")
    expected_training = stage in {"Z32", "RAW32"}
    if manifest.get("training_eligible") is not expected_training:
        raise CacheError("cache training eligibility is inconsistent with its stage")
    if not isinstance(manifest.get("production_eligible"), bool):
        raise CacheError("cache production eligibility is invalid")
    if stage == "BF16":
        if manifest.get("logical_dtype") != "bfloat16":
            raise CacheError("BF16 cache is missing its logical dtype")
    elif "logical_dtype" in manifest:
        raise CacheError("only BF16 cache may declare a logical dtype")
    _require_sha256(manifest.get("parent_manifest_sha256"), "parent manifest identity")
    _require_sha256(manifest.get("frame_ledger_sha256"), "frame ledger identity")
    _require_sha256(
        manifest.get("trajectory_alias_ledger_sha256"),
        "trajectory alias ledger identity",
    )
    _validate_run_config(manifest.get("run_config"))
    _validate_reconstruction_command(manifest.get("reconstruction_command"))
    if type(manifest.get("shard_rows")) is not int or manifest["shard_rows"] <= 0:
        raise CacheError("cache shard_rows must be a positive integer")
    _validate_identity_ledger(manifest.get("identities"), stage=stage)
    return stage, dtype


def _validate_store_manifest(
    manifest: Mapping[str, Any],
    *,
    stage: str,
    dtype: np.dtype[Any],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[int, ...]]:
    """Validate the manifest's bounded shard ledger without opening payloads."""

    total_rows = manifest.get("rows")
    if type(total_rows) is not int or total_rows <= 0:
        raise CacheError("cache total row count is invalid")
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise CacheError("cache shard ledger is invalid")

    shards: list[Mapping[str, Any]] = []
    starts: list[int] = []
    offset = 0
    expected_shape_tail = list(_spatial_shape(manifest.get("spatial_shape", (_TOKENS, _FEATURES))))
    for shard_number, raw_shard in enumerate(raw_shards):
        if not isinstance(raw_shard, Mapping):
            raise CacheError("cache shard entry is invalid")
        expected_id = f"shard-{shard_number:06d}"
        if raw_shard.get("shard_id") != expected_id:
            raise CacheError("cache shard ids are not canonical")
        rows = raw_shard.get("rows")
        if type(rows) is not int or rows <= 0:
            raise CacheError("cache shard row count is invalid")
        if raw_shard.get("shape") != [rows, *expected_shape_tail]:
            raise CacheError("cache shard shape ledger is invalid")
        if raw_shard.get("dtype") != dtype.str:
            raise CacheError("cache shard dtype ledger is invalid")
        for name in ("grid_bytes", "index_bytes"):
            value = raw_shard.get(name)
            if type(value) is not int or value <= 0:
                raise CacheError(f"cache shard {name} is invalid")
        for name in ("grid_sha256", "index_sha256"):
            _require_sha256(raw_shard.get(name), f"cache shard {name}")
        for name in ("grid_path", "index_path"):
            value = raw_shard.get(name)
            if not isinstance(value, str) or not value:
                raise CacheError(f"cache shard {name} is invalid")
            path = Path(value)
            if path.is_absolute() or ".." in path.parts:
                raise CacheError("cache shard paths must stay below the output directory")
        starts.append(offset)
        offset += rows
        shards.append(raw_shard)

    if offset != total_rows:
        raise CacheError("cache total row count does not match shard ledger")
    return tuple(shards), tuple(starts)


def open_cache_store(
    output_dir: str | os.PathLike[str],
    *,
    expected_parent_manifest_sha256: str,
    expected_identities: Mapping[str, Any],
    require_stage: str = "Z32",
    require_training_eligible: bool = True,
    require_production_eligible: bool = True,
) -> _CacheStore:
    """Open an authority-checked cache and lazily map requested shards.

    Only the manifest and caller-supplied authority are touched here.  Grid
    and index payloads are validated and memmapped on the first request for a
    shard, then reused by subsequent reads through the same store.
    """

    if not isinstance(require_stage, str) or require_stage not in _STAGE_DTYPES:
        raise CacheError("required cache stage is invalid")
    if not isinstance(require_training_eligible, bool):
        raise CacheError("training eligibility requirement must be boolean")
    if not isinstance(require_production_eligible, bool):
        raise CacheError("production eligibility requirement must be boolean")

    output = Path(output_dir)
    manifest = _load_manifest(output)
    stage, dtype = _validate_manifest(manifest)
    if stage != require_stage:
        raise CacheError("cache stage does not satisfy the requested stage")
    if require_training_eligible and manifest.get("training_eligible") is not True:
        raise CacheError("cache is not training eligible")
    if require_production_eligible and manifest.get("production_eligible") is not True:
        raise CacheError("cache is not production eligible")

    _require_sha256(expected_parent_manifest_sha256, "expected parent identity")
    if manifest["parent_manifest_sha256"] != expected_parent_manifest_sha256:
        raise CacheError("cache parent manifest identity mismatch")
    caller_identities = _validate_identity_ledger(
        expected_identities,
        stage=stage,
    )
    if manifest["identities"] != caller_identities:
        raise CacheError("cache identities do not match the caller authority")

    shards, starts = _validate_store_manifest(
        manifest,
        stage=stage,
        dtype=dtype,
    )
    # Consume every bounded index in canonical shard order before exposing a
    # store.  This preserves the legacy reader's global frame-key and ledger
    # semantics without opening or materializing any grid payload.
    index_rows: list[tuple[dict[str, Any], ...]] = []
    global_previous: bytes | None = None
    frame_ledger = hashlib.sha256()
    for descriptor in shards:
        rows, global_previous = _read_index_metadata(
            output / str(descriptor["index_path"]),
            descriptor,
            previous_frame_key=global_previous,
            ledger=frame_ledger,
        )
        index_rows.append(rows)
    if frame_ledger.hexdigest() != manifest["frame_ledger_sha256"]:
        raise CacheError("cache frame ledger SHA-256 mismatch")
    return _CacheStore(
        output=output,
        manifest=manifest,
        stage=stage,
        dtype=dtype,
        shards=shards,
        starts=starts,
        index_rows=tuple(index_rows),
        opened_shards={},
    )


def iter_cache_rows(
    output_dir: str | os.PathLike[str],
    *,
    expected_parent_manifest_sha256: str | None = None,
    expected_identities: Mapping[str, Any],
) -> Iterator[CacheRecord]:
    """Validate and stream rows from a portable cache generation."""

    output = Path(output_dir)
    manifest = _load_manifest(output)
    stage, dtype = _validate_manifest(manifest)
    caller_identities = _validate_identity_ledger(
        expected_identities,
        stage=stage,
    )
    if manifest["identities"] != caller_identities:
        raise CacheError("cache identities do not match the caller authority")
    parent_sha = manifest["parent_manifest_sha256"]
    if expected_parent_manifest_sha256 is not None:
        _require_sha256(expected_parent_manifest_sha256, "expected parent identity")
        if parent_sha != expected_parent_manifest_sha256:
            raise CacheError("cache parent manifest identity mismatch")

    shards = manifest.get("shards")
    total_rows = manifest.get("rows")
    if not isinstance(shards, list) or not isinstance(total_rows, int) or total_rows <= 0:
        raise CacheError("cache shard ledger is invalid")
    expected_ids = [f"shard-{index:06d}" for index in range(len(shards))]
    if [shard.get("shard_id") if isinstance(shard, Mapping) else None for shard in shards] != expected_ids:
        raise CacheError("cache shard ids are not canonical")

    global_previous: bytes | None = None
    ledger = hashlib.sha256()
    observed_rows = 0
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise CacheError("cache shard entry is invalid")
        rows = shard.get("rows")
        if not isinstance(rows, int) or rows <= 0:
            raise CacheError("cache shard row count is invalid")
        expected_shape = [rows, *_spatial_shape(manifest.get("spatial_shape", (_TOKENS, _FEATURES))) ]
        if shard.get("shape") != expected_shape or shard.get("dtype") != dtype.str:
            raise CacheError("cache shard shape or dtype ledger is invalid")
        grid_path = output / str(shard.get("grid_path"))
        index_path = output / str(shard.get("index_path"))
        if not grid_path.is_file() or not index_path.is_file():
            raise CacheError("cache shard file is missing")
        if grid_path.stat().st_size != shard.get("grid_bytes"):
            raise CacheError("cache grid byte count mismatch")
        if index_path.stat().st_size != shard.get("index_bytes"):
            raise CacheError("cache index byte count mismatch")
        if _sha256_path(grid_path) != shard.get("grid_sha256"):
            raise CacheError("cache grid SHA-256 mismatch")
        if _sha256_path(index_path) != shard.get("index_sha256"):
            raise CacheError("cache index SHA-256 mismatch")
        try:
            mapped = np.load(grid_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise CacheError("cache grid is not a readable NumPy array") from exc
        if not isinstance(mapped, np.memmap):
            raise CacheError("cache reader must use a memory map")
        if list(mapped.shape) != expected_shape or mapped.dtype != dtype:
            raise CacheError("cache grid payload shape or dtype mismatch")
        if dtype.kind == "f" and not np.isfinite(mapped).all():
            raise CacheError("cache grid payload is non-finite")

        with index_path.open("rb") as index_handle:
            for local_index in range(rows):
                line = index_handle.readline()
                if not line or not line.endswith(b"\n"):
                    raise CacheError("cache index is truncated")
                raw = line[:-1]
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise CacheError("cache index row is invalid JSON") from exc
                if not isinstance(row, dict) or raw != _canonical_json_bytes(row):
                    raise CacheError("cache index row is not canonical JSON")
                ledger.update(raw + b"\n")
                try:
                    frame_key = bytes.fromhex(row["frame_key"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise CacheError("cache index frame key is invalid") from exc
                if len(frame_key) != _FRAME_KEY_BYTES:
                    raise CacheError("cache index frame key has the wrong length")
                if global_previous is not None and frame_key <= global_previous:
                    raise CacheError("cache frame keys are duplicated or not strictly sorted")
                global_previous = frame_key
                grid = mapped[local_index]
                if _sha256_bytes(grid.tobytes(order="C")) != row.get("tensor_row_sha256"):
                    raise CacheError("cache tensor row SHA-256 mismatch")
                try:
                    record = CacheRecord(
                        frame_key=frame_key,
                        grid=grid,
                        compressed_jpeg_sha256=row["compressed_jpeg_sha256"],
                        decoded_rgb_sha256=row["decoded_rgb_sha256"],
                        source_dataset=row["source_dataset"],
                        revision=row["revision"],
                        building=row["building"],
                        partition=row["partition"],
                        frame_preimage=row["frame_preimage"],
                        canonical=row["canonical"],
                        trajectory_alias=row["trajectory_alias"],
                    )
                except KeyError as exc:
                    raise CacheError("cache index row is missing provenance") from exc
                _validate_record_metadata(record, whitening_fit=False)
                _validate_grid(record.grid, dtype, tuple(expected_shape[1:]))
                observed_rows += 1
                yield record
            if index_handle.read(1):
                raise CacheError("cache index contains extra rows")

    if observed_rows != total_rows:
        raise CacheError("cache total row count mismatch")
    if ledger.hexdigest() != manifest["frame_ledger_sha256"]:
        raise CacheError("cache frame ledger SHA-256 mismatch")


__all__ = [
    "CacheError",
    "CacheRecord",
    "WhiteningTransform",
    "apply_whitening",
    "fit_whitening",
    "iter_cache_rows",
    "open_cache_store",
    "load_whitening",
    "save_whitening",
    "validate_reencode_pilot",
    "write_cache",
]
