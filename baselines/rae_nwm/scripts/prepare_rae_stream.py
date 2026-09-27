#!/usr/bin/env python3
"""Convert released StreamVLN RGB/action annotations to the RAE directory ABI.

This is deliberately a data-only command.  It never imports torch, Habitat, or
the RAE model.  Frame links are created by one producer in deterministic
annotation order; optional CPU worker settings are recorded for provenance and
can be used by a future indexing front-end without introducing concurrent
writes to a trajectory directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence


def _bootstrap_repo() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _bootstrap_repo()

from rae_stream.annotations import (  # noqa: E402
    AnnotationRecord,
    find_frame_dir,
    load_annotation_file,
    numeric_frame_paths,
)  # noqa: E402
from rae_stream.config_guard import (  # noqa: E402
    STREAMVLN_ANNOTATION_ROW_COUNTS,
    STREAMVLN_ANNOTATION_SHA256,
    STREAMVLN_ARCHIVE_SHA256,
    STREAMVLN_REVISION,
    converter_bundle_sha256,
    converter_closure_digest_map,
)
from rae_stream.manifest import frame_sequence_sha256, write_jsonl  # noqa: E402
from rae_stream.materialize import MaterializationResult, materialize_record  # noqa: E402


CONVERTER_VERSION = "rae_stream_converter_v2"
DEFAULT_MIN_LENGTH = 68  # official context (4) + prediction horizon (64)
DEFAULT_FRAME_FSYNC_INTERVAL = 256


def _make_tree_readonly(root: str | Path) -> None:
    """Remove write bits from a generated tree without following symlinks.

    Formal preparation uses this only for the private tar-selected snapshot,
    generated trajectory tree, and split lists.  It is an accidental-mutation
    guard, not a claim that a privileged operator cannot rewrite the bytes.
    Symlinks are left untouched because chmod-ing their target would violate
    the source ownership boundary.
    """

    root_path = Path(root)
    if not root_path.exists() or root_path.is_symlink():
        raise ValueError(f"readonly tree root must be a regular directory: {root_path}")
    if not root_path.is_dir():
        raise ValueError(f"readonly tree root must be a directory: {root_path}")
    paths = sorted(root_path.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    paths.append(root_path)
    for path in paths:
        try:
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                continue
            os.chmod(path, stat.S_IMODE(info.st_mode) & ~0o222, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"could not make generated tree readonly: {path}") from exc


def _make_file_readonly(path: str | Path) -> None:
    target = Path(path)
    info = target.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"readonly target must be a regular file: {target}")
    os.chmod(target, stat.S_IMODE(info.st_mode) & ~0o222, follow_symlinks=False)


def _make_directory_readonly(path: str | Path) -> None:
    """Freeze one generated container directory without traversing siblings."""

    target = Path(path)
    info = target.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"readonly parent must be a regular directory: {target}")
    os.chmod(target, stat.S_IMODE(info.st_mode) & ~0o222, follow_symlinks=False)


def _freeze_source_parent_dirs(source_roots: Iterable[str | Path]) -> dict[str, bool]:
    """Freeze each private selected-source container and return its receipt map.

    A readonly source tree still has a writable parent by default, which
    permits an accidental rename/unlink-and-replace of the whole snapshot.
    The selected roots are created together under one private container; only
    those immediate containers are frozen here, never a published input tree.
    """

    parents = {Path(root).resolve().parent for root in source_roots}
    if not parents:
        raise ValueError("at least one source root is required")
    result: dict[str, bool] = {}
    for parent in sorted(parents, key=str):
        _make_directory_readonly(parent)
        result[str(parent)] = True
    return result


def _immutable_path_identity(path: str | Path) -> dict[str, int]:
    """Return a compact lstat identity for one generated regular path."""

    target = Path(path).absolute()
    info = target.lstat()
    if stat.S_ISLNK(info.st_mode) or not (
        stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
    ):
        raise ValueError(f"immutable identity target must be a regular path: {target}")
    return {
        "st_dev": int(info.st_dev),
        "st_ino": int(info.st_ino),
        "mode": int(stat.S_IMODE(info.st_mode)),
    }


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    annotation_path: Path
    source_root: Path
    source_revision: str | None = None
    archive_path: Path | None = None
    archive_paths: tuple[Path, ...] | None = None

    def resolved_archive_paths(self) -> tuple[Path, ...]:
        """Return one or more ordered archive parts for a single stream."""

        if self.archive_paths:
            if self.archive_path is not None:
                raise ValueError("provide archive_path or archive_paths, not both")
            paths = tuple(Path(path).resolve() for path in self.archive_paths)
        elif self.archive_path is not None:
            paths = (Path(self.archive_path).resolve(),)
        else:
            paths = ()
        if any(not path.is_file() for path in paths):
            missing = next(path for path in paths if not path.is_file())
            raise FileNotFoundError(missing)
        return paths


_TAR_FRAME_RE = re.compile(r"^(?P<video>.+?)/(?:rgb|rgb_images)/(?P<frame>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _source_metadata(spec: SourceSpec) -> dict[str, Any]:
    annotation = spec.annotation_path.resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    source = spec.source_root.resolve()
    archive_paths = spec.resolved_archive_paths()
    if not source.exists() and not archive_paths:
        raise FileNotFoundError(source)
    if source.exists() and not source.is_dir():
        raise ValueError(f"source root must be a directory: {source}")
    result: dict[str, Any] = {
        "dataset": spec.dataset,
        "annotation_path": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "source_root": str(source),
    }
    if spec.source_revision is not None:
        result["source_revision"] = str(spec.source_revision)
    if archive_paths:
        result["archive_parts"] = [str(path) for path in archive_paths]
        result["archive_bytes"] = sum(path.stat().st_size for path in archive_paths)
        result["archive_sha256"] = [sha256_file(path) for path in archive_paths]
        # Preserve a scalar alias for single-file receipts/readers.
        if len(archive_paths) == 1:
            result["archive_path"] = str(archive_paths[0])
            result["archive_sha256"] = result["archive_sha256"][0]
    return result


def _normalise_member(name: str) -> str:
    """Validate a tar member path before it can reach the temporary root."""

    # Tar names always use POSIX separators, even on a host with another
    # platform.  Reject links and traversal rather than relying on
    # ``TarFile.extract`` (which historically permitted path escapes).
    name = name.replace("\\", "/")
    if not name or name.startswith("/"):
        raise ValueError(f"unsafe tar member path: {name!r}")
    parts = [part for part in name.split("/") if part not in {""}]
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"unsafe tar member path: {name!r}")
    return "/".join(parts)


def _build_video_index(records: Sequence[AnnotationRecord]) -> dict[str, str]:
    """Build exact/suffix lookup keys once (never scan 20k records per tar row)."""

    index: dict[str, str] = {}
    for record in records:
        video = record.video.strip("/")
        keys = {video}
        if "/images/" in video:
            keys.add("images/" + video.split("/images/", 1)[1])
        for key in keys:
            previous = index.get(key)
            if previous is not None and previous != video:
                raise ValueError(f"ambiguous video suffix in annotation set: {key!r}")
            index[key] = video
    return index


def _member_relative_video(member_name: str, video_index: Mapping[str, str]) -> tuple[str, str] | None:
    """Return ``(video, frame_name)`` when a tar member matches a record."""

    match = _TAR_FRAME_RE.match(member_name)
    if match is None:
        return None
    member_video = match.group("video").strip("/")
    frame_name = Path(match.group("frame")).name + Path(member_name).suffix.lower()
    candidates = [member_video]
    if "/images/" in member_video:
        candidates.append("images/" + member_video.split("/images/", 1)[1])
    for candidate in candidates:
        video = video_index.get(candidate)
        if video is not None:
            return video, frame_name
    # Archives may use an arbitrary prefix.  The suffix after the final
    # ``images/`` marker is the only accepted fallback; no broad basename
    # matching is attempted because it could merge two scenes.
    return None


class _ConcatenatedReader:
    """Small forward-only binary reader over ordered archive parts.

    StreamVLN's RxR tarball is distributed as byte-split ``.part`` files.  The
    reader lets ``tarfile`` consume them as one gzip/tar stream without first
    materializing a second 68-GB concatenated archive.
    """

    def __init__(self, paths: Sequence[Path]) -> None:
        if not paths:
            raise ValueError("at least one archive part is required")
        self._paths = tuple(Path(path).resolve() for path in paths)
        self._handles: list[Any] = []
        self._index = 0
        self._closed = False

    def _open_current(self) -> Any | None:
        while self._index < len(self._paths):
            if not self._handles:
                self._handles.append(self._paths[self._index].open("rb"))
            handle = self._handles[-1]
            return handle
        return None

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            return b""
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = int(size)
        while self._index < len(self._paths):
            handle = self._open_current()
            if handle is None:
                break
            chunk = handle.read() if size < 0 else handle.read(remaining)
            if chunk:
                chunks.append(chunk)
                if size >= 0:
                    remaining -= len(chunk)
                    if remaining <= 0:
                        break
            if size >= 0 and remaining <= 0:
                break
            handle.close()
            self._handles.clear()
            self._index += 1
        return b"".join(chunks)

    def readable(self) -> bool:
        return not self._closed

    def seekable(self) -> bool:
        return False

    def close(self) -> None:
        self._closed = True
        for handle in self._handles:
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "_ConcatenatedReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def _extract_selected_tar(
    spec: SourceSpec,
    records: Sequence[AnnotationRecord],
    temporary_parent: Path,
    *,
    min_length: int = DEFAULT_MIN_LENGTH,
    frame_fsync_interval: int = DEFAULT_FRAME_FSYNC_INTERVAL,
) -> tuple[Path, dict[str, int]]:
    """Extract only annotation-matched RGB members in one sequential pass."""

    archive_paths = spec.resolved_archive_paths()
    if not archive_paths:
        raise ValueError("archive path is required for tar extraction")
    if isinstance(frame_fsync_interval, bool) or int(frame_fsync_interval) <= 0:
        raise ValueError("frame_fsync_interval must be a positive integer")
    frame_fsync_interval = int(frame_fsync_interval)
    source_root = Path(tempfile.mkdtemp(prefix=f"rae-stream-{spec.dataset.lower()}-", dir=str(temporary_parent)))
    video_index = _build_video_index(records)
    records_by_video: dict[str, AnnotationRecord] = {}
    for record in records:
        key = record.video.strip("/")
        if key in records_by_video:
            raise ValueError(f"duplicate video in annotation set: {record.video!r}")
        records_by_video[key] = record
    eligible_videos = {
        record.video.strip("/")
        for record in records
        if len(record.actions) >= int(min_length)
    }
    frame_counts: dict[str, int] = {key: 0 for key in records_by_video}
    extracted = 0
    try:
        if len(archive_paths) == 1:
            tar_context = tarfile.open(archive_paths[0], mode="r:*")
            reader_context = None
        else:
            reader_context = _ConcatenatedReader(archive_paths)
            # Stream mode is required: the concatenated parts are not a
            # seekable file and no joined temporary archive is created.
            tar_context = tarfile.open(fileobj=reader_context, mode="r|*")
        with tar_context as tar:
            for member in tar:
                normal = _normalise_member(member.name)
                selected_match = _member_relative_video(normal, video_index)
                if not member.isreg():
                    # A selected symlink/device would make provenance and
                    # safety ambiguous; reject it rather than skipping a
                    # potentially required frame silently.
                    if selected_match is not None:
                        raise ValueError(f"selected tar member is not a regular file: {member.name!r}")
                    continue
                match = selected_match
                if match is None:
                    continue
                video, frame_name = match
                canonical_video = video.strip("/")
                if canonical_video not in frame_counts:
                    raise ValueError(f"tar member matched unknown annotation video: {video!r}")
                frame_counts[canonical_video] += 1
                # Short trajectories are recorded as explicit exclusions and
                # do not need payload extraction.  We still count their tar
                # members so action/frame alignment is checked below.
                if canonical_video not in eligible_videos:
                    continue
                # Keep exactly ``images/<video basename>/rgb/...`` semantics
                # expected by annotations.find_frame_dir.  The video value is
                # already traversal-free from parse_annotation_row.
                relative = Path(video) / ("rgb_images" if "/rgb_images/" in normal else "rgb") / frame_name
                destination = source_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    raise ValueError(f"duplicate selected tar frame: {destination}")
                extracted_file = tar.extractfile(member)
                if extracted_file is None:
                    raise ValueError(f"cannot read selected tar member: {member.name!r}")
                temporary = destination.with_name(f".{destination.name}.tmp")
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(extracted_file, handle, length=1024 * 1024)
                    # A periodic sync marks a selected-frame checkpoint
                    # without issuing one expensive storage barrier per RGB
                    # frame.  It does not synchronize every preceding inode
                    # or claim complete crash consistency; the final
                    # manifest/receipt remain the admission gates.
                    if (extracted + 1) % frame_fsync_interval == 0:
                        handle.flush()
                        os.fsync(handle.fileno())
                os.replace(temporary, destination)
                extracted += 1
        if reader_context is not None:
            reader_context.close()
    except Exception:
        if reader_context is not None:
            reader_context.close()
        shutil.rmtree(source_root, ignore_errors=True)
        raise
    for video, record in records_by_video.items():
        expected = len(record.actions)
        actual = frame_counts.get(video, 0)
        if actual != expected:
            shutil.rmtree(source_root, ignore_errors=True)
            raise ValueError(
                f"len(actions)={expected} != tar RGB members={actual} for {record.video}"
            )
    if extracted == 0 and eligible_videos:
        shutil.rmtree(source_root, ignore_errors=True)
        raise FileNotFoundError(
            f"no annotation-matched RGB members found in archive parts {[str(path) for path in archive_paths]}"
        )
    return source_root, frame_counts


def _write_split_names(split_root: Path, split_to_names: dict[str, list[str]]) -> None:
    for split in ("train", "test"):
        path = split_root / split / "traj_names.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        names = sorted(split_to_names.get(split, []))
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")
        os.replace(temporary, path)


def _freeze_split_lists(split_root: str | Path) -> None:
    """Freeze only traj-name lists; official datasets write index caches nearby."""

    root = Path(split_root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"split root must be a regular directory: {root}")
    for split in ("train", "test"):
        path = root / split / "traj_names.txt"
        _make_file_readonly(path)


def _validate_split_presence(split_to_names: dict[str, list[str]], *, allow_empty_test: bool) -> None:
    if not split_to_names.get("train"):
        raise RuntimeError("building-first split produced no train trajectories")
    if not split_to_names.get("test") and not allow_empty_test:
        raise RuntimeError(
            "building-first split produced no test trajectories; refusing a one-sided formal dataset "
            "(use --allow-empty-test only for non-production mini probes)"
        )


def prepare_dataset(
    specs: Sequence[SourceSpec],
    *,
    data_root: str | Path,
    split_root: str | Path,
    manifest_path: str | Path,
    receipt_path: str | Path,
    min_length: int = DEFAULT_MIN_LENGTH,
    workers: int = 1,
    copy_images: bool = False,
    allow_empty_test: bool = False,
    max_records: int | None = None,
    production_eligible: bool = True,
    frame_fsync_interval: int = DEFAULT_FRAME_FSYNC_INTERVAL,
) -> dict[str, Any]:
    """Prepare one deterministic data set and return its receipt mapping."""

    if not specs:
        raise ValueError("at least one annotation/source specification is required")
    if production_eligible and copy_images:
        # A formal run uses hardlinks (or, only when the filesystem forbids
        # them, source-root-bound symlinks).  Copying the released RGB corpus
        # would create a second tens-of-GB payload and invalidate the resource
        # budget without changing the model or labels.
        raise ValueError("formal RAE-stream preparation forbids copy_images=True")
    if isinstance(min_length, bool) or int(min_length) <= 0:
        raise ValueError("min_length must be a positive integer")
    if isinstance(workers, bool) or int(workers) <= 0:
        raise ValueError("workers must be a positive integer")
    if isinstance(frame_fsync_interval, bool) or int(frame_fsync_interval) <= 0:
        raise ValueError("frame_fsync_interval must be a positive integer")
    frame_fsync_interval = int(frame_fsync_interval)
    if max_records is not None and (isinstance(max_records, bool) or int(max_records) <= 0):
        raise ValueError("max_records must be positive when supplied")
    if production_eligible and int(min_length) < DEFAULT_MIN_LENGTH:
        raise ValueError(
            f"formal RAE-stream data requires min_length>={DEFAULT_MIN_LENGTH}; "
            "use --non-production for a tiny probe"
        )
    if production_eligible and int(min_length) != DEFAULT_MIN_LENGTH:
        raise ValueError(
            f"formal RAE-stream data fixes min_length={DEFAULT_MIN_LENGTH}; "
            "a different filter is a separately reviewed data identity"
        )
    if production_eligible and allow_empty_test:
        raise ValueError("formal RAE-stream data requires a non-empty test split")
    if production_eligible:
        datasets = {str(spec.dataset) for spec in specs}
        if datasets != {"R2R", "RxR"} or len(specs) != 2:
            raise ValueError("formal RAE-stream data requires exactly one R2R and one RxR source")
        if max_records is not None:
            raise ValueError("formal RAE-stream data cannot use --max-records")
        for spec in specs:
            if str(spec.source_revision) != STREAMVLN_REVISION:
                raise ValueError("formal RAE-stream data requires the fixed StreamVLN revision")
            archive_hashes = spec.resolved_archive_paths()
            if not archive_hashes:
                raise ValueError("formal RAE-stream data requires the released parent archives")
    names = [str(spec.dataset) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate dataset names are not allowed")

    data_root = Path(data_root).resolve()
    split_root = Path(split_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    if production_eligible:
        # The generated data container is frozen before the receipt is
        # published.  Keep the receipt in the dedicated receipts/ area (or
        # another sibling) so publication never depends on reopening a frozen
        # directory and cannot be mistaken for a mutable training input.
        try:
            receipt_path.relative_to(data_root.parent)
        except ValueError:
            pass
        else:
            raise ValueError(
                "formal prepare receipt must live outside the frozen data container"
            )
    converter_sha256 = sha256_file(Path(__file__).resolve())
    converter_source_sha256 = converter_closure_digest_map(REPO_ROOT)
    converter_bundle_digest = converter_bundle_sha256(converter_source_sha256)
    if production_eligible:
        # Fail before touching the large data tree if the checked-in converter
        # no longer matches the independently reviewed byte identity.
        from rae_stream.config_guard import verify_converter_source

        verify_converter_source(REPO_ROOT, expected_sha256=converter_sha256)
    data_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    metadata = [_source_metadata(spec) for spec in specs]
    initial_input_metadata = [dict(item) for item in metadata]
    if production_eligible:
        for item in metadata:
            dataset = str(item["dataset"])
            expected_annotation = STREAMVLN_ANNOTATION_SHA256[dataset]
            if item.get("annotation_sha256") != expected_annotation:
                raise ValueError(
                    f"{dataset} annotation SHA does not match fixed StreamVLN revision"
                )
            archive_value = item.get("archive_sha256")
            actual_archives = (archive_value,) if isinstance(archive_value, str) else tuple(archive_value or ())
            if actual_archives != STREAMVLN_ARCHIVE_SHA256[dataset]:
                raise ValueError(
                    f"{dataset} parent archive SHA does not match fixed StreamVLN release"
                )
    all_rows: list[dict[str, Any]] = []
    split_to_names: dict[str, list[str]] = {"train": [], "test": []}
    seen_names: set[str] = set()
    counts: dict[str, int] = {"records_seen": 0, "materialized": 0, "excluded_short": 0}
    frame_count = 0
    frame_bytes = 0
    link_count = 0
    copy_count = 0
    archive_frames_counted = 0
    archive_frames_extracted = 0
    archive_short_frames_skipped = 0
    effective_source_roots: dict[str, Path] = {}
    effective_source_modes: dict[str, str] = {}
    started = time.monotonic()

    # One producer intentionally performs materialization in canonical order.
    # Annotation parsing and JPEG stat calls are CPU work, but parallel writers
    # would make inode/provenance accounting and failure recovery ambiguous.
    tar_parent = data_root.parent / "tar_selected_sources"
    for spec in specs:
        records = load_annotation_file(spec.annotation_path, spec.dataset)
        if production_eligible and len(records) != STREAMVLN_ANNOTATION_ROW_COUNTS[str(spec.dataset)]:
            raise ValueError(
                f"{spec.dataset} annotation row count {len(records)} does not match fixed release "
                f"({STREAMVLN_ANNOTATION_ROW_COUNTS[str(spec.dataset)]})"
            )
        if max_records is not None:
            records = records[: int(max_records)]
        source_meta = next(item for item in metadata if item["dataset"] == spec.dataset)
        effective_source = spec.source_root.resolve()
        extracted_from_tar = False
        archive_frame_counts: dict[str, int] = {}
        # An explicitly supplied archive always gets a fresh selected staging
        # tree.  Reusing a stale placeholder would silently mix frames from a
        # previous archive identity.
        archive_paths = spec.resolved_archive_paths()
        if archive_paths:
            tar_parent.mkdir(parents=True, exist_ok=True)
            effective_source, archive_frame_counts = _extract_selected_tar(
                spec,
                records,
                tar_parent,
                min_length=int(min_length),
                frame_fsync_interval=frame_fsync_interval,
            )
            extracted_from_tar = True
            counted = sum(archive_frame_counts.values())
            extracted = sum(len(record.actions) for record in records if len(record.actions) >= int(min_length))
            archive_frames_counted += counted
            archive_frames_extracted += extracted
            archive_short_frames_skipped += counted - extracted
            source_meta["effective_source_root"] = str(effective_source)
            source_meta["source_mode"] = "tar-selected"
        elif not effective_source.exists():
            if not archive_paths:
                raise FileNotFoundError(effective_source)
            tar_parent.mkdir(parents=True, exist_ok=True)
            # Keep the selected-frame staging tree as part of the output
            # identity.  This is small (only annotation-matched RGB), avoids
            # broken cross-filesystem symlinks after the process exits, and
            # still performs exactly one sequential tar scan per archive.
            effective_source, archive_frame_counts = _extract_selected_tar(
                spec,
                records,
                tar_parent,
                min_length=int(min_length),
                frame_fsync_interval=frame_fsync_interval,
            )
            extracted_from_tar = True
            counted = sum(archive_frame_counts.values())
            extracted = sum(len(record.actions) for record in records if len(record.actions) >= int(min_length))
            archive_frames_counted += counted
            archive_frames_extracted += extracted
            archive_short_frames_skipped += counted - extracted
            source_meta["effective_source_root"] = str(effective_source)
            source_meta["source_mode"] = "tar-selected"
        else:
            source_meta["effective_source_root"] = str(effective_source)
            source_meta["source_mode"] = "published-source"
        effective_source_roots[str(spec.dataset)] = effective_source.resolve()
        effective_source_modes[str(spec.dataset)] = str(source_meta["source_mode"])
        for record in records:
            counts["records_seen"] += 1
            if extracted_from_tar and len(record.actions) < int(min_length):
                # The archive pass already counted every selected member and
                # checked exact action/frame alignment.  Avoid asking the
                # materializer to rediscover a deliberately unextracted short
                # trajectory.
                result = MaterializationResult(
                    status="excluded_short",
                    trajectory_name=f"{record.dataset.lower()}__{record.basename}",
                    dataset=record.dataset,
                    video=record.video,
                    scene_token=record.scene_token,
                    frame_count=int(archive_frame_counts[record.video.strip("/")]),
                    measured_pose=False,
                    source_mode="tar-selected",
                    link_count=0,
                    copy_count=0,
                    elapsed_seconds=0.0,
                )
            else:
                result = materialize_record(
                    record,
                    effective_source,
                    data_root,
                    min_length=int(min_length),
                    copy_images=bool(copy_images),
                )
            row = result.as_manifest_row()
            row.update(
                {
                    "converter_version": CONVERTER_VERSION,
                    "converter_sha256": converter_sha256,
                    "converter_source_sha256": dict(converter_source_sha256),
                    "converter_bundle_sha256": converter_bundle_digest,
                    "source_index": record.source_index,
                    "source_id": f"{record.dataset}:{record.source_index}",
                    "annotation_row_sha256": record.row_sha256,
                    "source_revision": spec.source_revision,
                    "source_root": str(effective_source),
                    "source_mode": "tar-selected" if extracted_from_tar else result.source_mode,
                    "archive_path": (
                        str(archive_paths[0]) if len(archive_paths) == 1 else None
                    ),
                    "archive_parts": [str(path) for path in archive_paths] if archive_paths else None,
                    "measured_pose": False,
                    "pose_source": "command-derived-se2",
                    "production_eligible": bool(production_eligible),
                    # Keep the source command sequence in the manifest so a
                    # later validator can independently reconstruct the
                    # command-derived SE(2) pose and verify frame/action
                    # alignment without trusting arbitrary finite arrays.
                    "command_actions": [int(action) for action in record.actions],
                    "command_actions_sha256": hashlib.sha256(
                        json.dumps(
                            [int(action) for action in record.actions],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    # The validator reconstructs this relative directory
                    # below the receipt-bound effective source root and
                    # checks every output frame's inode/target against it.
                    "source_frame_subdir": (
                        str(find_frame_dir(effective_source, record.video).relative_to(effective_source))
                        if result.status == "materialized"
                        else None
                    ),
                }
            )
            if result.status == "materialized":
                counts["materialized"] += 1
                split = str(row["split"])
                name = str(row["trajectory_name"])
                if name in seen_names:
                    raise RuntimeError(f"duplicate trajectory name across sources: {name}")
                seen_names.add(name)
                split_to_names.setdefault(split, []).append(name)
                frame_count += int(result.frame_count)
                link_count += int(result.link_count)
                copy_count += int(result.copy_count)
                output_dir = Path(str(result.output_dir))
                row["traj_data_sha256"] = sha256_file(output_dir / "traj_data.pkl")
                output_frames = sorted(output_dir.glob("*.jpg"), key=lambda path: int(path.stem))
                if production_eligible:
                    # Compute the selected-source digest from the private
                    # tar snapshot itself.  Hardlinks/symlinks make it equal
                    # to the output digest, while the validator independently
                    # recomputes and binds both paths.
                    source_dir = find_frame_dir(effective_source, record.video)
                    source_frames = numeric_frame_paths(source_dir)
                    if len(source_frames) != len(output_frames):
                        raise RuntimeError("source/output frame count changed during digesting")
                    try:
                        same_inode = all(
                            os.path.samefile(output, source)
                            for output, source in zip(output_frames, source_frames)
                        )
                    except OSError:
                        same_inode = False
                    frame_digest = (
                        frame_sequence_sha256(source_frames)
                        if same_inode
                        else frame_sequence_sha256(output_frames)
                    )
                else:
                    frame_digest = frame_sequence_sha256(output_frames)
                row["frame_sequence_sha256"] = frame_digest
                row["source_frame_sequence_sha256"] = frame_digest
                for frame in output_dir.glob("*.jpg"):
                    try:
                        frame_bytes += int(frame.stat().st_size)
                    except OSError:
                        pass
            elif result.status == "excluded_short":
                counts["excluded_short"] += 1
            else:
                raise RuntimeError(f"unknown materialization status: {result.status!r}")
            row["source_metadata"] = source_meta
            all_rows.append(row)

    # Re-read every released input identity after conversion.  Annotation or
    # archive replacement during a long tar scan must fail before publication;
    # this is a sequential hash pass over the already-required immutable
    # inputs, not a second extraction or GPU computation.
    final_input_metadata = [_source_metadata(spec) for spec in specs]
    if final_input_metadata != initial_input_metadata:
        raise RuntimeError("released annotation/archive bytes changed during preparation")
    if production_eligible:
        for item in final_input_metadata:
            dataset = str(item["dataset"])
            if item.get("annotation_sha256") != STREAMVLN_ANNOTATION_SHA256[dataset]:
                raise RuntimeError(f"{dataset} annotation changed during preparation")
            archive_value = item.get("archive_sha256")
            actual_archives = (archive_value,) if isinstance(archive_value, str) else tuple(archive_value or ())
            if actual_archives != STREAMVLN_ARCHIVE_SHA256[dataset]:
                raise RuntimeError(f"{dataset} parent archive changed during preparation")
    # If a dependency changed while a long conversion was running, do not
    # publish a manifest whose rows mix two converter closures.
    final_converter_source_sha256 = converter_closure_digest_map(REPO_ROOT)
    if final_converter_source_sha256 != converter_source_sha256:
        raise RuntimeError("converter source closure changed during preparation; refusing publication")
    _validate_split_presence(split_to_names, allow_empty_test=allow_empty_test)
    all_rows.sort(key=lambda row: (str(row.get("dataset", "")), int(row.get("source_index", -1))))
    write_jsonl(manifest_path, all_rows)
    _write_split_names(split_root, split_to_names)

    immutable_data_tree = False
    immutable_split_tree = False
    immutable_split_lists = False
    immutable_manifest = False
    immutable_parent_dirs: dict[str, bool] = {}
    immutable_source_parent_dirs: dict[str, bool] = {}
    immutable_path_identities: dict[str, dict[str, int]] = {}
    split_list_sha256: dict[str, str] = {}
    immutable_source_trees: dict[str, bool] = {
        dataset: False for dataset in effective_source_roots
    }
    if production_eligible:
        # Freeze only private generated trees.  A published-source input may
        # belong to another experiment/user and must never be chmod-ed by the
        # converter; formal runs use tar-selected staging and therefore get a
        # complete immutable source snapshot here.
        _make_tree_readonly(data_root)
        _freeze_split_lists(split_root)
        _make_file_readonly(manifest_path)
        # Freeze the immediate containers as well, so an accidental rename or
        # unlink cannot replace an otherwise readonly generated child tree.
        # These parents are created inside the dedicated RAE-stream output
        # root; no published/shared source parent is modified.
        for parent in {data_root.parent}:
            _make_directory_readonly(parent)
            immutable_parent_dirs[str(parent)] = True
        immutable_data_tree = True
        # The official BaseDataset writes its deterministic index cache into
        # each split directory at train startup.  Keep those directories
        # writable while freezing the source-of-truth traj_names.txt files.
        immutable_split_lists = True
        split_list_sha256 = {
            split: sha256_file(split_root / split / "traj_names.txt")
            for split in ("train", "test")
        }
        immutable_manifest = True
        for dataset, source_root in effective_source_roots.items():
            if effective_source_modes.get(dataset) != "tar-selected":
                raise RuntimeError(
                    "formal preparation requires a private tar-selected source snapshot"
                )
            _make_tree_readonly(source_root)
            immutable_source_trees[dataset] = True
        immutable_source_parent_dirs = _freeze_source_parent_dirs(effective_source_roots.values())
        immutable_paths = {
            data_root,
            data_root.parent,
            data_root.parent.parent,
            split_root,
            split_root.parent,
            manifest_path,
            *(split_root / split for split in ("train", "test")),
            *(split_root / split / "traj_names.txt" for split in ("train", "test")),
            *effective_source_roots.values(),
            *(root.parent for root in effective_source_roots.values()),
        }
        immutable_path_identities = {
            str(path.resolve()): _immutable_path_identity(path)
            for path in sorted(immutable_paths, key=lambda value: str(value.resolve()))
        }

    try:
        stat = os.statvfs(data_root)
        inode_stats = {
            "free_inodes": int(stat.f_favail),
            "total_inodes": int(stat.f_files),
            "free_bytes": int(stat.f_bavail * stat.f_frsize),
        }
    except OSError as exc:
        inode_stats = {"error": str(exc)}

    receipt: dict[str, Any] = {
        "status": "PASS",
        "converter_version": CONVERTER_VERSION,
        "converter_sha256": converter_sha256,
        "converter_source_sha256": dict(converter_source_sha256),
        "converter_bundle_sha256": converter_bundle_digest,
        "dataset_identity": "RAE-NWM-StreamVLN-command-derived",
        "measured_pose": False,
        "pose_source": "command-derived-se2",
        "production_eligible": bool(production_eligible),
        "data_root": str(data_root),
        "split_root": str(split_root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "min_length": int(min_length),
        "formal_min_length": DEFAULT_MIN_LENGTH,
        "context_size": 4,
        "len_traj_pred": 64,
        "frame_fsync_interval": frame_fsync_interval,
        "workers_requested": int(workers),
        "single_writer": True,
        "copy_images": bool(copy_images),
        "counts": counts,
        "split_counts": {key: len(value) for key, value in split_to_names.items()},
        "frame_count_output": frame_count,
        "frame_bytes_stat_sum": frame_bytes,
        "link_count": link_count,
        "copy_count": copy_count,
        "immutable_data_tree": immutable_data_tree,
        "immutable_split_tree": immutable_split_tree,
        "immutable_split_lists": immutable_split_lists,
        "split_list_sha256": split_list_sha256,
        "immutable_manifest": immutable_manifest,
        "immutable_parent_dirs": immutable_parent_dirs,
        "immutable_source_parent_dirs": immutable_source_parent_dirs,
        "immutable_path_identities": immutable_path_identities,
        "immutable_source_trees": immutable_source_trees,
        "archive_frames_counted": archive_frames_counted,
        "archive_frames_extracted": archive_frames_extracted,
        "archive_short_frames_skipped": archive_short_frames_skipped,
        "inode_stats_after": inode_stats,
        "sources": metadata,
        "source_metadata_initial": initial_input_metadata,
        "source_metadata_final": final_input_metadata,
        "annotation_row_counts": {
            str(spec.dataset): sum(1 for row in all_rows if row.get("dataset") == str(spec.dataset))
            for spec in specs
        },
        "host": platform.node(),
        "python": platform.python_version(),
        "elapsed_seconds": time.monotonic() - started,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _atomic_json(receipt_path, receipt)
    return receipt


def _parse_dataset_arg(value: str) -> SourceSpec:
    # Accept ``NAME=ANNOTATIONS,SOURCE_ROOT`` and a colon-separated variant
    # for shell convenience.  Commas are preferred because paths may contain
    # colons on Windows-like mounts (the server is Linux, but parsing is cheap).
    text = str(value)
    if "=" not in text:
        raise argparse.ArgumentTypeError("--dataset must be NAME=ANNOTATIONS,SOURCE_ROOT")
    dataset, rest = text.split("=", 1)
    parts = rest.split(",", 1)
    if len(parts) != 2:
        parts = rest.split(":", 1)
    if len(parts) != 2 or not dataset.strip():
        raise argparse.ArgumentTypeError("--dataset must be NAME=ANNOTATIONS,SOURCE_ROOT")
    source = Path(parts[1])
    archive = source if source.is_file() or source.suffix.lower() in {".tar", ".tgz", ".gz"} else None
    # For archive-only specs the source root is a deterministic placeholder;
    # prepare_dataset replaces it with the selected staging root after the
    # single scan.  The placeholder itself is never used for extraction.
    return SourceSpec(
        dataset.strip(),
        Path(parts[0]),
        source if archive is None else source.with_name(f".{source.stem}.extracted"),
        source_revision=STREAMVLN_REVISION,
        archive_path=archive,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, help="parent containing data/rae_stream and data_splits/rae_stream")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--dataset", action="append", type=_parse_dataset_arg, help="NAME=ANNOTATIONS,SOURCE_ROOT (repeatable)")
    parser.add_argument("--r2r-annotations", type=Path)
    parser.add_argument("--rxr-annotations", type=Path)
    parser.add_argument("--r2r-source-root", type=Path)
    parser.add_argument("--rxr-source-root", type=Path)
    parser.add_argument("--r2r-archive", type=Path)
    parser.add_argument("--rxr-archive", type=Path)
    parser.add_argument(
        "--rxr-archive-part",
        action="append",
        type=Path,
        default=[],
        help="ordered byte-split RxR archive part (repeat for .part0, .part1, ...)",
    )
    parser.add_argument("--source-revision", default=STREAMVLN_REVISION)
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 16)))
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--allow-empty-test", action="store_true")
    parser.add_argument("--non-production", action="store_true", help="mark output ineligible for formal training")
    parser.add_argument(
        "--frame-fsync-interval",
        type=int,
        default=DEFAULT_FRAME_FSYNC_INTERVAL,
        help="durability barrier interval for selected tar frames (default: 256; not a model setting)",
    )
    return parser


def _resolve_specs(args: argparse.Namespace) -> list[SourceSpec]:
    specs = list(args.dataset or [])
    for dataset, annotation, source, archive in (
        ("R2R", args.r2r_annotations, args.r2r_source_root, args.r2r_archive),
        ("RxR", args.rxr_annotations, args.rxr_source_root, args.rxr_archive),
    ):
        if annotation is None and source is None and archive is None:
            continue
        if annotation is None:
            raise ValueError(f"{dataset} requires annotation")
        if dataset == "RxR" and args.rxr_archive_part and archive is not None:
            raise ValueError("RxR cannot provide --rxr-archive and --rxr-archive-part together")
        if dataset == "RxR" and args.rxr_archive_part:
            if source is not None:
                raise ValueError("RxR archive parts cannot be combined with --rxr-source-root")
            if annotation is None:
                raise ValueError("RxR archive parts require annotation")
            specs.append(
                SourceSpec(
                    dataset,
                    annotation,
                    Path(args.rxr_archive_part[0]).with_name(".rxr_parts.extracted"),
                    args.source_revision,
                    archive_paths=tuple(args.rxr_archive_part),
                )
            )
            continue
        if archive is not None:
            if source is not None:
                raise ValueError(f"{dataset} cannot provide both source-root and archive")
            source = archive.with_name(f".{archive.stem}.extracted")
        if source is None:
            raise ValueError(f"{dataset} requires source-root or archive")
        specs.append(SourceSpec(dataset, annotation, source, args.source_revision, archive))
    if not specs:
        raise ValueError("supply --dataset or at least one --*-annotations/--*-source-root pair")
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        specs = _resolve_specs(args)
        if args.output_root is not None:
            output = args.output_root
            data_root = args.data_root or output / "data" / "rae_stream"
            split_root = args.split_root or output / "data_splits" / "rae_stream"
            manifest = args.manifest or output / "rae_stream_manifest.jsonl"
            receipt = args.receipt or output / "rae_stream_prepare_receipt.json"
        else:
            if args.data_root is None or args.split_root is None:
                raise ValueError("--output-root or all of --data-root/--split-root are required")
            data_root = args.data_root
            split_root = args.split_root
            manifest = args.manifest or Path(data_root).parent / "rae_stream_manifest.jsonl"
            receipt = args.receipt or Path(data_root).parent / "rae_stream_prepare_receipt.json"
        prepare_dataset(
            specs,
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            receipt_path=receipt,
            min_length=args.min_length,
            workers=args.workers,
            copy_images=args.copy_images,
            allow_empty_test=args.allow_empty_test,
            max_records=args.max_records,
            production_eligible=not args.non_production,
            frame_fsync_interval=args.frame_fsync_interval,
        )
    except Exception as exc:
        print(f"prepare_rae_stream: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"prepare_rae_stream: PASS receipt={Path(receipt).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
