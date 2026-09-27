"""StreamVLN annotation and RGB-frame ABI validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .geometry import ACTION_BOS, ACTION_FWD, ACTION_LEFT, ACTION_RIGHT, ACTION_STOP

_SCENE_RE = re.compile(r"^(.+?)_(?:r2r|rxr)_[^/]+$", re.IGNORECASE)
_FRAME_RE = re.compile(r"^(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
_VALID_ACTIONS = {ACTION_FWD, ACTION_LEFT, ACTION_RIGHT}


@dataclass(frozen=True)
class AnnotationRecord:
    dataset: str
    video: str
    actions: tuple[int, ...]
    row_sha256: str
    scene_token: str
    source_index: int | None = None

    @property
    def basename(self) -> str:
        return Path(self.video).name


def canonical_row_sha256(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scene_token_from_video(video: str) -> str:
    basename = Path(video).name
    match = _SCENE_RE.match(basename)
    if not match:
        raise ValueError(f"cannot extract scene token from video: {video!r}")
    return match.group(1)


def parse_annotation_row(row: dict, dataset: str, source_index: int | None = None) -> AnnotationRecord:
    if not isinstance(row, dict):
        raise ValueError("annotation row must be an object")
    video = row.get("video")
    actions = row.get("actions")
    if not isinstance(video, str) or not video:
        raise ValueError("annotation row requires non-empty video")
    if Path(video).is_absolute() or ".." in Path(video).parts:
        raise ValueError(f"video path must be relative and traversal-free: {video!r}")
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("annotation row requires non-empty actions")
    values = tuple(int(a) for a in actions)
    if values[0] != ACTION_BOS:
        raise ValueError("first action must be BOS (-1)")
    for index, action in enumerate(values[1:], start=1):
        if action == ACTION_STOP:
            raise ValueError(f"STOP has no visual successor (index {index})")
        if action == ACTION_BOS:
            raise ValueError(f"BOS is only valid at index zero (index {index})")
        if action not in _VALID_ACTIONS:
            raise ValueError(f"unknown action code {action} at index {index}")
    return AnnotationRecord(
        dataset=str(dataset),
        video=video,
        actions=values,
        row_sha256=canonical_row_sha256(row),
        scene_token=scene_token_from_video(video),
        source_index=source_index,
    )


def load_annotation_file(path: str | Path, dataset: str) -> list[AnnotationRecord]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"annotation root must be a list: {path}")
    return [parse_annotation_row(row, dataset, i) for i, row in enumerate(rows)]


def numeric_frame_paths(frame_dir: str | Path) -> list[Path]:
    """Return direct child RGB files sorted by numeric frame index."""

    frame_dir = Path(frame_dir)
    if not frame_dir.is_dir():
        raise ValueError(f"frame directory does not exist: {frame_dir}")
    found: list[tuple[int, Path]] = []
    seen: set[int] = set()
    for path in frame_dir.iterdir():
        if not path.is_file():
            continue
        match = _FRAME_RE.match(path.name)
        if not match:
            continue
        index = int(match.group(1))
        if index in seen:
            raise ValueError(f"duplicate numeric frame index {index} in {frame_dir}")
        seen.add(index)
        found.append((index, path))
    if not found:
        raise ValueError(f"no numeric JPG/PNG frames in {frame_dir}")
    return [path for _, path in sorted(found, key=lambda item: item[0])]


def _safe_video_path(source_root: Path, video: str) -> Path:
    relative = Path(video)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"video path must stay below source root: {video!r}")
    candidate = (source_root / relative).resolve()
    root = source_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"video path escapes source root: {video!r}") from exc
    return candidate


def find_frame_dir(source_root: str | Path, video: str) -> Path:
    """Find the published ``rgb`` or legacy ``rgb_images`` directory."""

    root = Path(source_root)
    base = _safe_video_path(root, video)
    candidates = (base / "rgb", base / "rgb_images", base)
    for candidate in candidates:
        if candidate.is_dir():
            try:
                numeric_frame_paths(candidate)
            except ValueError:
                continue
            return candidate
    raise FileNotFoundError(f"no numeric RGB frame directory for {video!r} below {root}")


def validate_alignment(record: AnnotationRecord, frame_paths: Sequence[Path]) -> None:
    if len(record.actions) != len(frame_paths):
        raise ValueError(
            f"len(actions)={len(record.actions)} != len(rgb_frames)={len(frame_paths)} for {record.video}"
        )

