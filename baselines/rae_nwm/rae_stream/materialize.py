"""Materialize StreamVLN trajectories into the official RAE directory ABI."""

from __future__ import annotations

import errno
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .annotations import AnnotationRecord, find_frame_dir, numeric_frame_paths, validate_alignment
from .geometry import integrate_actions


@dataclass(frozen=True)
class MaterializationResult:
    status: str
    trajectory_name: str
    dataset: str
    video: str
    scene_token: str
    frame_count: int
    measured_pose: bool
    source_mode: str
    link_count: int
    copy_count: int
    elapsed_seconds: float
    output_dir: str | None = None

    def as_manifest_row(self) -> dict:
        return {
            "status": self.status,
            "trajectory_name": self.trajectory_name,
            "dataset": self.dataset,
            "video": self.video,
            "scene_token": self.scene_token,
            "split": split_for_scene(self.scene_token),
            "frame_count": self.frame_count,
            "measured_pose": self.measured_pose,
            "pose_source": "command-derived-se2",
            "source_mode": self.source_mode,
            "link_count": self.link_count,
            "copy_count": self.copy_count,
            "elapsed_seconds": self.elapsed_seconds,
            "output_dir": self.output_dir,
        }


def split_for_scene(scene_token: str) -> str:
    import hashlib

    bucket = hashlib.sha256(str(scene_token).encode("utf-8")).digest()[0] % 10
    return "test" if bucket == 0 else "train"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _link_frame(source: Path, destination: Path, source_root: Path, copy_images: bool) -> str:
    resolved_source = source.resolve()
    if not _inside(resolved_source, source_root):
        raise ValueError(f"source frame escapes declared source root: {source}")
    if copy_images:
        shutil.copy2(resolved_source, destination)
        return "copy"
    try:
        os.link(resolved_source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        # Relative symlinks keep the output relocatable and do not duplicate RGB bytes.
        destination.symlink_to(os.path.relpath(resolved_source, destination.parent))
        return "symlink"


def materialize_record(
    record: AnnotationRecord,
    source_root: str | Path,
    data_root: str | Path,
    *,
    min_length: int = 68,
    copy_images: bool = False,
) -> MaterializationResult:
    """Create one RAE trajectory, or return an explicit short-trajectory exclusion."""

    started = time.monotonic()
    source_root = Path(source_root).resolve()
    data_root = Path(data_root)
    frame_dir = find_frame_dir(source_root, record.video)
    frame_paths = numeric_frame_paths(frame_dir)
    validate_alignment(record, frame_paths)
    trajectory_name = f"{record.dataset.lower()}__{record.basename}"
    if len(frame_paths) < int(min_length):
        return MaterializationResult(
            status="excluded_short",
            trajectory_name=trajectory_name,
            dataset=record.dataset,
            video=record.video,
            scene_token=record.scene_token,
            frame_count=len(frame_paths),
            measured_pose=False,
            source_mode="none",
            link_count=0,
            copy_count=0,
            elapsed_seconds=time.monotonic() - started,
        )

    data_root.mkdir(parents=True, exist_ok=True)
    output = data_root / trajectory_name
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing trajectory: {output}")
    temp = Path(tempfile.mkdtemp(prefix=f".{trajectory_name}.", dir=str(data_root)))
    modes: list[str] = []
    try:
        for index, source in enumerate(frame_paths):
            modes.append(_link_frame(source, temp / f"{index}.jpg", source_root, copy_images))
        poses = integrate_actions(record.actions)
        payload = {"position": poses[:, :2].astype(np.float32), "yaw": poses[:, 2].astype(np.float32)}
        pickle_path = temp / "traj_data.pkl"
        with pickle_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, output)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise

    unique_modes = sorted(set(modes))
    source_mode = unique_modes[0] if len(unique_modes) == 1 else "mixed"
    return MaterializationResult(
        status="materialized",
        trajectory_name=trajectory_name,
        dataset=record.dataset,
        video=record.video,
        scene_token=record.scene_token,
        frame_count=len(frame_paths),
        measured_pose=False,
        source_mode=source_mode,
        link_count=sum(mode in {"hardlink", "symlink"} for mode in modes),
        copy_count=modes.count("copy"),
        elapsed_seconds=time.monotonic() - started,
        output_dir=str(output),
    )

