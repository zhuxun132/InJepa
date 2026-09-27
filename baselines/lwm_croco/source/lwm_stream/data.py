"""StreamVLN replay metadata and local planar geometry."""

from collections.abc import Mapping
from pathlib import PurePosixPath
import numbers
import re

import numpy as np


def _integer_id(value):
    if isinstance(value, bool):
        raise ValueError("boolean is not an episode id")
    if isinstance(value, numbers.Integral) and value >= 0:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    raise ValueError("episode id must be a nonnegative integer")


def _identity(record):
    if not isinstance(record, Mapping):
        raise ValueError("annotation must be a mapping")
    video = record.get("video")
    if not isinstance(video, str):
        raise ValueError("annotation video is required")
    match = re.fullmatch(r"(.+)_(r2r|rxr)_([0-9]+)", PurePosixPath(video).name)
    if match is None:
        raise ValueError("video must identify scene, source and episode")
    scene, source, eid = match.groups()
    if _integer_id(record.get("id")) != int(eid):
        raise ValueError("video and annotation episode ids differ")
    return scene, source, int(eid)


def _array(value, width, name):
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if arr.ndim != 2 or arr.shape[1] != width or not np.isfinite(arr).all():
        raise ValueError(f"{name} must have finite shape (N,{width})")
    return arr


def _quaternions(value):
    q = _array(value, 4, "xyzw quaternions")
    # Scaling before normalization avoids overflow with finite input components.
    scale = np.max(np.abs(q), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero quaternion does not define a rotation")
    q = q / scale
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def validate_annotation(record, frame_names):
    """Check incoming-action alignment and preserve actual 0/1-based filenames."""
    _identity(record)
    actions = record.get("actions")
    if not isinstance(actions, (list, tuple)) or not actions:
        raise ValueError("actions must contain the BOS reset observation")
    if any(isinstance(a, bool) or not isinstance(a, numbers.Integral) for a in actions):
        raise ValueError("actions must be integer codes")
    if actions[0] != -1 or any(a not in (1, 2, 3) for a in actions[1:]):
        raise ValueError("expected BOS then motion codes; STOP has no successor frame")
    names = list(frame_names)
    if len(names) != len(actions):
        raise ValueError("frame/action counts differ")
    numbered = []
    for name in names:
        match = re.fullmatch(r"([0-9]+)\.jpg", PurePosixPath(str(name)).name)
        if match is None:
            raise ValueError("frame must have a numeric JPEG name")
        numbered.append((int(match.group(1)), name))
    numbered.sort(key=lambda pair: pair[0])
    ids = [pair[0] for pair in numbered]
    if ids[0] not in (0, 1) or ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError("frames must be contiguous, unique and start at zero or one")
    return [pair[1] for pair in numbered]


def join_episode(record, episodes, dataset_source):
    """Find one same-source/scene/id episode; never guess a reset pose."""
    scene, source, eid = _identity(record)
    if dataset_source != source:
        raise ValueError("dataset source differs from annotation")
    matches = []
    for row in episodes:
        if not isinstance(row, Mapping):
            raise ValueError("episode must be a mapping")
        if _integer_id(row.get("episode_id")) != eid:
            continue
        path = row.get("scene_id")
        if not isinstance(path, str):
            raise ValueError("episode scene_id is required")
        if PurePosixPath(path).stem != scene:
            continue
        if row.get("dataset_source", source) != source:
            raise ValueError("episode source differs from annotation")
        matches.append(row)
    if len(matches) != 1:
        raise ValueError("expected one exact episode match")
    result = matches[0]
    _array([result.get("start_position")], 3, "start position")
    _quaternions([result.get("start_rotation")])
    return result


def local_trajectory(positions, quaternions_xyzw, origin_index=0):
    """Cumulative local (forward, left, yaw), in metres/radians, without scaling.

    Habitat is Y-up with local -Z forward; positive Y rotation is a left turn.
    All original rows are returned, including rows preceding a nonzero origin.
    """
    p = _array(positions, 3, "positions")
    q = _quaternions(quaternions_xyzw)
    if len(p) != len(q) or not len(p):
        raise ValueError("positions and rotations must have the same nonzero length")
    if (isinstance(origin_index, bool) or not isinstance(origin_index, numbers.Integral)
            or not 0 <= origin_index < len(p)):
        raise ValueError("origin index is out of range")
    x, y, z, w = q.T
    forward_x = -2 * (x * z + w * y)
    forward_z = -(1 - 2 * (x * x + y * y))
    if np.any(np.hypot(forward_x, forward_z) < np.finfo(np.float64).eps):
        raise ValueError("vertical viewing direction has no planar heading")
    yaw = np.arctan2(-forward_x, -forward_z)
    theta = yaw[origin_index]
    delta = p - p[origin_index]
    forward = -np.sin(theta) * delta[:, 0] - np.cos(theta) * delta[:, 2]
    left = -np.cos(theta) * delta[:, 0] + np.sin(theta) * delta[:, 2]
    relative_yaw = (yaw - theta + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack((forward, left, relative_yaw))


def select_keyframes(positions, min_distance=0.2):
    """Select original indices by planar distance from the last selected pose."""
    if (isinstance(min_distance, bool) or not isinstance(min_distance, numbers.Real)
            or not np.isfinite(min_distance) or min_distance <= 0):
        raise ValueError("keyframe distance must be finite and positive")
    p = _array(positions, 3, "positions")
    if not len(p):
        return []
    selected = [0]
    for idx in range(1, len(p)):
        displacement = p[idx, (0, 2)] - p[selected[-1], (0, 2)]
        if np.linalg.norm(displacement) >= min_distance:
            selected.append(idx)
    return selected
