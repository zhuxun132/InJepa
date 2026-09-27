"""Dependency-light command-derived SE(2) geometry.

The convention intentionally matches the official RAE ``to_local_coords``
row-vector convention: heading zero points along world +x and positive yaw is
counter-clockwise.  This module never reads simulator pose.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

ACTION_STOP = 0
ACTION_BOS = -1
ACTION_FWD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
TURN_RADIANS = math.pi / 12.0
FORWARD_METERS = 0.25


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to ``[-pi, pi)`` while preserving scalar/array shape."""

    wrapped = (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi
    if np.ndim(angle) == 0:
        return float(wrapped)
    return wrapped


def action_to_twist(action: int) -> tuple[float, float, float]:
    """Return the local-frame primitive displacement ``(dx, dy, dtheta)``."""

    action = int(action)
    if action == ACTION_BOS:
        return 0.0, 0.0, 0.0
    if action == ACTION_FWD:
        return FORWARD_METERS, 0.0, 0.0
    if action == ACTION_LEFT:
        return 0.0, 0.0, TURN_RADIANS
    if action == ACTION_RIGHT:
        return 0.0, 0.0, -TURN_RADIANS
    if action == ACTION_STOP:
        raise ValueError("STOP has no visual successor in StreamVLN")
    raise ValueError(f"unknown action code: {action}")


def compose_pose(pose: Sequence[float], local_delta: Sequence[float]) -> np.ndarray:
    """Compose a world pose with a local-frame SE(2) displacement."""

    x, y, theta = (float(v) for v in pose)
    dx, dy, dtheta = (float(v) for v in local_delta)
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(
        [x + c * dx - s * dy, y + s * dx + c * dy, wrap_angle(theta + dtheta)],
        dtype=np.float64,
    )


def integrate_actions(actions: Iterable[int]) -> np.ndarray:
    """Integrate a trajectory's discrete actions into cumulative poses.

    ``actions[0]`` is the non-executed BOS marker and each ``actions[i]``
    drives the transition from frame ``i-1`` to frame ``i``.  The returned
    array has shape ``(T, 3)`` and starts at the origin.
    """

    values = [int(a) for a in actions]
    if not values:
        raise ValueError("trajectory must contain a BOS action")
    if values[0] != ACTION_BOS:
        raise ValueError("first action must be BOS (-1)")
    if any(a == ACTION_BOS for a in values[1:]):
        raise ValueError("BOS is only valid at trajectory index zero")

    poses = np.zeros((len(values), 3), dtype=np.float32)
    for i, action in enumerate(values[1:], start=1):
        poses[i] = compose_pose(poses[i - 1], action_to_twist(action)).astype(np.float32)
    return poses


def relative_pose(start: Sequence[float], goal: Sequence[float]) -> np.ndarray:
    """Express ``goal`` in ``start``'s local SE(2) frame."""

    sx, sy, stheta = (float(v) for v in start)
    gx, gy, gtheta = (float(v) for v in goal)
    dx, dy = gx - sx, gy - sy
    c, s = math.cos(stheta), math.sin(stheta)
    # Row-vector multiplication by R(theta), equivalent to R(-theta) column form.
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return np.asarray([local_x, local_y, wrap_angle(gtheta - stheta)], dtype=np.float64)


def inverse_pose(start: Sequence[float], local_delta: Sequence[float]) -> np.ndarray:
    """Recover a world goal from a start pose and local relative pose."""

    return compose_pose(start, local_delta)

