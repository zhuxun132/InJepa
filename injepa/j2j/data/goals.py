"""Deterministic goal-view hashing for factual trajectories."""

from __future__ import annotations

import hashlib
import struct


_GOAL_VIEW_DOMAIN = b"J2J_GOAL_VIEW_V1\x00"


def goal_view_digest(key: bytes, *, t: int) -> bytes:
    """Hash a trajectory origin without incorporating a horizon."""
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("key must be exactly 32 raw bytes")
    if type(t) is not int or not 0 <= t < 2**32:
        raise ValueError("t must be an unsigned 32-bit integer")
    return hashlib.sha256(_GOAL_VIEW_DOMAIN + key + struct.pack("<I", t)).digest()


def goal_view_u64(key: bytes, *, t: int) -> int:
    """Interpret the first eight goal-view digest bytes as uint64 big-endian."""
    return int.from_bytes(goal_view_digest(key, t=t)[:8], "big", signed=False)


def hashed_goal_index(
    key: bytes, *, t: int, terminal: int, horizon: int
) -> int:
    """Choose a deterministic factual goal at least ``horizon`` steps ahead."""
    if any(type(value) is not int for value in (t, terminal, horizon)):
        raise ValueError("t, terminal, and horizon must be integers")
    if horizon < 1 or not 0 <= t < terminal or terminal - t < horizon:
        raise ValueError("origin is not eligible for the requested goal horizon")

    width = terminal - t - horizon + 1
    return t + horizon + goal_view_u64(key, t=t) % width
