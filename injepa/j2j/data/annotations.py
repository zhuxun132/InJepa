"""Strict StreamVLN annotation parsing."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


_VIDEO_PATTERN = re.compile(
    r"^images/(?P<scan>[A-Za-z0-9]+)_"
    r"(?P<dataset>r2r|rxr)_(?P<episode>[0-9]{6})$"
)


@dataclass(frozen=True)
class AnnotationRow:
    """Validated identity and action fields from one annotation row."""

    source_id: Literal["R2R", "RxR"]
    row_index: int
    episode_id: int
    scan_id: str
    video_prefix: str
    actions: tuple[int, ...]


def parse_annotation_row(
    raw: object, *, source_id: str, row_index: int
) -> AnnotationRow:
    """Parse one R2R or RxR row and fail closed on any schema mismatch."""
    if source_id not in ("R2R", "RxR"):
        raise ValueError("invalid annotation source")
    if type(row_index) is not int or row_index < 0:
        raise ValueError("row_index must be a non-negative integer")
    if not isinstance(raw, Mapping):
        raise ValueError("annotation row must be a mapping")

    try:
        episode_id = raw["id"]
        video_prefix = raw["video"]
        actions = raw["actions"]
    except (KeyError, TypeError) as exc:
        raise ValueError("annotation row is missing required fields") from exc

    if type(episode_id) is not int or episode_id < 0:
        raise ValueError("annotation id must be a non-negative integer")
    if type(video_prefix) is not str:
        raise ValueError("annotation video must be a string")

    match = _VIDEO_PATTERN.fullmatch(video_prefix)
    if match is None:
        raise ValueError("annotation video does not match the canonical pattern")
    if match.group("dataset") != source_id.lower():
        raise ValueError("annotation dataset does not match its source")
    if int(match.group("episode")) != episode_id:
        raise ValueError("annotation video episode does not match its id")

    if type(actions) is not list or not actions:
        raise ValueError("annotation actions must be a non-empty list")
    if type(actions[0]) is not int or actions[0] != -1:
        raise ValueError("annotation actions must start with exact BOS -1")
    if any(type(action) is not int or not 0 <= action <= 3 for action in actions[1:]):
        raise ValueError("annotation action payload must contain ids 0 through 3")

    return AnnotationRow(
        source_id=source_id,
        row_index=row_index,
        episode_id=episode_id,
        scan_id=match.group("scan"),
        video_prefix=video_prefix,
        actions=tuple(actions),
    )
