"""Canonical trajectory, frame, and transition key construction."""

from __future__ import annotations

import hashlib
import re
import struct
import unicodedata

from .annotations import AnnotationRow


_TRAJECTORY_DOMAIN = b"J2J_TRAJECTORY_KEY_V1\x00"
_SHA256_HEX = re.compile(r"[0-9A-Fa-f]{64}")
_WINDOWS_DRIVE_ABSOLUTE = re.compile(r"^[A-Za-z]:/")


def _length_prefix(value: bytes) -> bytes:
    if len(value) >= 2**64:
        raise ValueError("length-prefixed value is too large")
    return struct.pack("<Q", len(value)) + value


def _canonical_prefix(prefix: object) -> bytes:
    if type(prefix) is not str:
        raise ValueError("video prefix must be a string")
    normalized = unicodedata.normalize("NFC", prefix).replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_DRIVE_ABSOLUTE.match(normalized):
        raise ValueError("video prefix must be relative")
    segments = normalized.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ValueError("video prefix contains an invalid segment")
    return normalized.encode("utf-8")


def _validated_actions(actions: object) -> bytes:
    if type(actions) is not tuple or not actions:
        raise ValueError("actions must be a full non-empty tuple")
    if type(actions[0]) is not int or actions[0] != -1:
        raise ValueError("actions must start with exact BOS -1")
    if any(type(action) is not int or not 0 <= action <= 3 for action in actions[1:]):
        raise ValueError("action payload must contain ids 0 through 3")
    return bytes(actions[1:])


def trajectory_key(
    row: AnnotationRow, *, annotation_revision_sha256: str
) -> bytes:
    """Return the frozen SHA-256 identity for a complete annotation trajectory."""
    if not isinstance(row, AnnotationRow):
        raise ValueError("row must be an AnnotationRow")
    if row.source_id not in ("R2R", "RxR"):
        raise ValueError("invalid annotation source")
    if type(row.row_index) is not int or not 0 <= row.row_index < 2**64:
        raise ValueError("row_index must be an unsigned 64-bit integer")
    if type(annotation_revision_sha256) is not str or _SHA256_HEX.fullmatch(
        annotation_revision_sha256
    ) is None:
        raise ValueError("annotation revision must be exactly 64 hexadecimal characters")

    source = unicodedata.normalize("NFC", row.source_id).encode("utf-8")
    revision = bytes.fromhex(annotation_revision_sha256)
    prefix = _canonical_prefix(row.video_prefix)
    actions = _validated_actions(row.actions)

    digest = hashlib.sha256()
    digest.update(_TRAJECTORY_DOMAIN)
    digest.update(_length_prefix(source))
    digest.update(revision)
    digest.update(struct.pack("<Q", row.row_index))
    digest.update(_length_prefix(prefix))
    digest.update(_length_prefix(actions))
    return digest.digest()


def _indexed_key(key: bytes, index: int) -> bytes:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("key must be exactly 32 raw bytes")
    if type(index) is not int or not 0 <= index < 2**32:
        raise ValueError("index must be an unsigned 32-bit integer")
    return key + struct.pack("<I", index)


def frame_key(key: bytes, step: int) -> bytes:
    """Append a little-endian uint32 frame step to a raw trajectory key."""
    return _indexed_key(key, step)


def transition_key(key: bytes, t: int) -> bytes:
    """Append a little-endian uint32 transition origin to a trajectory key."""
    return _indexed_key(key, t)
