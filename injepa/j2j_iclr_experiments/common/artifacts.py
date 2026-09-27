"""Canonical immutable artifact primitives for current Context4 evaluation."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any

from j2j.receipts import canonical_json_bytes


_SHA256 = re.compile(r"[0-9a-f]{64}")
_FILE_IDENTITY_FIELDS = {"path", "bytes", "sha256"}


def deep_freeze(value: object) -> object:
    """Recursively detach and freeze a JSON-compatible value."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("artifact mapping keys must be strings")
        return MappingProxyType(
            {key: deep_freeze(child) for key, child in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(child) for child in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError("artifact contains a non-JSON value")


def to_plain_json(value: object) -> Any:
    """Return a detached JSON-native copy of a possibly frozen value."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("artifact mapping keys must be strings")
        return {key: to_plain_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_json(child) for child in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError("artifact contains a non-JSON value")


def require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def sha256_path(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path, *, name: str = "artifact") -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is unavailable: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError(f"{name} must contain positive bytes")
    return {
        "path": str(resolved),
        "bytes": size,
        "sha256": sha256_path(resolved),
    }


def validate_file_identity(
    value: object, *, name: str = "artifact"
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} FileIdentity must be a mapping")
    if set(value) != _FILE_IDENTITY_FIELDS:
        raise ValueError(f"{name} FileIdentity fields are incomplete or unknown")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} FileIdentity path must be a nonempty absolute string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute() or str(path.resolve()) != raw_path:
        raise ValueError(f"{name} FileIdentity path must be canonical and absolute")
    byte_count = value.get("bytes")
    if type(byte_count) is not int or byte_count <= 0:
        raise ValueError(f"{name} FileIdentity bytes must be a positive integer")
    expected_sha = require_sha256(value.get("sha256"), name=f"{name} SHA")
    observed = file_identity(path, name=name)
    if observed["bytes"] != byte_count or observed["sha256"] != expected_sha:
        raise ValueError(f"{name} live bytes/SHA identity drift")
    frozen = deep_freeze(observed)
    assert isinstance(frozen, Mapping)
    return frozen


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"artifact contains duplicate JSON field {key!r}")
        result[key] = value
    return result


def load_json_file_identity(
    value: object, *, name: str = "artifact"
) -> tuple[dict[str, Any], Mapping[str, object]]:
    identity = validate_file_identity(value, name=name)
    path = Path(str(identity["path"]))
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON value {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{name} JSON root must be a mapping")
    return payload, identity


def canonical_mapping_sha256(value: object) -> str:
    """Hash a JSON mapping using the package's canonical artifact framing."""

    return hashlib.sha256(canonical_json_bytes(to_plain_json(value))).hexdigest()


def create_once_json(path: str | Path, payload: object) -> Mapping[str, object]:
    """Publish canonical JSON atomically without ever replacing prior bytes."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"artifact destination already exists: {destination}")
    encoded = canonical_json_bytes(to_plain_json(payload))
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    identity = deep_freeze(file_identity(destination))
    assert isinstance(identity, Mapping)
    return identity


__all__ = [
    "canonical_mapping_sha256",
    "create_once_json",
    "deep_freeze",
    "file_identity",
    "load_json_file_identity",
    "require_sha256",
    "sha256_path",
    "to_plain_json",
    "validate_file_identity",
]
