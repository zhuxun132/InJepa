"""External byte-authority verification for a formal RAE-stream launch.

The authority JSON is intentionally kept outside the mutable source checkout.
It is generated after review and copied to the server separately.  The
launcher verifies both the authority file's externally supplied SHA-256 and
every listed source/config byte before it can create the official trainer
process.
"""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping

from .config_guard import (
    OFFICIAL_COMMIT,
    _clean_git_environment,
    _git_path_records,
    canonical_digest,
    sha256_file,
)


def _runtime_paths(root: Path) -> set[str]:
    """Enumerate the source/config surface that must appear in authority."""

    import subprocess

    try:
        tracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "-z"],
            text=True,
            stderr=subprocess.STDOUT,
            env=_clean_git_environment(),
        )
        untracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "-z"],
            text=True,
            stderr=subprocess.STDOUT,
            env=_clean_git_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not enumerate authority source surface below {root}") from exc
    paths = set(_git_path_records(tracked))
    for path in _git_path_records(untracked):
        if path.startswith(("rae_stream/", "scripts/", "tests/")) or path == "config/rae_stream.yaml":
            paths.add(path)
    return paths


def _validate_relative_path(path: str) -> None:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not path or "\\" in path:
        raise ValueError(f"authority contains unsafe relative path: {path!r}")


def _regular_target(root: Path, relative: str) -> Path:
    """Resolve a listed file only through canonical, non-symlink parents."""

    candidate = root / relative
    current = root
    parts = Path(relative).parts
    for component in parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f"authority source parent is a symlink: {current}")
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(f"authority target is not a regular file: {candidate}")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"authority target escapes repository root: {candidate}") from exc
    return candidate


def _checkout_bytecode(root: Path, source_paths: Iterable[str]) -> list[str]:
    """Find importable ``.pyc/.pyo`` siblings for the reviewed source set.

    The scan is intentionally limited to directories containing an authority
    listed Python source file; it never walks the potentially multi-terabyte
    RGB/data trees.  Formal launchers set ``PYTHONDONTWRITEBYTECODE`` but that
    flag does not stop Python from reading a forged stale cache, so a clean
    source checkout is an explicit gate.
    """

    code_dirs = {
        (root / Path(relative)).parent
        for relative in source_paths
        if Path(relative).suffix.lower() == ".py"
    }
    found: set[Path] = set()
    for directory in code_dirs:
        if not directory.is_dir():
            continue
        for candidate in directory.glob("*.pyc"):
            if candidate.is_file() or candidate.is_symlink():
                found.add(candidate)
        for candidate in directory.glob("*.pyo"):
            if candidate.is_file() or candidate.is_symlink():
                found.add(candidate)
        cache = directory / "__pycache__"
        if cache.is_dir():
            for pattern in ("*.pyc", "*.pyo"):
                for candidate in cache.glob(pattern):
                    if candidate.is_file() or candidate.is_symlink():
                        found.add(candidate)
    return sorted(str(path.relative_to(root)) for path in found)


def verify_code_authority(
    repo_root: str | Path,
    authority_path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Verify an externally pinned source/config byte map."""

    root = Path(repo_root).resolve()
    path = Path(authority_path).expanduser()
    if not path.is_absolute():
        path = path.absolute()
    # The authority is meaningful only when its bytes are pinned outside the
    # mutable checkout they attest.  Rejecting a checkout-local copy also
    # prevents a misleading "external" label in formal receipts.
    try:
        path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("code authority must be external and outside the mutable checkout")
    if path.is_symlink():
        raise ValueError(f"authority file must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise FileNotFoundError(f"authority file must be a regular file: {path}") from exc
    if resolved != path:
        raise ValueError(f"authority path must be canonical and contain no symlink components: {path}")

    # Read, hash, and parse one immutable descriptor snapshot.  The previous
    # implementation hashed by pathname and then opened the pathname again to
    # parse it, allowing replacement between those two operations.
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise FileNotFoundError(f"authority file must be a regular file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(f"authority file must be a regular non-symlink: {path}") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError("authority file changed before it was opened")
            authority_bytes = handle.read()
            after_read = os.fstat(handle.fileno())
    except Exception:
        raise
    after_path = os.stat(path, follow_symlinks=False)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened, field) != getattr(after_read, field) for field in identity_fields):
        raise RuntimeError("authority file changed while it was being read")
    if any(getattr(opened, field) != getattr(after_path, field) for field in identity_fields):
        raise RuntimeError("authority file was replaced while it was being read")

    actual_file_sha = hashlib.sha256(authority_bytes).hexdigest()
    if actual_file_sha != str(expected_sha256):
        raise RuntimeError(
            f"RAE-stream authority SHA mismatch: got {actual_file_sha}, expected {expected_sha256}"
        )
    try:
        payload = json.loads(authority_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"could not parse authority JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("authority JSON root must be an object")
    declared_payload_sha = payload.get("payload_sha256")
    without_digest = dict(payload)
    without_digest.pop("payload_sha256", None)
    computed_payload_sha = canonical_digest(without_digest)
    if declared_payload_sha != computed_payload_sha:
        raise RuntimeError("authority payload self-hash mismatch")
    if payload.get("schema") != "rae_stream_code_authority_v1":
        raise ValueError("unsupported RAE-stream authority schema")
    if payload.get("official_commit") != OFFICIAL_COMMIT:
        raise RuntimeError("authority is not bound to the admitted official RAE commit")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("authority must contain a non-empty files map")
    normalized: dict[str, str] = {}
    for raw_relative, raw_expected in files.items():
        relative = str(raw_relative)
        _validate_relative_path(relative)
        expected = str(raw_expected)
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
            raise ValueError(f"authority has invalid SHA for {relative}")
        target = _regular_target(root, relative)
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(f"authority source SHA mismatch: {relative}")
        normalized[relative] = actual
    expected_paths = _runtime_paths(root)
    actual_paths = set(normalized)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        details = []
        if missing:
            details.append("missing=" + ",".join(missing[:8]))
        if extra:
            details.append("extra=" + ",".join(extra[:8]))
        raise RuntimeError("authority source surface mismatch: " + "; ".join(details))
    bytecode = _checkout_bytecode(root, normalized)
    if bytecode:
        raise RuntimeError("checkout bytecode must be removed before formal launch: " + ", ".join(bytecode[:8]))
    return {
        "path": str(path),
        "sha256": actual_file_sha,
        "payload_sha256": computed_payload_sha,
        "official_commit": OFFICIAL_COMMIT,
        "files": dict(sorted(normalized.items())),
        "bytecode_files": [],
        "file_identity": {
            "device": int(opened.st_dev),
            "inode": int(opened.st_ino),
            "mode": int(opened.st_mode),
            "size": int(opened.st_size),
            "mtime_ns": int(opened.st_mtime_ns),
            "ctime_ns": int(opened.st_ctime_ns),
        },
    }


__all__ = ["verify_code_authority"]
