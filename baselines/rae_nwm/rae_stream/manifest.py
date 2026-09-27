"""Canonical, atomically written JSONL manifests."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable


def frame_sequence_sha256(paths: Iterable[str | Path]) -> str:
    """Hash every ordered frame with explicit index/length framing.

    The digest is used only for data provenance.  It is deliberately
    independent of JPEG decoding and therefore catches replacement/truncation
    of any frame while the validator is already walking the output tree.
    """

    digest = hashlib.sha256()
    for index, raw_path in enumerate(paths):
        path = Path(raw_path)
        digest.update(index.to_bytes(8, "big", signed=False))
        size = path.stat().st_size
        digest.update(int(size).to_bytes(8, "big", signed=False))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                if not isinstance(row, dict):
                    raise TypeError("manifest rows must be dictionaries")
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
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


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not an object")
            rows.append(row)
    return rows
