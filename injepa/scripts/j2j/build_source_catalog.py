#!/usr/bin/env python3
"""Build the compact released StreamVLN source catalog from admitted caches.

This command is only a parameterized adapter around the existing cache opener
and source-catalog writer.  It does not parse annotations or archives, read
grid payloads, construct samples, or create a training runtime.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from j2j.data import source as source_module
from j2j.encoding import cache as cache_module


class SourceCatalogDriverError(ValueError):
    """Raised when the caller has not supplied a complete cache authority."""


def _required(config: Mapping[str, Any], name: str) -> Any:
    try:
        value = config[name]
    except (KeyError, TypeError) as exc:
        raise SourceCatalogDriverError(
            f"source catalog config is missing {name!r}"
        ) from exc
    if value is None:
        raise SourceCatalogDriverError(
            f"source catalog config is missing {name!r}"
        )
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise SourceCatalogDriverError(f"{name} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise SourceCatalogDriverError(f"{name} must be a SHA-256 string") from exc
    return value.lower()


def _pathlike(value: object, name: str) -> str | os.PathLike[str]:
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise SourceCatalogDriverError(f"{name} must be a non-empty path")
    return value


def build_source_catalog(config: Mapping[str, Any]) -> dict[str, Any]:
    """Open one admitted Z32 cache and delegate catalog publication."""

    if not isinstance(config, Mapping):
        raise SourceCatalogDriverError("source catalog config must be a mapping")

    canonical_manifest = _pathlike(
        _required(config, "canonical_manifest"), "canonical_manifest"
    )
    manifest_sha = _sha256(
        _required(config, "expected_manifest_sha256"),
        "expected_manifest_sha256",
    )
    u_cache_manifest = _pathlike(
        _required(config, "u_cache_manifest"), "u_cache_manifest"
    )
    z32_cache_dir = _pathlike(
        _required(config, "z32_cache_dir"), "z32_cache_dir"
    )
    parent_sha = _sha256(
        _required(config, "expected_parent_manifest_sha256"),
        "expected_parent_manifest_sha256",
    )
    identities = _required(config, "expected_identities")
    if not isinstance(identities, Mapping):
        raise SourceCatalogDriverError("expected_identities must be a mapping")
    require_stage = config.get("require_stage", "Z32")
    if type(require_stage) is not str or not require_stage:
        raise SourceCatalogDriverError("require_stage must be a non-empty string")
    production = config.get("require_production_eligible", True)
    if type(production) is not bool:
        raise SourceCatalogDriverError(
            "require_production_eligible must be boolean"
        )
    output_dir = _pathlike(_required(config, "output_dir"), "output_dir")

    store = cache_module.open_cache_store(
        z32_cache_dir,
        expected_parent_manifest_sha256=parent_sha,
        expected_identities=dict(identities),
        require_stage=require_stage,
        require_training_eligible=True,
        require_production_eligible=production,
    )
    catalog_path = source_module.write_source_catalog(
        output_dir,
        canonical_manifest=canonical_manifest,
        expected_manifest_sha256=manifest_sha,
        u_cache_manifest=u_cache_manifest,
        z32_cache_store=store,
    )
    path = Path(catalog_path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SourceCatalogDriverError(
            "source catalog writer did not publish a readable artifact"
        ) from exc
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            value = yaml.safe_load(raw.decode("utf-8"))
        else:
            value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise SourceCatalogDriverError(
            "source catalog config is unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise SourceCatalogDriverError("source catalog config must be a mapping")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_source_catalog(_load_config(args.config))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
