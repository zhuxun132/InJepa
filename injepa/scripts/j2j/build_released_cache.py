#!/usr/bin/env python3
"""Build the released StreamVLN U -> whitening -> Z32 cache chain.

This is intentionally a thin orchestration layer.  Archive verification,
V-JEPA encoding, whitening, cache writing, and cache reading remain owned by
their existing modules; this file only wires those owners together and keeps
all run-specific paths and resource values in a caller-supplied mapping.
"""

from __future__ import annotations

from dataclasses import replace
import argparse
import copy
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any

import numpy as np
import torch
from PIL import Image


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from j2j.data import archives
from j2j.data import source as source_bridge
from j2j.data.keys import frame_key
from j2j.encoding import cache
from j2j.encoding import vjepa2


class ReleasedCacheDriverError(ValueError):
    """Raised when a cache-build authority or stage contract is incomplete."""


def raw_spatial_recipe_sha256(spatial_shape: Sequence[int]) -> str:
    if (not isinstance(spatial_shape, (list, tuple)) or len(spatial_shape) != 2
            or any(type(value) is not int or value <= 0 for value in spatial_shape)):
        raise ReleasedCacheDriverError("spatial_shape must contain two positive integers")
    recipe = {"operation": "identity", "representation": "native_spatial_tokens",
              "spatial_shape": list(spatial_shape)}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_raw_identity(config: Mapping[str, Any]) -> None:
    if "encoder_family" in config or "completed_local_spool" in config:
        raise ReleasedCacheDriverError("unsupported cache input configuration")
    identities = config.get("identities")
    if isinstance(identities, Mapping) and "encoder_family" in identities:
        raise ReleasedCacheDriverError("unsupported cache encoder identity")
    if config.get("representation") != "raw" or config.get("production_eligible") is not True:
        return
    identities = _required(config, "identities")
    if not isinstance(identities, Mapping):
        raise ReleasedCacheDriverError("identities must be a mapping")
    if identities.get("pool_sha256") != raw_spatial_recipe_sha256(_required(config, "spatial_shape")):
        raise ReleasedCacheDriverError("raw pool identity does not match native spatial recipe")
    checkpoint_sha = _sha256(_required(config, "checkpoint_sha256"), "checkpoint_sha256")
    if identities.get("checkpoint_sha256") != checkpoint_sha:
        raise ReleasedCacheDriverError("raw checkpoint identity does not match the loaded checkpoint")


def _required(config: Mapping[str, Any], name: str) -> Any:
    try:
        return config[name]
    except (KeyError, TypeError) as exc:
        raise ReleasedCacheDriverError(f"cache driver config is missing {name!r}") from exc


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReleasedCacheDriverError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ReleasedCacheDriverError(f"{name} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ReleasedCacheDriverError(f"{name} must be a SHA-256 string") from exc
    return value.lower()


def _resolve_parent_manifest(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the U parent from the admitted outer manifest in production.

    A production cache must not accept a caller-authored set of ledger
    digests.  Non-production fixtures retain the historical explicit parent
    seam so the focused unit tests and mechanical diagnostics remain small.
    """

    production = config.get("production_eligible")
    if production is True:
        manifest_locator = config.get("canonical_source_manifest", config.get("canonical_manifest"))
        if not isinstance(manifest_locator, (str, Path)):
            raise ReleasedCacheDriverError(
                "production cache requires a canonical source manifest"
            )
        expected = config.get(
            "canonical_source_manifest_sha256",
            config.get("canonical_manifest_sha256"),
        )
        if expected is None:
            raise ReleasedCacheDriverError(
                "production cache requires the canonical source manifest SHA-256"
            )
        try:
            from scripts.j2j.build_census import derive_task4_cache_parent

            parent = derive_task4_cache_parent(
                manifest_locator,
                expected_manifest_sha256=_sha256(
                    expected, "canonical source manifest SHA-256"
                ),
            )
        except ReleasedCacheDriverError:
            raise
        except Exception as exc:
            raise ReleasedCacheDriverError(
                "canonical source manifest could not derive the production cache parent"
            ) from exc
        supplied = config.get("parent_manifest")
        if supplied is not None and dict(supplied) != dict(parent):
            raise ReleasedCacheDriverError(
                "caller parent_manifest disagrees with the canonical source manifest"
            )
        return parent

    parent = config.get("parent_manifest")
    if not isinstance(parent, Mapping):
        raise ReleasedCacheDriverError("parent_manifest must be a mapping")
    return parent


def _resolve_alias_ledger_sha256(config: Mapping[str, Any]) -> str:
    """Resolve the cache alias identity from the same production outer manifest."""

    supplied = config.get("trajectory_alias_ledger_sha256")
    production = config.get("production_eligible")
    if production is True:
        manifest_locator = config.get("canonical_source_manifest", config.get("canonical_manifest"))
        expected = config.get(
            "canonical_source_manifest_sha256",
            config.get("canonical_manifest_sha256"),
        )
        if not isinstance(manifest_locator, (str, Path)) or expected is None:
            raise ReleasedCacheDriverError(
                "production cache requires a canonical source manifest for alias identity"
            )
        try:
            from scripts.j2j.build_census import derive_task4_cache_parent_and_alias

            _parent, derived = derive_task4_cache_parent_and_alias(
                manifest_locator,
                expected_manifest_sha256=_sha256(
                    expected, "canonical source manifest SHA-256"
                ),
            )
        except Exception as exc:
            if isinstance(exc, ReleasedCacheDriverError):
                raise
            raise ReleasedCacheDriverError(
                "canonical source manifest could not derive the alias ledger identity"
            ) from exc
        if supplied is not None and _sha256(supplied, "trajectory_alias_ledger_sha256") != derived:
            raise ReleasedCacheDriverError(
                "caller alias ledger identity disagrees with the canonical source manifest"
            )
        return derived
    if supplied is None:
        raise ReleasedCacheDriverError("trajectory_alias_ledger_sha256 is required")
    return _sha256(supplied, "trajectory_alias_ledger_sha256")


def _run_config(config: Mapping[str, Any], *, output: Path) -> dict[str, Any]:
    """Translate explicit driver settings to the existing cache run ABI."""

    devices = _required(config, "devices")
    if not isinstance(devices, list) or not devices or not all(
        isinstance(device, str) and device for device in devices
    ):
        raise ReleasedCacheDriverError("devices must be a non-empty configured list")
    workers = _required(config, "workers")
    if type(workers) is not int or workers < 0:
        raise ReleasedCacheDriverError("workers must be a non-negative integer")
    cache_batch = _positive_int(_required(config, "cache_batch"), "cache_batch")
    workspace = Path(str(_required(config, "workspace"))).expanduser()
    return {
        "cache_batch": cache_batch,
        "devices": list(devices),
        "workers": workers,
        "workspace": str(workspace),
        "output": str(output),
    }


def _resolve_whitening_fit_receipt_path(
    config: Mapping[str, Any],
) -> Path | None:
    """Resolve the optional fit-receipt path, required for production runs.

    The whitening owner already returns the complete scientific fit receipt.
    Keeping the path at the driver boundary avoids changing that numerical
    owner while ensuring a production cache run cannot silently discard its
    runtime/resource evidence.
    """

    value = config.get("whitening_fit_receipt")
    if value is None:
        if config.get("production_eligible") is True:
            raise ReleasedCacheDriverError(
                "production cache requires whitening_fit_receipt"
            )
        return None
    if not isinstance(value, (str, Path)) or not str(value):
        raise ReleasedCacheDriverError(
            "whitening_fit_receipt must be a non-empty path"
        )
    return Path(value).expanduser()


def _persist_whitening_fit_receipt(
    path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically persist the existing fit receipt as canonical JSON.

    This function deliberately performs no transformation of the receipt's
    fields or values.  Canonical serialization only fixes byte ordering and
    newline representation for reproducible hashing; the returned descriptor
    identifies exactly those bytes.
    """

    if not isinstance(receipt, Mapping):
        raise ReleasedCacheDriverError("whitening fit receipt must be a mapping")
    try:
        payload = (
            json.dumps(
                dict(receipt),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ReleasedCacheDriverError(
            "whitening fit receipt is not canonical-JSON serializable"
        ) from exc

    target = path
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    except OSError as exc:
        raise ReleasedCacheDriverError(
            "whitening fit receipt could not be written"
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    return {
        "path": str(target),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _base_identities(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _required(config, "identities")
    if not isinstance(raw, Mapping):
        raise ReleasedCacheDriverError("identities must be a mapping")
    expected = {
        "vjepa_source_commit",
        "checkpoint_sha256",
        "preprocess_sha256",
        "pool_sha256",
        "whitening_sha256",
    }
    if set(raw) != expected:
        raise ReleasedCacheDriverError("identities have an unexpected field ledger")
    result = dict(raw)
    for name in ("checkpoint_sha256", "preprocess_sha256", "pool_sha256"):
        result[name] = _sha256(result[name], name)
    if type(result["vjepa_source_commit"]) is not str or not result["vjepa_source_commit"]:
        raise ReleasedCacheDriverError("vjepa_source_commit must be non-empty")
    if result["whitening_sha256"] is not None:
        result["whitening_sha256"] = _sha256(result["whitening_sha256"], "whitening_sha256")
    return result


def _stage_kwargs(
    config: Mapping[str, Any],
    *,
    output: Path,
    stage: str,
    identities: Mapping[str, Any],
    parent_manifest: Mapping[str, Any],
    whitening_transform: cache.WhiteningTransform | None,
) -> dict[str, Any]:
    kwargs = {
        "expected_rows": _positive_int(_required(config, "expected_rows"), "expected_rows"),
        "stage": stage,
        "shard_rows": _positive_int(_required(config, "shard_rows"), "shard_rows"),
        "run_config": _run_config(config, output=output),
        "identities": dict(identities),
        "parent_manifest": parent_manifest,
        "whitening_transform": whitening_transform,
        "trajectory_alias_ledger_sha256": _resolve_alias_ledger_sha256(config),
        "reconstruction_command": list(
            _required(config, "reconstruction_command")
        ),
        "production_eligible": _required(config, "production_eligible"),
    }
    if "cache_write_workers" in config:
        kwargs["parallel_workers"] = _positive_int(
            config["cache_write_workers"], "cache_write_workers"
        )
    if "cache_write_window_rows" in config:
        kwargs["parallel_window_rows"] = _positive_int(
            config["cache_write_window_rows"], "cache_write_window_rows"
        )
    if "shard_subdirs" in config:
        kwargs["shard_subdirs"] = config["shard_subdirs"]
    return kwargs


def _with_artifact_sha(
    transform: cache.WhiteningTransform, artifact_sha256: str
) -> cache.WhiteningTransform:
    """Attach the saved artifact identity without changing whitening values."""

    if not isinstance(transform, cache.WhiteningTransform):
        # Test doubles and future cache-compatible transforms may expose the
        # same values without using the concrete dataclass.  The real writer
        # still receives the owner-validated object in production.
        return transform
    return replace(transform, artifact_sha256=_sha256(artifact_sha256, "whitening_sha256"))


def _train_only_rows(rows: Iterable[cache.CacheRecord]) -> Iterable[cache.CacheRecord]:
    """Expose only canonical project-train rows to the whitening owner."""

    for row in rows:
        partition = getattr(row, "partition", None)
        # The concrete cache reader always returns CacheRecord.  The fallback
        # keeps this narrow seam easy to exercise with protocol test doubles,
        # while never allowing a concrete dev/excluded row into fit_whitening.
        if partition is None or partition == "project-train":
            yield row


def run_cache_pipeline(
    records: Iterable[cache.CacheRecord], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Write direct RAW32 or run the legacy U/whitening/Z32 pipeline.

    ``records`` must already be a canonical, frame-key-sorted iterable.  The
    function never calls ``len`` or otherwise materializes it; the exact row
    count is supplied by the admitted census in ``config['expected_rows']``.
    """

    if not isinstance(config, Mapping):
        raise ReleasedCacheDriverError("cache driver config must be a mapping")
    _validate_raw_identity(config)
    if config.get("horizon") is not None:
        _positive_int(config["horizon"], "horizon")
    expected_rows = _positive_int(_required(config, "expected_rows"), "expected_rows")
    parent_manifest = _resolve_parent_manifest(config)
    identities = _base_identities(config)
    if identities["whitening_sha256"] is not None:
        raise ReleasedCacheDriverError("base U identities must have no whitening SHA")
    representation = config.get("representation", "whitened")
    if representation not in {"raw", "whitened"}:
        raise ReleasedCacheDriverError("representation must be raw or whitened")
    if representation == "raw":
        raw_output = Path(str(_required(config, "raw_output"))).expanduser()
        raw_manifest = cache.write_cache_streaming(
            raw_output,
            records,
            spatial_shape=_required(config, "spatial_shape"),
            **_stage_kwargs(
                config,
                output=raw_output,
                stage="RAW32",
                identities=identities,
                parent_manifest=parent_manifest,
                whitening_transform=None,
            ),
        )
        return {
            "raw_manifest": raw_manifest,
            "whitening": None,
            "expected_rows": expected_rows,
            "horizon": config.get("horizon"),
        }
    u_output = Path(str(_required(config, "u_output"))).expanduser()
    z32_output = Path(str(_required(config, "z32_output"))).expanduser()
    whitening_path = Path(str(_required(config, "whitening_path"))).expanduser()
    resource_limits = _required(config, "resource_limits")
    if not isinstance(resource_limits, Mapping):
        raise ReleasedCacheDriverError("resource_limits must be a mapping")
    fit_receipt_path = _resolve_whitening_fit_receipt_path(config)

    u_manifest = cache.write_cache_streaming(
        u_output,
        records,
        **_stage_kwargs(
            config,
            output=u_output,
            stage="U",
            identities=identities,
            parent_manifest=parent_manifest,
            whitening_transform=None,
        ),
    )

    u_rows = cache.iter_cache_rows(
        u_output,
        expected_parent_manifest_sha256=u_manifest["parent_manifest_sha256"],
        expected_identities=identities,
    )
    transform, fit_receipt = cache.fit_whitening(
        _train_only_rows(u_rows),
        workspace_root=_required(config, "workspace"),
        resource_limits=dict(resource_limits),
    )
    whitening_artifact = cache.save_whitening(whitening_path, transform)
    whitening_sha = _sha256(whitening_artifact["sha256"], "whitening artifact SHA")
    frozen_transform = _with_artifact_sha(transform, whitening_sha)
    z32_identities = dict(identities)
    z32_identities["whitening_sha256"] = whitening_sha

    z32_rows = cache.iter_cache_rows(
        u_output,
        expected_parent_manifest_sha256=u_manifest["parent_manifest_sha256"],
        expected_identities=identities,
    )
    z32_manifest = cache.write_cache_streaming(
        z32_output,
        z32_rows,
        **_stage_kwargs(
            config,
            output=z32_output,
            stage="Z32",
            identities=z32_identities,
            parent_manifest=u_manifest,
            whitening_transform=frozen_transform,
        ),
    )
    fit_receipt_artifact = None
    if fit_receipt_path is not None:
        fit_receipt_artifact = _persist_whitening_fit_receipt(
            fit_receipt_path,
            fit_receipt,
        )
    whitening_result: dict[str, Any] = {
        "fit": fit_receipt,
        **dict(whitening_artifact),
    }
    if fit_receipt_artifact is not None:
        whitening_result["fit_receipt_path"] = fit_receipt_artifact["path"]
        whitening_result["fit_receipt_bytes"] = fit_receipt_artifact["bytes"]
        whitening_result["fit_receipt_sha256"] = fit_receipt_artifact["sha256"]
    return {
        "u_manifest": u_manifest,
        "z32_manifest": z32_manifest,
        "whitening": whitening_result,
        "expected_rows": expected_rows,
        "horizon": config.get("horizon"),
    }


def _decode_payload(
    raw: bytes, *, expected_decoded_sha256: str | None = None
) -> torch.Tensor:
    """Decode one already verified JPEG to the V-JEPA CPU RGB ABI."""

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            rgb = image.convert("RGB")
            array = np.array(rgb, dtype=np.uint8, copy=True, order="C")
    except (OSError, ValueError, TypeError) as exc:
        raise ReleasedCacheDriverError("verified JPEG could not be decoded as RGB") from exc
    if array.ndim != 3 or array.shape[2] != 3:
        raise ReleasedCacheDriverError("decoded JPEG is not RGB")
    if expected_decoded_sha256 is not None:
        expected = _sha256(expected_decoded_sha256, "decoded_rgb_sha256")
        actual = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        if actual != expected:
            raise ReleasedCacheDriverError("decoded RGB identity disagrees with canonical manifest")
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()


def _iter_payloads(config: Mapping[str, Any]) -> Iterable[archives.VerifiedArchivePayload]:
    """Yield the two verified StreamVLN archive streams in fixed source order."""

    r2r = _required(config, "r2r_archive")
    r2r_sha = _required(config, "r2r_archive_sha256")
    parts = _required(config, "rxr_archive_parts")
    part_shas = _required(config, "rxr_archive_part_sha256s")
    if not isinstance(parts, (list, tuple)) or len(parts) != 2:
        raise ReleasedCacheDriverError("rxr_archive_parts must contain part0 and part1")
    if not isinstance(part_shas, (list, tuple)) or len(part_shas) != 2:
        raise ReleasedCacheDriverError("rxr_archive_part_sha256s must contain two hashes")
    selected = config.get('_selected_identities')
    if selected is not None:
        selected = set(selected)
        r2r_selected = {identity for identity in selected if '_r2r_' in identity[0]}
        rxr_selected = {identity for identity in selected if '_rxr_' in identity[0]}
        unclassified = selected - r2r_selected - rxr_selected
        if unclassified:
            raise ReleasedCacheDriverError('selected fallback identities must identify an R2R or RxR source')
    else:
        r2r_selected = None
        rxr_selected = None
    fast = selected is not None and config.get('preverified_archive_receipt') is not None
    if fast:
        receipt = Path(str(config['preverified_archive_receipt']))
        try:
            admitted = {item['name']: item['sha256'] for item in json.loads(receipt.read_text())['archives']}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReleasedCacheDriverError('preverified archive receipt is unreadable') from exc
        expected = {Path(str(r2r)).name: _sha256(r2r_sha, 'r2r_archive_sha256')}
        expected.update({Path(str(p)).name: _sha256(s, 'rxr archive part SHA') for p, s in zip(parts, part_shas)})
        if admitted != expected:
            raise ReleasedCacheDriverError('preverified archive receipt does not admit exact fallback archives')
    if r2r_selected is None or r2r_selected:
        yield from archives.iter_verified_payloads(
            r2r, _sha256(r2r_sha, "r2r_archive_sha256"), workers=config.get('archive_workers', 1),
            selected_identities=r2r_selected, verify_archive_sha=not fast, stop_when_selected_found=fast,
        )
    if rxr_selected is None or rxr_selected:
        yield from archives.iter_verified_payloads_from_parts(
            parts,
            tuple(_sha256(value, "rxr archive part SHA") for value in part_shas),
            logical_archive_name=str(_required(config, "rxr_logical_archive_name")),
            workers=config.get('archive_workers', 1), selected_identities=rxr_selected,
            verify_archive_sha=not fast, stop_when_selected_found=fast,
        )


def _load_plan(config: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[bytes]]:
    """Load canonical frame metadata through the existing source owner."""

    manifest_path = Path(str(_required(config, "canonical_manifest"))).expanduser()
    manifest_sha = _sha256(_required(config, "canonical_manifest_sha256"), "canonical_manifest_sha256")
    try:
        parsed = source_bridge._load_manifest(manifest_path, expected_sha256=manifest_sha)
    except Exception as exc:
        raise ReleasedCacheDriverError("canonical manifest could not be admitted") from exc

    revision = str(_required(config, "streamvln_revision"))
    if revision != cache._STREAMVLN_REVISION:
        raise ReleasedCacheDriverError("streamvln_revision does not match the frozen revision")
    by_member: dict[str, dict[str, Any]] = {}
    for item in parsed.items:
        if item.projection_partition == "excluded":
            continue
        trajectory = item.canonical_trajectory
        for step, (compressed, decoded) in enumerate(
            zip(trajectory.compressed_jpeg_sha256s, trajectory.decoded_rgb_sha256s)
        ):
            key = frame_key(trajectory.canonical_trajectory_key, step)
            try:
                # Physical RxR trajectories may use either verified layout;
                # this owner-derived source-preferred spelling is only the
                # plan label.  Spool joins below use parsed frame identity so
                # either spelling maps to the same canonical frame.
                member_path = archives.frame_member_path(
                    trajectory.canonical_video_prefix,
                    step + 1,
                )
            except Exception as exc:
                raise ReleasedCacheDriverError(
                    "canonical trajectory frame exceeds the released member grammar"
                ) from exc
            if member_path in by_member:
                raise ReleasedCacheDriverError("canonical manifest has duplicate frame members")
            by_member[member_path] = {
                "frame_key": key,
                "compressed_jpeg_sha256": compressed,
                "decoded_rgb_sha256": decoded,
                "source_dataset": trajectory.source_id,
                "revision": revision,
                "building": trajectory.scan_id,
                "partition": item.projection_partition,
                "frame_preimage": {
                    "canonical_trajectory_key": trajectory.canonical_trajectory_key.hex(),
                    "step": step,
                },
                "canonical": True,
                "trajectory_alias": False,
            }
    ordered = sorted((entry["frame_key"] for entry in by_member.values()))
    if len(ordered) != len(by_member) or not ordered:
        raise ReleasedCacheDriverError("canonical frame plan is empty or duplicated")
    return by_member, ordered


def _plan_by_frame_identity(
    plan: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Index a canonical frame plan by the archive owner's stable identity.

    Directory spelling is not part of a trajectory/frame identity: the
    published RxR release contains ``rgb/NNN.jpg`` plus
    ``rgb_images/NNN.jpg``/``rgb_images/NNNN.jpg`` layouts.  Parsing every plan key through
    ``j2j.data.archives`` keeps one grammar owner and rejects accidental
    aliases or cross-source paths.
    """

    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for member_path, metadata in plan.items():
        try:
            identity = archives.parse_frame_member(member_path)
        except Exception as exc:
            raise ReleasedCacheDriverError(
                "canonical frame plan contains an invalid released member path"
            ) from exc
        key = (identity.video_prefix, identity.frame_index)
        if key in indexed:
            raise ReleasedCacheDriverError(
                "canonical frame plan contains duplicate frame identities"
            )
        indexed[key] = metadata
    return indexed


def _iter_plan_payloads(config, plan):
    roots = [Path(p) for p in config.get('extracted_rgb_roots', [])]
    if not roots:
        yield from _iter_payloads(config)
        return
    workers = config.get('archive_workers', 1)
    trust_extracted = config.get('trusted_extracted_rgb') is True
    folders = {}
    for name in plan:
        identity = archives.parse_frame_member(name)
        prefix = identity.video_prefix
        if prefix not in folders:
            folders[prefix] = [root / prefix for root in roots if (root / prefix).is_dir()]
    def read(item):
        name, metadata = item
        identity = archives.parse_frame_member(name)
        for folder in folders[identity.video_prefix]:
            for layout in ('rgb', 'rgb_images'):
                for width in (3, 4):
                    p = folder / layout / (str(identity.frame_index).zfill(width) + '.jpg')
                    if not p.is_file():
                        continue
                    raw = p.read_bytes()
                    if trust_extracted:
                        compressed = hashlib.sha256(raw).hexdigest()
                        member = archives.VerifiedArchiveMember(archive_path=str(folder), member_path=name,
                            compressed_sha256=compressed,
                            decoded_rgb_sha256=metadata['decoded_rgb_sha256'], width=0, height=0)
                        return archives.VerifiedArchivePayload(member=member, jpeg_bytes=raw)
                    compressed = hashlib.sha256(raw).hexdigest()
                    decoded, w, h = archives._verify_jpeg(raw)
                    if compressed != metadata['compressed_jpeg_sha256'] or decoded != metadata['decoded_rgb_sha256']:
                        raise ReleasedCacheDriverError('extracted RGB identity mismatch: ' + str(p))
                    member = archives.VerifiedArchiveMember(archive_path=str(folder), member_path=name,
                        compressed_sha256=compressed, decoded_rgb_sha256=decoded, width=w, height=h)
                    return archives.VerifiedArchivePayload(member=member, jpeg_bytes=raw)
        return (identity.video_prefix, identity.frame_index)
    items = list(plan.items())
    missing = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(items), 2 * workers):
            for result in pool.map(read, items[start:start + 2 * workers]):
                if isinstance(result, tuple):
                    missing.add(result)
                else:
                    yield result
    print('EXTRACTED_RGB_COMPLETE', len(plan) - len(missing), 'MISSING', len(missing), flush=True)
    if missing:
        yield from _iter_payloads(dict(config, _selected_identities=missing))


def _spool_payloads(
    config: Mapping[str, Any], plan: Mapping[str, Mapping[str, Any]], spool_path: Path
) -> dict[bytes, tuple[int, int]]:
    """Verify archives once and spool only exact JPEG bytes, bounded by disk."""

    plan_by_identity = _plan_by_frame_identity(plan)
    references: dict[bytes, tuple[int, int]] = {}
    started = time.monotonic()
    spool_path.parent.mkdir(parents=True, exist_ok=True)
    with spool_path.open("xb") as handle:
        for payload in _iter_plan_payloads(config, plan):
            try:
                identity = archives.parse_frame_member(payload.member_path)
            except Exception as exc:
                raise ReleasedCacheDriverError(
                    "verified archive payload contains an invalid released member path"
                ) from exc
            metadata = plan_by_identity.get(
                (identity.video_prefix, identity.frame_index)
            )
            if metadata is None:
                # Aliases and excluded rows are verified for archive coverage
                # but are not cache occurrences.
                continue
            source_dataset = metadata.get("source_dataset")
            if source_dataset is not None and source_dataset != identity.source_id:
                raise ReleasedCacheDriverError(
                    "archive payload source disagrees with canonical manifest"
                )
            if payload.compressed_sha256 != metadata["compressed_jpeg_sha256"] or payload.decoded_rgb_sha256 != metadata["decoded_rgb_sha256"]:
                raise ReleasedCacheDriverError("archive payload identity disagrees with canonical manifest")
            key = metadata["frame_key"]
            if key in references:
                raise ReleasedCacheDriverError("canonical frame payload is duplicated")
            offset = handle.tell()
            raw = payload.jpeg_bytes
            handle.write(struct.pack("<Q", len(raw)))
            handle.write(raw)
            references[key] = (offset, len(raw))
            if len(references) % 10000 == 0:
                print('JPEG_VERIFIED', len(references), '/', len(plan),
                      'frames_s', round(len(references) / (time.monotonic() - started), 2), flush=True)
    expected_keys = {metadata["frame_key"] for metadata in plan_by_identity.values()}
    missing = expected_keys.difference(references)
    if missing:
        raise ReleasedCacheDriverError("verified archives are missing canonical frame payloads")
    return references


def _spool_descriptor_path(
    root: Path, descriptor: object, *, label: str, verify_sha256: bool = True
) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate one census payload-spool descriptor."""

    if not isinstance(descriptor, Mapping):
        raise ReleasedCacheDriverError(f"{label} descriptor is invalid")
    relpath = descriptor.get("relpath")
    if type(relpath) is not str or not relpath or "\\" in relpath:
        raise ReleasedCacheDriverError(f"{label} relpath is invalid")
    candidate = Path(relpath)
    if candidate.is_absolute() or ".." in candidate.parts or any(
        part in {"", "."} for part in candidate.parts
    ):
        raise ReleasedCacheDriverError(f"{label} relpath must stay below its root")
    path = root / candidate
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ReleasedCacheDriverError(f"{label} path is missing or escapes its root") from exc
    if not resolved.is_file():
        raise ReleasedCacheDriverError(f"{label} path is not a regular file")
    declared_bytes = descriptor.get("bytes")
    declared_sha = descriptor.get("sha256")
    if type(declared_bytes) is not int or declared_bytes < 0:
        raise ReleasedCacheDriverError(f"{label} bytes is invalid")
    declared_sha = _sha256(declared_sha, f"{label} sha256")
    try:
        actual_bytes = resolved.stat().st_size
    except OSError as exc:
        raise ReleasedCacheDriverError(f"{label} stat failed") from exc
    if actual_bytes != declared_bytes:
        raise ReleasedCacheDriverError(f"{label} byte count mismatch")
    if not verify_sha256:
        return resolved, dict(descriptor)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ReleasedCacheDriverError(f"{label} is unreadable") from exc
    if digest.hexdigest() != declared_sha:
        raise ReleasedCacheDriverError(f"{label} SHA-256 mismatch")
    return resolved, dict(descriptor)


def _strict_spool_json(raw: bytes, *, label: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReleasedCacheDriverError(f"{label} is not strict JSON") from exc


def _canonical_spool_row_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ReleasedCacheDriverError("payload spool index row is not canonical JSON") from exc


def _load_verified_payload_spool(
    receipt_path: Path | str,
    plan: Mapping[str, Mapping[str, Any]],
    *, verify_spool_sha256: bool = True,
) -> tuple[Path, dict[bytes, tuple[int, int]]]:
    """Load a census-produced payload spool without opening an archive.

    The index is checked in full, while only canonical frame members are
    retained in the in-memory locator map.  JPEG bytes are hash-checked again
    when ``_iter_encoded_records`` reads each requested frame.
    """

    receipt = Path(receipt_path).expanduser()
    if not receipt.is_file():
        raise ReleasedCacheDriverError("verified payload spool receipt is missing")
    try:
        sidecar_raw = receipt.read_bytes()
    except OSError as exc:
        raise ReleasedCacheDriverError("verified payload spool receipt is unreadable") from exc
    sidecar_value = _strict_spool_json(sidecar_raw, label="verified payload spool receipt")
    if not isinstance(sidecar_value, Mapping):
        raise ReleasedCacheDriverError("verified payload spool receipt must be a mapping")
    if not sidecar_raw.endswith(b"\n"):
        raise ReleasedCacheDriverError("verified payload spool receipt must end with LF")
    try:
        canonical_sidecar = (
            json.dumps(
                sidecar_value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ReleasedCacheDriverError("verified payload spool receipt is not canonical JSON") from exc
    if canonical_sidecar != sidecar_raw:
        raise ReleasedCacheDriverError("verified payload spool receipt is not canonical JSON")
    if sidecar_value.get("schema") != "J2J_VERIFIED_RGB_PAYLOAD_SPOOL_V1":
        raise ReleasedCacheDriverError("verified payload spool schema mismatch")
    if sidecar_value.get("format") != "u64_le_length_then_exact_jpeg_bytes":
        raise ReleasedCacheDriverError("verified payload spool format mismatch")
    total_rows = sidecar_value.get("row_count")
    if type(total_rows) is not int or total_rows < 0:
        raise ReleasedCacheDriverError("verified payload spool row_count is invalid")
    root = receipt.parent.resolve()
    spool_path, spool_desc = _spool_descriptor_path(
        root, sidecar_value.get("spool"), label="verified payload spool", verify_sha256=verify_spool_sha256
    )
    index_path, index_desc = _spool_descriptor_path(
        root, sidecar_value.get("index"), label="verified payload index"
    )
    index_rows = index_desc.get("row_count")
    if type(index_rows) is not int or index_rows != total_rows:
        raise ReleasedCacheDriverError("verified payload index row_count mismatch")
    spool_size = int(spool_desc["bytes"])

    # Plan labels are owner-derived paths.  Join physical spool rows by the
    # parsed (trajectory prefix, one-based frame index) identity because RxR
    # has two verified directory/width spellings.
    plan_by_identity = _plan_by_frame_identity(plan)
    references: dict[bytes, tuple[int, int]] = {}
    seen_canonical_keys: set[bytes] = set()
    seen_members: set[str] = set()
    digest = hashlib.sha256()
    actual_bytes = 0
    actual_rows = 0
    next_offset = 0
    try:
        with index_path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    raise ReleasedCacheDriverError("verified payload index row is missing a final LF")
                value = _strict_spool_json(line[:-1], label="verified payload index row")
                if not isinstance(value, Mapping):
                    raise ReleasedCacheDriverError("verified payload index row must be a mapping")
                if _canonical_spool_row_bytes(value) != line:
                    raise ReleasedCacheDriverError("verified payload index row is not canonical JSON")
                member_path = value.get("member_path")
                if type(member_path) is not str or not member_path:
                    raise ReleasedCacheDriverError("verified payload index member_path is invalid")
                try:
                    identity = archives.parse_frame_member(member_path)
                except Exception as exc:
                    raise ReleasedCacheDriverError(
                        "verified payload index member_path has an invalid released grammar"
                    ) from exc
                if member_path in seen_members:
                    raise ReleasedCacheDriverError("duplicate verified payload index member")
                seen_members.add(member_path)
                source_id = value.get("source_id")
                if source_id not in {"R2R", "RxR"}:
                    raise ReleasedCacheDriverError("verified payload index source_id is invalid")
                if source_id != identity.source_id:
                    raise ReleasedCacheDriverError(
                        "verified payload index source disagrees with member path"
                    )
                compressed = _sha256(
                    value.get("compressed_jpeg_sha256"),
                    "verified payload compressed SHA",
                )
                decoded = _sha256(
                    value.get("decoded_rgb_sha256"),
                    "verified payload decoded SHA",
                )
                offset = value.get("offset")
                length = value.get("length")
                if type(offset) is not int or offset < 0:
                    raise ReleasedCacheDriverError("verified payload offset is invalid")
                if type(length) is not int or length < 0:
                    raise ReleasedCacheDriverError("verified payload length is invalid")
                if offset != next_offset or offset + 8 + length > spool_size:
                    raise ReleasedCacheDriverError(
                        "verified payload index has a non-contiguous locator"
                    )
                next_offset = offset + 8 + length
                expected = plan_by_identity.get(
                    (identity.video_prefix, identity.frame_index)
                )
                if expected is not None:
                    expected_source = expected.get("source_dataset")
                    if expected_source is not None and expected_source != source_id:
                        raise ReleasedCacheDriverError(
                            "verified payload source disagrees with canonical manifest"
                        )
                    if compressed != expected.get("compressed_jpeg_sha256"):
                        raise ReleasedCacheDriverError(
                            "verified payload compressed identity disagrees with canonical manifest"
                        )
                    if decoded != expected.get("decoded_rgb_sha256"):
                        raise ReleasedCacheDriverError(
                            "verified payload decoded identity disagrees with canonical manifest"
                        )
                    key = expected.get("frame_key")
                    if type(key) is not bytes or key in references:
                        raise ReleasedCacheDriverError("duplicate canonical frame in verified payload index")
                    references[key] = (offset, length)
                    seen_canonical_keys.add(key)
                digest.update(line)
                actual_bytes += len(line)
                actual_rows += 1
    except OSError as exc:
        raise ReleasedCacheDriverError("verified payload index is unreadable") from exc

    if actual_rows != total_rows or actual_bytes != index_desc["bytes"]:
        raise ReleasedCacheDriverError("verified payload index descriptor does not match file")
    if digest.hexdigest() != index_desc["sha256"]:
        raise ReleasedCacheDriverError("verified payload index SHA-256 mismatch")
    if next_offset != spool_size:
        raise ReleasedCacheDriverError("verified payload index does not cover the spool exactly")
    expected_keys = {
        metadata["frame_key"] for metadata in plan_by_identity.values()
    }
    if expected_keys.difference(seen_canonical_keys):
        raise ReleasedCacheDriverError("missing canonical frame payload in verified spool")
    if len(references) != len(plan_by_identity):
        raise ReleasedCacheDriverError("canonical frame payload locator count mismatch")
    return spool_path, references


def _encode_frozen_images(model, images, *, raw):
    return vjepa2.encode_images(model, images, **({'pool_kernel_size': None} if raw else {}))


def _encode_device_batches(
    models, tensors, per_device_batch, raw, *, return_chunks=False
):
    if type(per_device_batch) is int:
        capacities = [per_device_batch] * len(models)
    elif isinstance(per_device_batch, (list, tuple)):
        capacities = list(per_device_batch)
    else:
        raise ReleasedCacheDriverError('per-device batch capacities must be an integer or a list')
    if len(capacities) != len(models) or any(type(value) is not int or value <= 0 for value in capacities):
        raise ReleasedCacheDriverError('per-device batch capacities must be positive and match devices')
    if len(tensors) > sum(capacities):
        raise ReleasedCacheDriverError('batch exceeds configured device capacity')
    def run(item):
        model, chunk = item
        result = _encode_frozen_images(model, chunk, raw=raw)
        return result.grid.detach().cpu().numpy()
    chunks = []
    start = 0
    for model, capacity in zip(models, capacities):
        chunk = tensors[start : start + capacity]
        if len(chunk):
            chunks.append((model, chunk))
        start += capacity
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        results = list(pool.map(run, chunks))
    if return_chunks:
        return results
    return np.concatenate(results, axis=0)


def _iter_encoded_records(
    config: Mapping[str, Any],
    plan: Mapping[str, Mapping[str, Any]],
    ordered_keys: Sequence[bytes],
    references: Mapping[bytes, tuple[int, int]],
    spool_path: Path,
    model: torch.nn.Module,
) -> Iterable[cache.CacheRecord]:
    """Read sorted spooled frames, batch-decode, and call frozen V-JEPA."""

    configured_batch = _positive_int(_required(config, "cache_batch"), "cache_batch")
    models = model if isinstance(model, list) else [model]
    per_device_batch = config.get("encoding_device_batches")
    if per_device_batch is None:
        per_device_batch = [configured_batch] * len(models)
    elif not isinstance(per_device_batch, list) or len(per_device_batch) != len(models) or any(
        type(value) is not int or value <= 0 for value in per_device_batch
    ):
        raise ReleasedCacheDriverError('encoding_device_batches must provide one positive batch size per device')
    batch_size = sum(per_device_batch)
    configured_workers = config.get("workers", 0)
    if type(configured_workers) is not int or configured_workers < 0:
        raise ReleasedCacheDriverError("workers must be a non-negative integer")
    plan_by_key = {metadata["frame_key"]: metadata for metadata in plan.values()}
    if len(plan_by_key) != len(plan):
        raise ReleasedCacheDriverError("canonical frame keys are duplicated")
    descriptor = os.open(spool_path, os.O_RDONLY)
    try:
        pending: list[tuple[dict[str, Any], torch.Tensor]] = []

        def flush() -> Iterable[cache.CacheRecord]:
            nonlocal pending
            if not pending:
                return ()
            # Images in a batch must have a common source shape for stacking;
            # flush on shape changes rather than padding or resizing outside
            # the official V-JEPA preprocessor.
            tensors = torch.stack([image for _, image in pending], dim=0)
            if len(models) > 1:
                grid_chunks = _encode_device_batches(
                    models,
                    tensors,
                    per_device_batch,
                    config.get('representation') == 'raw',
                    return_chunks=True,
                )
                grids = (grid for chunk in grid_chunks for grid in chunk)
            elif config.get("representation") == "raw":
                encoded = _encode_frozen_images(model, tensors, raw=True)
                grids = encoded.grid.detach().cpu().numpy()
            else:
                encoded = vjepa2.encode_images(model, tensors)
                grids = encoded.grid.detach().cpu().numpy()
            output: list[cache.CacheRecord] = []
            for (metadata, _), grid in zip(pending, grids):
                output.append(
                    cache.CacheRecord(
                        frame_key=metadata["frame_key"],
                        grid=np.asarray(grid, dtype="<f4", order="C"),
                        compressed_jpeg_sha256=metadata["compressed_jpeg_sha256"],
                        decoded_rgb_sha256=metadata["decoded_rgb_sha256"],
                        source_dataset=metadata["source_dataset"],
                        revision=metadata["revision"],
                        building=metadata["building"],
                        partition=metadata["partition"],
                        frame_preimage=metadata["frame_preimage"],
                        canonical=True,
                        trajectory_alias=False,
                    )
                )
            pending = []
            return tuple(output)

        def pread_exact(length: int, offset: int) -> bytes:
            chunks: list[bytes] = []
            remaining = length
            position = offset
            while remaining:
                chunk = os.pread(descriptor, remaining, position)
                if not chunk:
                    raise ReleasedCacheDriverError("payload spool record is truncated")
                chunks.append(chunk)
                position += len(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def read_payload(key: bytes) -> tuple[dict[str, Any], bytes]:
            try:
                offset, length = references[key]
            except KeyError as exc:
                raise ReleasedCacheDriverError("sorted frame plan lacks a payload") from exc
            header = pread_exact(8, offset)
            if len(header) != 8 or struct.unpack("<Q", header)[0] != length:
                raise ReleasedCacheDriverError("payload spool record is truncated")
            raw = pread_exact(length, offset + 8)
            metadata = plan_by_key[key]
            if hashlib.sha256(raw).hexdigest() != metadata["compressed_jpeg_sha256"]:
                raise ReleasedCacheDriverError("payload spool compressed identity mismatch")
            return metadata, raw

        def decode_payload(item: tuple[dict[str, Any], bytes]) -> tuple[dict[str, Any], torch.Tensor]:
            metadata, raw = item
            return metadata, _decode_payload(
                raw,
                expected_decoded_sha256=metadata["decoded_rgb_sha256"],
            )

        def read_and_decode(key: bytes) -> tuple[dict[str, Any], torch.Tensor]:
            return decode_payload(read_payload(key))

        def consume_decoded(
            decoded_items: Iterable[tuple[dict[str, Any], torch.Tensor]],
        ) -> Iterable[cache.CacheRecord]:
            nonlocal shape
            for metadata, image in decoded_items:
                current_shape = (int(image.shape[1]), int(image.shape[2]))
                if shape is not None and (
                    current_shape != shape or len(pending) >= batch_size
                ):
                    yield from flush()
                shape = current_shape
                pending.append((metadata, image))

        shape: tuple[int, int] | None = None
        executor: ThreadPoolExecutor | None = None
        if configured_workers > 1:
            executor = ThreadPoolExecutor(max_workers=configured_workers)
        try:
            for start in range(0, len(ordered_keys), batch_size):
                keys = ordered_keys[start : start + batch_size]
                if executor is None:
                    decoded = (read_and_decode(key) for key in keys)
                else:
                    # executor.map preserves input order, so scheduling cannot
                    # change cache row order or V-JEPA batch membership.
                    decoded = executor.map(read_and_decode, keys)
                yield from consume_decoded(decoded)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
        yield from flush()
    finally:
        os.close(descriptor)


def _load_and_encode_records(config: Mapping[str, Any]) -> tuple[Iterable[cache.CacheRecord], int]:
    """Prepare a replayable sorted record stream and its census row count."""

    _validate_raw_identity(config)
    plan, ordered = _load_plan(config)
    expected_rows = _positive_int(_required(config, "expected_rows"), "expected_rows")
    if expected_rows != len(ordered):
        raise ReleasedCacheDriverError("expected_rows disagrees with canonical frame plan")
    workspace = Path(str(_required(config, "workspace"))).expanduser()
    external_spool = config.get("verified_payload_spool_receipt")
    if external_spool is None:
        spool_path = workspace / "verified-jpeg-payloads.spool"
        if spool_path.exists():
            raise ReleasedCacheDriverError("payload spool already exists; choose a fresh workspace")
        references = _spool_payloads(config, plan, spool_path)
    else:
        spool_path, references = _load_verified_payload_spool(
            Path(str(external_spool)).expanduser(), plan,
            verify_spool_sha256=config.get("verify_payload_spool_sha256", True),
        )
    source_root = Path(str(_required(config, "vjepa_source_root"))).expanduser()
    checkpoint = Path(str(_required(config, "checkpoint"))).expanduser()
    model = vjepa2.load_frozen_vjepa2(
        source_root=source_root,
        checkpoint=checkpoint,
        expected_sha256=_sha256(_required(config, "checkpoint_sha256"), "checkpoint_sha256"),
        expected_bytes=_positive_int(_required(config, "checkpoint_bytes"), "checkpoint_bytes"),
    )
    devices = config.get('encoding_devices', [str(_required(config, 'device'))])
    if not isinstance(devices, list) or not devices or len(set(devices)) != len(devices):
        raise ReleasedCacheDriverError('encoding_devices must be distinct nonempty devices')
    if len(devices) > 1:
        model = [copy.deepcopy(model).to(torch.device(device)) for device in devices]
    else:
        model.to(torch.device(devices[0]))
    return (
        _iter_encoded_records(config, plan, ordered, references, spool_path, model),
        expected_rows,
    )


def build_released_cache(config: Mapping[str, Any]) -> dict[str, Any]:
    """Admit verified assets, encode U, fit/save whitening, and emit Z32."""

    # Resolve all production authority before opening archives or loading the
    # frozen encoder.  This keeps an invalid parent a cheap, deterministic
    # admission failure rather than a late failure after hours of encoding.
    _validate_raw_identity(config)
    if config.get("production_eligible") is True:
        _resolve_parent_manifest(config)
        _resolve_alias_ledger_sha256(config)
    records, expected_rows = _load_and_encode_records(config)
    # The record generator is replayed from the spool for exactly one U pass;
    # Z32 is then derived from the immutable U cache, so no second RGB decode
    # or encoder call can alter the published U rows.
    cfg = dict(config)
    cfg["expected_rows"] = expected_rows
    result = run_cache_pipeline(records, cfg)
    # A census-produced spool is an immutable cross-stage artifact and must
    # survive for its own receipt/readback audit.  Only delete the temporary
    # spool created by this invocation when no external receipt was supplied.
    if config.get("verified_payload_spool_receipt") is None:
        spool_path = Path(str(_required(config, "workspace"))).expanduser() / "verified-jpeg-payloads.spool"
        try:
            spool_path.unlink()
        except FileNotFoundError:
            pass
    return result


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            value = yaml.safe_load(raw.decode("utf-8"))
        else:
            value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ReleasedCacheDriverError("cache driver config is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ReleasedCacheDriverError("cache driver config must be a mapping")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = build_released_cache(_load_config(args.config))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"ok": True, "expected_rows": result["expected_rows"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
