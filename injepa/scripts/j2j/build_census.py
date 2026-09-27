#!/usr/bin/env python3
"""Build a fail-closed StreamVLN H4 census and machine-readable receipt."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml

from j2j.data.annotations import AnnotationRow, parse_annotation_row
from j2j.data.archives import (
    ArchiveValidationError,
    ArchiveIdentityError,
    VerifiedArchiveMember,
    VerifiedArchivePayload,
    iter_verified_members,
    iter_verified_members_from_parts,
    iter_verified_payloads,
    iter_verified_payloads_from_parts,
    ordered_frame_member_paths,
    parse_frame_member,
)
from j2j.data.census import (
    CensusTrajectory,
    CompactCensusAccumulator,
    GoalViewOrigin,
    ProductionCensus,
    census,
)
from j2j.data.keys import trajectory_key
from j2j.data.splits import (
    BuildingSets,
    CanonicalTrajectory,
    PointNavIdentityError,
    TrajectoryRecord,
    assert_official_set_authority,
    assign_building_partition,
    canonicalize_trajectories,
    freeze_building_sets,
    load_pointnav_split_identity,
)


_SHA256_HEX = re.compile(r"[0-9A-Fa-f]{64}")
_WINDOWS_DRIVE_ABSOLUTE = re.compile(r"^[A-Za-z]:/")
_DEV_DOMAIN = b"J2J_DEV_BUILDING_V1\x00"
_DEDUP_DOMAIN = b"J2J_TRAJECTORY_DEDUP_V1\x00"
_FROZEN_ASSET_SCOPES = frozenset({"synthetic_fixture", "official"})
_CENSUS_MAIN_HORIZON = 4
_SCOPE_KEYS = frozenset(
    {"census_only", "training_claim", "H_main", "H_run", "horizon_role"}
)

_OFFICIAL_STREAM_REVISION = "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9"
_OFFICIAL_ASSET_IDENTITY = {
    ("annotations", "R2R", "annotation", 0): (
        13_018_034,
        "d0a1c255c2641c61e2d586756a3208ed58e78cf7e781359305ce8f1360d54305",
    ),
    ("annotations", "RxR", "annotation", 1): (
        42_093_931,
        "b40ecb7d8773edd9c045ca042446f962b1c9305c4149cbaeaf851fc7014ee330",
    ),
    ("rgb_archives", "R2R", "rgb_archive", 0): (
        23_578_753_959,
        "9dccce7260f8db486b99cb0dcf5f443604a4c7dca3d980d367ffc516da06c59a",
    ),
    ("rgb_archives", "RxR", "rgb_archive_part", 0): (
        42_949_672_960,
        "c855412c0af87e19dcdd8c31fe46357cc5bdebc9b9666412e1252c6ca3e5345d",
    ),
    ("rgb_archives", "RxR", "rgb_archive_part", 1): (
        25_498_645_409,
        "7227ce92e90b07ff7e8c70c9c855060f00d3fcd364f2c02fd94135539e8493b5",
    ),
    ("episode_archives", "mp3d", "pointnav_episodes", 0): (
        418_680_926,
        "c9c28135cf572dc2a9b208d486bfe7bcd7562474486d06d2fc552058afcc1b6b",
    ),
    ("episode_archives", "gibson", "pointnav_episodes", 1): (
        403_133_836,
        "ec46befe911c080662f2c01ac56c7015c28f31c83f432fa836ff784c99e26e5a",
    ),
}


class ManifestValidationError(ValueError):
    """Raised for malformed or non-authoritative asset manifests."""


class AssetIdentityError(ManifestValidationError):
    """Raised when consumed annotation bytes miss their manifest identity."""


class PipelineValidationError(ValueError):
    """Raised when verified assets fail parsing, alignment, or census gates."""


@dataclass(frozen=True)
class _CapturedJson:
    value: object
    raw_bytes: bytes


class AssetEntry:
    """Validated manifest metadata plus its caller-owned asset root."""

    __slots__ = (
        "section",
        "logical_name",
        "dataset",
        "role",
        "revision",
        "order",
        "relative_path",
        "bytes",
        "sha256",
        "root",
    )

    def __init__(
        self,
        *,
        section: str,
        logical_name: str,
        dataset: str,
        role: str,
        revision: str,
        order: int,
        relative_path: str,
        bytes: int,
        sha256: str,
        root: Path,
    ) -> None:
        self.section = section
        self.logical_name = logical_name
        self.dataset = dataset
        self.role = role
        self.revision = revision
        self.order = order
        self.relative_path = relative_path
        self.bytes = bytes
        self.sha256 = sha256
        self.root = root

    @property
    def path(self) -> Path:
        return self.root.joinpath(*self.relative_path.split("/"))

    def receipt_identity(self) -> dict[str, object]:
        return {
            "section": self.section,
            "logical_name": self.logical_name,
            "dataset": self.dataset,
            "role": self.role,
            "revision": self.revision,
            "order": self.order,
            "relative_path": self.relative_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


def _validate_run_authority(*, asset_scope: str, dev_count: int) -> bool:
    """Return whether this run has the exact frozen production authority."""
    if asset_scope not in _FROZEN_ASSET_SCOPES:
        raise PipelineValidationError("asset_scope is not frozen for this pipeline")
    if asset_scope == "official":
        if type(dev_count) is int and dev_count == 11:
            return True
        raise PipelineValidationError(
            "official asset_scope requires exact dev_count 11"
        )
    if type(dev_count) is not int or dev_count < 1:
        raise PipelineValidationError("dev_count must be a positive exact integer")
    return False


def execution_scope(*, h_run: int) -> dict[str, object]:
    """Build the non-training scope attached to a census result.

    ``H_run`` is a library parameter, not a finite experiment whitelist.  The
    scope is deliberately small: it records only the evidence level and the
    relationship to the V9 H=4 main run.  It is not an execution or access
    control mechanism.
    """

    if type(h_run) is not int or h_run < 1:
        raise PipelineValidationError(
            "H_run must be a positive exact integer for census scope"
        )
    role = (
        "main_h4_census"
        if h_run == _CENSUS_MAIN_HORIZON
        else "nonmain_census_extension"
    )
    return {
        "census_only": True,
        "training_claim": False,
        "H_main": _CENSUS_MAIN_HORIZON,
        "H_run": h_run,
        "horizon_role": role,
    }


def validate_execution_scope(scope: Mapping[str, object]) -> dict[str, object]:
    """Validate and canonicalize a previously serialized census scope."""

    if not isinstance(scope, Mapping):
        raise PipelineValidationError("execution_scope must be a mapping")
    if set(scope) != set(_SCOPE_KEYS):
        raise PipelineValidationError(
            "execution_scope fields must be exactly the frozen scope schema"
        )
    if type(scope["census_only"]) is not bool or scope["census_only"] is not True:
        raise PipelineValidationError("execution_scope.census_only must be true")
    if (
        type(scope["training_claim"]) is not bool
        or scope["training_claim"] is not False
    ):
        raise PipelineValidationError(
            "execution_scope.training_claim must be false"
        )
    if type(scope["H_main"]) is not int or scope["H_main"] != _CENSUS_MAIN_HORIZON:
        raise PipelineValidationError("execution_scope.H_main must be exact H=4")
    h_run = scope["H_run"]
    if type(h_run) is not int or h_run < 1:
        raise PipelineValidationError(
            "execution_scope.H_run must be a positive exact integer"
        )
    expected_role = (
        "main_h4_census"
        if h_run == _CENSUS_MAIN_HORIZON
        else "nonmain_census_extension"
    )
    if scope["horizon_role"] != expected_role:
        raise PipelineValidationError(
            "execution_scope.horizon_role disagrees with H_run"
        )
    # Return a fresh ordinary dict so callers cannot mutate the input mapping
    # while a receipt is being assembled.
    return {
        "census_only": True,
        "training_claim": False,
        "H_main": _CENSUS_MAIN_HORIZON,
        "H_run": h_run,
        "horizon_role": expected_role,
    }


def require_released_production_identity(
    identity: object = None,
    *,
    asset_scope: object = None,
    entries: Sequence[AssetEntry] | None = None,
    streamvln_revision: object = None,
    dev_count: object = 11,
) -> bool:
    """Require the frozen official StreamVLN identity before production use.

    The existing manifest validator remains the source of the asset-entry
    schema and official byte/SHA table.  This wrapper only binds that validator
    to the production census owner; it does not create a second identity
    ledger.  ``identity`` accepts the historical positional asset-scope form
    and a mapping carrying the same fields, which keeps the narrow helper
    usable by focused checks without weakening the official path.
    """

    candidate = asset_scope if asset_scope is not None else identity
    if isinstance(candidate, Mapping):
        mapping = candidate
        candidate = mapping.get("asset_scope", mapping.get("scope"))
        if streamvln_revision is None:
            streamvln_revision = mapping.get("streamvln_revision")
        if dev_count == 11 and "dev_count" in mapping:
            dev_count = mapping["dev_count"]
        if entries is None and isinstance(mapping.get("entries"), Sequence):
            entries = mapping["entries"]  # type: ignore[assignment]

    def _identity_error(detail: str) -> PipelineValidationError:
        # Keep the pre-implementation RED sentinel in the diagnostic so an
        # old focused harness reports one stable missing-contract cause.  The
        # specific RELEASED_ASSET_REQUIRED wording remains visible as well.
        return PipelineValidationError(
            "MISSING_CENSUS_HORIZON_V2: RELEASED_ASSET_REQUIRED: " + detail
        )

    if candidate != "official":
        raise _identity_error("production census requires official asset identity")
    if type(dev_count) is not int or dev_count != 11:
        raise _identity_error("official identity requires exact dev_count 11")
    if streamvln_revision is not None and streamvln_revision != _OFFICIAL_STREAM_REVISION:
        raise _identity_error("StreamVLN revision does not match the frozen official revision")
    if entries is not None:
        try:
            _assert_official_asset_manifest(entries)
        except Exception as exc:
            raise _identity_error(str(exc)) from exc
    return True


def _strict_json_bytes(payload: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )


def _strict_json_file(path: Path) -> _CapturedJson:
    with path.open("rb") as handle:
        payload = handle.read()
    return _CapturedJson(value=_strict_json_bytes(payload), raw_bytes=payload)


def _canonical_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ManifestValidationError(f"{label} must be a non-empty POSIX path")
    normalized = unicodedata.normalize("NFC", value)
    if normalized.startswith("/") or _WINDOWS_DRIVE_ABSOLUTE.match(normalized):
        raise ManifestValidationError(f"{label} must be relative")
    if any(segment in ("", ".", "..") for segment in normalized.split("/")):
        raise ManifestValidationError(f"{label} contains an invalid segment")
    return normalized


def _exact_string(mapping: Mapping[str, object], field: str) -> str:
    value = mapping.get(field)
    if type(value) is not str or not value:
        raise ManifestValidationError(f"manifest field {field} must be a non-empty string")
    return value


def _parse_entry(
    raw: object, *, section: str, root: Path
) -> AssetEntry:
    if not isinstance(raw, Mapping):
        raise ManifestValidationError(f"{section} entries must be objects")
    expected_fields = {
        "logical_name",
        "dataset",
        "role",
        "revision",
        "order",
        "relative_path",
        "bytes",
        "sha256",
    }
    if set(raw) != expected_fields:
        raise ManifestValidationError(f"{section} entry fields do not match the schema")
    order = raw["order"]
    byte_count = raw["bytes"]
    sha256 = raw["sha256"]
    if type(order) is not int or order < 0:
        raise ManifestValidationError("asset order must be a non-negative exact integer")
    if type(byte_count) is not int or byte_count < 1:
        raise ManifestValidationError("asset bytes must be a positive exact integer")
    if type(sha256) is not str or _SHA256_HEX.fullmatch(sha256) is None:
        raise ManifestValidationError("asset SHA-256 must be exactly 64 hexadecimal characters")
    return AssetEntry(
        section=section,
        logical_name=_canonical_relative_path(raw["logical_name"], label="logical_name"),
        dataset=_exact_string(raw, "dataset"),
        role=_exact_string(raw, "role"),
        revision=_exact_string(raw, "revision"),
        order=order,
        relative_path=_canonical_relative_path(raw["relative_path"], label="relative_path"),
        bytes=byte_count,
        sha256=sha256.lower(),
        root=root,
    )


def _validate_manifest(
    payload: object,
    *,
    annotation_root: Path,
    rgb_root: Path,
    episode_root: Path,
) -> tuple[str, tuple[AssetEntry, ...]]:
    if not isinstance(payload, Mapping):
        raise ManifestValidationError("asset manifest must be a JSON object")
    if set(payload) != {
        "schema",
        "asset_scope",
        "annotations",
        "rgb_archives",
        "episode_archives",
    }:
        raise ManifestValidationError("asset manifest top-level fields do not match the schema")
    if payload["schema"] != "J2J_ASSET_MANIFEST_V1":
        raise ManifestValidationError("unsupported asset manifest schema")
    asset_scope = payload["asset_scope"]
    if type(asset_scope) is not str or asset_scope not in _FROZEN_ASSET_SCOPES:
        raise ManifestValidationError(
            "asset_scope must be exactly synthetic_fixture or official"
        )

    specifications = (
        ("annotations", annotation_root, 2),
        ("rgb_archives", rgb_root, 3),
        ("episode_archives", episode_root, 2),
    )
    entries: list[AssetEntry] = []
    for section, root, exact_count in specifications:
        raw_entries = payload[section]
        if type(raw_entries) is not list or len(raw_entries) != exact_count:
            raise ManifestValidationError(
                f"{section} must contain exactly {exact_count} entries"
            )
        entries.extend(
            _parse_entry(raw, section=section, root=root) for raw in raw_entries
        )

    identities = [(entry.section, entry.dataset, entry.role, entry.order) for entry in entries]
    expected_identities = {
        ("annotations", "R2R", "annotation", 0),
        ("annotations", "RxR", "annotation", 1),
        ("rgb_archives", "R2R", "rgb_archive", 0),
        ("rgb_archives", "RxR", "rgb_archive_part", 0),
        ("rgb_archives", "RxR", "rgb_archive_part", 1),
        ("episode_archives", "mp3d", "pointnav_episodes", 0),
        ("episode_archives", "gibson", "pointnav_episodes", 1),
    }
    if len(set(identities)) != len(identities) or set(identities) != expected_identities:
        raise ManifestValidationError("asset roles, datasets, or part orders are invalid")
    if len({entry.logical_name for entry in entries}) != len(entries):
        raise ManifestValidationError("manifest logical names must be unique")

    by_identity = {
        (entry.section, entry.dataset, entry.role, entry.order): entry
        for entry in entries
    }
    for dataset in ("R2R", "RxR"):
        annotation = by_identity[("annotations", dataset, "annotation", 0 if dataset == "R2R" else 1)]
        rgb_entries = [
            entry
            for entry in entries
            if entry.section == "rgb_archives" and entry.dataset == dataset
        ]
        if any(entry.revision != annotation.revision for entry in rgb_entries):
            raise ManifestValidationError(
                f"{dataset} annotation and RGB revisions must agree"
            )
    return asset_scope, tuple(entries)


def _missing_assets(entries: Sequence[AssetEntry]) -> list[str]:
    return sorted(entry.logical_name for entry in entries if not entry.path.is_file())


def _verify_manifest_identities(entries: Sequence[AssetEntry]) -> list[str]:
    errors: list[str] = []
    for entry in entries:
        actual_bytes = entry.path.stat().st_size
        if actual_bytes != entry.bytes:
            errors.append(f"{entry.logical_name}: bytes mismatch")
    return errors


def _assert_official_asset_manifest(entries: Sequence[AssetEntry]) -> None:
    by_identity = {
        (entry.section, entry.dataset, entry.role, entry.order): entry
        for entry in entries
    }
    for identity, expected in _OFFICIAL_ASSET_IDENTITY.items():
        entry = by_identity[identity]
        if (entry.bytes, entry.sha256) != expected:
            raise ManifestValidationError(
                f"{entry.logical_name} does not match frozen official asset identity"
            )
        if entry.section in ("annotations", "rgb_archives") and entry.revision != _OFFICIAL_STREAM_REVISION:
            raise ManifestValidationError(
                f"{entry.logical_name} does not match the frozen StreamVLN revision"
            )


def _parse_horizon_settings(payload: object) -> int:
    """Validate the one dense positive-integer horizon configuration."""

    if not isinstance(payload, Mapping):
        raise PipelineValidationError("config must be a mapping")
    horizon = payload.get("horizon")
    if not isinstance(horizon, Mapping):
        raise PipelineValidationError("config is missing horizon settings")
    max_steps = horizon.get("max_steps")
    if type(max_steps) is not int or max_steps < 1:
        raise PipelineValidationError(
            "horizon.max_steps must be a positive exact integer"
        )
    prefixes = horizon.get("prefixes")
    if type(prefixes) is not list:
        raise PipelineValidationError("horizon.prefixes must be a list")
    if any(type(value) is not int for value in prefixes):
        raise PipelineValidationError(
            "horizon.prefixes must contain exact integers"
        )
    expected = list(range(1, max_steps + 1))
    if prefixes != expected:
        raise PipelineValidationError(
            "horizon.prefixes must be the complete dense prefix 1..max_steps"
        )
    return max_steps


def _read_config_with_horizon(path: Path) -> tuple[bytes, int]:
    with path.open("rb") as handle:
        raw_bytes = handle.read()
    payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    return raw_bytes, _parse_horizon_settings(payload)


def _validate_config(path: Path) -> bytes:
    """Validate config bytes while retaining the historical bytes-only API."""

    raw_bytes, _ = _read_config_with_horizon(path)
    return raw_bytes


def _horizon_from_config_bytes(raw_bytes: object) -> int:
    """Recover H from bytes returned by the compatibility config API."""

    if not isinstance(raw_bytes, bytes):
        # Existing tests patch ``_validate_config`` with a no-op to isolate
        # earlier authority gates.  Preserve that seam as the H4 baseline.
        return _CENSUS_MAIN_HORIZON
    try:
        payload = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PipelineValidationError("config bytes are not valid YAML") from exc
    return _parse_horizon_settings(payload)


def _entry(
    entries: Sequence[AssetEntry],
    *,
    section: str,
    dataset: str,
    order: int,
) -> AssetEntry:
    matches = [
        entry
        for entry in entries
        if entry.section == section and entry.dataset == dataset and entry.order == order
    ]
    if len(matches) != 1:
        raise PipelineValidationError("validated manifest identity lookup failed")
    return matches[0]


def _load_annotations(
    entries: Sequence[AssetEntry],
) -> tuple[dict[str, tuple[AnnotationRow, ...]], dict[tuple[str, int], bytes]]:
    rows_by_source: dict[str, tuple[AnnotationRow, ...]] = {}
    keys: dict[tuple[str, int], bytes] = {}
    for source_id, order in (("R2R", 0), ("RxR", 1)):
        entry = _entry(entries, section="annotations", dataset=source_id, order=order)
        with entry.path.open("rb") as handle:
            annotation_bytes = handle.read()
        if (
            len(annotation_bytes) != entry.bytes
            or hashlib.sha256(annotation_bytes).hexdigest() != entry.sha256
        ):
            raise AssetIdentityError(
                f"{entry.logical_name}: consumed annotation bytes or SHA-256 mismatch"
            )
        raw_rows = _strict_json_bytes(annotation_bytes)
        if type(raw_rows) is not list:
            raise PipelineValidationError(f"{source_id} annotation root must be a JSON array")
        parsed: list[AnnotationRow] = []
        for row_index, raw in enumerate(raw_rows):
            row = parse_annotation_row(raw, source_id=source_id, row_index=row_index)
            parsed.append(row)
            keys[(source_id, row_index)] = trajectory_key(
                row, annotation_revision_sha256=entry.sha256
            )
        rows_by_source[source_id] = tuple(parsed)
    return rows_by_source, keys


def _member_prefix(member_path: str) -> str:
    try:
        return parse_frame_member(member_path).video_prefix
    except ArchiveValidationError as exc:
        raise PipelineValidationError("verified RGB member has an invalid released grammar") from exc


@dataclass(frozen=True)
class _CompactTrajectoryLayout:
    row: AnnotationRow
    trajectory_key: bytes
    frame_offset: int

    @property
    def frame_count(self) -> int:
        return len(self.row.actions)


@dataclass(frozen=True)
class _CompactCanonicalTrajectory:
    representative_index: int
    dedup_key: bytes
    alias_indices: tuple[int, ...]


class _OfficialFrameStore:
    """Trajectory-indexed raw32 frame hashes for the official archive path."""

    def __init__(
        self,
        rows_by_source: Mapping[str, tuple[AnnotationRow, ...]],
        keys: Mapping[tuple[str, int], bytes],
    ) -> None:
        layouts: list[_CompactTrajectoryLayout] = []
        by_prefix: dict[str, int] = {}
        frame_offset = 0
        for source_id in ("R2R", "RxR"):
            for row in rows_by_source[source_id]:
                # R2R uses three-digit members; RxR additionally has a
                # verified four-digit ``rgb_images`` layout.  The owner
                # parser/sequence check below remains the authority for the
                # exact per-trajectory width and frame range.
                if len(row.actions) > 9999:
                    raise PipelineValidationError(
                        "RGB trajectory exceeds the released frame grammar"
                    )
                if row.video_prefix in by_prefix:
                    raise PipelineValidationError(
                        "annotation video prefixes must be globally unique"
                    )
                index = len(layouts)
                by_prefix[row.video_prefix] = index
                layouts.append(
                    _CompactTrajectoryLayout(
                        row=row,
                        trajectory_key=keys[(source_id, row.row_index)],
                        frame_offset=frame_offset,
                    )
                )
                frame_offset += len(row.actions)

        self.layouts = layouts
        self._by_prefix = by_prefix
        self._compressed = bytearray(frame_offset * 32)
        self._decoded = bytearray(frame_offset * 32)
        self._seen = bytearray((frame_offset + 7) // 8)
        self._total_frames = frame_offset
        self._seen_frames = 0
        # ``layout`` is sequence-level; rgb_images frame width is an observed
        # per-member property because unpadded names cross digit boundaries.
        self._frame_layouts: dict[str, str] = {}
        # The official RxR archive carries six scene-only ``rgb_images``
        # prefixes that are not present in the published annotations.  Keep a
        # separate contiguous-frame ledger for those verified members rather
        # than mapping them to training trajectories.
        self._ignored_extra_masks: dict[str, int] = {}
        self._ignored_extra_layouts: dict[str, str] = {}
        self._ignored_extra_widths: dict[str, set[int]] = {}

    @staticmethod
    def _raw32(value: object, *, label: str) -> bytes:
        if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
            raise PipelineValidationError(f"{label} must be a 64-digit SHA-256")
        return bytes.fromhex(value)

    def consume(self, source_id: str, members: object) -> None:
        for member in members:
            try:
                identity = parse_frame_member(member.member_path)
            except ArchiveValidationError as exc:
                raise PipelineValidationError(
                    "verified RGB member has an invalid released grammar"
                ) from exc
            if identity.source_id != source_id:
                raise PipelineValidationError(
                    f"{source_id} RGB archive contains a cross-dataset member"
                )
            prefix = identity.video_prefix
            layout_index = self._by_prefix.get(prefix)
            if layout_index is None:
                if identity.source_id != "RxR" or identity.layout != "rgb_images":
                    raise PipelineValidationError(
                        "RGB archive contains an unmapped prefix outside the verified RxR rgb_images layout"
                    )
                current_layout = identity.layout
                previous_layout = self._ignored_extra_layouts.get(prefix)
                if previous_layout is not None and previous_layout != current_layout:
                    raise PipelineValidationError(
                        "unmapped RGB prefix mixes physical layouts"
                    )
                self._ignored_extra_layouts[prefix] = current_layout
                self._ignored_extra_widths.setdefault(prefix, set()).add(
                    identity.frame_width
                )
                bit = 1 << (identity.frame_index - 1)
                mask = self._ignored_extra_masks.get(prefix, 0)
                if mask & bit:
                    raise PipelineValidationError(
                        "duplicate unmapped RGB member across logical archives"
                    )
                # Validate the same hash fields as annotated members even
                # though these bytes are intentionally excluded from the
                # trajectory store.
                self._raw32(
                    member.compressed_sha256, label="compressed JPEG SHA-256"
                )
                self._raw32(member.decoded_rgb_sha256, label="decoded RGB SHA-256")
                self._ignored_extra_masks[prefix] = mask | bit
                continue
            layout = self.layouts[layout_index]
            if layout.row.source_id != source_id:
                raise PipelineValidationError(
                    "RGB member source disagrees with annotation ownership"
                )
            current_layout = identity.layout
            previous_layout = self._frame_layouts.get(prefix)
            if previous_layout is not None and previous_layout != current_layout:
                raise PipelineValidationError(
                    "RGB frame sequence mixes physical layouts"
                )
            self._frame_layouts[prefix] = current_layout
            frame_index = identity.frame_index
            if not 1 <= frame_index <= layout.frame_count:
                raise PipelineValidationError(
                    "RGB frame index falls outside its annotation trajectory"
                )
            slot = layout.frame_offset + frame_index - 1
            byte_index, bit_index = divmod(slot, 8)
            bit = 1 << bit_index
            if self._seen[byte_index] & bit:
                raise PipelineValidationError(
                    "duplicate RGB member across logical archives"
                )
            compressed = self._raw32(
                member.compressed_sha256, label="compressed JPEG SHA-256"
            )
            decoded = self._raw32(
                member.decoded_rgb_sha256, label="decoded RGB SHA-256"
            )
            raw_offset = slot * 32
            self._compressed[raw_offset : raw_offset + 32] = compressed
            self._decoded[raw_offset : raw_offset + 32] = decoded
            self._seen[byte_index] |= bit
            self._seen_frames += 1

    def finish(self) -> int:
        for prefix, mask in self._ignored_extra_masks.items():
            last_frame = mask.bit_length()
            expected = (1 << last_frame) - 1
            if mask != expected:
                raise PipelineValidationError(
                    f"unmapped RGB prefix has a non-contiguous frame sequence: {prefix}"
                )
        if self._seen_frames != self._total_frames:
            raise PipelineValidationError(
                "RGB archives do not contain exactly every annotated frame"
            )
        return self._seen_frames

    @property
    def ignored_extra_frame_count(self) -> int:
        """Number of verified scene-only frames excluded from annotations."""

        return sum(mask.bit_count() for mask in self._ignored_extra_masks.values())

    @property
    def ignored_extra_prefixes(self) -> tuple[str, ...]:
        """Sorted scene-only prefixes admitted by the narrow extra policy."""

        return tuple(sorted(self._ignored_extra_masks))

    def ignored_extra_summary(self) -> tuple[dict[str, object], ...]:
        """Return deterministic identity facts for excluded archive prefixes."""

        summary: list[dict[str, object]] = []
        for prefix in self.ignored_extra_prefixes:
            mask = self._ignored_extra_masks[prefix]
            summary.append(
                {
                    "video_prefix": prefix,
                    "source_id": "RxR",
                    "layout": self._ignored_extra_layouts[prefix],
                    "observed_frame_widths": sorted(
                        self._ignored_extra_widths[prefix]
                    ),
                    "frame_count": mask.bit_count(),
                    "first_frame": 1,
                    "last_frame": mask.bit_length(),
                }
            )
        return tuple(summary)

    def _raw_view(self, layout_index: int, *, decoded: bool) -> memoryview:
        layout = self.layouts[layout_index]
        start = layout.frame_offset * 32
        end = start + layout.frame_count * 32
        storage = self._decoded if decoded else self._compressed
        return memoryview(storage)[start:end]

    def compressed_view(self, layout_index: int) -> memoryview:
        return self._raw_view(layout_index, decoded=False)

    def decoded_view(self, layout_index: int) -> memoryview:
        return self._raw_view(layout_index, decoded=True)

    def compressed_hexes(self, layout_index: int) -> list[str]:
        raw = self.compressed_view(layout_index)
        return [raw[offset : offset + 32].hex() for offset in range(0, len(raw), 32)]

    def decoded_hexes(self, layout_index: int) -> list[str]:
        raw = self.decoded_view(layout_index)
        return [raw[offset : offset + 32].hex() for offset in range(0, len(raw), 32)]


def _compact_representative_key(
    store: _OfficialFrameStore, layout_index: int
) -> tuple[bytes, int, bytes, bytes]:
    layout = store.layouts[layout_index]
    return (
        layout.row.source_id.encode("utf-8"),
        layout.row.row_index,
        layout.row.video_prefix.encode("utf-8"),
        layout.trajectory_key,
    )


def _compact_dedup_key(store: _OfficialFrameStore, layout_index: int) -> bytes:
    layout = store.layouts[layout_index]
    action_payload = bytes(layout.row.actions[1:])
    digest = hashlib.sha256()
    digest.update(_DEDUP_DOMAIN)
    digest.update(struct.pack("<Q", layout.frame_count))
    digest.update(store.compressed_view(layout_index))
    digest.update(struct.pack("<Q", len(action_payload)))
    digest.update(action_payload)
    return digest.digest()


def _same_compact_preimage(
    store: _OfficialFrameStore, left: int, right: int
) -> bool:
    return (
        store.layouts[left].row.actions == store.layouts[right].row.actions
        and store.compressed_view(left) == store.compressed_view(right)
    )


def _canonicalize_compact_trajectories(
    store: _OfficialFrameStore,
) -> list[_CompactCanonicalTrajectory]:
    families: dict[bytes, list[int]] = {}
    for layout_index in range(len(store.layouts)):
        digest = _compact_dedup_key(store, layout_index)
        family = families.setdefault(digest, [])
        if family:
            representative = family[0]
            if not _same_compact_preimage(store, representative, layout_index):
                raise PipelineValidationError("distinct dedup preimages collided")
            if (
                store.layouts[representative].row.scan_id
                != store.layouts[layout_index].row.scan_id
            ):
                raise PipelineValidationError(
                    "a dedup family crosses scan identities"
                )
            if store.decoded_view(representative) != store.decoded_view(layout_index):
                raise PipelineValidationError(
                    "identical compressed trajectories decode to different RGB bytes"
                )
        family.append(layout_index)

    canonical: list[_CompactCanonicalTrajectory] = []
    for dedup_key, family in families.items():
        ordered = tuple(
            sorted(
                family,
                key=lambda index: _compact_representative_key(store, index),
            )
        )
        canonical.append(
            _CompactCanonicalTrajectory(
                representative_index=ordered[0],
                dedup_key=dedup_key,
                alias_indices=ordered,
            )
        )
    canonical.sort(
        key=lambda value: _compact_representative_key(
            store, value.representative_index
        )
    )
    return canonical


def _collect_members(
    members: Sequence[tuple[str, object]],
) -> tuple[dict[str, dict[str, VerifiedArchiveMember]], int]:
    grouped: dict[str, dict[str, VerifiedArchiveMember]] = {}
    decoded_count = 0
    for source_id, iterator in members:
        dataset_token = f"_{source_id.lower()}_"
        for member in iterator:
            prefix = _member_prefix(member.member_path)
            if dataset_token not in prefix:
                raise PipelineValidationError(
                    f"{source_id} RGB archive contains a cross-dataset member"
                )
            prefix_members = grouped.setdefault(prefix, {})
            if member.member_path in prefix_members:
                raise PipelineValidationError("duplicate RGB member across logical archives")
            prefix_members[member.member_path] = member
            decoded_count += 1
    return grouped, decoded_count


def _build_trajectory_records(
    rows_by_source: Mapping[str, tuple[AnnotationRow, ...]],
    keys: Mapping[tuple[str, int], bytes],
    members_by_prefix: Mapping[str, Mapping[str, VerifiedArchiveMember]],
) -> tuple[TrajectoryRecord, ...]:
    records: list[TrajectoryRecord] = []
    used_prefixes: set[str] = set()
    for source_id in ("R2R", "RxR"):
        for row in rows_by_source[source_id]:
            prefix_members = members_by_prefix.get(row.video_prefix, {})
            member_paths = tuple(prefix_members)
            ordered_paths = ordered_frame_member_paths(
                member_paths,
                video_prefix=row.video_prefix,
                expected_frame_count=len(row.actions),
            )
            ordered_members = tuple(prefix_members[path] for path in ordered_paths)
            records.append(
                TrajectoryRecord(
                    trajectory_key=keys[(source_id, row.row_index)],
                    source_id=source_id,
                    annotation_row_index=row.row_index,
                    canonical_video_prefix=row.video_prefix,
                    scan_id=row.scan_id,
                    actions=row.actions,
                    compressed_jpeg_sha256s=tuple(
                        member.compressed_sha256 for member in ordered_members
                    ),
                    decoded_rgb_sha256s=tuple(
                        member.decoded_rgb_sha256 for member in ordered_members
                    ),
                )
            )
            used_prefixes.add(row.video_prefix)
    extras = set(members_by_prefix) - used_prefixes
    if extras:
        raise PipelineValidationError("RGB archive contains prefixes absent from annotations")
    return tuple(records)


def _origins(key: bytes, terminal: int) -> tuple[GoalViewOrigin, ...]:
    return tuple(
        GoalViewOrigin(
            trajectory_key=key,
            t=t,
            terminal=terminal,
            pre_eviction_support=False,
        )
        for t in range(terminal)
    )


def _annotation_pre_audit(
    rows_by_source: Mapping[str, tuple[AnnotationRow, ...]],
    keys: Mapping[tuple[str, int], bytes],
    *,
    max_horizon: int = _CENSUS_MAIN_HORIZON,
) -> ProductionCensus:
    trajectories: list[CensusTrajectory] = []
    for source_id in ("R2R", "RxR"):
        for row in rows_by_source[source_id]:
            key = keys[(source_id, row.row_index)]
            terminal = len(row.actions) - 1
            trajectories.append(
                CensusTrajectory(
                    partition="all",
                    canonical_trajectory_key=key,
                    dedup_key=key,
                    scan_id=row.scan_id,
                    actions=row.actions,
                    dedup_alias_count=0,
                    zero_motion_terminal=row.actions == (-1,),
                    goal_view_origins=_origins(key, terminal),
                )
            )
    return census(
        trajectories,
        audit_kind="annotation_pre_audit",
        max_horizon=max_horizon,
    )


def _annotation_pre_audit_compact(
    rows_by_source: Mapping[str, tuple[AnnotationRow, ...]],
    keys: Mapping[tuple[str, int], bytes],
    *,
    max_horizon: int = _CENSUS_MAIN_HORIZON,
) -> ProductionCensus:
    accumulator = CompactCensusAccumulator(
        audit_kind="annotation_pre_audit", max_horizon=max_horizon
    )
    for source_id in ("R2R", "RxR"):
        for row in rows_by_source[source_id]:
            key = keys[(source_id, row.row_index)]
            accumulator.add_complete_trajectory(
                partition="all",
                canonical_trajectory_key=key,
                dedup_key=key,
                scan_id=row.scan_id,
                actions=row.actions,
                dedup_alias_count=0,
                zero_motion_terminal=row.actions == (-1,),
            )
    return accumulator.finish()


def _production_census(
    canonical: Sequence[CanonicalTrajectory],
    sets: BuildingSets,
    *,
    max_horizon: int = _CENSUS_MAIN_HORIZON,
) -> tuple[ProductionCensus, dict[bytes, str]]:
    trajectories: list[CensusTrajectory] = []
    partitions: dict[bytes, str] = {}
    for record in canonical:
        partition = assign_building_partition(record.scan_id, sets)
        partitions[record.canonical_trajectory_key] = partition
        if partition not in ("train", "dev"):
            continue
        terminal = len(record.actions) - 1
        trajectories.append(
            CensusTrajectory(
                partition=partition,
                canonical_trajectory_key=record.canonical_trajectory_key,
                dedup_key=record.dedup_key,
                scan_id=record.scan_id,
                actions=record.actions,
                dedup_alias_count=len(record.aliases) - 1,
                zero_motion_terminal=record.actions == (-1,),
                goal_view_origins=_origins(record.canonical_trajectory_key, terminal),
            )
        )
    return (
        census(
            trajectories,
            audit_kind="production_post_rgb",
            max_horizon=max_horizon,
        ),
        partitions,
    )


def _production_census_compact(
    canonical: Sequence[_CompactCanonicalTrajectory],
    store: _OfficialFrameStore,
    sets: BuildingSets,
    *,
    max_horizon: int = _CENSUS_MAIN_HORIZON,
) -> tuple[ProductionCensus, list[str]]:
    accumulator = CompactCensusAccumulator(
        audit_kind="production_post_rgb", max_horizon=max_horizon
    )
    partitions: list[str] = []
    for record in canonical:
        layout = store.layouts[record.representative_index]
        partition = assign_building_partition(layout.row.scan_id, sets)
        partitions.append(partition)
        if partition not in ("train", "dev"):
            continue
        accumulator.add_complete_trajectory(
            partition=partition,
            canonical_trajectory_key=layout.trajectory_key,
            dedup_key=record.dedup_key,
            scan_id=layout.row.scan_id,
            actions=layout.row.actions,
            dedup_alias_count=len(record.alias_indices) - 1,
            zero_motion_terminal=layout.row.actions == (-1,),
        )
    return accumulator.finish(), partitions


def _census_receipt(value: ProductionCensus) -> dict[str, object]:
    return {
        "kind": value.audit_kind,
        "max_horizon": value.max_horizon,
        "partitions": [asdict(partition) for partition in value.partitions],
    }


def _pointnav_receipt(identity: object) -> dict[str, object]:
    return {
        "split": identity.split,
        "dataset": identity.dataset,
        "content_basenames": sorted(identity.content_basenames),
        "explicit_content_scenes": sorted(identity.explicit_content_scenes),
        "episode_scene_ids": sorted(identity.episode_scene_ids),
        "scan_ids": sorted(identity.scan_ids),
        "member_ledger": [asdict(member) for member in identity.member_ledger],
    }


def _alias_ledger(canonical: Sequence[CanonicalTrajectory]) -> list[dict[str, object]]:
    return [
        {
            "canonical_trajectory_key": record.canonical_trajectory_key.hex(),
            "dedup_key": record.dedup_key.hex(),
            "aliases": [
                {
                    "source_id": alias.source_id,
                    "annotation_row_index": alias.annotation_row_index,
                    "canonical_video_prefix": alias.canonical_video_prefix,
                    "trajectory_key": alias.trajectory_key.hex(),
                    "scan_id": alias.scan_id,
                }
                for alias in record.aliases
            ],
        }
        for record in canonical
    ]


def _canonical_jsonl_summary(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    digest = hashlib.sha256()
    row_count = 0
    for row in rows:
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
        row_count += 1
    return {"row_count": row_count, "sha256": digest.hexdigest()}


def _canonical_mapping_sha256(value: Mapping[str, object]) -> str:
    """Hash one cache/ledger mapping with the existing compact-JSON rule."""

    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PipelineValidationError("mapping contains non-canonical JSON values") from exc
    return hashlib.sha256(encoded).hexdigest()


def _annotation_ledger_preimage(
    assets: Sequence[Mapping[str, object]], *, streamvln_revision: str
) -> dict[str, object]:
    """Build the bounded, deterministic annotation identity ledger.

    The ledger contains only identities already present in the consumed asset
    manifest; it never hashes a rewritten annotation subset.  Keeping the
    preimage in the outer manifest lets the later cache-parent derivation
    recompute and verify the digest instead of trusting a caller-provided SHA.
    """

    selected = [
        dict(asset)
        for asset in assets
        if isinstance(asset, Mapping) and asset.get("role") == "annotation"
    ]
    selected.sort(key=lambda asset: (str(asset.get("dataset", "")), int(asset.get("order", 0))))
    return {
        "schema": "J2J_ANNOTATION_LEDGER_V1",
        "streamvln_revision": streamvln_revision,
        "assets": selected,
    }


def _split_ledger_preimage(
    ledger: Sequence[Mapping[str, object]], *, streamvln_revision: str
) -> dict[str, object]:
    """Build the split/building identity preimage from the census owner."""

    return {
        "schema": "J2J_SPLIT_LEDGER_V1",
        "streamvln_revision": streamvln_revision,
        "ledger": [dict(row) for row in ledger],
    }


def _write_jsonl_rows(
    destination: Path,
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Write an ordered JSONL stream and return its exact byte identity.

    The caller owns the row iterator; this function never materializes the
    canonical trajectory or alias ledger.  JSON encoding is deliberately the
    same canonical encoding used by :func:`_canonical_jsonl_summary`, so the
    returned digest is directly usable as the ledger identity in a receipt.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for row in rows:
                encoded = (
                    json.dumps(
                        row,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                    + b"\n"
                )
                handle.write(encoded)
                digest.update(encoded)
                row_count += 1
                byte_count += len(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return {
        "relpath": destination.name,
        "row_count": row_count,
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
    }


def _write_partitioned_trajectory_rows(
    root: Path,
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Write full, eligible and excluded trajectory ledgers in one pass."""

    names = {
        "full": "J2J_CANONICAL_TRAJECTORIES_V1.jsonl",
        "eligible": "J2J_CANONICAL_TRAJECTORIES_ELIGIBLE_V1.jsonl",
        "excluded": "J2J_CANONICAL_TRAJECTORIES_EXCLUDED_V1.jsonl",
    }
    handles: dict[str, object] = {}
    descriptors: dict[str, dict[str, object]] = {}
    temporary: dict[str, Path] = {}
    try:
        for role, name in names.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{name}.", suffix=".tmp", dir=root
            )
            handles[role] = os.fdopen(descriptor, "wb")
            temporary[role] = Path(temporary_name)
        digests = {role: hashlib.sha256() for role in names}
        counts = {role: 0 for role in names}
        bytes_written = {role: 0 for role in names}
        for row in rows:
            partition = row.get("partition")
            if type(partition) is not str:
                raise PipelineValidationError(
                    "canonical trajectory row partition must be a string"
                )
            target = "eligible" if partition in {"train", "dev"} else "excluded"
            encoded = (
                json.dumps(
                    row,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
                + b"\n"
            )
            for role in ("full", target):
                handle = handles[role]
                assert hasattr(handle, "write")
                handle.write(encoded)  # type: ignore[union-attr]
                digests[role].update(encoded)
                counts[role] += 1
                bytes_written[role] += len(encoded)
        for role, handle in handles.items():
            assert hasattr(handle, "flush")
            handle.flush()  # type: ignore[union-attr]
            assert hasattr(handle, "fileno")
            os.fsync(handle.fileno())  # type: ignore[union-attr]
            handle.close()  # type: ignore[union-attr]
            destination = root / names[role]
            os.replace(temporary[role], destination)
            descriptors[role] = {
                "relpath": destination.name,
                "row_count": counts[role],
                "bytes": bytes_written[role],
                "sha256": digests[role].hexdigest(),
            }
    except Exception:
        for handle in handles.values():
            try:
                handle.close()  # type: ignore[union-attr]
            except Exception:
                pass
        for temporary_name in temporary.values():
            try:
                temporary_name.unlink()
            except OSError:
                pass
        raise
    return descriptors


def write_canonical_manifests(
    output_root: Path | os.PathLike[str],
    *,
    trajectory_rows: Iterable[Mapping[str, object]],
    alias_rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Persist the complete canonical trajectory and alias JSONL products.

    This is a thin output seam over the already-canonicalized row iterators;
    it does not parse annotations, inspect RGB, alter ordering, or create
    records.  The output root is explicit so server runs can place the full
    product outside the bounded receipt directory while local tests may use a
    temporary directory.
    """

    root = Path(output_root)
    if not root.is_dir():
        raise PipelineValidationError(
            "canonical manifest output root must already be an existing directory"
        )
    trajectory_ledgers = _write_partitioned_trajectory_rows(root, trajectory_rows)
    trajectory = trajectory_ledgers["full"]
    trajectory["projection_ledgers"] = {
        "eligible": trajectory_ledgers["eligible"],
        "excluded": trajectory_ledgers["excluded"],
    }
    alias = _write_jsonl_rows(
        root / "J2J_CANONICAL_ALIASES_V1.jsonl", alias_rows
    )
    return {
        "schema": "J2J_CANONICAL_MANIFEST_V1",
        "trajectory": trajectory,
        "alias": alias,
    }


def _canonical_manifest_descriptor(
    root: Path,
    descriptor: object,
    *,
    label: str,
    require_nonempty: bool,
    require_raw_trajectory_order: bool = False,
) -> dict[str, object]:
    """Revalidate one writer-produced JSONL descriptor without RGB replay.

    This is deliberately a byte/JSONL check only.  It does not parse an
    archive or decode JPEGs; all frame identities must already be present in
    the full ledger emitted by :func:`write_canonical_manifests`.
    """

    if not isinstance(descriptor, Mapping):
        raise PipelineValidationError(f"{label} descriptor is missing")
    relpath = _canonical_relative_path(descriptor.get("relpath"), label=f"{label} relpath")
    row_count = descriptor.get("row_count")
    byte_count = descriptor.get("bytes")
    declared_sha = descriptor.get("sha256")
    if type(row_count) is not int or row_count < 0:
        raise PipelineValidationError(f"{label} row_count must be a non-negative integer")
    if type(byte_count) is not int or byte_count < 0:
        raise PipelineValidationError(f"{label} bytes must be a non-negative integer")
    if type(declared_sha) is not str or _SHA256_HEX.fullmatch(declared_sha) is None:
        raise PipelineValidationError(f"{label} sha256 must be a 64-digit SHA-256")
    if require_nonempty and row_count == 0:
        raise PipelineValidationError(f"{label} cannot be empty")

    # Resolve through the existing canonical relative-path gate and reject
    # symlink escapes as well.  The latter matters when a receipt is resumed
    # on a shared filesystem where an artifact directory may be mutable.
    root_resolved = root.resolve()
    path = root / relpath
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root_resolved)
    except (OSError, ValueError) as exc:
        raise PipelineValidationError(f"{label} path is missing or escapes its root") from exc
    if not resolved.is_file():
        raise PipelineValidationError(f"{label} path is not a regular file")

    digest = hashlib.sha256()
    actual_bytes = 0
    actual_rows = 0
    previous_trajectory_key: bytes | None = None
    try:
        with resolved.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    raise PipelineValidationError(f"{label} row is missing a final LF")
                payload = line[:-1]
                try:
                    value = _strict_json_bytes(payload)
                    if not isinstance(value, Mapping):
                        raise ValueError("row is not a JSON object")
                    if require_raw_trajectory_order:
                        key_value = value.get("canonical_trajectory_key")
                        if (
                            type(key_value) is not str
                            or re.fullmatch(r"[0-9a-fA-F]{64}", key_value) is None
                        ):
                            raise ValueError("trajectory row has an invalid canonical key")
                        key = bytes.fromhex(key_value)
                        if previous_trajectory_key is not None and key <= previous_trajectory_key:
                            raise ValueError("trajectory rows are not strictly raw-key sorted")
                        previous_trajectory_key = key
                    canonical = json.dumps(
                        value,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                        allow_nan=False,
                    ).encode("utf-8")
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise PipelineValidationError(f"{label} contains non-canonical JSONL") from exc
                if canonical != payload:
                    raise PipelineValidationError(f"{label} row is not canonical JSON")
                digest.update(line)
                actual_bytes += len(line)
                actual_rows += 1
    except OSError as exc:
        raise PipelineValidationError(f"{label} path is unreadable") from exc

    if actual_rows != row_count:
        raise PipelineValidationError(f"{label} row_count does not match bytes")
    if actual_bytes != byte_count:
        raise PipelineValidationError(f"{label} byte count does not match file")
    actual_sha = digest.hexdigest()
    if actual_sha != declared_sha.lower():
        raise PipelineValidationError(f"{label} SHA-256 does not match file")
    # Normalize the descriptor to the exact fields consumed by source.py;
    # preserving arbitrary receipt keys here would make the resumed artifact
    # depend on unvalidated caller metadata.
    return {
        "relpath": relpath,
        "row_count": row_count,
        "bytes": byte_count,
        "sha256": actual_sha,
    }


def _atomic_write_bytes(destination: Path, encoded: bytes, *, prefix: str) -> None:
    """Atomically publish one explicitly named artifact beside its inputs."""

    if not destination.parent.is_dir():
        raise PipelineValidationError("artifact output parent must already be an existing directory")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix, suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


class _VerifiedPayloadSpoolWriter:
    """Persist exact verified JPEG members while a census consumes one stream.

    The spool is an explicitly opt-in transport bridge.  It records the
    original tar-member bytes and a canonical metadata JSONL index; no bytes
    are decoded or re-encoded here.  ``wrap`` yields the same payload object
    after writing it, so a caller can feed the existing ``_OfficialFrameStore``
    without opening a second archive iterator.
    """

    _SCHEMA = "J2J_VERIFIED_RGB_PAYLOAD_SPOOL_V1"
    _SPOOL_NAME = "J2J_VERIFIED_RGB_PAYLOADS_V1.spool"
    _INDEX_NAME = "J2J_VERIFIED_RGB_PAYLOAD_INDEX_V1.jsonl"
    _RECEIPT_NAME = "J2J_VERIFIED_RGB_PAYLOAD_SPOOL_V1.json"

    def __init__(self, output_root: Path | os.PathLike[str]) -> None:
        root = Path(output_root)
        if not root.is_dir():
            raise PipelineValidationError(
                "payload spool root must already be an existing directory"
            )
        self.root = root
        self._spool_path = root / self._SPOOL_NAME
        self._index_path = root / self._INDEX_NAME
        self._receipt_path = root / self._RECEIPT_NAME
        if any(path.exists() for path in (self._spool_path, self._index_path, self._receipt_path)):
            raise PipelineValidationError(
                "payload spool output already exists; choose a fresh directory"
            )
        spool_fd, spool_tmp = tempfile.mkstemp(
            prefix=f".{self._SPOOL_NAME}.", suffix=".tmp", dir=root
        )
        index_fd, index_tmp = tempfile.mkstemp(
            prefix=f".{self._INDEX_NAME}.", suffix=".tmp", dir=root
        )
        self._spool_tmp = Path(spool_tmp)
        self._index_tmp = Path(index_tmp)
        self._spool = os.fdopen(spool_fd, "wb")
        self._index = os.fdopen(index_fd, "wb")
        self._spool_digest = hashlib.sha256()
        self._index_digest = hashlib.sha256()
        self._spool_bytes = 0
        self._index_bytes = 0
        self._row_count = 0
        self._seen_members: set[str] = set()
        self._closed = False
        self._published = False

    @staticmethod
    def _jsonl_bytes(value: Mapping[str, object]) -> bytes:
        try:
            return (
                json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise PipelineValidationError("payload spool index row is not canonical JSON") from exc

    def wrap(
        self,
        source_id: str,
        payloads: Iterable[VerifiedArchivePayload],
    ) -> Iterable[VerifiedArchivePayload]:
        """Write and yield one source's verified payload stream in order."""

        if source_id not in {"R2R", "RxR"}:
            raise PipelineValidationError("payload spool source_id is invalid")
        if self._closed:
            raise PipelineValidationError("payload spool writer is already closed")
        for payload in payloads:
            if not isinstance(payload, VerifiedArchivePayload):
                raise PipelineValidationError(
                    "payload spool requires VerifiedArchivePayload values"
                )
            member_path = payload.member_path
            if type(member_path) is not str or not member_path:
                raise PipelineValidationError("payload spool member path is invalid")
            if member_path in self._seen_members:
                raise PipelineValidationError(
                    f"duplicate payload spool member: {member_path}"
                )
            self._seen_members.add(member_path)
            raw = payload.jpeg_bytes
            if hashlib.sha256(raw).hexdigest() != payload.compressed_sha256:
                raise PipelineValidationError("payload spool compressed identity mismatch")
            offset = self._spool.tell()
            header = struct.pack("<Q", len(raw))
            self._spool.write(header)
            self._spool.write(raw)
            self._spool_digest.update(header)
            self._spool_digest.update(raw)
            self._spool_bytes += len(header) + len(raw)
            row = {
                "source_id": source_id,
                "member_path": member_path,
                "compressed_jpeg_sha256": payload.compressed_sha256,
                "decoded_rgb_sha256": payload.decoded_rgb_sha256,
                "offset": offset,
                "length": len(raw),
                "width": payload.width,
                "height": payload.height,
            }
            encoded = self._jsonl_bytes(row)
            self._index.write(encoded)
            self._index_digest.update(encoded)
            self._index_bytes += len(encoded)
            self._row_count += 1
            yield payload

    def _close_handles(self) -> None:
        if self._closed:
            return
        try:
            self._spool.flush()
            os.fsync(self._spool.fileno())
        finally:
            try:
                self._spool.close()
            finally:
                self._index.flush()
                os.fsync(self._index.fileno())
                self._index.close()
                self._closed = True

    @staticmethod
    def _descriptor(
        path: Path,
        *,
        row_count: int | None = None,
        byte_count: int | None = None,
        sha256: str | None = None,
    ) -> dict[str, object]:
        if byte_count is None or sha256 is None:
            try:
                byte_count = path.stat().st_size
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while chunk := handle.read(1024 * 1024):
                        digest.update(chunk)
                sha256 = digest.hexdigest()
            except OSError as exc:
                raise PipelineValidationError("payload spool artifact is unreadable") from exc
        result: dict[str, object] = {
            "relpath": path.name,
            "bytes": byte_count,
            "sha256": sha256,
        }
        if row_count is not None:
            result["row_count"] = row_count
        return result

    def finalize(self) -> dict[str, object]:
        """Publish spool, index, and a small identity receipt atomically."""

        if self._published:
            raise PipelineValidationError("payload spool has already been finalized")
        self._close_handles()
        try:
            os.replace(self._spool_tmp, self._spool_path)
            os.replace(self._index_tmp, self._index_path)
            sidecar: dict[str, object] = {
                "schema": self._SCHEMA,
                "format": "u64_le_length_then_exact_jpeg_bytes",
                "row_count": self._row_count,
                "spool": self._descriptor(
                    self._spool_path,
                    byte_count=self._spool_bytes,
                    sha256=self._spool_digest.hexdigest(),
                ),
                "index": self._descriptor(
                    self._index_path,
                    row_count=self._row_count,
                    byte_count=self._index_bytes,
                    sha256=self._index_digest.hexdigest(),
                ),
            }
            encoded = (
                json.dumps(
                    sidecar,
                    sort_keys=True,
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("utf-8")
                + b"\n"
            )
            _atomic_write_bytes(
                self._receipt_path,
                encoded,
                prefix=".j2j-payload-spool-receipt-",
            )
        except Exception:
            self.abort()
            raise
        self._published = True
        return sidecar

    def abort(self) -> None:
        """Remove only this writer's unpublished temporary/published files."""

        if not self._closed:
            try:
                self._spool.close()
            finally:
                self._index.close()
            self._closed = True
        for path in (
            self._spool_tmp,
            self._index_tmp,
            self._spool_path if not self._published else None,
            self._index_path if not self._published else None,
            self._receipt_path if not self._published else None,
        ):
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass


def write_canonical_outer_manifest_from_receipt(
    receipt_path: Path | os.PathLike[str],
    *,
    canonical_root: Path | os.PathLike[str] | None = None,
    output_path: Path | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Resume the source seam from an already completed full census output.

    A bounded census receipt cannot be upgraded into canonical rows: this
    helper therefore succeeds only when the receipt already references the
    complete JSONL products and each referenced file still matches its exact
    bytes/count/SHA.  No tar stream, JPEG parser, or model is touched.
    """

    receipt = Path(receipt_path)
    if not receipt.is_file():
        raise PipelineValidationError("census receipt is missing")
    if receipt.name != "J2J_BUILD_CENSUS_RECEIPT_V1.json":
        raise PipelineValidationError(
            "outer manifest resume requires a J2J_BUILD_CENSUS_RECEIPT_V1.json receipt"
        )
    try:
        captured = _strict_json_file(receipt)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PipelineValidationError("census receipt is not strict JSON") from exc
    payload = captured.value
    if not isinstance(payload, Mapping) or payload.get("schema") != "J2J_BUILD_CENSUS_RECEIPT_V1":
        raise PipelineValidationError("census receipt schema mismatch")
    if payload.get("production_eligible") is not True:
        raise PipelineValidationError("outer manifest resume requires production-eligible census")
    if type(payload.get("max_horizon")) is not int or payload["max_horizon"] <= 0:
        raise PipelineValidationError("census receipt max_horizon must be a positive integer")
    input_identity = payload.get("input_identity")
    if not isinstance(input_identity, Mapping):
        raise PipelineValidationError("census receipt input_identity is missing")

    raw_manifest = payload.get("canonical_manifest")
    if not isinstance(raw_manifest, Mapping) or raw_manifest.get("schema") != "J2J_CANONICAL_MANIFEST_V1":
        raise PipelineValidationError(
            "census receipt has no complete canonical manifest; rerun census with --canonical-manifest-root"
        )
    root = Path(canonical_root) if canonical_root is not None else receipt.parent
    if not root.is_dir():
        raise PipelineValidationError("canonical manifest root must already be an existing directory")
    root = root.resolve()
    destination = (
        Path(output_path)
        if output_path is not None
        else root / "J2J_CANONICAL_SOURCE_MANIFEST_V1.json"
    )
    destination = destination.resolve()
    try:
        destination.parent.relative_to(root)
    except ValueError as exc:
        raise PipelineValidationError("outer manifest output must stay below canonical root") from exc
    if destination.parent != root:
        raise PipelineValidationError("outer manifest output must be directly inside canonical root")

    trajectory = _canonical_manifest_descriptor(
        root,
        raw_manifest.get("trajectory"),
        label="trajectory ledger",
        require_nonempty=True,
        require_raw_trajectory_order=True,
    )
    raw_projections = raw_manifest.get("trajectory")
    if not isinstance(raw_projections, Mapping):  # guarded by descriptor above
        raise PipelineValidationError("trajectory descriptor is invalid")
    projections = raw_projections.get("projection_ledgers")
    if not isinstance(projections, Mapping):
        raise PipelineValidationError("canonical manifest projection ledgers are missing")
    trajectory["projection_ledgers"] = {
        "eligible": _canonical_manifest_descriptor(
            root,
            projections.get("eligible"),
            label="eligible trajectory ledger",
            require_nonempty=True,
            require_raw_trajectory_order=True,
        ),
        "excluded": _canonical_manifest_descriptor(
            root,
            projections.get("excluded"),
            label="excluded trajectory ledger",
            require_nonempty=False,
            require_raw_trajectory_order=True,
        ),
    }
    alias = _canonical_manifest_descriptor(
        root,
        raw_manifest.get("alias"),
        label="alias ledger",
        require_nonempty=True,
    )

    annotation_sha256s: dict[str, str] = {}
    assets = input_identity.get("assets")
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise PipelineValidationError("census input_identity.assets is missing")
    for asset in assets:
        if not isinstance(asset, Mapping) or asset.get("role") != "annotation":
            continue
        dataset = asset.get("dataset")
        sha = asset.get("sha256")
        if dataset not in {"R2R", "RxR"} or type(sha) is not str or _SHA256_HEX.fullmatch(sha) is None:
            raise PipelineValidationError("annotation asset identity is malformed")
        normalized = sha.lower()
        previous = annotation_sha256s.setdefault(str(dataset), normalized)
        if previous != normalized:
            raise PipelineValidationError("annotation asset identity is duplicated with different SHA")
    if set(annotation_sha256s) != {"R2R", "RxR"}:
        raise PipelineValidationError("both R2R and RxR annotation identities are required")

    streamvln_revision = payload.get("streamvln_revision", _OFFICIAL_STREAM_REVISION)
    if type(streamvln_revision) is not str or not streamvln_revision:
        raise PipelineValidationError("census StreamVLN revision is missing")
    annotation_assets = [
        asset
        for asset in assets
        if isinstance(asset, Mapping) and asset.get("role") == "annotation"
    ]
    annotation_preimage = _annotation_ledger_preimage(
        annotation_assets, streamvln_revision=streamvln_revision
    )
    annotation_ledger_sha256 = _canonical_mapping_sha256(annotation_preimage)

    # ``building_sets.ledger`` is emitted by the same census/split owner.  A
    # legacy fixture may omit it (the bounded resume test only exercises the
    # old source-manifest shape); such an artifact can still be inspected by
    # source.py, but cannot later pass formal parent derivation.
    building_sets = payload.get("building_sets")
    split_ledger: list[Mapping[str, object]] | None = None
    if isinstance(building_sets, Mapping) and isinstance(
        building_sets.get("ledger"), list
    ):
        split_ledger = [
            row for row in building_sets["ledger"] if isinstance(row, Mapping)
        ]
        if len(split_ledger) != len(building_sets["ledger"]):
            raise PipelineValidationError("census split ledger contains a non-mapping row")
    split_preimage = _split_ledger_preimage(
        split_ledger or [], streamvln_revision=streamvln_revision
    )
    split_ledger_sha256 = _canonical_mapping_sha256(split_preimage)

    outer: dict[str, object] = {
        "schema": "J2J_CANONICAL_SOURCE_MANIFEST_V1",
        "asset_scope": payload.get("asset_scope"),
        "audit_kind": "production_post_rgb",
        "production_eligible": True,
        "population_scope": "official_streamvln_full",
        "streamvln_revision": streamvln_revision,
        "max_horizon": payload["max_horizon"],
        "input_identity": dict(input_identity),
        "annotation_sha256s": dict(annotation_sha256s),
        "annotation_ledger": annotation_preimage,
        "annotation_ledger_sha256": annotation_ledger_sha256,
        "split_ledger_sha256": split_ledger_sha256,
        "trajectory": trajectory,
        "alias": alias,
        "trajectory_alias_ledger_sha256": alias["sha256"],
        "canonical_trajectory_ledger_sha256": trajectory["projection_ledgers"]["eligible"]["sha256"],
        "census_receipt_sha256": hashlib.sha256(captured.raw_bytes).hexdigest(),
    }
    if split_ledger is not None:
        outer["split_ledger"] = split_preimage
    try:
        encoded = (
            json.dumps(
                outer,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PipelineValidationError("outer manifest contains non-canonical JSON values") from exc
    _atomic_write_bytes(destination, encoded, prefix=".j2j-source-manifest-")
    return {
        "schema": outer["schema"],
        "relpath": destination.name,
        "path": str(destination),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "census_receipt_sha256": outer["census_receipt_sha256"],
    }


def _load_outer_manifest_for_parent(
    canonical_manifest: Path | os.PathLike[str] | Mapping[str, object],
    *,
    expected_manifest_sha256: str | None,
) -> tuple[Mapping[str, object], Path | None]:
    """Load an outer manifest without introducing a second manifest parser."""

    if isinstance(canonical_manifest, Mapping):
        value: object = canonical_manifest
        root: Path | None = None
        if expected_manifest_sha256 is not None:
            expected = expected_manifest_sha256.lower()
            if not _SHA256_HEX.fullmatch(expected):
                raise PipelineValidationError("expected canonical manifest SHA-256 is invalid")
            if _canonical_mapping_sha256(dict(canonical_manifest)) != expected:
                raise PipelineValidationError("canonical manifest mapping SHA-256 mismatch")
    else:
        path = Path(canonical_manifest)
        if not path.is_file():
            raise PipelineValidationError("canonical source manifest is missing")
        try:
            captured = _strict_json_file(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise PipelineValidationError("canonical source manifest is not strict JSON") from exc
        value = captured.value
        root = path.parent.resolve()
        actual = hashlib.sha256(captured.raw_bytes).hexdigest()
        if expected_manifest_sha256 is not None:
            expected = expected_manifest_sha256.lower()
            if _SHA256_HEX.fullmatch(expected) is None or actual != expected:
                raise PipelineValidationError("canonical source manifest SHA-256 mismatch")
    if not isinstance(value, Mapping):
        raise PipelineValidationError("canonical source manifest must be a mapping")
    return value, root


def _validate_manifest_descriptor_for_parent(
    root: Path | None,
    descriptor: object,
    *,
    label: str,
    require_nonempty: bool,
) -> dict[str, object]:
    """Validate descriptor shape and, for a file manifest, exact file bytes."""

    if root is not None:
        return _canonical_manifest_descriptor(
            root, descriptor, label=label, require_nonempty=require_nonempty
        )
    if not isinstance(descriptor, Mapping):
        raise PipelineValidationError(f"{label} descriptor is missing")
    relpath = descriptor.get("relpath")
    row_count = descriptor.get("row_count")
    byte_count = descriptor.get("bytes")
    declared_sha = descriptor.get("sha256")
    if type(relpath) is not str or not relpath:
        raise PipelineValidationError(f"{label} relpath is invalid")
    if type(row_count) is not int or row_count < 0:
        raise PipelineValidationError(f"{label} row_count is invalid")
    if require_nonempty and row_count == 0:
        raise PipelineValidationError(f"{label} cannot be empty")
    if type(byte_count) is not int or byte_count < 0:
        raise PipelineValidationError(f"{label} bytes is invalid")
    if type(declared_sha) is not str or _SHA256_HEX.fullmatch(declared_sha) is None:
        raise PipelineValidationError(f"{label} sha256 is invalid")
    return {
        "relpath": relpath,
        "row_count": row_count,
        "bytes": byte_count,
        "sha256": declared_sha.lower(),
    }


def _require_manifest_sha(value: object, label: str) -> str:
    if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
        raise PipelineValidationError(f"{label} must be a SHA-256 string")
    return value.lower()


def task4_cache_parent_sha256(parent_manifest: Mapping[str, object]) -> str:
    """Return the exact parent identity used by the cache writer."""

    if not isinstance(parent_manifest, Mapping):
        raise PipelineValidationError("Task 4 parent must be a mapping")
    return _canonical_mapping_sha256(dict(parent_manifest))


def derive_task4_cache_parent(
    canonical_manifest: Path | os.PathLike[str] | Mapping[str, object],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Derive the unique production U-cache parent from eligible rows.

    The caller supplies only the already-admitted outer manifest.  All parent
    leaves are checked against the outer's verified input/ledger preimages;
    in particular, the full or excluded trajectory digest can never be
    substituted for the eligible projection digest.
    """

    outer, root = _load_outer_manifest_for_parent(
        canonical_manifest, expected_manifest_sha256=expected_manifest_sha256
    )
    if outer.get("schema") != "J2J_CANONICAL_SOURCE_MANIFEST_V1":
        raise PipelineValidationError("Task 4 parent derivation requires the canonical source manifest")
    if outer.get("asset_scope") != "official" or outer.get("production_eligible") is not True:
        raise PipelineValidationError("Task 4 parent derivation requires official production authority")
    if outer.get("audit_kind") != "production_post_rgb":
        raise PipelineValidationError("Task 4 parent derivation requires production_post_rgb audit")
    revision = outer.get("streamvln_revision")
    if revision != _OFFICIAL_STREAM_REVISION:
        raise PipelineValidationError("Task 4 parent StreamVLN revision is not the frozen official revision")

    input_identity = outer.get("input_identity")
    if not isinstance(input_identity, Mapping):
        raise PipelineValidationError("canonical source manifest input_identity is missing")
    asset_manifest_sha256 = _require_manifest_sha(
        input_identity.get("asset_manifest_sha256"), "asset_manifest_sha256"
    )
    assets = input_identity.get("assets")
    if not isinstance(assets, list):
        raise PipelineValidationError("canonical source manifest input_identity.assets is missing")
    annotation_assets = [
        asset for asset in assets if isinstance(asset, Mapping) and asset.get("role") == "annotation"
    ]
    if len(annotation_assets) != 2:
        raise PipelineValidationError("canonical source manifest must bind exactly two annotation assets")
    annotation_by_dataset: dict[str, Mapping[str, object]] = {}
    for asset in annotation_assets:
        dataset = asset.get("dataset")
        if dataset not in {"R2R", "RxR"} or dataset in annotation_by_dataset:
            raise PipelineValidationError("annotation asset dataset identity is invalid")
        if asset.get("revision") != _OFFICIAL_STREAM_REVISION:
            raise PipelineValidationError("annotation asset revision is not the frozen official revision")
        annotation_by_dataset[str(dataset)] = asset
    if set(annotation_by_dataset) != {"R2R", "RxR"}:
        raise PipelineValidationError("R2R and RxR annotation assets are both required")

    annotation_sha256s = outer.get("annotation_sha256s")
    if not isinstance(annotation_sha256s, Mapping) or set(annotation_sha256s) != {"R2R", "RxR"}:
        raise PipelineValidationError("per-source annotation SHA ledger is missing")
    for dataset, asset in annotation_by_dataset.items():
        if _require_manifest_sha(annotation_sha256s.get(dataset), f"annotation_sha256s[{dataset}]") != _require_manifest_sha(
            asset.get("sha256"), f"annotation asset {dataset} SHA"
        ):
            raise PipelineValidationError("per-source annotation SHA does not match input identity")

    annotation_preimage = outer.get("annotation_ledger")
    if not isinstance(annotation_preimage, Mapping):
        raise PipelineValidationError("annotation ledger preimage is missing")
    if annotation_preimage.get("schema") != "J2J_ANNOTATION_LEDGER_V1":
        raise PipelineValidationError("annotation ledger schema mismatch")
    if annotation_preimage.get("streamvln_revision") != _OFFICIAL_STREAM_REVISION:
        raise PipelineValidationError("annotation ledger revision mismatch")
    preimage_assets = annotation_preimage.get("assets")
    if not isinstance(preimage_assets, list) or [dict(x) for x in preimage_assets if isinstance(x, Mapping)] != [dict(x) for x in preimage_assets]:
        raise PipelineValidationError("annotation ledger preimage is malformed")
    expected_annotation_preimage = _annotation_ledger_preimage(
        annotation_assets, streamvln_revision=_OFFICIAL_STREAM_REVISION
    )
    if dict(annotation_preimage) != expected_annotation_preimage:
        raise PipelineValidationError("annotation ledger preimage does not match input identity")
    annotation_ledger_sha256 = _require_manifest_sha(
        outer.get("annotation_ledger_sha256"), "annotation_ledger_sha256"
    )
    if annotation_ledger_sha256 != _canonical_mapping_sha256(dict(annotation_preimage)):
        raise PipelineValidationError("annotation ledger SHA does not match its preimage")

    trajectory_raw = outer.get("trajectory")
    if not isinstance(trajectory_raw, Mapping):
        raise PipelineValidationError("canonical trajectory descriptor is missing")
    trajectory = _validate_manifest_descriptor_for_parent(
        root, trajectory_raw, label="trajectory ledger", require_nonempty=True
    )
    projections = trajectory_raw.get("projection_ledgers")
    if not isinstance(projections, Mapping):
        raise PipelineValidationError("trajectory projection ledgers are missing")
    eligible = _validate_manifest_descriptor_for_parent(
        root, projections.get("eligible"), label="eligible trajectory ledger", require_nonempty=True
    )
    excluded = _validate_manifest_descriptor_for_parent(
        root, projections.get("excluded"), label="excluded trajectory ledger", require_nonempty=False
    )
    if trajectory["row_count"] != eligible["row_count"] + excluded["row_count"]:
        raise PipelineValidationError("full/eligible/excluded trajectory counts disagree")
    canonical_ledger_sha256 = _require_manifest_sha(
        outer.get("canonical_trajectory_ledger_sha256"),
        "canonical_trajectory_ledger_sha256",
    )
    if canonical_ledger_sha256 != eligible["sha256"]:
        raise PipelineValidationError("Task 4 parent must use the eligible trajectory ledger SHA")

    alias = _validate_manifest_descriptor_for_parent(
        root, outer.get("alias"), label="alias ledger", require_nonempty=True
    )
    if alias["row_count"] != trajectory["row_count"]:
        raise PipelineValidationError("alias and full trajectory counts disagree")
    alias_sha256 = _require_manifest_sha(
        outer.get("trajectory_alias_ledger_sha256"), "trajectory_alias_ledger_sha256"
    )
    if alias_sha256 != alias["sha256"]:
        raise PipelineValidationError("alias ledger SHA does not match its descriptor")

    split_preimage = outer.get("split_ledger")
    if not isinstance(split_preimage, Mapping):
        raise PipelineValidationError("split ledger preimage is missing")
    if split_preimage.get("schema") != "J2J_SPLIT_LEDGER_V1" or split_preimage.get("streamvln_revision") != _OFFICIAL_STREAM_REVISION:
        raise PipelineValidationError("split ledger preimage schema or revision mismatch")
    split_rows = split_preimage.get("ledger")
    if not isinstance(split_rows, list) or any(not isinstance(row, Mapping) for row in split_rows):
        raise PipelineValidationError("split ledger preimage is malformed")
    split_ledger_sha256 = _require_manifest_sha(
        outer.get("split_ledger_sha256"), "split_ledger_sha256"
    )
    if split_ledger_sha256 != _canonical_mapping_sha256(dict(split_preimage)):
        raise PipelineValidationError("split ledger SHA does not match its preimage")

    return {
        "schema": "J2J_TASK4_CACHE_PARENT_V1",
        "asset_scope": "official",
        "audit_kind": "production_post_rgb",
        "production_eligible": True,
        "streamvln_revision": _OFFICIAL_STREAM_REVISION,
        "asset_manifest_sha256": asset_manifest_sha256,
        "annotation_ledger_sha256": annotation_ledger_sha256,
        "split_ledger_sha256": split_ledger_sha256,
        "canonical_trajectory_ledger_sha256": canonical_ledger_sha256,
    }


def derive_task4_cache_parent_and_alias(
    canonical_manifest: Path | os.PathLike[str] | Mapping[str, object],
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, object], str]:
    """Return the production U parent and the already-bound alias ledger SHA."""

    parent = derive_task4_cache_parent(
        canonical_manifest, expected_manifest_sha256=expected_manifest_sha256
    )
    outer, _ = _load_outer_manifest_for_parent(
        canonical_manifest, expected_manifest_sha256=expected_manifest_sha256
    )
    alias_sha = _require_manifest_sha(
        outer.get("trajectory_alias_ledger_sha256"),
        "trajectory_alias_ledger_sha256",
    )
    return parent, alias_sha


def write_canonical_manifest(
    output_root: Path | os.PathLike[str],
    *,
    trajectory_rows: Iterable[Mapping[str, object]],
    alias_rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    """Singular public spelling retained for the source/catalog bridge ABI."""

    return write_canonical_manifests(
        output_root,
        trajectory_rows=trajectory_rows,
        alias_rows=alias_rows,
    )


def _compact_trajectory_rows(
    canonical: Sequence[_CompactCanonicalTrajectory],
    store: _OfficialFrameStore,
    partitions: Sequence[str],
) -> Iterable[dict[str, object]]:
    if len(canonical) != len(partitions):
        raise PipelineValidationError("compact partition ledger length mismatch")
    for record, partition in zip(canonical, partitions):
        layout = store.layouts[record.representative_index]
        yield {
            "canonical_video_prefix": layout.row.video_prefix,
            "compressed_jpeg_sha256s": store.compressed_hexes(
                record.representative_index
            ),
            "dedup_key": record.dedup_key.hex(),
            "canonical_trajectory_key": layout.trajectory_key.hex(),
            "scan_id": layout.row.scan_id,
            "partition": partition,
        }


def _compact_alias_rows(
    canonical: Sequence[_CompactCanonicalTrajectory],
    store: _OfficialFrameStore,
) -> Iterable[dict[str, object]]:
    for record in canonical:
        representative = store.layouts[record.representative_index]
        yield {
            "canonical_trajectory_key": representative.trajectory_key.hex(),
            "dedup_key": record.dedup_key.hex(),
            "aliases": [
                {
                    "source_id": store.layouts[index].row.source_id,
                    "annotation_row_index": store.layouts[index].row.row_index,
                    "canonical_video_prefix": store.layouts[index].row.video_prefix,
                    "trajectory_key": store.layouts[index].trajectory_key.hex(),
                    "scan_id": store.layouts[index].row.scan_id,
                }
                for index in record.alias_indices
            ],
        }


def _compact_full_trajectory_rows(
    canonical: Sequence[_CompactCanonicalTrajectory],
    store: _OfficialFrameStore,
    partitions: Sequence[str],
) -> Iterable[dict[str, object]]:
    """Yield the complete canonical trajectory product for production output.

    The bounded receipt keeps its historical six-field summary.  This richer
    stream is used only when the caller explicitly requests the full server
    manifest and adds facts already held by the verified annotation/frame
    store (source/row identity, BOS-inclusive actions and decoded RGB hashes).
    """

    if len(canonical) != len(partitions):
        raise PipelineValidationError("compact partition ledger length mismatch")
    # ``canonicalize_trajectories`` intentionally chooses representatives by
    # source/annotation provenance.  The persisted source ABI, however,
    # requires the full trajectory ledger to be strictly ordered by the raw
    # canonical trajectory key.  Sort only the bounded trajectory metadata
    # here; frame payloads remain in the verified store and are not copied.
    ordered_indices = sorted(
        range(len(canonical)),
        key=lambda index: store.layouts[canonical[index].representative_index].trajectory_key,
    )
    for index in ordered_indices:
        record = canonical[index]
        partition = partitions[index]
        layout = store.layouts[record.representative_index]
        row = layout.row
        projection_partition = (
            "project-train"
            if partition == "train"
            else "project-dev"
            if partition == "dev"
            else "excluded"
        )
        yield {
            "source_id": row.source_id,
            "annotation_row_index": row.row_index,
            "canonical_video_prefix": row.video_prefix,
            "actions": list(row.actions),
            "compressed_jpeg_sha256s": store.compressed_hexes(
                record.representative_index
            ),
            "decoded_rgb_sha256s": store.decoded_hexes(record.representative_index),
            "dedup_key": record.dedup_key.hex(),
            "canonical_trajectory_key": layout.trajectory_key.hex(),
            "scan_id": row.scan_id,
            "partition": partition,
            "projection_partition": projection_partition,
        }


def _building_ledger(sets: BuildingSets) -> list[dict[str, object]]:
    eligible = sets.train | sets.dev
    ranked = sorted(
        eligible,
        key=lambda scan_id: (
            hashlib.sha256(
                _DEV_DOMAIN + unicodedata.normalize("NFC", scan_id).encode("utf-8")
            ).digest(),
            scan_id.encode("utf-8"),
        ),
    )
    ranks = {scan_id: rank for rank, scan_id in enumerate(ranked)}
    all_scans = sorted(
        sets.S_stream | sets.S_pn_train | sets.S_final_mp3d | sets.S_final_gibson
    )
    ledger: list[dict[str, object]] = []
    for scan_id in all_scans:
        if scan_id in sets.train:
            partition = "train"
        elif scan_id in sets.dev:
            partition = "dev"
        elif scan_id in sets.S_final_mp3d:
            partition = "final_mp3d"
        elif scan_id in sets.S_final_gibson:
            partition = "final_gibson"
        elif scan_id in sets.excluded:
            partition = "excluded"
        else:
            partition = "not_stream"
        digest = (
            hashlib.sha256(
                _DEV_DOMAIN + unicodedata.normalize("NFC", scan_id).encode("utf-8")
            ).hexdigest()
            if scan_id in eligible
            else None
        )
        ledger.append(
            {
                "scan_id": scan_id,
                "in_stream": scan_id in sets.S_stream,
                "in_pointnav_train": scan_id in sets.S_pn_train,
                "in_final_mp3d": scan_id in sets.S_final_mp3d,
                "in_final_gibson": scan_id in sets.S_final_gibson,
                "dev_hash": digest,
                "dev_rank": ranks.get(scan_id),
                "partition": partition,
            }
        )
    return ledger


def _relative_to_manifest(path: Path, manifest_parent: Path) -> str:
    return Path(os.path.relpath(path, manifest_parent)).as_posix()


def _atomic_write_json(receipt_root: Path, receipt: Mapping[str, object]) -> str:
    if not receipt_root.is_dir():
        raise PipelineValidationError("receipt_root must already be an existing directory")
    receipt_name = "J2J_BUILD_CENSUS_RECEIPT_V1.json"
    destination = receipt_root / receipt_name
    encoded = (
        json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".j2j-census-", suffix=".tmp", dir=receipt_root
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return receipt_name


def build_scope_sidecar(receipt_path: Path | os.PathLike[str]) -> Path:
    """Write the H4 scope sidecar for one newly written V1 receipt.

    The digest is computed from the exact bytes on disk, rather than from a
    parsed/re-serialized object.  This keeps the sidecar an identity witness
    for the existing V1 receipt and does not alter that receipt's bytes.
    """

    receipt = Path(receipt_path)
    if not receipt.is_file():
        raise PipelineValidationError("census receipt is missing")
    if receipt.name != "J2J_BUILD_CENSUS_RECEIPT_V1.json":
        raise PipelineValidationError(
            "scope sidecar requires a J2J_BUILD_CENSUS_RECEIPT_V1.json receipt"
        )
    try:
        raw_bytes = receipt.read_bytes()
    except OSError as exc:
        raise PipelineValidationError("census receipt bytes are unreadable") from exc
    try:
        payload = _strict_json_bytes(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PipelineValidationError("census receipt is not strict JSON") from exc
    if not isinstance(payload, Mapping) or type(payload.get("max_horizon")) is not int:
        raise PipelineValidationError("V1 census receipt is missing max_horizon")
    if payload["max_horizon"] != _CENSUS_MAIN_HORIZON:
        raise PipelineValidationError("scope sidecar is only valid for H4 V1 receipts")
    declared_schema = payload.get("schema")
    if declared_schema is not None and declared_schema != "J2J_BUILD_CENSUS_RECEIPT_V1":
        raise PipelineValidationError("scope sidecar receipt schema mismatch")

    sidecar_payload = {
        "schema": "J2J_BUILD_CENSUS_SCOPE_RECEIPT_V1",
        "census_receipt_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "execution_scope": execution_scope(h_run=_CENSUS_MAIN_HORIZON),
    }
    validate_execution_scope(sidecar_payload["execution_scope"])
    destination = receipt.parent / "J2J_BUILD_CENSUS_SCOPE_RECEIPT_V1.json"
    if not receipt.parent.is_dir():
        raise PipelineValidationError("scope sidecar parent must be an existing directory")
    encoded = (
        json.dumps(sidecar_payload, sort_keys=True, indent=2, ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".j2j-census-scope-", suffix=".tmp", dir=receipt.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return destination


def _emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--annotation-root", required=True, type=Path)
    parser.add_argument("--rgb-root", required=True, type=Path)
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument("--receipt-root", required=True, type=Path)
    parser.add_argument(
        "--canonical-manifest-root",
        type=Path,
        help=(
            "Existing directory in which to persist the complete canonical "
            "trajectory and alias JSONL products. When omitted, the receipt "
            "retains its bounded ledger summaries for compatibility."
        ),
    )
    parser.add_argument(
        "--payload-spool-root",
        "--verified-payload-spool-root",
        dest="payload_spool_root",
        type=Path,
        help=(
            "Existing directory in which to retain the exact verified JPEG "
            "payload stream and metadata index for the released-cache driver. "
            "This opt-in bridge is production-only; the default path is unchanged."
        ),
    )
    parser.add_argument("--dev-count", type=int, default=11)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest_document = _strict_json_file(args.asset_manifest)
        if isinstance(manifest_document, _CapturedJson):
            manifest_payload = manifest_document.value
            manifest_bytes: bytes | None = manifest_document.raw_bytes
        else:
            # Test doubles may provide an already-parsed payload for an early gate.
            manifest_payload = manifest_document
            manifest_bytes = None
        asset_scope, entries = _validate_manifest(
            manifest_payload,
            annotation_root=args.annotation_root,
            rgb_root=args.rgb_root,
            episode_root=args.episode_root,
        )
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "failure_kind": "asset_manifest_mismatch",
                "manifest_errors": [str(exc)],
            }
        )
        return 2

    try:
        production_eligible = _validate_run_authority(
            asset_scope=asset_scope,
            dev_count=args.dev_count,
        )
    except PipelineValidationError as exc:
        _emit(
            {
                "ok": False,
                "failure_kind": "pipeline_validation_error",
                "errors": [str(exc)],
            }
        )
        return 5

    missing = _missing_assets(entries)
    if missing:
        _emit(
            {
                "ok": False,
                "failure_kind": "missing_assets",
                "missing_assets": missing,
            }
        )
        return 3

    try:
        manifest_errors = _verify_manifest_identities(entries)
        if production_eligible:
            require_released_production_identity(
                asset_scope=asset_scope,
                entries=entries,
                dev_count=args.dev_count,
            )
        if manifest_errors:
            raise ManifestValidationError("; ".join(manifest_errors))
    except Exception as exc:
        _emit(
            {
                "ok": False,
                "failure_kind": "asset_manifest_mismatch",
                "manifest_errors": [str(exc)],
            }
        )
        return 4

    payload_spool_writer: _VerifiedPayloadSpoolWriter | None = None
    try:
        if manifest_bytes is None:
            raise PipelineValidationError(
                "asset manifest bytes were not captured for consumed-byte authority"
            )
        config_bytes = _validate_config(args.config)
        h_run = _horizon_from_config_bytes(config_bytes)
        rows_by_source, trajectory_keys = _load_annotations(entries)
        if production_eligible:
            annotation_pre_audit = _annotation_pre_audit_compact(
                rows_by_source, trajectory_keys, max_horizon=h_run
            )
        else:
            annotation_pre_audit = _annotation_pre_audit(
                rows_by_source, trajectory_keys, max_horizon=h_run
            )

        r2r_rgb = _entry(
            entries, section="rgb_archives", dataset="R2R", order=0
        )
        rxr_part0 = _entry(
            entries, section="rgb_archives", dataset="RxR", order=0
        )
        rxr_part1 = _entry(
            entries, section="rgb_archives", dataset="RxR", order=1
        )
        if args.payload_spool_root is not None:
            if not production_eligible:
                raise PipelineValidationError(
                    "payload spool is available only for the official production path"
                )
            if args.canonical_manifest_root is None:
                raise PipelineValidationError(
                    "payload spool requires --canonical-manifest-root so cache input is complete"
                )
            payload_spool_writer = _VerifiedPayloadSpoolWriter(
                args.payload_spool_root
            )
            r2r_members = iter_verified_payloads(
                r2r_rgb.path, expected_sha256=r2r_rgb.sha256
            )
            rxr_members = iter_verified_payloads_from_parts(
                (rxr_part0.path, rxr_part1.path),
                expected_part_sha256s=(rxr_part0.sha256, rxr_part1.sha256),
                logical_archive_name="rgb/rxr_part0_part1.tar.gz",
            )
        else:
            r2r_members = iter_verified_members(
                r2r_rgb.path, expected_sha256=r2r_rgb.sha256
            )
            rxr_members = iter_verified_members_from_parts(
                (rxr_part0.path, rxr_part1.path),
                expected_part_sha256s=(rxr_part0.sha256, rxr_part1.sha256),
                logical_archive_name="rgb/rxr_part0_part1.tar.gz",
            )
        if production_eligible:
            compact_store = _OfficialFrameStore(rows_by_source, trajectory_keys)
            if payload_spool_writer is not None:
                compact_store.consume(
                    "R2R", payload_spool_writer.wrap("R2R", r2r_members)
                )
                compact_store.consume(
                    "RxR", payload_spool_writer.wrap("RxR", rxr_members)
                )
            else:
                compact_store.consume("R2R", r2r_members)
                compact_store.consume("RxR", rxr_members)
            decoded_jpeg_count = compact_store.finish()
            compact_canonical = _canonicalize_compact_trajectories(compact_store)
            canonical = None
        else:
            members_by_prefix, decoded_jpeg_count = _collect_members(
                (("R2R", r2r_members), ("RxR", rxr_members))
            )
            trajectory_records = _build_trajectory_records(
                rows_by_source, trajectory_keys, members_by_prefix
            )
            canonical = canonicalize_trajectories(trajectory_records)
            compact_store = None
            compact_canonical = None

        mp3d_entry = _entry(
            entries, section="episode_archives", dataset="mp3d", order=0
        )
        gibson_entry = _entry(
            entries, section="episode_archives", dataset="gibson", order=1
        )
        pn_train = load_pointnav_split_identity(
            mp3d_entry.path,
            split="train",
            dataset="mp3d",
            expected_identity=(mp3d_entry.bytes, mp3d_entry.sha256),
        )
        final_mp3d = load_pointnav_split_identity(
            mp3d_entry.path,
            split="val",
            dataset="mp3d",
            expected_identity=(mp3d_entry.bytes, mp3d_entry.sha256),
        )
        final_gibson = load_pointnav_split_identity(
            gibson_entry.path,
            split="val",
            dataset="gibson",
            expected_identity=(gibson_entry.bytes, gibson_entry.sha256),
        )
        stream_r2r = frozenset(row.scan_id for row in rows_by_source["R2R"])
        stream_rxr = frozenset(row.scan_id for row in rows_by_source["RxR"])
        stream = stream_r2r | stream_rxr
        if production_eligible:
            assert_official_set_authority(
                stream_r2r,
                stream_rxr,
                pn_train.scan_ids,
                final_mp3d.scan_ids,
                final_gibson.scan_ids,
            )
        building_sets = freeze_building_sets(
            stream,
            S_pn_train=pn_train.scan_ids,
            S_final_mp3d=final_mp3d.scan_ids,
            S_final_gibson=final_gibson.scan_ids,
            dev_count=args.dev_count,
        )
        canonical_manifest: dict[str, object] | None = None
        if production_eligible:
            if compact_store is None or compact_canonical is None:
                raise PipelineValidationError("official compact path was not initialized")
            production_post_rgb, compact_partitions = _production_census_compact(
                compact_canonical,
                compact_store,
                building_sets,
                max_horizon=h_run,
            )
            if args.canonical_manifest_root is not None:
                canonical_manifest = write_canonical_manifests(
                    args.canonical_manifest_root,
                    trajectory_rows=_compact_full_trajectory_rows(
                        compact_canonical, compact_store, compact_partitions
                    ),
                    alias_rows=_compact_alias_rows(compact_canonical, compact_store),
                )
                trajectory_ledger = canonical_manifest["trajectory"]
                alias_ledger = canonical_manifest["alias"]
            else:
                trajectory_ledger = _canonical_jsonl_summary(
                    _compact_trajectory_rows(
                        compact_canonical, compact_store, compact_partitions
                    )
                )
                alias_ledger = _canonical_jsonl_summary(
                    _compact_alias_rows(compact_canonical, compact_store)
                )
            canonical_count = len(compact_canonical)
            dedup_alias_count = sum(
                len(record.alias_indices) - 1 for record in compact_canonical
            )
        else:
            if canonical is None:
                raise PipelineValidationError("synthetic exhaustive path was not initialized")
            production_post_rgb, partitions = _production_census(
                canonical,
                building_sets,
                max_horizon=h_run,
            )
            trajectory_ledger = [
                {
                    "canonical_video_prefix": record.canonical_video_prefix,
                    "compressed_jpeg_sha256s": list(record.compressed_jpeg_sha256s),
                    "dedup_key": record.dedup_key.hex(),
                    "canonical_trajectory_key": record.canonical_trajectory_key.hex(),
                    "scan_id": record.scan_id,
                    "partition": partitions[record.canonical_trajectory_key],
                }
                for record in canonical
            ]
            alias_ledger = _alias_ledger(canonical)
            canonical_count = len(canonical)
            dedup_alias_count = sum(len(record.aliases) - 1 for record in canonical)
        if compact_store is None:
            stream_rgb_receipt = {
                "decoded_jpeg_count": decoded_jpeg_count,
                "ignored_extra_frame_count": 0,
                "ignored_extra_prefixes": [],
            }
        else:
            ignored_extra_summary = compact_store.ignored_extra_summary()
            stream_rgb_receipt = {
                # Preserve the historical field as the count of annotated
                # frames admitted to the compact store.  The separate fields
                # make the six verified scene-only prefixes explicit rather
                # than silently folding them into training data.
                "decoded_jpeg_count": decoded_jpeg_count,
                "ignored_extra_frame_count": compact_store.ignored_extra_frame_count,
                "ignored_extra_prefixes": [dict(row) for row in ignored_extra_summary],
                "verified_regular_jpeg_count": (
                    decoded_jpeg_count + compact_store.ignored_extra_frame_count
                ),
            }
        receipt = {
            "schema": "J2J_BUILD_CENSUS_RECEIPT_V1",
            "asset_scope": asset_scope,
            "production_eligible": production_eligible,
            "max_horizon": h_run,
            "input_identity": {
                "asset_manifest_relpath": args.asset_manifest.name,
                "asset_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "annotation_root_rel": _relative_to_manifest(
                    args.annotation_root, args.asset_manifest.parent
                ),
                "rgb_root_rel": _relative_to_manifest(
                    args.rgb_root, args.asset_manifest.parent
                ),
                "episode_root_rel": _relative_to_manifest(
                    args.episode_root, args.asset_manifest.parent
                ),
                "config_name": args.config.name,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "assets": [entry.receipt_identity() for entry in entries],
            },
            "annotation_rows": {
                "R2R": len(rows_by_source["R2R"]),
                "RxR": len(rows_by_source["RxR"]),
                "total": len(rows_by_source["R2R"]) + len(rows_by_source["RxR"]),
            },
            "annotation_pre_audit": _census_receipt(annotation_pre_audit),
            "stream_rgb": stream_rgb_receipt,
            "dedup": {
                "canonical_trajectories": canonical_count,
                "dedup_alias_count": dedup_alias_count,
                "alias_ledger": alias_ledger,
            },
            "pointnav": {
                "mp3d_train": _pointnav_receipt(pn_train),
                "mp3d_val": _pointnav_receipt(final_mp3d),
                "gibson_val": _pointnav_receipt(final_gibson),
            },
            "building_sets": {
                "S_stream": sorted(building_sets.S_stream),
                "S_pn_train": sorted(building_sets.S_pn_train),
                "S_final_mp3d": sorted(building_sets.S_final_mp3d),
                "S_final_gibson": sorted(building_sets.S_final_gibson),
                "excluded": sorted(building_sets.excluded),
                "train": sorted(building_sets.train),
                "dev": sorted(building_sets.dev),
                "ledger": _building_ledger(building_sets),
            },
            "trajectory_ledger": trajectory_ledger,
            "production_post_rgb": _census_receipt(production_post_rgb),
        }
        if canonical_manifest is not None:
            receipt["canonical_manifest"] = canonical_manifest
        # Publish the optional payload bridge only after all archive coverage,
        # deduplication, split, and census checks have succeeded.  It is a
        # sidecar by design, so the historical V1 receipt bytes/schema remain
        # unchanged regardless of the opt-in flag.
        payload_spool_sidecar: dict[str, object] | None = None
        if payload_spool_writer is not None:
            payload_spool_sidecar = payload_spool_writer.finalize()
        receipt_relpath = _atomic_write_json(args.receipt_root, receipt)
        outer_manifest_result: dict[str, object] | None = None
        if canonical_manifest is not None:
            # The receipt is written first so the outer manifest can bind its
            # exact bytes.  This remains a finite, same-process publication;
            # no second archive/model pass is introduced.
            outer_manifest_result = write_canonical_outer_manifest_from_receipt(
                Path(args.receipt_root) / receipt_relpath,
                canonical_root=args.canonical_manifest_root,
            )
    except (AssetIdentityError, ArchiveIdentityError, PointNavIdentityError) as exc:
        if payload_spool_writer is not None:
            payload_spool_writer.abort()
        _emit(
            {
                "ok": False,
                "failure_kind": "asset_manifest_mismatch",
                "manifest_errors": [str(exc)],
            }
        )
        return 4
    except Exception as exc:
        if payload_spool_writer is not None:
            payload_spool_writer.abort()
        _emit(
            {
                "ok": False,
                "failure_kind": "pipeline_validation_error",
                "errors": [str(exc)],
            }
        )
        return 5

    result_payload: dict[str, object] = {"ok": True, "receipt_relpath": receipt_relpath}
    if outer_manifest_result is not None:
        result_payload["canonical_source_manifest"] = outer_manifest_result
    if payload_spool_sidecar is not None:
        result_payload["payload_spool_relpath"] = (
            _VerifiedPayloadSpoolWriter._RECEIPT_NAME
        )
        result_payload["payload_spool_receipt"] = str(
            Path(args.payload_spool_root)
            / _VerifiedPayloadSpoolWriter._RECEIPT_NAME
        )
    _emit(result_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
