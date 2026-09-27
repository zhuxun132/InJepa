"""Trajectory deduplication and fail-closed building-first split construction."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import re
import stat
import struct
import unicodedata
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence


_DEDUP_DOMAIN = b"J2J_TRAJECTORY_DEDUP_V1\x00"
_DEV_DOMAIN = b"J2J_DEV_BUILDING_V1\x00"
_SHA256_HEX = re.compile(r"[0-9A-Fa-f]{64}")
_SCAN_ID = re.compile(r"[A-Za-z0-9]+")
_WINDOWS_DRIVE_ABSOLUTE = re.compile(r"^[A-Za-z]:/")
_VIDEO_PREFIX = re.compile(r"images/[A-Za-z0-9]+_(?:r2r|rxr)_[0-9]{6}")
_READ_CHUNK_BYTES = 1024 * 1024


OFFICIAL_SET_AUTHORITY = {
    "stream_r2r": {
        "count": 61,
        "sha256": "4ed1c414b66fd37c761e012974dc28a2b34f56f9315391cf1469bac0a9ef162c",
    },
    "stream_rxr": {
        "count": 59,
        "sha256": "7e08fa9083584a689f2c382d5e707a5c687ecf83d66f3aa8b1e69ff27743ef79",
    },
    "pn_train": {
        "count": 61,
        "sha256": "4ed1c414b66fd37c761e012974dc28a2b34f56f9315391cf1469bac0a9ef162c",
    },
    "final_mp3d": {
        "count": 11,
        "sha256": "c032d26cbe9ae52fe73c3f6d9ef6aae8f713a49bf4c4bc4cd7fb3b16e276291e",
    },
    "final_gibson": {
        "count": 14,
        "sha256": "32e1a4562a8c791d4e8e9420a3c5115cecc5e514a5dc7f4b8578c98524029607",
    },
}


class SplitConstructionError(ValueError):
    """Raised when identity, deduplication, or partition invariants fail."""


class PointNavIdentityError(SplitConstructionError):
    """Raised when PointNav bytes miss their expected input identity."""


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_key: bytes
    source_id: str
    annotation_row_index: int
    canonical_video_prefix: str
    scan_id: str
    actions: tuple[int, ...]
    compressed_jpeg_sha256s: tuple[str, ...]
    decoded_rgb_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalAlias:
    source_id: str
    annotation_row_index: int
    canonical_video_prefix: str
    trajectory_key: bytes
    scan_id: str


@dataclass(frozen=True)
class CanonicalTrajectory:
    canonical_trajectory_key: bytes
    dedup_key: bytes
    scan_id: str
    source_id: str
    annotation_row_index: int
    canonical_video_prefix: str
    actions: tuple[int, ...]
    compressed_jpeg_sha256s: tuple[str, ...]
    decoded_rgb_sha256s: tuple[str, ...]
    aliases: tuple[CanonicalAlias, ...]


@dataclass(frozen=True)
class PointNavMemberLedger:
    split: str
    dataset: str
    member_path: str
    compressed_bytes: int
    compressed_sha256: str
    decompressed_bytes: int
    decompressed_sha256: str


@dataclass(frozen=True)
class PointNavSplitIdentity:
    split: str
    dataset: str
    content_basenames: frozenset[str]
    explicit_content_scenes: frozenset[str]
    episode_scene_ids: frozenset[str]
    scan_ids: frozenset[str]
    member_ledger: tuple[PointNavMemberLedger, ...]


@dataclass(frozen=True)
class BuildingSets:
    S_stream: frozenset[str]
    S_pn_train: frozenset[str]
    S_final_mp3d: frozenset[str]
    S_final_gibson: frozenset[str]
    excluded: frozenset[str]
    train: frozenset[str]
    dev: frozenset[str]


def _length_prefix(payload: bytes) -> bytes:
    if len(payload) >= 2**64:
        raise ValueError("payload is too large")
    return struct.pack("<Q", len(payload)) + payload


def _raw_sha256(value: object, *, label: str) -> bytes:
    if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
        raise ValueError(f"{label} must be exactly 64 hexadecimal characters")
    return bytes.fromhex(value)


def _validated_actions(actions: object) -> bytes:
    if type(actions) is not tuple or not actions:
        raise ValueError("actions must be a full non-empty tuple")
    if type(actions[0]) is not int or actions[0] != -1:
        raise ValueError("actions must start with exact BOS -1")
    if any(type(action) is not int or not 0 <= action <= 3 for action in actions[1:]):
        raise ValueError("action payload must contain exact integer ids 0 through 3")
    return bytes(actions[1:])


def _dedup_preimage(frame_sha256s: object, actions: object) -> bytes:
    if type(frame_sha256s) is not tuple or not frame_sha256s:
        raise ValueError("frame_sha256s must be a non-empty tuple")
    action_payload = _validated_actions(actions)
    if len(frame_sha256s) != len(actions):
        raise ValueError("frame and action sequences must have equal length")
    frame_payload = b"".join(
        _raw_sha256(value, label="frame SHA-256") for value in frame_sha256s
    )
    return (
        _DEDUP_DOMAIN
        + struct.pack("<Q", len(frame_sha256s))
        + frame_payload
        + _length_prefix(action_payload)
    )


def trajectory_dedup_key(frame_sha256s: object, actions: object) -> bytes:
    """Hash the complete compressed-JPEG sequence and factual action payload."""
    return hashlib.sha256(_dedup_preimage(frame_sha256s, actions)).digest()


def _canonical_text(value: object, *, label: str) -> str:
    if type(value) is not str or not value:
        raise SplitConstructionError(f"{label} must be a non-empty string")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise SplitConstructionError(f"{label} must already be NFC-normalized")
    return normalized


def _representative_key(record: TrajectoryRecord) -> tuple[bytes, int, bytes, bytes]:
    return (
        unicodedata.normalize("NFC", record.source_id).encode("utf-8"),
        record.annotation_row_index,
        unicodedata.normalize("NFC", record.canonical_video_prefix).encode("utf-8"),
        record.trajectory_key,
    )


def _validate_trajectory_record(record: object) -> tuple[bytes, bytes]:
    if not isinstance(record, TrajectoryRecord):
        raise SplitConstructionError("records must contain TrajectoryRecord values")
    if type(record.trajectory_key) is not bytes or len(record.trajectory_key) != 32:
        raise SplitConstructionError("trajectory_key must be exactly 32 raw bytes")
    if record.source_id not in ("R2R", "RxR"):
        raise SplitConstructionError("invalid trajectory source")
    if (
        type(record.annotation_row_index) is not int
        or not 0 <= record.annotation_row_index < 2**64
    ):
        raise SplitConstructionError("annotation row index must be uint64")
    prefix = _canonical_text(record.canonical_video_prefix, label="video prefix")
    if _VIDEO_PREFIX.fullmatch(prefix) is None:
        raise SplitConstructionError("video prefix does not use the canonical grammar")
    scan_id = _canonical_text(record.scan_id, label="scan id")
    if _SCAN_ID.fullmatch(scan_id) is None:
        raise SplitConstructionError("scan id must be ASCII alphanumeric")
    try:
        preimage = _dedup_preimage(record.compressed_jpeg_sha256s, record.actions)
    except ValueError as exc:
        raise SplitConstructionError("invalid trajectory dedup preimage") from exc
    if type(record.decoded_rgb_sha256s) is not tuple:
        raise SplitConstructionError("decoded RGB hashes must be a tuple")
    if len(record.decoded_rgb_sha256s) != len(record.compressed_jpeg_sha256s):
        raise SplitConstructionError("compressed and decoded frame counts disagree")
    try:
        for value in record.decoded_rgb_sha256s:
            _raw_sha256(value, label="decoded RGB SHA-256")
    except ValueError as exc:
        raise SplitConstructionError("invalid decoded RGB SHA-256") from exc
    try:
        digest = trajectory_dedup_key(record.compressed_jpeg_sha256s, record.actions)
    except ValueError as exc:
        raise SplitConstructionError("invalid trajectory dedup key") from exc
    if type(digest) is not bytes or len(digest) != 32:
        raise SplitConstructionError("trajectory_dedup_key must return 32 raw bytes")
    return digest, preimage


def canonicalize_trajectories(
    records: Iterable[TrajectoryRecord],
) -> tuple[CanonicalTrajectory, ...]:
    """Deduplicate complete trajectories while preserving every source alias."""
    try:
        materialized = tuple(records)
    except TypeError as exc:
        raise SplitConstructionError("records must be iterable") from exc

    families: dict[bytes, list[tuple[TrajectoryRecord, bytes]]] = {}
    identities: set[tuple[str, int, str, bytes]] = set()
    for record in materialized:
        digest, preimage = _validate_trajectory_record(record)
        identity = (
            record.source_id,
            record.annotation_row_index,
            record.canonical_video_prefix,
            record.trajectory_key,
        )
        if identity in identities:
            raise SplitConstructionError("duplicate trajectory provenance identity")
        identities.add(identity)
        families.setdefault(digest, []).append((record, preimage))

    canonical: list[CanonicalTrajectory] = []
    for dedup_key, family in families.items():
        if len({preimage for _, preimage in family}) != 1:
            raise SplitConstructionError("distinct dedup preimages collided")
        records_in_family = [record for record, _ in family]
        if len({record.scan_id for record in records_in_family}) != 1:
            raise SplitConstructionError("a dedup family crosses scan identities")
        if len({record.decoded_rgb_sha256s for record in records_in_family}) != 1:
            raise SplitConstructionError(
                "identical compressed trajectories decode to different RGB bytes"
            )

        ordered = sorted(records_in_family, key=_representative_key)
        representative = ordered[0]
        aliases = tuple(
            CanonicalAlias(
                source_id=record.source_id,
                annotation_row_index=record.annotation_row_index,
                canonical_video_prefix=record.canonical_video_prefix,
                trajectory_key=record.trajectory_key,
                scan_id=record.scan_id,
            )
            for record in ordered
        )
        canonical.append(
            CanonicalTrajectory(
                canonical_trajectory_key=representative.trajectory_key,
                dedup_key=dedup_key,
                scan_id=representative.scan_id,
                source_id=representative.source_id,
                annotation_row_index=representative.annotation_row_index,
                canonical_video_prefix=representative.canonical_video_prefix,
                actions=representative.actions,
                compressed_jpeg_sha256s=representative.compressed_jpeg_sha256s,
                decoded_rgb_sha256s=representative.decoded_rgb_sha256s,
                aliases=aliases,
            )
        )
    return tuple(
        sorted(
            canonical,
            key=lambda record: (
                record.source_id.encode("utf-8"),
                record.annotation_row_index,
                record.canonical_video_prefix.encode("utf-8"),
                record.canonical_trajectory_key,
            ),
        )
    )


def _canonical_zip_member(name: object, *, is_dir: bool) -> str:
    if type(name) is not str or not name:
        raise SplitConstructionError("zip member path must be non-empty")
    if "\\" in name:
        raise SplitConstructionError("zip member paths must use forward slashes")
    candidate = name[:-1] if is_dir and name.endswith("/") else name
    normalized = unicodedata.normalize("NFC", candidate)
    if normalized.startswith("/") or _WINDOWS_DRIVE_ABSOLUTE.match(normalized):
        raise SplitConstructionError("zip member path must be relative")
    segments = normalized.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise SplitConstructionError("zip member path contains an invalid segment")
    return normalized


def _reject_special_zip_member(info: zipfile.ZipInfo) -> None:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise SplitConstructionError("special zip members are forbidden")


def _strict_json(
    payload: bytes, *, member_path: str
) -> tuple[Mapping[str, object], bytes]:
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
        decoded = gzip.decompress(payload)
        value = json.loads(
            decoded.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SplitConstructionError(f"invalid gzip JSON member: {member_path}") from exc
    if not isinstance(value, Mapping):
        raise SplitConstructionError(f"JSON member is not an object: {member_path}")
    return value, decoded


def _explicit_scenes(payload: Mapping[str, object]) -> frozenset[str]:
    raw = payload.get("content_scenes", [])
    if type(raw) is not list:
        raise SplitConstructionError("content_scenes must be a list")
    if raw == ["*"]:
        return frozenset()
    if "*" in raw:
        raise SplitConstructionError("content_scenes wildcard cannot be mixed")
    result: set[str] = set()
    for value in raw:
        if type(value) is not str or _SCAN_ID.fullmatch(value) is None:
            raise SplitConstructionError("invalid explicit content scene")
        if value in result:
            raise SplitConstructionError("duplicate explicit content scene")
        result.add(value)
    return frozenset(result)


def _scene_id(raw: object, *, dataset: str) -> str:
    if type(raw) is not str or not raw or "\\" in raw:
        raise SplitConstructionError("episode scene_id must be a canonical string")
    normalized = unicodedata.normalize("NFC", raw)
    if normalized.startswith("/") or _WINDOWS_DRIVE_ABSOLUTE.match(normalized):
        raise SplitConstructionError("episode scene_id must be relative")
    segments = normalized.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise SplitConstructionError("episode scene_id contains an invalid segment")
    if dataset == "mp3d":
        if len(segments) < 3 or segments[-3] != "mp3d":
            raise SplitConstructionError("MP3D scene_id has the wrong dataset grammar")
        scan = segments[-2]
        filename = segments[-1]
        if _SCAN_ID.fullmatch(scan) is None or filename != f"{scan}.glb":
            raise SplitConstructionError("MP3D parent and scene stem must agree")
        return scan
    if dataset == "gibson":
        if len(segments) < 2 or segments[-2] != "gibson":
            raise SplitConstructionError("Gibson scene_id has the wrong dataset grammar")
        filename = segments[-1]
        if not filename.endswith(".glb"):
            raise SplitConstructionError("Gibson scene_id must end in .glb")
        scene = filename[:-4]
        if _SCAN_ID.fullmatch(scene) is None:
            raise SplitConstructionError("invalid Gibson scene identity")
        return scene
    raise SplitConstructionError("unsupported PointNav dataset")


def _episode_scenes(payload: Mapping[str, object], *, dataset: str) -> frozenset[str]:
    raw_episodes = payload.get("episodes", [])
    if type(raw_episodes) is not list:
        raise SplitConstructionError("episodes must be a list")
    result: set[str] = set()
    for episode in raw_episodes:
        if not isinstance(episode, Mapping) or "scene_id" not in episode:
            raise SplitConstructionError("each episode must contain scene_id")
        result.add(_scene_id(episode["scene_id"], dataset=dataset))
    return frozenset(result)


def _validated_expected_identity(value: object) -> tuple[int, str] | None:
    if value is None:
        return None
    if type(value) is not tuple or len(value) != 2:
        raise SplitConstructionError(
            "expected_identity must be an exact (bytes, SHA-256) tuple"
        )
    expected_bytes, expected_sha256 = value
    if type(expected_bytes) is not int or expected_bytes < 1:
        raise SplitConstructionError("expected PointNav bytes must be a positive integer")
    if (
        type(expected_sha256) is not str
        or _SHA256_HEX.fullmatch(expected_sha256) is None
    ):
        raise SplitConstructionError(
            "expected PointNav SHA-256 must be exactly 64 hexadecimal characters"
        )
    return expected_bytes, expected_sha256.lower()


@contextmanager
def _open_verified_pointnav_zip(
    path: Path, expected_identity: object
) -> Iterator[zipfile.ZipFile]:
    expected = _validated_expected_identity(expected_identity)
    with path.open("rb") as handle:
        if expected is not None:
            digest = hashlib.sha256()
            actual_bytes = 0
            while chunk := handle.read(_READ_CHUNK_BYTES):
                actual_bytes += len(chunk)
                digest.update(chunk)
            if (actual_bytes, digest.hexdigest()) != expected:
                raise PointNavIdentityError(
                    f"PointNav ZIP bytes or SHA-256 mismatch: {path}"
                )
            handle.seek(0)
        with zipfile.ZipFile(handle, "r") as archive:
            yield archive


def load_pointnav_split_identity(
    episode_zip: str | Path,
    *,
    split: str,
    dataset: str,
    expected_identity: tuple[int, str] | None = None,
) -> PointNavSplitIdentity:
    """Load one allowlisted PointNav split and reconcile every identity source."""
    if dataset not in ("mp3d", "gibson"):
        raise SplitConstructionError("unsupported PointNav dataset")
    if (dataset, split) not in (("mp3d", "train"), ("mp3d", "val"), ("gibson", "val")):
        raise SplitConstructionError("unsupported PointNav dataset/split pair")

    path = Path(episode_zip)
    root_path = f"{split}/{split}.json.gz"
    selected: dict[str, zipfile.ZipInfo] = {}
    try:
        with _open_verified_pointnav_zip(path, expected_identity) as archive:
            seen: set[str] = set()
            for info in archive.infolist():
                _reject_special_zip_member(info)
                member_path = _canonical_zip_member(info.filename, is_dir=info.is_dir())
                if member_path in seen:
                    raise SplitConstructionError(f"duplicate zip member: {member_path}")
                seen.add(member_path)
                if not (member_path == split or member_path.startswith(f"{split}/")):
                    continue
                if info.is_dir():
                    allowed_dirs = {split}
                    if dataset == "mp3d" and split == "train":
                        allowed_dirs.add(f"{split}/content")
                    if member_path not in allowed_dirs:
                        raise SplitConstructionError(
                            f"unexpected directory in PointNav split: {member_path}"
                        )
                    continue
                if member_path == root_path:
                    selected[member_path] = info
                    continue
                if dataset == "mp3d" and split == "train":
                    match = re.fullmatch(r"train/content/([A-Za-z0-9]+)\.json\.gz", member_path)
                    if match is not None:
                        selected[member_path] = info
                        continue
                raise SplitConstructionError(
                    f"unexpected file in PointNav split: {member_path}"
                )

            if root_path not in selected:
                raise SplitConstructionError(f"missing PointNav root member: {root_path}")
            content_paths = sorted(
                member_path
                for member_path in selected
                if member_path != root_path
            )
            if dataset == "mp3d" and split == "train" and not content_paths:
                raise SplitConstructionError("MP3D train must contain content members")

            ordered_paths = (root_path, *content_paths)
            content_basenames = frozenset(
                member_path.removeprefix("train/content/").removesuffix(".json.gz")
                for member_path in content_paths
            )
            explicit = frozenset()
            episode_scene_ids: set[str] = set()
            ledgers: list[PointNavMemberLedger] = []
            for member_path in ordered_paths:
                compressed = archive.read(selected[member_path])
                payload, decompressed = _strict_json(
                    compressed, member_path=member_path
                )
                ledgers.append(
                    PointNavMemberLedger(
                        split=split,
                        dataset=dataset,
                        member_path=member_path,
                        compressed_bytes=len(compressed),
                        compressed_sha256=hashlib.sha256(compressed).hexdigest(),
                        decompressed_bytes=len(decompressed),
                        decompressed_sha256=hashlib.sha256(decompressed).hexdigest(),
                    )
                )
                if member_path == root_path:
                    explicit = _explicit_scenes(payload)
                    root_episode_scenes = _episode_scenes(payload, dataset=dataset)
                    if dataset == "mp3d" and split == "train" and root_episode_scenes:
                        raise SplitConstructionError(
                            "MP3D train root episodes must be empty"
                        )
                    episode_scene_ids.update(root_episode_scenes)
                else:
                    member_scenes = _episode_scenes(payload, dataset=dataset)
                    basename = member_path.removeprefix(
                        "train/content/"
                    ).removesuffix(".json.gz")
                    if member_scenes and member_scenes != {basename}:
                        raise SplitConstructionError(
                            "content member scenes disagree with its basename"
                        )
                    episode_scene_ids.update(member_scenes)
                del payload, decompressed, compressed
    except SplitConstructionError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError, RuntimeError, EOFError) as exc:
        raise SplitConstructionError(f"invalid PointNav zip archive: {path}") from exc

    episode_ids = frozenset(episode_scene_ids)

    sources = [source for source in (explicit, content_basenames, episode_ids) if source]
    if not sources:
        raise SplitConstructionError("PointNav split contains no scene identity source")
    if any(source != sources[0] for source in sources[1:]):
        raise SplitConstructionError("PointNav identity sources disagree")

    return PointNavSplitIdentity(
        split=split,
        dataset=dataset,
        content_basenames=content_basenames,
        explicit_content_scenes=explicit,
        episode_scene_ids=episode_ids,
        scan_ids=sources[0],
        member_ledger=tuple(ledgers),
    )


load_pointnav_split_identity.__signature__ = inspect.Signature(
    parameters=(
        inspect.Parameter("episode_zip", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("split", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter("dataset", inspect.Parameter.KEYWORD_ONLY),
    )
)


def _validated_scan_set(values: object, *, label: str) -> frozenset[str]:
    try:
        materialized = frozenset(values)
    except TypeError as exc:
        raise SplitConstructionError(f"{label} must be an iterable of scan ids") from exc
    for scan_id in materialized:
        if (
            type(scan_id) is not str
            or unicodedata.normalize("NFC", scan_id) != scan_id
            or _SCAN_ID.fullmatch(scan_id) is None
        ):
            raise SplitConstructionError(f"{label} contains an invalid ASCII scan id")
        try:
            scan_id.encode("ascii")
        except UnicodeEncodeError as exc:
            raise SplitConstructionError(f"{label} contains a non-ASCII scan id") from exc
    return materialized


def ascii_sorted_lf_sha256(scan_ids: Iterable[str]) -> str:
    """Hash unique ASCII scan ids sorted by raw bytes with a final LF per item."""
    values = _validated_scan_set(scan_ids, label="scan_ids")
    payload = b"".join(scan_id.encode("ascii") + b"\n" for scan_id in sorted(values))
    return hashlib.sha256(payload).hexdigest()


def assert_official_set_authority(
    stream_r2r: Iterable[str],
    stream_rxr: Iterable[str],
    pn_train: Iterable[str],
    final_mp3d: Iterable[str],
    final_gibson: Iterable[str],
) -> None:
    """Require exact official set identities plus frozen subset/equality relations."""
    named = {
        "stream_r2r": _validated_scan_set(stream_r2r, label="stream_r2r"),
        "stream_rxr": _validated_scan_set(stream_rxr, label="stream_rxr"),
        "pn_train": _validated_scan_set(pn_train, label="pn_train"),
        "final_mp3d": _validated_scan_set(final_mp3d, label="final_mp3d"),
        "final_gibson": _validated_scan_set(final_gibson, label="final_gibson"),
    }
    for label, values in named.items():
        authority = OFFICIAL_SET_AUTHORITY[label]
        if len(values) != authority["count"] or ascii_sorted_lf_sha256(values) != authority[
            "sha256"
        ]:
            raise SplitConstructionError(f"{label} does not match official set authority")
    if not named["stream_rxr"] <= named["stream_r2r"]:
        raise SplitConstructionError("RxR scans must be a subset of R2R scans")
    if named["pn_train"] != named["stream_r2r"]:
        raise SplitConstructionError("PointNav train scans must exactly equal R2R scans")


def _dev_rank(scan_id: str) -> tuple[bytes, bytes]:
    encoded = unicodedata.normalize("NFC", scan_id).encode("utf-8")
    return hashlib.sha256(_DEV_DOMAIN + encoded).digest(), encoded


def freeze_building_sets(
    S_stream: Iterable[str],
    *,
    S_pn_train: Iterable[str],
    S_final_mp3d: Iterable[str],
    S_final_gibson: Iterable[str],
    dev_count: int = 11,
) -> BuildingSets:
    """Freeze eligible StreamVLN buildings before selecting the hash-ranked dev set."""
    if type(dev_count) is not int or dev_count < 1:
        raise SplitConstructionError("dev_count must be a positive integer")
    stream = _validated_scan_set(S_stream, label="S_stream")
    pn_train = _validated_scan_set(S_pn_train, label="S_pn_train")
    final_mp3d = _validated_scan_set(S_final_mp3d, label="S_final_mp3d")
    final_gibson = _validated_scan_set(S_final_gibson, label="S_final_gibson")
    if final_mp3d & final_gibson:
        raise SplitConstructionError("MP3D and Gibson final building sets overlap")

    final = final_mp3d | final_gibson
    eligible = (stream & pn_train) - final
    if len(eligible) < dev_count + 1:
        raise SplitConstructionError("not enough eligible buildings for dev and train")
    dev = frozenset(sorted(eligible, key=_dev_rank)[:dev_count])
    train = frozenset(eligible - dev)
    excluded = frozenset(stream - (eligible | final))
    return BuildingSets(
        S_stream=stream,
        S_pn_train=pn_train,
        S_final_mp3d=final_mp3d,
        S_final_gibson=final_gibson,
        excluded=excluded,
        train=train,
        dev=dev,
    )


def assign_building_partition(scan_id: str, sets: BuildingSets) -> str:
    """Return the unique frozen partition for a known scan identity."""
    if type(scan_id) is not str or not isinstance(sets, BuildingSets):
        raise SplitConstructionError("invalid building partition query")
    memberships = [
        ("train", scan_id in sets.train),
        ("dev", scan_id in sets.dev),
        ("final_mp3d", scan_id in sets.S_final_mp3d),
        ("final_gibson", scan_id in sets.S_final_gibson),
        ("excluded", scan_id in sets.excluded),
    ]
    labels = [label for label, present in memberships if present]
    if len(labels) != 1:
        raise SplitConstructionError("scan identity has zero or conflicting partitions")
    return labels[0]
