"""Fail-closed streaming readers for StreamVLN RGB archives."""

from __future__ import annotations

from contextlib import ExitStack
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
import tarfile
import unicodedata
from collections.abc import Iterable, Iterator, Sequence

from PIL import Image


_SHA256_HEX = re.compile(r"[0-9A-Fa-f]{64}")
_WINDOWS_DRIVE_ABSOLUTE = re.compile(r"^[A-Za-z]:/")
_VIDEO_PREFIX = re.compile(
    r"^images/[A-Za-z0-9]+_(?P<source>r2r|rxr)_[0-9]{6}$"
)
_FRAME_MEMBER_RGB = re.compile(
    r"^(?P<prefix>images/[A-Za-z0-9]+_(?:r2r|rxr)_[0-9]{6})/"
    r"rgb/(?P<frame>[0-9]{3})\.jpg$"
)
_FRAME_MEMBER_RGB_IMAGES = re.compile(
    r"^(?P<prefix>images/[A-Za-z0-9]+_rxr_[0-9]{6})/"
    # The released scene-only RxR archive uses three-digit names through 999
    # and four-digit names thereafter (for example 055.jpg and 1135.jpg).
    r"rgb_images/(?P<frame>[0-9]{3,4})\.jpg$"
)
# Kept as a compatibility spelling for downstream code that imported the old
# private constant.  New parsing must go through ``parse_frame_member`` so the
# two verified release layouts share one owner.
_FRAME_MEMBER = _FRAME_MEMBER_RGB
_READ_CHUNK_BYTES = 1024 * 1024


class ArchiveValidationError(ValueError):
    """Raised when an RGB archive or one of its members is invalid."""


class ArchiveIdentityError(ArchiveValidationError):
    """Raised when bytes read from an RGB archive miss their authority hash."""


class FrameSequenceError(ValueError):
    """Raised when archive members do not exactly cover an expected sequence."""


@dataclass(frozen=True)
class VerifiedArchiveMember:
    """Identity and decoded-image facts for one verified JPEG member."""

    archive_path: str
    member_path: str
    compressed_sha256: str
    decoded_rgb_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class VerifiedArchivePayload:
    """A verified member together with its original JPEG bytes.

    ``VerifiedArchiveMember`` is intentionally left byte-for-byte compatible
    with the original archive API.  This companion view is produced only by
    the payload iterators below, after the same archive identity, tar-member,
    JPEG decode, and hash checks have completed.  The bytes are the exact
    member bytes read from the tar stream; they are never re-encoded.
    """

    member: VerifiedArchiveMember
    jpeg_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.member, VerifiedArchiveMember):
            raise ArchiveValidationError("payload view requires a verified archive member")
        if type(self.jpeg_bytes) is not bytes:
            raise ArchiveValidationError("verified JPEG payload must be immutable bytes")
        if hashlib.sha256(self.jpeg_bytes).hexdigest() != self.member.compressed_sha256:
            raise ArchiveValidationError("verified JPEG payload does not match member hash")

    @property
    def payload(self) -> bytes:
        """Compatibility spelling for callers that call the bytes ``payload``."""

        return self.jpeg_bytes

    # Forwarding identity fields keeps the view convenient for cache writers
    # without creating a second member metadata representation.
    @property
    def archive_path(self) -> str:
        return self.member.archive_path

    @property
    def member_path(self) -> str:
        return self.member.member_path

    @property
    def compressed_sha256(self) -> str:
        return self.member.compressed_sha256

    @property
    def decoded_rgb_sha256(self) -> str:
        return self.member.decoded_rgb_sha256

    @property
    def width(self) -> int:
        return self.member.width

    @property
    def height(self) -> int:
        return self.member.height


@dataclass(frozen=True)
class FrameMemberIdentity:
    """Parsed identity for one released RGB frame member.

    ``layout`` is either the three-digit ``rgb`` directory used by R2R and
    some RxR trajectories, or the three/four-digit ``rgb_images`` directory
    observed in the official RxR release. ``frame_width`` records the
    observed number of decimal digits and may vary within one ``rgb_images``
    sequence. The frame index is always one-based.
    """

    video_prefix: str
    source_id: str
    layout: str
    frame_width: int
    frame_index: int


def _validated_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
        raise ArchiveValidationError(f"{label} must be exactly 64 hexadecimal characters")
    return value.lower()


def _stream_sha256(handle: object) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(_READ_CHUNK_BYTES):
        digest.update(chunk)
    return digest.hexdigest()


def _canonical_member_path(raw_name: object) -> str:
    if type(raw_name) is not str or not raw_name:
        raise ArchiveValidationError("tar member path must be a non-empty string")
    if "\\" in raw_name:
        raise ArchiveValidationError("tar member paths must use forward slashes")
    normalized = unicodedata.normalize("NFC", raw_name)
    if normalized.startswith("/") or _WINDOWS_DRIVE_ABSOLUTE.match(normalized):
        raise ArchiveValidationError("tar member path must be relative")
    segments = normalized.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ArchiveValidationError("tar member path contains an invalid segment")
    return normalized


def _validated_video_prefix(video_prefix: object) -> tuple[str, str]:
    if type(video_prefix) is not str:
        raise ArchiveValidationError("video_prefix must be a string")
    match = _VIDEO_PREFIX.fullmatch(video_prefix)
    if match is None:
        raise ArchiveValidationError("video_prefix does not use the canonical grammar")
    source_id = {"r2r": "R2R", "rxr": "RxR"}[match.group("source")]
    return video_prefix, source_id


def parse_frame_member(member_path: str) -> FrameMemberIdentity:
    """Parse one exact released RGB member path through the archive owner.

    Only the observed StreamVLN layouts are accepted: ``rgb/NNN.jpg`` for R2R
    (and annotated RxR trajectories), plus RxR ``rgb_images/NNN.jpg`` and
    ``rgb_images/NNNN.jpg`` scene-only members. The latter uses one-based
    zero-padded decimal names; zero-valued names are rejected.
    """

    canonical = _canonical_member_path(member_path)
    match = _FRAME_MEMBER_RGB.fullmatch(canonical)
    layout = "rgb"
    frame_width = 3
    if match is None:
        match = _FRAME_MEMBER_RGB_IMAGES.fullmatch(canonical)
        layout = "rgb_images"
        if match is not None:
            frame_width = len(match.group("frame"))
    if match is None:
        raise ArchiveValidationError("RGB file does not use the canonical member grammar")
    prefix = match.group("prefix")
    _, source_id = _validated_video_prefix(prefix)
    if layout == "rgb_images" and source_id != "RxR":
        # The restricted regex already enforces this; retain an explicit
        # ownership check so future grammar edits cannot weaken it silently.
        raise ArchiveValidationError("rgb_images layout is reserved for RxR members")
    frame_index = int(match.group("frame"))
    if frame_index < 1:
        raise ArchiveValidationError("RGB frame indices are one-based")
    if layout == "rgb_images":
        expected_width = 3 if frame_index <= 999 else 4
        if frame_width != expected_width:
            raise ArchiveValidationError(
                "rgb_images frame width does not use the released three/four-digit transition"
            )
    return FrameMemberIdentity(
        video_prefix=prefix,
        source_id=source_id,
        layout=layout,
        frame_width=frame_width,
        frame_index=frame_index,
    )


def frame_member_path(
    video_prefix: str,
    frame_index: int,
    *,
    layout: str | None = None,
    frame_width: int | None = None,
) -> str:
    """Derive an exact member path using the sole released-path owner.

    When no layout is supplied, the annotation-compatible ``rgb/NNN.jpg``
    spelling is used for both sources.  Callers that have observed a physical
    RxR scene-only layout may pass ``layout="rgb_images"`` and an explicit
    an explicit observed frame width.
    """

    try:
        prefix, source_id = _validated_video_prefix(video_prefix)
    except ArchiveValidationError as exc:
        raise FrameSequenceError(str(exc)) from exc
    if type(frame_index) is not int or frame_index < 1:
        raise FrameSequenceError("frame_index must be a positive integer")
    if layout is None:
        layout = "rgb"
    if layout == "rgb":
        if frame_width not in (None, 3):
            raise FrameSequenceError("rgb layout requires a three-digit frame width")
        if frame_index > 999:
            raise FrameSequenceError("rgb frame index exceeds the three-digit grammar")
        return f"{prefix}/rgb/{frame_index:03d}.jpg"
    if layout == "rgb_images":
        if source_id != "RxR":
            raise FrameSequenceError("rgb_images layout is reserved for RxR members")
        natural_width = 3 if frame_index <= 999 else len(str(frame_index))
        if natural_width > 4:
            raise FrameSequenceError("frame index exceeds the four-digit RGB grammar")
        if frame_width is None:
            frame_width = natural_width
        if frame_width != natural_width:
            raise FrameSequenceError(
                "rgb_images frame width must match the released three/four-digit transition"
            )
        return f"{prefix}/rgb_images/{frame_index:0{frame_width}d}.jpg"
    raise FrameSequenceError("unknown RGB frame layout")


def _validated_frame_member(member_path: str) -> tuple[str, int]:
    """Compatibility view returning only prefix and one-based frame index."""

    identity = parse_frame_member(member_path)
    return identity.video_prefix, identity.frame_index


def _verify_jpeg(payload: bytes) -> tuple[str, int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            rgb = image.convert("RGB")
            decoded_sha256 = hashlib.sha256(rgb.tobytes()).hexdigest()
            return decoded_sha256, rgb.width, rgb.height
    except (OSError, SyntaxError, ValueError) as exc:
        raise ArchiveValidationError("tar member is not a valid decodable JPEG") from exc


def _iter_tar_stream(
    fileobj: object,
    *,
    archive_path: str,
    include_payload: bool = False,
    workers: int = 1,
    selected_identities: set | None = None,
    stop_when_selected_found: bool = False,
) -> Iterator[VerifiedArchiveMember | VerifiedArchivePayload]:
    if type(workers) is not int or workers < 1:
        raise ArchiveValidationError('workers must be a positive integer')
    def verify(member_path, payload):
        decoded, width, height = _verify_jpeg(payload)
        member = VerifiedArchiveMember(archive_path=archive_path, member_path=member_path,
            compressed_sha256=hashlib.sha256(payload).hexdigest(),
            decoded_rgb_sha256=decoded, width=width, height=height)
        return VerifiedArchivePayload(member=member, jpeg_bytes=payload) if include_payload else member
    pending = deque()
    remaining = set(selected_identities) if stop_when_selected_found and selected_identities else None
    seen_directories: set[str] = set()
    seen_frame_masks: dict[str, int] = {}
    seen_frame_layouts: dict[str, str] = {}
    try:
        with ExitStack() as stack:
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers)) if workers > 1 else None
            archive = stack.enter_context(tarfile.open(fileobj=fileobj, mode="r|*"))
            for member in archive:
                member_path = _canonical_member_path(member.name)
                if member.isdir():
                    match = _FRAME_MEMBER.fullmatch(member_path)
                    frame_duplicate = False
                    if match is not None:
                        frame_index = int(match.group("frame"))
                        frame_duplicate = frame_index >= 1 and bool(
                            seen_frame_masks.get(match.group("prefix"), 0)
                            & (1 << (frame_index - 1))
                        )
                    if member_path in seen_directories or frame_duplicate:
                        raise ArchiveValidationError(
                            f"duplicate tar member: {member_path}"
                        )
                    seen_directories.add(member_path)
                    continue
                if not member.isfile():
                    raise ArchiveValidationError(
                        f"links and special tar members are forbidden: {member_path}"
                    )
                identity = parse_frame_member(member_path)
                prefix = identity.video_prefix
                frame_index = identity.frame_index
                # Width is an observed property, not a sequence-level layout:
                # Released rgb_images names naturally cross the three/four
                # digit boundary.
                current_layout = identity.layout
                previous_layout = seen_frame_layouts.get(prefix)
                if previous_layout is not None and previous_layout != current_layout:
                    raise ArchiveValidationError(
                        f"mixed RGB frame layouts for trajectory: {prefix}"
                    )
                seen_frame_layouts[prefix] = current_layout
                bit = 1 << (frame_index - 1)
                if member_path in seen_directories or seen_frame_masks.get(prefix, 0) & bit:
                    raise ArchiveValidationError(f"duplicate tar member: {member_path}")
                seen_frame_masks[prefix] = seen_frame_masks.get(prefix, 0) | bit
                if selected_identities is not None and (prefix, frame_index) not in selected_identities:
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ArchiveValidationError(f"cannot read tar member: {member_path}")
                payload = extracted.read()
                if len(payload) != member.size:
                    raise ArchiveValidationError(f"truncated tar member: {member_path}")
                if pool is not None:
                    pending.append(pool.submit(verify, member_path, payload))
                    if len(pending) >= 2 * workers:
                        yield pending.popleft().result()
                else:
                    yield verify(member_path, payload)
                if remaining is not None:
                    remaining.discard((prefix, frame_index))
                    if not remaining:
                        while pending:
                            yield pending.popleft().result()
                        return
            while pending:
                yield pending.popleft().result()
    except ArchiveValidationError:
        raise
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveValidationError(f"invalid streaming tar archive: {archive_path}") from exc


def iter_verified_members(
    archive: str | Path, expected_sha256: str
) -> Iterator[VerifiedArchiveMember]:
    """Stream a tar archive after verifying its complete byte identity."""
    archive_path = Path(archive)
    expected = _validated_sha256(expected_sha256, label="archive SHA-256")
    try:
        with archive_path.open("rb") as handle:
            if _stream_sha256(handle) != expected:
                raise ArchiveIdentityError(
                    f"archive SHA-256 mismatch: {archive_path}"
                )
            handle.seek(0)
            yield from _iter_tar_stream(handle, archive_path=str(archive_path))
    except ArchiveValidationError:
        raise
    except OSError as exc:
        raise ArchiveValidationError(f"cannot open archive: {archive_path}") from exc


def iter_verified_payloads(
    archive: str | Path, expected_sha256: str, *, workers: int = 1, selected_identities: set | None = None,
    verify_archive_sha: bool = True, stop_when_selected_found: bool = False
) -> Iterator[VerifiedArchivePayload]:
    """Stream exact verified JPEG bytes from a single RGB archive.

    The complete archive SHA is checked before the first payload is exposed;
    member validation then follows ``_iter_tar_stream`` exactly as for
    :func:`iter_verified_members`.
    """

    archive_path = Path(archive)
    expected = _validated_sha256(expected_sha256, label="archive SHA-256")
    try:
        with archive_path.open("rb") as handle:
            if verify_archive_sha and _stream_sha256(handle) != expected:
                raise ArchiveIdentityError(
                    f"archive SHA-256 mismatch: {archive_path}"
                )
            handle.seek(0)
            for item in _iter_tar_stream(
                handle,
                archive_path=str(archive_path),
                include_payload=True,
                workers=workers,
                selected_identities=selected_identities,
                stop_when_selected_found=stop_when_selected_found,
            ):
                # The flag above makes this invariant true while retaining a
                # single parser/validator for both public views.
                if not isinstance(item, VerifiedArchivePayload):
                    raise ArchiveValidationError("payload iterator received metadata-only member")
                yield item
    except ArchiveValidationError:
        raise
    except OSError as exc:
        raise ArchiveValidationError(f"cannot open archive: {archive_path}") from exc


class _ConcatenatedReader:
    """Read an ordered sequence of already-open binary streams as one stream."""

    def __init__(self, handles: Sequence[object]) -> None:
        self._handles = handles
        self._index = 0

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while self._index < len(self._handles) and (remaining != 0):
            request = -1 if size < 0 else remaining
            chunk = self._handles[self._index].read(request)
            if chunk:
                chunks.append(chunk)
                if size >= 0:
                    remaining -= len(chunk)
            else:
                self._index += 1
        return b"".join(chunks)


def iter_verified_members_from_parts(
    parts: Sequence[str | Path],
    expected_part_sha256s: Sequence[str],
    *,
    logical_archive_name: str,
) -> Iterator[VerifiedArchiveMember]:
    """Read exactly ``part0 || part1`` as one logical streaming tar archive."""
    try:
        part_paths = tuple(Path(part) for part in parts)
        expected_hashes = tuple(expected_part_sha256s)
    except (TypeError, ValueError) as exc:
        raise ArchiveValidationError("split archive inputs must be finite sequences") from exc
    if len(part_paths) != 2 or len(expected_hashes) != 2:
        raise ArchiveValidationError("split archive requires exactly part0 and part1")
    if not part_paths[0].name.endswith(".part0") or not part_paths[1].name.endswith(
        ".part1"
    ):
        raise ArchiveValidationError("split archive parts must be ordered part0 then part1")
    if type(logical_archive_name) is not str or not logical_archive_name:
        raise ArchiveValidationError("logical archive name must be a non-empty string")

    validated_hashes = tuple(
        _validated_sha256(raw_expected, label=f"part{index} SHA-256")
        for index, raw_expected in enumerate(expected_hashes)
    )

    try:
        with ExitStack() as stack:
            handles = tuple(stack.enter_context(path.open("rb")) for path in part_paths)
            for index, (path, handle, expected) in enumerate(
                zip(part_paths, handles, validated_hashes)
            ):
                if _stream_sha256(handle) != expected:
                    raise ArchiveIdentityError(
                        f"part{index} SHA-256 mismatch: {path}"
                    )
                handle.seek(0)
            concatenated = _ConcatenatedReader(handles)
            yield from _iter_tar_stream(
                concatenated, archive_path=logical_archive_name
            )
    except ArchiveValidationError:
        raise
    except OSError as exc:
        raise ArchiveValidationError("cannot open split archive inputs") from exc


def iter_verified_payloads_from_parts(
    parts: Sequence[str | Path],
    expected_part_sha256s: Sequence[str],
    *,
    logical_archive_name: str,
    workers: int = 1,
    selected_identities: set | None = None,
    verify_archive_sha: bool = True,
    stop_when_selected_found: bool = False,
) -> Iterator[VerifiedArchivePayload]:
    """Stream exact verified JPEG bytes from the logical ``part0 || part1`` archive."""

    try:
        part_paths = tuple(Path(part) for part in parts)
        expected_hashes = tuple(expected_part_sha256s)
    except (TypeError, ValueError) as exc:
        raise ArchiveValidationError("split archive inputs must be finite sequences") from exc
    if len(part_paths) != 2 or len(expected_hashes) != 2:
        raise ArchiveValidationError("split archive requires exactly part0 and part1")
    if not part_paths[0].name.endswith(".part0") or not part_paths[1].name.endswith(
        ".part1"
    ):
        raise ArchiveValidationError("split archive parts must be ordered part0 then part1")
    if type(logical_archive_name) is not str or not logical_archive_name:
        raise ArchiveValidationError("logical archive name must be a non-empty string")

    validated_hashes = tuple(
        _validated_sha256(raw_expected, label=f"part{index} SHA-256")
        for index, raw_expected in enumerate(expected_hashes)
    )

    try:
        with ExitStack() as stack:
            handles = tuple(stack.enter_context(path.open("rb")) for path in part_paths)
            for index, (path, handle, expected) in enumerate(
                zip(part_paths, handles, validated_hashes)
            ):
                if verify_archive_sha and _stream_sha256(handle) != expected:
                    raise ArchiveIdentityError(
                        f"part{index} SHA-256 mismatch: {path}"
                    )
                handle.seek(0)
            concatenated = _ConcatenatedReader(handles)
            for item in _iter_tar_stream(
                concatenated,
                archive_path=logical_archive_name,
                include_payload=True,
                workers=workers,
                selected_identities=selected_identities,
                stop_when_selected_found=stop_when_selected_found,
            ):
                if not isinstance(item, VerifiedArchivePayload):
                    raise ArchiveValidationError("payload iterator received metadata-only member")
                yield item
    except ArchiveValidationError:
        raise
    except OSError as exc:
        raise ArchiveValidationError("cannot open split archive inputs") from exc


# Explicitly named aliases make the RGB nature of the bytes clear to cache
# callers while keeping one implementation and one parser.
iter_verified_rgb_payloads = iter_verified_payloads
iter_verified_rgb_payloads_from_parts = iter_verified_payloads_from_parts


def ordered_frame_member_paths(
    member_paths: Iterable[str], *, video_prefix: str, expected_frame_count: int
) -> tuple[str, ...]:
    """Validate and order one trajectory's complete physical frame sequence.

    The returned paths preserve the exact spelling/layout observed in the
    archive while imposing the canonical one-based numeric order. A single
    trajectory cannot mix the two verified layouts, but ``rgb_images`` names
    may naturally vary from three to four digits. The
    returned paths preserve each observed spelling.
    """

    try:
        _, source_id = _validated_video_prefix(video_prefix)
    except ArchiveValidationError as exc:
        raise FrameSequenceError(str(exc)) from exc
    if type(expected_frame_count) is not int or expected_frame_count < 0:
        raise FrameSequenceError("expected_frame_count must be a non-negative integer")
    if expected_frame_count > 9999:
        raise FrameSequenceError("expected_frame_count exceeds the released frame grammar")
    try:
        actual = tuple(member_paths)
    except TypeError as exc:
        raise FrameSequenceError("member_paths must be iterable") from exc
    if any(type(path) is not str for path in actual):
        raise FrameSequenceError("member paths must be strings")
    if len(set(actual)) != len(actual):
        raise FrameSequenceError("frame sequence contains duplicate members")

    by_index: dict[int, str] = {}
    layout: str | None = None
    for path in actual:
        try:
            identity = parse_frame_member(path)
        except ArchiveValidationError as exc:
            raise FrameSequenceError("frame member uses an invalid released grammar") from exc
        if identity.video_prefix != video_prefix or identity.source_id != source_id:
            raise FrameSequenceError("frame member prefix/source disagrees with the trajectory")
        if layout is None:
            layout = identity.layout
        elif identity.layout != layout:
            raise FrameSequenceError("frame sequence mixes RGB layouts")
        if identity.frame_index in by_index:
            raise FrameSequenceError("frame sequence contains duplicate numeric indices")
        by_index[identity.frame_index] = path

    expected_indices = set(range(1, expected_frame_count + 1))
    if set(by_index) != expected_indices:
        raise FrameSequenceError("frame members do not exactly match the expected sequence")
    return tuple(by_index[index] for index in range(1, expected_frame_count + 1))


def validate_expected_frame_sequence(
    member_paths: Iterable[str], *, video_prefix: str, expected_frame_count: int
) -> None:
    """Require exactly the one-based frame set admitted by the path owner."""

    ordered_frame_member_paths(
        member_paths,
        video_prefix=video_prefix,
        expected_frame_count=expected_frame_count,
    )
