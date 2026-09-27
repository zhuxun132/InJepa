import json
import tarfile
from pathlib import Path

import pytest

import scripts.prepare_rae_stream as prepare_module
from scripts.prepare_rae_stream import SourceSpec, prepare_dataset
from scripts.prepare_rae_stream import (
    _freeze_split_lists,
    _make_directory_readonly,
    _make_tree_readonly,
)
from rae_stream.config_guard import converter_bundle_sha256, converter_closure_digest_map
from rae_stream.manifest import read_jsonl


def _make_archive(tmp_path: Path, *, frame_count: int = 68) -> tuple[Path, Path]:
    annotation = tmp_path / "annotations.json"
    video = "images/TinyScene_r2r_000001"
    actions = [-1] + [1] * (frame_count - 1)
    annotation.write_text(json.dumps([{"video": video, "actions": actions}]), encoding="utf-8")
    archive = tmp_path / "images.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        # Include an unrelated trajectory so matching does not rely on the
        # first member or on extracting the complete archive.
        for other in ("images/Other_r2r_000002/rgb/000.jpg",):
            info = tarfile.TarInfo(other)
            payload = b"other"
            info.size = len(payload)
            tar.addfile(info, __import__("io").BytesIO(payload))
        for index in range(len(actions)):
            name = f"{video}/rgb/{index:03d}.jpg"
            info = tarfile.TarInfo(name)
            payload = f"frame-{index}".encode()
            info.size = len(payload)
            tar.addfile(info, __import__("io").BytesIO(payload))
    return annotation, archive


def test_archive_only_conversion_extracts_selected_members_once(tmp_path: Path):
    annotation, archive = _make_archive(tmp_path)
    output = tmp_path / "out"
    receipt = prepare_dataset(
        [
            SourceSpec(
                "R2R",
                annotation,
                tmp_path / "absent-source-root",
                source_revision="test-rev",
                archive_path=archive,
            )
        ],
        data_root=output / "data" / "rae_stream",
        split_root=output / "splits" / "rae_stream",
        manifest_path=output / "manifest.jsonl",
        receipt_path=output / "receipt.json",
        min_length=68,
        allow_empty_test=True,
        production_eligible=False,
    )
    assert receipt["counts"]["materialized"] == 1
    rows = read_jsonl(output / "manifest.jsonl")
    row = next(item for item in rows if item["status"] == "materialized")
    assert row["source_mode"] == "tar-selected"
    trajectory = Path(row["output_dir"])
    assert (trajectory / "0.jpg").read_bytes() == b"frame-0"
    assert (trajectory / "67.jpg").read_bytes() == b"frame-67"


def test_concatenated_archive_parts_are_streamed_without_joining_payload(tmp_path: Path):
    annotation, archive = _make_archive(tmp_path)
    payload = archive.read_bytes()
    cut = max(1, len(payload) // 2)
    part0 = tmp_path / "images.tar.gz.part0"
    part1 = tmp_path / "images.tar.gz.part1"
    part0.write_bytes(payload[:cut])
    part1.write_bytes(payload[cut:])
    output = tmp_path / "split-out"
    receipt = prepare_dataset(
        [
            SourceSpec(
                "R2R",
                annotation,
                tmp_path / "absent-source-root",
                source_revision="test-rev",
                archive_paths=(part0, part1),
            )
        ],
        data_root=output / "data" / "rae_stream",
        split_root=output / "splits" / "rae_stream",
        manifest_path=output / "manifest.jsonl",
        receipt_path=output / "receipt.json",
        min_length=68,
        allow_empty_test=True,
        production_eligible=False,
    )
    assert receipt["sources"][0]["archive_parts"] == [str(part0.resolve()), str(part1.resolve())]
    row = next(item for item in read_jsonl(output / "manifest.jsonl") if item["status"] == "materialized")
    assert Path(row["output_dir"]) .joinpath("67.jpg").read_bytes() == b"frame-67"


def test_archive_extraction_does_not_materialize_short_trajectories(tmp_path: Path):
    long_video = "images/LongScene_r2r_000001"
    short_video = "images/ShortScene_r2r_000002"
    annotation = tmp_path / "annotations.json"
    annotation.write_text(
        json.dumps(
            [
                {"video": long_video, "actions": [-1] + [1] * 67},
                {"video": short_video, "actions": [-1, 1, 2]},
            ]
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "images.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for video, count in ((long_video, 68), (short_video, 3)):
            for index in range(count):
                name = f"{video}/rgb/{index:03d}.jpg"
                info = tarfile.TarInfo(name)
                payload = f"{video}-{index}".encode()
                info.size = len(payload)
                tar.addfile(info, __import__("io").BytesIO(payload))
    output = tmp_path / "short-out"
    receipt = prepare_dataset(
        [
            SourceSpec(
                "R2R",
                annotation,
                tmp_path / "absent-source-root",
                source_revision="test-rev",
                archive_path=archive,
            )
        ],
        data_root=output / "data" / "rae_stream",
        split_root=output / "splits" / "rae_stream",
        manifest_path=output / "manifest.jsonl",
        receipt_path=output / "receipt.json",
        min_length=68,
        allow_empty_test=True,
        production_eligible=False,
    )
    assert receipt["counts"]["excluded_short"] == 1
    assert receipt["archive_frames_counted"] == 71
    assert receipt["archive_frames_extracted"] == 68
    assert receipt["archive_short_frames_skipped"] == 3
    assert not list((output / "data" / "tar_selected_sources").glob("**/ShortScene*"))


def test_archive_extraction_uses_periodic_frame_fsync_not_one_call_per_rgb(
    tmp_path: Path, monkeypatch
):
    """Selected-frame extraction must not turn millions of links into sync calls."""

    # Cross two default 256-frame checkpoints so the periodic branch is
    # exercised, rather than merely asserting that a short fixture has no
    # per-frame barrier.
    annotation, archive = _make_archive(tmp_path, frame_count=512)
    fsync_calls: list[int] = []
    monkeypatch.setattr(prepare_module.os, "fsync", lambda fd: fsync_calls.append(int(fd)))
    output = tmp_path / "fsync-out"
    receipt = prepare_dataset(
        [
            SourceSpec(
                "R2R",
                annotation,
                tmp_path / "absent-source-root",
                source_revision="test-rev",
                archive_path=archive,
            )
        ],
        data_root=output / "data" / "rae_stream",
        split_root=output / "splits" / "rae_stream",
        manifest_path=output / "manifest.jsonl",
        receipt_path=output / "receipt.json",
        min_length=68,
        allow_empty_test=True,
        production_eligible=False,
    )
    # Two frame barriers (256 and 512), one pickle, one manifest, and one
    # receipt.  A call per selected RGB would make the full released corpus
    # needlessly hours slower.
    assert len(fsync_calls) == 5
    assert receipt["frame_fsync_interval"] == 256
    converter_path = Path(prepare_module.__file__).resolve()
    assert receipt["converter_sha256"] == prepare_module.sha256_file(converter_path)
    expected_closure = converter_closure_digest_map(converter_path.parents[1])
    assert receipt["converter_source_sha256"] == expected_closure
    assert receipt["converter_bundle_sha256"] == converter_bundle_sha256(expected_closure)
    row = read_jsonl(output / "manifest.jsonl")[0]
    assert row["converter_sha256"] == receipt["converter_sha256"]
    assert row["converter_bundle_sha256"] == receipt["converter_bundle_sha256"]


def test_archive_extraction_rejects_nonpositive_frame_fsync_interval(tmp_path: Path):
    annotation, archive = _make_archive(tmp_path)
    with pytest.raises(ValueError, match="frame_fsync_interval"):
        prepare_dataset(
            [
                SourceSpec(
                    "R2R",
                    annotation,
                    tmp_path / "absent-source-root",
                    source_revision="test-rev",
                    archive_path=archive,
                )
            ],
            data_root=tmp_path / "out" / "data",
            split_root=tmp_path / "out" / "splits",
            manifest_path=tmp_path / "out" / "manifest.jsonl",
            receipt_path=tmp_path / "out" / "receipt.json",
            min_length=68,
            allow_empty_test=True,
            production_eligible=False,
            frame_fsync_interval=0,
        )


def test_formal_prepare_rejects_rgb_copy_mode(tmp_path: Path):
    """The released archive must not be duplicated into a second RGB tree."""

    annotation, archive = _make_archive(tmp_path)
    with pytest.raises(ValueError, match="copy_images|copy"):
        prepare_dataset(
            [
                SourceSpec(
                    "R2R",
                    annotation,
                    tmp_path / "absent-source-root",
                    source_revision=prepare_module.STREAMVLN_REVISION,
                    archive_path=archive,
                )
            ],
            data_root=tmp_path / "out" / "data",
            split_root=tmp_path / "out" / "splits",
            manifest_path=tmp_path / "out" / "manifest.jsonl",
            receipt_path=tmp_path / "out" / "receipt.json",
            min_length=68,
            allow_empty_test=False,
            production_eligible=True,
            copy_images=True,
        )


def test_readonly_data_tree_helper_removes_write_bits(tmp_path: Path):
    root = tmp_path / "tree"
    child = root / "nested" / "frame.jpg"
    child.parent.mkdir(parents=True)
    child.write_bytes(b"frame")
    _make_tree_readonly(root)
    assert child.stat().st_mode & 0o222 == 0
    assert root.stat().st_mode & 0o222 == 0


def test_readonly_directory_helper_removes_only_container_write_bits(tmp_path: Path):
    from scripts import prepare_rae_stream

    root = tmp_path / "container"
    child = root / "child.txt"
    root.mkdir()
    child.write_bytes(b"x")
    _make_directory_readonly(root)
    assert root.stat().st_mode & 0o222 == 0
    assert child.stat().st_mode & 0o222 != 0


def test_split_freeze_keeps_official_index_cache_writable(tmp_path: Path):
    root = tmp_path / "splits"
    for split in ("train", "test"):
        directory = root / split
        directory.mkdir(parents=True)
        (directory / "traj_names.txt").write_text("one\n", encoding="utf-8")
    _freeze_split_lists(root)
    assert root.stat().st_mode & 0o222 != 0
    assert (root / "train").stat().st_mode & 0o222 != 0
    assert (root / "train" / "traj_names.txt").stat().st_mode & 0o222 == 0


def test_source_container_freeze_removes_replacement_permission(tmp_path: Path):
    """The selected-source container is frozen after tar extraction."""

    from scripts.prepare_rae_stream import _freeze_source_parent_dirs

    container = tmp_path / "tar_selected_sources"
    source = container / "rae-stream-r2r"
    source.mkdir(parents=True)
    (source / "frame.jpg").write_bytes(b"frame")
    flags = _freeze_source_parent_dirs([source])
    assert container.stat().st_mode & 0o222 == 0
    assert flags == {str(container.resolve()): True}
    # Freezing the container does not needlessly rewrite the source payload;
    # the production path freezes that tree separately.
    assert source.stat().st_mode & 0o222 != 0


def test_immutable_path_identity_records_inode_and_mode(tmp_path: Path):
    from scripts.prepare_rae_stream import _immutable_path_identity

    path = tmp_path / "snapshot"
    path.mkdir()
    identity = _immutable_path_identity(path)
    assert identity["st_dev"] == path.stat().st_dev
    assert identity["st_ino"] == path.stat().st_ino
    assert identity["mode"] == path.stat().st_mode & 0o7777


def test_immutable_path_identity_rejects_symlink(tmp_path: Path):
    from scripts.prepare_rae_stream import _immutable_path_identity

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="regular path"):
        _immutable_path_identity(link)
