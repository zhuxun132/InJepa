import json
from pathlib import Path

import pytest

from rae_stream.annotations import (
    find_frame_dir,
    numeric_frame_paths,
    parse_annotation_row,
    validate_alignment,
)


def _row(actions=(-1, 1, 2)):
    return {"video": "images/SceneA_r2r_000001", "actions": list(actions), "id": 1}


def test_parse_preserves_bos_and_uses_scene_token():
    record = parse_annotation_row(_row(), dataset="R2R", source_index=7)
    assert record.actions == (-1, 1, 2)
    assert record.scene_token == "SceneA"
    assert record.dataset == "R2R"
    assert len(record.row_sha256) == 64


def test_stop_and_misplaced_bos_are_rejected():
    with pytest.raises(ValueError, match="STOP"):
        parse_annotation_row(_row((-1, 1, 0)), dataset="R2R")
    with pytest.raises(ValueError, match="first action"):
        parse_annotation_row(_row((1, 2)), dataset="R2R")
    with pytest.raises(ValueError, match="BOS"):
        parse_annotation_row(_row((-1, 1, -1)), dataset="R2R")


def test_numeric_sort_and_frame_alignment(tmp_path: Path):
    frame_dir = tmp_path / "images" / "SceneA_r2r_000001" / "rgb"
    frame_dir.mkdir(parents=True)
    for name in ("10.jpg", "2.jpg", "001.jpg"):
        (frame_dir / name).write_bytes(b"jpeg")
    paths = numeric_frame_paths(frame_dir)
    assert [p.stem for p in paths] == ["001", "2", "10"]
    record = parse_annotation_row(_row(), dataset="R2R")
    validate_alignment(record, paths)


def test_frame_dir_discovery_accepts_rgb_images(tmp_path: Path):
    root = tmp_path / "R2R"
    frame_dir = root / "images" / "SceneA_r2r_000001" / "rgb_images"
    frame_dir.mkdir(parents=True)
    (frame_dir / "000.jpg").write_bytes(b"jpeg")
    found = find_frame_dir(root, "images/SceneA_r2r_000001")
    assert found == frame_dir


def test_alignment_catches_action_frame_shift(tmp_path: Path):
    frame_dir = tmp_path / "rgb"
    frame_dir.mkdir()
    for i in range(2):
        (frame_dir / f"{i}.jpg").write_bytes(b"jpeg")
    record = parse_annotation_row(_row((-1, 1, 2)), dataset="R2R")
    with pytest.raises(ValueError, match=r"len\(actions\)"):
        validate_alignment(record, numeric_frame_paths(frame_dir))
