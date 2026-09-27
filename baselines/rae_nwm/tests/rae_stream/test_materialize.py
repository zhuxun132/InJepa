import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from rae_stream.annotations import parse_annotation_row
from rae_stream.materialize import materialize_record, split_for_scene
from rae_stream.manifest import frame_sequence_sha256, read_jsonl, write_jsonl


def _make_source(root: Path, actions=(-1, 1, 2, 1)):
    video = "images/SceneA_r2r_000001"
    frame_dir = root / video / "rgb"
    frame_dir.mkdir(parents=True)
    # Deliberately non-contiguous names: output must become 0.jpg..N-1.jpg.
    for name in ("10.jpg", "2.jpg", "001.jpg", "7.jpg")[: len(actions)]:
        (frame_dir / name).write_bytes(b"jpeg-" + name.encode())
    return parse_annotation_row({"video": video, "actions": list(actions)}, "R2R")


def test_materialize_short_trajectory_is_excluded_without_lowering_horizon(tmp_path: Path):
    record = _make_source(tmp_path / "src")
    result = materialize_record(record, tmp_path / "src", tmp_path / "data", min_length=5)
    assert result.status == "excluded_short"
    assert not (tmp_path / "data").exists()


def test_materialize_writes_numeric_pickle_and_safe_links(tmp_path: Path):
    record = _make_source(tmp_path / "src")
    result = materialize_record(record, tmp_path / "src", tmp_path / "data", min_length=2)
    assert result.status == "materialized"
    out = tmp_path / "data" / result.trajectory_name
    assert [p.name for p in sorted(out.glob("*.jpg"), key=lambda p: int(p.stem))] == [
        "0.jpg", "1.jpg", "2.jpg", "3.jpg"
    ]
    with (out / "traj_data.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    assert set(payload) == {"position", "yaw"}
    assert payload["position"].dtype == np.float32
    assert payload["position"].shape == (4, 2)
    assert payload["yaw"].shape == (4,)
    assert result.measured_pose is False
    for path in out.glob("*.jpg"):
        if path.is_symlink():
            assert path.resolve().is_relative_to((tmp_path / "src").resolve())
        else:
            # Hardlinks share the source inode without duplicating JPEG bytes.
            assert path.stat().st_nlink >= 2


def test_split_is_deterministic_and_scene_stays_together():
    assert split_for_scene("SceneA") == split_for_scene("SceneA")
    assert split_for_scene("SceneA") in {"train", "test"}


def test_manifest_writer_is_canonical_and_atomic(tmp_path: Path):
    path = tmp_path / "manifest.jsonl"
    rows = [{"b": 2, "a": 1}, {"status": "ok", "n": 3}]
    write_jsonl(path, rows)
    assert not (tmp_path / "manifest.jsonl.tmp").exists()
    assert read_jsonl(path) == rows
    assert path.read_text(encoding="utf-8").splitlines()[0] == '{"a":1,"b":2}'


def test_frame_sequence_digest_binds_every_frame_and_order(tmp_path: Path):
    first = tmp_path / "0.jpg"
    second = tmp_path / "1.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    digest = frame_sequence_sha256([first, second])
    assert digest == frame_sequence_sha256([first, second])
    assert digest != frame_sequence_sha256([second, first])
    second.write_bytes(b"changed")
    assert digest != frame_sequence_sha256([first, second])
