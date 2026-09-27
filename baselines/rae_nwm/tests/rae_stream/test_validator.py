import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from rae_stream.manifest import write_jsonl


def _make_row(root: Path, name: str, dataset: str, image_bytes: bytes, split: str) -> dict:
    scene = "TrainScene" if split == "train" else "SceneR2R"
    video = f"images/{scene}_{dataset.lower()}_000001"
    trajectory_name = f"{dataset.lower()}__{Path(video).name}"
    output = root / trajectory_name
    output.mkdir(parents=True)
    (output / "0.jpg").write_bytes(image_bytes)
    with (output / "traj_data.pkl").open("wb") as handle:
        pickle.dump(
            {"position": np.zeros((1, 2), dtype=np.float32), "yaw": np.zeros((1,), dtype=np.float32)},
            handle,
        )
    return {
        "status": "materialized",
        "trajectory_name": trajectory_name,
        "dataset": dataset,
        "video": video,
        "scene_token": scene,
        "split": split,
        "frame_count": 1,
        "measured_pose": False,
        "pose_source": "command-derived-se2",
        "output_dir": str(output),
        "annotation_row_sha256": "0" * 64,
        "command_actions": [-1],
        "source_revision": "rev",
        "source_index": 0,
        "source_root": str(root),
        "production_eligible": True,
    }


def test_validator_decodes_a_bounded_deterministic_image_sample(tmp_path: Path):
    from PIL import Image
    from scripts.validate_rae_stream import validate_dataset

    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid_path, format="JPEG")
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = [
        _make_row(data_root, "r2r__a", "R2R", valid_path.read_bytes(), "train"),
        _make_row(data_root, "rxr__b", "RxR", valid_path.read_bytes(), "test"),
    ]
    split_root = tmp_path / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text("r2r__TrainScene_r2r_000001\n", encoding="utf-8")
    (split_root / "test" / "traj_names.txt").write_text("rxr__SceneR2R_rxr_000001\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    receipt = validate_dataset(
        data_root=data_root,
        split_root=split_root,
        manifest_path=manifest,
        expected_revision="rev",
        image_decode_samples=2,
    )
    assert receipt["image_decode_samples"] == 2

    (data_root / "r2r__TrainScene_r2r_000001" / "0.jpg").write_bytes(b"not-a-jpeg")
    with pytest.raises(ValueError, match="JPEG decode"):
        validate_dataset(
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            expected_revision="rev",
            image_decode_samples=2,
        )


def test_validator_rejects_manifest_with_converter_bytes_not_matching_official_root(
    tmp_path: Path, monkeypatch
):
    """The formal gate must bind each row to the exact converter source bytes."""

    from PIL import Image
    import scripts.validate_rae_stream as validator_module

    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid_path, format="JPEG")
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = [
        _make_row(data_root, "r2r__a", "R2R", valid_path.read_bytes(), "train"),
        _make_row(data_root, "rxr__b", "RxR", valid_path.read_bytes(), "test"),
    ]
    # Deliberately use an incorrect identity.  The current validator does not
    # yet check this field, so this assertion is RED until the gate is added.
    for row in rows:
        row["converter_version"] = "rae_stream_converter_v2"
        row["converter_sha256"] = "0" * 64
    split_root = tmp_path / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text("r2r__TrainScene_r2r_000001\n", encoding="utf-8")
    (split_root / "test" / "traj_names.txt").write_text("rxr__SceneR2R_rxr_000001\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    official_root = tmp_path / "official"
    (official_root / "scripts").mkdir(parents=True)
    (official_root / "scripts" / "prepare_rae_stream.py").write_text(
        "# converter bytes\n", encoding="utf-8"
    )
    monkeypatch.setattr(validator_module, "verify_upstream_source", lambda *args, **kwargs: {})
    expected_converter_sha = validator_module.sha256_file(
        official_root / "scripts" / "prepare_rae_stream.py"
    )
    monkeypatch.setattr(
        validator_module,
        "verify_converter_source",
        lambda root: expected_converter_sha,
    )
    monkeypatch.setattr(
        validator_module,
        "converter_closure_digest_map",
        lambda root: {"scripts/prepare_rae_stream.py": expected_converter_sha},
    )
    monkeypatch.setattr(
        validator_module,
        "compute_converter_bundle_sha256",
        lambda value: "bundle-test",
    )

    with pytest.raises(ValueError, match="converter"):
        validator_module.validate_dataset(
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            official_root=official_root,
            expected_revision="rev",
        )


def test_formal_validator_requires_prepare_receipt(tmp_path: Path):
    """A production-shaped manifest cannot bypass the converter/source receipt gate."""

    from PIL import Image
    from scripts.validate_rae_stream import validate_dataset

    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid_path, format="JPEG")
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = [
        _make_row(data_root, "r2r__a", "R2R", valid_path.read_bytes(), "train"),
        _make_row(data_root, "rxr__b", "RxR", valid_path.read_bytes(), "test"),
    ]
    split_root = tmp_path / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text("r2r__TrainScene_r2r_000001\n", encoding="utf-8")
    (split_root / "test" / "traj_names.txt").write_text("rxr__SceneR2R_rxr_000001\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)

    with pytest.raises(ValueError, match="prepare receipt"):
        validate_dataset(
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            official_root=tmp_path,
            expected_revision="rev",
            require_production=True,
        )


def test_validator_recomputes_command_derived_pose(tmp_path: Path):
    """Finite arbitrary pose arrays must not pass as command-derived labels."""

    from PIL import Image
    from scripts.validate_rae_stream import validate_dataset

    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid_path, format="JPEG")
    data_root = tmp_path / "data"
    data_root.mkdir()
    row = _make_row(data_root, "r2r__a", "R2R", valid_path.read_bytes(), "train")
    # Keep a second split row so the normal split-leakage gate remains active.
    other = _make_row(data_root, "rxr__b", "RxR", valid_path.read_bytes(), "test")
    with (data_root / "r2r__TrainScene_r2r_000001" / "traj_data.pkl").open("wb") as handle:
        pickle.dump(
            {"position": np.ones((1, 2), dtype=np.float32), "yaw": np.zeros((1,), dtype=np.float32)},
            handle,
        )
    rows = [row, other]
    split_root = tmp_path / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text("r2r__TrainScene_r2r_000001\n", encoding="utf-8")
    (split_root / "test" / "traj_names.txt").write_text("rxr__SceneR2R_rxr_000001\n", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match="command-derived pose"):
        validate_dataset(
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            expected_revision="rev",
        )


def test_validator_requires_loader_canonical_direct_child_output(tmp_path: Path):
    """A manifest row must name exactly data_root/trajectory_name.

    The upstream dataset loader reconstructs this path from the split list and
    trajectory name.  Accepting an arbitrary nested output would validate one
    tree while training reads another.
    """

    from PIL import Image
    from scripts.validate_rae_stream import validate_dataset

    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid_path, format="JPEG")
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = [
        _make_row(data_root, "r2r__a", "R2R", valid_path.read_bytes(), "train"),
        _make_row(data_root, "rxr__b", "RxR", valid_path.read_bytes(), "test"),
    ]
    row = rows[0]
    canonical = Path(row["output_dir"])
    nested = data_root / "nested" / canonical.name
    nested.parent.mkdir()
    canonical.rename(nested)
    row["output_dir"] = str(nested)
    split_root = tmp_path / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text(
        f"{row['trajectory_name']}\n", encoding="utf-8"
    )
    (split_root / "test" / "traj_names.txt").write_text(
        f"{rows[1]['trajectory_name']}\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match="canonical|direct child|output path"):
        validate_dataset(
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            expected_revision="rev",
        )


def test_validator_rejects_pickle_symlink(tmp_path: Path):
    from PIL import Image
    from scripts.validate_rae_stream import validate_dataset

    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid_path, format="JPEG")
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = [
        _make_row(data_root, "r2r__a", "R2R", valid_path.read_bytes(), "train"),
        _make_row(data_root, "rxr__b", "RxR", valid_path.read_bytes(), "test"),
    ]
    pickle_path = data_root / rows[0]["trajectory_name"] / "traj_data.pkl"
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(pickle_path.read_bytes())
    pickle_path.unlink()
    pickle_path.symlink_to(outside)
    split_root = tmp_path / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text(
        f"{rows[0]['trajectory_name']}\n", encoding="utf-8"
    )
    (split_root / "test" / "traj_names.txt").write_text(
        f"{rows[1]['trajectory_name']}\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match="pickle|symlink"):
        validate_dataset(
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            expected_revision="rev",
        )


def test_validator_rejects_duplicate_or_noncanonical_split_listing(tmp_path: Path):
    from PIL import Image
    from scripts.validate_rae_stream import validate_dataset

    valid_path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(valid_path, format="JPEG")
    data_root = tmp_path / "data"
    data_root.mkdir()
    rows = [
        _make_row(data_root, "r2r__a", "R2R", valid_path.read_bytes(), "train"),
        _make_row(data_root, "rxr__b", "RxR", valid_path.read_bytes(), "test"),
    ]
    split_root = tmp_path / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text(
        f"{rows[0]['trajectory_name']}\n{rows[0]['trajectory_name']}\n", encoding="utf-8"
    )
    (split_root / "test" / "traj_names.txt").write_text(
        f"{rows[1]['trajectory_name']}\n", encoding="utf-8"
    )
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, rows)
    with pytest.raises(ValueError, match="duplicate|canonical|traj_names"):
        validate_dataset(
            data_root=data_root,
            split_root=split_root,
            manifest_path=manifest,
            expected_revision="rev",
        )


def test_readonly_gate_rejects_writable_required_path(tmp_path: Path):
    from scripts.validate_rae_stream import _assert_readonly_path

    path = tmp_path / "frame.jpg"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="writable|readonly"):
        _assert_readonly_path(path, label="frame")


def test_formal_layout_keeps_official_split_cache_directories_writable(
    tmp_path: Path, monkeypatch
):
    """Only split list bytes are frozen; upstream may write index caches beside them."""

    import scripts.validate_rae_stream as validator

    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    split_root = tmp_path / "split_parent" / "splits"
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    # The split containers intentionally remain writable for official
    # dataset_dist_*.pkl cache files; the list files themselves are frozen by
    # the caller/receipt gate.
    checked: list[Path] = []
    monkeypatch.setattr(
        validator,
        "_assert_readonly_path",
        lambda path, *, label: checked.append(Path(path)),
    )
    validator._assert_formal_layout_readonly(data_root, split_root, manifest)
    assert checked == [data_root, manifest, data_root.parent]
    assert split_root not in checked
    assert split_root.parent not in checked


def test_formal_validator_requires_external_prepare_receipt_sha(tmp_path: Path):
    from scripts.validate_rae_stream import validate_dataset

    with pytest.raises(ValueError, match="prepare receipt SHA"):
        validate_dataset(
            data_root=tmp_path / "data",
            split_root=tmp_path / "splits",
            manifest_path=tmp_path / "manifest.jsonl",
            official_root=tmp_path,
            expected_revision="rev",
            require_production=True,
            prepare_receipt_path=tmp_path / "prepare.json",
        )


def test_formal_validator_requires_external_code_authority(tmp_path: Path):
    from scripts.validate_rae_stream import validate_dataset

    with pytest.raises(ValueError, match="code authority"):
        validate_dataset(
            data_root=tmp_path / "data",
            split_root=tmp_path / "splits",
            manifest_path=tmp_path / "manifest.jsonl",
            official_root=tmp_path,
            expected_revision="rev",
            require_production=True,
            prepare_receipt_path=tmp_path / "prepare.json",
            prepare_receipt_sha256="a" * 64,
        )


def test_formal_validator_allows_authority_outside_official_checkout(tmp_path: Path):
    """The external SHA authority must not share the mutable source tree."""

    from scripts.validate_rae_stream import validate_dataset

    official = tmp_path / "official"
    official.mkdir()
    prepare = official / "receipts" / "prepare.json"
    prepare.parent.mkdir()
    prepare.write_text("{}", encoding="utf-8")
    authority = tmp_path / "trusted" / "authority.json"
    authority.parent.mkdir()
    authority.write_text("{}", encoding="utf-8")

    # Path admission happens before the missing data-root check.  Reaching
    # FileNotFoundError proves the external authority was not incorrectly
    # forced through the checkout-owned path gate.
    with pytest.raises(FileNotFoundError):
        validate_dataset(
            data_root=official / "data" / "rae_stream",
            split_root=official / "data_splits" / "rae_stream",
            manifest_path=official / "rae_stream_manifest.jsonl",
            official_root=official,
            require_production=True,
            prepare_receipt_path=prepare,
            prepare_receipt_sha256="a" * 64,
            authority_path=authority,
            authority_sha256="b" * 64,
        )


def test_prepare_receipt_identity_helper_checks_converter_closure():
    from scripts.validate_rae_stream import _verify_formal_prepare_receipt_identity

    receipt = {
        "converter_sha256": "a" * 64,
        "converter_source_sha256": {"scripts/prepare_rae_stream.py": "a" * 64},
        "converter_bundle_sha256": "b" * 64,
        "immutable_source_trees": {"R2R": True, "RxR": True},
    }
    result = _verify_formal_prepare_receipt_identity(
        receipt,
        expected_converter_sha="a" * 64,
        expected_converter_source_sha256={"scripts/prepare_rae_stream.py": "a" * 64},
        expected_converter_bundle_sha256="b" * 64,
    )
    assert result is None


def test_formal_source_parent_identity_requires_frozen_containers(tmp_path: Path, monkeypatch):
    from scripts.validate_rae_stream import _verify_formal_source_parent_identity

    source_a = tmp_path / "selected" / "r2r"
    source_b = tmp_path / "selected" / "rxr"
    source_a.mkdir(parents=True)
    source_b.mkdir()
    checked: list[Path] = []
    import scripts.validate_rae_stream as validator

    monkeypatch.setattr(
        validator,
        "_assert_readonly_path",
        lambda path, *, label: checked.append(Path(path)),
    )
    _verify_formal_source_parent_identity(
        {"immutable_source_parent_dirs": {str((tmp_path / "selected").resolve()): True}},
        [source_a, source_b],
    )
    assert checked == [tmp_path / "selected"]


def test_immutable_path_identity_detects_container_replacement(tmp_path: Path):
    from scripts.validate_rae_stream import _verify_immutable_path_identities
    from scripts.prepare_rae_stream import _immutable_path_identity

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    receipt = {"immutable_path_identities": {str(snapshot): _immutable_path_identity(snapshot)}}
    _verify_immutable_path_identities(receipt, tmp_path)
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    snapshot.rename(tmp_path / "old_snapshot")
    replacement.rename(snapshot)
    with pytest.raises(ValueError, match="identity"):
        _verify_immutable_path_identities(receipt, tmp_path)


def test_source_frame_digest_helper_recomputes_source_bytes(tmp_path: Path):
    from scripts.validate_rae_stream import _verify_source_frame_digest
    from rae_stream.manifest import frame_sequence_sha256

    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source-bytes")
    output.write_bytes(b"different-bytes")
    source_digest = frame_sequence_sha256([source])
    with pytest.raises(ValueError, match="source frame digest"):
        _verify_source_frame_digest(
            [output],
            [source],
            declared_output_digest=frame_sequence_sha256([output]),
            declared_source_digest="0" * 64,
        )
    assert _verify_source_frame_digest(
        [output],
        [source],
        declared_output_digest=frame_sequence_sha256([output]),
        declared_source_digest=source_digest,
    ) is None
