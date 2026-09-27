from pathlib import Path

import pytest

from rae_stream.config_guard import (
    OFFICIAL_CHECKOUT_PROFILE,
    STREAMVLN_ARCHIVE_SHA256,
    assert_data_overlay,
    assert_fresh_training_output,
    assert_no_resume_settings,
    assert_training_profile,
    paper_batch_mapping,
    runtime_paper_config,
    assert_runtime_paper_config,
    changed_keys,
    _git_path_records,
    load_yaml_mapping,
    is_allowed_untracked_path,
    verify_converter_source,
)


def test_r2r_archive_identity_matches_fixed_release_lfs_oid():
    assert STREAMVLN_ARCHIVE_SHA256["R2R"] == (
        "9dccce7260f8db486b99cb0dcf5f443604a4c7dca3d980d367ffc516da06c59a",
    )


def test_formal_data_outputs_are_allowed_but_source_shadows_are_not():
    """The source guard admits only the exact generated data-only files."""

    assert is_allowed_untracked_path("data_splits/rae_stream/train/traj_names.txt")
    assert is_allowed_untracked_path("data_splits/rae_stream/test/traj_names.txt")
    assert is_allowed_untracked_path("rae_stream_manifest.jsonl")
    for path in (
        "data_splits/rae_stream/train/evil.py",
        "data_splits/rae_stream/other.txt",
        "rae_stream_manifest.jsonl.bak",
        "rae_stream_manifest.jsonl ",
        "rae_stream_manifest.jsonl\t",
        " data_splits/rae_stream/train/traj_names.txt",
        "data/rae_stream/injected.py",
        "models.py",
    ):
        assert not is_allowed_untracked_path(path)


def test_canonical_runtime_reservation_lock_is_allowed():
    """The launcher's own canonical lock must not be treated as source injection."""

    assert is_allowed_untracked_path(".rae_stream_gpu.lock")


def test_git_path_records_do_not_split_unicode_separators():
    """Only Git's NUL/newline record terminators may delimit a path."""

    unicode_path = "rae_stream_manifest.jsonl\u2028\u2029\u0085"
    assert _git_path_records(unicode_path + "\nrae_stream/adapter.py\n") == [
        unicode_path,
        "rae_stream/adapter.py",
    ]
    assert _git_path_records(unicode_path + "\0rae_stream/adapter.py\0") == [
        unicode_path,
        "rae_stream/adapter.py",
    ]


def test_checkout_profile_is_explicit_and_exact():
    base = {
        "batch_size": 96,
        "lr": 2e-4,
        "final_lr": 2e-6,
        "lr_schedule": "linear",
        "weight_decay": 0.0,
        "num_workers": 16,
        "run_name": "raenwm",
        "datasets": {"recon": {"data_folder": "data/recon"}},
    }
    resolved = dict(base)
    resolved["run_name"] = "rae_stream"
    resolved["datasets"] = {"rae_stream": {"data_folder": "data/rae_stream"}}
    assert_training_profile(resolved, epochs=50)
    assert OFFICIAL_CHECKOUT_PROFILE["batch_size"] == 96
    assert OFFICIAL_CHECKOUT_PROFILE["global_batch_size"] == 96


def test_training_override_is_rejected():
    with pytest.raises(ValueError, match="official-checkout-config"):
        assert_training_profile({"batch_size": 4, "lr": 1e-4, "final_lr": 1e-6, "num_workers": 16}, epochs=300)


def test_stale_checkout_values_are_not_admitted_as_paper_profile():
    with pytest.raises(ValueError, match="paper-reproduction"):
        assert_training_profile(
            {
                "batch_size": 4,
                "lr": 1e-4,
                "final_lr": 1e-6,
                "lr_schedule": "linear",
                "weight_decay": 0.0,
                "num_workers": 16,
            },
            epochs=300,
        )


def test_paper_batch_mapping_records_total_and_per_rank_values():
    assert paper_batch_mapping({"batch_size": 96}, world_size=1) == {
        "global_batch_size": 96,
        "world_size": 1,
        "per_rank_batch_size": 96,
    }
    assert paper_batch_mapping({"batch_size": 48}, world_size=2) == {
        "global_batch_size": 96,
        "world_size": 2,
        "per_rank_batch_size": 48,
    }
    with pytest.raises(ValueError, match="global batch"):
        paper_batch_mapping({"batch_size": 4}, world_size=1)
    with pytest.raises(ValueError, match="divide"):
        paper_batch_mapping({"batch_size": 10}, world_size=7)


def test_runtime_paper_config_derives_per_rank_batch_without_mutating_template():
    template = {
        "batch_size": 96,
        "lr": 2e-4,
        "final_lr": 2e-6,
        "lr_schedule": "linear",
        "weight_decay": 0.0,
        "num_workers": 16,
        "run_name": "rae_stream",
    }
    resolved = runtime_paper_config(template, world_size=4)
    assert resolved["batch_size"] == 24
    assert template["batch_size"] == 96
    assert_runtime_paper_config(template, resolved, world_size=4, epochs=50)


def test_runtime_paper_config_rejects_any_non_batch_training_change():
    template = {
        "batch_size": 96,
        "lr": 2e-4,
        "final_lr": 2e-6,
        "lr_schedule": "linear",
        "weight_decay": 0.0,
        "num_workers": 16,
    }
    changed = dict(template, lr=1e-4)
    with pytest.raises(ValueError, match="runtime config|batch_size"):
        assert_runtime_paper_config(template, changed, world_size=1, epochs=50)


def test_only_declared_data_identity_keys_change():
    base = {"batch_size": 4, "datasets": {"recon": 1}, "run_name": "raenwm"}
    overlay = {"batch_size": 4, "datasets": {"rae_stream": 1}, "run_name": "rae_stream"}
    assert set(changed_keys(base, overlay)) == {"datasets", "run_name"}
    with pytest.raises(ValueError, match="not allowed"):
        changed_keys(base, {"batch_size": 8})
    with pytest.raises(ValueError, match="not allowed"):
        changed_keys(base, {**base, "from_checkpoint": "checkpoint.pth.tar"})


def test_checked_in_overlay_changes_only_data_and_run_identity():
    root = Path(__file__).resolve().parents[2]
    base = load_yaml_mapping(root / "config" / "raenwm.yaml")
    overlay = load_yaml_mapping(root / "config" / "rae_stream.yaml")
    changed = set(assert_data_overlay(base, overlay))
    assert changed <= {
        "datasets",
        "run_name",
        "results_dir",
        "wandb",
        "from_checkpoint",
        "checkpoint_path",
        "batch_size",
        "lr",
        "final_lr",
        "lr_schedule",
        "weight_decay",
    }
    assert set(overlay["datasets"]) == {"rae_stream"}
    assert_training_profile(overlay, epochs=50)
    assert overlay["batch_size"] == 96
    assert float(overlay["lr"]) == pytest.approx(2e-4)
    assert float(overlay["final_lr"]) == pytest.approx(2e-6)
    assert overlay["weight_decay"] == pytest.approx(0.0)


def test_reviewed_converter_bytes_are_pinned():
    root = Path(__file__).resolve().parents[2]
    digest = verify_converter_source(root)
    assert len(digest) == 64


def test_resume_settings_and_existing_latest_are_rejected(tmp_path: Path):
    config = {
        "results_dir": "logs",
        "run_name": "rae_stream",
    }
    assert_no_resume_settings(config)
    output = assert_fresh_training_output(tmp_path, config)
    assert output == (tmp_path / "logs" / "rae_stream").resolve()

    with pytest.raises(ValueError, match="from_checkpoint"):
        assert_no_resume_settings({**config, "from_checkpoint": "other.pth.tar"})

    latest = output / "checkpoints" / "latest.pth.tar"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="resume"):
        assert_fresh_training_output(tmp_path, config)


def test_output_identity_cannot_escape_repo_root(tmp_path: Path):
    with pytest.raises(ValueError, match="output"):
        assert_fresh_training_output(
            tmp_path,
            {"results_dir": "../../outside", "run_name": "rae_stream"},
        )


def test_output_identity_rejects_preexisting_symlink_components(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    results = tmp_path / "logs"
    results.mkdir()
    (results / "rae_stream").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|output"):
        assert_fresh_training_output(
            tmp_path,
            {"results_dir": "logs", "run_name": "rae_stream"},
        )


def test_dataset_overlay_pins_loader_sampling_abi():
    base = {
        "datasets": {
            "recon": {
                "data_folder": "data/recon",
                "train": "splits/recon/train",
                "test": "splits/recon/test",
                "goals_per_obs": 4,
            }
        }
    }
    valid = {
        "datasets": {
            "rae_stream": {
                "data_folder": "data/rae_stream",
                "train": "splits/rae_stream/train",
                "test": "splits/rae_stream/test",
                "goals_per_obs": 4,
            }
        }
    }
    assert_data_overlay(base, valid)
    for bad in (
        {**valid, "datasets": {"rae_stream": {**valid["datasets"]["rae_stream"], "goals_per_obs": 8}}},
        {**valid, "datasets": {"rae_stream": {**valid["datasets"]["rae_stream"], "unknown": 1}}},
    ):
        with pytest.raises(ValueError, match="dataset|goals_per_obs|loader"):
            assert_data_overlay(base, bad)
