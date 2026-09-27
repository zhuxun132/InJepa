from pathlib import Path
import hashlib

import pytest

from rae_stream.launcher import (
    OFFICIAL_TRAIN_CLI_DEFAULTS,
    build_train_argv,
    build_child_environment,
    choose_free_gpu,
    choose_free_gpus,
    resolve_world_size,
    query_gpus,
    recheck_gpu_inventory,
    recheck_gpu_inventory_set,
    python_executable_identity,
    verify_memory_preflight,
    verify_child_started,
)


def test_paper_cli_defaults_are_explicitly_registered():
    assert OFFICIAL_TRAIN_CLI_DEFAULTS == {
        "--epochs": 50,
        "--global-seed": 42,
        "--log-every": 100,
        "--ckpt-every": 5000,
        "--eval-every": 1000,
        "--bfloat16": 1,
        "--torch-compile": 1,
    }


def test_train_argv_calls_official_train_without_reimplementing_it(tmp_path: Path):
    argv = build_train_argv(
        repo_root=tmp_path,
        config="config/rae_stream.yaml",
        extra_args=["--bfloat16", "1"],
    )
    assert argv[:2] == ["python", "train.py"]
    assert "--config" in argv and "config/rae_stream.yaml" in argv
    assert "--bfloat16" in argv


def test_train_argv_uses_official_torchrun_for_explicit_multi_gpu_world(tmp_path: Path):
    argv = build_train_argv(
        repo_root=tmp_path,
        config="receipts/runtime_configs/world4.yaml",
        world_size=4,
    )
    assert argv[:6] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "4",
    ]
    assert argv[6] == "train.py"


def test_train_argv_fills_all_released_paper_runtime_flags(tmp_path: Path):
    argv = build_train_argv(repo_root=tmp_path, config="config/rae_stream.yaml")
    pairs = dict(zip(argv[4::2], argv[5::2]))
    assert pairs == {name: str(value) for name, value in OFFICIAL_TRAIN_CLI_DEFAULTS.items()}


def test_accumulation_option_is_explicit_and_reaches_official_argv(tmp_path: Path):
    from scripts.launch_rae_stream import _extra_args, build_arg_parser

    args = build_arg_parser().parse_args(
        ["plan", "--gradient-accumulation-steps", "3"]
    )
    assert args.gradient_accumulation_steps == 3
    extra = _extra_args(args)
    # The launcher-only option is materialized into the derived YAML; it is
    # deliberately not forwarded as an unsupported upstream CLI flag.
    assert "--gradient-accumulation-steps" not in extra


def test_validation_closure_allows_only_reviewed_runtime_guard_transition():
    from rae_stream.launcher import (
        VALIDATION_RECEIPT_AUTHORITY_FILES,
        verify_validation_receipt_closure,
    )

    unchanged = {
        relative: f"{index:064x}"
        for index, relative in enumerate(VALIDATION_RECEIPT_AUTHORITY_FILES, start=1)
    }
    recorded = dict(unchanged)
    current = dict(unchanged)
    recorded["rae_stream/config_guard.py"] = "fdad9cabf0afca5f80ab914b8a3820b7593671051a664253427a7915e9058695"
    current["rae_stream/config_guard.py"] = "fa085002dfac2b25b6d8ba3f5eb43a3ca15b346cf32573948cd5f4a2be07440c"

    transitions = verify_validation_receipt_closure(recorded, current)
    assert transitions == {
        "rae_stream/config_guard.py": {
            "from_sha256": recorded["rae_stream/config_guard.py"],
            "to_sha256": current["rae_stream/config_guard.py"],
            "reason": "accumulation_runtime_profile_guard",
        }
    }

    current["rae_stream/config_guard.py"] = "0" * 64
    with pytest.raises(ValueError, match="validation-closure"):
        verify_validation_receipt_closure(recorded, current)


@pytest.mark.parametrize(
    "tokens",
    [
        ["--config", "evil.yaml"],
        ["--config=evil.yaml"],
        ["--epochs", "1"],
        ["--bfloat16=0"],
        ["--torch-compile", "0"],
    ],
)
def test_train_argv_rejects_direct_api_training_overrides(tmp_path: Path, tokens):
    with pytest.raises(ValueError, match="protected"):
        build_train_argv(
            repo_root=tmp_path,
            config="config/rae_stream.yaml",
            extra_args=tokens,
        )


def test_child_environment_rejects_import_shadowing(monkeypatch):
    from rae_stream.launcher import build_child_environment

    monkeypatch.setenv("PYTHONPATH", "/tmp/shadow")
    environment = build_child_environment({"WANDB_MODE": "offline"}, gpu_index=3)
    assert "PYTHONPATH" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "3"
    assert environment["WANDB_MODE"] == "offline"
    with pytest.raises(ValueError, match="environment override"):
        build_child_environment({"PYTHONPATH": "/tmp/shadow"}, gpu_index=3)


def test_child_environment_binds_physical_gpu_list_for_torchrun():
    environment = build_child_environment({}, gpu_indices=[3, 7])
    assert environment["CUDA_VISIBLE_DEVICES"] == "3,7"
    with pytest.raises(ValueError, match="GPU indices"):
        build_child_environment({}, gpu_indices=[3, 3])


def test_child_environment_binds_dedicated_offline_hf_home(tmp_path: Path):
    from rae_stream.launcher import build_child_environment

    environment = build_child_environment({}, gpu_index=1, hf_home=tmp_path / "hf")
    assert environment["HF_HOME"] == str((tmp_path / "hf").resolve())
    assert environment["HF_HUB_CACHE"] == str((tmp_path / "hf" / "hub").resolve())
    assert environment["TRANSFORMERS_CACHE"] == str((tmp_path / "hf" / "hub").resolve())
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    with pytest.raises(ValueError, match="HF|cache"):
        build_child_environment(
            {"HF_HOME": str(tmp_path / "other")}, gpu_index=1, hf_home=tmp_path / "hf"
        )


def test_child_environment_clears_inherited_distributed_topology(monkeypatch):
    from rae_stream.launcher import build_child_environment

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
    environment = build_child_environment({}, gpu_index=2)
    assert "RANK" not in environment
    assert "WORLD_SIZE" not in environment
    assert "MASTER_ADDR" not in environment


def test_python_runtime_identity_records_dependency_surface():
    import sys

    from rae_stream.launcher import runtime_environment_identity

    identity = runtime_environment_identity(sys.executable)
    assert identity["python_version"]
    assert isinstance(identity["sys_path"], list)
    assert "packages" in identity


def test_python_executable_identity_preserves_lexical_venv_entrypoint(tmp_path: Path):
    target = tmp_path / "python3.9"
    target.write_bytes(b"fake-python-binary")
    target.chmod(0o755)
    lexical = tmp_path / "venv" / "bin" / "python"
    lexical.parent.mkdir(parents=True)
    lexical.symlink_to(target)

    identity = python_executable_identity(str(lexical))

    assert identity["path"] == str(lexical)
    assert identity["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()


def test_formal_revision_cannot_be_overridden():
    from rae_stream.launcher import assert_formal_revision

    assert assert_formal_revision(None)
    assert assert_formal_revision(
        "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9"
    )
    with pytest.raises(ValueError, match="fixed StreamVLN revision"):
        assert_formal_revision("other-revision")


def test_gpu_selection_is_runtime_driven_and_fail_closed():
    rows = [
        {"index": 0, "free_mib": 1024, "util": 0, "name": "RTX 3090"},
        {"index": 1, "free_mib": 8192, "util": 2, "name": "RTX 3090"},
    ]
    assert choose_free_gpu(rows, min_free_mib=4096, max_util=25)["index"] == 1
    with pytest.raises(RuntimeError, match="no GPU"):
        choose_free_gpu(rows, min_free_mib=9000, max_util=25)


def test_multi_gpu_selection_and_world_size_are_runtime_driven():
    rows = [
        {"index": 1, "uuid": "GPU-1", "free_mib": 8192, "util": 2, "name": "RTX 3090", "apps": [], "apps_query_status": "ok"},
        {"index": 4, "uuid": "GPU-4", "free_mib": 8192, "util": 3, "name": "RTX 3090", "apps": [], "apps_query_status": "ok"},
        {"index": 6, "uuid": "GPU-6", "free_mib": 8192, "util": 1, "name": "RTX 3090", "apps": [], "apps_query_status": "ok"},
        {"index": 7, "uuid": "GPU-7", "free_mib": 8192, "util": 1, "name": "RTX 3090", "apps": [], "apps_query_status": "ok"},
    ]
    assert resolve_world_size(None, eligible_count=4) == 4
    assert resolve_world_size(2, eligible_count=4) == 2
    selected = choose_free_gpus(rows, count=4, min_free_mib=4096, max_util=25, require_apps_census=True)
    assert [row["index"] for row in selected] == [1, 4, 6, 7]
    with pytest.raises(ValueError, match="divide"):
        resolve_world_size(5, eligible_count=5)


def test_recheck_multi_gpu_set_rejects_changed_membership():
    initial = [
        {"index": 1, "uuid": "GPU-1", "free_mib": 8192, "util": 1, "name": "RTX", "apps": [], "apps_query_status": "ok"},
        {"index": 4, "uuid": "GPU-4", "free_mib": 8192, "util": 1, "name": "RTX", "apps": [], "apps_query_status": "ok"},
    ]
    changed = [
        {"index": 1, "uuid": "GPU-1", "free_mib": 8192, "util": 1, "name": "RTX", "apps": [], "apps_query_status": "ok"},
        {"index": 5, "uuid": "GPU-5", "free_mib": 8192, "util": 1, "name": "RTX", "apps": [], "apps_query_status": "ok"},
    ]
    with pytest.raises(RuntimeError, match="changed"):
        recheck_gpu_inventory_set(initial, lambda: changed, world_size=2)


def test_memory_preflight_receipt_binds_batch_world_and_config(tmp_path: Path):
    import json
    from rae_stream.config_guard import canonical_digest, OFFICIAL_COMMIT, OFFICIAL_RESTRICTED_SHA256

    payload = {
        "status": "PASS",
        "probe_kind": "official_train_step_memory",
        "official_commit": OFFICIAL_COMMIT,
        "official_source_sha256": dict(OFFICIAL_RESTRICTED_SHA256),
        "official_source": {
            "training_source_variant": "official_with_gradient_accumulation_adapter_v1",
        },
        "global_batch_size": 96,
        "world_size": 2,
        "per_rank_batch_size": 48,
        "gpu_indices": [1, 4],
        "gpu_uuids": ["GPU-1", "GPU-4"],
        "config_sha256": "c" * 64,
        "probe_steps": 1,
        "optimizer_step": True,
        "oom": False,
        "bfloat16": 1,
        "torch_compile": 1,
        "free_mib_before": [12000, 12000],
        "peak_memory_allocated_mib": [9000, 9000],
        "safety_margin_mib": 1024,
    }
    payload["receipt_sha256"] = canonical_digest(payload)
    path = tmp_path / "memory.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = verify_memory_preflight(
        path,
        expected_sha256=digest,
        world_size=2,
        config_sha256="c" * 64,
        gpu_indices=[1, 4],
        gpu_uuids=["GPU-1", "GPU-4"],
    )
    assert verified["status"] == "PASS"
    with pytest.raises(ValueError, match="batch|world"):
        verify_memory_preflight(
            path,
            expected_sha256=digest,
            world_size=4,
            config_sha256="c" * 64,
            gpu_indices=[1, 4, 6, 7],
            gpu_uuids=["GPU-1", "GPU-4", "GPU-6", "GPU-7"],
        )


def test_unknown_gpu_utilization_requires_explicit_admission():
    rows = [{"index": 0, "free_mib": 8192, "util": None, "name": "RTX 3090"}]
    with pytest.raises(RuntimeError, match="unknown utilization"):
        choose_free_gpu(rows)
    assert choose_free_gpu(rows, allow_unknown_util=True)["index"] == 0


def test_gpu_with_reported_compute_app_is_never_admitted():
    rows = [
        {
            "index": 0,
            "free_mib": 8192,
            "util": None,
            "name": "RTX 3090",
            "apps": [{"pid": "7", "process_name": "other", "used_memory": "1 MiB"}],
        }
    ]
    with pytest.raises(RuntimeError, match="no GPU"):
        choose_free_gpu(rows, allow_unknown_util=True)


def test_formal_gpu_selection_rejects_unknown_even_with_empty_compute_census():
    """An empty process list is not a numerical utilization measurement."""

    rows = [
        {
            "index": 0,
            "uuid": "GPU-0",
            "free_mib": 8192,
            "util": None,
            "name": "RTX 3090",
            "apps": None,
            "apps_query_status": "error:OSError",
        }
    ]
    with pytest.raises(RuntimeError, match="no GPU"):
        choose_free_gpu(rows, allow_unknown_util=True, require_apps_census=True)

    rows[0].update({"apps": [], "apps_query_status": "ok"})
    with pytest.raises(RuntimeError, match="no GPU|unknown utilization"):
        choose_free_gpu(rows, allow_unknown_util=True, require_apps_census=True)


def test_query_gpus_preserves_nvidia_smi_na(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "0, 24124, [N/A], NVIDIA GeForce RTX 3090\n",
    )
    assert query_gpus(executable="nvidia-smi") == [
        {"index": 0, "free_mib": 24124, "util": None, "name": "NVIDIA GeForce RTX 3090"}
    ]


def test_query_gpus_rejects_duplicate_inventory_index(monkeypatch):
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: (
            "0, GPU-a, 24124, 0, 24576, 0, RTX 3090\n"
            "0, GPU-b, 24124, 0, 24576, 0, RTX 3090\n"
        ),
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        query_gpus(executable="nvidia-smi")


def test_compute_app_census_rejects_unknown_gpu_uuid(monkeypatch):
    import subprocess
    from rae_stream.launcher import _query_compute_apps

    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "GPU-not-in-inventory, 7, python, 1 MiB\n",
    )
    rows = [{"index": 0, "uuid": "GPU-0", "free_mib": 8192, "util": None}]
    _query_compute_apps("nvidia-smi", rows)
    assert rows[0]["apps"] is None
    assert rows[0]["apps_query_status"] == "malformed"


def test_gpu_inventory_is_rechecked_before_spawn():
    initial = [
        {"index": 0, "free_mib": 8192, "util": 0, "name": "RTX 3090"},
        {"index": 1, "free_mib": 1024, "util": 0, "name": "RTX 3090"},
    ]
    changed = [
        {"index": 0, "free_mib": 1024, "util": 90, "name": "RTX 3090"},
        {"index": 1, "free_mib": 8192, "util": 0, "name": "RTX 3090"},
    ]
    with pytest.raises(RuntimeError, match="changed"):
        recheck_gpu_inventory(initial, lambda: changed)


def test_child_start_gate_rejects_immediate_exit():
    class Dead:
        pid = 123

        def poll(self):
            return 1

    with pytest.raises(RuntimeError, match="exited"):
        verify_child_started(Dead(), wait_seconds=0)


def test_child_start_gate_accepts_live_process_without_waiting():
    class Live:
        pid = 123

        def poll(self):
            return None

    assert verify_child_started(Live(), wait_seconds=0) is True


def test_launcher_rejects_extra_config_override():
    from scripts.launch_rae_stream import _extra_args, build_arg_parser

    args = build_arg_parser().parse_args(["--extra-arg=--config", "--extra-arg=other.yaml"])
    with pytest.raises(ValueError, match="protected"):
        _extra_args(args)


def test_launch_helper_rejects_forged_manifest_validation_before_any_spawn(tmp_path: Path):
    from rae_stream.launcher import launch_official_train

    with pytest.raises(ValueError, match="manifest_validation injection"):
        launch_official_train(
            repo_root=tmp_path,
            config=tmp_path / "config.yaml",
            manifest_path=tmp_path / "manifest.jsonl",
            manifest_validation={"status": "PASS"},
        )


def test_formal_launcher_requires_dedicated_authority_and_receipt(tmp_path: Path):
    from rae_stream.launcher import launch_official_train

    with pytest.raises(ValueError, match="authority"):
        launch_official_train(
            repo_root=tmp_path,
            config=tmp_path / "config.yaml",
            manifest_path=tmp_path / "manifest.jsonl",
            prepare_receipt_path=tmp_path / "prepare.json",
            prepare_receipt_sha256="0" * 64,
            hf_home=tmp_path / "hf",
            receipt_path=tmp_path / "launch.json",
        )


def test_formal_launcher_requires_fresh_dedicated_log(tmp_path: Path):
    from rae_stream.launcher import launch_official_train

    with pytest.raises(ValueError, match="log"):
        launch_official_train(
            repo_root=tmp_path,
            config=tmp_path / "config.yaml",
            manifest_path=tmp_path / "manifest.jsonl",
            prepare_receipt_path=tmp_path / "prepare.json",
            prepare_receipt_sha256="0" * 64,
            hf_home=tmp_path / "hf",
            authority_path=tmp_path / "authority.json",
            authority_sha256="0" * 64,
            receipt_path=tmp_path / "launch.json",
        )


def test_nvidia_smi_identity_rejects_non_system_wrapper(tmp_path: Path):
    from rae_stream.launcher import nvidia_smi_identity

    wrapper = tmp_path / "nvidia-smi"
    wrapper.write_text("#!/bin/sh\necho fake\n", encoding="utf-8")
    wrapper.chmod(0o755)
    with pytest.raises(ValueError, match="system|root-owned|nvidia-smi"):
        nvidia_smi_identity(wrapper, formal=True)


def test_nvidia_smi_identity_binds_external_sha(tmp_path: Path, monkeypatch):
    from rae_stream import launcher

    # The test substitutes the platform ownership probe; the production gate
    # still requires a root-owned, canonical system executable.
    binary = tmp_path / "nvidia-smi"
    binary.write_bytes(b"trusted-test-binary")
    binary.chmod(0o755)
    monkeypatch.setattr(launcher, "_is_root_owned_system_executable", lambda path: True)
    digest = launcher.sha256_file(binary)
    identity = launcher.nvidia_smi_identity(binary, expected_sha256=digest, formal=True)
    assert identity["sha256"] == digest
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        launcher.nvidia_smi_identity(binary, expected_sha256="0" * 64, formal=True)


def test_reservation_path_is_single_canonical_file(tmp_path: Path):
    from rae_stream.launcher import canonical_reservation_path

    expected = (tmp_path / ".rae_stream_gpu.lock").resolve()
    assert canonical_reservation_path(tmp_path, None) == expected
    assert canonical_reservation_path(tmp_path, expected) == expected
    with pytest.raises(ValueError, match="canonical reservation"):
        canonical_reservation_path(tmp_path, tmp_path / "other.lock")


def test_runtime_reservation_rejects_dangling_symlink_without_following_it(tmp_path: Path):
    """A lock path must never resolve through a dangling symlink outside root."""

    from rae_stream.launcher import _gpu_reservation, _validate_owned_runtime_path

    outside = tmp_path / "outside"
    outside.mkdir()
    lock = tmp_path / ".rae_stream_gpu.lock"
    lock.symlink_to(outside / "created_if_followed")

    with pytest.raises(ValueError, match="symlink"):
        _validate_owned_runtime_path(tmp_path, lock, label="reservation lock")
    with pytest.raises((ValueError, OSError, RuntimeError)):
        with _gpu_reservation(lock):
            pass
    assert not (outside / "created_if_followed").exists()


def test_prepare_layout_identity_rechecks_inode_before_spawn(tmp_path: Path, monkeypatch):
    import rae_stream.launcher as launcher
    import scripts.validate_rae_stream as validator

    monkeypatch.setattr(launcher, "_load_validation_module", lambda _root: validator)
    verify_prepare_layout_identity = launcher.verify_prepare_layout_identity
    from scripts.prepare_rae_stream import _immutable_path_identity
    import json

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    receipt = tmp_path / "prepare.json"
    payload = {
        "immutable_path_identities": {str(snapshot): _immutable_path_identity(snapshot)}
    }
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    digest = __import__("hashlib").sha256(receipt.read_bytes()).hexdigest()
    result = verify_prepare_layout_identity(
        receipt,
        expected_sha256=digest,
        official_root=tmp_path,
    )
    assert str(snapshot) in result
    snapshot.rename(tmp_path / "old")
    (tmp_path / "replacement").mkdir()
    (tmp_path / "replacement").rename(snapshot)
    with pytest.raises(ValueError, match="identity"):
        verify_prepare_layout_identity(
            receipt,
            expected_sha256=digest,
            official_root=tmp_path,
        )


def _pinned_validation_fixture(tmp_path: Path):
    import hashlib
    import json
    from rae_stream.config_guard import (
        ALLOWED_MODIFIED_CONFIG_SHA256,
        OFFICIAL_BASELINE_RESTRICTED_SHA256,
    )
    from rae_stream.launcher import VALIDATION_RECEIPT_AUTHORITY_FILES

    root = tmp_path.resolve()
    data_root = root / "data" / "rae_stream"
    split_root = root / "data_splits" / "rae_stream"
    data_root.mkdir(parents=True)
    (split_root / "train").mkdir(parents=True)
    (split_root / "test").mkdir(parents=True)
    (split_root / "train" / "traj_names.txt").write_text("t0\n", encoding="utf-8")
    (split_root / "test" / "traj_names.txt").write_text("t1\n", encoding="utf-8")
    source_parent = data_root.parent / "tar_selected_sources"
    source_r2r = source_parent / "r2r"
    source_rxr = source_parent / "rxr"
    source_r2r.mkdir(parents=True)
    source_rxr.mkdir(parents=True)
    manifest = root / "rae_stream_manifest.jsonl"
    manifest.write_text('{"trajectory_id":"t0"}\n', encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    closure_files = {
        relative: f"{index + 100:064x}"
        for index, relative in enumerate(VALIDATION_RECEIPT_AUTHORITY_FILES)
    }
    # This fixture models the real production receipt generated before the
    # accumulation runtime-profile guard was added.
    closure_files["rae_stream/config_guard.py"] = (
        "fdad9cabf0afca5f80ab914b8a3820b7593671051a664253427a7915e9058695"
    )
    prepare = root / "receipts" / "prepare.json"
    prepare.parent.mkdir(parents=True)
    converter_source = {"scripts/prepare_rae_stream.py": "b" * 64}
    immutable_paths = {
        data_root,
        data_root.parent,
        data_root.parent.parent,
        split_root,
        split_root.parent,
        manifest,
        split_root / "train",
        split_root / "test",
        split_root / "train" / "traj_names.txt",
        split_root / "test" / "traj_names.txt",
        source_r2r,
        source_rxr,
        source_parent,
    }
    identity_map = {
        str(path): {"st_dev": 1, "st_ino": index, "mode": 0o755}
        for index, path in enumerate(sorted(immutable_paths, key=str), start=1)
    }
    prepare_payload = {
        "status": "PASS",
        "production_eligible": True,
        "receipt_sha256": None,
        "converter_version": "rae_stream_converter_v2",
        "converter_sha256": "a" * 64,
        "converter_source_sha256": converter_source,
        "converter_bundle_sha256": "c" * 64,
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "data_root": str(data_root),
        "split_root": str(split_root),
        "min_length": 68,
        "formal_min_length": 68,
        "context_size": 4,
        "len_traj_pred": 64,
        "immutable_data_tree": True,
        "immutable_split_lists": True,
        "immutable_manifest": True,
        "immutable_parent_dirs": {str(data_root.parent): True},
        "immutable_source_parent_dirs": {str(source_parent): True},
        "immutable_source_trees": {"R2R": True, "RxR": True},
        "split_list_sha256": {
            "train": hashlib.sha256((split_root / "train" / "traj_names.txt").read_bytes()).hexdigest(),
            "test": hashlib.sha256((split_root / "test" / "traj_names.txt").read_bytes()).hexdigest(),
        },
        "immutable_path_identities": identity_map,
        "sources": [
            {
                "dataset": "R2R",
                "source_revision": "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9",
                "source_mode": "tar-selected",
                "effective_source_root": str(source_r2r),
            },
            {
                "dataset": "RxR",
                "source_revision": "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9",
                "source_mode": "tar-selected",
                "effective_source_root": str(source_rxr),
            },
        ],
    }
    prepare_payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in prepare_payload.items() if key != "receipt_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    prepare.write_text(json.dumps(prepare_payload, sort_keys=True), encoding="utf-8")
    prepare_sha = hashlib.sha256(prepare.read_bytes()).hexdigest()
    payload = {
        "status": "PASS",
        "production_eligible": True,
        "claim_boundary": "FORMAL_DATA_VALIDATION_ONLY",
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha,
        "prepare_receipt_path": str(prepare),
        "prepare_receipt_sha256": prepare_sha,
        "data_root": str(data_root),
        "split_root": str(split_root),
        "rows": 1,
        "train_trajectories": 1,
        "test_trajectories": 1,
        "frame_count_materialized": 2,
        "image_decode_samples": 8,
        "converter_identity_checked": True,
        "converter_sha256": "a" * 64,
        "converter_source_sha256": converter_source,
        "converter_bundle_sha256": "c" * 64,
        "official_source": {
            "commit": "0219ce41c44d515f86719dd763c1efe7c7f72519",
            "files": dict(OFFICIAL_BASELINE_RESTRICTED_SHA256),
            "data_config_files": dict(ALLOWED_MODIFIED_CONFIG_SHA256),
        },
        "code_authority": {
            "official_commit": "0219ce41c44d515f86719dd763c1efe7c7f72519",
            "files": {
                "models.py": OFFICIAL_BASELINE_RESTRICTED_SHA256["models.py"],
                "planning_eval.py": OFFICIAL_BASELINE_RESTRICTED_SHA256["planning_eval.py"],
                **closure_files,
            },
        },
    }
    receipt = root / "receipts" / "validate.json"
    receipt.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    config = {
        "datasets": {
            "rae_stream": {
                "data_folder": str(data_root),
                "train": str(split_root / "train"),
                "test": str(split_root / "test"),
            }
        }
    }
    return {
        "root": root,
        "data_root": data_root,
        "split_root": split_root,
        "manifest": manifest,
        "manifest_sha": manifest_sha,
        "prepare": prepare,
        "prepare_sha": prepare_sha,
        "receipt": receipt,
        "receipt_sha": receipt_sha,
        "converter_source": converter_source,
        "closure_files": closure_files,
        "config": config,
    }


def test_formal_validation_reuses_pinned_receipt_without_rgb_rescan(tmp_path: Path, monkeypatch):
    import rae_stream.launcher as launcher

    fixture = _pinned_validation_fixture(tmp_path)
    monkeypatch.setattr(
        launcher,
        "_load_validation_module",
        lambda _root: (_ for _ in ()).throw(AssertionError("full validator must not run")),
    )
    monkeypatch.setattr(
        launcher,
        "verify_prepare_layout_identity",
        lambda *_args, **_kwargs: {str(fixture["data_root"]): {"st_ino": 1}},
    )
    monkeypatch.setattr(launcher, "verify_converter_source", lambda _root: "a" * 64)
    monkeypatch.setattr(
        launcher,
        "converter_closure_digest_map",
        lambda _root: fixture["converter_source"],
    )
    monkeypatch.setattr(launcher, "converter_bundle_sha256", lambda _mapping: "c" * 64)
    monkeypatch.setattr(
        launcher,
        "verify_upstream_source",
        lambda _root: {
            "commit": "0219ce41c44d515f86719dd763c1efe7c7f72519",
            "training_source_variant": "official_with_gradient_accumulation_adapter_v1",
            "files": {
                "models.py": "c6a917dc008421f1c42dbbff614584702010f1853834cac167bc5fee7ddbe254",
                "planning_eval.py": "4b3be912a583df2f3e3038323dd3f05641b2ff403a8a7d84b3bfd71dd5367bb8",
                "train.py": "a" * 64,
            },
            "data_config_files": {
                "config/data_config.yaml": "4aa34752817a07da4ee57f6015b60b054fc7bebefad5fa3f5bcaa9affc9351dd",
                "config/data_hyperparams_plan.yaml": "d3fc6dca8817ec3e416447adb226327d7686be47a03f89c6adddef6c4b94b6e1",
                "config/eval_config.yaml": "492ce9797ba4ae883ff0406b44c3bb2b0880df636b32ccd40de2efb952c5e4e0",
            },
        },
    )
    monkeypatch.setattr(
        launcher,
        "verify_code_authority",
        lambda *_args, **_kwargs: {
            "sha256": "d" * 64,
            "official_commit": "0219ce41c44d515f86719dd763c1efe7c7f72519",
            "files": {
                "models.py": "c6a917dc008421f1c42dbbff614584702010f1853834cac167bc5fee7ddbe254",
                "planning_eval.py": "4b3be912a583df2f3e3038323dd3f05641b2ff403a8a7d84b3bfd71dd5367bb8",
                **{
                    **fixture["closure_files"],
                    "rae_stream/config_guard.py": "fa085002dfac2b25b6d8ba3f5eb43a3ca15b346cf32573948cd5f4a2be07440c",
                },
            },
        },
    )

    result = launcher.validate_manifest_for_training(
        repo_root=fixture["root"],
        manifest_path=fixture["manifest"],
        config=fixture["config"],
        expected_revision="dc61ee9b4e90aa7ba63c1163b2134df5610dccb9",
        prepare_receipt_path=fixture["prepare"],
        prepare_receipt_sha256=fixture["prepare_sha"],
        authority_path=fixture["root"].parent / "authority.json",
        authority_sha256="d" * 64,
        validation_receipt_path=fixture["receipt"],
        validation_receipt_sha256=fixture["receipt_sha"],
    )
    assert result["status"] == "PASS"
    assert result["validation_mode"] == "PINNED_RECEIPT_WITH_IMMUTABLE_LAYOUT_RECHECK"
    assert result["validation_receipt_sha256"] == fixture["receipt_sha"]
    assert result["current_official_source"]["training_source_variant"].endswith(
        "accumulation_adapter_v1"
    )
    assert result["validation_receipt_closure_transitions"]["rae_stream/config_guard.py"][
        "reason"
    ] == "accumulation_runtime_profile_guard"


def test_pinned_validation_receipt_rejects_manifest_or_converter_mismatch(tmp_path: Path, monkeypatch):
    import json
    import hashlib
    import rae_stream.launcher as launcher

    fixture = _pinned_validation_fixture(tmp_path)
    monkeypatch.setattr(
        launcher,
        "verify_prepare_layout_identity",
        lambda *_args, **_kwargs: {str(fixture["data_root"]): {"st_ino": 1}},
    )
    monkeypatch.setattr(launcher, "verify_converter_source", lambda _root: "a" * 64)
    monkeypatch.setattr(
        launcher,
        "converter_closure_digest_map",
        lambda _root: fixture["converter_source"],
    )
    monkeypatch.setattr(launcher, "converter_bundle_sha256", lambda _mapping: "c" * 64)
    monkeypatch.setattr(
        launcher,
        "verify_upstream_source",
        lambda _root: {
            "commit": "0219ce41c44d515f86719dd763c1efe7c7f72519",
            "files": {
                "models.py": "c6a917dc008421f1c42dbbff614584702010f1853834cac167bc5fee7ddbe254",
                "planning_eval.py": "4b3be912a583df2f3e3038323dd3f05641b2ff403a8a7d84b3bfd71dd5367bb8",
                "train.py": "a" * 64,
            },
            "data_config_files": {
                "config/data_config.yaml": "4aa34752817a07da4ee57f6015b60b054fc7bebefad5fa3f5bcaa9affc9351dd",
                "config/data_hyperparams_plan.yaml": "d3fc6dca8817ec3e416447adb226327d7686be47a03f89c6adddef6c4b94b6e1",
                "config/eval_config.yaml": "492ce9797ba4ae883ff0406b44c3bb2b0880df636b32ccd40de2efb952c5e4e0",
            },
        },
    )
    monkeypatch.setattr(
        launcher,
        "verify_code_authority",
        lambda *_args, **_kwargs: {
            "sha256": "d" * 64,
            "official_commit": "0219ce41c44d515f86719dd763c1efe7c7f72519",
            "files": {
                "models.py": "c6a917dc008421f1c42dbbff614584702010f1853834cac167bc5fee7ddbe254",
                "planning_eval.py": "4b3be912a583df2f3e3038323dd3f05641b2ff403a8a7d84b3bfd71dd5367bb8",
                **fixture["closure_files"],
            },
        },
    )

    payload = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    payload["converter_sha256"] = "0" * 64
    fixture["receipt"].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tampered_receipt_sha = hashlib.sha256(fixture["receipt"].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="converter"):
        launcher.validate_manifest_for_training(
            repo_root=fixture["root"],
            manifest_path=fixture["manifest"],
            config=fixture["config"],
            expected_revision="dc61ee9b4e90aa7ba63c1163b2134df5610dccb9",
            prepare_receipt_path=fixture["prepare"],
            prepare_receipt_sha256=fixture["prepare_sha"],
            authority_path=fixture["root"].parent / "authority.json",
            authority_sha256="d" * 64,
            validation_receipt_path=fixture["receipt"],
            validation_receipt_sha256=tampered_receipt_sha,
        )


def test_pinned_validation_receipt_rejects_prepare_self_hash_tampering(tmp_path: Path):
    import hashlib
    import json
    import rae_stream.launcher as launcher

    fixture = _pinned_validation_fixture(tmp_path)
    prepare_payload = json.loads(fixture["prepare"].read_text(encoding="utf-8"))
    prepare_payload["receipt_sha256"] = "0" * 64
    fixture["prepare"].write_text(json.dumps(prepare_payload, sort_keys=True), encoding="utf-8")
    prepare_sha = hashlib.sha256(fixture["prepare"].read_bytes()).hexdigest()
    validation_payload = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
    validation_payload["prepare_receipt_sha256"] = prepare_sha
    fixture["receipt"].write_text(json.dumps(validation_payload, sort_keys=True), encoding="utf-8")
    validation_sha = hashlib.sha256(fixture["receipt"].read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="self-hash"):
        launcher.validate_manifest_for_training(
            repo_root=fixture["root"],
            manifest_path=fixture["manifest"],
            config=fixture["config"],
            expected_revision="dc61ee9b4e90aa7ba63c1163b2134df5610dccb9",
            prepare_receipt_path=fixture["prepare"],
            prepare_receipt_sha256=prepare_sha,
            authority_path=fixture["root"].parent / "authority.json",
            authority_sha256="d" * 64,
            validation_receipt_path=fixture["receipt"],
            validation_receipt_sha256=validation_sha,
        )


def test_formal_launcher_requires_pinned_validation_receipt(tmp_path: Path):
    from rae_stream.launcher import launch_official_train

    with pytest.raises(ValueError, match="validation receipt"):
        launch_official_train(
            repo_root=tmp_path,
            config=tmp_path / "config.yaml",
            manifest_path=tmp_path / "manifest.jsonl",
            prepare_receipt_path=tmp_path / "prepare.json",
            prepare_receipt_sha256="0" * 64,
            hf_home=tmp_path / "hf",
            authority_path=tmp_path / "authority.json",
            authority_sha256="0" * 64,
            receipt_path=tmp_path / "launch.json",
            log_path=tmp_path / "launch.log",
            memory_preflight_path=tmp_path / "memory.json",
            memory_preflight_sha256="0" * 64,
        )


def test_formal_launcher_binds_validation_receipt_before_initial_identity(
    tmp_path: Path, monkeypatch
):
    """The pinned receipt path must be available to the first identity snapshot."""

    import hashlib
    import rae_stream.launcher as launcher

    root = tmp_path.resolve()
    (root / "config").mkdir()
    template = root / "config" / "rae_stream.yaml"
    template.write_text("datasets: {}\n", encoding="utf-8")
    manifest = root / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    prepare = root / "prepare.json"
    prepare.write_text("{}\n", encoding="utf-8")
    validation = root / "validate.json"
    validation.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(launcher, "_assert_formal_config_paths", lambda *_args: None)
    monkeypatch.setattr(
        launcher,
        "python_executable_identity",
        lambda _value: {"path": "/usr/bin/python3", "sha256": "p" * 64},
    )
    monkeypatch.setattr(
        launcher,
        "nvidia_smi_identity",
        lambda *_args, **_kwargs: {"path": "/usr/bin/nvidia-smi", "sha256": "n" * 64},
    )
    monkeypatch.setattr(
        launcher,
        "resolved_training_identity",
        lambda **_kwargs: {"config": {}, "config_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        launcher,
        "validate_manifest_for_training",
        lambda **_kwargs: {
            "status": "PASS",
            "manifest_path": str(manifest),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "validation_receipt_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(launcher, "resolve_world_size", lambda *_args, **_kwargs: 1)

    with pytest.raises(RuntimeError, match="only 0 GPU"):
        launcher.launch_official_train(
            repo_root=root,
            config=template,
            manifest_path=manifest,
            prepare_receipt_path=prepare,
            prepare_receipt_sha256=hashlib.sha256(prepare.read_bytes()).hexdigest(),
            hf_home=root / "hf",
            authority_path=root / "authority.json",
            authority_sha256="a" * 64,
            receipt_path=root / "launch.json",
            log_path=root / "launch.log",
            memory_preflight_path=root / "memory.json",
            memory_preflight_sha256="m" * 64,
            validation_receipt_path=validation,
            validation_receipt_sha256=hashlib.sha256(validation.read_bytes()).hexdigest(),
            gpu_rows=[
                {
                    "index": 0,
                    "uuid": "GPU-0",
                    "name": "RTX 3090",
                    "total_mib": 24576,
                    "free_mib": 0,
                    "util": 0,
                    "apps": [],
                    "apps_query_status": "ok",
                }
            ],
        )
