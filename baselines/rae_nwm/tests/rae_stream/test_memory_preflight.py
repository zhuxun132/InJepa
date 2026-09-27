"""Scientific contract tests for the bounded RAE memory probe.

The first RED run is intentional: the production ``run_memory_preflight`` API
has not yet been implemented.  These tests define the narrow receipt/ABI
contract before implementation; they never allocate CUDA or start training.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from rae_stream.config_guard import (
    OFFICIAL_COMMIT,
    OFFICIAL_RESTRICTED_SHA256,
    canonical_digest,
)
from rae_stream.launcher import assert_memory_preflight_headroom, verify_memory_preflight
from rae_stream.memory_preflight import _worker_command

# Deliberately import the not-yet-present probe API.  This must fail during the
# RED phase; once implemented, the signature test below protects the ABI.
from rae_stream.memory_preflight import run_memory_preflight


def _valid_payload(*, world_size: int = 2) -> dict:
    indices = [1, 4][:world_size]
    uuids = [f"GPU-{index}" for index in indices]
    payload = {
        "status": "PASS",
        "probe_kind": "official_train_step_memory",
        "official_commit": OFFICIAL_COMMIT,
        "official_source_sha256": dict(OFFICIAL_RESTRICTED_SHA256),
        "official_source": {
            "training_source_variant": "official_with_gradient_accumulation_adapter_v1",
        },
        "global_batch_size": 96,
        "world_size": world_size,
        "per_rank_batch_size": 96 // world_size,
        "gpu_indices": indices,
        "gpu_uuids": uuids,
        "config_sha256": "c" * 64,
        "probe_steps": 1,
        "optimizer_step": True,
        "oom": False,
        "bfloat16": 1,
        "torch_compile": 1,
        "free_mib_before": [12000] * world_size,
        "peak_memory_allocated_mib": [9000] * world_size,
        "safety_margin_mib": 1024,
    }
    payload["receipt_sha256"] = canonical_digest(payload)
    return payload


def _write_receipt(tmp_path: Path, payload: dict) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "memory_preflight.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_probe_api_exposes_explicit_runtime_identity_contract():
    """The probe must accept all identity inputs instead of reading globals."""

    parameters = set(inspect.signature(run_memory_preflight).parameters)
    assert {
        "repo_root",
        "config_path",
        "output_path",
        "world_size",
        "gpu_indices",
        "gpu_uuids",
    } <= parameters


def test_probe_worker_is_invoked_as_package_module_for_both_topologies():
    args = ["--worker", "--repo-root", "/tmp/rae-stream"]
    assert _worker_command("python", 1, args) == [
        "python",
        "-m",
        "rae_stream.memory_preflight",
        *args,
    ]
    multi = _worker_command("python", 2, args)
    assert multi[:7] == [
        "python",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        "2",
        "-m",
    ]
    assert multi[7:9] == ["rae_stream.memory_preflight", "--worker"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("global_batch_size", 4, "global batch"),
        ("per_rank_batch_size", 96, "world/per-rank"),
        ("bfloat16", 0, "BF16"),
        ("torch_compile", 0, "compile"),
        ("optimizer_step", False, "optimizer"),
        ("oom", True, "optimizer"),
        ("probe_steps", 0, "probe_steps"),
        ("safety_margin_mib", -1, "safety margin"),
    ],
)
def test_receipt_rejects_training_or_memory_semantic_mutation(
    tmp_path: Path, field, value, match
):
    payload = _valid_payload()
    payload[field] = value
    payload["receipt_sha256"] = canonical_digest({k: v for k, v in payload.items() if k != "receipt_sha256"})
    path, digest = _write_receipt(tmp_path, payload)
    with pytest.raises((ValueError, RuntimeError), match=match):
        verify_memory_preflight(
            path,
            expected_sha256=digest,
            world_size=2,
            config_sha256="c" * 64,
            gpu_indices=[1, 4],
            gpu_uuids=["GPU-1", "GPU-4"],
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("gpu_indices", [4, 1], "GPU list"),
        ("gpu_uuids", ["GPU-1", "GPU-X"], "GPU UUIDs"),
        ("config_sha256", "d" * 64, "config"),
        ("official_commit", "0" * 40, "official commit"),
    ],
)
def test_receipt_rejects_identity_rebinding(tmp_path: Path, field, value, match):
    payload = _valid_payload()
    payload[field] = value
    payload["receipt_sha256"] = canonical_digest({k: v for k, v in payload.items() if k != "receipt_sha256"})
    path, digest = _write_receipt(tmp_path, payload)
    with pytest.raises((ValueError, RuntimeError), match=match):
        verify_memory_preflight(
            path,
            expected_sha256=digest,
            world_size=2,
            config_sha256="c" * 64,
            gpu_indices=[1, 4],
            gpu_uuids=["GPU-1", "GPU-4"],
        )


def test_receipt_rejects_peak_plus_margin_exceeding_free(tmp_path: Path):
    payload = _valid_payload()
    payload["peak_memory_allocated_mib"] = [11900, 9000]
    payload["receipt_sha256"] = canonical_digest({k: v for k, v in payload.items() if k != "receipt_sha256"})
    path, digest = _write_receipt(tmp_path, payload)
    with pytest.raises(ValueError, match="margin|exceeds"):
        verify_memory_preflight(
            path,
            expected_sha256=digest,
            world_size=2,
            config_sha256="c" * 64,
            gpu_indices=[1, 4],
            gpu_uuids=["GPU-1", "GPU-4"],
        )


def test_launch_headroom_gate_compares_latest_free_memory_to_probe_requirement():
    payload = _valid_payload()
    selected = [
        {"index": 1, "free_mib": 10024, "uuid": "GPU-1"},
        {"index": 4, "free_mib": 11000, "uuid": "GPU-4"},
    ]
    result = assert_memory_preflight_headroom(payload, selected)
    assert result["required_mib"] == [10024, 10024]
    selected[1]["free_mib"] = 10023
    with pytest.raises(RuntimeError, match="current free|preflight"):
        assert_memory_preflight_headroom(payload, selected)


def test_receipt_rejects_non_owned_path_before_probe_consumption(tmp_path: Path):
    """The launcher-level path gate must not consume an arbitrary external receipt."""

    # This is an ABI-level assertion for the probe API: it must require an
    # explicit output path, leaving ownership/traversal enforcement to the
    # launcher.  No filesystem mutation outside tmp_path is attempted here.
    parameters = inspect.signature(run_memory_preflight).parameters
    assert parameters["output_path"].default is inspect.Parameter.empty


def test_accumulation_receipt_binds_microbatch_and_effective_global_batch(tmp_path: Path):
    payload = _valid_payload(world_size=4)
    payload["per_rank_batch_size"] = 8
    payload["gradient_accumulation_steps"] = 3
    payload["training_profile"] = "RAE-stream-accumulation"
    payload["paper_training_profile"] = "paper-reproduction-v1"
    payload["microbatch_global_size"] = 32
    payload["effective_global_batch_size"] = 96
    payload["evaluation_batch_size_per_rank"] = 8
    payload["evaluation_batch_semantics"] = "physical_microbatch"
    payload["runtime_config"] = {
        "config_sha256": "c" * 64,
        "derived": True,
        "training_profile": "RAE-stream-accumulation",
        "paper_training_profile": "paper-reproduction-v1",
        "gradient_accumulation_steps": 3,
        "microbatch_size": 8,
        "microbatch_global_size": 32,
        "effective_global_batch_size": 96,
        "evaluation_batch_size_per_rank": 8,
        "evaluation_batch_semantics": "physical_microbatch",
        "batch_mapping": {
            "global_batch_size": 96,
            "world_size": 4,
            "per_rank_batch_size": 8,
            "gradient_accumulation_steps": 3,
            "microbatch_global_size": 32,
            "effective_global_batch_size": 96,
        },
    }
    payload["gpu_indices"] = [0, 5, 6, 7]
    payload["gpu_uuids"] = ["GPU-0", "GPU-5", "GPU-6", "GPU-7"]
    payload["free_mib_before"] = [12000] * 4
    payload["peak_memory_allocated_mib"] = [9000] * 4
    payload["receipt_sha256"] = canonical_digest(
        {k: v for k, v in payload.items() if k != "receipt_sha256"}
    )
    path, digest = _write_receipt(tmp_path, payload)
    verified = verify_memory_preflight(
        path,
        expected_sha256=digest,
        world_size=4,
        gradient_accumulation_steps=3,
        config_sha256="c" * 64,
        gpu_indices=[0, 5, 6, 7],
        gpu_uuids=["GPU-0", "GPU-5", "GPU-6", "GPU-7"],
    )
    assert verified["gradient_accumulation_steps"] == 3
    with pytest.raises(ValueError, match="accumulation|microbatch|batch"):
        verify_memory_preflight(
            path,
            expected_sha256=digest,
            world_size=4,
            gradient_accumulation_steps=2,
            config_sha256="c" * 64,
            gpu_indices=[0, 5, 6, 7],
            gpu_uuids=["GPU-0", "GPU-5", "GPU-6", "GPU-7"],
        )
    missing = dict(payload)
    missing.pop("effective_global_batch_size")
    missing["receipt_sha256"] = canonical_digest(
        {k: v for k, v in missing.items() if k != "receipt_sha256"}
    )
    missing_path, missing_digest = _write_receipt(tmp_path / "missing", missing)
    with pytest.raises(ValueError, match="effective|batch"):
        verify_memory_preflight(
            missing_path,
            expected_sha256=missing_digest,
            world_size=4,
            gradient_accumulation_steps=3,
            config_sha256="c" * 64,
            gpu_indices=[0, 5, 6, 7],
            gpu_uuids=["GPU-0", "GPU-5", "GPU-6", "GPU-7"],
        )

    wrong_profile = dict(payload)
    wrong_profile["training_profile"] = "paper-reproduction-v1"
    wrong_profile["receipt_sha256"] = canonical_digest(
        {k: v for k, v in wrong_profile.items() if k != "receipt_sha256"}
    )
    wrong_path, wrong_digest = _write_receipt(tmp_path / "wrong_profile", wrong_profile)
    with pytest.raises(ValueError, match="profile"):
        verify_memory_preflight(
            wrong_path,
            expected_sha256=wrong_digest,
            world_size=4,
            gradient_accumulation_steps=3,
            config_sha256="c" * 64,
            gpu_indices=[0, 5, 6, 7],
            gpu_uuids=["GPU-0", "GPU-5", "GPU-6", "GPU-7"],
        )

    missing_profile = dict(payload)
    missing_profile.pop("paper_training_profile")
    missing_profile["receipt_sha256"] = canonical_digest(
        {k: v for k, v in missing_profile.items() if k != "receipt_sha256"}
    )
    missing_profile_path, missing_profile_digest = _write_receipt(
        tmp_path / "missing_profile", missing_profile
    )
    with pytest.raises(ValueError, match="paper_training_profile|profile"):
        verify_memory_preflight(
            missing_profile_path,
            expected_sha256=missing_profile_digest,
            world_size=4,
            gradient_accumulation_steps=3,
            config_sha256="c" * 64,
            gpu_indices=[0, 5, 6, 7],
            gpu_uuids=["GPU-0", "GPU-5", "GPU-6", "GPU-7"],
        )

    wrong_source = dict(payload)
    wrong_source["official_source"] = {
        "training_source_variant": "official_baseline"
    }
    wrong_source["receipt_sha256"] = canonical_digest(
        {k: v for k, v in wrong_source.items() if k != "receipt_sha256"}
    )
    wrong_source_path, wrong_source_digest = _write_receipt(
        tmp_path / "wrong_source", wrong_source
    )
    with pytest.raises(ValueError, match="official_source|variant"):
        verify_memory_preflight(
            wrong_source_path,
            expected_sha256=wrong_source_digest,
            world_size=4,
            gradient_accumulation_steps=3,
            config_sha256="c" * 64,
            gpu_indices=[0, 5, 6, 7],
            gpu_uuids=["GPU-0", "GPU-5", "GPU-6", "GPU-7"],
        )

    wrong_runtime = dict(payload)
    wrong_runtime["runtime_config"] = dict(payload["runtime_config"])
    wrong_runtime["runtime_config"]["training_profile"] = "paper-reproduction-v1"
    wrong_runtime["receipt_sha256"] = canonical_digest(
        {k: v for k, v in wrong_runtime.items() if k != "receipt_sha256"}
    )
    wrong_runtime_path, wrong_runtime_digest = _write_receipt(
        tmp_path / "wrong_runtime", wrong_runtime
    )
    with pytest.raises(ValueError, match="runtime config|runtime_config|profile"):
        verify_memory_preflight(
            wrong_runtime_path,
            expected_sha256=wrong_runtime_digest,
            world_size=4,
            gradient_accumulation_steps=3,
            config_sha256="c" * 64,
            gpu_indices=[0, 5, 6, 7],
            gpu_uuids=["GPU-0", "GPU-5", "GPU-6", "GPU-7"],
        )
