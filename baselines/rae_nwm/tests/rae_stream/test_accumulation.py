"""Scientific contracts for the paper-total-batch accumulation adapter.

These tests intentionally define the runtime arithmetic before the adapter is
implemented.  They do not allocate CUDA or invoke the official trainer.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from rae_stream.accumulation import (
    accumulation_update_count,
    make_accumulation_plan,
    scaled_loss_value,
)
from rae_stream.config_guard import (
    ACCUMULATION_PROFILE_NAME,
    PAPER_PROFILE_NAME,
    assert_runtime_paper_config,
    paper_batch_mapping,
    runtime_paper_config,
    training_profile_name,
)
from rae_stream.launcher import materialize_runtime_config, resolve_world_size


def _template() -> dict:
    return {
        "batch_size": 96,
        "lr": 2e-4,
        "final_lr": 2e-6,
        "lr_schedule": "linear",
        "weight_decay": 0.0,
        "num_workers": 16,
        "run_name": "rae_stream",
    }


def test_world4_accum3_preserves_effective_global_batch_and_derives_microbatch():
    plan = make_accumulation_plan(global_batch_size=96, world_size=4, accumulation_steps=3)
    assert plan.microbatch_size == 8
    assert plan.microbatch_global_size == 32
    assert plan.effective_global_batch_size == 96
    assert paper_batch_mapping(
        {"batch_size": 8}, world_size=4, gradient_accumulation_steps=3
    ) == {
        "global_batch_size": 96,
        "world_size": 4,
        "per_rank_batch_size": 8,
        "gradient_accumulation_steps": 3,
        "microbatch_global_size": 32,
        "effective_global_batch_size": 96,
    }


def test_accumulation_has_a_distinct_receipt_profile_identity():
    assert training_profile_name(1) == PAPER_PROFILE_NAME == "paper-reproduction-v1"
    assert training_profile_name(3) == ACCUMULATION_PROFILE_NAME == "RAE-stream-accumulation"
    with pytest.raises(ValueError, match="positive integer"):
        training_profile_name(1.5)


def test_runtime_config_records_accumulation_without_changing_paper_fields():
    template = _template()
    resolved = runtime_paper_config(
        template, world_size=4, gradient_accumulation_steps=3
    )
    assert resolved["batch_size"] == 8
    assert resolved["gradient_accumulation_steps"] == 3
    assert resolved["effective_global_batch_size"] == 96
    assert resolved["training_profile"] == "RAE-stream-accumulation"
    assert resolved["paper_training_profile"] == "paper-reproduction-v1"
    assert template["batch_size"] == 96
    assert_runtime_paper_config(
        template,
        resolved,
        world_size=4,
        gradient_accumulation_steps=3,
        epochs=50,
    )


@pytest.mark.parametrize(
    ("world_size", "accumulation_steps"),
    [(4, 5), (7, 1), (5, 4)],
)
def test_accumulation_requires_exact_divisibility(world_size, accumulation_steps):
    with pytest.raises(ValueError, match="divide|divisible|batch"):
        make_accumulation_plan(
            global_batch_size=96,
            world_size=world_size,
            accumulation_steps=accumulation_steps,
        )


def test_update_count_drops_only_incomplete_tail_and_scales_loss_once():
    assert accumulation_update_count(10, 3) == 3
    assert accumulation_update_count(9, 3) == 3
    assert scaled_loss_value(12.0, 3) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="positive|accumulation"):
        accumulation_update_count(3, 0)


def test_runtime_yaml_is_deterministic_and_separate_for_accumulation(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source_root = Path(__file__).resolve().parents[2]
    (config_dir / "rae_stream.yaml").write_bytes(
        (source_root / "config" / "rae_stream.yaml").read_bytes()
    )
    path, resolved, info = materialize_runtime_config(
        tmp_path,
        world_size=4,
        gradient_accumulation_steps=3,
    )
    assert path.name.endswith("_world4_accum3.yaml")
    assert resolved["batch_size"] == 8
    assert info["batch_mapping"]["effective_global_batch_size"] == 96
    assert info["training_profile"] == "RAE-stream-accumulation"
    assert info["paper_training_profile"] == "paper-reproduction-v1"
    assert info["evaluation_batch_size_per_rank"] == 8
    assert info["evaluation_batch_semantics"] == "physical_microbatch"
    path2, resolved2, info2 = materialize_runtime_config(
        tmp_path,
        world_size=4,
        gradient_accumulation_steps=3,
    )
    assert path2 == path
    assert resolved2 == resolved
    assert info2["config_sha256"] == info["config_sha256"]


def test_paper_profile_runtime_identity_includes_microbatch_size(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source_root = Path(__file__).resolve().parents[2]
    (config_dir / "rae_stream.yaml").write_bytes(
        (source_root / "config" / "rae_stream.yaml").read_bytes()
    )
    path, _resolved, info = materialize_runtime_config(
        tmp_path,
        world_size=1,
        gradient_accumulation_steps=1,
    )
    assert path == config_dir / "rae_stream.yaml"
    assert info["microbatch_size"] == 96


def test_world_selection_respects_accumulation_divisibility():
    assert resolve_world_size(None, eligible_count=8, gradient_accumulation_steps=3) == 8
    with pytest.raises(ValueError, match="divide"):
        resolve_world_size(5, eligible_count=8, gradient_accumulation_steps=3)
    with pytest.raises(ValueError, match="positive divisor|positive integer"):
        resolve_world_size(None, eligible_count=8, gradient_accumulation_steps=3.5)


def test_train_loop_contains_only_explicit_accumulation_adapter_contract():
    source = (Path(__file__).resolve().parents[2] / "train.py").read_text(encoding="utf-8")
    assert "gradient_accumulation_steps" in source
    assert "model.no_sync()" in source
    assert "scaled_loss = loss / float(accumulation_steps)" in source
    assert "total_steps = args.epochs * updates_per_epoch" in source
    assert "RAE-stream-accumulation" in source
    assert "effective_global_batch_size" in source


def test_loss_scaling_matches_one_effective_batch_gradient_on_cpu():
    # Scalar linear regression is enough to test the exact mean-gradient
    # identity without importing CUDA/PyTorch in the local test environment.
    x = (1.0, -2.0, 0.5, 3.0, -1.5, 2.5)
    target = (0.2, -0.4, 0.7, 1.1, -0.8, 0.3)
    weight = 0.37
    full_gradient = sum(2.0 * (weight * xi - yi) * xi for xi, yi in zip(x, target)) / len(x)
    accumulated_gradient = 0.0
    for start in (0, 2, 4):
        micro = sum(
            2.0 * (weight * xi - yi) * xi
            for xi, yi in zip(x[start : start + 2], target[start : start + 2])
        ) / 2.0
        accumulated_gradient += micro / 3.0
    assert accumulated_gradient == pytest.approx(full_gradient)
