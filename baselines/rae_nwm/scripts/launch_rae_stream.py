#!/usr/bin/env python3
"""Launch the unchanged official RAE training entry point for RAE-stream.

The launcher is a gate and an argv builder, not a trainer.  It verifies the
restricted official source surface, the frozen paper-reproduction profile,
and a live GPU quota before creating exactly one child ``train.py`` process.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


def _bootstrap_repo() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _bootstrap_repo()
from rae_stream.config_guard import OFFICIAL_CHECKOUT_PROFILE, STREAMVLN_REVISION  # noqa: E402
from rae_stream.launcher import (  # noqa: E402
    build_train_argv,
    cpu_snapshot,
    launch_official_train,
    materialize_runtime_config,
    resolved_training_identity,
    validate_manifest_for_training,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("plan", "train"), default="plan")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=Path("config/rae_stream.yaml"))
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("config/raenwm.yaml"),
        help="untouched official config used for the data-overlay guard",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--prepare-receipt",
        type=Path,
        help="PASS receipt emitted by prepare_rae_stream.py (required for formal train)",
    )
    parser.add_argument(
        "--prepare-receipt-sha256",
        help="external SHA-256 pin for the immutable production prepare receipt",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        help="dedicated local Hugging Face cache containing the pinned offline DINOv2 snapshot",
    )
    parser.add_argument(
        "--hf-snapshot-root",
        type=Path,
        help="optional explicit pinned HF snapshot directory (must agree with --hf-home)",
    )
    parser.add_argument(
        "--authority",
        type=Path,
        help="external code-authority JSON (required for formal train)",
    )
    parser.add_argument(
        "--authority-sha256",
        help="SHA-256 of the external code-authority JSON",
    )
    parser.add_argument(
        "--nvidia-smi-sha256",
        help="external SHA-256 pin for the trusted system nvidia-smi binary",
    )
    parser.add_argument("--expected-revision", default=STREAMVLN_REVISION)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--python", dest="python_executable", default="python")
    parser.add_argument("--epochs", type=int, default=OFFICIAL_CHECKOUT_PROFILE["epochs"])
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="runtime gradient accumulation factor; derived YAML preserves effective global batch 96",
    )
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument(
        "--gpu-indices",
        help="comma-separated physical GPU indices; omit to choose a runtime-eligible set",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="explicit torchrun world size (must divide paper total batch 96); omit for runtime auto-selection",
    )
    parser.add_argument(
        "--memory-preflight",
        type=Path,
        help="PASS receipt from the bounded official one-step memory probe (required for train)",
    )
    parser.add_argument(
        "--memory-preflight-sha256",
        help="SHA-256 pin for --memory-preflight",
    )
    parser.add_argument(
        "--validation-receipt",
        type=Path,
        help="existing PASS receipt from validate_rae_stream.py (required for formal train)",
    )
    parser.add_argument(
        "--validation-receipt-sha256",
        help="SHA-256 pin for --validation-receipt",
    )
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--max-util", type=int, default=25)
    parser.add_argument(
        "--allow-unknown-util",
        action="store_true",
        help="deprecated safety override; formal launch rejects unknown utilization",
    )
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument(
        "--reservation-lock",
        type=Path,
        help="cooperative lock shared by RAE-stream launchers (defaults inside --repo-root)",
    )
    parser.add_argument(
        "--startup-wait-seconds",
        type=float,
        default=5.0,
        help="bounded grace period used to verify the official train child remains alive",
    )
    parser.add_argument("--startup-poll-seconds", type=float, default=0.2)
    parser.add_argument("--dry-run", action="store_true", help="build/print the official argv without spawning training")
    parser.add_argument("--extra-arg", action="append", default=[], help="one official train.py argv token (repeatable)")
    return parser


def _extra_args(args: argparse.Namespace) -> list[str]:
    result = list(args.extra_arg or [])
    protected = {
        "--config",
        "--epochs",
        "--global-seed",
        "--log-every",
        "--ckpt-every",
        "--eval-every",
        "--bfloat16",
        "--torch-compile",
        "--gradient-accumulation-steps",
    }
    for token in result:
        option = str(token).split("=", 1)[0]
        if option in protected:
            raise ValueError(
                f"extra-arg {token!r} is protected; use the dedicated launcher option instead"
            )
    values = (
        ("--epochs", args.epochs),
        ("--global-seed", args.global_seed),
        ("--log-every", args.log_every),
        ("--ckpt-every", args.ckpt_every),
        ("--eval-every", args.eval_every),
        ("--bfloat16", args.bfloat16),
        ("--torch-compile", args.torch_compile),
    )
    # Explicit options are appended in stable order; callers can still pass
    # arbitrary official flags via --extra-arg.  Duplicate options are rejected
    # to prevent a later token from silently changing the frozen profile.
    supplied_names = {token for token in result if token.startswith("--")}
    for name, value in values:
        if value is None:
            continue
        if name in supplied_names:
            raise ValueError(f"duplicate training option {name}; use one explicit value")
        result.extend([name, str(value)])
        supplied_names.add(name)
    if int(args.epochs) != int(OFFICIAL_CHECKOUT_PROFILE["epochs"]):
        raise ValueError(
            f"paper-reproduction profile requires --epochs {OFFICIAL_CHECKOUT_PROFILE['epochs']}; "
            "a different training budget needs a separately reviewed profile"
        )
    return result


def _gpu_indices(args: argparse.Namespace) -> list[int] | None:
    if args.gpu_indices is None:
        return None
    if not str(args.gpu_indices).strip():
        raise ValueError("--gpu-indices must not be empty")
    try:
        values = [int(part.strip()) for part in str(args.gpu_indices).split(",")]
    except ValueError as exc:
        raise ValueError("--gpu-indices must be comma-separated integers") from exc
    if not values or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("--gpu-indices must be a unique list of nonnegative integers")
    return values


def plan(args: argparse.Namespace) -> dict:
    root = Path(args.repo_root).resolve()
    config = Path(args.config)
    config_abs = config if config.is_absolute() else root / config
    gpu_indices = _gpu_indices(args)
    planned_world_size = args.world_size
    if planned_world_size is None and gpu_indices is not None:
        planned_world_size = len(gpu_indices)
    # Plan mode does not query GPUs.  With no explicit topology it emits a
    # paper-faithful single-process preview and marks world selection pending;
    # formal train resolves the largest valid divisor from the live inventory.
    # A deterministic derived YAML is materialized so the printed argv and
    # identity refer to the same accumulation-aware config the child would use.
    preview_world_size = int(planned_world_size or 1)
    runtime_config_path, _runtime_config, runtime_config_info = materialize_runtime_config(
        root,
        template_path=config_abs,
        world_size=preview_world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    identity = resolved_training_identity(
        repo_root=root,
        config_path=runtime_config_path,
        manifest_path=args.manifest,
        prepare_receipt_path=args.prepare_receipt,
        prepare_receipt_sha256=args.prepare_receipt_sha256,
        base_config_path=args.base_config,
        hf_home=args.hf_home,
        hf_snapshot_root=args.hf_snapshot_root,
        authority_path=args.authority,
        authority_sha256=args.authority_sha256,
        nvidia_smi_sha256=args.nvidia_smi_sha256,
        python_executable=args.python_executable,
        world_size=preview_world_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    manifest_validation = None
    if args.manifest is not None:
        manifest_validation = validate_manifest_for_training(
            repo_root=root,
            manifest_path=args.manifest,
            config=identity["config"],
            expected_revision=args.expected_revision,
            prepare_receipt_path=args.prepare_receipt,
            prepare_receipt_sha256=args.prepare_receipt_sha256,
            authority_path=args.authority,
            authority_sha256=args.authority_sha256,
            validation_receipt_path=args.validation_receipt,
            validation_receipt_sha256=args.validation_receipt_sha256,
        )
    argv = build_train_argv(
        repo_root=root,
        config=runtime_config_path,
        extra_args=_extra_args(args),
        python_executable=args.python_executable,
        world_size=preview_world_size,
    )
    result = {
        "status": "PLAN_ONLY",
        "argv": argv,
        "cwd": str(root),
        "identity": identity,
        "runtime_config": runtime_config_info,
        "cpu": cpu_snapshot(),
        "world_size_requested": args.world_size,
        "gpu_indices_requested": gpu_indices,
        "world_size_resolution": (
            "runtime_auto_pending" if args.world_size is None and gpu_indices is None else "explicit"
        ),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "training_profile": identity["training_profile"],
    }
    if manifest_validation is not None:
        result["manifest_validation"] = manifest_validation
    if not args.dry_run:
        # Plan mode does not query or hold a GPU; only the deterministic
        # runtime-config receipt may have been materialized above.
        result["gpu"] = "not queried in plan mode"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        if args.command == "plan" or args.dry_run:
            result = plan(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        root = Path(args.repo_root).resolve()
        config = Path(args.config)
        if args.manifest is None:
            raise ValueError("formal training requires --manifest from prepare/validate_rae_stream.py")
        if args.prepare_receipt is None:
            raise ValueError("formal training requires --prepare-receipt from prepare_rae_stream.py")
        if args.prepare_receipt_sha256 is None:
            raise ValueError("formal training requires --prepare-receipt-sha256")
        if args.hf_home is None:
            raise ValueError("formal training requires --hf-home for verified offline RAE assets")
        if args.authority is None or args.authority_sha256 is None:
            raise ValueError("formal training requires --authority and --authority-sha256")
        if args.nvidia_smi_sha256 is None:
            raise ValueError("formal training requires --nvidia-smi-sha256")
        if args.validation_receipt is None or args.validation_receipt_sha256 is None:
            raise ValueError(
                "formal training requires --validation-receipt and --validation-receipt-sha256"
            )
        if args.expected_revision != STREAMVLN_REVISION:
            raise ValueError(
                f"formal training requires fixed StreamVLN revision {STREAMVLN_REVISION}"
            )
        extra = _extra_args(args)
        # The lower-level helper owns the single authoritative manifest gate.
        # Keeping validation there prevents a forged ``manifest_validation``
        # mapping from bypassing production/converter/image checks through the
        # Python API, while still ensuring no GPU query occurs before the gate.
        # GPU inventory is queried only after the helper's authoritative data
        # validation.  This avoids consuming a resource snapshot for a
        # manifest that would fail closed, and leaves the query→lock→spawn
        # race gate in one place.
        process = launch_official_train(
            repo_root=root,
            config=config,
            extra_args=extra,
            gpu_rows=None,
            gpu_index=args.gpu_index,
            gpu_indices=_gpu_indices(args),
            world_size=args.world_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            min_free_mib=args.min_free_mib,
            max_util=args.max_util,
            allow_unknown_util=args.allow_unknown_util,
            python_executable=args.python_executable,
            nvidia_smi_executable=args.nvidia_smi,
            log_path=args.log,
            receipt_path=args.receipt,
            manifest_path=args.manifest,
            prepare_receipt_path=args.prepare_receipt,
            prepare_receipt_sha256=args.prepare_receipt_sha256,
            expected_revision=args.expected_revision,
            base_config_path=args.base_config,
            hf_home=args.hf_home,
            hf_snapshot_root=args.hf_snapshot_root,
            authority_path=args.authority,
            authority_sha256=args.authority_sha256,
            nvidia_smi_sha256=args.nvidia_smi_sha256,
            memory_preflight_path=args.memory_preflight,
            memory_preflight_sha256=args.memory_preflight_sha256,
            validation_receipt_path=args.validation_receipt,
            validation_receipt_sha256=args.validation_receipt_sha256,
            manifest_validation=None,
            reservation_path=args.reservation_lock,
            startup_wait_seconds=args.startup_wait_seconds,
            startup_poll_seconds=args.startup_poll_seconds,
        )
        launch_receipt = getattr(process, "rae_stream_launch_receipt", {})
        print(
            json.dumps(
                {
                    "status": "FORMAL_TRAINING_STARTED",
                    "pid": process.pid,
                    "gpu": launch_receipt.get("gpu"),
                    "gpu_initial": launch_receipt.get("gpu_initial"),
                    "startup_verified": launch_receipt.get("startup_verified", False),
                    "cuda_visible_devices": launch_receipt.get("cuda_visible_devices"),
                    "world_size": launch_receipt.get("world_size"),
                    "physical_gpu_indices": launch_receipt.get("physical_gpu_indices"),
                    "cwd": str(root),
                    "manifest_validation": launch_receipt.get("manifest_validation"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"launch_rae_stream: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
