"""Bounded, official-ABI memory preflight for the RAE-stream paper run.

The preflight is deliberately outside ``train.py``.  It imports the exact
official model, RAE encoder, transport loss and optimizer, executes one real
BF16/compile training step per rank, and emits a receipt consumed by the
launcher.  It is a feasibility probe only: it never writes a checkpoint or a
formal training output and its result is not a navigation metric.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from .config_guard import (
    OFFICIAL_COMMIT,
    OFFICIAL_RESTRICTED_SHA256,
    OFFICIAL_CHECKOUT_PROFILE,
    PAPER_PROFILE_NAME,
    assert_training_profile,
    load_yaml_mapping,
    paper_batch_mapping,
    sha256_file,
    training_profile_name,
    verify_upstream_source,
)
from .accumulation import make_accumulation_plan, require_positive_int
from .launcher import (
    build_child_environment,
    choose_free_gpus,
    materialize_runtime_config,
    query_gpus,
    _atomic_json,
    _with_self_hash,
    _validate_owned_runtime_path,
)


def _normalize_indices(values: Sequence[int]) -> list[int]:
    try:
        result = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU indices must be integers") from exc
    if not result or len(set(result)) != len(result) or any(value < 0 for value in result):
        raise ValueError("GPU indices must be a non-empty unique list of nonnegative integers")
    return result


def _normalize_uuids(values: Sequence[str], expected_length: int) -> list[str]:
    result = [str(value) for value in values]
    if len(result) != expected_length or any(not value.strip() for value in result):
        raise ValueError("GPU UUIDs must be non-empty and match the GPU list")
    if len(set(result)) != len(result):
        raise ValueError("GPU UUIDs must be unique")
    return result


def _autocast(enabled: bool):
    import torch

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled, dtype=torch.bfloat16)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=torch.bfloat16)


def _worker_command(
    python_executable: str,
    world_size: int,
    worker_args: Sequence[str],
) -> list[str]:
    """Build the package-module command used for the isolated probe worker."""

    module_args = ["-m", "rae_stream.memory_preflight", *[str(arg) for arg in worker_args]]
    if world_size == 1:
        return [str(python_executable), *module_args]
    return [
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        *module_args,
    ]


def _load_effective_training_config(root: Path, config_path: Path) -> dict[str, Any]:
    """Apply the same default-then-update YAML merge as official ``train.py``."""

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required for the official memory preflight") from exc
    default_path = root / "config" / "eval_config.yaml"
    if not default_path.is_file():
        raise FileNotFoundError(default_path)
    default = yaml.safe_load(default_path.read_text(encoding="utf-8"))
    user = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(default, Mapping) or not isinstance(user, Mapping):
        raise ValueError("official training configs must have mapping roots")
    effective = dict(default)
    effective.update(dict(user))
    return effective


def _worker(
    *,
    repo_root: Path,
    config_path: Path,
    report_dir: Path,
    seed: int,
    bfloat16: int,
    torch_compile: int,
    gradient_accumulation_steps: int,
) -> int:
    """Run one exact official optimization step and write a rank report."""

    root = repo_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    report_dir.mkdir(parents=True, exist_ok=True)
    initialized = False
    try:
        import torch
        import torch.distributed as dist
        from copy import deepcopy

        # Import the upstream helpers rather than copying model/loss code.
        from distributed import init_distributed
        from train import update_ema, requires_grad
        from RAE.src.stage2.transport.transport import (
            ModelType,
            PathType,
            Transport,
            Sampler,
            WeightType,
        )
        from RAE.src.utils.model_utils import instantiate_from_config
        from RAE.src.utils.train_utils import parse_configs
        from models import CDiT_models
        config = _load_effective_training_config(root, config_path)
        world_hint = int(os.environ.get("WORLD_SIZE", "1"))
        assert_training_profile(
            config,
            epochs=50,
            world_size=world_hint,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        if int(bfloat16) != 1 or int(torch_compile) != 1:
            raise ValueError("memory preflight is fixed to BF16=1 and torch_compile=1")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; memory preflight is fail-closed")

        _, rank, gpu, _ = init_distributed()
        initialized = True
        world = dist.get_world_size()
        if world <= 0:
            raise RuntimeError("official distributed initialization returned invalid world size")
        plan = make_accumulation_plan(
            global_batch_size=int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]),
            world_size=world,
            accumulation_steps=gradient_accumulation_steps,
        )
        expected_local = plan.microbatch_size
        if int(config["batch_size"]) != expected_local:
            raise ValueError(
                "resolved config microbatch="
                f"{config['batch_size']} does not match world={world}, "
                f"accumulation={gradient_accumulation_steps}, effective global batch 96"
            )
        device = torch.device(f"cuda:{gpu}")
        torch.manual_seed(int(seed) * world + rank)

        # These construction steps intentionally mirror train.py's official
        # order and dimensions; no alternate architecture or reduced batch is
        # introduced.
        rae_config, *_ = parse_configs(config["config_path"])
        rae = instantiate_from_config(rae_config).to(device).eval()
        if int(config["image_size"]) % 14 != 0:
            raise ValueError("official image_size must be divisible by the DINOv2 patch size")
        latent_size = int(config["image_size"]) // 14
        learn_sigma_cfg = config.get("learn_sigma", False)
        if isinstance(learn_sigma_cfg, str):
            learn_sigma_cfg = learn_sigma_cfg.strip().lower() == "true"
        model_kwargs = {
            "context_size": int(config["context_size"]),
            "input_size": latent_size,
            "in_channels": rae.latent_dim,
            "learn_sigma": bool(learn_sigma_cfg),
            "head_width": config.get("head_width", rae.latent_dim),
            "head_depth": int(config.get("head_depth", 2)),
            "head_num_heads": int(config.get("head_num_heads", 16)),
        }
        model = CDiT_models[config["model"]](**model_kwargs).to(device)
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
        ema.eval()
        base_lr = float(config["lr"])
        betas = tuple(config.get("betas", (0.9, 0.95)))
        opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=betas, weight_decay=0.0)
        if int(torch_compile):
            if not hasattr(torch, "compile"):
                raise RuntimeError("official torch_compile=1 requested but torch.compile is unavailable")
            model = torch.compile(model)
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        shift_dim = int(rae.latent_dim) * latent_size * latent_size
        tp = config.get("transport", {})
        shift = math.sqrt(shift_dim / float(tp.get("time_dist_shift_base", 4096)))
        if tp.get("time_dist_shift") is not None:
            shift = float(tp["time_dist_shift"])
        if bool(tp.get("time_dist_shift_disable", False)):
            shift = 1.0
        transport = Transport(
            model_type=getattr(ModelType, str(tp.get("model_type", "velocity")).upper()),
            path_type=getattr(PathType, str(tp.get("path_type", "linear")).upper()),
            loss_type=getattr(WeightType, str(tp.get("loss_type", "velocity")).upper()),
            time_dist_type=str(tp.get("time_dist_type", "uniform")),
            time_dist_shift=shift,
            train_eps=1e-3,
            sample_eps=1e-3,
        )
        sampler_transport = Sampler(transport)
        del sampler_transport

        # The official loop creates its LambdaLR after the loader.  A one-step
        # probe cannot afford to materialize the full dataset/index, so use a
        # one-update schedule solely to exercise the same scheduler state
        # transition without changing the paper's declared linear endpoints.
        final_lr = float(config["final_lr"])
        base_lr = float(config["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda step: (final_lr / base_lr) * min(float(step), 1.0)
            + (1.0 - min(float(step), 1.0)),
        )

        # Establish the baseline after the frozen model/optimizer/DDP objects
        # are resident.  The reported peak is the additional allocation caused
        # by the real batch/forward/backward/step path, so the margin check is
        # meaningful on a 24-GiB card and does not confuse model residency with
        # available activation headroom.
        torch.cuda.synchronize(device)
        allocated_before = float(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        free_before, _ = torch.cuda.mem_get_info(device)

        local_batch = int(config["batch_size"])
        context_size = int(config["context_size"])
        goals = 4
        image_size = int(config["image_size"])
        accumulation_steps = require_positive_int(
            gradient_accumulation_steps, name="gradient_accumulation_steps"
        )
        # The shape is exactly the official TrainingDataset batch: context
        # frames followed by four sampled goals.  A fresh synthetic
        # microbatch is used for each accumulation slot, so the probe measures
        # the same peak activation path without retaining multiple graphs.
        model.train()
        scaler = torch.amp.GradScaler() if int(bfloat16) else None
        opt.zero_grad()
        micro_losses: list[torch.Tensor] = []
        for micro_idx in range(accumulation_steps):
            x = (
                torch.rand(
                    local_batch,
                    context_size + goals,
                    3,
                    image_size,
                    image_size,
                    device=device,
                )
                * 2.0
                - 1.0
            )
            y = torch.rand(local_batch * goals, 3, device=device)
            rel_t = torch.rand(local_batch * goals, device=device)
            sync_context = (
                model.no_sync()
                if micro_idx < accumulation_steps - 1 and hasattr(model, "no_sync")
                else contextlib.nullcontext()
            )
            with sync_context:
                with _autocast(True):
                    with torch.no_grad():
                        b, t = x.shape[:2]
                        encoded = rae.encode((x.flatten(0, 1) * 0.5) + 0.5).unflatten(0, (b, t))
                    num_goals = t - context_size
                    x_start = encoded[:, context_size:].flatten(0, 1)
                    x_cond = (
                        encoded[:, :context_size]
                        .unsqueeze(1)
                        .expand(
                            b,
                            num_goals,
                            context_size,
                            encoded.shape[2],
                            encoded.shape[3],
                            encoded.shape[4],
                        )
                        .flatten(0, 1)
                    )
                    terms = transport.training_losses(
                        model,
                        x_start,
                        {"y": y, "x_cond": x_cond, "rel_t": rel_t},
                    )
                    micro_loss = terms["loss"].mean()
                    scaled_loss = micro_loss / float(accumulation_steps)
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
            micro_losses.append(micro_loss.detach())
            # Release references before the next microbatch; gradients remain
            # attached to parameters while activations/inputs do not.
            del x, y, rel_t, encoded, x_start, x_cond, terms, micro_loss, scaled_loss

        loss = torch.stack(micro_losses).mean()
        clip_val = float(config.get("grad_clip_val", 1.0))
        if scaler is None:
            if clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            opt.step()
        else:
            if clip_val > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            scaler.step(opt)
            scaler.update()
        scheduler.step()
        update_ema(ema, model.module)
        torch.cuda.synchronize(device)
        peak_total_mib = float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
        peak_mib = max(0.0, peak_total_mib - allocated_before / (1024.0 * 1024.0))
        reserved_peak_mib = float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0)
        total_memory_mib = float(torch.cuda.get_device_properties(device).total_memory) / (1024.0 * 1024.0)
        free_mib = float(free_before) / (1024.0 * 1024.0)
        record = {
            "rank": int(rank),
            "free_mib_before": free_mib,
            "peak_memory_allocated_mib": peak_mib,
            "allocated_mib_before": allocated_before / (1024.0 * 1024.0),
            "peak_memory_reserved_mib": reserved_peak_mib,
            "total_memory_mib": total_memory_mib,
            "device_name": torch.cuda.get_device_name(device),
            "torch_cuda_version": str(torch.version.cuda),
            "optimizer_step": True,
            "scheduler_step": True,
            "gradient_accumulation_steps": accumulation_steps,
            "microbatch_size": local_batch,
            "microbatch_global_size": int(plan.microbatch_global_size),
            "effective_global_batch_size": int(plan.effective_global_batch_size),
            "microbatch_steps": accumulation_steps,
            "oom": False,
            "loss_finite": bool(torch.isfinite(loss.detach()).item()),
        }
        gathered: list[dict[str, Any] | None] = [None] * world
        dist.all_gather_object(gathered, record)
        if rank == 0:
            for item in gathered:
                if not isinstance(item, Mapping):
                    raise RuntimeError("official memory probe returned a malformed rank report")
            _atomic_json(report_dir / "all_ranks.json", {"records": gathered})
        dist.barrier()
        report_path = report_dir / f"rank_{rank}.json"
        _atomic_json(report_path, record)
        if not record["loss_finite"]:
            raise RuntimeError("memory preflight loss is non-finite")
        return 0
    except Exception as exc:
        rank = os.environ.get("RANK", "0")
        report_path = report_dir / f"rank_{rank}.json"
        try:
            _atomic_json(
                report_path,
                {"rank": int(rank), "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
            )
        except Exception:
            pass
        return 1
    finally:
        if initialized:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass


def run_memory_preflight(
    repo_root: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    world_size: int,
    gpu_indices: Sequence[int],
    gpu_uuids: Sequence[str],
    *,
    gradient_accumulation_steps: int = 1,
    python_executable: str = "python",
    hf_home: str | Path | None = None,
    nvidia_smi_executable: str = "nvidia-smi",
    seed: int = 42,
    bfloat16: int = 1,
    torch_compile: int = 1,
    safety_margin_mib: float = 1024.0,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Run the bounded probe and write a self-hashed PASS receipt.

    The function is intentionally explicit about every identity input.  It
    is safe to call from a scheduler wrapper; it does not choose GPUs or alter
    the selected topology implicitly.
    """

    root = Path(repo_root).resolve()
    indices = _normalize_indices(gpu_indices)
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if len(indices) != world_size:
        raise ValueError("GPU list length must equal world_size")
    uuids = _normalize_uuids(gpu_uuids, world_size)
    try:
        accumulation_steps = require_positive_int(
            gradient_accumulation_steps, name="gradient_accumulation_steps"
        )
        plan = make_accumulation_plan(
            global_batch_size=int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]),
            world_size=world_size,
            accumulation_steps=accumulation_steps,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid world/gradient accumulation mapping: {exc}") from exc
    if int(bfloat16) != 1 or int(torch_compile) != 1:
        raise ValueError("paper preflight requires bfloat16=1 and torch_compile=1")
    if hf_home is None:
        raise ValueError("memory preflight requires the dedicated verified --hf-home")
    try:
        margin = float(safety_margin_mib)
    except (TypeError, ValueError) as exc:
        raise ValueError("safety_margin_mib must be numeric") from exc
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("safety_margin_mib must be finite and nonnegative")
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout_seconds must be numeric") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")

    template = Path(config_path)
    if not template.is_absolute():
        template = root / template
    template = template.resolve()
    expected_template = (root / "config" / "rae_stream.yaml").resolve()
    if template != expected_template:
        raise ValueError(f"memory preflight requires the dedicated template {expected_template}")
    assert_training_profile(load_yaml_mapping(template), epochs=50, world_size=1)
    source = verify_upstream_source(root)
    runtime_config, resolved, runtime_info = materialize_runtime_config(
        root,
        template_path=template,
        world_size=world_size,
        gradient_accumulation_steps=accumulation_steps,
    )
    mapping = paper_batch_mapping(
        resolved,
        world_size=world_size,
        gradient_accumulation_steps=accumulation_steps,
    )
    # Resolve UUIDs from the live inventory when possible and refuse an
    # accidental index/UUID mismatch.  This is read-only and does not admit
    # unknown utilization or reserve a card.
    # The probe itself is a resource-consuming CUDA process.  Require the
    # same measured-utilization and UUID-keyed empty-process census as the
    # formal launcher before starting it; a free-memory-only snapshot is not
    # sufficient evidence of feasibility.
    inventory = query_gpus(executable=nvidia_smi_executable, include_apps=True)
    selected_live = choose_free_gpus(
        inventory,
        count=world_size,
        indices=indices,
        min_free_mib=4096,
        max_util=25,
        allow_unknown_util=False,
        require_apps_census=True,
    )
    by_index = {int(row["index"]): str(row.get("uuid", "")) for row in inventory}
    for index, uuid in zip(indices, uuids):
        if index not in by_index or not by_index[index]:
            raise RuntimeError(f"GPU index {index} disappeared before memory preflight")
        if by_index[index] != uuid:
            raise RuntimeError(f"GPU UUID changed before memory preflight for index {index}")
    if [int(row["index"]) for row in selected_live] != indices:
        raise RuntimeError("live GPU set changed before memory preflight")

    output = Path(output_path)
    if not output.is_absolute():
        output = root / output
    output = _validate_owned_runtime_path(root, output, label="memory preflight output", require_new=True)
    report_dir = output.parent / f".memory_probe_{os.getpid()}"
    if report_dir.exists():
        raise RuntimeError(f"memory probe report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True, exist_ok=False)
    log_path = report_dir / "probe.log"
    # Invoke the worker as a package module.  Executing the file path directly
    # would make its relative imports fail before the probe starts; ``cwd`` is
    # the dedicated root, so the package is importable without injecting a
    # caller-controlled PYTHONPATH.
    worker_args = [
        "--worker",
        "--repo-root",
        str(root),
        "--config",
        str(runtime_config),
        "--report-dir",
        str(report_dir),
        "--seed",
        str(int(seed)),
        "--bfloat16",
        "1",
        "--torch-compile",
        "1",
        "--gradient-accumulation-steps",
        str(accumulation_steps),
    ]
    python_info = str(python_executable)
    command = _worker_command(python_info, world_size, worker_args)
    environment = build_child_environment(
        {},
        gpu_indices=indices,
        hf_home=hf_home,
    )
    try:
        with log_path.open("x", encoding="utf-8", buffering=1) as log:
            completed = subprocess.run(
                command,
                cwd=str(root),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
    except Exception:
        # Keep the exact probe log/report directory for diagnosis; no broad
        # cleanup is attempted.
        raise
    if completed.returncode != 0:
        raise RuntimeError(
            f"official memory preflight failed with return code {completed.returncode}; see {log_path}"
        )
    records: list[Mapping[str, Any]] = []
    for rank in range(world_size):
        path = report_dir / f"rank_{rank}.json"
        if not path.is_file():
            raise RuntimeError(f"memory preflight did not produce rank report {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping) or value.get("status") == "FAIL":
            raise RuntimeError(f"memory preflight rank {rank} failed: {value}")
        records.append(value)
    free_values = [float(record["free_mib_before"]) for record in records]
    peak_values = [float(record["peak_memory_allocated_mib"]) for record in records]
    reserved_values = [float(record.get("peak_memory_reserved_mib", 0.0)) for record in records]
    total_values = [float(record.get("total_memory_mib", 0.0)) for record in records]
    if any(not math.isfinite(value) or value <= 0 for value in free_values + peak_values):
        raise RuntimeError("memory preflight returned invalid memory measurements")
    if any(not math.isfinite(value) or value < 0 for value in reserved_values) or any(
        not math.isfinite(value) or value <= 0 for value in total_values
    ):
        raise RuntimeError("memory preflight returned invalid allocator/device measurements")
    if any(peak + margin > free for peak, free in zip(peak_values, free_values)):
        raise RuntimeError("memory preflight peak plus safety margin exceeds free memory")
    payload: dict[str, Any] = {
        "status": "PASS",
        "probe_kind": "official_train_step_memory",
        "official_commit": OFFICIAL_COMMIT,
        "official_source_sha256": dict(OFFICIAL_RESTRICTED_SHA256),
        "official_source": source,
        "global_batch_size": mapping["global_batch_size"],
        "world_size": mapping["world_size"],
        "per_rank_batch_size": mapping["per_rank_batch_size"],
        "training_profile": training_profile_name(accumulation_steps),
        "paper_training_profile": PAPER_PROFILE_NAME,
        "evaluation_batch_size_per_rank": int(plan.microbatch_size),
        "evaluation_batch_semantics": (
            "physical_microbatch" if accumulation_steps > 1 else "paper_per_rank_batch"
        ),
        "gradient_accumulation_steps": accumulation_steps,
        "microbatch_global_size": int(plan.microbatch_global_size),
        "effective_global_batch_size": int(plan.effective_global_batch_size),
        "gpu_indices": indices,
        "gpu_uuids": uuids,
        "gpu_inventory": [dict(row) for row in selected_live],
        "config_sha256": str(runtime_info["config_sha256"]),
        "runtime_config": runtime_info,
        "probe_steps": 1,
        "optimizer_step": all(record.get("optimizer_step") is True for record in records),
        "scheduler_step": all(record.get("scheduler_step") is True for record in records),
        "oom": any(record.get("oom") is not False for record in records),
        "loss_finite": all(record.get("loss_finite") is True for record in records),
        "bfloat16": 1,
        "torch_compile": 1,
        "free_mib_before": free_values,
        "peak_memory_allocated_mib": peak_values,
        "peak_memory_reserved_mib": reserved_values,
        "total_memory_mib": total_values,
        "device_names": [str(record.get("device_name", "")) for record in records],
        "torch_cuda_versions": [str(record.get("torch_cuda_version", "")) for record in records],
        "safety_margin_mib": margin,
        "probe_command": command,
        "report_dir": str(report_dir),
        "python_executable": python_info,
    }
    if not payload["optimizer_step"] or payload["oom"] or not payload["loss_finite"]:
        raise RuntimeError("memory preflight did not complete a finite optimizer step")
    receipt = _with_self_hash(payload, "receipt_sha256")
    _atomic_json(output, receipt)
    result = dict(receipt)
    result["path"] = str(output)
    result["sha256"] = sha256_file(output)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--gpu-indices")
    parser.add_argument("--gpu-uuids")
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--hf-home", type=Path)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bfloat16", type=int, default=1)
    parser.add_argument("--torch-compile", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--safety-margin-mib", type=float, default=1024.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.worker:
            if args.report_dir is None:
                raise ValueError("worker requires --report-dir")
            return _worker(
                repo_root=args.repo_root.resolve(),
                config_path=args.config.resolve(),
                report_dir=args.report_dir.resolve(),
                seed=args.seed,
                bfloat16=args.bfloat16,
                torch_compile=args.torch_compile,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )
        if args.output is None or args.world_size is None or args.gpu_indices is None or args.gpu_uuids is None:
            raise ValueError("parent mode requires --output, --world-size, --gpu-indices and --gpu-uuids")
        indices = [int(value.strip()) for value in str(args.gpu_indices).split(",")]
        uuids = [value.strip() for value in str(args.gpu_uuids).split(",")]
        result = run_memory_preflight(
            args.repo_root,
            args.config,
            args.output,
            args.world_size,
            indices,
            uuids,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            python_executable=args.python_executable,
            hf_home=args.hf_home,
            nvidia_smi_executable=args.nvidia_smi,
            seed=args.seed,
            bfloat16=args.bfloat16,
            torch_compile=args.torch_compile,
            safety_margin_mib=args.safety_margin_mib,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(f"memory_preflight: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
