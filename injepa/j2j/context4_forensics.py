"""Read-only numerical observations and finite-gradient forensic evidence."""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

import math
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


@torch.no_grad()
def layernorm_input_statistics(module: nn.LayerNorm, input_tensor: Tensor) -> dict:
    """Population variance and an upper bound, including every executed row.

    This does not identify padding, functional layer_norm calls or the actual
    gradient gain. The bound follows ||J_LN||₂ <= ||gamma||∞ / sqrt(var+eps).
    """
    if not isinstance(module, nn.LayerNorm):
        raise TypeError("statistics require an explicit LayerNorm module")
    shape = tuple(module.normalized_shape)
    if tuple(input_tensor.shape[-len(shape):]) != shape or not input_tensor.numel():
        raise ValueError("LayerNorm input shape or row count is invalid")
    values = input_tensor.detach().to(dtype=torch.float64).reshape(-1, math.prod(shape))
    variance = values.var(dim=-1, unbiased=False)
    minimum, mean, maximum = (float(value) for value in
                              torch.stack((variance.amin(), variance.mean(), variance.amax())).cpu())
    effective_weight = getattr(module, "effective_weight", module.weight)
    gamma = 1.0 if effective_weight is None else float(effective_weight.detach().double().abs().amax().cpu())
    result = {"scope": "all_executed_rows", "row_count": values.shape[0],
              "variance_min": minimum, "variance_mean": mean, "variance_max": maximum,
              "epsilon": float(module.eps), "gamma_maxabs": gamma,
              "jacobian_l2_upper_bound_max": gamma / math.sqrt(minimum + module.eps)}
    if any(not math.isfinite(value) for value in result.values() if isinstance(value, (int, float))):
        raise FloatingPointError("LayerNorm input statistics are non-finite")
    return result


@torch.no_grad()
def packed_qkv_gradient_statistics(attention: nn.MultiheadAttention) -> dict:
    """Partition accumulated pre-clip packed projection gradients, Q then K then V."""
    if (not isinstance(attention, nn.MultiheadAttention)
            or attention.in_proj_weight is None
            or tuple(attention.in_proj_weight.shape) != (3 * attention.embed_dim, attention.embed_dim)):
        raise ValueError("statistics require equal-width packed QKV projection")
    result = {}
    for kind, parameter in (("weight", attention.in_proj_weight), ("bias", attention.in_proj_bias)):
        if parameter is None or parameter.grad is None:
            result[kind] = None
            continue
        values = parameter.grad.detach().double().reshape(3, -1)
        norms = torch.linalg.vector_norm(values, dim=1).cpu().tolist()
        maxima = values.abs().amax(dim=1).cpu().tolist()
        if any(not math.isfinite(value) for value in norms + maxima):
            raise FloatingPointError("packed QKV gradient statistics are non-finite")
        result[kind] = {name: {"numel": values.shape[1], "l2": norm, "max_abs": maximum}
                        for name, norm, maximum in zip(("q", "k", "v"), norms, maxima, strict=True)}
    return result


def _cpu_copy(value):
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu", copy=True)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _cpu_copy(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _cpu_copy(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(child) for child in value)
    if isinstance(value, list):
        return [_cpu_copy(child) for child in value]
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    raise TypeError(f"unsupported forensic payload type: {type(value).__name__}")


def _rng_state(device: torch.device) -> dict:
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": {device.index: torch.cuda.get_rng_state(device).clone()} if device.type == "cuda" else None}


def _all_values(value, world_size: int) -> list:
    if world_size == 1:
        return [value]
    values = [None] * world_size
    torch.distributed.all_gather_object(values, value)
    return values


def _agree_error(error: str | None, world_size: int) -> None:
    errors = _all_values(error, world_size)
    if any(item is not None for item in errors):
        raise RuntimeError(f"finite gradient capture failed before optimizer: {errors}")


def _save_exclusive(path: Path, value) -> dict:
    with path.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_capture_config(config: Mapping) -> dict:
    if not isinstance(config, Mapping) or set(config) != {"q_preclip_threshold", "max_events"}:
        raise ValueError("anomaly_forensics requires exactly q_preclip_threshold and max_events")
    threshold, maximum = config["q_preclip_threshold"], config["max_events"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError("q_preclip_threshold must be finite and positive")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("max_events must be a positive integer")
    return {"q_preclip_threshold": float(threshold), "max_events": maximum}


class FiniteGradientCapture:
    """Bounded post-backward, pre-clip evidence; never an exact-resume checkpoint.

    Capture failure is coordinated for ranks that have completed backward. This
    observer cannot recover from rank death, a hung filesystem or a failed DDP
    backward. Ordinary execution retains only RNG bytes and input references.
    """

    def __init__(self, output_root, *, q_preclip_threshold: float, max_events: int):
        validate_capture_config({"q_preclip_threshold": q_preclip_threshold, "max_events": max_events})
        self.output_root = Path(output_root)
        self.q_preclip_threshold = float(q_preclip_threshold)
        self.max_events = max_events
        existing = tuple(self.output_root.glob("step_*"))
        if any(not (path / "manifest.json").is_file() for path in existing):
            raise ValueError("incomplete forensic event requires explicit recovery")
        self.events = len(existing)

    def begin_update(self, *, model, microbatches, optimizer, scheduler, world_size,
                     successful_updates, global_denominators, context):
        if self.events >= self.max_events:
            return _InactiveCaptureSession()
        if world_size > 1 and (not torch.distributed.is_initialized()
                              or torch.distributed.get_world_size() != world_size):
            raise ValueError("forensic world size disagrees with the active process group")
        if not isinstance(context, Mapping):
            raise TypeError("forensic context must be a mapping")
        # Reject unserializable context before forward, without advancing RNG.
        context = json.loads(json.dumps(dict(context), allow_nan=False))
        wrapper = getattr(model, "module", model)
        joint = getattr(wrapper, "joint_model", wrapper)
        device = next(joint.parameters()).device
        return _FiniteCaptureSession(self, joint, tuple(microbatches), optimizer, scheduler,
                                     world_size, successful_updates, dict(global_denominators),
                                     context, _rng_state(device), device)


class _InactiveCaptureSession:
    def capture_if_needed(self, q_preclip: float):
        return None


class _FiniteCaptureSession:
    def __init__(self, owner, model, slots, optimizer, scheduler, world_size,
                 successful_updates, denominators, context, rng_start, device):
        self.owner, self.model, self.slots = owner, model, slots
        self.optimizer, self.scheduler = optimizer, scheduler
        self.world_size, self.successful_updates = world_size, successful_updates
        self.denominators, self.context = denominators, context
        self.rng_start, self.device = rng_start, device

    def capture_if_needed(self, q_preclip: float):
        norms = _all_values(float(q_preclip), self.world_size)
        if any(not math.isfinite(value) for value in norms):
            raise FloatingPointError("finite capture does not admit non-finite backward results")
        if max(norms) <= self.owner.q_preclip_threshold:
            return None
        rank = torch.distributed.get_rank() if self.world_size > 1 else 0
        event = self.owner.output_root / f"step_{self.successful_updates + 1:09d}"
        rng_end = _rng_state(self.device)
        error = None
        if rank == 0:
            try:
                event.mkdir(parents=True, exist_ok=False)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        _agree_error(error, self.world_size)
        files = {}
        error = None
        try:
            payload = {"rank": rank, "world_size": self.world_size,
                       "microbatches": _cpu_copy(self.slots), "rng_start": self.rng_start,
                       "rng_end": rng_end, "context": self.context}
            rank_name = f"rank_{rank:04d}.pt"
            files[rank_name] = _save_exclusive(event / rank_name, payload)
            del payload
            if rank == 0:
                shared = {"model": _cpu_copy(self.model.state_dict()),
                          "optimizer": _cpu_copy(self.optimizer.state_dict()),
                          "scheduler": _cpu_copy(self.scheduler.state_dict()),
                          "gradients": {name: _cpu_copy(parameter.grad) for name, parameter in self.model.named_parameters()},
                          "module_training_flags": {name: child.training for name, child in self.model.named_modules()}}
                files["shared.pt"] = _save_exclusive(event / "shared.pt", shared)
                del shared
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        _agree_error(error, self.world_size)
        all_files = _all_values(files, self.world_size)
        error = None
        if rank == 0:
            try:
                manifest = {"schema": "j2j.finite_gradient_forensics.v1", "forensic_not_resume": True,
                            "state_phase": "post_backward_pre_clip_pre_optimizer",
                            "original_forward_trace": "not_captured; scheduled diagnostics are separate",
                            "successful_updates": self.successful_updates,
                            "attempted_update": self.successful_updates + 1, "world_size": self.world_size,
                            "global_denominators": self.denominators, "context": self.context,
                            "q_preclip_by_rank": norms, "q_preclip_threshold": self.owner.q_preclip_threshold,
                            "files": {name: identity for group in all_files for name, identity in group.items()}}
                pending = event / "manifest.pending.json"
                try:
                    with pending.open("x", encoding="utf-8") as stream:
                        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
                        stream.flush()
                        os.fsync(stream.fileno())
                    # Atomic create-only publication, never replace an event.
                    os.link(pending, event / "manifest.json")
                finally:
                    pending.unlink(missing_ok=True)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        _agree_error(error, self.world_size)
        self.owner.events += 1
        return str(event / "manifest.json")
