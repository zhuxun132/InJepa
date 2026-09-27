"""Bounded efficiency accounting for the Context4 offline suite."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from pathlib import Path
import resource
from typing import Any

import torch
from torch import nn

from j2j.evaluation.metrics import LatencyRecorder


def no_g_tree_nodes(horizon: int, *, branching: int = 3) -> int:
    """Return the exact number of unique non-empty prefixes in an F tree."""

    for name, value in (("horizon", horizon), ("branching", branching)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return sum(branching**depth for depth in range(1, horizon + 1))


def _model_device(model: nn.Module | None) -> torch.device | None:
    if model is None:
        return None
    parameters = tuple(model.parameters())
    return parameters[0].device if parameters else None


def _host_peak_rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB.  This project is Linux-only, but keep the
    # conversion isolated so receipts do not ambiguously label raw units.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def benchmark_decision(
    decision_fn: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    component_fns: Mapping[str, Callable[[], Any]] | None = None,
    component_fns_factory: Callable[[], Mapping[str, Callable[[], Any]]] | None = None,
    model: nn.Module | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    counters: Mapping[str, int] | None = None,
    synchronizer: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Measure a production-equivalent decision callable without I/O.

    The caller owns B=1 inputs and must disable diagnostics, shuffling and
    serialization before constructing ``decision_fn``.  Warm-up samples are
    retained but excluded from the steady-state summary.
    """

    if not callable(decision_fn):
        raise TypeError("decision_fn must be callable")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    if counters is not None:
        for name, value in counters.items():
            if not isinstance(name, str) or type(value) is not int or value < 0:
                raise ValueError("invocation counters must be non-negative integers")
    if component_fns is not None and component_fns_factory is not None:
        raise ValueError(
            "component_fns and component_fns_factory are mutually exclusive"
        )
    if component_fns_factory is not None and not callable(component_fns_factory):
        raise TypeError("component_fns_factory must be callable")
    components = dict(component_fns or {})
    allowed_components = {"q", "g", "f", "ranking"}
    if set(components) - allowed_components:
        raise ValueError("component_fns may contain only q/g/f/ranking")
    if any(not callable(fn) for fn in components.values()):
        raise TypeError("every component_fns value must be callable")

    device = _model_device(model)
    cuda_device = device if device is not None and device.type == "cuda" else None
    effective_synchronizer = synchronizer
    if effective_synchronizer is None and cuda_device is not None:
        effective_synchronizer = lambda: torch.cuda.synchronize(cuda_device)
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)

    recorder = LatencyRecorder(
        warmup_samples=warmup,
        synchronizer=effective_synchronizer,
    )
    last_result: Any = None
    with torch.inference_mode():
        for _ in range(warmup + repeats):
            last_result = recorder.measure("decision_cycle_ms", decision_fn)

    # Snapshot the production whole-decision footprint before constructing or
    # running any isolated component fixture.  Captured G/F arguments may be
    # large and must not be mistaken for whole-decision peak VRAM.
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(cuda_device))
        if cuda_device is not None
        else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(cuda_device))
        if cuda_device is not None
        else None
    )

    if component_fns_factory is not None:
        built = component_fns_factory()
        if not isinstance(built, Mapping):
            raise TypeError("component_fns_factory must return a mapping")
        components = dict(built)
        if set(components) - allowed_components:
            raise ValueError("component_fns may contain only q/g/f/ranking")
        if any(not callable(fn) for fn in components.values()):
            raise TypeError("every component_fns value must be callable")

    component_ms: dict[str, Any] = {}
    with torch.inference_mode():
        for name, fn in sorted(components.items()):
            component_recorder = LatencyRecorder(
                warmup_samples=warmup,
                synchronizer=effective_synchronizer,
            )
            for _ in range(warmup + repeats):
                component_recorder.measure("decision_cycle_ms", fn)
            component_ms[name] = component_recorder.summary()["decision_cycle_ms"]

    summary = recorder.summary()["decision_cycle_ms"]
    checkpoint_bytes: int | None = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint is unavailable: {path}")
        checkpoint_bytes = path.stat().st_size
    trainable_parameters = (
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if model is not None
        else None
    )
    result_counters = getattr(last_result, "counters", None)
    if counters is None and isinstance(result_counters, Mapping):
        counters = {str(name): int(value) for name, value in result_counters.items()}

    return {
        "schema": "J2J_CONTEXT4_LATENCY_V1",
        "warmup": warmup,
        "repeats": repeats,
        "decision_cycle_ms": summary,
        "component_ms": component_ms,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "host_peak_rss_bytes": _host_peak_rss_bytes(),
        "checkpoint_bytes": checkpoint_bytes,
        "trainable_parameters": trainable_parameters,
        "invocations_per_decision": dict(counters or {}),
        "profiler_enabled": False,
        "component_measurement": {
            "mode": "isolated_existing_call_replay",
            "included_in_decision_cycle_ms": False,
            "warmup": warmup,
            "repeats": repeats,
        },
    }


def throughput_summary(*, rows: int, trajectories: int, elapsed_seconds: float) -> dict[str, Any]:
    """Summarize traversal throughput separately from decision latency."""

    for name, value in (("rows", rows), ("trajectories", trajectories)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(elapsed_seconds, bool) or not isinstance(elapsed_seconds, (int, float)):
        raise TypeError("elapsed_seconds must be numeric")
    elapsed = float(elapsed_seconds)
    if not torch.isfinite(torch.tensor(elapsed)) or elapsed <= 0.0:
        raise ValueError("elapsed_seconds must be finite and positive")
    return {
        "elapsed_seconds": elapsed,
        "rows_per_second": rows / elapsed,
        "trajectories_per_second": trajectories / elapsed,
    }


__all__ = ["benchmark_decision", "no_g_tree_nodes", "throughput_summary"]
