"""Habitat-compatible metric and latency accounting for evaluation only."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import numbers
import statistics
import time
from typing import Any, Callable

from j2j.adapter import ActionId


METRIC_NAMES = (
    "success",
    "spl",
    "soft_spl",
    "distance_to_goal",
    "path_length",
    "num_steps",
    "collisions",
    "timeout",
)

LATENCY_NAMES = (
    "policy_inference_ms",
    "decision_cycle_ms",
    "environment_step_ms",
    "episode_wall_ms",
)


def _finite(value: Any, name: str) -> float:
    # Habitat-Sim exposes positions as ``numpy.ndarray`` with ``numpy.float32``
    # elements.  ``numbers.Real`` covers those scalar types without making
    # NumPy a hard dependency of the pure evaluator.
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        # A zero-dimensional tensor/scalar is also a legitimate backend value;
        # convert only when it has an unambiguous ``item`` method.
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                value = None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _position(value: Any, name: str = "position") -> tuple[float, ...]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of real values")
    result = tuple(_finite(item, name) for item in value)
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def _action_id(action: Any) -> ActionId:
    if isinstance(action, ActionId):
        return action
    if isinstance(action, bool):
        raise TypeError("action must be an ImageNav ActionId")
    if isinstance(action, str):
        names = {
            "STOP": ActionId.STOP,
            "FWD": ActionId.FWD,
            "MOVE_FORWARD": ActionId.FWD,
            "LEFT": ActionId.LEFT,
            "TURN_LEFT": ActionId.LEFT,
            "RIGHT": ActionId.RIGHT,
            "TURN_RIGHT": ActionId.RIGHT,
        }
        try:
            return names[action.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown action {action!r}") from exc
    try:
        return ActionId(int(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown action {action!r}") from exc


def _euclidean(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("positions must have equal dimensions")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    # NumPy's default linear interpolation, implemented without a NumPy
    # dependency so this module remains usable in the pure contract runner.
    index = (len(ordered) - 1) * q / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(values: Iterable[float]) -> dict[str, Any]:
    """Return finite count/mean/median/p50/p95/p99/min/max statistics."""

    samples = [_finite(value, "sample") for value in values]
    if not samples:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p50": _percentile(samples, 50.0),
        "p95": _percentile(samples, 95.0),
        "p99": _percentile(samples, 99.0),
        "min": min(samples),
        "max": max(samples),
    }


@dataclass(frozen=True)
class LatencySample:
    """One measured latency sample and its warm-up label."""

    milliseconds: float
    warmup: bool


class LatencyRecorder:
    """Monotonic timing recorder with explicit warm-up/steady-state groups."""

    def __init__(
        self,
        *,
        warmup_samples: int = 0,
        synchronizer: Callable[[], None] | None = None,
    ) -> None:
        if isinstance(warmup_samples, bool) or not isinstance(warmup_samples, int):
            raise TypeError("warmup_samples must be an integer")
        if warmup_samples < 0:
            raise ValueError("warmup_samples must be nonnegative")
        self.warmup_samples = warmup_samples
        self._samples: dict[str, list[LatencySample]] = defaultdict(list)
        self._synchronizer = synchronizer

    @staticmethod
    def synchronize_cuda() -> None:
        """Synchronize CUDA when Torch and a CUDA device are available."""

        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            # Timing must remain usable for CPU-only contract tests.  A custom
            # synchronizer can be supplied when a backend needs strict errors.
            return

    def _sync(self) -> None:
        if self._synchronizer is not None:
            self._synchronizer()
        else:
            self.synchronize_cuda()

    # Public spelling used by the runner at asynchronous GPU policy timing
    # boundaries.  Keeping it separate from ``record`` makes the timing
    # contract explicit and easy to replace in a backend test.
    def synchronize(self) -> None:
        self._sync()

    def record(
        self,
        name: str,
        milliseconds: float,
        *,
        warmup: bool | None = None,
    ) -> float:
        if name not in LATENCY_NAMES:
            raise ValueError(f"unknown latency name {name!r}")
        value = _finite(milliseconds, f"{name}.milliseconds")
        if value < 0.0:
            raise ValueError(f"{name}.milliseconds must be nonnegative")
        if warmup is None:
            warmup = len(self._samples[name]) < self.warmup_samples
        if not isinstance(warmup, bool):
            raise TypeError("warmup must be boolean")
        self._samples[name].append(LatencySample(value, warmup))
        return value

    def measure(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        warmup: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call ``fn`` and record elapsed monotonic wall time in milliseconds."""

        if not callable(fn):
            raise TypeError("fn must be callable")
        self._sync()
        start = time.perf_counter_ns()
        try:
            return_value = fn(*args, **kwargs)
        finally:
            self._sync()
            elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
            self.record(name, elapsed, warmup=warmup)
        return return_value

    def samples(self, name: str, *, group: str = "all") -> list[float]:
        if name not in LATENCY_NAMES:
            raise ValueError(f"unknown latency name {name!r}")
        if group not in {"all", "warmup", "steady_state"}:
            raise ValueError("group must be all, warmup, or steady_state")
        values = self._samples[name]
        if group == "warmup":
            return [sample.milliseconds for sample in values if sample.warmup]
        if group == "steady_state":
            return [sample.milliseconds for sample in values if not sample.warmup]
        return [sample.milliseconds for sample in values]

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in LATENCY_NAMES:
            all_summary = summarize_samples(self.samples(name))
            result[name] = {
                # Keep the flat fields convenient for tabular reports while
                # retaining explicit warm-up/steady-state groups.
                **all_summary,
                "all": all_summary,
                "warmup": summarize_samples(self.samples(name, group="warmup")),
                "steady_state": summarize_samples(
                    self.samples(name, group="steady_state")
                ),
                "samples_ms": self.samples(name),
                "warmup_flags": [sample.warmup for sample in self._samples[name]],
            }
        return result

    # Convenient alias for receipt writers.
    to_dict = summary


class HabitatMetricsAccumulator:
    """Pure fallback implementation matching Habitat-Lab navigation measures.

    In a real Habitat run the runner prefers the values returned by
    ``env.get_metrics()``.  This accumulator exists for deterministic smoke
    tests and for environments exposing only raw position/distance values; it
    never consumes model predictions.
    """

    def __init__(self, *, success_distance: float = 0.2) -> None:
        self.success_distance = _finite(success_distance, "success_distance")
        if self.success_distance < 0.0:
            raise ValueError("success_distance must be nonnegative")
        self._started = False
        self._start_position: tuple[float, ...] = ()
        self._previous_position: tuple[float, ...] = ()
        self._start_distance = 0.0
        self._distance_to_goal = 0.0
        self._path_length = 0.0
        self._num_steps = 0
        self._collisions = 0
        self._stopped = False
        self._timeout = False

    def reset(self, *, start_position: Any, start_distance: float) -> None:
        self._start_position = _position(start_position, "start_position")
        self._previous_position = self._start_position
        self._start_distance = _finite(start_distance, "start_distance")
        if self._start_distance < 0.0:
            raise ValueError("start_distance must be nonnegative")
        self._distance_to_goal = self._start_distance
        self._path_length = 0.0
        self._num_steps = 0
        self._collisions = 0
        self._stopped = False
        self._timeout = False
        self._started = True

    def observe(
        self,
        *,
        position: Any,
        distance_to_goal: float,
        action: Any,
        collisions: int | bool | None = None,
        collision: bool | None = None,
        timeout: bool = False,
        done: bool = False,
        stop_called: bool | None = None,
    ) -> None:
        if not self._started:
            raise RuntimeError("reset must be called before observe")
        if stop_called is None:
            observed_stop = _action_id(action) == ActionId.STOP
        else:
            if not isinstance(stop_called, bool):
                raise TypeError("stop_called must be boolean when supplied")
            if action != "CONTINUOUS" and (_action_id(action) == ActionId.STOP) != stop_called:
                raise ValueError("action and stop_called disagree")
            observed_stop = stop_called
        current = _position(position)
        distance = _finite(distance_to_goal, "distance_to_goal")
        if distance < 0.0:
            raise ValueError("distance_to_goal must be nonnegative")
        # Habitat's SPL measure accumulates displacement between successive
        # simulator states.  A STOP action itself normally leaves the state
        # unchanged; if a test double reports a changed state, preserving the
        # reported displacement is the least surprising official semantics.
        self._path_length += _euclidean(self._previous_position, current)
        self._previous_position = current
        self._distance_to_goal = distance
        self._num_steps += 1
        self._stopped = self._stopped or observed_stop
        if collisions is not None:
            if isinstance(collisions, bool):
                self._collisions += int(collisions)
            elif isinstance(collisions, int) and collisions >= 0:
                # A measure may report either a per-step count or a cumulative
                # count.  Taking the maximum handles both without double count.
                self._collisions = max(self._collisions, collisions)
            else:
                raise TypeError("collisions must be a nonnegative integer/bool")
        if collision is not None:
            if not isinstance(collision, bool):
                raise TypeError("collision must be boolean")
            self._collisions += int(collision)
        if not isinstance(timeout, bool) or not isinstance(done, bool):
            raise TypeError("timeout and done must be boolean")
        self._timeout = self._timeout or timeout

    def mark_timeout(self) -> None:
        if not self._started:
            raise RuntimeError("reset must be called before mark_timeout")
        self._timeout = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    def finalize(
        self,
        *,
        timeout: bool | None = None,
        official_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("reset must be called before finalize")
        if timeout is not None:
            if not isinstance(timeout, bool):
                raise TypeError("timeout must be boolean")
            self._timeout = self._timeout or timeout
        if self._start_distance > 0.0:
            denominator = max(self._start_distance, self._path_length)
            progress = max(
                0.0,
                1.0 - self._distance_to_goal / self._start_distance,
            )
            efficiency = self._start_distance / denominator
        else:
            # Degenerate episodes are not expected in MP3D.  Keep all outputs
            # finite and give a zero score unless no movement was needed and a
            # valid STOP was issued.
            efficiency = 1.0 if self._path_length == 0.0 else 0.0
            progress = 1.0 if self._distance_to_goal == 0.0 else 0.0
        success = float(
            self._stopped and self._distance_to_goal < self.success_distance
        )
        result: dict[str, Any] = {
            "success": success,
            "spl": success * efficiency,
            "soft_spl": progress * efficiency,
            "distance_to_goal": self._distance_to_goal,
            "path_length": self._path_length,
            "num_steps": self._num_steps,
            "steps": self._num_steps,
            "collisions": self._collisions,
            "timeout": bool(self._timeout),
            "start_distance": self._start_distance,
        }
        if official_metrics is not None:
            for name in ("success", "spl", "soft_spl", "distance_to_goal"):
                if name in official_metrics and official_metrics[name] is not None:
                    result[name] = _finite(official_metrics[name], name)
        return result


def aggregate_episode_metrics(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode numeric measures by arithmetic mean."""

    rows = list(episodes)
    result: dict[str, Any] = {"episode_count": len(rows)}
    for name in METRIC_NAMES:
        values = []
        for row in rows:
            if name in row and row[name] is not None:
                value = row[name]
                if isinstance(value, bool):
                    value = float(value)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
        result[name] = statistics.fmean(values) if values else None
        result[f"{name}_count"] = len(values)
    return result


def aggregate_episode_latency(
    episodes: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool measured latency samples across episodes.

    ``run_imagegoal_episode`` records raw per-call samples and warm-up labels
    for every episode.  The final evaluation receipt needs one comparable
    summary for the complete official split, so this helper combines those
    samples without averaging already-aggregated episode statistics.  Missing
    latency rows are treated as empty (useful for metadata-only fixtures),
    while malformed sample/flag lengths are rejected rather than producing a
    misleading timing table.
    """

    recorder = LatencyRecorder(warmup_samples=0)
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TypeError(f"episodes[{episode_index}] must be a mapping")
        latency = episode.get("latency")
        if latency is None:
            continue
        if not isinstance(latency, Mapping):
            raise TypeError(f"episodes[{episode_index}].latency must be a mapping")
        for name in LATENCY_NAMES:
            row = latency.get(name)
            if row is None:
                continue
            if not isinstance(row, Mapping):
                raise TypeError(
                    f"episodes[{episode_index}].latency.{name} must be a mapping"
                )
            samples = row.get("samples_ms", [])
            flags = row.get("warmup_flags", [])
            if not isinstance(samples, (list, tuple)):
                raise TypeError(
                    f"episodes[{episode_index}].latency.{name}.samples_ms must be a sequence"
                )
            if not isinstance(flags, (list, tuple)):
                raise TypeError(
                    f"episodes[{episode_index}].latency.{name}.warmup_flags must be a sequence"
                )
            if len(samples) != len(flags):
                raise ValueError(
                    f"episodes[{episode_index}].latency.{name} samples/flags length mismatch"
                )
            for sample, warmup in zip(samples, flags):
                recorder.record(name, sample, warmup=warmup)
    return recorder.summary()
