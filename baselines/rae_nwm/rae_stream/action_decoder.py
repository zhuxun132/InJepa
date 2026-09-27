"""External continuous-to-Habitat action adapter for RAE-stream.

The RAE planner predicts the first two components in the same normalized
coordinate system used by the official repository (``[-64, 64]`` metric
waypoint bounds mapped to ``[-1, 1]``).  Yaw is already expressed in radians
and is *not* passed through that XY normalization.  This module deliberately
contains no model or simulator code: it only provides the fixed, auditable
compatibility mapping from a three-dimensional command to one of Habitat's
four primitive actions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
from typing import Any

import numpy as np


# Keep these integer values aligned with the StreamVLN/Habitat action ABI.
ACTION_STOP = 0
ACTION_FWD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

FORWARD_METERS = 0.25
TURN_RADIANS = math.pi / 12.0
DEFAULT_XY_MIN = -64.0
DEFAULT_XY_MAX = 64.0


def _finite_real(value: Any, name: str) -> float:
    """Convert a scalar tensor/NumPy value while rejecting bool/NaN/Inf."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real scalar")
    if not isinstance(value, numbers.Real):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception as exc:  # pragma: no cover - defensive tensor ABI
                raise TypeError(f"{name} must be a finite real scalar") from exc
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _triplet(command: Any) -> tuple[float, float, float]:
    """Read an explicit ``(dx, dy, dyaw)`` command from common tensor ABIs."""

    if isinstance(command, Mapping):
        # The planner seam may expose either named NWM fields or a nested
        # continuous_action payload.  Do not silently accept an ambiguous
        # mapping with missing fields.
        if "continuous_action" in command:
            return _triplet(command["continuous_action"])
        if "command" in command and not any(
            key in command for key in ("dx", "dy", "dyaw", "u_x", "u_y", "omega")
        ):
            return _triplet(command["command"])
        if "action_continuous" in command:
            return _triplet(command["action_continuous"])
        if "action" in command and not any(
            key in command for key in ("dx", "dy", "dyaw", "u_x", "u_y", "omega")
        ):
            return _triplet(command["action"])
        keys = (
            ("dx", "u_x", "x"),
            ("dy", "u_y", "y"),
            ("dyaw", "omega", "yaw", "theta"),
        )
        values: list[Any] = []
        for choices in keys:
            found = next((command[key] for key in choices if key in command), None)
            if found is None:
                raise ValueError("continuous command must contain dx, dy, dyaw")
            values.append(found)
        return tuple(_finite_real(value, f"command[{i}]") for i, value in enumerate(values))  # type: ignore[return-value]

    if isinstance(command, (str, bytes)):
        raise TypeError("continuous command must contain dx, dy, dyaw")
    if isinstance(command, np.ndarray):
        if command.ndim != 1:
            raise ValueError("continuous command must be one single (3,) vector")
        values = command.tolist()
    else:
        detach = getattr(command, "detach", None)
        if callable(detach):
            try:
                tensor = command.detach().cpu()
                if getattr(tensor, "ndim", None) != 1:
                    raise ValueError("continuous command must be one single (3,) vector")
                values = tensor.tolist()
            except ValueError:
                raise
            except Exception as exc:  # pragma: no cover - optional torch ABI
                raise TypeError("continuous command must contain dx, dy, dyaw") from exc
        elif callable(getattr(command, "tolist", None)):
            try:
                converted = command.tolist()
            except Exception as exc:  # pragma: no cover
                raise TypeError("continuous command must contain dx, dy, dyaw") from exc
            if isinstance(converted, (list, tuple)) and any(
                isinstance(item, (list, tuple, np.ndarray)) for item in converted
            ):
                raise ValueError("continuous command must be one single (3,) vector")
            values = converted if isinstance(converted, (list, tuple)) else [converted]
        elif isinstance(command, Sequence):
            values = list(command)
        else:
            try:
                values = list(command)
            except TypeError as exc:
                raise TypeError("continuous command must contain dx, dy, dyaw") from exc

    # A batch-shaped tensor is never silently flattened into a first action;
    # only one command is accepted at this seam.
    if len(values) != 3:
        raise ValueError("continuous command must have exactly three values")
    return tuple(_finite_real(value, f"command[{i}]") for i, value in enumerate(values))  # type: ignore[return-value]


def _stats_bounds(action_stats: Any) -> tuple[float, float]:
    """Extract official XY min/max from a mapping or a two-item pair."""

    if action_stats is None:
        return DEFAULT_XY_MIN, DEFAULT_XY_MAX
    if isinstance(action_stats, Mapping):
        minimum = action_stats.get("min")
        maximum = action_stats.get("max")
    else:
        try:
            minimum, maximum = action_stats
        except (TypeError, ValueError) as exc:
            raise ValueError("action_stats must provide min and max XY bounds") from exc
    if minimum is None or maximum is None:
        raise ValueError("action_stats must provide min and max XY bounds")
    # Official config stores two-element arrays.  A scalar pair is accepted as
    # a convenience only when both values are scalars.
    if isinstance(minimum, numbers.Real):
        min_values = [minimum, minimum]
    else:
        min_values = list(minimum)
    if isinstance(maximum, numbers.Real):
        max_values = [maximum, maximum]
    else:
        max_values = list(maximum)
    if len(min_values) != 2 or len(max_values) != 2:
        raise ValueError("action_stats min/max must each have two XY values")
    lo = _finite_real(min_values[0], "action_stats.min[0]")
    hi = _finite_real(max_values[0], "action_stats.max[0]")
    if not math.isclose(lo, _finite_real(min_values[1], "action_stats.min[1]")):
        raise ValueError("anisotropic XY bounds are unsupported by the fixed decoder")
    if not math.isclose(hi, _finite_real(max_values[1], "action_stats.max[1]")):
        raise ValueError("anisotropic XY bounds are unsupported by the fixed decoder")
    if not hi > lo:
        raise ValueError("action_stats max must exceed min")
    return lo, hi


def unnormalize_xy(
    xy: Sequence[float] | np.ndarray,
    *,
    spacing: float = FORWARD_METERS,
    action_stats: Any = None,
) -> np.ndarray:
    """Convert official normalized XY to metric metres.

    ``action_stats`` follows ``config/data_config.yaml``.  With the default
    ``[-64, 64]`` bounds, normalized ``1/64`` is one waypoint unit and hence
    ``0.25 m`` at the default spacing.  The yaw component is intentionally
    handled separately and is never scaled here.
    """

    if isinstance(xy, (str, bytes)):
        raise TypeError("xy must contain exactly two numeric values")
    try:
        values = list(xy)
    except TypeError as exc:
        raise TypeError("xy must contain exactly two numeric values") from exc
    if len(values) != 2:
        raise ValueError("xy must contain exactly two values")
    spacing_value = _finite_real(spacing, "spacing")
    if spacing_value <= 0.0:
        raise ValueError("spacing must be positive")
    lo, hi = _stats_bounds(action_stats)
    normalized = np.asarray([_finite_real(v, f"xy[{i}]") for i, v in enumerate(values)], dtype=np.float64)
    # Inverse of official misc.normalize_data: n = 2*(raw-lo)/(hi-lo)-1.
    raw = (normalized + 1.0) * 0.5 * (hi - lo) + lo
    return raw * spacing_value


def normalize_xy(
    metric_xy: Sequence[float] | np.ndarray,
    *,
    spacing: float = FORWARD_METERS,
    action_stats: Any = None,
) -> np.ndarray:
    """Apply the official XY metric-to-normalized affine transform."""

    if isinstance(metric_xy, (str, bytes)):
        raise TypeError("metric_xy must contain exactly two numeric values")
    values = list(metric_xy)
    if len(values) != 2:
        raise ValueError("metric_xy must contain exactly two values")
    spacing_value = _finite_real(spacing, "spacing")
    if spacing_value <= 0.0:
        raise ValueError("spacing must be positive")
    lo, hi = _stats_bounds(action_stats)
    raw = np.asarray([_finite_real(v, f"metric_xy[{i}]") for i, v in enumerate(values)], dtype=np.float64) / spacing_value
    return 2.0 * (raw - lo) / (hi - lo) - 1.0


@dataclass(frozen=True)
class DecodedAction:
    """Auditable result of one fixed compatibility decode."""

    action_name: str
    action: int
    metric_delta: tuple[float, float, float]
    distance: float
    normalized_delta: tuple[float, float, float]


class RAEStreamActionDecoder:
    """Nearest-neighbour decoder for the four Habitat primitives.

    The class is intentionally stateless with respect to model predictions;
    it records only optional conversion events for a run receipt.  A distance
    threshold is fail-closed: an out-of-codebook command raises instead of
    silently executing a potentially unsafe primitive.
    """

    _ORDER = (ACTION_STOP, ACTION_FWD, ACTION_LEFT, ACTION_RIGHT)
    _NAMES = {ACTION_STOP: "STOP", ACTION_FWD: "FWD", ACTION_LEFT: "LEFT", ACTION_RIGHT: "RIGHT"}

    def __init__(
        self,
        *,
        spacing: float = FORWARD_METERS,
        action_stats: Any = None,
        max_distance: float | None = None,
    ) -> None:
        self.spacing = _finite_real(spacing, "spacing")
        if self.spacing <= 0.0:
            raise ValueError("spacing must be positive")
        self.action_stats = action_stats
        self.max_distance = None if max_distance is None else _finite_real(max_distance, "max_distance")
        if self.max_distance is not None and self.max_distance < 0.0:
            raise ValueError("max_distance must be nonnegative")
        self.events: list[DecodedAction] = []

    def decode(self, command: Any) -> int:
        values = _triplet(command)
        metric_xy = unnormalize_xy(values[:2], spacing=self.spacing, action_stats=self.action_stats)
        metric = (float(metric_xy[0]), float(metric_xy[1]), values[2])
        # Normalize each axis by the corresponding primitive scale.  This
        # prevents metres from dominating radians and makes the tie order
        # explicit and reproducible.
        codebook = {
            ACTION_STOP: (0.0, 0.0, 0.0),
            ACTION_FWD: (self.spacing, 0.0, 0.0),
            ACTION_LEFT: (0.0, 0.0, TURN_RADIANS),
            ACTION_RIGHT: (0.0, 0.0, -TURN_RADIANS),
        }
        scored: list[tuple[float, int, int]] = []
        for priority, action in enumerate(self._ORDER):
            tx, ty, tyaw = codebook[action]
            score = math.sqrt(
                ((metric[0] - tx) / self.spacing) ** 2
                + ((metric[1] - ty) / self.spacing) ** 2
                + ((metric[2] - tyaw) / TURN_RADIANS) ** 2
            )
            scored.append((score, priority, action))
        distance, _, action = min(scored, key=lambda item: (item[0], item[1]))
        if self.max_distance is not None and distance > self.max_distance:
            raise ValueError(f"decode distance {distance:.6g} exceeds max_distance {self.max_distance:.6g}")
        result = DecodedAction(
            action_name=self._NAMES[action],
            action=action,
            metric_delta=metric,
            distance=float(distance),
            normalized_delta=values,
        )
        self.events.append(result)
        return action

    # Names used by the existing V9 runner's explicit decoder seam.
    quantize = decode
    to_action = decode
    decode_continuous = decode

    def conversion_summary(self) -> dict[str, Any]:
        counts = {name: 0 for name in self._NAMES.values()}
        for event in self.events:
            counts[event.action_name] += 1
        return {
            "decoder": "rae_stream_fixed_codebook_v1",
            "spacing_m": self.spacing,
            "action_stats": self.action_stats,
            "max_distance": self.max_distance,
            "count": len(self.events),
            "counts": counts,
            "events": [
                {
                    "normalized_delta": list(event.normalized_delta),
                    "metric_delta": list(event.metric_delta),
                    "distance": event.distance,
                    "action": event.action_name,
                }
                for event in self.events
            ],
        }


def decode_first_delta(
    command: Any,
    *,
    spacing: float = FORWARD_METERS,
    action_stats: Any = None,
    max_distance: float | None = None,
) -> DecodedAction:
    """Decode one planner delta and return its complete auditable result."""

    decoder = RAEStreamActionDecoder(
        spacing=spacing,
        action_stats=action_stats,
        max_distance=max_distance,
    )
    # ``decode`` returns only the canonical integer for runner compatibility;
    # expose the richer event to callers of this convenience function.
    decoder.decode(command)
    return decoder.events[-1]


# A concise alias is useful in CLI code and keeps the public seam discoverable.
ActionDecoder = RAEStreamActionDecoder


__all__ = [
    "ACTION_STOP",
    "ACTION_FWD",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "DecodedAction",
    "RAEStreamActionDecoder",
    "ActionDecoder",
    "decode_first_delta",
    "normalize_xy",
    "unnormalize_xy",
]
