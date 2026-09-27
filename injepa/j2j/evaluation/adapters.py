"""Policy/action adapters used only by the ImageGoal evaluation runner."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
from typing import Any, Callable

from j2j.adapter import ActionId


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
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


def _canonical_action(value: Any) -> ActionId:
    if isinstance(value, ActionId):
        return value
    if isinstance(value, bool):
        raise TypeError("action must be a canonical ImageNav action")
    if isinstance(value, str):
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
            return names[value.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown action {value!r}") from exc
    try:
        return ActionId(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown action {value!r}") from exc


@dataclass(frozen=True)
class ActionConversion:
    """An auditable continuous-to-discrete conversion event."""

    command: tuple[float, float, float] | None
    action: ActionId
    stop_reason: str | None = None


class NWMContinuousActionAdapter:
    """Map NWM's continuous planar command ABI to four Habitat primitives.

    This class intentionally contains no NWM model code.  It is an external
    evaluation adapter.  NWM has no native STOP class, so zero/near-zero
    commands can be converted to STOP only by an explicitly external rule.
    """

    native_metric_names = ("ATE", "RPE")
    evaluation_metric_names = (
        "success",
        "spl",
        "soft_spl",
        "distance_to_goal",
    )

    def __init__(
        self,
        *,
        forward_step_m: float = 0.25,
        turn_angle_deg: float = 15.0,
        yaw_unit: str = "radians",
        stop_translation_threshold_m: float | None = None,
        stop_yaw_threshold: float | None = None,
        command_threshold: float | None = None,
        lateral_weight: float = 1.0,
        yaw_weight: float = 1.0,
    ) -> None:
        self.forward_step_m = _real(forward_step_m, "forward_step_m")
        self.turn_angle_deg = _real(turn_angle_deg, "turn_angle_deg")
        if self.forward_step_m <= 0.0 or self.turn_angle_deg <= 0.0:
            raise ValueError("forward_step_m and turn_angle_deg must be positive")
        if yaw_unit not in {"radians", "degrees"}:
            raise ValueError("yaw_unit must be 'radians' or 'degrees'")
        self.yaw_unit = yaw_unit
        self.turn_angle = (
            math.radians(self.turn_angle_deg)
            if yaw_unit == "radians"
            else self.turn_angle_deg
        )
        self.stop_translation_threshold_m = (
            self.forward_step_m * 0.1
            if stop_translation_threshold_m is None
            else _real(stop_translation_threshold_m, "stop_translation_threshold_m")
        )
        self.stop_yaw_threshold = (
            self.turn_angle * 0.1
            if stop_yaw_threshold is None
            else _real(stop_yaw_threshold, "stop_yaw_threshold")
        )
        if self.stop_translation_threshold_m < 0.0 or self.stop_yaw_threshold < 0.0:
            raise ValueError("STOP thresholds must be nonnegative")
        self.command_threshold = (
            None
            if command_threshold is None
            else _real(command_threshold, "command_threshold")
        )
        if self.command_threshold is not None and self.command_threshold < 0.0:
            raise ValueError("command_threshold must be nonnegative")
        self.lateral_weight = _real(lateral_weight, "lateral_weight")
        self.yaw_weight = _real(yaw_weight, "yaw_weight")
        if self.lateral_weight <= 0.0 or self.yaw_weight <= 0.0:
            raise ValueError("action metric weights must be positive")
        self.conversions: list[ActionConversion] = []

    def encode(self, action: Any) -> tuple[float, float, float] | None:
        """Encode one canonical primitive as ``(dx, dy, dyaw)``."""

        action_id = _canonical_action(action)
        if action_id == ActionId.STOP:
            return None
        if action_id == ActionId.FWD:
            return (self.forward_step_m, 0.0, 0.0)
        if action_id == ActionId.LEFT:
            return (0.0, 0.0, self.turn_angle)
        if action_id == ActionId.RIGHT:
            return (0.0, 0.0, -self.turn_angle)
        raise ValueError(f"unsupported action {action_id!r}")

    def _command(self, command: Any) -> tuple[float, float, float]:
        if isinstance(command, Mapping):
            command = (
                command.get("dx", command.get("u_x", command.get("x"))),
                command.get("dy", command.get("u_y", command.get("y"))),
                command.get("dyaw", command.get("yaw", command.get("omega"))),
            )
        if isinstance(command, (str, bytes)):
            raise TypeError("continuous command must contain dx, dy, dyaw")
        if not isinstance(command, Sequence):
            # NumPy arrays and Torch tensors are iterable but do not register
            # as ``collections.abc.Sequence`` on all supported versions.
            tolist = getattr(command, "tolist", None)
            if callable(tolist):
                command = tolist()
            elif isinstance(command, Iterable):
                command = tuple(command)
            else:
                raise TypeError("continuous command must contain dx, dy, dyaw")
        if len(command) != 3:
            raise ValueError("continuous command must have exactly three values")
        return tuple(_real(value, f"command[{index}]") for index, value in enumerate(command))  # type: ignore[return-value]

    def should_stop(
        self,
        command: Any,
        *,
        target_distance: float | None = None,
        latent_distance: float | None = None,
        threshold: float | None = None,
    ) -> bool:
        """Apply the preconfigured external STOP rule.

        A target/latent distance is optional and is only consulted when a
        threshold is explicitly supplied.  This prevents accidental use of a
        simulator measurement as a policy input while still allowing a frozen
        experiment rule to be represented in receipts.
        """

        values = self._command(command)
        dx, dy, dyaw = values
        translation = math.hypot(dx, dy)
        zero_command = (
            translation <= self.stop_translation_threshold_m
            and abs(dyaw) <= self.stop_yaw_threshold
        )
        if zero_command:
            return True
        if threshold is None:
            threshold = self.command_threshold
        if threshold is not None:
            limit = _real(threshold, "threshold")
            if limit < 0.0:
                raise ValueError("threshold must be nonnegative")
            # ``command_threshold`` is an external NWM rule over the native
            # command vector.  It must affect the decision even when no goal
            # distance is supplied (the latter is evaluator-only and is not a
            # legal policy input).  The norm is intentionally computed in the
            # command's declared units; callers that use a normalized NWM
            # command can therefore set the threshold in that same space.
            command_norm = math.sqrt(dx * dx + dy * dy + dyaw * dyaw)
            if command_norm <= limit:
                return True
            for value, name in (
                (target_distance, "target_distance"),
                (latent_distance, "latent_distance"),
            ):
                if value is not None and _real(value, name) <= limit:
                    return True
        return False

    def decode(
        self,
        command: Any,
        *,
        stop: bool | None = None,
        stop_reason: str | None = None,
        target_distance: float | None = None,
        latent_distance: float | None = None,
        stop_threshold: float | None = None,
    ) -> ActionId:
        """Quantize a continuous NWM command to the nearest primitive."""

        values = self._command(command)
        if stop_threshold is None:
            stop_threshold = self.command_threshold
        if stop is None:
            stop = self.should_stop(
                values,
                target_distance=target_distance,
                latent_distance=latent_distance,
                threshold=stop_threshold,
            )
        if not isinstance(stop, bool):
            raise TypeError("stop must be boolean")
        if stop:
            action = ActionId.STOP
            if stop_reason is not None:
                reason = stop_reason
            elif stop_threshold is not None:
                values_norm = math.sqrt(sum(value * value for value in values))
                threshold_value = _real(stop_threshold, "stop_threshold")
                reason = (
                    "command_threshold"
                    if values_norm <= threshold_value
                    else "external_stop_rule"
                )
            else:
                reason = "external_stop_rule"
            self.conversions.append(ActionConversion(values, action, reason))
            return action

        canonical = {
            ActionId.FWD: self.encode(ActionId.FWD),
            ActionId.LEFT: self.encode(ActionId.LEFT),
            ActionId.RIGHT: self.encode(ActionId.RIGHT),
        }
        # Compare in normalized units so metres and radians do not overwhelm
        # one another.  Lateral translation is not directly executable by the
        # four-action Habitat agent; it is treated as a directional deviation
        # and therefore contributes to the turn distance.
        dx, dy, dyaw = values
        scored: list[tuple[float, int, ActionId]] = []
        for priority, action in enumerate(
            (ActionId.FWD, ActionId.LEFT, ActionId.RIGHT)
        ):
            target = canonical[action]
            assert target is not None
            tx, ty, tyaw = target
            score = (
                ((dx - tx) / self.forward_step_m) ** 2
                + self.lateral_weight * ((dy - ty) / self.forward_step_m) ** 2
                + self.yaw_weight * ((dyaw - tyaw) / self.turn_angle) ** 2
            )
            scored.append((score, priority, action))
        _, _, action = min(scored, key=lambda item: (item[0], item[1]))
        self.conversions.append(ActionConversion(values, action, None))
        return action

    # Explicit aliases make the ABI self-documenting and accommodate callers
    # that use ``quantize``/``to_action`` terminology.
    quantize = decode
    to_action = decode
    decode_continuous = decode

    def conversion_summary(self) -> dict[str, Any]:
        counts = {action.name: 0 for action in ActionId}
        for event in self.conversions:
            counts[event.action.name] += 1
        return {
            "count": len(self.conversions),
            "counts": counts,
            "events": [
                {
                    "command": list(event.command) if event.command is not None else None,
                    "action": event.action.name,
                    "stop_reason": event.stop_reason,
                }
                for event in self.conversions
            ],
            "yaw_unit": self.yaw_unit,
            "forward_step_m": self.forward_step_m,
            "turn_angle_deg": self.turn_angle_deg,
            "command_threshold": self.command_threshold,
        }


class CallablePolicyAdapter:
    """Small adapter for a user-supplied callable in smoke tests/experiments."""

    def __init__(self, act_fn: Callable[..., Any], *, reset_fn: Callable[..., Any] | None = None):
        if not callable(act_fn):
            raise TypeError("act_fn must be callable")
        self.act_fn = act_fn
        self.reset_fn = reset_fn

    def reset(self, goal_rgb: Any) -> None:
        if self.reset_fn is not None:
            self.reset_fn(goal_rgb)

    def act(self, current_rgb: Any, goal_rgb: Any, history: Any) -> Any:
        return self.act_fn(current_rgb, goal_rgb, history)

    def close(self) -> None:
        return None
