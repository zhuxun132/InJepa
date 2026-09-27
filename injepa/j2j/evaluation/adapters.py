"""Policy/action adapters used only by the ImageGoal evaluation runner."""

from __future__ import annotations

from typing import Any, Callable

from j2j.adapter import ActionId


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
