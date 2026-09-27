"""Thin RAE planner adapter for the existing Habitat ImageGoal runner.

The policy owns only observation formatting and the continuous-to-discrete
compatibility seam.  It never stores imagined frames, simulator pose, or
privileged observations.  The heavy RAE model/planner is injected as a
backend, which keeps unit tests CPU-only and makes an ABI mismatch fail
closed instead of silently reimplementing CEM.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import deque
from dataclasses import dataclass
import contextlib
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Callable, Protocol

import numpy as np

from .action_decoder import ActionDecoder, RAEStreamActionDecoder, _triplet


# This is the official Habitat-planning rollout horizon from the checked-out
# ``config/eval_config.yaml``.  It is deliberately distinct from the 64-step
# *training* prediction horizon in ``config/rae_stream.yaml``; using the latter
# here would multiply every CEM decision's rollout work by eight.
OFFICIAL_PLANNER_ROLLOUT_STEPS = 8


class PlannerBackend(Protocol):
    """Minimal backend ABI used by :class:`RAEStreamPolicy`."""

    def plan(self, context: np.ndarray, goal: np.ndarray) -> Any:
        """Return a canonical integer or one continuous ``(dx,dy,dyaw)``."""


def _rgb_array(value: Any, name: str) -> np.ndarray:
    """Validate/copy one RGB observation without retaining caller ownership."""

    array = np.asarray(value)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H,W,3), got {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{name} must have positive spatial dimensions")
    if array.dtype.kind not in "uif":
        raise TypeError(f"{name} must be a numeric RGB array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return np.array(array, copy=True)


def _history_rgb(item: Any) -> np.ndarray | None:
    """Extract only the factual RGB field from a history entry.

    Keys such as ``predicted_rgb`` are intentionally ignored.  A bare array
    is accepted for lightweight callers; mappings must contain an explicit
    ``rgb``/``observation`` key.
    """

    if isinstance(item, Mapping):
        if "rgb" in item:
            return _rgb_array(item["rgb"], "history.rgb")
        if "observation" in item:
            value = item["observation"]
            if isinstance(value, Mapping) and "rgb" in value:
                return _rgb_array(value["rgb"], "history.observation.rgb")
        return None
    try:
        return _rgb_array(item, "history item")
    except (TypeError, ValueError):
        return None


def _canonical_output(value: Any) -> int | None:
    """Return an explicit primitive integer, or ``None`` for a continuous cmd."""

    if isinstance(value, Mapping):
        for key in ("action_id", "primitive", "action"):
            if key in value:
                nested = _canonical_output(value[key])
                # ``action`` is also used by a few planner wrappers for a
                # continuous triplet.  Leave that payload for the explicit
                # decoder below instead of rejecting it before the decoder
                # gets a chance to run.
                if nested is not None:
                    return nested
                return None
        for key in ("continuous_action", "command", "action_continuous", "dx", "dy", "dyaw", "u_x", "u_y", "omega"):
            if key in value:
                return None
        raise ValueError("planner output mapping has no action or continuous command")
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("planner returned boolean instead of a Habitat action")
    if isinstance(value, (int, np.integer)):
        action = int(value)
        if action not in (0, 1, 2, 3):
            raise ValueError(f"planner returned invalid Habitat action {action}")
        return action
    if isinstance(value, str):
        names = {
            "STOP": 0,
            "FWD": 1,
            "MOVE_FORWARD": 1,
            "LEFT": 2,
            "TURN_LEFT": 2,
            "RIGHT": 3,
            "TURN_RIGHT": 3,
        }
        try:
            return names[value.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown planner action {value!r}") from exc
    # A scalar torch tensor is an explicit action; a length-3 tensor is a
    # continuous command and is handled by the decoder below.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = None
        if isinstance(scalar, (int, np.integer)) and not isinstance(scalar, bool):
            return _canonical_output(int(scalar))
    return None


@dataclass(frozen=True)
class PolicyDecision:
    """Optional diagnostics retained for the caller, never fed into history."""

    action: int | str
    continuous_command: tuple[float, float, float] | None = None


class RAEStreamPolicy:
    """Adapt an injected RAE backend to the V9 runner policy ABI.

    Parameters
    ----------
    backend:
        Object exposing ``plan(context, goal)`` or a compatible callable.
    decoder:
        :class:`RAEStreamActionDecoder` (or an object exposing ``decode``).
        It is required whenever the backend returns a continuous command.
    context_size:
        Official RAE context length; fixed at four by the default profile but
        kept explicit so a malformed value cannot be hidden in code.
    """

    def __init__(
        self,
        backend: PlannerBackend | Callable[[np.ndarray, np.ndarray], Any],
        *,
        decoder: ActionDecoder | Any | None = None,
        context_size: int = 4,
        mode: str = 'discrete',
    ) -> None:
        if not callable(getattr(backend, "plan", None)) and not callable(backend):
            raise TypeError("backend must expose plan(context, goal) or be callable")
        if isinstance(context_size, bool) or int(context_size) <= 0:
            raise ValueError("context_size must be a positive integer")
        self.backend = backend
        if mode not in ('discrete', 'continuous'):
            raise ValueError('mode must be discrete or continuous')
        self.mode = mode
        # The fixed decoder is part of the RAE-stream compatibility contract,
        # not a learned/model component.  Callers can inject a configured
        # instance (for example with a fail-closed distance threshold), while
        # the lightweight fake backend seam remains usable without extra
        # plumbing.
        self.decoder = decoder if decoder is not None else RAEStreamActionDecoder()
        self.context_size = int(context_size)
        self._goal_rgb: np.ndarray | None = None
        self._factual_context: deque[np.ndarray] = deque(maxlen=max(self.context_size - 1, 0))
        self.last_decision: PolicyDecision | None = None
        self._closed = False

    def reset(self, goal_rgb: Any) -> None:
        if self._closed:
            raise RuntimeError("policy is closed")
        self._goal_rgb = _rgb_array(goal_rgb, "goal_rgb")
        self._factual_context.clear()
        self.last_decision = None

    def _context(self, current_rgb: Any, factual_history: Any) -> tuple[np.ndarray, np.ndarray]:
        current = _rgb_array(current_rgb, "current_rgb")
        if self._goal_rgb is None:
            raise RuntimeError("reset(goal_rgb) must be called before act")
        goal = np.array(self._goal_rgb, copy=True)

        # The runner passes the authoritative factual history.  Rebuild from
        # that argument every call rather than trusting an internal imagined
        # state.  This also makes a stale/reordered history visible in tests.
        entries: list[np.ndarray] = []
        if factual_history is not None:
            try:
                iterator = iter(factual_history)
            except TypeError as exc:
                raise TypeError("factual_history must be iterable") from exc
            for item in iterator:
                rgb = _history_rgb(item)
                if rgb is not None:
                    entries.append(rgb)
        history_slots = max(self.context_size - 1, 0)
        recent = entries[-history_slots:] if history_slots else []
        if len(recent) < history_slots:
            # RAE requires exactly four frames.  Padding with the current
            # factual RGB is the official first-observation warm-up rule.
            recent = [np.array(current, copy=True)] * (history_slots - len(recent)) + recent
        context = np.stack([*recent, current], axis=0)
        return context, goal

    def _call_backend(self, context: np.ndarray, goal: np.ndarray) -> Any:
        planner = getattr(self.backend, "plan", None)
        if callable(planner):
            return planner(np.array(context, copy=True), np.array(goal, copy=True))
        return self.backend(np.array(context, copy=True), np.array(goal, copy=True))  # type: ignore[misc]

    def act(self, current_rgb: Any, goal_rgb: Any, factual_history: Any) -> Any:
        if self._closed:
            raise RuntimeError("policy is closed")
        # The runner supplies goal_rgb each decision.  Require consistency
        # with reset while allowing a fresh immutable copy from the firewall.
        supplied_goal = _rgb_array(goal_rgb, "goal_rgb")
        if self._goal_rgb is None:
            self.reset(supplied_goal)
        elif self._goal_rgb.shape != supplied_goal.shape or not np.array_equal(self._goal_rgb, supplied_goal):
            raise ValueError("goal_rgb changed after reset")
        context, goal = self._context(current_rgb, factual_history)
        output = self._call_backend(context, goal)
        if self.mode == 'continuous':
            command = _triplet(output)
            self.last_decision = PolicyDecision(action='CONTINUOUS', continuous_command=command)
            return {'continuous_action': command}
        primitive = _canonical_output(output)
        command: tuple[float, float, float] | None = None
        if primitive is None:
            command = _triplet(output)
            if self.decoder is None or not callable(getattr(self.decoder, "decode", None)):
                raise TypeError("continuous planner output requires an explicit action decoder")
            primitive = int(self.decoder.decode(command))
        if primitive not in (0, 1, 2, 3):
            raise ValueError(f"decoded action outside Habitat ABI: {primitive!r}")
        # Do not append context/predictions to internal history.  The existing
        # runner owns factual history and appends only post-step RGB frames.
        self.last_decision = PolicyDecision(action=primitive, continuous_command=command)
        return primitive

    def close(self) -> None:
        self._closed = True
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()


class OfficialRAEPlannerBackend:
    """Adapter around an already initialized official ``WM_Planning_Evaluator``.

    Constructing the evaluator is intentionally kept outside this class (and
    therefore outside every episode).  Use ``from_factory`` in a server
    launcher to dynamically import the official planner and inject it.  The
    callback capture invokes the official ``generate_actions`` method exactly
    once and does not write prediction files.
    """

    def __init__(
        self,
        evaluator: Any,
        *,
        transform: Callable[[Any], Any],
        torch_module: Any,
        len_traj_pred: int = OFFICIAL_PLANNER_ROLLOUT_STEPS,
        dataset_name: str = "rae_stream",
    ) -> None:
        if not callable(getattr(evaluator, "generate_actions", None)):
            raise TypeError("official evaluator must expose generate_actions")
        if not callable(transform):
            raise TypeError("official RAE transform must be callable")
        if not hasattr(torch_module, "as_tensor"):
            raise TypeError("torch_module must expose as_tensor")
        if int(len_traj_pred) <= 0:
            raise ValueError("len_traj_pred must be positive")
        self.evaluator = evaluator
        self.transform = transform
        self.torch = torch_module
        self.len_traj_pred = int(len_traj_pred)
        self.dataset_name = str(dataset_name)
        self._counter = 0

    def plan(self, context: np.ndarray, goal: np.ndarray) -> tuple[float, float, float]:
        if context.ndim != 4 or context.shape[0] != 4 or context.shape[-1] != 3:
            raise ValueError("official planner context must be (4,H,W,3)")
        if goal.ndim != 3 or goal.shape[-1] != 3:
            raise ValueError("official planner goal must be (H,W,3)")
        try:
            transformed = [self.transform(_to_pil(frame)) for frame in context]
            obs = self.torch.stack(transformed, dim=0).unsqueeze(0)
            goal_tensor = self.torch.stack([self.transform(_to_pil(goal))], dim=0).unsqueeze(0)
            zeros = self.torch.zeros((1, self.len_traj_pred, 3), dtype=getattr(self.torch, "float32", None))
            idxs = self.torch.zeros((1, 1), dtype=getattr(self.torch, "float32", None))
        except Exception as exc:
            raise RuntimeError(f"could not construct official planner tensors: {exc}") from exc

        planning_module = sys.modules.get(getattr(self.evaluator.__class__, "__module__", ""))
        if planning_module is None or not callable(getattr(planning_module, "save_planning_pred", None)):
            raise RuntimeError("official planner save_planning_pred callback ABI is unavailable")
        original_callback = planning_module.save_planning_pred
        captured: dict[str, Any] = {}

        def capture(*args: Any, **kwargs: Any) -> None:
            # Official signature: (..., preds, deltas, loss, gt_actions, ...).
            deltas = kwargs.get("deltas")
            if deltas is None and len(args) >= 7:
                deltas = args[6]
            if deltas is None:
                raise RuntimeError("official save_planning_pred did not expose deltas")
            captured["deltas"] = deltas

        planning_module.save_planning_pred = capture
        args_obj: Any = None
        old_save: Any = None
        try:
            # Callback capture requires the official branch to be enabled.  We
            # restore the flag and callback in a finally block, so no state or
            # output path leaks between decisions.
            args_obj = getattr(self.evaluator, "args", None)
            old_save = getattr(args_obj, "save_preds", None) if args_obj is not None else None
            if args_obj is not None:
                args_obj.save_preds = True
            # Match the official evaluate() invocation, including both its
            # no_grad decorator and CUDA BF16 autocast. generate_actions()
            # itself has neither context.
            with self.torch.no_grad(), self.torch.amp.autocast(
                'cuda', enabled=True, dtype=self.torch.bfloat16
            ):
                self.evaluator.generate_actions(
                    None,
                    self.dataset_name,
                    idxs,
                    obs,
                    goal_tensor,
                    zeros,
                    self.len_traj_pred,
                )
            if args_obj is not None:
                args_obj.save_preds = old_save
        except Exception as exc:
            raise RuntimeError(f"official planner generate_actions ABI failed: {exc}") from exc
        finally:
            planning_module.save_planning_pred = original_callback
            if args_obj is not None:
                args_obj.save_preds = old_save
        deltas = captured.get("deltas")
        if deltas is None:
            raise RuntimeError("official planner produced no captured continuous deltas")
        try:
            first = deltas[0, 0]
            values = first.detach().float().cpu().reshape(-1).tolist()
            if len(values) != 3:
                raise ValueError
            return tuple(float(v) for v in values)  # type: ignore[return-value]
        except Exception as exc:
            raise RuntimeError("official planner deltas have unexpected shape; refusing fallback decode") from exc


def _to_pil(frame: np.ndarray) -> Any:
    """Lazily convert an RGB array to PIL without importing it in unit tests."""

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - server-only path
        raise RuntimeError("Pillow is required by the official RAE transform") from exc
    array = np.asarray(frame)
    if array.dtype.kind == "f":
        # Habitat RGB is uint8; float input is accepted only in [0,1].
        if np.any(array < 0.0) or np.any(array > 1.0):
            raise ValueError("float RGB frames must lie in [0,1]")
        array = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


__all__ = [
    "PlannerBackend",
    "PolicyDecision",
    "RAEStreamPolicy",
    "OfficialRAEPlannerBackend",
    "OFFICIAL_PLANNER_ROLLOUT_STEPS",
]
