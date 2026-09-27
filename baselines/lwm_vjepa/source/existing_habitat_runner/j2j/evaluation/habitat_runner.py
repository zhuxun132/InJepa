"""Thin, lazy Habitat 0.2.4 ImageGoal runner.

Only this module knows about the optional simulator.  The pure contract and
metric code remains importable on machines that do not have Habitat installed.
The runner follows the approved receding-horizon lifecycle: one primitive per
cycle, then a fresh real RGB observation; imagined policy state is never copied
into the factual history.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import importlib
import inspect
import math
import numbers
import gzip
import json
import time
from typing import Any

from j2j.adapter import ActionId

from .contracts import (
    EXPECTED_HABITAT_VERSION,
    PRIVILEGED_OBSERVATION_KEYS,
    RGBOnlyObservationFirewall,
    canonical_episode_id,
    canonical_scene_id,
    validate_imagegoal_config,
)
from .metrics import HabitatMetricsAccumulator, LatencyRecorder


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        item = getattr(value, "item", None)
        if callable(item):
            try:
                value = item()
            except Exception:
                return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _unpack_reset(value: Any) -> tuple[Any, Mapping[str, Any] | None]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], Mapping):
        return value[0], value[1]
    return value, None


def _unpack_step(value: Any) -> tuple[Any, bool, Mapping[str, Any]]:
    """Normalize Habitat core/Gym step return variants."""

    if isinstance(value, tuple):
        if len(value) == 5:
            observation, _reward, terminated, truncated, info = value
            return observation, bool(terminated or truncated), (
                info if isinstance(info, Mapping) else {}
            )
        if len(value) == 4:
            observation, _reward, done, info = value
            return observation, bool(done), (
                info if isinstance(info, Mapping) else {}
            )
        if len(value) == 2 and isinstance(value[1], Mapping):
            return value[0], False, value[1]
    return value, False, {}


def _mapping_value(mapping: Mapping[str, Any] | None, *keys: str) -> Any:
    if mapping is None:
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _position_from(env: Any, observation: Mapping[str, Any], info: Mapping[str, Any] | None) -> Any:
    # These values are evaluator-only and are never passed through the firewall.
    for source in (observation, info or {}):
        for key in ("position", "pose"):
            if key in source:
                value = source[key]
                if key == "pose" and isinstance(value, Mapping):
                    value = value.get("position")
                if value is not None:
                    return value
    sim = getattr(env, "sim", None)
    if sim is not None:
        getter = getattr(sim, "get_agent_state", None)
        if callable(getter):
            state = getter()
            value = getattr(state, "position", None)
            if value is not None:
                return value
    getter = getattr(env, "get_agent_state", None)
    if callable(getter):
        state = getter()
        value = getattr(state, "position", None)
        if value is not None:
            return value
    return None


def _distance_from(
    env: Any,
    observation: Mapping[str, Any],
    info: Mapping[str, Any] | None,
    metrics: Mapping[str, Any] | None,
) -> float | None:
    for source in (metrics or {}, observation, info or {}):
        value = _mapping_value(source, "distance_to_goal", "distance_to_target")
        value = _finite_or_none(value)
        if value is not None and value >= 0.0:
            return value
    return None


def _metrics_from_env(env: Any) -> Mapping[str, Any]:
    getter = getattr(env, "get_metrics", None)
    if callable(getter):
        values = getter()
        if isinstance(values, Mapping):
            return values
    # Some wrappers expose ``info``/``metrics`` as a property.
    for name in ("metrics", "info"):
        values = getattr(env, name, None)
        if isinstance(values, Mapping):
            return values
    return {}


def _extract_rgb(
    observation: Mapping[str, Any], *, require_goal: bool = True
) -> tuple[Any, Any | None]:
    if not isinstance(observation, Mapping):
        raise TypeError("Habitat observation must be a mapping")
    current = observation.get("rgb", observation.get("current_rgb"))
    goal = observation.get("imagegoal", observation.get("goal_rgb"))
    if current is None:
        raise ValueError("Habitat observation has no rgb/current_rgb")
    if goal is None and require_goal:
        raise ValueError("Habitat observation has no imagegoal/goal_rgb")
    return current, goal


def _current_episode_identity(env: Any) -> str | None:
    """Read Habitat's actual current episode identity after ``reset``."""

    current_episode = getattr(env, "current_episode", None)
    if isinstance(current_episode, Mapping):
        value = current_episode.get("episode_id", current_episode.get("id"))
    else:
        value = getattr(current_episode, "episode_id", None)
        if value is None:
            value = getattr(current_episode, "id", None)
    if value is None or not str(value).strip():
        return None
    return canonical_episode_id(value)


def _current_episode_scene_identity(env: Any) -> str | None:
    """Read Habitat's current episode scene path/token after ``reset``."""

    current_episode = getattr(env, "current_episode", None)
    if isinstance(current_episode, Mapping):
        value = current_episode.get("scene_id")
    else:
        value = getattr(current_episode, "scene_id", None)
    if value is None or not str(value).strip():
        return None
    return str(value)


def _as_action(value: Any) -> ActionId:
    if isinstance(value, Mapping):
        for key in ("action_id", "action", "primitive"):
            if key in value:
                return _as_action(value[key])
        raise ValueError("policy result mapping has no action/action_id")
    if isinstance(value, tuple):
        # A two-tuple is accepted only when its first member is an explicit
        # canonical action and the second member is diagnostics.  In
        # particular, an NWM continuous (dx, dy, dyaw) tuple must never be
        # truncated to its first float (which used to turn 0.25 into STOP).
        if len(value) == 2:
            first = value[0]
            if isinstance(first, (ActionId, str)) or (
                isinstance(first, numbers.Integral) and not isinstance(first, bool)
            ):
                return _as_action(first)
        raise TypeError(
            "continuous/ambiguous tuple returned by policy; supply an explicit "
            "NWM action decoder"
        )
    if isinstance(value, list):
        raise TypeError(
            "continuous/ambiguous sequence returned by policy; supply an explicit "
            "NWM action decoder"
        )
    if isinstance(value, ActionId):
        return value
    if isinstance(value, bool):
        raise TypeError("policy returned boolean instead of ActionId")
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
        if value.upper() not in names:
            raise ValueError(f"unknown policy action {value!r}")
        return names[value.upper()]
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"policy returned non-integral action {value!r}")
    try:
        return ActionId(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"policy returned invalid action {value!r}") from exc


def _continuous_payload(value: Any) -> Any | None:
    """Extract an explicit NWM continuous command, if one is present."""

    if isinstance(value, Mapping):
        for key in ("continuous_action", "command", "action_continuous"):
            if key in value:
                return value[key]
        # A named dx/dy/dyaw mapping is also unambiguous.
        if any(key in value for key in ("dx", "dy", "dyaw", "u_x", "u_y", "omega")):
            return value
        return None
    if isinstance(value, tuple) and len(value) == 3:
        return value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            converted = tolist()
        except Exception:
            return None
        if isinstance(converted, (list, tuple)) and len(converted) == 3:
            return value
    if isinstance(value, list) and len(value) == 3:
        return value
    return None


def _call_policy_reset(adapter: Any, goal_rgb: Any) -> None:
    reset = getattr(adapter, "reset", None)
    if callable(reset):
        reset(goal_rgb)


def _call_policy_act(adapter: Any, current: Any, goal: Any, history: Any) -> Any:
    act = getattr(adapter, "act", None)
    if callable(act):
        # Inspect binding before invocation so an implementation TypeError is
        # never mistaken for a signature mismatch (and therefore never causes
        # a stateful policy to run twice).
        try:
            signature = inspect.signature(act)
        except (TypeError, ValueError):
            return act(current, goal, history)
        try:
            signature.bind(current, goal, history)
        except TypeError:
            try:
                signature.bind(current_rgb=current, goal_rgb=goal, history=history)
            except TypeError:
                raise TypeError(
                    "policy act signature must accept (current_rgb, goal_rgb, history)"
                )
            return act(current_rgb=current, goal_rgb=goal, history=history)
        return act(current, goal, history)
    if callable(adapter):
        return adapter(current, goal, history)
    raise TypeError("adapter must expose act(...) or be callable")


def _call_env_step(env: Any, action: ActionId) -> Any:
    # Habitat core Env accepts an int and wraps it as {"action": int}; Gym
    # wrappers accept either.  Passing the integer avoids leaking our enum type
    # into a simulator-specific serialization path.
    return env.step(int(action))


def _validated_continuous_execution(value: Any) -> tuple[str, bool, dict[str, Any]]:
    """Validate the explicit simulator seam without inventing a primitive."""

    if not isinstance(value, Mapping) or set(value) != {"name", "is_stop", "habitat_payload"}:
        raise TypeError("continuous handler must return name, is_stop and habitat_payload")
    name, is_stop, payload = value["name"], value["is_stop"], value["habitat_payload"]
    if not isinstance(name, str) or name not in {"CONTINUOUS", "STOP"}:
        raise ValueError("continuous execution name must be CONTINUOUS or STOP")
    if not isinstance(is_stop, bool):
        raise TypeError("continuous execution is_stop must be boolean")
    if is_stop != (name == "STOP"):
        raise ValueError("continuous execution name and is_stop disagree")
    if not isinstance(payload, Mapping) or "action" not in payload or set(payload) - {"action", "action_args"}:
        raise TypeError("habitat_payload must contain action and optional action_args only")
    if not isinstance(payload["action"], str) or not payload["action"].strip():
        raise TypeError("habitat_payload.action must be a nonempty action name")
    if is_stop and dict(payload) != {"action": "stop"}:
        raise ValueError("STOP execution requires the native {'action': 'stop'} payload")
    if not is_stop and payload["action"].strip().lower() == "stop":
        raise ValueError("CONTINUOUS execution cannot dispatch STOP")
    if "action_args" in payload and not isinstance(payload["action_args"], Mapping):
        raise TypeError("habitat_payload.action_args must be a mapping")

    def validate_json(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str) or key.lower() in PRIVILEGED_OBSERVATION_KEYS:
                    raise ValueError("continuous payload contains a privileged/non-string key")
                validate_json(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                validate_json(child)
        elif item is None or isinstance(item, (str, bool, int)):
            return
        elif isinstance(item, float) and math.isfinite(item):
            return
        else:
            raise TypeError("continuous payload must contain finite JSON values")

    validate_json(payload)
    # Own a plain, finite JSON snapshot before invoking an external simulator.
    return name, is_stop, json.loads(json.dumps(payload, allow_nan=False))


def _observer_policy_evidence(adapter: Any, action: ActionId) -> dict[str, Any]:
    factual = getattr(adapter, "factual_state_receipt", None)
    if not callable(factual):
        raise TypeError("decision observer requires adapter.factual_state_receipt()")
    latest = getattr(adapter, "latest_step_diagnostics", None)
    if callable(latest):
        diagnostics = latest()
    else:
        drain = getattr(adapter, "drain_step_diagnostics", None)
        if not callable(drain):
            raise TypeError("decision observer requires scalar policy diagnostics")
        rows = drain()
        if not isinstance(rows, (list, tuple)) or not rows:
            raise ValueError("decision observer has no policy diagnostic row")
        diagnostics = rows[-1]
    if not isinstance(diagnostics, Mapping):
        raise TypeError("decision observer diagnostics must be a mapping")
    receipt = factual()
    if not isinstance(receipt, Mapping):
        raise TypeError("decision observer factual receipt must be a mapping")
    return {
        "action": action.name,
        "diagnostics": copy.deepcopy(dict(diagnostics)),
        "factual_state_receipt": copy.deepcopy(dict(receipt)),
    }


def _observer_rotation(env: Any) -> list[float] | None:
    simulator = getattr(env, "sim", None)
    getter = getattr(simulator, "get_agent_state", None)
    if not callable(getter):
        return None
    rotation = getattr(getter(), "rotation", None)
    if rotation is None:
        return None
    if hasattr(rotation, "real") and hasattr(rotation, "imag"):
        values = [float(rotation.real), *[float(value) for value in rotation.imag]]
    elif all(hasattr(rotation, name) for name in ("w", "x", "y", "z")):
        values = [float(getattr(rotation, name)) for name in ("w", "x", "y", "z")]
    else:
        try:
            values = [float(value) for value in rotation]
        except TypeError:
            return None
    return values if len(values) == 4 and all(math.isfinite(value) for value in values) else None


def _observer_goal_yaw_degrees(env: Any) -> float | None:
    episode = getattr(env, "current_episode", None)
    goals = getattr(episode, "goals", None)
    if not isinstance(goals, Sequence) or not goals:
        return None
    rotation = getattr(goals[0], "rotation", None)
    if rotation is None:
        return None
    if hasattr(rotation, "real") and hasattr(rotation, "imag"):
        imag = list(rotation.imag)
        if len(imag) != 3:
            return None
        w, x, y, z = float(rotation.real), *(float(value) for value in imag)
    elif all(hasattr(rotation, name) for name in ("w", "x", "y", "z")):
        w, x, y, z = (float(getattr(rotation, name)) for name in ("w", "x", "y", "z"))
    else:
        try:
            x, y, z, w = (float(value) for value in rotation)
        except (TypeError, ValueError):
            return None
    values = (w, x, y, z)
    if not all(math.isfinite(value) for value in values):
        return None
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
    return math.degrees(yaw)


def run_imagegoal_episode(
    env: Any,
    adapter: Any,
    *,
    max_steps: int = 1000,
    success_distance: float = 0.2,
    firewall: RGBOnlyObservationFirewall | None = None,
    metrics: HabitatMetricsAccumulator | None = None,
    latency: LatencyRecorder | None = None,
    warmup_steps: int = 0,
    episode_id: str | None = None,
    expected_scene_id: str | None = None,
    expected_episode_id: str | None = None,
    action_decoder: Any = None,
    continuous_action_handler: Any = None,
    decision_observer: Any = None,
    frame_observer: Any = None,
    diagnostic_reach_radius: float | None = None,
) -> dict[str, Any]:
    """Run one closed-loop ImageGoal episode and return a JSON-ready receipt."""

    if diagnostic_reach_radius is not None:
        if (isinstance(diagnostic_reach_radius, bool)
                or not isinstance(diagnostic_reach_radius, numbers.Real)
                or not math.isfinite(diagnostic_reach_radius)
                or diagnostic_reach_radius <= 0):
            raise ValueError("diagnostic reach radius must be positive and finite")
        diagnostic_reach_radius = float(diagnostic_reach_radius)

    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if decision_observer is not None and not callable(decision_observer):
        raise TypeError("decision_observer must be callable")
    if frame_observer is not None and not callable(frame_observer):
        raise TypeError("frame_observer must be callable")
    if continuous_action_handler is not None:
        if not callable(continuous_action_handler):
            raise TypeError("continuous_action_handler must be callable")
        if action_decoder is not None:
            raise ValueError("continuous_action_handler and action_decoder are mutually exclusive")
        if decision_observer is not None:
            raise ValueError("continuous execution does not support the primitive decision_observer")
    expected_episode_token = (
        canonical_episode_id(expected_episode_id)
        if expected_episode_id is not None
        else None
    )
    supplied_episode_token = (
        canonical_episode_id(episode_id) if episode_id is not None else None
    )
    episode_id = supplied_episode_token
    if expected_scene_id is not None and not str(expected_scene_id).strip():
        raise ValueError("expected_scene_id must be a nonempty string when supplied")
    if firewall is None:
        firewall = RGBOnlyObservationFirewall()
    if metrics is None:
        metrics = HabitatMetricsAccumulator(success_distance=success_distance)
    if latency is None:
        latency = LatencyRecorder(warmup_samples=warmup_steps)

    episode_start = time.perf_counter_ns()
    reset_value = env.reset()
    raw_observation, reset_info = _unpack_reset(reset_value)
    if not isinstance(raw_observation, Mapping):
        raise TypeError("env.reset() must return a mapping observation")
    current_rgb, goal_rgb = _extract_rgb(raw_observation)
    # Simulator sensor arrays may share/reuse storage on the next step. Own
    # factual snapshots before any stepping, including the fixed reset goal.
    current_rgb, goal_rgb = copy.deepcopy(current_rgb), copy.deepcopy(goal_rgb)
    actual_episode_id = _current_episode_identity(env)
    actual_scene_value = _current_episode_scene_identity(env)
    expected_scene_token = (
        canonical_scene_id(expected_scene_id) if expected_scene_id is not None else None
    )
    actual_scene_token = None
    if actual_scene_value is not None:
        try:
            actual_scene_token = canonical_scene_id(actual_scene_value)
        except ValueError:
            # A non-ledger fake environment may expose an opaque scene name;
            # formal calls with an expected scene still fail below.
            if expected_scene_token is not None:
                raise
    if actual_episode_id is None:
        # Pure fake environments used by contract tests need not invent a
        # Habitat identity.  A formal ledger-bound call, however, must always
        # compare against the simulator's current episode object.
        if expected_episode_token is not None or expected_scene_token is not None:
            raise RuntimeError(
                "Habitat current_episode has no official episode identity for the ledger row"
            )
        if episode_id is not None:
            raise RuntimeError("Habitat current_episode has no official episode identity")
        actual_episode_id = None
    if actual_episode_id is not None:
        if supplied_episode_token is not None and supplied_episode_token != actual_episode_id:
            raise ValueError(
                f"Habitat current_episode id {actual_episode_id!r} does not match supplied episode id {supplied_episode_token!r}"
            )
        if expected_episode_token is not None and expected_episode_token != actual_episode_id:
            raise ValueError(
                f"Habitat current_episode id {actual_episode_id!r} does not match ledger episode id {expected_episode_token!r}"
            )
        if expected_scene_token is not None:
            if actual_scene_token is None:
                raise RuntimeError(
                    "Habitat current_episode has no official scene identity for the ledger row"
                )
            if actual_scene_token != expected_scene_token:
                raise ValueError(
                    f"Habitat current_episode scene {actual_scene_token!r} does not match "
                    f"ledger scene {expected_scene_token!r}"
                )
        episode_id = actual_episode_id
    # Only this two-key view crosses the firewall.  Evaluator measurements in
    # the simulator mapping are deliberately extracted separately.
    if frame_observer is not None:
        frame_observer({"rgb": copy.deepcopy(current_rgb), "imagegoal": copy.deepcopy(goal_rgb), "step": 0, "action": None, "phase": "reset"})
    policy_observation = firewall.filter({"rgb": current_rgb, "imagegoal": goal_rgb})
    _call_policy_reset(adapter, goal_rgb)

    initial_metrics = _metrics_from_env(env)
    start_position = _position_from(env, raw_observation, reset_info)
    start_distance = _distance_from(env, raw_observation, reset_info, initial_metrics)
    if start_position is None:
        # A core Habitat env always exposes sim.get_agent_state; this fallback
        # keeps fake environments useful while making missing metric data clear.
        start_position = (0.0,)
    if start_distance is None:
        raise RuntimeError(
            "official DistanceToGoal is unavailable at episode reset; "
            "refusing to fabricate a zero distance"
        )
    metrics.reset(start_position=start_position, start_distance=start_distance)
    previous_position = start_position
    current_distance = float(start_distance)

    factual_history: list[dict[str, Any]] = []
    actions: list[str] = []
    termination_reason = "MAX_STEPS"
    done = False
    step_count = 0
    first_arrival_step = None
    if diagnostic_reach_radius is not None and current_distance < diagnostic_reach_radius:
        first_arrival_step = 0
        termination_reason = "REACHED_GOAL"
        done = True
    while step_count < max_steps and not done:
        # The model may enqueue asynchronous CUDA work.  Synchronize at the
        # beginning/end of the decision boundary so CPU wall-clock timing does
        # not under-report policy latency.  ``LatencyRecorder`` is a no-op on
        # CPU-only runs and can be replaced by a deterministic test hook.
        latency.synchronize()
        decision_start = time.perf_counter_ns()
        policy_start = time.perf_counter_ns()
        # Rebuild the safe view every cycle.  This prevents a policy from
        # retaining a mutable simulator mapping and makes the no-privileged-key
        # invariant explicit at each call.
        policy_observation = firewall.filter(
            {"rgb": current_rgb, "imagegoal": goal_rgb},
            history=tuple(factual_history),
        )
        raw_action = _call_policy_act(
            adapter,
            policy_observation["rgb"],
            policy_observation["imagegoal"],
            policy_observation["factual_history"],
        )
        latency.synchronize()
        policy_elapsed = (time.perf_counter_ns() - policy_start) / 1_000_000.0
        latency.record(
            "policy_inference_ms",
            policy_elapsed,
            warmup=step_count < warmup_steps,
        )
        continuous = _continuous_payload(raw_action)
        habitat_payload = None
        if continuous_action_handler is not None:
            if continuous is None:
                raise TypeError("continuous handler requires an explicit continuous policy command")
            action_name, stop_called, habitat_payload = _validated_continuous_execution(
                continuous_action_handler(raw_action)
            )
            action = action_name
        elif continuous is not None:
            if action_decoder is None:
                raise TypeError(
                    "continuous/ambiguous NWM command requires an explicit action_decoder"
                )
            # Keep this path opt-in; a V9 policy must return a canonical action
            # and cannot accidentally enter the NWM continuous adapter.
            raw_action = action_decoder.decode(continuous)
        if continuous_action_handler is None:
            action = _as_action(raw_action)
            action_name = action.name
            stop_called = action == ActionId.STOP
        actions.append(action_name)

        # ``decision_cycle_ms`` is the observation-to-action boundary.  Keep
        # simulator stepping and all post-step bookkeeping in their dedicated
        # timing buckets below; otherwise the reported policy decision time
        # would depend on Habitat rendering/physics latency.
        decision_elapsed = (time.perf_counter_ns() - decision_start) / 1_000_000.0
        latency.record(
            "decision_cycle_ms",
            decision_elapsed,
            warmup=step_count < warmup_steps,
        )

        if decision_observer is not None:
            # The observer receives detached copies after the canonical action
            # is fixed and before simulator stepping. Its return value is
            # deliberately ignored, so it cannot become a second policy.
            policy_evidence = _observer_policy_evidence(adapter, action)
            evaluator_evidence = {
                "scene_id": actual_scene_token,
                "episode_id": episode_id,
                "step": step_count,
                "position": copy.deepcopy(previous_position),
                "rotation": _observer_rotation(env),
                "goal_yaw_degrees": _observer_goal_yaw_degrees(env),
                "distance_to_goal": current_distance,
            }
            decision_observer(
                copy.deepcopy(policy_evidence), copy.deepcopy(evaluator_evidence)
            )

        environment_start = time.perf_counter_ns()
        step_value = env.step(habitat_payload) if habitat_payload is not None else _call_env_step(env, action)
        environment_elapsed = (time.perf_counter_ns() - environment_start) / 1_000_000.0
        latency.record(
            "environment_step_ms",
            environment_elapsed,
            warmup=step_count < warmup_steps,
        )
        new_observation, step_done, step_info = _unpack_step(step_value)
        if not isinstance(new_observation, Mapping):
            raise TypeError("env.step() must return a mapping observation")
        env_metrics = _metrics_from_env(env)
        merged_metrics = dict(env_metrics)
        merged_metrics.update(step_info)
        position = _position_from(env, new_observation, merged_metrics)
        if position is None:
            # Keep this evaluator-only fallback separate from factual history;
            # position must never be persisted in the policy view.
            position = previous_position
        distance = _distance_from(env, new_observation, step_info, env_metrics)
        if distance is None:
            distance = _finite_or_none(merged_metrics.get("distance_to_goal"))
        if distance is None:
            raise RuntimeError(
                "official DistanceToGoal is unavailable after env.step; "
                "refusing to fabricate a metric value"
            )
        current_distance = float(distance)
        collision_value = merged_metrics.get(
            "collisions", merged_metrics.get("collision")
        )
        collisions: int | bool | None
        if isinstance(collision_value, Mapping):
            if "count" in collision_value:
                count = collision_value["count"]
                if isinstance(count, numbers.Integral) and not isinstance(count, bool):
                    collisions = int(count)
                else:
                    raise TypeError("official collisions.count must be an integer")
            elif "is_collision" in collision_value:
                collisions = bool(collision_value["is_collision"])
            else:
                collisions = None
        elif isinstance(collision_value, (bool, numbers.Integral)):
            collisions = collision_value
        else:
            collisions = None
        metrics.observe(
            position=position,
            distance_to_goal=distance,
            action=action,
            collisions=collisions,
            timeout=False,
            done=step_done,
            **({"stop_called": stop_called} if continuous_action_handler is not None else {}),
        )
        previous_position = position
        factual_history.append(
            {
                # Only real RGB, the action actually executed, and ordering /
                # validity metadata cross the next policy boundary.  Position,
                # geodesic distance and collision measures stay evaluator-only.
                # RAE continuous policy appends the new current frame itself;
                # retain the pre-action frame as past context, avoiding a
                # duplicated current frame and a dropped oldest observation.
                # Preserve the existing discrete adapter's post-action ABI.
                "rgb": copy.deepcopy(
                    current_rgb if continuous_action_handler is not None
                    else new_observation.get("rgb", new_observation.get("current_rgb"))
                ),
                "action": action_name,
                "order": step_count,
                "mask": True,
            }
        )
        step_count += 1
        current_rgb, next_goal = _extract_rgb(new_observation, require_goal=False)
        current_rgb = copy.deepcopy(current_rgb)
        # ImageGoal remains fixed for an episode.  If a wrapper omits the goal
        # on later observations, retain the real goal image from reset.
        if next_goal is not None:
            goal_rgb = copy.deepcopy(next_goal)
        if frame_observer is not None:
            frame_observer({"rgb": copy.deepcopy(current_rgb), "imagegoal": copy.deepcopy(goal_rgb), "step": step_count, "action": action_name, "phase": "step"})
        episode_over = getattr(env, "episode_over", False)
        if callable(episode_over):
            episode_over = episode_over()
        done = bool(step_done or episode_over)
        if diagnostic_reach_radius is not None and current_distance < diagnostic_reach_radius:
            first_arrival_step = step_count
            termination_reason = "REACHED_GOAL"
            done = True
        elif stop_called:
            termination_reason = "STOP"
            done = True
        elif step_count >= max_steps:
            # A core Habitat env can set episode_over on its budget boundary;
            # classify this as timeout before the generic ENV_DONE reason.
            termination_reason = "MAX_STEPS"
            metrics.mark_timeout()
            done = True
        elif done:
            termination_reason = "ENV_DONE"

        if done:
            break

    if not done and step_count >= max_steps:
        metrics.mark_timeout()
        termination_reason = "MAX_STEPS"

    # Prefer official Habitat metrics whenever exposed.  The pure accumulator
    # still supplies path/step/timeout diagnostics and acts as a fake-env
    # fallback.
    official = _metrics_from_env(env)
    result_metrics = metrics.finalize(
        timeout=termination_reason == "MAX_STEPS",
        official_metrics=official,
    )
    episode_elapsed = (time.perf_counter_ns() - episode_start) / 1_000_000.0
    latency.record("episode_wall_ms", episode_elapsed, warmup=False)
    receipt: dict[str, Any] = {
        "episode_id": episode_id,
        "scene_id": actual_scene_token,
        "ledger_scene_id": expected_scene_token,
        "episode_key": (
            [actual_scene_token, episode_id]
            if actual_scene_token is not None and episode_id is not None
            else None
        ),
        "ledger_episode_id": (
            expected_episode_token
            if expected_episode_token is not None
            else None
        ),
        "termination_reason": termination_reason,
        "num_steps": step_count,
        "actions": actions,
        "metrics": result_metrics,
        "latency": latency.summary(),
        "factual_history_length": len(factual_history),
        "video_recording_in_episode_wall_time": frame_observer is not None,
    }
    # Flat aliases are convenient for aggregators and preserve the official
    # measure names in top-level JSON rows.
    receipt.update({name: result_metrics.get(name) for name in result_metrics})
    if diagnostic_reach_radius is not None:
        arrived = first_arrival_step is not None
        denominator = max(float(start_distance), float(result_metrics["path_length"]))
        reach_spl = (1.0 if first_arrival_step == 0 else
                     float(start_distance) / denominator if arrived and denominator > 0 else 0.0)
        # Oracle termination is an evaluator intervention, not a learned STOP
        # or an official Habitat Success/SPL. Preserve those fields verbatim.
        receipt.update(arrival_success=float(arrived), reach_spl=reach_spl,
                       first_arrival_step=first_arrival_step,
                       diagnostic_reach_radius=diagnostic_reach_radius,
                       evaluation_success_criterion="reach_only")
    return receipt


def load_habitat_environment(
    *,
    config_path: str | None = None,
    config: Any = None,
    dataset: Any = None,
    episode_indices: list[int] | None = None,
    episodes_path: str | None = None,
    scenes_dir: str | None = None,
    evaluation_contract: Mapping[str, Any] | None = None,
) -> Any:
    """Lazily construct a Habitat 0.2.4 ``Env`` from an official config.

    This function deliberately does not catch arbitrary constructor errors:
    asset/config failures should remain visible in the evaluation receipt.
    """

    if evaluation_contract is not None:
        # Validate the same resolved contract that the launcher recorded before
        # constructing Habitat.  This avoids validating one metadata mapping
        # and then silently loading a different default 256x256/90° config.
        validate_imagegoal_config(evaluation_contract)

    try:
        habitat = importlib.import_module("habitat")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Habitat-Lab 0.2.4 is unavailable; install the verified official "
            "runtime before launching MP3D evaluation"
        ) from exc

    runtime_version = getattr(habitat, "__version__", None)
    if runtime_version is None or str(runtime_version) != EXPECTED_HABITAT_VERSION:
        raise RuntimeError(
            f"Habitat runtime identity must expose version {EXPECTED_HABITAT_VERSION}, got {runtime_version!r}"
        )
    # Habitat-Lab 0.2.4 imports Habitat-Sim lazily in some installations.  A
    # separate identity check prevents a Lab source tree from being paired
    # with an arbitrary simulator binary.  The official package may expose
    # either ``__version__`` or a distribution version, so use both without
    # making packaging metadata a hard dependency.
    try:
        habitat_sim = importlib.import_module("habitat_sim")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Habitat-Sim runtime is unavailable; ImageGoal evaluation requires the verified 0.2.4 pair"
        ) from exc
    sim_version = getattr(habitat_sim, "__version__", None)
    if sim_version is None:
        try:
            from importlib import metadata as importlib_metadata

            sim_version = importlib_metadata.version("habitat-sim")
        except Exception:
            sim_version = None
    if sim_version is None or str(sim_version) != EXPECTED_HABITAT_VERSION:
        raise RuntimeError(
            f"Habitat-Sim runtime identity must be {EXPECTED_HABITAT_VERSION}, got {sim_version!r}"
        )

    if config is None:
        if config_path is None:
            raise ValueError("config_path or resolved Habitat config is required")
        try:
            get_config = importlib.import_module("habitat.config.default").get_config
            config = get_config(config_path)
        except Exception as exc:
            raise RuntimeError(f"could not load Habitat config {config_path!r}: {exc}") from exc
    if evaluation_contract is not None:
        _validate_runtime_settings(config, evaluation_contract)
    if dataset is None and episodes_path is not None:
        dataset = load_habitat_dataset(episodes_path, scenes_dir=scenes_dir)
    if episode_indices is not None:
        if dataset is None:
            raise ValueError("episode selection requires the canonical dataset")
        if not episode_indices or any(type(i) is not int or not 0 <= i < len(dataset.episodes) for i in episode_indices) or len(set(episode_indices)) != len(episode_indices):
            raise ValueError("episode indices must be unique valid canonical row indices")
        import copy
        selected = copy.copy(dataset)
        selected.episodes = [dataset.episodes[i] for i in episode_indices]
        dataset = selected
    if any(key.rsplit(".", 1)[-1].lower() == "type" and value == "ViewAlignedImageGoalSensor"
           for key, value in _iter_mapping_values(config)):
        from diagnostics.view_aligned_200_20260910.sensor import register_sensor
        register_sensor()
    env_cls = getattr(habitat, "Env", None)
    if env_cls is None:
        raise RuntimeError("installed Habitat package does not expose habitat.Env")
    return env_cls(config=config, dataset=dataset)


def load_habitat_dataset(
    episodes_path: str,
    *,
    scenes_dir: str | None = None,
) -> Any:
    """Load the supplied public episode ledger through Habitat's dataset API.

    This is deliberately a thin call to ``PointNavDatasetV1.from_json``.  It
    does not construct or alter episodes and therefore cannot silently replace
    a caller's ledger with a count-only loop.
    """

    path = str(episodes_path)
    opener = gzip.open if path.endswith(".gz") else open
    try:
        dataset_cls = importlib.import_module(
            "habitat.datasets.pointnav.pointnav_dataset"
        ).PointNavDatasetV1
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            "Habitat PointNavDatasetV1 is unavailable; cannot bind the supplied episode ledger"
        ) from exc
    dataset = dataset_cls()
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        payload = handle.read()
    dataset.from_json(payload, scenes_dir=scenes_dir)
    return dataset


def habitat_runtime_version() -> str | None:
    try:
        habitat = importlib.import_module("habitat")
    except ModuleNotFoundError:
        return None
    return getattr(habitat, "__version__", None)


def _iter_mapping_values(value: Any, prefix: str = ""):
    """Yield dotted keys from OmegaConf/dict-like runtime configs."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from _iter_mapping_values(child, path)


def _validate_runtime_settings(runtime_config: Any, contract: Mapping[str, Any]) -> None:
    """Check explicit sensor/action values in a resolved Habitat config.

    Habitat has changed nesting names between minor releases.  We therefore
    discover equivalent leaf names, but require every RGB/action field in the
    resolved config.  Internal depth dimensions remain outside this contract.
    """

    expected = {
        "width": int(contract["rgb"]["width"]),
        "height": int(contract["rgb"]["height"]),
        "hfov": float(contract["rgb"]["hfov"]),
        "forward_step_size": float(contract["actions"]["forward_step"]),
        "turn_angle": float(contract["actions"]["turn_angle_deg"]),
        "turn_angle_deg": float(contract["actions"]["turn_angle_deg"]),
    }
    if not isinstance(runtime_config, Mapping):
        raise ValueError("resolved Habitat runtime config must be a mapping")
    found: dict[str, bool] = {key: False for key in ("width", "height", "hfov", "position", "forward_step_size", "turn_angle")}
    for dotted, value in _iter_mapping_values(runtime_config):
        leaf = dotted.rsplit(".", 1)[-1].lower()
        if leaf == "position":
            path_parts = [part.lower() for part in dotted.split(".")]
            rgb_context = any(("rgb" in part) or part in {"camera", "camera_sensor", "color_sensor", "color"} for part in path_parts[:-1])
            if not rgb_context:
                continue
            actual_value = value
            target_value = contract["rgb"]["position"]
            try:
                if tuple(float(item) for item in actual_value) != tuple(float(item) for item in target_value):
                    raise ValueError(f"Habitat runtime setting {dotted}={value!r} contradicts frozen value {target_value!r}")
            except TypeError as exc:
                raise ValueError(f"Habitat runtime setting {dotted} is not a position sequence") from exc
            found["position"] = True
            continue
        if leaf not in expected:
            continue
        path_parts = [part.lower() for part in dotted.split(".")]
        # Width/height/HFOV are contract values only for the RGB sensor or
        # camera.  Habitat commonly keeps a 256x256 internal depth sensor;
        # treating its dimensions as RGB drift would reject a valid runtime.
        if leaf in {"width", "height", "hfov"}:
            rgb_context = any(
                ("rgb" in part)
                or part in {"camera", "camera_sensor", "color_sensor", "color"}
                for part in path_parts[:-1]
            )
            if not rgb_context:
                continue
        key = "turn_angle" if leaf == "turn_angle_deg" else leaf
        found[key] = True
        try:
            actual = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Habitat runtime setting {dotted} is not numeric")
        target = expected[leaf]
        if actual != target:
            raise ValueError(
                f"Habitat runtime setting {dotted}={value!r} contradicts frozen value {target!r}"
            )
    missing = [key for key, present in found.items() if not present]
    if missing:
        raise ValueError("resolved Habitat runtime config is missing required RGB/action fields: " + ", ".join(missing))


def validate_habitat_runtime_config(
    runtime_config: Any, evaluation_contract: Mapping[str, Any]
) -> None:
    """Public testable wrapper for runtime/contract binding."""

    validate_imagegoal_config(evaluation_contract)
    _validate_runtime_settings(runtime_config, evaluation_contract)
