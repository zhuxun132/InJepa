"""Pure contracts for the MP3D Habitat ImageGoal evaluation seam.

The module deliberately has no Habitat, simulator, or model dependency.  It
only validates the frozen experiment contract and creates an immutable policy
view of an observation.  Keeping these checks pure makes it possible to run
the scientific preflight and unit tests on a machine without Habitat installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


EXPECTED_HABITAT_VERSION = "0.2.4"
EXPECTED_TASK = "ImageNav"
EXPECTED_SPLIT = "val"
EXPECTED_EPISODE_COUNT = 495

# The public ImageGoal validation ledger used by this project.  Keep the order
# stable because it is part of the resolved evaluation identity.
EXPECTED_SCENE_IDS = (
    "2azQ1b91cZZ",
    "8194nk5LbLH",
    "EU6Fwq7SyZv",
    "QUCTc6BB5sX",
    "TbHJrupSAjP",
    "X7HyMhZNoso",
    "Z6MFQCViBuw",
    "oLBMNvg9in8",
    "pLe4wQe7qrG",
    "x8F5xyUWy9e",
    "zsNo4HB9uLZ",
)

EXPECTED_ACTIONS = {
    "BOS": -1,
    "STOP": 0,
    "MOVE_FORWARD": 1,
    "TURN_LEFT": 2,
    "TURN_RIGHT": 3,
}


def canonical_scene_id(
    value: Any,
    *,
    expected_scene_ids: tuple[str, ...] = EXPECTED_SCENE_IDS,
) -> str:
    """Resolve a Habitat scene path/token to the frozen scene identifier.

    Habitat episode JSON stores ``scene_id`` as a path (usually ending in
    ``<scene>.glb``), while ``current_episode.scene_id`` may use a different
    root prefix.  Comparing the canonical token keeps those equivalent forms
    stable without treating a bare episode ID as globally unique.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("scene_id must be a nonempty string")
    normalized = value.replace("\\", "/")
    components = [part for part in normalized.split("/") if part]
    matches: set[str] = set()
    for scene in expected_scene_ids:
        for component in components:
            stem = component
            for suffix in (".glb", ".navmesh", ".navmesh.bin"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
            if component == scene or stem == scene:
                matches.add(scene)
    if len(matches) != 1:
        raise ValueError(f"scene_id {value!r} is not one frozen MP3D scene")
    return next(iter(matches))


def canonical_episode_id(value: Any) -> str:
    """Normalize one official Habitat episode ID without inventing one."""

    if value is None:
        raise ValueError("episode identity is required")
    identity = str(value).strip()
    if not identity:
        raise ValueError("episode identity is required")
    return identity


def canonical_episode_key(
    scene_id: Any,
    episode_id: Any,
    *,
    expected_scene_ids: tuple[str, ...] = EXPECTED_SCENE_IDS,
) -> tuple[str, str]:
    """Return the stable ``(scene_id, episode_id)`` key used by evaluation."""

    return (
        canonical_scene_id(scene_id, expected_scene_ids=expected_scene_ids),
        canonical_episode_id(episode_id),
    )

PRIVILEGED_OBSERVATION_KEYS = frozenset(
    {
        "depth",
        "gps",
        "compass",
        "pose",
        "position",
        "rotation",
        "map",
        "top_down_map",
        "geodesic_distance",
        "distance_to_goal",
        "collision",
        "collisions",
        "measurements",
        "simulator_state",
        "agent_state",
    }
)

# These are the only keys that may be exposed to a policy by the common
# firewall.  ``history`` values are factual records supplied by the runner;
# they are not imagined rollouts.
DEFAULT_POLICY_KEYS = frozenset(
    {
        "rgb",
        "current_rgb",
        "imagegoal",
        "goal_rgb",
        "factual_history",
        "history",
        "previous_action",
        "action_history",
    }
)

FACTUAL_HISTORY_KEYS = frozenset({"rgb", "action", "order", "mask"})


def _plain(value: Any) -> Any:
    """Convert OmegaConf-like mappings/lists into ordinary Python values."""

    # Import lazily: evaluation contracts must work without OmegaConf.
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore

        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:  # pragma: no cover - optional dependency path
        pass
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_policy_value(value: Any) -> Any:
    """Deep-freeze policy-visible records without requiring NumPy/Torch."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_policy_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_policy_value(item) for item in value)
    # RGB arrays/tensors are often mutable.  Make a private read-only copy when
    # the backend exposes the conventional NumPy ``copy/setflags`` ABI; other
    # opaque image handles remain untouched and cannot be changed through the
    # immutable mapping itself.
    copier = getattr(value, "copy", None)
    if callable(copier) and hasattr(value, "setflags"):
        try:
            copied = copier()
            copied.setflags(write=False)
            return copied
        except Exception:
            pass
    return value


def _validate_factual_history(history: Any) -> tuple[Any, ...]:
    if history is None:
        return ()
    if isinstance(history, (str, bytes, Mapping)):
        raise TypeError("factual history must be a sequence of records")
    try:
        records = tuple(history)
    except TypeError as exc:
        raise TypeError("factual history must be a sequence of records") from exc
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"factual_history[{index}] must be a mapping")
        keys = {str(key) for key in record}
        unknown = keys - FACTUAL_HISTORY_KEYS
        if unknown:
            raise ValueError(
                f"factual_history[{index}] contains non-factual keys: "
                + ", ".join(sorted(unknown))
            )
        if "rgb" not in record or "action" not in record:
            raise ValueError(
                f"factual_history[{index}] requires real rgb and executed action"
            )
        if "order" in record:
            order = record["order"]
            if isinstance(order, bool) or not isinstance(order, int) or order < 0:
                raise ValueError(f"factual_history[{index}].order must be nonnegative integer")
        if "mask" in record and not isinstance(record["mask"], bool):
            raise TypeError(f"factual_history[{index}].mask must be boolean")
    return tuple(_freeze_policy_value(record) for record in records)


def _number_equal(actual: Any, expected: float, *, name: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    if float(actual) != float(expected):
        raise ValueError(f"{name} must equal {expected!r}; got {actual!r}")


def _require_mapping(config: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return config


def _require_key(mapping: Mapping[str, Any], key: str, parent: str = "config") -> Any:
    if key not in mapping:
        raise ValueError(f"{parent}.{key} is required")
    return mapping[key]


def validate_imagegoal_config(
    config: Mapping[str, Any], *, require_assets: bool = False
) -> Mapping[str, Any]:
    """Validate and freeze the approved MP3D ImageGoal evaluation contract.

    ``require_assets=False`` is intentionally the default so pure unit tests
    can validate the scientific task without knowing machine-specific paths.
    The preflight CLI enables ``require_assets`` after loading the resolved
    asset manifest.  No GPU count, worker count, or server path is assumed.
    """

    raw = _plain(config)
    root = _require_mapping(raw, "config")

    # A resolved Habitat config may be wrapped under ``evaluation``.  Do not
    # silently choose a nested section when the caller supplied contradictory
    # top-level values; only use the wrapper when the canonical fields are not
    # present at the root.
    if "evaluation" in root and "task" not in root:
        nested = _require_mapping(root["evaluation"], "config.evaluation")
        merged = dict(root)
        merged.update(nested)
        root = merged

    version = _require_key(root, "habitat_version")
    if str(version) != EXPECTED_HABITAT_VERSION:
        raise ValueError(
            f"habitat_version must be {EXPECTED_HABITAT_VERSION!r}; got {version!r}"
        )
    task = _require_key(root, "task")
    if str(task) != EXPECTED_TASK:
        raise ValueError(f"task must be {EXPECTED_TASK!r}; got {task!r}")
    split = _require_key(root, "split")
    if str(split) != EXPECTED_SPLIT:
        raise ValueError(f"split must be {EXPECTED_SPLIT!r}; got {split!r}")

    episode_count = _require_key(root, "episode_count")
    if isinstance(episode_count, bool) or not isinstance(episode_count, int):
        raise TypeError("episode_count must be an integer")
    profile = root.get("evaluation_profile")
    if profile not in (None, "distance_stratified_visible_v1"):
        raise ValueError("unknown evaluation_profile")
    if profile == "distance_stratified_visible_v1":
        if episode_count <= 0:
            raise ValueError("custom episode_count must be positive")
        radius = root.get("success_distance")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius != 1.0:
            raise ValueError("distance_stratified_visible_v1 requires success_distance 1m")
    elif episode_count != EXPECTED_EPISODE_COUNT:
        raise ValueError(
            f"episode_count must be {EXPECTED_EPISODE_COUNT}; got {episode_count}"
        )

    scene_ids = _require_key(root, "scene_ids")
    if not isinstance(scene_ids, (list, tuple)):
        raise TypeError("scene_ids must be a list or tuple")
    if tuple(str(item) for item in scene_ids) != EXPECTED_SCENE_IDS:
        raise ValueError("scene_ids do not match the frozen 11-scene ledger")

    agent = _require_mapping(_require_key(root, "agent"), "config.agent")
    _number_equal(_require_key(agent, "height", "agent"), 1.5, name="agent.height")
    _number_equal(_require_key(agent, "radius", "agent"), 0.1, name="agent.radius")
    if "allow_sliding" in agent and not isinstance(agent["allow_sliding"], bool):
        raise TypeError("agent.allow_sliding must be boolean")
    if "physics" in agent and not isinstance(agent["physics"], bool):
        raise TypeError("agent.physics must be boolean")

    actions = _require_mapping(_require_key(root, "actions"), "config.actions")
    for key, expected in EXPECTED_ACTIONS.items():
        actual = _require_key(actions, key, "actions")
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"actions.{key} must be an integer")
        if actual != expected:
            raise ValueError(f"actions.{key} must equal {expected}; got {actual}")
    _number_equal(
        _require_key(actions, "forward_step", "actions"),
        0.25,
        name="actions.forward_step",
    )
    _number_equal(
        _require_key(actions, "turn_angle_deg", "actions"),
        15.0,
        name="actions.turn_angle_deg",
    )

    rgb = _require_mapping(_require_key(root, "rgb"), "config.rgb")
    for key, expected in (("width", 640), ("height", 480)):
        actual = _require_key(rgb, key, "rgb")
        if isinstance(actual, bool) or not isinstance(actual, int):
            raise TypeError(f"rgb.{key} must be an integer")
        if actual != expected:
            raise ValueError(f"rgb.{key} must equal {expected}; got {actual}")
    _number_equal(_require_key(rgb, "hfov", "rgb"), 79.0, name="rgb.hfov")
    position = _require_key(rgb, "position", "rgb")
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        raise ValueError("rgb.position must contain three values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in position):
        raise TypeError("rgb.position must contain real scalars")
    if tuple(float(v) for v in position) != (0.0, 1.25, 0.0):
        raise ValueError("rgb.position must equal [0.0, 1.25, 0.0]")

    if "policy_observation_keys" in root:
        policy_keys = root["policy_observation_keys"]
        if not isinstance(policy_keys, (list, tuple)):
            raise TypeError("policy_observation_keys must be a list")
        if not {str(k) for k in policy_keys}.issubset(DEFAULT_POLICY_KEYS):
            raise ValueError("policy_observation_keys contains a privileged/unknown key")
    if "privileged_observation_keys" in root:
        privileged = root["privileged_observation_keys"]
        if not isinstance(privileged, (list, tuple)):
            raise TypeError("privileged_observation_keys must be a list")
        if not PRIVILEGED_OBSERVATION_KEYS.issuperset(str(k) for k in privileged):
            raise ValueError("privileged_observation_keys contains an unknown key")

    if "max_episode_steps" in root:
        maximum = root["max_episode_steps"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("max_episode_steps must be a positive integer")

    if require_assets:
        assets = root.get("assets")
        if not isinstance(assets, Mapping):
            raise ValueError("assets mapping is required for resolved preflight config")
        for key in (
            "mp3d_archive",
            "scene_root",
            "episodes",
            "habitat_lab_revision",
            "habitat_sim_revision",
        ):
            value = assets.get(key)
            if value is None or value == "":
                raise ValueError(f"assets.{key} is required for resolved preflight config")

    # Normalize aliases used by the runner while retaining all user-provided
    # metadata.  The immutable result prevents a later component from silently
    # changing the experiment identity after preflight.
    normalized = dict(root)
    normalized["habitat_version"] = EXPECTED_HABITAT_VERSION
    normalized["task"] = EXPECTED_TASK
    normalized["split"] = EXPECTED_SPLIT
    normalized["episode_count"] = episode_count
    normalized["scene_ids"] = list(EXPECTED_SCENE_IDS)
    return _freeze(normalized)


class RGBOnlyObservationFirewall:
    """Reject privileged simulator measurements before policy invocation."""

    def __init__(
        self,
        *,
        allowed_keys: Any = None,
        privileged_keys: Any = None,
    ) -> None:
        self.allowed_keys = frozenset(
            DEFAULT_POLICY_KEYS
            if allowed_keys is None
            else (str(k) for k in allowed_keys)
        )
        self.privileged_keys = frozenset(
            PRIVILEGED_OBSERVATION_KEYS
            if privileged_keys is None
            else (str(k).lower() for k in privileged_keys)
        )

    def filter(
        self,
        observation: Mapping[str, Any],
        *,
        history: Any = None,
    ) -> Mapping[str, Any]:
        """Return an immutable RGB/goal/history view without mutating input."""

        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        lowered = {str(key).lower(): key for key in observation}
        leaked = sorted(
            key
            for lowered_key, key in lowered.items()
            if lowered_key in self.privileged_keys
            or any(lowered_key.startswith(f"{name}.") for name in self.privileged_keys)
        )
        if leaked:
            raise ValueError(
                "privileged simulator observation cannot reach policy: "
                + ", ".join(str(k) for k in leaked)
            )
        unknown = [key for key in observation if str(key) not in self.allowed_keys]
        if unknown:
            raise ValueError(
                "unknown/non-RGB observation key cannot reach policy: "
                + ", ".join(str(k) for k in unknown)
            )
        if "rgb" not in observation and "current_rgb" not in observation:
            raise ValueError("policy observation requires current rgb")
        if "imagegoal" not in observation and "goal_rgb" not in observation:
            raise ValueError("policy observation requires imagegoal/goal_rgb")
        result = {
            str(key): _freeze_policy_value(value)
            for key, value in observation.items()
        }
        if "factual_history" in result:
            result["factual_history"] = _validate_factual_history(
                result["factual_history"]
            )
        if "history" in result:
            result["history"] = _validate_factual_history(result["history"])
        if "action_history" in result:
            result["action_history"] = _validate_factual_history(
                result["action_history"]
            )
        if history is not None:
            result["factual_history"] = _validate_factual_history(history)
        return MappingProxyType(result)

    # A descriptive alias is useful at call sites and keeps compatibility with
    # the design document's terminology.
    policy_view = filter
