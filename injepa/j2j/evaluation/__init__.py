"""Public ImageGoal evaluation seam.

Imports are intentionally lightweight: importing :mod:`j2j.evaluation` does
not import Habitat, NWM, or a checkpoint.  Those dependencies are loaded only
when the optional runner/adapter is explicitly used.
"""

from .adapters import ActionConversion, CallablePolicyAdapter, NWMContinuousActionAdapter
from .contracts import (
    DEFAULT_POLICY_KEYS,
    EXPECTED_ACTIONS,
    EXPECTED_EPISODE_COUNT,
    EXPECTED_HABITAT_VERSION,
    EXPECTED_SCENE_IDS,
    EXPECTED_SPLIT,
    EXPECTED_TASK,
    PRIVILEGED_OBSERVATION_KEYS,
    RGBOnlyObservationFirewall,
    canonical_episode_id,
    canonical_episode_key,
    canonical_scene_id,
    validate_imagegoal_config,
)
from .habitat_runner import (
    habitat_runtime_version,
    load_habitat_environment,
    run_imagegoal_episode,
    validate_habitat_runtime_config,
)
from .metrics import (
    HabitatMetricsAccumulator,
    LATENCY_NAMES,
    LatencyRecorder,
    METRIC_NAMES,
    aggregate_episode_metrics,
    aggregate_episode_latency,
    summarize_samples,
)
from .nwm_adapter import NWMOfficialChainAdapter, NWMPolicyAdapter
from .v9_adapter import V9PolicyAdapter
from .v9_factory import V9FactoryError, V9FactoryPending, V9RuntimeAdapter, build_v9_adapter

__all__ = [
    "ActionConversion",
    "CallablePolicyAdapter",
    "DEFAULT_POLICY_KEYS",
    "EXPECTED_ACTIONS",
    "EXPECTED_EPISODE_COUNT",
    "EXPECTED_HABITAT_VERSION",
    "EXPECTED_SCENE_IDS",
    "EXPECTED_SPLIT",
    "EXPECTED_TASK",
    "HabitatMetricsAccumulator",
    "LATENCY_NAMES",
    "LatencyRecorder",
    "METRIC_NAMES",
    "NWMContinuousActionAdapter",
    "NWMOfficialChainAdapter",
    "NWMPolicyAdapter",
    "PRIVILEGED_OBSERVATION_KEYS",
    "RGBOnlyObservationFirewall",
    "aggregate_episode_metrics",
    "aggregate_episode_latency",
    "canonical_episode_id",
    "canonical_episode_key",
    "canonical_scene_id",
    "habitat_runtime_version",
    "load_habitat_environment",
    "run_imagegoal_episode",
    "summarize_samples",
    "validate_imagegoal_config",
    "validate_habitat_runtime_config",
    "V9PolicyAdapter",
    "V9FactoryError",
    "V9FactoryPending",
    "V9RuntimeAdapter",
    "build_v9_adapter",
]
