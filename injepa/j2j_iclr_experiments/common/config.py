"""Strict resolution for the shared Context4 experiment-suite config."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
import re
from typing import Any


_SECRET_KEY = re.compile(r"credential|secret|token|password|proxy", re.IGNORECASE)
_SHA256 = re.compile(r"[0-9a-f]{64}")
PREREGISTERED_ACTIVE_PREFIXES = frozenset({1, 2, 4})
QUALITATIVE_REQUIRED_STRATA = frozenset(
    {
        *(
            f"g:{view}:{outcome}"
            for view in ("local", "goal", "terminal")
            for outcome in ("correct", "incorrect")
        ),
        "ranking:full:winner",
        "ranking:full:nonwinner",
    }
)
_REQUIRED_SECTIONS = {
    "checkpoint",
    "model",
    "offline",
    "metrics",
    "statistics",
    "qualitative",
    "online",
    "stop",
    "local",
    "latency",
    "runtime",
    "output",
}
_OPTIONAL_SECTIONS: set[str] = set()
_SECTION_FIELDS = {
    "checkpoint": {
        "path",
        "sha256",
        "training_resolved_config_path",
        "training_resolved_config_sha256",
        "training_code_sha256",
    },
    "model": {"k_model", "h_model", "k_active", "h_active"},
    "offline": {"coverage", "arms", "trajectory_budget"},
    "metrics": {"ece_bins", "cosine_epsilon", "calibration_bins"},
    "statistics": {"replicates", "seed", "ci_level"},
    "qualitative": {"limit", "per_stratum_limit"},
    "online": {"max_diagnostics"},
    "stop": {
        "mode",
        "receipt",
        "receipt_sha256",
        "calibration_ledger_path",
        "calibration_ledger_sha256",
        "calibration_provenance_path",
        "calibration_provenance_sha256",
    },
    "local": {
        "state_budget",
        "episode_budget",
        "step_budget",
        "methods",
        "candidate_budget_per_method",
        "capability_receipt",
        "capability_receipt_sha256",
        "evaluation_overlay_path",
        "evaluation_overlay_sha256",
        "sensor_config_path",
        "sensor_config_sha256",
        "preflight_receipt_path",
        "preflight_receipt_sha256",
        "episode_ledger_path",
        "episode_ledger_sha256",
        "decision_path",
        "decision_sha256",
        "analysis_manifest_path",
        "analysis_manifest_sha256",
    },
    "latency": {"warmup", "repeats", "fixture_trajectory_key"},
    "runtime": {
        "world_size",
        "device",
        "workers",
        "distributed_timeout_seconds",
    },
    "output": {"root", "allow_overwrite"},
}


def _reject_secret_keys(value: object, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            if _SECRET_KEY.search(key):
                raise ValueError(
                    f"secret/credential/token/password/proxy field is forbidden: "
                    f"{path}.{key}"
                )
            _reject_secret_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secret_keys(child, path=f"{path}[{index}]")


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"config section {name!r} must be a mapping")
    result = dict(value)
    unknown = set(result) - _SECTION_FIELDS[name]
    if unknown:
        raise ValueError(f"config section {name!r} has unknown fields: {sorted(unknown)}")
    return result


def _positive_integer(section: Mapping[str, Any], key: str, *, path: str) -> int:
    value = section.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path}.{key} must be a positive integer (bool is invalid)")
    return value


def _nonnegative_integer(section: Mapping[str, Any], key: str, *, path: str) -> int:
    value = section.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"{path}.{key} must be a non-negative integer (bool is invalid)")
    return value


def _finite_positive(section: Mapping[str, Any], key: str, *, path: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path}.{key} must be a finite positive number, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{path}.{key} must be finite and positive")
    return converted


def _nonempty_string(section: Mapping[str, Any], key: str, *, path: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _sha256(section: Mapping[str, Any], key: str, *, path: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{path}.{key} must be a lowercase SHA-256 digest")
    return value


def _optional_nonempty_string(
    section: Mapping[str, Any], key: str, *, path: str
) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path}.{key} must be null or a non-empty string")
    return value


def _optional_sha256(
    section: Mapping[str, Any], key: str, *, path: str
) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    return _sha256(section, key, path=path)


def resolve_suite_config(raw: Mapping[str, object]) -> dict[str, Any]:
    """Validate and deep-copy one parameterized experiment-suite config.

    Variant labels are deliberately absent: identity is derived from the
    checkpoint-bound training sidecar by the loader.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("suite config must be a mapping")
    _reject_secret_keys(raw)
    config = copy.deepcopy(dict(raw))

    if "variant" in config:
        raise ValueError(
            "evaluator-side variant label is forbidden; use checkpoint sidecar identity"
        )
    missing = _REQUIRED_SECTIONS - set(config)
    if missing:
        raise ValueError(f"suite config is missing required sections: {sorted(missing)}")
    unknown = set(config) - _REQUIRED_SECTIONS - _OPTIONAL_SECTIONS
    if unknown:
        raise ValueError(f"suite config has unknown sections: {sorted(unknown)}")

    checkpoint = _section(config, "checkpoint")
    for key in ("path", "training_resolved_config_path"):
        _nonempty_string(checkpoint, key, path="checkpoint")
    for key in ("sha256", "training_resolved_config_sha256", "training_code_sha256"):
        _sha256(checkpoint, key, path="checkpoint")

    model = _section(config, "model")
    for key in ("k_model", "h_model", "k_active", "h_active"):
        _positive_integer(model, key, path="model")
    if model["k_active"] > model["k_model"]:
        raise ValueError("active K cannot exceed checkpoint model K")
    if model["h_active"] > model["h_model"]:
        raise ValueError("active H cannot exceed checkpoint model H")
    for field in ("k_active", "h_active"):
        if model[field] not in PREREGISTERED_ACTIVE_PREFIXES:
            raise ValueError(
                f"model.{field} is not in the preregistered active K/H set "
                f"{sorted(PREREGISTERED_ACTIVE_PREFIXES)}"
            )

    offline = _section(config, "offline")
    coverage = _nonempty_string(offline, "coverage", path="offline")
    if coverage == "full_project_dev":
        if "trajectory_budget" in offline:
            raise ValueError("offline full_project_dev forbids trajectory_budget")
    elif coverage == "bounded":
        if "trajectory_budget" not in offline:
            raise ValueError("offline bounded coverage requires trajectory_budget")
        _positive_integer(offline, "trajectory_budget", path="offline")
    else:
        raise ValueError(
            "offline.coverage must be full_project_dev or bounded"
        )
    arms = offline.get("arms")
    if not isinstance(arms, (list, tuple)) or not arms:
        raise ValueError("offline.arms must be a non-empty sequence")
    if any(not isinstance(arm, str) or not arm for arm in arms):
        raise TypeError("offline.arms entries must be non-empty strings")
    if len(set(arms)) != len(arms):
        raise ValueError("offline.arms must not contain duplicates")

    metrics = _section(config, "metrics")
    _positive_integer(metrics, "ece_bins", path="metrics")
    metrics["cosine_epsilon"] = _finite_positive(
        metrics, "cosine_epsilon", path="metrics"
    )
    if "calibration_bins" in metrics:
        _positive_integer(metrics, "calibration_bins", path="metrics")

    statistics = _section(config, "statistics")
    _positive_integer(statistics, "replicates", path="statistics")
    _nonnegative_integer(statistics, "seed", path="statistics")
    ci_level = _finite_positive(statistics, "ci_level", path="statistics")
    if ci_level >= 1.0:
        raise ValueError("statistics.ci_level must be finite and between zero and one")
    statistics["ci_level"] = ci_level

    qualitative = _section(config, "qualitative")
    _positive_integer(qualitative, "limit", path="qualitative")
    _positive_integer(
        qualitative,
        "per_stratum_limit",
        path="qualitative",
    )
    if qualitative["limit"] < len(QUALITATIVE_REQUIRED_STRATA):
        raise ValueError(
            "qualitative.limit must cover all "
            f"{len(QUALITATIVE_REQUIRED_STRATA)} required strata"
        )

    online = _section(config, "online")
    _positive_integer(online, "max_diagnostics", path="online")

    stop = _section(config, "stop")
    mode = _nonempty_string(stop, "mode", path="stop")
    if mode not in {"never_stop", "calibrated"}:
        raise ValueError("stop.mode must be never_stop or calibrated")
    receipt = _optional_nonempty_string(stop, "receipt", path="stop")
    receipt_sha = _optional_sha256(stop, "receipt_sha256", path="stop")
    if (receipt is None) != (receipt_sha is None):
        raise ValueError("stop receipt path and SHA must be supplied together")
    if mode == "calibrated" and receipt is None:
        raise ValueError("calibrated STOP requires an admitted receipt path and SHA")
    if mode == "never_stop" and receipt is not None:
        raise ValueError("never_stop cannot claim a calibrated STOP receipt")
    ledger_path = _optional_nonempty_string(
        stop, "calibration_ledger_path", path="stop"
    )
    ledger_sha = _optional_sha256(
        stop, "calibration_ledger_sha256", path="stop"
    )
    if (ledger_path is None) != (ledger_sha is None):
        raise ValueError("STOP calibration ledger path and SHA must be supplied together")
    provenance_path = _optional_nonempty_string(
        stop, "calibration_provenance_path", path="stop"
    )
    provenance_sha = _optional_sha256(
        stop, "calibration_provenance_sha256", path="stop"
    )
    if (provenance_path is None) != (provenance_sha is None):
        raise ValueError(
            "STOP calibration provenance path and SHA must be supplied together"
        )

    local = _section(config, "local")
    _positive_integer(local, "state_budget", path="local")
    for key in ("episode_budget", "step_budget"):
        _positive_integer(local, key, path="local")
    methods = local.get("methods")
    if methods != ["Full", "NoG-enumerate"]:
        raise ValueError(
            "local.methods must be exactly [Full, NoG-enumerate] in that order"
        )
    budgets = local.get("candidate_budget_per_method")
    if not isinstance(budgets, Mapping):
        raise TypeError("local.candidate_budget_per_method must be a mapping")
    expected_methods = {"Full", "NoG-enumerate"}
    if set(budgets) != expected_methods:
        raise ValueError(
            "local candidate budget mapping must contain exactly Full and NoG-enumerate"
        )
    complete_candidate_count = int(model["k_active"]) * int(model["h_active"])
    for method in ("Full", "NoG-enumerate"):
        value = budgets[method]
        if type(value) is not int or value != complete_candidate_count:
            raise ValueError(
                "local candidate budget for each method must equal the complete K*H set"
            )
    local["candidate_budget_per_method"] = dict(budgets)
    required_identity_pairs = (
        ("capability_receipt", "capability_receipt_sha256"),
        ("evaluation_overlay_path", "evaluation_overlay_sha256"),
        ("sensor_config_path", "sensor_config_sha256"),
        ("preflight_receipt_path", "preflight_receipt_sha256"),
        ("episode_ledger_path", "episode_ledger_sha256"),
        ("decision_path", "decision_sha256"),
        ("analysis_manifest_path", "analysis_manifest_sha256"),
    )
    for path_key, sha_key in required_identity_pairs:
        if path_key not in local or sha_key not in local:
            raise ValueError(
                f"local {path_key} path and {sha_key} SHA are required together"
            )
        _nonempty_string(local, path_key, path="local")
        _sha256(local, sha_key, path="local")

    latency = _section(config, "latency")
    _nonnegative_integer(latency, "warmup", path="latency")
    _positive_integer(latency, "repeats", path="latency")
    fixture_key = _nonempty_string(
        latency, "fixture_trajectory_key", path="latency"
    )
    if _SHA256.fullmatch(fixture_key) is None:
        raise ValueError(
            "latency.fixture_trajectory_key must be a lowercase SHA-256 trajectory key"
        )

    output = _section(config, "output")
    _nonempty_string(output, "root", path="output")
    allow_overwrite = output.get("allow_overwrite", False)
    if type(allow_overwrite) is not bool:
        raise TypeError("output.allow_overwrite must be bool")
    if allow_overwrite:
        raise ValueError("output overwrite is forbidden")
    output["allow_overwrite"] = False

    runtime = _section(config, "runtime")
    _positive_integer(runtime, "world_size", path="runtime")
    device = _nonempty_string(runtime, "device", path="runtime")
    if device not in {"cpu", "cuda"}:
        raise ValueError("runtime.device must be cpu or cuda")
    _nonnegative_integer(runtime, "workers", path="runtime")
    _positive_integer(
        runtime,
        "distributed_timeout_seconds",
        path="runtime",
    )

    config.update(
        {
            "checkpoint": checkpoint,
            "model": model,
            "offline": offline,
            "metrics": metrics,
            "statistics": statistics,
            "qualitative": qualitative,
            "online": online,
            "stop": stop,
            "local": local,
            "latency": latency,
            "runtime": runtime,
            "output": output,
        }
    )
    return config


__all__ = [
    "PREREGISTERED_ACTIVE_PREFIXES",
    "QUALITATIVE_REQUIRED_STRATA",
    "resolve_suite_config",
]
