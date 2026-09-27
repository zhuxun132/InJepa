"""Run one recurrent RAW closed-loop configuration against an ordered ledger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import math
from pathlib import Path
from typing import Any
from j2j.evaluation.episode_sharding import episode_shard_indices
from j2j.evaluation.video import FirstPersonVideoRecorder

import yaml

import j2j.evaluation.habitat_runner as habitat_runner_module
import j2j.evaluation.metrics as metrics_module
import j2j.evaluation.vjepa_grid_encoder as encoder_module
import j2j_iclr_experiments.common.checkpoint as checkpoint_module
import j2j_iclr_experiments.common.statistics as statistics_module
import j2j_iclr_experiments.local_habitat.capability as capability_module
import j2j_iclr_experiments.online.adapter as adapter_module
import j2j_iclr_experiments.online.planner as planner_module
import j2j_iclr_experiments.offline.rankers as rankers_module
import j2j_iclr_experiments.offline.whole_branch_rank as whole_branch_module
import j2j_iclr_experiments.offline.stagnation_guard as stagnation_module
import j2j_recurrent_experiments.closed_loop.factory as factory_module
import j2j_recurrent_experiments.closed_loop.raw_stop as raw_stop_module
from j2j.evaluation.habitat_runner import (
    load_habitat_environment,
    run_imagegoal_episode,
)
from j2j.evaluation.vjepa_grid_encoder import build_native_vjepa_grid_encoder
from j2j_iclr_experiments.common.artifacts import (
    canonical_mapping_sha256,
    create_once_json,
    file_identity,
    require_sha256,
    to_plain_json,
)
from j2j_iclr_experiments.common.checkpoint import (
    load_eval_checkpoint,
    require_exact_final_checkpoint,
    require_completed_epoch_checkpoint,
)
from j2j_iclr_experiments.local_habitat.capability import (
    validate_formal_reset_replay_capability,
)
from scripts.preflight_imagegoal_assets import _load_episodes, validate_episode_ledger

from .factory import build_recurrent_adapter
from .raw_stop import validate_raw_stop_calibration_receipt


_CONFIG_SCHEMA = "J2J_RECURRENT_RAW_CLOSED_LOOP_CONFIG_V1"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _bound_file(section: Mapping[str, Any], path_key: str, sha_key: str,
                name: str) -> dict[str, Any]:
    expected = require_sha256(section.get(sha_key), name=f"{name} SHA")
    observed = file_identity(section.get(path_key), name=name)
    if observed["sha256"] != expected:
        raise ValueError(f"{name} live bytes/SHA identity drift")
    return observed


def _load_json(identity: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(str(identity["path"])).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    return _mapping(value, name)


def _load_evaluation_contract(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(Path(str(identity["path"])).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("Habitat evaluation config is not valid YAML/JSON") from exc
    return _mapping(value, "Habitat evaluation config")


def _artifact_matches(value: object, identity: Mapping[str, Any], name: str) -> None:
    item = _mapping(value, f"capability {name}")
    if Path(str(item.get("path", ""))).expanduser().resolve() != Path(
        str(identity["path"])
    ) or item.get("sha256") != identity["sha256"]:
        raise ValueError(f"Habitat capability {name} differs from configured artifact")


def _close_capability_artifacts(
    capability: Mapping[str, Any], *, sensor: Mapping[str, Any],
    evaluation: Mapping[str, Any], ledger: Mapping[str, Any],
) -> None:
    artifacts = _mapping(capability.get("artifacts"), "Habitat capability artifacts")
    _artifact_matches(artifacts.get("sensor_config"), sensor, "sensor config")
    _artifact_matches(artifacts.get("episode_ledger"), ledger, "episode ledger")
    evaluation_source = artifacts.get("evaluation_config_source")
    # A validated V3 receipt always has this field.  Its absence is tolerated
    # only at an injected validator seam used by the bounded unit fixture.
    if evaluation_source is not None:
        source = _mapping(evaluation_source, "capability evaluation config source")
        if Path(str(source.get("path", ""))).expanduser().resolve() != Path(
            str(evaluation["path"])
        ):
            raise ValueError(
                "Habitat capability evaluation config differs from configured artifact"
            )


def _scene_token(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("episode ledger scene_id must be a nonempty string")
    name = Path(value.replace("\\", "/")).name
    for suffix in (".navmesh.bin", ".navmesh", ".glb"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        raise ValueError("episode ledger scene_id has no canonical token")
    return name


def _validate_ordered_ledger(
    episodes: list[dict[str, Any]], evaluation_contract: Mapping[str, Any],
    *, formal: bool,
) -> Mapping[str, Any]:
    expected_count = evaluation_contract.get("episode_count")
    scene_ids = evaluation_contract.get("scene_ids")
    if expected_count is None or scene_ids is None:
        if formal:
            raise ValueError(
                "formal evaluation contract must declare episode_count and scene_ids"
            )
        expected_count = len(episodes)
        scene_ids = list(dict.fromkeys(_scene_token(row.get("scene_id")) for row in episodes))
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("evaluation episode_count must be a positive integer")
    if (
        not isinstance(scene_ids, (list, tuple))
        or not scene_ids
        or any(not isinstance(scene, str) or not scene for scene in scene_ids)
        or len(set(scene_ids)) != len(scene_ids)
    ):
        raise ValueError("evaluation scene_ids must be unique nonempty strings")
    return validate_episode_ledger(
        episodes,
        expected_count=expected_count,
        expected_scene_ids=tuple(scene_ids),
    )


def _close_evaluation_contract(
    evaluation: Mapping[str, Any], habitat: Mapping[str, Any], *, formal: bool,
) -> None:
    assets = evaluation.get("assets")
    if assets is None:
        if formal:
            raise ValueError("formal evaluation contract must declare its assets")
        return
    assets = _mapping(assets, "Habitat evaluation assets")
    configured_ledger = Path(str(habitat.get("episode_ledger", ""))).expanduser().resolve()
    configured_scenes = Path(str(habitat.get("scene_root", ""))).expanduser().resolve()
    if Path(str(assets.get("episodes", ""))).expanduser().resolve() != configured_ledger:
        raise ValueError("evaluation contract episode ledger differs from configured artifact")
    if Path(str(assets.get("scene_root", ""))).expanduser().resolve() != configured_scenes:
        raise ValueError("evaluation contract scene root differs from configured artifact")
    if "max_episode_steps" in evaluation and evaluation["max_episode_steps"] != habitat.get(
        "max_episode_steps"
    ):
        raise ValueError("evaluation max_episode_steps differs from runner config")
    if "success_distance" in evaluation and float(evaluation["success_distance"]) != float(
        habitat.get("success_distance")
    ):
        raise ValueError("evaluation success_distance differs from runner config")


def build_configuration_cases(
    base_config: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply explicit K/H/deployment cases without mutating shared identities."""
    _mapping(base_config, "closed-loop base config")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ValueError("closed-loop cases must be a nonempty sequence")
    allowed = {"configuration_id", "deployment", "active_k", "active_h"}
    result: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw in enumerate(cases):
        case = _mapping(raw, f"closed-loop case {index}")
        if set(case) != allowed:
            raise ValueError("closed-loop case fields must be configuration/deployment/K/H")
        identifier = case["configuration_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("closed-loop configuration IDs must be unique nonempty strings")
        if case["deployment"] not in {
            "full", "no_f", "proposal_only", "posthoc_f", "f_feedback_q", "goal_gf"
        }:
            raise ValueError(
                "closed-loop deployment must be full, no_f, proposal_only, posthoc_f, f_feedback_q, or goal_gf"
            )
        _positive_integer(case["active_k"], "active_k")
        _positive_integer(case["active_h"], "active_h")
        configured = copy.deepcopy(dict(base_config))
        configured.update(copy.deepcopy(dict(case)))
        result.append(configured)
        identifiers.add(identifier)
    return result


def _validate_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema") != _CONFIG_SCHEMA:
        raise ValueError("closed-loop config schema differs")
    protocol = require_sha256(config.get("protocol_sha256"), name="protocol SHA")
    identifier = config.get("configuration_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("configuration_id must be a nonempty string")
    tier = config.get("run_tier")
    if tier not in {"formal", "diagnostic"}:
        raise ValueError("run_tier must be formal or diagnostic")
    step_trace = config.get("diagnostic_step_trace", False)
    if type(step_trace) is not bool or (step_trace and tier != "diagnostic"):
        raise ValueError("diagnostic_step_trace must be boolean and diagnostic-only")
    stop_override = config.get("diagnostic_stop_threshold")
    reach_radius = config.get("diagnostic_reach_radius")
    if reach_radius is not None:
        if tier != "diagnostic":
            raise ValueError("reach-only requires diagnostic run tier")
        if (isinstance(reach_radius, bool) or not isinstance(reach_radius, (int, float))
                or not math.isfinite(reach_radius) or reach_radius <= 0):
            raise ValueError("diagnostic reach radius must be positive and finite")
        if stop_override is not None:
            raise ValueError("reach-only and diagnostic STOP threshold are mutually exclusive")
    if stop_override is not None:
        if tier != "diagnostic":
            raise ValueError("STOP threshold override requires diagnostic run tier")
        if (isinstance(stop_override, bool) or not isinstance(stop_override, (int, float))
                or not math.isfinite(stop_override) or stop_override < 0):
            raise ValueError("diagnostic STOP threshold must be finite and nonnegative")
    admission = config.get("checkpoint_admission", "exact_final")
    if admission not in {"exact_final", "completed_epoch"}:
        raise ValueError("unknown checkpoint_admission")
    if tier == "formal" and admission != "exact_final":
        raise ValueError("formal evaluation requires exact-final checkpoint")
    shard_index = _nonnegative_integer(config.get("shard_index", 0), "shard_index")
    shard_count = _positive_integer(config.get("shard_count", 1), "shard_count")
    if shard_index >= shard_count:
        raise ValueError("shard_index must be smaller than shard_count")
    if tier == "formal" and shard_count != 1:
        raise ValueError("formal RAW evaluation requires an unsharded complete result")
    episode_indices = None
    if "diagnostic_episode_indices" in config:
        selection = config["diagnostic_episode_indices"]
        if (tier != "diagnostic" or not isinstance(selection, list) or not selection
                or any(type(index) is not int or index < 0 for index in selection)
                or selection != sorted(set(selection))):
            raise ValueError("diagnostic subset must be a nonempty sorted unique integer list")
        episode_indices = list(selection)
    video = _mapping(config.get("video", {}), "video")
    video_budget = _nonnegative_integer(video.get("episode_budget", 0), "video episode_budget")
    fps = video.get("fps", 4)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("video fps must be positive and finite")
    if video_budget and (not isinstance(video.get("directory"), str) or not video["directory"]):
        raise ValueError("video directory is required when recording")
    deployment = config.get("deployment")
    if deployment not in {"full", "no_f", "proposal_only", "posthoc_f", "f_feedback_q", "goal_gf"}:
        raise ValueError(
            "deployment must be full, no_f, proposal_only, posthoc_f, f_feedback_q, or goal_gf"
        )
    whole_branch = None
    if "diagnostic_whole_branch_ranking" in config:
        from j2j_iclr_experiments.offline.whole_branch_rank import validate_whole_branch_options
        if tier != "diagnostic" or deployment not in {
            "full", "posthoc_f", "f_feedback_q", "goal_gf"
        }:
            raise ValueError(
                "whole branch ranking requires diagnostic full, posthoc_f, f_feedback_q, or goal_gf deployment"
            )
        whole_branch = validate_whole_branch_options(config["diagnostic_whole_branch_ranking"])
    if deployment in {"posthoc_f", "f_feedback_q"} and whole_branch is None:
        raise ValueError(f"{deployment} requires diagnostic whole branch ranking")
    active_k = _positive_integer(config.get("active_k"), "active_k")
    active_h = _positive_integer(config.get("active_h"), "active_h")
    execution_interval = config.get("diagnostic_execution_interval", 1)
    if type(execution_interval) is not int or execution_interval < 1:
        raise ValueError("diagnostic_execution_interval must be a positive integer")
    if deployment == "goal_gf":
        if (tier != "diagnostic" or reach_radius is None or execution_interval != 1
                or whole_branch is None):
            raise ValueError("goal_gf requires diagnostic reach-only whole-branch risk0 ext1")
        if whole_branch["risk_weight"] != 0 or whole_branch.get("score_mode", "l1_consistency") != "l1_consistency":
            raise ValueError("goal_gf requires matched normalized L1 endpoint-only risk0")
    if execution_interval > 1 and (tier != "diagnostic" or deployment != "full"
            or whole_branch is None or reach_radius is None or active_h < execution_interval):
        raise ValueError("sequence execution requires diagnostic full whole-branch reach-only H>=interval")

    stagnation_guard = None
    if "diagnostic_stagnation_guard" in config:
        if tier != "diagnostic" or deployment != "full" or whole_branch is None:
            raise ValueError("stagnation guard requires diagnostic full whole-branch planning")
        stagnation_guard = stagnation_module.validate_stagnation_options(config["diagnostic_stagnation_guard"])

    evaluation_seed = _nonnegative_integer(config.get("evaluation_seed"), "evaluation_seed")
    namespace = config.get("sampling_namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("sampling_namespace must be a nonempty string")
    device = config.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a nonempty string")
    return {
        "protocol": protocol,
        "configuration_id": identifier,
        "tier": tier,
        "diagnostic_stop_threshold": stop_override,
        "diagnostic_reach_radius": reach_radius,
        "diagnostic_step_trace": step_trace,
        "checkpoint_admission": admission,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "video": dict(video, episode_budget=video_budget, fps=fps),
        "deployment": deployment,
        "active_k": active_k,
        "active_h": active_h,
        "evaluation_seed": evaluation_seed,
        "namespace": namespace,
        "device": device,
        "whole_branch_ranking": whole_branch,
        "stagnation_guard": stagnation_guard,
        "execution_interval": execution_interval,
        "diagnostic_episode_indices": episode_indices,
    }


def _source_manifest() -> dict[str, dict[str, Any]]:
    from diagnostics.view_aligned_200_20260910 import core as view_core, sensor as view_sensor
    modules = {
        "runner": Path(__file__),
        "recurrent_factory": Path(factory_module.__file__),
        "planner": Path(planner_module.__file__),
        "prediction_role_rankers": Path(rankers_module.__file__),
        "whole_branch_ranking": Path(whole_branch_module.__file__),
        "stagnation_guard": Path(stagnation_module.__file__),
        "view_aligned_core": Path(view_core.__file__),
        "view_aligned_sensor": Path(view_sensor.__file__),
        "policy_adapter": Path(adapter_module.__file__),
        "raw_stop": Path(raw_stop_module.__file__),
        "statistics": Path(statistics_module.__file__),
        "checkpoint_loader": Path(checkpoint_module.__file__),
        "native_encoder": Path(encoder_module.__file__),
        "habitat_runner": Path(habitat_runner_module.__file__),
        "habitat_metrics": Path(metrics_module.__file__),
        "habitat_capability": Path(capability_module.__file__),
    }
    return {
        name: file_identity(path, name=f"closed-loop source {name}")
        for name, path in modules.items()
    }


def prepare_recurrent_habitat_runtime(
    config: Mapping[str, Any],
    *,
    checkpoint_loader=load_eval_checkpoint,
    encoder_builder=build_native_vjepa_grid_encoder,
    adapter_builder=build_recurrent_adapter,
    habitat_loader=load_habitat_environment,
) -> Mapping[str, Any]:
    """Build the one admitted exact-final runtime shared by evaluation entrypoints."""

    values = _validate_configuration(_mapping(config, "closed-loop config"))
    checkpoint = _mapping(config.get("checkpoint"), "checkpoint")
    visual = _mapping(config.get("visual"), "visual")
    stop_config = _mapping(config.get("stop"), "stop")
    habitat = _mapping(config.get("habitat"), "habitat")

    if "encoder_family" in visual:
        raise ValueError("unsupported explicit visual encoder family")
    stop_identity = _bound_file(stop_config, "receipt", "receipt_sha256", "RAW STOP receipt")
    stop_receipt = validate_raw_stop_calibration_receipt(_load_json(stop_identity, "RAW STOP receipt"))
    if stop_receipt.get("protocol_sha256") != values["protocol"]:
        raise ValueError("RAW STOP protocol differs from closed-loop config")

    sensor_identity = _bound_file(
        habitat, "sensor_config", "sensor_config_sha256", "Habitat sensor config"
    )
    evaluation_identity = _bound_file(
        habitat,
        "evaluation_config",
        "evaluation_config_sha256",
        "Habitat evaluation config",
    )
    ledger_identity = _bound_file(
        habitat, "episode_ledger", "episode_ledger_sha256", "episode ledger"
    )
    capability_identity = _bound_file(
        habitat,
        "capability_receipt",
        "capability_receipt_sha256",
        "Habitat capability receipt",
    )
    evaluation_contract = _load_evaluation_contract(evaluation_identity)
    _close_evaluation_contract(
        evaluation_contract, habitat, formal=values["tier"] == "formal"
    )
    capability = validate_formal_reset_replay_capability(
        _load_json(capability_identity, "Habitat capability receipt")
    )
    _close_capability_artifacts(
        capability,
        sensor=sensor_identity,
        evaluation=evaluation_identity,
        ledger=ledger_identity,
    )
    episodes, observed_ledger_sha = _load_episodes(
        Path(str(ledger_identity["path"]))
    )
    if observed_ledger_sha != ledger_identity["sha256"]:
        raise ValueError("episode ledger changed while loading")
    ledger_census = _validate_ordered_ledger(
        episodes, evaluation_contract, formal=values["tier"] == "formal"
    )
    ordered_keys = ledger_census["episode_keys"]

    configured_max_steps = _positive_integer(
        habitat.get("max_episode_steps"), "max_episode_steps"
    )
    success_distance = habitat.get("success_distance")
    if (
        isinstance(success_distance, bool)
        or not isinstance(success_distance, (int, float))
        or not math.isfinite(float(success_distance))
        or not 0.0 < float(success_distance)
    ):
        raise ValueError("success_distance must be positive and finite")
    warmup_steps = _nonnegative_integer(
        habitat.get("latency_warmup_steps", 0), "latency_warmup_steps"
    )

    episode_budget = len(ordered_keys)
    if values["tier"] == "diagnostic":
        diagnostic = _mapping(config.get("diagnostic"), "diagnostic")
        episode_budget = _positive_integer(diagnostic.get("episode_budget"), "diagnostic episode_budget")
        step_budget = _positive_integer(diagnostic.get("step_budget"), "diagnostic step_budget")
        if episode_budget > len(ordered_keys) or step_budget > configured_max_steps:
            raise ValueError("diagnostic budget exceeds configured budget")
    subset = values["diagnostic_episode_indices"]
    if subset is not None:
        if len(subset) > episode_budget or subset[-1] >= len(ordered_keys):
            raise ValueError("diagnostic subset exceeds ledger or episode budget")
        positions = episode_shard_indices(len(subset),
            shard_index=values["shard_index"], shard_count=values["shard_count"])
        selected_indices = [subset[position] for position in positions]
    else:
        selected_indices = episode_shard_indices(episode_budget,
            shard_index=values["shard_index"], shard_count=values["shard_count"])
    checkpoint_options = {"device": values["device"]}
    if values["checkpoint_admission"] == "completed_epoch":
        checkpoint_options["checkpoint_admission"] = "completed_epoch"

    loaded = checkpoint_loader(
        checkpoint.get("path"),
        checkpoint.get("sha256"),
        checkpoint.get("training_resolved_config_path"),
        checkpoint.get("training_resolved_config_sha256"),
        checkpoint.get("training_code_sha256"),
        **checkpoint_options,
    )
    completed_identity = None
    if values["checkpoint_admission"] == "completed_epoch":
        completed_identity = require_completed_epoch_checkpoint(loaded)
        final_identity = loaded.provenance.get("exact_final_identity")
    else:
        final_identity = require_exact_final_checkpoint(loaded)
    training = _mapping(loaded.training_config, "training sidecar")
    model_config = _mapping(training.get("model"), "training sidecar model")
    grid_side = _positive_integer(model_config.get("grid_side"), "model grid_side")
    latent_dim = _positive_integer(model_config.get("latent_dim"), "model latent_dim")
    training_seed = _nonnegative_integer(
        model_config.get("global_seed"), "training global_seed"
    )
    expected_shape = (grid_side * grid_side, latent_dim)
    encode_grid, encoder_provenance = encoder_builder(
        **visual,
        device=values["device"],
        expected_spatial_shape=expected_shape,
    )
    adapter = adapter_builder(
        loaded=loaded,
        encode_grid=encode_grid,
        encoder_provenance=encoder_provenance,
        stop_receipt=stop_receipt,
        **({"run_tier": values["tier"]} if values["deployment"] == "goal_gf" else {}),
        expected_protocol_sha256=values["protocol"],
        active_k=values["active_k"],
        active_h=values["active_h"],
        sampling_seed=values["evaluation_seed"],
        namespace=values["namespace"],
        deployment=values["deployment"],
        **({"diagnostic_stop_threshold": values["diagnostic_stop_threshold"]}
           if values["diagnostic_stop_threshold"] is not None else {}),
        **({"diagnostic_reach_radius": values["diagnostic_reach_radius"]}
           if values["diagnostic_reach_radius"] is not None else {}),
        **({"execution_interval": values["execution_interval"]}
           if values["execution_interval"] != 1 else {}),
        **({"stagnation_guard": values["stagnation_guard"]}
           if values["stagnation_guard"] is not None else {}),
        **({"whole_branch_ranking": values["whole_branch_ranking"]}
           if values["whole_branch_ranking"] is not None else {}),
    )
    environment = None
    try:
        environment = habitat_loader(
            config_path=str(sensor_identity["path"]),
            episodes_path=str(ledger_identity["path"]),
            scenes_dir=str(habitat.get("scene_root")),
            evaluation_contract=evaluation_contract,
            **({"episode_indices": selected_indices}
               if values["shard_count"] > 1 or subset is not None else {}),
        )
        seed_environment = getattr(environment, "seed", None)
        if not callable(seed_environment):
            raise TypeError("Habitat environment must expose seed(seed)")
        seed_environment(values["evaluation_seed"])
    except Exception:
        close_adapter = getattr(adapter, "close", None)
        try:
            if callable(close_adapter):
                close_adapter()
        finally:
            if environment is not None:
                close_environment = getattr(environment, "close", None)
                if callable(close_environment):
                    close_environment()
        raise

    runtime_values = dict(values)
    runtime_values.update(
        evaluation_contract=evaluation_contract,
        ledger_census=ledger_census,
        configured_max_steps=configured_max_steps,
        success_distance=float(success_distance),
        warmup_steps=warmup_steps,
        training=training,
        training_seed=training_seed,
        completed_epoch_identity=completed_identity,
        selected_indices=selected_indices,
        effective_stop_threshold=(float(stop_receipt["threshold"])
            if values["diagnostic_stop_threshold"] is None else float(values["diagnostic_stop_threshold"])),
    )
    return {
        "adapter": adapter,
        "environment": environment,
        "planner": getattr(adapter, "_planner", None),
        "capability_receipt": capability,
        "frozen_seed": values["evaluation_seed"],
        "artifact_identities": {
            "stop_receipt": stop_identity,
            "sensor_config": sensor_identity,
            "evaluation_config": evaluation_identity,
            "episode_ledger": ledger_identity,
            "habitat_capability": capability_identity,
        },
        "ordered_keys": ordered_keys,
        "loaded": loaded,
        "final_checkpoint_identity": final_identity,
        "encoder_provenance": encoder_provenance,
        "values": runtime_values,
    }


def run_closed_loop_configuration(
    *, config: Mapping[str, Any], output_path: str | Path,
    checkpoint_loader=load_eval_checkpoint,
    encoder_builder=build_native_vjepa_grid_encoder,
    adapter_builder=build_recurrent_adapter,
    habitat_loader=load_habitat_environment,
    episode_runner=run_imagegoal_episode,
) -> Mapping[str, Any]:
    """Run one exact-final configuration and atomically publish its result."""
    destination = Path(output_path).expanduser().resolve()
    admission_destination = destination.with_name(destination.name + ".admission.json")
    if destination.exists() or admission_destination.exists():
        raise FileExistsError(
            f"closed-loop result/admission destination already exists: {destination}"
        )
    runtime = prepare_recurrent_habitat_runtime(
        config,
        checkpoint_loader=checkpoint_loader,
        encoder_builder=encoder_builder,
        adapter_builder=adapter_builder,
        habitat_loader=habitat_loader,
    )
    values = runtime["values"]
    artifacts = runtime["artifact_identities"]
    stop_identity = artifacts["stop_receipt"]
    sensor_identity = artifacts["sensor_config"]
    evaluation_identity = artifacts["evaluation_config"]
    ledger_identity = artifacts["episode_ledger"]
    capability_identity = artifacts["habitat_capability"]
    evaluation_contract = values["evaluation_contract"]
    ledger_census = values["ledger_census"]
    ordered_keys = runtime["ordered_keys"]
    configured_max_steps = values["configured_max_steps"]
    success_distance = values["success_distance"]
    warmup_steps = values["warmup_steps"]
    loaded = runtime["loaded"]
    final_identity = runtime["final_checkpoint_identity"]
    training_seed = values["training_seed"]
    encoder_provenance = runtime["encoder_provenance"]
    adapter = runtime["adapter"]
    environment = runtime["environment"]
    selected_keys = ordered_keys
    max_steps = configured_max_steps
    if values["tier"] == "diagnostic":
        diagnostic = _mapping(config.get("diagnostic"), "diagnostic")
        episode_budget = _positive_integer(
            diagnostic.get("episode_budget"), "diagnostic episode_budget"
        )
        step_budget = _positive_integer(
            diagnostic.get("step_budget"), "diagnostic step_budget"
        )
        if episode_budget > len(ordered_keys) or step_budget > configured_max_steps:
            raise ValueError("diagnostic budget exceeds the configured formal budget")
        selected_keys = ordered_keys[:episode_budget]
        max_steps = step_budget

    selected_indices = values["selected_indices"]
    selected_keys = [ordered_keys[index] for index in selected_indices]

    episode_rows: list[dict[str, Any]] = []
    admission_identity = None
    try:
        analysis = config.get("analysis")
        if values["tier"] == "formal":
            analysis = _mapping(analysis, "formal analysis plan")
            if not analysis:
                raise ValueError("formal analysis plan must be nonempty")
        elif analysis is None:
            analysis = {
                "status": "NOT_FORMAL_ANALYSIS",
                "interpretation": "bounded runtime diagnostic only",
            }
        admission = {
            "schema": "J2J_RECURRENT_RAW_CLOSED_LOOP_ADMISSION_V1",
            "status": "ADMITTED_BEFORE_EPISODE_EXECUTION",
            "formal_eligible": values["tier"] == "formal",
            "protocol_sha256": values["protocol"],
            "configuration": to_plain_json(config),
            "configuration_sha256": canonical_mapping_sha256(config),
            "analysis": to_plain_json(analysis),
            "run_decision": {
                "configuration_id": values["configuration_id"],
                "deployment": values["deployment"],
                "active_k": values["active_k"],
                "active_h": values["active_h"],
                "evaluation_seed": values["evaluation_seed"],
                "sampling_namespace": values["namespace"],
                "tier": values["tier"],
                "episode_budget": len(selected_keys),
                "step_budget": max_steps,
                "artifact_identities": {
                    "checkpoint_provenance": to_plain_json(loaded.provenance),
                    "final_checkpoint_identity": to_plain_json(final_identity),
                    "completed_epoch_identity": to_plain_json(values["completed_epoch_identity"]),
                    "encoder_provenance": to_plain_json(encoder_provenance),
                    "stop_receipt": stop_identity,
                    "habitat_capability": capability_identity,
                    "sensor_config": sensor_identity,
                    "evaluation_config": evaluation_identity,
                    "episode_ledger": ledger_identity,
                },
            },
            "source_manifest": _source_manifest(),
        }
        admission_identity = create_once_json(admission_destination, admission)
        for local_index, (ledger_index, (scene_id, episode_id)) in enumerate(zip(selected_indices, selected_keys)):
            recorder = None
            video = values["video"]
            if local_index < video["episode_budget"]:
                recorder = FirstPersonVideoRecorder(
                    Path(video["directory"]) / f"episode_{ledger_index:06d}.mp4",
                    fps=video["fps"], episode_key=[scene_id, episode_id],
                    identities={"checkpoint_sha256": loaded.provenance["checkpoint_sha256"],
                                "episode_ledger_sha256": ledger_identity["sha256"],
                                "ledger_index": ledger_index, "shard_index": values["shard_index"]})
            trace_file = None
            observer_options = {}
            if values["diagnostic_step_trace"] or values["whole_branch_ranking"] is not None:
                trace_path = (destination.parent / (destination.stem + "_episodes")
                              / f"episode_{ledger_index:06d}.steps.jsonl")
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_file = trace_path.open("x")
                def record_decision(policy, evaluator):
                    # Existing observer runs after the action is fixed. Distances
                    # are pre-action evaluator evidence, never policy inputs.
                    evidence = {"ledger_index": ledger_index, "step": evaluator["step"],
                        "action": policy["action"],
                        "canonical_stop_distance": float(policy["diagnostics"]["canonical_stop_distance"]),
                        "distance_to_goal": float(evaluator["distance_to_goal"]),
                        "effective_threshold": values["effective_stop_threshold"]}
                    probe = getattr(adapter, "diagnostic_goal_stop_scores", None)
                    if values["diagnostic_step_trace"] and callable(probe):
                        evidence["goal_stop_scores"] = probe()
                    if values["whole_branch_ranking"] is not None:
                        if policy["action"] == "STOP" and policy["diagnostics"].get("stop_category") == "executed":
                            evidence.update(whole_branch_ranking=None,
                                            whole_branch_ranking_status="not_run_stop")
                        else:
                            evidence["whole_branch_ranking"] = to_plain_json(
                                policy["diagnostics"]["whole_branch_ranking"])
                    if "diagnostic_execution_interval" in config:
                        diagnostics = policy["diagnostics"]
                        defaults = dict(execution_interval=1, real_step_index=evaluator["step"],
                            planning_step_index=evaluator["step"], plan_origin_real_step=evaluator["step"],
                            execution_offset=0, replanned=True)
                        evidence.update({key: diagnostics.get(key, value) for key, value in defaults.items()})
                        if diagnostics.get("whole_branch_ranking_status") == "reused_plan":
                            evidence["whole_branch_ranking_status"] = "reused_plan"
                    trace_file.write(json.dumps(evidence, allow_nan=False) + "\n")
                    trace_file.flush()
                observer_options["decision_observer"] = record_decision
            try:
                observed = _mapping(
                    episode_runner(
                        environment,
                        adapter,
                        max_steps=max_steps,
                        success_distance=float(success_distance),
                        warmup_steps=warmup_steps,
                        expected_scene_id=scene_id,
                        expected_episode_id=episode_id,
                        **({"frame_observer": recorder} if recorder is not None else {}),
                        **observer_options,
                        **({"diagnostic_reach_radius": values["diagnostic_reach_radius"]}
                           if values["diagnostic_reach_radius"] is not None else {}),
                    ),
                    "Habitat episode result",
                )
            finally:
                if trace_file is not None:
                    trace_file.close()
            if observed.get("scene_id") != scene_id or str(observed.get("episode_id")) != episode_id:
                raise RuntimeError("Habitat episode result differs from ordered ledger identity")
            row = dict(to_plain_json(observed))
            if recorder is not None:
                row["first_person_video"] = recorder.close()
            row.update(
                configuration_id=values["configuration_id"],
                deployment=values["deployment"],
                active_k=values["active_k"],
                active_h=values["active_h"],
                evaluation_seed=values["evaluation_seed"],
                training_seed=training_seed,
                checkpoint_sha256=loaded.provenance["checkpoint_sha256"],
                status="FORMAL" if values["tier"] == "formal" else "DIAGNOSTIC",
                building=scene_id,
                episode_id=episode_id,
                composite_episode_key=[scene_id, episode_id],
                ledger_index=ledger_index,
            )
            episode_rows.append(row)
            create_once_json(destination.parent / (destination.stem + "_episodes")
                             / f"episode_{ledger_index:06d}.json", row)
            print(json.dumps({"event": "episode_complete", "shard_index": values["shard_index"],
                "ledger_index": ledger_index, "completed": len(episode_rows), "assigned": len(selected_keys),
                "success": row.get("success"), "spl": row.get("spl"),
                **({"arrival_success": row.get("arrival_success"),
                    "reach_spl": row.get("reach_spl"),
                    "first_arrival_step": row.get("first_arrival_step")}
                   if values["diagnostic_reach_radius"] is not None else {})}), flush=True)
    finally:
        close_adapter = getattr(adapter, "close", None)
        try:
            if callable(close_adapter):
                close_adapter()
        finally:
            if environment is not None:
                close_environment = getattr(environment, "close", None)
                if callable(close_environment):
                    close_environment()

    formal = values["tier"] == "formal"
    result = {
        "schema": "J2J_RECURRENT_RAW_CLOSED_LOOP_RESULT_V1",
        "status": "COMPLETE" if formal else "DIAGNOSTIC_COMPLETE",
        "formal_eligible": formal,
        "evidence_tier": "frozen_ledger_closed_loop" if formal else "bounded_runtime_diagnostic",
        "protocol_sha256": values["protocol"],
        "configuration": to_plain_json(config),
        "configuration_sha256": canonical_mapping_sha256(config),
        "checkpoint_provenance": to_plain_json(loaded.provenance),
        "final_checkpoint_identity": to_plain_json(final_identity),
        "completed_epoch_identity": to_plain_json(values["completed_epoch_identity"]),
        "shard_index": values["shard_index"],
        "shard_count": values["shard_count"],
        "selected_indices": selected_indices,
        "variant_identity": to_plain_json(loaded.variant_identity),
        "encoder_provenance": to_plain_json(encoder_provenance),
        "stop_receipt_identity": stop_identity,
        "capability_receipt_identity": capability_identity,
        "sensor_config_identity": sensor_identity,
        "evaluation_config_identity": evaluation_identity,
        "episode_ledger_identity": ledger_identity,
        "episode_ledger_census": to_plain_json(ledger_census),
        "ordered_episode_key_sha256": canonical_mapping_sha256(
            {"episode_keys": ordered_keys}
        ),
        "runner_source_identity": file_identity(Path(__file__), name="closed-loop runner"),
        "admission_identity": to_plain_json(admission_identity),
        "episode_budget": len(selected_keys),
        "step_budget": max_steps,
        "episodes": episode_rows,
    }
    if "diagnostic_execution_interval" in config:
        result["diagnostic_execution_interval"] = values["execution_interval"]
    if values["diagnostic_reach_radius"] is not None:
        result.update(evaluation_success_criterion="reach_only",
                      diagnostic_reach_radius=values["diagnostic_reach_radius"])
    create_once_json(destination, result)
    return result


__all__ = ["build_configuration_cases", "run_closed_loop_configuration"]
