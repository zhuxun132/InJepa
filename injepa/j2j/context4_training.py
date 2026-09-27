"""Fail-closed runtime primitives for context-four spatial joint training."""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

import copy
from contextlib import nullcontext
from dataclasses import dataclass, replace, asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import time
import warnings
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from j2j.context4_data import (
    JointBatch,
    PreparedJointTrajectory,
    RankOriginPlan,
    TrajectoryDescriptor,
    build_rank_origin_plan,
    iter_materialized_rank_slices,
    materialize_joint_batch,
    prepare_joint_trajectory,
)
from j2j.context4_metrics import ModeDiagnosticAccumulator, TrainingMetricsStore
from j2j.context4_objective import (
    JointNumerators,
    branch_denominators,
    graph_connected_zero,
    joint_numerators,
    weighted_joint_loss,
    weighted_joint_loss_value,
)
from j2j.context4_variants import (
    FULL_IDENTITY,
    FULL_LOSS_SPEC_SHA256,
    FULL_VARIANT_ID,
    RAW_IDENTITY,
    STOCHASTIC_IDENTITY,
    STABLE_STOCHASTIC_IDENTITY,
    RECURRENT_STOCHASTIC_IDENTITY,
    RECURRENT_STOCHASTIC_IDENTITIES,
    RECURRENT_STOCHASTIC_PROTOCOL_SHA256,
    STABLE_STOCHASTIC_IDENTITIES,
    STABLE_STOCHASTIC_PROTOCOL_SHA256,
    STOCHASTIC_IDENTITIES,
    RAW_IDENTITIES,
    STOCHASTIC_PROTOCOL_SHA256,
    RAW_PROTOCOL_SHA256,
    RAW_VARIANT_ID,
    VariantContract,
    resolve_variant_contract,
    variant_identity_dict,
)
from j2j.data.goals import hashed_goal_index
from j2j.spatial_intact import Context4SpatialJointModel


EXPERIMENT_ID = FULL_VARIANT_ID
DESIGN_SHA256 = "06e614c0bcdef6ef7fd99d7961670bf726faa62850ab705262284e3b460d1474"
FORMAL_PROTOCOL_SHA256 = FULL_LOSS_SPEC_SHA256
SCRATCH_OPTIMIZATION_PROTOCOL_SHA256 = (
    "0fdf0a5b63c983a10baf1e2f27cd46e4e60ad39f242e12635ab91bcb3377130d"
)
SCRATCH_TRAINING_CONTRACT_ID = (
    "context4_no_selector_spatial_joint_scratch_qagf_iso_agfclip2_v1"
)
RAW_TRAINING_CONTRACT_ID = "context4_raw_spatial_scratch_qagf_v1"
STOCHASTIC_TRAINING_CONTRACT_ID = "context4_stochastic_divided_raw_scratch_qagf_v1"
STABLE_STOCHASTIC_TRAINING_CONTRACT_ID = "context4_stochastic_divided_raw_scratch_qagf_v2"
RECURRENT_STOCHASTIC_TRAINING_CONTRACT_ID = "context4_stochastic_recurrent_raw_scratch_qagf_v1"
_RAW_TRAINING_IDS = frozenset({RECURRENT_STOCHASTIC_TRAINING_CONTRACT_ID, RAW_TRAINING_CONTRACT_ID, STOCHASTIC_TRAINING_CONTRACT_ID, STABLE_STOCHASTIC_TRAINING_CONTRACT_ID})
SCRATCH_INITIALIZATION_MODE = "scratch_trainable"
SCRATCH_INITIALIZATION_SCHEMA = "j2j_context4_trainable_init_v1"
SCRATCH_OPTIMIZER_SCHEME = "q_agf_isolated_v1"
RAE_NWM_COMMIT = "0219ce41c44d515f86719dd763c1efe7c7f72519"
INTACT_COMMIT = "235b6a3a92db4d0f1b3a40597ab1f407db4fd15b"
STREAMVLN_REVISION = "dc61ee9b4e90aa7ba63c1163b2134df5610dccb9"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BRANCHES = ("q", "f", "g_local", "g_goal", "qg")
_ADAMW_BEHAVIOR_DEFAULTS = (
    ("amsgrad", False),
    ("maximize", False),
    ("foreach", None),
    ("capturable", False),
    ("differentiable", False),
    ("fused", None),
)
_ADAMW_BASE_PARAM_GROUP_KEYS = frozenset(
    {
        "params",
        "lr",
        "group_name",
        "betas",
        "eps",
        "weight_decay",
        *(name for name, _value in _ADAMW_BEHAVIOR_DEFAULTS),
    }
)
_ADAMW_PARAM_GROUP_KEYS = _ADAMW_BASE_PARAM_GROUP_KEYS | {"initial_lr"}


def _contract_or_full(
    variant_contract: VariantContract | None,
) -> VariantContract:
    if variant_contract is None:
        return resolve_variant_contract(FULL_IDENTITY)
    if not isinstance(variant_contract, VariantContract):
        raise TypeError("variant_contract must be a resolved VariantContract")
    # Re-resolve the immutable values so hand-built or relabelled contracts
    # cannot bypass the exact registry.
    return resolve_variant_contract(variant_contract.identity)


def _variant_contract_from_config(root: Mapping[str, Any]) -> VariantContract:
    experiment_id = root.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("config variant experiment identity is missing")
    if experiment_id == EXPERIMENT_ID:
        if "variant_identity" in root:
            raise ValueError("Full variant identity must remain exact-derived")
        return resolve_variant_contract(FULL_IDENTITY)
    native = root.get("variant_identity")
    contract = resolve_variant_contract(native)
    if contract.identity.variant_id != experiment_id:
        raise ValueError("config experiment_id and native variant identity disagree")
    return contract


def _prepare_for_variant(
    trajectory_item: object,
    variant_contract: VariantContract,
):
    if (
        type(trajectory_item) is PreparedJointTrajectory
        and trajectory_item.transform_spec_sha
        == variant_contract.identity.transform_spec_sha
    ):
        # Production DataLoader/iterator owns first use.  Downstream slices
        # reuse the exact prepared object without calling the transform again.
        return trajectory_item
    prepared = prepare_joint_trajectory(trajectory_item)
    return variant_contract.prepared_trajectory_transform(prepared)


def _apply_context_variant(
    batch: JointBatch,
    variant_contract: VariantContract,
) -> JointBatch:
    if variant_contract.identity.variant_id != "context1":
        return batch
    transformed = variant_contract.context_transform(
        {
            "context_grid": batch.context_grid,
            "context_incoming_raw4": batch.context_incoming_raw4,
            "context_outgoing_raw4": batch.context_outgoing_raw4,
            "context_age": batch.context_age,
            "context_type": batch.context_type,
            "context_valid": batch.context_valid,
        }
    )
    return replace(batch, **dict(transformed))


_STOCHASTIC_STAT_FIELDS = (
    "occurrences", "reconstruction_mean_sum", "kl_total_sum", "kl_mean_sum",
    "reconstruction_sum", "target_scalars",
    "exact_nelbo_per_scalar_sum", "prior_mean_square_sum", "posterior_mean_square_sum",
    "prior_variance_sum", "posterior_variance_sum", "latent_dimensions",
)


def _accumulate_stochastic_statistics(totals: dict[str, float], values) -> None:
    if values is None:
        return  # An owner-empty/terminal-only micro contains no Q occurrence.
    if not isinstance(values, Mapping) or set(values) != set(_STOCHASTIC_STAT_FIELDS):
        raise ValueError("stochastic probability statistics have an invalid schema")
    # One device transfer for the complete small vector, not one synchronization
    # per statistic; the full future tape is never retained for telemetry.
    tensors = []
    for name in _STOCHASTIC_STAT_FIELDS:
        value = values[name]
        if not isinstance(value, Tensor) or value.ndim != 0 or value.requires_grad:
            raise ValueError("stochastic statistics must be detached scalar sums")
        tensors.append(value)
    numbers = torch.stack(tensors).double().cpu().tolist()
    for name, number in zip(_STOCHASTIC_STAT_FIELDS, numbers, strict=True):
        if not math.isfinite(number):
            raise FloatingPointError("stochastic probability statistic is nonfinite")
        totals[name] += number


def _global_stochastic_statistics(totals, *, world_size: int, device: torch.device):
    if totals is None:
        return None
    if world_size == 1:
        return dict(totals)
    values = torch.tensor([totals[name] for name in _STOCHASTIC_STAT_FIELDS],
                          dtype=torch.float64, device=device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    return dict(zip(_STOCHASTIC_STAT_FIELDS, values.cpu().tolist(), strict=True))


def _stochastic_probability_record(sums, *, kl_beta: float, sampling_namespace: str):
    if sums is None:
        return None
    occurrences = sums["occurrences"]
    dimensions = sums["latent_dimensions"]
    scalars = sums["target_scalars"]
    def divide(value, count):
        return value / count if count else 0.0
    reconstruction = divide(sums["reconstruction_mean_sum"], occurrences)
    kl_mean = divide(sums["kl_mean_sum"], occurrences)
    return {"raw_sums": dict(sums), "kl_beta": kl_beta,
            "sampling_namespace": sampling_namespace,
            "means": {
                "reconstruction_per_occurrence": reconstruction,
                "kl_total_per_occurrence": divide(sums["kl_total_sum"], occurrences),
                "kl_per_latent_dimension": kl_mean,
                "weighted_kl": kl_beta * kl_mean,
                "optimized_q": reconstruction + kl_beta * kl_mean,
                "exact_nelbo_per_scalar_occurrence_mean": divide(sums["exact_nelbo_per_scalar_sum"], occurrences),
                "summed_nll_per_target_scalar": divide(sums["reconstruction_sum"], scalars),
                "prior_mean_square": divide(sums["prior_mean_square_sum"], dimensions),
                "posterior_mean_square": divide(sums["posterior_mean_square_sum"], dimensions),
                "prior_variance": divide(sums["prior_variance_sum"], dimensions),
                "posterior_variance": divide(sums["posterior_variance_sum"], dimensions),
            }}


class Context4ObjectiveModule(nn.Module):
    """DDP-visible wrapper around the one owned Q/A/G/F tree.

    DDP must observe a forward on every rank and physical micro-slot.  An
    owner-empty rank therefore returns branch-specific graph-connected zeros
    instead of bypassing ``DistributedDataParallel.forward``.
    """

    def __init__(self, joint_model: Context4SpatialJointModel, *, precision: str,
                 qg_objective: str = "marginal", loss_weights: Mapping[str, float] | None = None) -> None:
        super().__init__()
        if not isinstance(joint_model, Context4SpatialJointModel):
            raise TypeError("joint_model must be Context4SpatialJointModel")
        if precision not in {"bf16-mixed", "fp32"}:
            raise ValueError("precision must be bf16-mixed or fp32")
        self.joint_model = joint_model
        self.precision = precision
        if qg_objective not in {"marginal", "posterior_weighted", "posterior_sample"}:
            raise ValueError("unknown QG objective")
        if bool(getattr(joint_model.proposal, "is_stochastic", False)) != (qg_objective == "posterior_sample"):
            raise ValueError("stochastic Q requires posterior_sample QG")
        self.loss_weights = loss_weights
        self.qg_objective = qg_objective
        self.collect_diagnostics = False

    def _empty(self) -> JointNumerators:
        model = self.joint_model
        q = graph_connected_zero(model.proposal.parameters())
        f = graph_connected_zero(
            parameter
            for module in (model.action_encoder, model.forward_core, model.pred_proj)
            for parameter in module.parameters()
        )
        g_local = graph_connected_zero(
            parameter
            for module in (model.action_encoder, model.actor)
            for parameter in module.parameters()
        )
        # Keep each mathematical owner visible to DDP without inventing a
        # synthetic data row or changing any branch denominator.
        return JointNumerators(
            numerators={
                "q": q,
                "f": f,
                "g_local": g_local,
                "g_goal": g_local * 1.0,
                "qg": g_local * 1.0,
            },
            denominators={name: 0 for name in _BRANCHES},
            counts={
                "transitions": 0,
                "q_occurrences": 0,
                "terminals": 0,
                "qg_mode_rows": 0,
            },
        )

    def forward(self, batch: JointBatch | None) -> JointNumerators:
        if batch is None:
            return self._empty()
        if type(batch) is not JointBatch:
            raise TypeError("context4 objective accepts only JointBatch or None")
        parameter = next(self.joint_model.parameters())
        enabled = self.precision == "bf16-mixed" and parameter.device.type == "cuda"
        with torch.autocast(
            device_type=parameter.device.type,
            dtype=torch.bfloat16,
            enabled=enabled,
        ):
            return joint_numerators(
                self.joint_model,
                batch,
                collect_diagnostics=self.collect_diagnostics,
                qg_objective=self.qg_objective,
                loss_weights=self.loss_weights,
            )


def build_q_goal_views(
    descriptors: Sequence[TrajectoryDescriptor],
    *,
    horizon: int,
) -> tuple[dict[int, tuple[tuple[int, int], ...]], Mapping[str, int]]:
    """Build terminal plus distinct eligible hashed goal views from metadata."""

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be positive")
    result: dict[int, tuple[tuple[int, int], ...]] = {}
    q_occurrences = 0
    active_blocks = 0
    for trajectory_index, descriptor in enumerate(tuple(descriptors)):
        if type(descriptor) is not TrajectoryDescriptor:
            raise TypeError("goal planning requires TrajectoryDescriptor values")
        terminal = descriptor.origin_count
        pairs: list[tuple[int, int]] = []
        for origin in range(terminal):
            goals = [terminal]
            if terminal - origin >= horizon:
                hashed = hashed_goal_index(
                    descriptor.trajectory_key,
                    t=origin,
                    terminal=terminal,
                    horizon=horizon,
                )
                goals.append(hashed)
            # Preserve the existing formal Q row ordering and dedup rule.
            for goal in sorted(set(goals)):
                pairs.append((origin, goal))
                q_occurrences += 1
                active_blocks += min(horizon, goal - origin)
        result[trajectory_index] = tuple(pairs)
    return result, {
        "q_occurrences": q_occurrences,
        "active_q_future_blocks": active_blocks,
    }


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_identity(plan: RankOriginPlan) -> str:
    return _canonical_json_sha256(
        {
            "world_size": plan.world_size,
            "effective_batch": plan.effective_batch,
            "descriptors": [
                [value.dataset_index, value.trajectory_key.hex(), value.origin_count]
                for value in plan.descriptors
            ],
            "owners": list(plan.owners),
            "updates": [
                [
                    [[row.trajectory_index, row.start, row.stop] for row in rank_rows]
                    for rank_rows in update
                ]
                for update in plan.updates
            ],
            "terminals": [
                [value.trajectory_index, value.rank, value.update_index, value.origin, value.side_row]
                for value in plan.terminals
            ],
        }
    )


def _next_batch_identity(
    plan: RankOriginPlan,
    q_goal_views_by_trajectory: Mapping[int, Sequence[tuple[int, int]]],
) -> str:
    first = plan.updates[0]
    touched = sorted({row.trajectory_index for rank_rows in first for row in rank_rows})
    return _canonical_json_sha256(
        {
            "first_update": [
                [[row.trajectory_index, row.start, row.stop] for row in rank_rows]
                for rank_rows in first
            ],
            "q_views": [
                [index, [[origin, goal] for origin, goal in q_goal_views_by_trajectory[index]]]
                for index in touched
            ],
        }
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"config {name} must be a mapping")
    return value


def _exact_number(value: object, expected: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"config {name} must be numeric")
    if not math.isfinite(float(value)) or float(value) != float(expected):
        raise ValueError(f"config {name} does not match the exact optimizer schedule")


def _positive_int(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"config {name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"config {name} must be {'non-negative' if allow_zero else 'positive'}")
    return value


def _require_sha(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value.lower()) is None:
        raise ValueError(f"config identity {name} must be a lowercase SHA-256")
    return value.lower()


def validate_context4_config(
    config: Mapping[str, Any],
    *,
    actual_world_size: int,
) -> dict[str, Any]:
    """Validate the scientific/numerical identity before model or cache use."""

    root = _mapping(config, "root")
    # Exact variant admission is deliberately first: invalid controls cannot
    # reach paths, cache payloads, RNG, or model construction.
    variant_contract = _variant_contract_from_config(root)
    from j2j.context4_variants import CROCO224_IDENTITY, DINOV3_256_IDENTITY
    if variant_contract.identity == CROCO224_IDENTITY:
        from j2j.encoding.croco224 import validate_training_config
        validate_training_config(root)
    elif variant_contract.identity == DINOV3_256_IDENTITY:
        from j2j.encoding.dinov3_256 import validate_training_config
        validate_training_config(root)
    elif root.get("encoder_contract") is not None or "encoder_family" in root.get("data", {}).get("expected_identities", {}):
        raise ValueError("encoder replacement requires its own encoder variant identity")
    stochastic = variant_contract.identity in STOCHASTIC_IDENTITIES
    raw = variant_contract.identity in RAW_IDENTITIES
    design_sha = variant_contract.identity.transform_spec_sha if raw else DESIGN_SHA256
    formal_sha = variant_contract.identity.transform_spec_sha if raw else FORMAL_PROTOCOL_SHA256
    identity = _mapping(root.get("identity"), "identity")
    if identity.get("design_sha256") != design_sha:
        raise ValueError("config design identity mismatch")
    formal_protocol = identity.get("formal_protocol_sha256")
    if formal_protocol is not None and formal_protocol != formal_sha:
        raise ValueError("config formal protocol identity mismatch")
    if identity.get("rae_nwm_commit") != RAE_NWM_COMMIT:
        raise ValueError("config RAE-NWM commit identity mismatch")
    if identity.get("intact_commit") != INTACT_COMMIT:
        raise ValueError("config INTACT commit identity mismatch")
    if identity.get("streamvln_revision") != STREAMVLN_REVISION:
        raise ValueError("config StreamVLN source commit identity mismatch")
    for name in ("source_manifest_sha256", "cache_manifest_sha256"):
        _require_sha(identity.get(name), name)

    initialization_raw = root.get("initialization")
    scratch = initialization_raw is not None
    if scratch:
        initialization = _mapping(initialization_raw, "initialization")
        if initialization != {
            "mode": SCRATCH_INITIALIZATION_MODE,
            "schema": SCRATCH_INITIALIZATION_SCHEMA,
        }:
            raise ValueError("scratch initialization identity is invalid")
        training_contract = _mapping(
            root.get("training_contract"), "training_contract"
        )
        if training_contract != {
            "id": RECURRENT_STOCHASTIC_TRAINING_CONTRACT_ID if variant_contract.identity in RECURRENT_STOCHASTIC_IDENTITIES else STABLE_STOCHASTIC_TRAINING_CONTRACT_ID if variant_contract.identity == STABLE_STOCHASTIC_IDENTITY else STOCHASTIC_TRAINING_CONTRACT_ID if stochastic else RAW_TRAINING_CONTRACT_ID if raw else SCRATCH_TRAINING_CONTRACT_ID,
            "optimization_protocol_sha256": design_sha if raw else SCRATCH_OPTIMIZATION_PROTOCOL_SHA256,
        }:
            raise ValueError("scratch training contract identity is invalid")
        forbidden_identity = {"q_warm_sha256", "agf_warm_sha256"}.intersection(
            identity
        )
        if forbidden_identity:
            raise ValueError(
                "scratch config must not contain warm checkpoint identity fields"
            )
        if "warm_start" in root:
            raise ValueError("scratch config must not contain warm_start fields")
    else:
        if "training_contract" in root:
            raise ValueError(
                "training_contract requires the scratch initialization contract"
            )
        for name in ("q_warm_sha256", "agf_warm_sha256"):
            _require_sha(identity.get(name), name)

    model = _mapping(root.get("model"), "model")
    if stochastic:
        expected_architecture = "stochastic_recurrent" if variant_contract.identity in RECURRENT_STOCHASTIC_IDENTITIES else "stochastic_divided"
        if model.get("proposal_architecture") != expected_architecture:
            raise ValueError("stochastic proposal architecture differs from its protocol")
        for name, expected in {"trajectory_latent_dim": 128, "kl_beta": 0.05,
                               "sigma_epsilon": 1e-4, "posterior_residual_init_std": 1e-3}.items():
            _exact_number(model.get(name), expected, f"stochastic model {name}")
    elif model.get("proposal_architecture", "fixed_mixture") != "fixed_mixture":
        raise ValueError("legacy protocol requires its fixed-mixture proposal")
    if variant_contract.identity in STABLE_STOCHASTIC_IDENTITIES:
        from dataclasses import asdict
        from j2j.proposal.stability import StabilityConfig
        controls = model.get("stability")
        parsed_controls = StabilityConfig.from_mapping(controls)
        if dict(controls) != asdict(StabilityConfig()) or asdict(parsed_controls) != dict(controls):
            raise ValueError("v2 requires its complete frozen stability controls")
    elif model.get("stability") is not None:
        raise ValueError("legacy protocols cannot acquire v2 stability controls")
    if "training_qkv_normalization_backend" in model:
        from j2j.proposal.stability import validate_training_qkv_normalization_backend
        validate_training_qkv_normalization_backend(
            model["training_qkv_normalization_backend"],
            stable=variant_contract.identity in STABLE_STOCHASTIC_IDENTITIES)
    if model.get("context_size") != 4:
        raise ValueError("config context_size must equal four")
    if model.get("selector_enabled") is not False:
        raise ValueError("config selector must be disabled")
    _positive_int(model.get("modes"), "model modes")
    _positive_int(model.get("horizon"), "model horizon")
    if variant_contract.identity in RECURRENT_STOCHASTIC_IDENTITIES:
        for name, expected in (("training_q_horizon", 1), ("goal_sampling_horizon", 4)):
            if _positive_int(model.get(name), f"model {name}") != expected:
                raise ValueError(f"recurrent protocol requires {name}={expected}")
    elif {"training_q_horizon", "goal_sampling_horizon"}.intersection(model):
        raise ValueError("separate Q horizon controls require the recurrent protocol")
    if model.get("latent_dim") != 768:
        raise ValueError("config latent_dim must equal the admitted V-JEPA width 768")
    if raw:
        if not scratch or model.get("representation") != "raw":
            raise ValueError("raw spatial run requires scratch initialization and raw representation")
        side = _positive_int(model.get("grid_side"), "model grid_side")
        if side != 24:
            warnings.warn(f"Non-default raw grid side {side}; cache shape must match.", UserWarning)
        if model.get("attention_backend") not in {"math", "auto"}:
            raise ValueError("raw attention backend must be math or auto")
        if type(model.get("activation_checkpointing")) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        if model.get("memory_norm") is not True or model.get("prediction_norm") != "layernorm":
            raise ValueError("raw protocol requires memory and prediction LayerNorm")
        raw_data = _mapping(root.get("data"), "data")
        if raw_data.get("require_stage") != "RAW32":
            raise ValueError("raw protocol requires RAW32 cache")
        raw_ids = _mapping(raw_data.get("expected_identities"), "data expected identities")
        if raw_ids.get("whitening_sha256") is not None:
            raise ValueError("raw protocol forbids whitening")
    elif model.get("grid_side") != 6:
        raise ValueError("config grid_side must equal the admitted cache grid 6")
    global_seed = _positive_int(
        model.get("global_seed", 3072), "model global_seed", allow_zero=True
    )
    if scratch and global_seed != 3072:
        raise ValueError("scratch model global_seed must equal 3072")
    exact_model = {
        "hidden_dim": 768,
        "heads": 16,
        "ffn_dim": 2048,
        "proposal_depth": 6,
        "actor_depth": 3,
        "forward_depth": 6,
    }
    for name, expected in exact_model.items():
        value = model.get(name, expected)
        if value != expected:
            raise ValueError(f"config model {name} does not match the reviewed architecture")
    if "retain_activation_blocks" in model:
        from j2j.proposal.blocks import validate_retained_activation_blocks
        if not isinstance(model["retain_activation_blocks"], list):
            raise TypeError("config retain_activation_blocks must be a JSON list")
        validate_retained_activation_blocks(
            model["retain_activation_blocks"],
            activation_checkpointing=model.get("activation_checkpointing", False),
            depths={"proposal.future.blocks": model.get("proposal_depth", exact_model["proposal_depth"]),
                    "forward_core.blocks": model.get("forward_depth", exact_model["forward_depth"]),
                    "actor.blocks": model.get("actor_depth", exact_model["actor_depth"])})
    dropout = model.get("dropout", 0.1)
    _exact_number(dropout, 0.1, "model dropout")

    training = _mapping(root.get("training"), "training")
    if not isinstance(training.get("validation_enabled", True), bool):
        raise ValueError("training.validation_enabled must be boolean")
    if "anomaly_forensics" in training:
        if not stochastic:
            raise ValueError("anomaly forensics require the stochastic scientific family")
        from j2j.context4_forensics import validate_capture_config
        validate_capture_config(training["anomaly_forensics"])
    if not stochastic and "model_probe" in training:
        raise ValueError("bounded model probes require the stochastic scientific family")
    if stochastic or "model_probe" in training:
        from j2j.context4_probes import validate_probe_config
        validate_probe_config(training.get("model_probe"))
    if stochastic and training.get("qg_objective") != "posterior_sample":
        raise ValueError("stochastic protocol requires posterior_sample QG")
    if raw and not stochastic and training.get("qg_objective") != "posterior_weighted":
        raise ValueError("raw protocol requires posterior_weighted QG")
    if not raw and training.get("qg_objective", "marginal") != "marginal":
        raise ValueError("legacy protocol requires marginal QG")
    kind = training.get("run_kind")
    if kind not in {"lr_pilot", "formal"}:
        raise ValueError("config training run kind is invalid")
    epochs = _positive_int(training.get("epochs"), "training epoch")
    if (kind == "lr_pilot" and epochs != 1) or (kind == "formal" and epochs != 30):
        raise ValueError("config training epoch count is invalid for the run kind")
    if scratch:
        legacy_fields = {
            "learning_rate",
            "gradient_clip_norm",
        }.intersection(training)
        if legacy_fields:
            raise ValueError(
                "scratch config must not mix legacy learning-rate or gradient-clip fields"
            )
        optimization = _mapping(
            training.get("optimization"), "training optimization"
        )
        if optimization.get("scheme") != SCRATCH_OPTIMIZER_SCHEME:
            raise ValueError("scratch optimizer scheme is invalid")
        peak_lrs = _mapping(
            optimization.get("peak_learning_rates"),
            "training optimization peak_learning_rates",
        )
        clip_norms = _mapping(
            optimization.get("gradient_clip_norms"),
            "training optimization gradient_clip_norms",
        )
        if set(peak_lrs) != {"q", "agf"} or set(clip_norms) != {"q", "agf"}:
            raise ValueError("scratch optimizer groups must be exactly q and agf")
        _exact_number(peak_lrs.get("q"), 2e-4, "scratch q peak learning rate")
        _exact_number(
            peak_lrs.get("agf"), 3e-4, "scratch agf peak learning rate"
        )
        _exact_number(clip_norms.get("q"), 1.0, "scratch q gradient clip")
        _exact_number(clip_norms.get("agf"), 1.0 if raw else 2.0, "scratch agf gradient clip")
    else:
        if "optimization" in training:
            raise ValueError(
                "legacy training config must not contain scratch optimization"
            )
        learning_rate = training.get("learning_rate")
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, (int, float)
        ):
            raise TypeError("config training learning rate must be numeric")
        if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise ValueError(
                "config training learning rate must be finite and positive"
            )

    optimizer = _mapping(training.get("optimizer"), "training optimizer")
    if optimizer.get("name") != "AdamW":
        raise ValueError("config optimizer must be AdamW")
    _exact_number(optimizer.get("weight_decay"), 1e-3, "optimizer weight_decay")
    betas = optimizer.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("config optimizer betas are invalid")
    _exact_number(betas[0], 0.9, "optimizer beta1")
    _exact_number(betas[1], 0.999, "optimizer beta2")
    _exact_number(optimizer.get("eps"), 1e-8, "optimizer eps")
    _exact_number(training.get("warmup_fraction"), 0.01, "warmup fraction")
    if training.get("schedule") != "cosine_to_zero":
        raise ValueError("config schedule must be cosine_to_zero")
    if not scratch:
        _exact_number(training.get("gradient_clip_norm"), 1.0, "gradient clip")
    effective_batch = _positive_int(
        training.get("effective_global_batch"),
        "effective global batch",
    )
    microbatch = _positive_int(training.get("microbatch_per_rank"), "microbatch")
    accumulation = _positive_int(training.get("accumulation_steps"), "accumulation batch")
    configured_world = _positive_int(training.get("world_size"), "world size")
    actual = _positive_int(actual_world_size, "actual world size")
    if configured_world != actual:
        raise ValueError("config world size does not match the runtime world")
    if raw:
        origins_per_rank = (effective_batch + configured_world - 1) // configured_world
        if accumulation != (origins_per_rank + microbatch - 1) // microbatch:
            raise ValueError("raw accumulation must cover the planned per-rank origins exactly")
    elif microbatch * accumulation * configured_world != effective_batch:
        raise ValueError(
            "config microbatch/accumulation/world product must equal effective batch"
        )
    _positive_int(
        training.get("diagnostic_interval_updates"),
        "training diagnostic_interval_updates",
    )

    runtime = _mapping(root.get("runtime"), "runtime")
    _positive_int(runtime.get("workers"), "runtime workers", allow_zero=True)
    _positive_int(runtime.get("prefetch"), "runtime prefetch")
    if not isinstance(runtime.get("cpu_affinity"), str) or not runtime["cpu_affinity"]:
        raise ValueError("config runtime cpu affinity must be declared")
    if runtime["cpu_affinity"] not in {"caller-bound", "launcher", "split-current"}:
        _parse_cpu_affinity(str(runtime["cpu_affinity"]))
    if runtime.get("device", "cuda") not in {"cuda", "cpu"}:
        raise ValueError("config runtime device is invalid")
    if runtime.get("precision", "bf16-mixed") not in {"bf16-mixed", "fp32"}:
        raise ValueError("config runtime precision is invalid")
    if runtime.get("torch_threads_per_rank") is not None:
        _positive_int(runtime["torch_threads_per_rank"], "runtime torch_threads_per_rank")
    if runtime.get("process_group_timeout_seconds") is not None:
        _positive_int(
            runtime["process_group_timeout_seconds"],
            "runtime process_group_timeout_seconds",
        )
    if runtime.get("timeout_seconds") is not None:
        _positive_int(runtime["timeout_seconds"], "runtime timeout_seconds")
    if (
        runtime.get("process_group_timeout_seconds") is not None
        and runtime.get("timeout_seconds") is not None
        and runtime["process_group_timeout_seconds"] != runtime["timeout_seconds"]
    ):
        raise ValueError("runtime timeout aliases disagree")
    _reject_secret_fields(root)
    return copy.deepcopy(dict(root))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_state(payload: object, *, label: str) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} warm checkpoint payload must be a mapping")
    state = None
    for key in ("state_dict", "proposal_state", "model_state", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if state is None and payload and all(
        isinstance(name, str) and isinstance(value, Tensor)
        for name, value in payload.items()
    ):
        state = payload
    if state is None:
        raise ValueError(f"{label} warm checkpoint has no recognized state mapping")
    if any(not isinstance(name, str) or not isinstance(value, Tensor) for name, value in state.items()):
        raise ValueError(f"{label} warm checkpoint state keys/tensors are invalid")
    return state  # type: ignore[return-value]


def _strip_prefixes(name: str, prefixes: tuple[str, ...]) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break
    return name


@dataclass(frozen=True)
class WarmStartReceipt:
    q_source_sha256: str
    agf_source_sha256: str
    imported_keys: tuple[str, ...]
    ignored_keys: tuple[str, ...]
    final_tensor_state_sha256: str


def load_model_only_warm_start(
    model: nn.Module,
    *,
    q_checkpoint: str | os.PathLike[str],
    expected_q_sha256: str,
    agf_checkpoint: str | os.PathLike[str],
    expected_agf_sha256: str,
) -> WarmStartReceipt:
    """Admit bytes first, then import only Q input/future, A and pred-proj."""

    q_path = Path(q_checkpoint)
    agf_path = Path(agf_checkpoint)
    expected_q = _require_sha(expected_q_sha256, "Q warm source")
    expected_agf = _require_sha(expected_agf_sha256, "AGF warm source")
    if not q_path.is_file() or not agf_path.is_file():
        raise ValueError("warm source checkpoint file is missing")
    actual_q = _sha256_file(q_path)
    actual_agf = _sha256_file(agf_path)
    if actual_q != expected_q:
        raise ValueError("Q warm source SHA mismatch")
    if actual_agf != expected_agf:
        raise ValueError("AGF warm source SHA mismatch")

    q_payload = torch.load(q_path, map_location="cpu", weights_only=False)
    agf_payload = torch.load(agf_path, map_location="cpu", weights_only=False)
    q_source = _checkpoint_state(q_payload, label="Q")
    agf_source = _checkpoint_state(agf_payload, label="AGF")

    target_state = model.state_dict()
    q_expected = {
        name.removeprefix("proposal.")
        for name in target_state
        if name.startswith(("proposal.input_proj.", "proposal.future."))
    }
    agf_expected = {
        name
        for name in target_state
        if name.startswith(("action_encoder.", "pred_proj."))
    }
    normalized_q: dict[str, Tensor] = {}
    ignored: list[str] = []
    unexpected: list[str] = []
    for source_name, value in q_source.items():
        name = _strip_prefixes(source_name, ("module.", "proposal."))
        if name.startswith(("input_proj.", "future.")):
            if name in normalized_q:
                unexpected.append(source_name)
            normalized_q[name] = value
        elif name.startswith("selector."):
            ignored.append(f"q:{source_name}")
        else:
            unexpected.append(source_name)
    missing_q = sorted(q_expected.difference(normalized_q))
    unexpected_q = sorted(set(normalized_q).difference(q_expected)) + sorted(unexpected)

    normalized_agf: dict[str, Tensor] = {}
    known_ignored_agf = (
        "actor.",
        "intent_actor.",
        "predictor.",
        "projector.",
        "sigreg.",
        "encoder.",
        "decoder.",
        "A.",
        "G.",
        "F.",
    )
    unexpected_agf: list[str] = []
    for source_name, value in agf_source.items():
        name = _strip_prefixes(source_name, ("module.", "model."))
        if name.startswith(("action_encoder.", "pred_proj.")):
            if name in normalized_agf:
                unexpected_agf.append(source_name)
            normalized_agf[name] = value
        elif name.startswith(known_ignored_agf):
            ignored.append(f"agf:{source_name}")
        else:
            unexpected_agf.append(source_name)
    missing_agf = sorted(agf_expected.difference(normalized_agf))
    extra_agf = sorted(set(normalized_agf).difference(agf_expected)) + sorted(unexpected_agf)
    if missing_q or unexpected_q or missing_agf or extra_agf:
        raise ValueError(
            "warm source key mismatch: "
            f"q_missing={missing_q}, q_unexpected={unexpected_q}, "
            f"agf_missing={missing_agf}, agf_unexpected={extra_agf}"
        )

    replacements: dict[str, Tensor] = {}
    for name, value in normalized_q.items():
        replacements[f"proposal.{name}"] = value
    replacements.update(normalized_agf)
    for name, value in replacements.items():
        expected = target_state[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"warm source key {name} has incompatible shape or dtype")

    admitted = {name: value.detach().clone() for name, value in target_state.items()}
    for name, value in replacements.items():
        admitted[name] = value.detach().clone()
    for name, value in tuple(admitted.items()):
        if not name.startswith("pred_proj."):
            continue
        if name.endswith("running_mean"):
            admitted[name] = torch.zeros_like(value)
        elif name.endswith("running_var"):
            admitted[name] = torch.ones_like(value)
        elif name.endswith("num_batches_tracked"):
            admitted[name] = torch.zeros_like(value)
    model.load_state_dict(admitted, strict=True)
    imported = tuple(sorted(replacements))
    return WarmStartReceipt(
        q_source_sha256=actual_q,
        agf_source_sha256=actual_agf,
        imported_keys=imported,
        ignored_keys=tuple(sorted(ignored)),
        final_tensor_state_sha256=_tensor_state_sha256(model),
    )


def _require_finite_gradients(
    parameters: Iterable[nn.Parameter], error_message: str
) -> None:
    flags_by_device: dict[torch.device, list[Tensor]] = {}
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            flags_by_device.setdefault(gradient.device, []).append(
                torch.isfinite(gradient.detach()).all()
            )
    # Inspect every gradient, but cross the device/host boundary only once
    # per device. Keep this gate on each side of clipping, before Adam.
    for flags in flags_by_device.values():
        if not bool(torch.stack(flags).all()):
            raise FloatingPointError(error_message)


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    partials: list[Tensor] = []
    for parameter in parameters:
        if parameter.grad is not None:
            gradient = parameter.grad.detach().double()
            partials.append(gradient.square().sum())
    if not partials:
        return 0.0
    # All Context4 DDP parameters are rank-local on one device.  Aggregate
    # there and cross the device/host boundary once, rather than synchronizing
    # once per parameter tensor merely to produce telemetry.
    return math.sqrt(float(torch.stack(partials).sum().cpu()))


def _owner_gradient_norms(
    groups: Mapping[str, Sequence[nn.Parameter]],
) -> Mapping[str, float]:
    names = tuple(groups)
    squared_norms: list[Tensor] = []
    for name in names:
        parameters = tuple(groups[name])
        if not parameters:
            raise ValueError(f"gradient owner group {name} is empty")
        partials = [
            parameter.grad.detach().double().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        squared_norms.append(
            torch.stack(partials).sum()
            if partials
            else parameters[0].new_zeros((), dtype=torch.float64)
        )
    values = torch.stack(squared_norms).sqrt().cpu().tolist()
    return {name: float(value) for name, value in zip(names, values, strict=True)}


@dataclass(frozen=True)
class OptimizerUpdate:
    success: bool
    pre_clip_grad_norm: float
    post_clip_grad_norm: float
    clip_scale: float
    reason: str | None = None


@dataclass(frozen=True)
class GlobalUpdateReceipt:
    success: bool
    micro_slots: int
    accumulation_steps: int
    tail_flushed: bool
    empty_local_slots: int
    optimizer_steps: int
    scheduler_steps: int
    pre_clip_grad_norm: float
    post_clip_grad_norm: float
    clip_scale: float | None
    global_denominators: Mapping[str, int]
    local_numerators: Mapping[str, float]
    local_denominators: Mapping[str, int]
    owner_grad_norms: Mapping[str, Mapping[str, object]]
    clip_domains: Mapping[str, Mapping[str, float]]
    diagnostics: Mapping[str, object] | None = None
    reason: str | None = None
    local_stochastic_statistics: Mapping[str, float] | None = None


def _floating_output_tensors(value: object) -> tuple[Tensor, ...]:
    if isinstance(value, Tensor):
        return (value,) if value.is_floating_point() else ()
    if isinstance(value, Mapping):
        return tuple(
            tensor
            for child in value.values()
            for tensor in _floating_output_tensors(child)
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            tensor for child in value for tensor in _floating_output_tensors(child)
        )
    return ()


class _ActivationCollector:
    """One-forward hook collector for actually invoked leaf-module FQNs."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.handles: list[object] = []
        self.sum_squares: dict[str, list[Tensor]] = {}
        self.maxima: dict[str, list[Tensor]] = {}
        self.numel: dict[str, int] = {}
        self.layernorm_inputs: dict[str, list[dict]] = {}
        self.enabled = True

    def start(self) -> None:
        if self.handles:
            raise RuntimeError("activation collector was started twice")
        for name, child in self.module.named_modules():
            if name.startswith("proposal.") and isinstance(child, nn.LayerNorm):
                def capture_norm(layer, inputs, *, fqn=name):
                    if self.enabled:
                        from j2j.context4_forensics import layernorm_input_statistics
                        self.layernorm_inputs.setdefault(fqn, []).append(layernorm_input_statistics(layer, inputs[0]))
                self.handles.append(child.register_forward_pre_hook(capture_norm))
            if not name or any(True for _ in child.children()):
                continue

            def capture(_module, _inputs, output, *, fqn=name):
                if not self.enabled:
                    return
                tensors = _floating_output_tensors(output)
                if not tensors:
                    return
                values = tuple(tensor.detach() for tensor in tensors)
                square = sum((torch.linalg.vector_norm(value, dtype=torch.float32).square()
                              for value in values), values[0].new_zeros((), dtype=torch.float32))
                maximum = torch.stack(
                    tuple(torch.maximum(value.amax().abs(), value.amin().abs()).float() for value in values)
                ).amax()
                self.sum_squares.setdefault(fqn, []).append(square.detach())
                self.maxima.setdefault(fqn, []).append(maximum.detach())
                self.numel[fqn] = self.numel.get(fqn, 0) + sum(
                    value.numel() for value in values
                )

            self.handles.append(child.register_forward_hook(capture))

    def close(self) -> None:
        for handle in self.handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self.handles.clear()

    def summary(self) -> Mapping[str, Mapping[str, float | int]]:
        result: dict[str, Mapping[str, float | int]] = {}
        for name in sorted(self.sum_squares):
            count = self.numel[name]
            sum_square = float(torch.stack(self.sum_squares[name]).sum().double().cpu())
            maximum = float(torch.stack(self.maxima[name]).amax().double().cpu())
            rms = math.sqrt(sum_square / count)
            if not math.isfinite(rms) or not math.isfinite(maximum):
                raise FloatingPointError(f"activation diagnostic {name} is non-finite")
            result[name] = {"rms": rms, "maxabs": maximum, "numel": count}
        return result

    def numerical_summary(self) -> dict:
        from j2j.context4_forensics import packed_qkv_gradient_statistics
        return {"scope": "explicit_modules_only", "layernorm_inputs": self.layernorm_inputs,
                "packed_qkv_pre_clip": {name: packed_qkv_gradient_statistics(child)
                    for name, child in self.module.named_modules()
                    if name.startswith("proposal.") and isinstance(child, nn.MultiheadAttention)}}


def _per_fqn_gradient_statistics(
    module: nn.Module,
) -> Mapping[str, Mapping[str, float | int]]:
    result: dict[str, Mapping[str, float | int]] = {}
    for name, parameter in sorted(module.named_parameters()):
        gradient = parameter.grad
        if gradient is None:
            l2 = 0.0
            maximum = 0.0
        else:
            detached = gradient.detach().float()
            l2 = float(torch.linalg.vector_norm(detached).double().cpu())
            maximum = float(detached.abs().amax().double().cpu())
        if not math.isfinite(l2) or not math.isfinite(maximum):
            raise FloatingPointError(f"gradient diagnostic {name} is non-finite")
        result[name] = {
            "l2": l2,
            "maxabs": maximum,
            "numel": parameter.numel(),
        }
    return result


def _owner_parameter_groups(model: nn.Module) -> Mapping[str, tuple[nn.Parameter, ...]]:
    candidate = getattr(model, "module", model)
    joint = getattr(candidate, "joint_model", candidate)
    required = ("proposal", "action_encoder", "actor", "forward_core", "pred_proj")
    if not all(isinstance(getattr(joint, name, None), nn.Module) for name in required):
        return {
            "joint": tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        }
    groups = {
        "q": tuple(joint.proposal.parameters()),
        "a": tuple(joint.action_encoder.parameters()),
        "g": tuple(joint.actor.parameters()),
        "f": tuple(joint.forward_core.parameters()) + tuple(joint.pred_proj.parameters()),
    }
    identifiers = [id(parameter) for values in groups.values() for parameter in values]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Q/A/G/F optimizer owner parameter groups overlap")
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(identifiers) != expected:
        raise RuntimeError("Q/A/G/F optimizer owner groups do not cover the trainable tree")
    return groups


@dataclass(frozen=True)
class OptimizerGroupReceipt:
    ordered_fqns: tuple[str, ...]
    fqn_sha256: str
    peak_lr: float
    clip_norm: float


@dataclass(frozen=True)
class OptimizerPartitionReceipt:
    scheme: str
    ordered_group_names: tuple[str, ...]
    partition_validated: bool
    groups: Mapping[str, OptimizerGroupReceipt]


def build_q_agf_optimizer(
    model: nn.Module,
    *,
    optimization: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    training_contract: Mapping[str, str] | None = None,
) -> tuple[torch.optim.AdamW, OptimizerPartitionReceipt]:
    """Build the one-optimizer, named Q/aggregate-AGF training contract."""

    optimization = _mapping(optimization, "training optimization")
    optimizer_config = _mapping(optimizer_config, "training optimizer")
    if optimization.get("scheme") != SCRATCH_OPTIMIZER_SCHEME:
        raise ValueError("Q/AGF optimizer scheme is invalid")
    peak_lrs = _mapping(
        optimization.get("peak_learning_rates"),
        "training optimization peak_learning_rates",
    )
    clip_norms = _mapping(
        optimization.get("gradient_clip_norms"),
        "training optimization gradient_clip_norms",
    )
    if set(peak_lrs) != {"q", "agf"} or set(clip_norms) != {"q", "agf"}:
        raise ValueError("Q/AGF optimizer groups must be exactly q and agf")
    _exact_number(peak_lrs.get("q"), 2e-4, "Q peak learning rate")
    _exact_number(peak_lrs.get("agf"), 3e-4, "AGF peak learning rate")
    _exact_number(clip_norms.get("q"), 1.0, "Q gradient clip")
    raw = training_contract is not None and _validated_training_contract(training_contract)["id"] in _RAW_TRAINING_IDS
    _exact_number(clip_norms.get("agf"), 1.0 if raw else 2.0, "AGF gradient clip")
    if optimizer_config.get("name") != "AdamW":
        raise ValueError("Q/AGF optimizer must be AdamW")
    _exact_number(
        optimizer_config.get("weight_decay"), 1e-3, "optimizer weight_decay"
    )
    betas = optimizer_config.get("betas")
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("Q/AGF optimizer betas are invalid")
    _exact_number(betas[0], 0.9, "optimizer beta1")
    _exact_number(betas[1], 0.999, "optimizer beta2")
    _exact_number(optimizer_config.get("eps"), 1e-8, "optimizer eps")

    owner_groups = _owner_parameter_groups(model)
    if tuple(owner_groups) != ("q", "a", "g", "f"):
        raise ValueError(
            "Q/AGF optimizer owner partition must contain q, a, g, f in order"
        )
    owner_parameters: dict[str, tuple[nn.Parameter, ...]] = {}
    for owner in ("q", "a", "g", "f"):
        values = tuple(owner_groups[owner])
        if not values or any(
            not isinstance(parameter, nn.Parameter) or not parameter.requires_grad
            for parameter in values
        ):
            raise ValueError(
                f"Q/AGF optimizer owner group {owner} has invalid parameters"
            )
        owner_parameters[owner] = values

    identifiers = [
        id(parameter)
        for owner in ("q", "a", "g", "f")
        for parameter in owner_parameters[owner]
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Q/AGF optimizer owner partition has parameter overlap")
    expected = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if set(identifiers) != expected:
        raise ValueError(
            "Q/AGF optimizer owner partition does not cover the trainable tree"
        )

    candidate = getattr(model, "module", model)
    joint = getattr(candidate, "joint_model", candidate)
    names_by_id = {
        id(parameter): name
        for name, parameter in joint.named_parameters()
        if parameter.requires_grad
    }
    if set(names_by_id) != expected:
        raise ValueError("Q/AGF optimizer cannot assign canonical owner FQNs")

    raw_groups = {
        "q": owner_parameters["q"],
        "agf": (
            owner_parameters["a"]
            + owner_parameters["g"]
            + owner_parameters["f"]
        ),
    }
    prefixes = {
        "q": ("proposal.",),
        "agf": ("action_encoder.", "actor.", "forward_core.", "pred_proj."),
    }
    optimizer_groups: list[dict[str, object]] = []
    receipts: dict[str, OptimizerGroupReceipt] = {}
    for group_name in ("q", "agf"):
        ordered = tuple(
            sorted(
                ((names_by_id[id(parameter)], parameter) for parameter in raw_groups[group_name]),
                key=lambda item: item[0],
            )
        )
        fqns = tuple(name for name, _parameter in ordered)
        if not fqns or any(
            not name.startswith(prefixes[group_name]) for name in fqns
        ):
            raise ValueError(
                f"Q/AGF optimizer owner group {group_name} has misnamed parameters"
            )
        peak_lr = float(peak_lrs[group_name])
        clip_norm = float(clip_norms[group_name])
        optimizer_groups.append(
            {
                "params": tuple(parameter for _name, parameter in ordered),
                "lr": peak_lr,
                "group_name": group_name,
            }
        )
        receipts[group_name] = OptimizerGroupReceipt(
            ordered_fqns=fqns,
            fqn_sha256=_canonical_json_sha256(list(fqns)),
            peak_lr=peak_lr,
            clip_norm=clip_norm,
        )

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=0.0,
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(optimizer_config["eps"]),
    )
    receipt = OptimizerPartitionReceipt(
        scheme=SCRATCH_OPTIMIZER_SCHEME,
        ordered_group_names=("q", "agf"),
        partition_validated=True,
        groups=receipts,
    )
    return optimizer, receipt


@torch.no_grad()
def _require_finite_stable_q(model: nn.Module, world_size: int) -> None:
    """One batched raw-state gate; smooth bounds must not hide corrupted Inf."""
    proposal = getattr(_checkpoint_module(model), "proposal", None)
    if getattr(proposal, "stability", None) is None:
        return
    flags = torch.stack([torch.isfinite(p).all() for p in proposal.parameters()])
    invalid = (~flags.all()).to(torch.int64)
    if world_size > 1:
        torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
    if bool(invalid):
        raise FloatingPointError("stable Q contains nonfinite raw parameters")


def run_global_update(
    *,
    model: nn.Module,
    microbatches: Iterable[object | None],
    joint_numerator_fn,
    global_denominators: Mapping[str, int],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    world_size: int,
    accumulation_steps: int,
    max_grad_norm: float | None = None,
    gradient_clip_norms: Mapping[str, float] | None = None,
    successful_updates: int = 0,
    diagnostic_interval_updates: int | None = None,
    loss_weights: Mapping[str, float] | None = None,
    anomaly_capture=None,
    anomaly_context: Mapping[str, object] | None = None,
) -> GlobalUpdateReceipt:
    """Accumulate one DDP global origin batch and advance exactly once."""

    branches = ("q", "f", "g_local", "g_goal", "qg")
    slots = tuple(microbatches)
    if not slots or len(slots) > accumulation_steps:
        raise ValueError("microbatch slots must form one bounded accumulation window")
    if isinstance(accumulation_steps, bool) or not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be positive")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be positive")
    if isinstance(successful_updates, bool) or not isinstance(successful_updates, int) or successful_updates < 0:
        raise ValueError("successful_updates must be a non-negative integer")
    if diagnostic_interval_updates is None:
        collect_diagnostics = False
    else:
        if (
            isinstance(diagnostic_interval_updates, bool)
            or not isinstance(diagnostic_interval_updates, int)
            or diagnostic_interval_updates <= 0
        ):
            raise ValueError("diagnostic_interval_updates must be a positive integer")
        collect_diagnostics = (
            (successful_updates + 1) % diagnostic_interval_updates == 0
        )
    if set(global_denominators) != set(branches):
        raise ValueError("global denominator ledger must contain exactly five branches")
    legacy_clip = max_grad_norm is not None
    isolated_clip = gradient_clip_norms is not None
    if legacy_clip == isolated_clip:
        raise ValueError(
            "exactly one of max_grad_norm or gradient_clip_norms is required"
        )
    if legacy_clip:
        if (
            isinstance(max_grad_norm, bool)
            or not isinstance(max_grad_norm, (int, float))
            or not math.isfinite(float(max_grad_norm))
            or float(max_grad_norm) <= 0.0
        ):
            raise ValueError("max_grad_norm must be finite and positive")
        resolved_max_grad_norm = float(max_grad_norm)
        resolved_clip_norms: Mapping[str, float] | None = None
    else:
        resolved_max_grad_norm = None
        if not isinstance(gradient_clip_norms, Mapping) or set(
            gradient_clip_norms
        ) != {"q", "agf"}:
            raise ValueError("gradient_clip_norms must contain exactly q and agf")
        resolved: dict[str, float] = {}
        for name in ("q", "agf"):
            value = gradient_clip_norms[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"gradient_clip_norms {name} must be finite and positive"
                )
            resolved[name] = float(value)
        resolved_clip_norms = resolved
    denominators = {
        name: _positive_int(value, f"global denominator {name}", allow_zero=True)
        for name, value in global_denominators.items()
    }
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("global update model has no trainable parameters")
    if not callable(joint_numerator_fn):
        raise TypeError("joint_numerator_fn must be callable")
    scheduler_step = getattr(scheduler, "step", None)
    if not callable(scheduler_step):
        raise TypeError("scheduler must expose step()")
    no_sync = getattr(model, "no_sync", None)
    _require_finite_stable_q(model, world_size)
    optimizer.zero_grad(set_to_none=True)
    empty_slots = 0
    local_numerator_totals = {name: 0.0 for name in branches}
    local_denominator_totals = {name: 0 for name in branches}
    diagnostic_module = _checkpoint_module(model)
    activation_collector = (
        _ActivationCollector(diagnostic_module) if collect_diagnostics else None
    )
    objective_wrapper = getattr(model, "module", model)
    stochastic = getattr(objective_wrapper, "qg_objective", "marginal") == "posterior_sample"
    local_stochastic = {name: 0.0 for name in _STOCHASTIC_STAT_FIELDS} if stochastic else None
    mode_accumulator = ModeDiagnosticAccumulator(
        qg_objective=getattr(objective_wrapper, "qg_objective", "marginal")
    ) if collect_diagnostics and not stochastic else None
    prior_collect = getattr(objective_wrapper, "collect_diagnostics", None)
    has_collect_flag = hasattr(objective_wrapper, "collect_diagnostics")
    if has_collect_flag:
        objective_wrapper.collect_diagnostics = collect_diagnostics
    if activation_collector is not None:
        activation_collector.start()
    diagnostics: Mapping[str, object] | None = None

    try:
        capture_session = None if anomaly_capture is None else anomaly_capture.begin_update(
            model=model, microbatches=slots, optimizer=optimizer, scheduler=scheduler,
            world_size=world_size, successful_updates=successful_updates,
            global_denominators=denominators, context=anomaly_context)
        for slot_index, microbatch in enumerate(slots):
            synchronize = slot_index == len(slots) - 1
            context = nullcontext() if synchronize or not callable(no_sync) else no_sync()
            with context:
                if activation_collector is not None:
                    activation_collector.enabled = True
                if microbatch is None:
                    empty_slots += 1
                # Empty rank slots deliberately traverse the caller's model
                # forward too.  Real DDP prepares its reducer during forward;
                # constructing zero gradients from parameters here would let
                # an empty rank skip that collective state transition.
                result = joint_numerator_fn(microbatch)
                if local_stochastic is not None:
                    _accumulate_stochastic_statistics(local_stochastic, getattr(result, "stochastic_statistics", None))
                numerators = getattr(result, "numerators", None)
                local_denominators = getattr(result, "denominators", None)
                if not isinstance(numerators, Mapping) or not isinstance(local_denominators, Mapping):
                    raise TypeError("joint numerator result must expose numerator/denominator mappings")
                if set(numerators) != set(branches) or set(local_denominators) != set(branches):
                    raise ValueError("joint numerator result must contain exactly five branches")
                if mode_accumulator is not None:
                    diagnostic_tensors = getattr(result, "diagnostic_tensors", None)
                    if diagnostic_tensors is not None:
                        if not isinstance(diagnostic_tensors, Mapping):
                            raise TypeError("diagnostic_tensors must be a mapping")
                        required_diagnostics = {
                            "q_responsibility",
                            "q_tape",
                            "q_active_h",
                            "qg_logits",
                            "qg_labels",
                        }
                        if set(diagnostic_tensors) != required_diagnostics:
                            raise ValueError("diagnostic_tensors have an invalid schema")
                        mode_accumulator.update(**dict(diagnostic_tensors))
                normalized: dict[str, Tensor] = {}
                for name in branches:
                    numerator = numerators[name]
                    if not isinstance(numerator, Tensor) or numerator.numel() != 1:
                        raise TypeError(f"{name} numerator must be a scalar tensor")
                    if not bool(torch.isfinite(numerator.detach())):
                        raise FloatingPointError(f"{name} numerator is non-finite")
                    local_count = local_denominators[name]
                    if isinstance(local_count, bool) or not isinstance(local_count, int) or local_count < 0:
                        raise ValueError(f"{name} local denominator is invalid")
                    local_numerator_totals[name] += float(numerator.detach())
                    local_denominator_totals[name] += local_count
                    if denominators[name] == 0:
                        if local_count != 0 or float(numerator.detach()) != 0.0:
                            raise ValueError(f"{name} has local work but a zero global denominator")
                        normalized[name] = numerator.float()
                    else:
                        normalized[name] = numerator.float() * float(world_size) / float(denominators[name])
                loss = weighted_joint_loss(
                    q=normalized["q"],
                    f=normalized["f"],
                    g_local=normalized["g_local"],
                    g_goal=normalized["g_goal"],
                    qg=normalized["qg"],
                    weights=loss_weights,
                )
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError("global update loss is non-finite")
                if activation_collector is not None:
                    activation_collector.enabled = False
                loss.backward()
        _require_finite_gradients(parameters, "global update gradient is non-finite")
        owner_groups = _owner_parameter_groups(model)
        owner_pre = _owner_gradient_norms(owner_groups)
        pre_clip = math.sqrt(sum(value * value for value in owner_pre.values()))
        if capture_session is not None:
            capture_session.capture_if_needed(owner_pre["q"])
        if activation_collector is not None:
            diagnostics = {
                "pre_clip_grad": _per_fqn_gradient_statistics(diagnostic_module),
                "activation": activation_collector.summary(),
                "modes": mode_accumulator.summary() if mode_accumulator is not None else None,
                "numerical_forensics": activation_collector.numerical_summary(),
            }
        if resolved_clip_norms is None:
            if resolved_max_grad_norm is None:
                raise RuntimeError("validated legacy gradient clip is missing")
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=resolved_max_grad_norm,
                error_if_nonfinite=True,
            )
            owner_post = _owner_gradient_norms(owner_groups)
            post_clip = math.sqrt(
                sum(value * value for value in owner_post.values())
            )
            scale = (
                1.0
                if pre_clip == 0.0
                else min(1.0, resolved_max_grad_norm / (pre_clip + 1.0e-6))
            )
            clip_domains = {
                "joint": {
                    "pre_clip": pre_clip,
                    "post_clip": post_clip,
                    "clip_scale": scale,
                    "max_norm": resolved_max_grad_norm,
                }
            }
            owner_records: Mapping[str, Mapping[str, object]] = {
                name: {
                    "pre_clip": owner_pre[name],
                    "post_clip": owner_post[name],
                    "clip_scale": scale,
                }
                for name in owner_pre
            }
            receipt_clip_scale: float | None = scale
        else:
            if tuple(owner_groups) != ("q", "a", "g", "f"):
                raise ValueError(
                    "isolated clipping requires exact q, a, g, f owner groups"
                )
            domain_parameters = {
                "q": tuple(owner_groups["q"]),
                "agf": (
                    tuple(owner_groups["a"])
                    + tuple(owner_groups["g"])
                    + tuple(owner_groups["f"])
                ),
            }
            domain_pre = {
                "q": owner_pre["q"],
                "agf": math.sqrt(
                    owner_pre["a"] ** 2
                    + owner_pre["g"] ** 2
                    + owner_pre["f"] ** 2
                ),
            }
            for name in ("q", "agf"):
                torch.nn.utils.clip_grad_norm_(
                    domain_parameters[name],
                    max_norm=resolved_clip_norms[name],
                    error_if_nonfinite=True,
                )
            owner_post = _owner_gradient_norms(owner_groups)
            post_clip = math.sqrt(
                sum(value * value for value in owner_post.values())
            )
            domain_post = {
                "q": owner_post["q"],
                "agf": math.sqrt(
                    owner_post["a"] ** 2
                    + owner_post["g"] ** 2
                    + owner_post["f"] ** 2
                ),
            }
            domain_scales = {
                name: (
                    1.0
                    if domain_pre[name] == 0.0
                    else min(
                        1.0,
                        resolved_clip_norms[name] / (domain_pre[name] + 1.0e-6),
                    )
                )
                for name in ("q", "agf")
            }
            clip_domains = {
                name: {
                    "pre_clip": domain_pre[name],
                    "post_clip": domain_post[name],
                    "clip_scale": domain_scales[name],
                    "max_norm": resolved_clip_norms[name],
                }
                for name in ("q", "agf")
            }
            owner_records = {
                name: {
                    "pre_clip": owner_pre[name],
                    "post_clip": owner_post[name],
                    "applied_clip_domain": "q" if name == "q" else "agf",
                }
                for name in ("q", "a", "g", "f")
            }
            receipt_clip_scale = None
        if not math.isfinite(pre_clip) or not math.isfinite(post_clip):
            raise FloatingPointError("global update clipped gradient is non-finite")
        if any(
            not math.isfinite(value)
            for values in (owner_pre, owner_post)
            for value in values.values()
        ):
            raise FloatingPointError("owner gradient norm is non-finite")
        _require_finite_gradients(
            parameters, "global update clipped gradient is non-finite"
        )
        optimizer.step()
        _require_finite_stable_q(model, world_size)
        scheduler_step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        if activation_collector is not None:
            activation_collector.close()
        if has_collect_flag:
            objective_wrapper.collect_diagnostics = prior_collect
    return GlobalUpdateReceipt(
        success=True,
        micro_slots=len(slots),
        accumulation_steps=accumulation_steps,
        tail_flushed=len(slots) < accumulation_steps,
        empty_local_slots=empty_slots,
        optimizer_steps=1,
        scheduler_steps=1,
        pre_clip_grad_norm=pre_clip,
        post_clip_grad_norm=post_clip,
        clip_scale=receipt_clip_scale,
        global_denominators=dict(denominators),
        local_numerators=dict(local_numerator_totals),
        local_denominators=dict(local_denominator_totals),
        owner_grad_norms=owner_records,
        clip_domains=clip_domains,
        diagnostics=diagnostics,
        local_stochastic_statistics=local_stochastic,
    )


def build_update_optimization_telemetry(
    *,
    optimizer: torch.optim.Optimizer,
    update: GlobalUpdateReceipt,
    optimizer_partition: OptimizerPartitionReceipt | None = None,
) -> Mapping[str, object]:
    """Render legacy or scratch optimization telemetry without ambiguity."""

    if not isinstance(update, GlobalUpdateReceipt):
        raise TypeError("optimization telemetry requires GlobalUpdateReceipt")
    if optimizer_partition is None:
        if update.clip_scale is None:
            raise ValueError("legacy telemetry requires one joint clip scale")
        learning_rates = {
            f"group_{index}": float(group["lr"])
            for index, group in enumerate(optimizer.param_groups)
        }
        return {
            "lr": learning_rates,
            "grad": {
                **dict(update.owner_grad_norms),
                "joint": {
                    "pre_clip": update.pre_clip_grad_norm,
                    "post_clip": update.post_clip_grad_norm,
                    "clip_scale": update.clip_scale,
                },
            },
        }

    if not isinstance(optimizer_partition, OptimizerPartitionReceipt):
        raise TypeError("scratch telemetry requires an optimizer partition receipt")
    if (
        optimizer_partition.scheme != SCRATCH_OPTIMIZER_SCHEME
        or tuple(optimizer_partition.ordered_group_names) != ("q", "agf")
        or not optimizer_partition.partition_validated
    ):
        raise ValueError("scratch optimizer partition identity is invalid")
    if len(optimizer.param_groups) != 2:
        raise ValueError("scratch optimizer must contain two parameter groups")
    group_names = tuple(group.get("group_name") for group in optimizer.param_groups)
    if group_names != optimizer_partition.ordered_group_names:
        raise ValueError("scratch optimizer group order disagrees with its receipt")
    if update.clip_scale is not None:
        raise ValueError("scratch telemetry must not carry a joint clip scale")
    if tuple(update.clip_domains) != ("q", "agf"):
        raise ValueError("scratch telemetry clip domains are invalid")
    if tuple(update.owner_grad_norms) != ("q", "a", "g", "f"):
        raise ValueError("scratch telemetry owner groups are invalid")
    pre_clip = float(update.pre_clip_grad_norm)
    post_clip = float(update.post_clip_grad_norm)
    if pre_clip == 0.0:
        if post_clip != 0.0:
            raise ValueError("joint post-clip norm cannot be positive when pre-clip is zero")
        post_over_pre = 1.0
    else:
        post_over_pre = post_clip / pre_clip
    if not all(math.isfinite(value) and value >= 0.0 for value in (pre_clip, post_clip, post_over_pre)):
        raise ValueError("scratch joint gradient telemetry must be finite and non-negative")
    learning_rates = {
        str(group_name): float(group["lr"])
        for group_name, group in zip(
            optimizer_partition.ordered_group_names,
            optimizer.param_groups,
            strict=True,
        )
    }
    return {
        "lr": learning_rates,
        "grad": {
            "clip_domains": dict(update.clip_domains),
            "owners": dict(update.owner_grad_norms),
            "joint": {
                "pre_clip": pre_clip,
                "post_clip": post_clip,
                "post_over_pre": post_over_pre,
            },
        },
    }


def _concatenate_joint_batches(batches: Sequence[JointBatch]) -> JointBatch:
    values = tuple(batches)
    if not values or any(type(value) is not JointBatch for value in values):
        raise ValueError("joint batch concatenation requires production JointBatch values")
    if len(values) == 1:
        return values[0]
    q_rows: list[Tensor] = []
    offset = 0
    for value in values:
        q_rows.append(value.q_origin_row + offset)
        offset += int(value.current_grid.shape[0])

    def combine(name: str) -> Tensor:
        tensors = tuple(getattr(value, name) for value in values)
        if any(not isinstance(item, Tensor) for item in tensors):
            raise TypeError(f"JointBatch field {name} must contain tensors")
        return torch.cat(tensors, dim=0)

    return JointBatch(
        origin_indices=combine("origin_indices"),
        current_grid=combine("current_grid"),
        next_grid=combine("next_grid"),
        local_intent=combine("local_intent"),
        previous_raw4=combine("previous_raw4"),
        outgoing_raw4=combine("outgoing_raw4"),
        action_ids=combine("action_ids"),
        context_grid=combine("context_grid"),
        context_incoming_raw4=combine("context_incoming_raw4"),
        context_outgoing_raw4=combine("context_outgoing_raw4"),
        context_age=combine("context_age"),
        context_type=combine("context_type"),
        context_valid=combine("context_valid"),
        q_goal_grid=combine("q_goal_grid"),
        q_target_grid=combine("q_target_grid"),
        q_active_h=combine("q_active_h"),
        q_origin_row=torch.cat(q_rows, dim=0),
        goal_intent=combine("goal_intent"),
        terminal_grid=combine("terminal_grid"),
        terminal_intent=combine("terminal_intent"),
        terminal_previous_raw4=combine("terminal_previous_raw4"),
        terminal_action_ids=combine("terminal_action_ids"),
        q_sample_keys=tuple(key for value in values for key in value.q_sample_keys),
    )


def _move_joint_batch(batch: JointBatch, device: torch.device) -> JointBatch:
    if type(batch) is not JointBatch:
        raise TypeError("only production JointBatch values may enter the runner")
    moved = {
        name: (getattr(batch, name).to(device=device, non_blocking=device.type == "cuda")
               if isinstance(getattr(batch, name), Tensor) else getattr(batch, name))
        for name in JointBatch.__dataclass_fields__
    }
    return JointBatch(**moved)


def _batch_denominators(batch: JointBatch) -> Mapping[str, int]:
    transitions = int(batch.current_grid.shape[0])
    terminals = int(batch.terminal_grid.shape[0])
    source = batch.current_grid if transitions else batch.terminal_grid
    if source.ndim != 3 or source.shape[0] == 0:
        raise ValueError("a physical joint batch must contain a transition or terminal")
    spatial, latent = source.shape[1:]
    return branch_denominators(
        q_occurrences=int(batch.q_goal_grid.shape[0]),
        transitions=transitions,
        terminals=terminals,
        spatial_tokens=int(spatial),
        latent_dim=int(latent),
    )


def _distributed_sum_int(value: int, *, world_size: int, device: torch.device) -> int:
    if world_size == 1:
        return int(value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("distributed reduction requires an initialized process group")
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def _distributed_sum_float(value: float, *, world_size: int, device: torch.device) -> float:
    if world_size == 1:
        return float(value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("distributed reduction requires an initialized process group")
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def _distributed_max_int(value: int, *, world_size: int, device: torch.device) -> int:
    if world_size == 1:
        return int(value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("distributed reduction requires an initialized process group")
    tensor = torch.tensor(value, dtype=torch.int64, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return int(tensor.item())


class _IndexedFirstUseDataset(Dataset):
    """Expose one deterministic first-use sequence through PyTorch DataLoader."""

    def __init__(
        self,
        dataset: object,
        dataset_indices: Sequence[int],
        *,
        variant_contract: VariantContract | None = None,
    ) -> None:
        getitem = getattr(dataset, "__getitem__", None)
        if not callable(getitem):
            raise TypeError("source dataset must support indexed materialization")
        indices = tuple(dataset_indices)
        if any(type(value) is not int or value < 0 for value in indices):
            raise ValueError("first-use dataset indices must be non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("first-use DataLoader sequence must not repeat a trajectory")
        self.dataset = dataset
        self.dataset_indices = indices
        self.variant_contract = _contract_or_full(variant_contract)

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def __getitem__(self, position: int) -> tuple[int, object]:
        if type(position) is not int or not 0 <= position < len(self.dataset_indices):
            raise IndexError("first-use position is outside the rank sequence")
        dataset_index = self.dataset_indices[position]
        return dataset_index, _prepare_for_variant(
            self.dataset[dataset_index], self.variant_contract
        )


def _identity_collate(value: object) -> object:
    return value


def _context4_worker_init(_worker_id: int) -> None:
    # Rank CPU affinity is inherited by forked workers.  One intra-op thread
    # per loader worker prevents nested pools from competing with each other.
    torch.set_num_threads(1)


def _rank_first_use_order(plan: RankOriginPlan, rank: int) -> tuple[int, ...]:
    if not 0 <= rank < plan.world_size:
        raise ValueError("rank is outside the plan")
    seen: set[int] = set()
    ordered: list[int] = []
    for update_index in range(plan.update_count):
        for row in plan.slices_for(rank, update_index):
            trajectory_index = row.trajectory_index
            if trajectory_index not in seen:
                seen.add(trajectory_index)
                ordered.append(plan.descriptors[trajectory_index].dataset_index)
        if update_index == 0:
            zero_motion = sorted(
                (
                    index
                    for index, descriptor in enumerate(plan.descriptors)
                    if descriptor.origin_count == 0 and plan.owner_for(index) == rank
                ),
                key=lambda index: (plan.descriptors[index].trajectory_key, index),
            )
            for trajectory_index in zero_motion:
                if trajectory_index in seen:
                    raise RuntimeError("zero-motion trajectory unexpectedly entered an origin slice")
                seen.add(trajectory_index)
                ordered.append(plan.descriptors[trajectory_index].dataset_index)
    expected = {index for index, owner in enumerate(plan.owners) if owner == rank}
    if seen != expected:
        raise RuntimeError("rank first-use order does not cover its trajectory owners exactly once")
    return tuple(ordered)


class _OrderedDataLoaderProxy:
    """Map expected dataset indices onto one asynchronous first-use iterator."""

    def __init__(self, loader: DataLoader) -> None:
        self._iterator: Iterator[object] = iter(loader)
        self._remaining = len(loader.dataset)  # type: ignore[arg-type]

    def __getitem__(self, expected_dataset_index: int) -> object:
        if self._remaining <= 0:
            raise RuntimeError("rank DataLoader was consumed beyond its one-pass plan")
        try:
            row = next(self._iterator)
        except StopIteration as exc:  # pragma: no cover - defensive loader boundary
            raise RuntimeError("rank DataLoader ended before the owner plan") from exc
        self._remaining -= 1
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise TypeError("rank DataLoader must return (dataset_index, item)")
        observed_index, item = row
        if isinstance(observed_index, Tensor):
            if observed_index.numel() != 1:
                raise TypeError("DataLoader index tensor must be scalar")
            observed_index = int(observed_index.item())
        if observed_index != expected_dataset_index:
            raise RuntimeError("DataLoader first-use order drifted from the owner plan")
        return item

    def assert_exhausted(self) -> None:
        if self._remaining != 0:
            raise RuntimeError("rank DataLoader did not consume every owned trajectory")


def _build_rank_dataloader(
    dataset: object,
    plan: RankOriginPlan,
    *,
    rank: int,
    workers: int,
    prefetch: int,
    pin_memory: bool,
    seed: int,
    variant_contract: VariantContract | None = None,
) -> DataLoader:
    order = _rank_first_use_order(plan, rank)
    indexed = _IndexedFirstUseDataset(
        dataset,
        order,
        variant_contract=variant_contract,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kwargs: dict[str, object] = {
        "batch_size": None,
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": _identity_collate,
        "pin_memory": pin_memory,
        "worker_init_fn": _context4_worker_init,
        "generator": generator,
    }
    if workers:
        kwargs.update(
            {
                "prefetch_factor": prefetch,
                "persistent_workers": True,
            }
        )
    return DataLoader(indexed, **kwargs)


def _rank_update_batches(
    *,
    rows: Sequence[object],
    plan: RankOriginPlan,
    rank: int,
    update_index: int,
    q_goal_views_by_trajectory: Mapping[int, Sequence[tuple[int, int]]],
    horizon: int,
    microbatch_per_rank: int,
    variant_contract: VariantContract | None = None,
) -> list[JointBatch]:
    contract = _contract_or_full(variant_contract)
    pending: list[JointBatch] = []
    pending_origins = 0
    result: list[JointBatch] = []

    def flush() -> None:
        nonlocal pending, pending_origins
        if pending:
            result.append(_concatenate_joint_batches(pending))
            pending = []
            pending_origins = 0

    for materialized in rows:
        trajectory_index = getattr(materialized, "trajectory_index", None)
        start = getattr(materialized, "start", None)
        stop = getattr(materialized, "stop", None)
        item = getattr(materialized, "trajectory_item", None)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (trajectory_index, start, stop)):
            raise TypeError("materialized origin slice metadata is invalid")
        item = _prepare_for_variant(item, contract)
        descriptor = plan.descriptors[trajectory_index]
        cursor = start
        while cursor < stop:
            room = microbatch_per_rank - pending_origins
            take = min(room, stop - cursor)
            piece_stop = cursor + take
            origins = tuple(range(cursor, piece_stop))
            pairs = tuple(
                (origin, goal)
                for origin, goal in q_goal_views_by_trajectory.get(trajectory_index, ())
                if cursor <= origin < piece_stop
            )
            if {origin for origin, _goal in pairs} != set(origins):
                raise ValueError("every factual origin must have at least one Q goal view")
            terminal = plan.terminal_for(trajectory_index)
            include_terminal = (
                terminal.rank == rank
                and terminal.update_index == update_index
                and not terminal.side_row
                and piece_stop == descriptor.origin_count
            )
            piece = materialize_joint_batch(
                item,
                origin_indices=torch.tensor(origins, dtype=torch.int64),
                q_origin_indices=torch.tensor([origin for origin, _goal in pairs], dtype=torch.int64),
                goal_indices=torch.tensor([goal for _origin, goal in pairs], dtype=torch.int64),
                horizon=horizon,
                context_size=4,
                include_terminal_stop=include_terminal,
                loss_weights=contract.loss_weights,
            )
            piece = _apply_context_variant(piece, contract)
            piece = replace(piece, q_sample_keys=tuple(
                json.dumps({"trajectory_key": descriptor.trajectory_key.hex(),
                            "origin": origin, "goal": goal},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                for origin, goal in pairs
            ))
            pending.append(piece)
            pending_origins += take
            cursor = piece_stop
            if pending_origins == microbatch_per_rank:
                flush()
    flush()
    return result


def _model_device(model: nn.Module) -> torch.device:
    parameter = next((value for value in model.parameters() if value.requires_grad), None)
    if parameter is None:
        raise ValueError("runner model has no trainable parameters")
    return parameter.device


def _checkpoint_module(model: nn.Module) -> nn.Module:
    value = getattr(model, "module", model)
    return getattr(value, "joint_model", value)


def _resource_record(rank: int, device: torch.device, cpu_seconds: float) -> Mapping[str, object]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if device.type != "cuda":
        return {
            "rank": rank,
            "gpu_uuid": "CPU_ONLY",
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
            "gpu_utilization_percent": 0,
            "cpu_affinity": affinity,
            "cpu_seconds": cpu_seconds,
        }
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    try:
        utilization = int(torch.cuda.utilization(device))
    except (AttributeError, RuntimeError, NotImplementedError):
        utilization = 0
    return {
        "rank": rank,
        "gpu_uuid": str(uuid) if uuid is not None else f"CUDA_DEVICE_{device.index}",
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu_utilization_percent": utilization,
        "cpu_affinity": affinity,
        "cpu_seconds": cpu_seconds,
    }


def _gather_rng_states(*, rank: int, world_size: int, device: torch.device) -> tuple[Mapping[str, object], ...]:
    local: Mapping[str, object] = {
        "rank": rank,
        "torch_rng_state": torch.random.get_rng_state().cpu(),
        "cuda_rng_state": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
    }
    if world_size == 1:
        return (local,)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("distributed RNG capture requires an initialized process group")
    gathered: list[object | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, local)
    if any(not isinstance(value, Mapping) for value in gathered):
        raise RuntimeError("distributed RNG capture returned an invalid ledger")
    ordered = tuple(gathered)  # type: ignore[arg-type]
    if tuple(value.get("rank") for value in ordered) != tuple(range(world_size)):
        raise RuntimeError("distributed RNG capture rank order is invalid")
    return ordered


@dataclass(frozen=True)
class QGradientEpochSummary:
    epoch: int
    updates: int
    window: int
    first_median: float
    last_median: float
    epoch_median: float
    within_epoch_ratio: float
    sequence_sha256: str


def _float64_sample_median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Q gradient sequence is empty")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return 0.5 * ordered[midpoint - 1] + 0.5 * ordered[midpoint]


def _positive_gradient_ratio(last: float, first: float) -> float:
    if not math.isfinite(first) or not math.isfinite(last):
        raise FloatingPointError("Q gradient ratio inputs must be finite")
    if first < 0.0 or last < 0.0:
        raise ValueError("Q gradient ratio inputs cannot be negative")
    if first > 0.0:
        return last / first
    if last == 0.0:
        return 1.0
    return math.inf


def summarize_q_gradient_epoch(
    *,
    epoch: int,
    q_pre_clip_grad_norms: Sequence[float],
) -> QGradientEpochSummary:
    """Freeze the protocol's float64 post-DDP/pre-clip Q statistic."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("Q gradient epoch must be a positive integer")
    if not isinstance(q_pre_clip_grad_norms, Sequence):
        raise TypeError("Q gradient sequence is missing")
    raw = tuple(q_pre_clip_grad_norms)
    if not raw:
        raise ValueError("Q gradient sequence is empty")
    values: list[float] = []
    digest = hashlib.sha256()
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Q gradient sequence must contain numeric values")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise FloatingPointError("Q gradient sequence must be finite")
        if resolved < 0.0:
            raise ValueError("Q gradient sequence cannot contain a negative norm")
        values.append(resolved)
        digest.update(struct.pack("<d", resolved))
    updates = len(values)
    window = max(1, (updates + 3) // 4)
    first_median = _float64_sample_median(values[:window])
    last_median = _float64_sample_median(values[-window:])
    epoch_median = _float64_sample_median(values)
    return QGradientEpochSummary(
        epoch=epoch,
        updates=updates,
        window=window,
        first_median=first_median,
        last_median=last_median,
        epoch_median=epoch_median,
        within_epoch_ratio=_positive_gradient_ratio(
            last_median,
            first_median,
        ),
        sequence_sha256=digest.hexdigest(),
    )


def _validated_q_gradient_summary(value: object) -> QGradientEpochSummary:
    if not isinstance(value, QGradientEpochSummary):
        raise TypeError("rank Q gradient summary has an invalid schema")
    if (
        isinstance(value.epoch, bool)
        or not isinstance(value.epoch, int)
        or value.epoch not in (1, 2)
    ):
        raise ValueError("early Q gradient gate only accepts epoch 1 or 2")
    if (
        isinstance(value.updates, bool)
        or not isinstance(value.updates, int)
        or value.updates < 1
        or value.window != max(1, (value.updates + 3) // 4)
    ):
        raise ValueError("rank Q gradient summary update/window count is invalid")
    for name in ("first_median", "last_median", "epoch_median"):
        number = getattr(value, name)
        if not math.isfinite(number) or number < 0.0:
            raise FloatingPointError(f"rank Q gradient summary {name} is invalid")
    expected_ratio = _positive_gradient_ratio(
        value.last_median,
        value.first_median,
    )
    if value.within_epoch_ratio != expected_ratio:
        raise ValueError("rank Q gradient summary ratio is inconsistent")
    if _SHA256.fullmatch(value.sequence_sha256) is None:
        raise ValueError("rank Q gradient sequence digest is invalid")
    return value


def _json_ratio(value: float) -> tuple[float | None, str]:
    if math.isfinite(value):
        return value, "finite"
    if value == math.inf:
        return None, "positive_infinity"
    raise FloatingPointError("early Q gradient ratio is invalid")


def _ratio_from_json(value: object, state: object, *, field_name: str) -> float:
    if state == "finite":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{field_name} finite value is invalid")
        return float(value)
    if state == "positive_infinity":
        if value is not None:
            raise ValueError(f"{field_name} positive infinity must use null JSON value")
        return math.inf
    raise ValueError(f"{field_name} state is invalid")


def _q_gradient_summary_record(
    value: QGradientEpochSummary,
    *,
    rank: int,
) -> Mapping[str, object]:
    ratio, ratio_state = _json_ratio(value.within_epoch_ratio)
    return {
        "rank": rank,
        "epoch": value.epoch,
        "updates": value.updates,
        "window": value.window,
        "first_median": value.first_median,
        "last_median": value.last_median,
        "epoch_median": value.epoch_median,
        "within_epoch_ratio": ratio,
        "within_epoch_ratio_state": ratio_state,
        "sequence_sha256": value.sequence_sha256,
    }


def _validated_gate_identity(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("early Q gradient gate identity must be a mapping")
    expected_keys = {
        "training_contract_id",
        "optimization_protocol_sha256",
        "plan_sha256",
        "config_sha256",
        "code_sha256",
        "next_batch_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("early Q gradient gate identity fields are invalid")
    contract = _validated_training_contract({
        "id": value.get("training_contract_id"),
        "optimization_protocol_sha256": value.get("optimization_protocol_sha256"),
    })
    return {
        "training_contract_id": contract["id"],
        "optimization_protocol_sha256": contract["optimization_protocol_sha256"],
        **{
            name: _require_sha(value.get(name), name)
            for name in (
                "plan_sha256",
                "config_sha256",
                "code_sha256",
                "next_batch_sha256",
            )
        },
    }


def synchronize_q_gradient_early_gate(
    *,
    local_summary: QGradientEpochSummary,
    epoch1_median: float | None,
    rank: int,
    world_size: int,
    output_root: str | os.PathLike[str],
    identity: Mapping[str, str],
) -> Mapping[str, object]:
    """Produce one rank-0 E1/E2 decision and broadcast success or failure."""

    local = _validated_q_gradient_summary(local_summary)
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("early Q gradient gate rank/world is invalid")
    resolved_identity = _validated_gate_identity(identity)
    if local.epoch == 1:
        if epoch1_median is not None:
            raise ValueError("epoch 1 gate must not receive an earlier epoch median")
        resolved_epoch1_median = None
    else:
        if (
            isinstance(epoch1_median, bool)
            or not isinstance(epoch1_median, (int, float))
            or not math.isfinite(float(epoch1_median))
            or float(epoch1_median) < 0.0
        ):
            raise ValueError("epoch 2 gate requires the canonical epoch 1 median")
        resolved_epoch1_median = float(epoch1_median)

    if world_size == 1:
        gathered: list[object] = [local]
    else:
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() != rank
            or torch.distributed.get_world_size() != world_size
        ):
            raise RuntimeError("early Q gradient gate distributed runtime mismatch")
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, local)

    package: list[object | None] = [None]
    if rank == 0:
        try:
            summaries = tuple(_validated_q_gradient_summary(value) for value in gathered)
            reference = summaries[0]
            if any(
                value.epoch != reference.epoch
                or value.updates != reference.updates
                or value.window != reference.window
                for value in summaries
            ):
                raise ValueError("rank Q gradient summary count does not agree")
            exact_digest = all(
                value.sequence_sha256 == reference.sequence_sha256
                for value in summaries
            )
            median_names = ("first_median", "last_median", "epoch_median")
            medians_agree = all(
                math.isclose(
                    getattr(value, name),
                    getattr(reference, name),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-12,
                )
                for value in summaries[1:]
                for name in median_names
            )
            if not medians_agree:
                raise ValueError("rank Q gradient medians do not agree")
            agreement_mode = "exact_digest" if exact_digest else "tolerant_medians"
            within_ratio = reference.within_epoch_ratio
            cross_ratio = (
                None
                if reference.epoch == 1
                else _positive_gradient_ratio(
                    reference.epoch_median,
                    resolved_epoch1_median,
                )
            )
            stop = within_ratio >= 10.0 or (
                cross_ratio is not None and cross_ratio >= 10.0
            )
            within_json, within_state = _json_ratio(within_ratio)
            if cross_ratio is None:
                cross_json: float | None = None
                cross_state = "not_applicable"
            else:
                cross_json, cross_state = _json_ratio(cross_ratio)
            receipt: Mapping[str, object] = {
                "schema": "J2J_CONTEXT4_Q_GRADIENT_EARLY_GATE_V1",
                "epoch": reference.epoch,
                "status": "STOPPED_EARLY" if stop else "CONTINUE",
                "threshold": 10.0,
                "comparison": "stop_if_any_ratio_greater_than_or_equal",
                "identity": dict(resolved_identity),
                "epoch_median": reference.epoch_median,
                "within_epoch_ratio": within_json,
                "within_epoch_ratio_state": within_state,
                "cross_epoch_ratio": cross_json,
                "cross_epoch_ratio_state": cross_state,
                "rank_agreement": {
                    "mode": agreement_mode,
                    "world_size": world_size,
                    "rtol": 1.0e-6,
                    "atol": 1.0e-12,
                },
                "rank_summaries": [
                    _q_gradient_summary_record(value, rank=index)
                    for index, value in enumerate(summaries)
                ],
            }
            destination = (
                Path(output_root)
                / "data"
                / "training_metrics"
                / "validation"
                / f"early_gate_epoch_{reference.epoch:04d}.json"
            )
            _write_atomic_json(destination, receipt)
            package[0] = receipt
        except Exception as exc:
            package[0] = {
                "schema": "J2J_CONTEXT4_Q_GRADIENT_EARLY_GATE_FAILURE_V1",
                "epoch": local.epoch,
                "status": "FAILED_CLOSED",
                "identity": dict(resolved_identity),
                "exception_type": type(exc).__name__,
                "reason": str(exc),
            }

    if world_size > 1:
        torch.distributed.broadcast_object_list(package, src=0)
    observed = package[0]
    if not isinstance(observed, Mapping):
        raise RuntimeError("early Q gradient gate broadcast package is invalid")
    if observed.get("status") == "FAILED_CLOSED":
        raise RuntimeError(
            "early Q gradient gate failed closed: "
            f"{observed.get('exception_type')}: {observed.get('reason')}"
        )
    if (
        observed.get("epoch") != local.epoch
        or observed.get("identity") != resolved_identity
        or observed.get("status") not in {"CONTINUE", "STOPPED_EARLY"}
    ):
        raise RuntimeError("early Q gradient gate broadcast decision is invalid")
    return observed


def _load_canonical_q_gradient_gate(
    *,
    output_root: str | os.PathLike[str],
    epoch: int,
    world_size: int,
    identity: Mapping[str, str],
    epoch1_median: float | None = None,
) -> Mapping[str, object]:
    """Admit an already-completed early gate before exact-resume advances."""

    if epoch not in (1, 2):
        raise ValueError("canonical early gate epoch must be 1 or 2")
    resolved_identity = _validated_gate_identity(identity)
    path = (
        Path(output_root)
        / "data"
        / "training_metrics"
        / "validation"
        / f"early_gate_epoch_{epoch:04d}.json"
    )
    if not path.is_file():
        raise ValueError(f"canonical early gate E{epoch} receipt is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"canonical early gate E{epoch} receipt is unreadable"
        ) from exc
    try:
        if not isinstance(value, Mapping):
            raise ValueError("receipt is not a mapping")
        expected_fields = {
            "schema",
            "epoch",
            "status",
            "threshold",
            "comparison",
            "identity",
            "epoch_median",
            "within_epoch_ratio",
            "within_epoch_ratio_state",
            "cross_epoch_ratio",
            "cross_epoch_ratio_state",
            "rank_agreement",
            "rank_summaries",
        }
        if set(value) != expected_fields:
            raise ValueError("receipt fields are incomplete or ambiguous")
        if value.get("schema") != "J2J_CONTEXT4_Q_GRADIENT_EARLY_GATE_V1":
            raise ValueError("schema is invalid")
        if value.get("epoch") != epoch or value.get("threshold") != 10.0:
            raise ValueError("epoch/threshold boundary is invalid")
        if value.get("comparison") != "stop_if_any_ratio_greater_than_or_equal":
            raise ValueError("comparison rule is invalid")
        if value.get("identity") != resolved_identity:
            raise ValueError("identity mismatch")

        raw_summaries = value.get("rank_summaries")
        if not isinstance(raw_summaries, list) or len(raw_summaries) != world_size:
            raise ValueError("rank summary count is invalid")
        summaries: list[QGradientEpochSummary] = []
        summary_fields = {
            "rank",
            "epoch",
            "updates",
            "window",
            "first_median",
            "last_median",
            "epoch_median",
            "within_epoch_ratio",
            "within_epoch_ratio_state",
            "sequence_sha256",
        }
        for expected_rank, raw_summary in enumerate(raw_summaries):
            if not isinstance(raw_summary, Mapping) or set(raw_summary) != summary_fields:
                raise ValueError("rank summary schema is invalid")
            if raw_summary.get("rank") != expected_rank:
                raise ValueError("rank summary order is invalid")
            summary = QGradientEpochSummary(
                epoch=raw_summary.get("epoch"),  # type: ignore[arg-type]
                updates=raw_summary.get("updates"),  # type: ignore[arg-type]
                window=raw_summary.get("window"),  # type: ignore[arg-type]
                first_median=raw_summary.get("first_median"),  # type: ignore[arg-type]
                last_median=raw_summary.get("last_median"),  # type: ignore[arg-type]
                epoch_median=raw_summary.get("epoch_median"),  # type: ignore[arg-type]
                within_epoch_ratio=_ratio_from_json(
                    raw_summary.get("within_epoch_ratio"),
                    raw_summary.get("within_epoch_ratio_state"),
                    field_name="rank within-epoch ratio",
                ),
                sequence_sha256=raw_summary.get("sequence_sha256"),  # type: ignore[arg-type]
            )
            summary = _validated_q_gradient_summary(summary)
            if summary.epoch != epoch:
                raise ValueError("rank summary epoch mismatch")
            summaries.append(summary)

        reference = summaries[0]
        if any(
            summary.updates != reference.updates
            or summary.window != reference.window
            for summary in summaries[1:]
        ):
            raise ValueError("rank update/window count does not agree")
        medians_agree = all(
            math.isclose(
                getattr(summary, name),
                getattr(reference, name),
                rel_tol=1.0e-6,
                abs_tol=1.0e-12,
            )
            for summary in summaries[1:]
            for name in ("first_median", "last_median", "epoch_median")
        )
        if not medians_agree:
            raise ValueError("rank medians do not agree")
        exact_digest = all(
            summary.sequence_sha256 == reference.sequence_sha256
            for summary in summaries[1:]
        )
        expected_agreement_mode = (
            "exact_digest" if exact_digest else "tolerant_medians"
        )
        agreement = value.get("rank_agreement")
        if (
            not isinstance(agreement, Mapping)
            or set(agreement) != {"mode", "world_size", "rtol", "atol"}
            or agreement.get("world_size") != world_size
            or agreement.get("mode") != expected_agreement_mode
            or agreement.get("rtol") != 1.0e-6
            or agreement.get("atol") != 1.0e-12
        ):
            raise ValueError("rank agreement is invalid")

        top_epoch_median = value.get("epoch_median")
        if (
            isinstance(top_epoch_median, bool)
            or not isinstance(top_epoch_median, (int, float))
            or not math.isfinite(float(top_epoch_median))
            or float(top_epoch_median) < 0.0
            or float(top_epoch_median) != reference.epoch_median
        ):
            raise ValueError("top-level epoch median is inconsistent")
        stored_within = _ratio_from_json(
            value.get("within_epoch_ratio"),
            value.get("within_epoch_ratio_state"),
            field_name="top-level within-epoch ratio",
        )
        if stored_within != reference.within_epoch_ratio:
            raise ValueError("within-epoch ratio is inconsistent")

        if epoch == 1:
            if epoch1_median is not None:
                raise ValueError("E1 must not receive a prior epoch median")
            if (
                value.get("cross_epoch_ratio") is not None
                or value.get("cross_epoch_ratio_state") != "not_applicable"
            ):
                raise ValueError("E1 cross-epoch ratio is invalid")
            cross_ratio: float | None = None
        else:
            if (
                isinstance(epoch1_median, bool)
                or not isinstance(epoch1_median, (int, float))
                or not math.isfinite(float(epoch1_median))
                or float(epoch1_median) < 0.0
            ):
                raise ValueError("E2 requires the canonical E1 median")
            cross_ratio = _positive_gradient_ratio(
                reference.epoch_median,
                float(epoch1_median),
            )
            stored_cross = _ratio_from_json(
                value.get("cross_epoch_ratio"),
                value.get("cross_epoch_ratio_state"),
                field_name="cross-epoch ratio",
            )
            if stored_cross != cross_ratio:
                raise ValueError("cross-epoch ratio is inconsistent")

        stop = reference.within_epoch_ratio >= 10.0 or (
            cross_ratio is not None and cross_ratio >= 10.0
        )
        expected_status = "STOPPED_EARLY" if stop else "CONTINUE"
        if value.get("status") != expected_status:
            raise ValueError("status contradicts the rederived gate decision")
        if expected_status != "CONTINUE":
            raise ValueError("did not authorize continuation")
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise ValueError(
            f"canonical early gate E{epoch} evidence is invalid: {exc}"
        ) from exc
    return value


@dataclass(frozen=True)
class EpochRunReceipt:
    epoch: int
    successful_updates: int
    source_reads: int
    checkpoint_path: str
    checkpoint_sha256: str
    numerators: Mapping[str, float]
    denominators: Mapping[str, int]
    losses: Mapping[str, float]
    q_pre_clip_grad_norms: tuple[float, ...] = ()


@dataclass(frozen=True)
class QualificationRunReceipt:
    """A real committed training prefix, never an epoch or resume checkpoint."""
    epoch: int
    successful_updates: int
    source_reads: int
    numerators: Mapping[str, float]
    denominators: Mapping[str, int]
    losses: Mapping[str, float]
    q_pre_clip_grad_norms: tuple[float, ...]


def _qualification_limit(limit, *, update_count, epoch, successful_updates, world_size):
    from j2j.context4_forensics import _all_values
    if world_size > 1 and not torch.distributed.is_initialized():
        if limit is None:
            return  # Unchanged non-distributed orchestration/default path.
        raise RuntimeError("distributed qualification requires an initialized process group")
    values = _all_values(limit, world_size)
    if any(value != values[0] or type(value) is not type(values[0]) for value in values):
        raise ValueError("qualification budget differs across ranks")
    if limit is None:
        return
    if type(limit) is not int or not 0 < limit < update_count:
        raise ValueError("qualification limit must be a positive strict epoch prefix")
    if epoch != 1 or successful_updates != 0:
        raise ValueError("qualification requires constructor-fresh epoch one")


@dataclass(frozen=True)
class ValidationReceipt:
    numerators: Mapping[str, float]
    denominators: Mapping[str, int]
    losses: Mapping[str, float]
    source_reads: int
    stochastic_statistics: Mapping[str, float] | None = None
    model_probe: Mapping[str, object] | None = None


def run_context4_epoch(
    *,
    dataset: object,
    descriptors: Sequence[TrajectoryDescriptor],
    q_goal_views_by_trajectory: Mapping[int, Sequence[tuple[int, int]]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    output_root: str | os.PathLike[str],
    rank: int,
    world_size: int,
    effective_global_batch: int,
    microbatch_per_rank: int,
    accumulation_steps: int,
    horizon: int,
    max_grad_norm: float | None = None,
    gradient_clip_norms: Mapping[str, float] | None = None,
    optimizer_partition: OptimizerPartitionReceipt | None = None,
    training_contract: Mapping[str, str] | None = None,
    initialization_identity: Mapping[str, object] | None = None,
    epoch: int,
    successful_updates: int,
    checkpoint_identity: Mapping[str, str],
    joint_numerator_fn: Callable[[nn.Module, JointBatch | None], object] | None = None,
    diagnostic_interval_updates: int | None = None,
    model_probe_config: Mapping[str, int] | None = None,
    anomaly_capture=None,
    variant_contract: VariantContract | None = None,
    qualification_update_limit: int | None = None,
) -> EpochRunReceipt | QualificationRunReceipt:
    """Run one exact trajectory pass without rereading a trajectory payload."""

    if model_probe_config is not None:
        from j2j.context4_probes import validate_probe_config
        model_probe_config = validate_probe_config(model_probe_config)
        if _contract_or_full(variant_contract).identity not in STOCHASTIC_IDENTITIES:
            raise ValueError("bounded model probes require the stochastic scientific family")

    scratch_epoch = _scratch_metadata_mode(
        gradient_clip_norms,
        optimizer_partition,
        training_contract,
        initialization_identity,
    )
    if scratch_epoch:
        if max_grad_norm is not None:
            raise ValueError("scratch epoch must not mix the legacy global clip")
        if not isinstance(optimizer_partition, OptimizerPartitionReceipt):
            raise TypeError("scratch epoch optimizer partition is invalid")
        if (
            optimizer_partition.scheme != SCRATCH_OPTIMIZER_SCHEME
            or optimizer_partition.ordered_group_names != ("q", "agf")
            or not optimizer_partition.partition_validated
            or tuple(optimizer_partition.groups) != ("q", "agf")
        ):
            raise ValueError("scratch epoch optimizer partition identity is invalid")
        _validated_training_contract(training_contract)
        _validated_initialization_identity(initialization_identity)
        if not isinstance(gradient_clip_norms, Mapping) or set(
            gradient_clip_norms
        ) != {"q", "agf"}:
            raise ValueError("scratch epoch clip domains must be exactly q and agf")
        for name in ("q", "agf"):
            clip_norm = float(gradient_clip_norms[name])
            if clip_norm != float(optimizer_partition.groups[name].clip_norm):
                raise ValueError(
                    "scratch epoch clip domain disagrees with optimizer partition"
                )
    elif max_grad_norm is None:
        raise ValueError("legacy epoch requires one global gradient clip")

    contract = _contract_or_full(variant_contract)
    resolved_variant = variant_identity_dict(contract.identity)
    descriptor_tuple = tuple(descriptors)
    plan = build_rank_origin_plan(
        descriptor_tuple,
        world_size=world_size,
        effective_batch=effective_global_batch,
    )
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside the configured world")
    _qualification_limit(qualification_update_limit, update_count=plan.update_count,
                         epoch=epoch, successful_updates=successful_updates, world_size=world_size)
    if contract.identity in RAW_IDENTITIES:
        origins_per_rank = (effective_global_batch + world_size - 1) // world_size
        if accumulation_steps != (origins_per_rank + microbatch_per_rank - 1) // microbatch_per_rank:
            raise ValueError("raw accumulation must cover the planned per-rank origins exactly")
    elif microbatch_per_rank * accumulation_steps * world_size != effective_global_batch:
        raise ValueError("microbatch, accumulation and world must preserve effective batch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError("epoch must be a positive integer")
    joint = _checkpoint_module(model)
    stochastic_proposal = getattr(joint, "proposal", None)
    if bool(getattr(stochastic_proposal, "is_stochastic", False)):
        stochastic_proposal.sampling_namespace = f"train/epoch-{epoch}"
    device = _model_device(model)
    store = TrainingMetricsStore(output_root)
    materialized_iterator = iter_materialized_rank_slices(
        dataset,
        plan,
        rank,
        prepare_trajectory=lambda item: _prepare_for_variant(item, contract),
    )
    seen_trajectories: set[int] = set()
    epoch_source_reads = 0
    completed_updates = successful_updates
    anomaly_context = None if anomaly_capture is None else {
        "epoch": epoch, "checkpoint_identity": dict(checkpoint_identity),
        "variant_identity": resolved_variant, "epoch_plan_sha256": _plan_identity(plan),
        "sampling_namespace": getattr(stochastic_proposal, "sampling_namespace", ""),
        "global_seed": getattr(stochastic_proposal, "global_seed", None),
    }
    epoch_numerators = {name: 0.0 for name in _BRANCHES}
    epoch_denominators = {name: 0 for name in _BRANCHES}
    q_pre_clip_grad_norms: list[float] = []
    zero_motion = tuple(
        sorted(
            (
                index
                for index, descriptor in enumerate(descriptor_tuple)
                if descriptor.origin_count == 0 and plan.owner_for(index) == rank
            ),
            key=lambda index: (descriptor_tuple[index].trajectory_key, index),
        )
    )

    for update_index in range(plan.update_count):
        update_started = time.perf_counter()
        cpu_started = time.process_time()
        data_started = time.perf_counter()
        actual_index, rows = next(materialized_iterator)
        if actual_index != update_index:
            raise RuntimeError("materialized iterator update order drifted from the plan")
        update_reads = 0
        for row in rows:
            trajectory_index = row.trajectory_index
            if trajectory_index not in seen_trajectories:
                seen_trajectories.add(trajectory_index)
                update_reads += 1
        batches = _rank_update_batches(
            rows=rows,
            plan=plan,
            rank=rank,
            update_index=update_index,
            q_goal_views_by_trajectory=q_goal_views_by_trajectory,
            horizon=horizon,
            microbatch_per_rank=microbatch_per_rank,
            variant_contract=contract,
        )
        if update_index == 0 and zero_motion:
            terminal_batches: list[JointBatch] = []
            for trajectory_index in zero_motion:
                descriptor = descriptor_tuple[trajectory_index]
                item = _prepare_for_variant(
                    dataset[descriptor.dataset_index], contract
                )
                update_reads += 1
                seen_trajectories.add(trajectory_index)
                terminal_batches.append(
                    materialize_joint_batch(
                        item,
                        origin_indices=torch.empty(0, dtype=torch.int64),
                        q_origin_indices=torch.empty(0, dtype=torch.int64),
                        goal_indices=torch.empty(0, dtype=torch.int64),
                        horizon=horizon,
                        context_size=4,
                        include_terminal_stop=True,
                        loss_weights=contract.loss_weights,
                    )
                )
            terminal_batches = [
                _apply_context_variant(batch, contract) for batch in terminal_batches
            ]
            terminal_batch = _concatenate_joint_batches(terminal_batches)
            if len(batches) < accumulation_steps:
                batches.append(terminal_batch)
            elif batches:
                batches[0] = _concatenate_joint_batches((batches[0], terminal_batch))
            else:  # pragma: no cover - a global update always has an origin owner
                batches.append(terminal_batch)
        if len(batches) > accumulation_steps:
            raise ValueError("rank-local owner work exceeds configured accumulation capacity")
        slot_count = _distributed_max_int(len(batches), world_size=world_size, device=device)
        if slot_count < 1 or slot_count > accumulation_steps:
            raise ValueError("global physical micro count is outside the accumulation contract")
        moved: list[JointBatch | None] = [
            _move_joint_batch(batch, device) for batch in batches
        ]
        moved.extend([None] * (slot_count - len(moved)))
        data_wait_seconds = max(time.perf_counter() - data_started, 0.0)
        local_denominators = {name: 0 for name in ("q", "f", "g_local", "g_goal", "qg")}
        for batch in moved:
            if batch is None:
                continue
            for name, value in _batch_denominators(batch).items():
                local_denominators[name] += value
        global_denominators = {
            name: _distributed_sum_int(value, world_size=world_size, device=device)
            for name, value in local_denominators.items()
        }

        def calculate(batch: JointBatch | None):
            if joint_numerator_fn is None:
                return model(batch)
            return joint_numerator_fn(model, batch)

        # Probe the parameters before this update. Any diagnostic failure
        # occurs before Adam/scheduler advance and cannot strand a committed
        # training update without its canonical loss/gradient ledger.
        probe = None
        probe_seconds = 0.0
        probe_parameter_update = completed_updates
        if (model_probe_config is not None and diagnostic_interval_updates is not None
            and (completed_updates + 1) % diagnostic_interval_updates == 0):
            from j2j.context4_probes import bounded_model_probe
            probe_batch = next((batch for batch in moved if batch is not None and
                                (batch.current_grid.shape[0] or batch.terminal_grid.shape[0])), None)
            if probe_batch is not None:
                probe_started = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda" and getattr(getattr(model, "module", model), "precision", "fp32") == "bf16-mixed"):
                    probe = bounded_model_probe(joint, probe_batch, **model_probe_config,
                                                sampling_namespace="train-probe/v1")
                probe_seconds = max(time.perf_counter() - probe_started, 0.0)

        update = run_global_update(
            model=model,
            microbatches=moved,
            joint_numerator_fn=calculate,
            global_denominators=global_denominators,
            optimizer=optimizer,
            scheduler=scheduler,
            world_size=world_size,
            accumulation_steps=accumulation_steps,
            max_grad_norm=max_grad_norm,
            gradient_clip_norms=gradient_clip_norms,
            successful_updates=completed_updates,
            diagnostic_interval_updates=diagnostic_interval_updates,
            loss_weights=contract.loss_weights,
            anomaly_capture=anomaly_capture,
            anomaly_context=anomaly_context,
        )
        if dict(update.local_denominators) != local_denominators:
            raise RuntimeError("forward denominators disagree with the pre-backward owner ledger")
        stochastic_sums = _global_stochastic_statistics(
            update.local_stochastic_statistics, world_size=world_size, device=device)
        stochastic_record = _stochastic_probability_record(
            stochastic_sums, kl_beta=getattr(joint, "kl_beta", 0.05),
            sampling_namespace=getattr(stochastic_proposal, "sampling_namespace", ""))
        global_numerators = {
            name: _distributed_sum_float(value, world_size=world_size, device=device)
            for name, value in update.local_numerators.items()
        }
        losses = {
            name: (
                global_numerators[name] / global_denominators[name]
                if global_denominators[name]
                else 0.0
            )
            for name in global_denominators
        }
        losses["total"] = weighted_joint_loss_value(
            losses,
            weights=contract.loss_weights,
        )
        global_reads = _distributed_sum_int(update_reads, world_size=world_size, device=device)
        epoch_source_reads += global_reads
        for name in _BRANCHES:
            epoch_numerators[name] += global_numerators[name]
            epoch_denominators[name] += global_denominators[name]
        completed_updates += 1
        elapsed = max(time.perf_counter() - update_started, 1e-12)
        resource = dict(
            _resource_record(
                rank,
                device,
                max(time.process_time() - cpu_started, 0.0),
            )
        )
        resource["successful_update"] = completed_updates
        resource["variant_identity"] = resolved_variant
        store.write_rank_resource(resource)
        if update.diagnostics is not None:
            store.write_rank_diagnostic(
                {
                    "rank": rank,
                    "successful_update": completed_updates,
                    "variant_identity": resolved_variant,
                    **dict(update.diagnostics),
                    **({"model_probe": probe, "model_probe_seconds": probe_seconds,
                        "model_probe_parameter_state": {"kind": "pre_update", "successful_update": probe_parameter_update}}
                       if probe is not None else {}),
                }
            )
        optimization_telemetry = build_update_optimization_telemetry(
            optimizer=optimizer,
            update=update,
            optimizer_partition=optimizer_partition,
        )
        if scratch_epoch:
            grad = _mapping(
                optimization_telemetry.get("grad"),
                "scratch optimization telemetry grad",
            )
            clip_domains = _mapping(
                grad.get("clip_domains"),
                "scratch optimization telemetry clip_domains",
            )
            q_domain = _mapping(
                clip_domains.get("q"),
                "scratch optimization telemetry Q domain",
            )
            q_pre_clip = q_domain.get("pre_clip")
            if (
                isinstance(q_pre_clip, bool)
                or not isinstance(q_pre_clip, (int, float))
                or not math.isfinite(float(q_pre_clip))
                or float(q_pre_clip) < 0.0
            ):
                raise FloatingPointError(
                    "scratch epoch Q pre-clip gradient telemetry is invalid"
                )
            q_pre_clip_grad_norms.append(float(q_pre_clip))
        if rank == 0:
            store.write_update(
                {
                    "successful_update": completed_updates,
                    "variant_identity": resolved_variant,
                    "loss": losses,
                    "numerator": global_numerators,
                    "denominator": global_denominators,
                    **({"stochastic_q": stochastic_record} if stochastic_record is not None else {}),
                    "grad": optimization_telemetry["grad"],
                    "lr": optimization_telemetry["lr"],
                    "timing": {
                        "updates_per_second": 1.0 / elapsed,
                        "data_wait_seconds": data_wait_seconds,
                        "source_reads": global_reads,
                    },
                    "resources": resource,
                }
            )

        if qualification_update_limit is not None and completed_updates == qualification_update_limit:
            prefix_losses = {name: epoch_numerators[name] / epoch_denominators[name]
                             if epoch_denominators[name] else 0.0 for name in _BRANCHES}
            prefix_losses["total"] = weighted_joint_loss_value(prefix_losses, weights=contract.loss_weights)
            return QualificationRunReceipt(epoch, completed_updates, epoch_source_reads,
                dict(epoch_numerators), dict(epoch_denominators), prefix_losses, tuple(q_pre_clip_grad_norms))

    checkpoint_path = store.root / "checkpoints" / f"epoch_{epoch:04d}.pt"
    checkpoint_sha = ""
    rng_states = _gather_rng_states(rank=rank, world_size=world_size, device=device)
    if rank == 0:
        checkpoint_sha = save_epoch_checkpoint(
            checkpoint_path,
            model=_checkpoint_module(model),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            successful_updates=completed_updates,
            rng_state_by_rank=rng_states,
            variant_identity=(
                None
                if contract.identity == FULL_IDENTITY
                else resolved_variant
            ),
            training_contract=(training_contract if scratch_epoch else None),
            initialization_identity=(
                initialization_identity if scratch_epoch else None
            ),
            optimizer_partition=(optimizer_partition if scratch_epoch else None),
            **dict(checkpoint_identity),
        )
        store.write_checkpoint(
            {
                "epoch": epoch,
                "successful_updates": completed_updates,
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "variant_identity": resolved_variant,
            }
        )
    if world_size > 1:
        torch.distributed.barrier()
    epoch_losses = {
        name: (
            epoch_numerators[name] / epoch_denominators[name]
            if epoch_denominators[name]
            else 0.0
        )
        for name in _BRANCHES
    }
    epoch_losses["total"] = weighted_joint_loss_value(
        epoch_losses,
        weights=contract.loss_weights,
    )
    return EpochRunReceipt(
        epoch=epoch,
        successful_updates=completed_updates,
        source_reads=epoch_source_reads,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        numerators=dict(epoch_numerators),
        denominators=dict(epoch_denominators),
        losses=epoch_losses,
        q_pre_clip_grad_norms=tuple(q_pre_clip_grad_norms),
    )


def evaluate_context4(
    *,
    dataset: object,
    descriptors: Sequence[TrajectoryDescriptor],
    q_goal_views_by_trajectory: Mapping[int, Sequence[tuple[int, int]]],
    model: nn.Module,
    rank: int,
    world_size: int,
    effective_global_batch: int,
    microbatch_per_rank: int,
    horizon: int,
    variant_contract: VariantContract | None = None,
    model_probe_config: Mapping[str, int] | None = None,
) -> ValidationReceipt:
    """Evaluate the same five objectives on a rank-exclusive factual pass."""

    if model_probe_config is not None:
        from j2j.context4_probes import validate_probe_config
        model_probe_config = validate_probe_config(model_probe_config)
        if _contract_or_full(variant_contract).identity not in STOCHASTIC_IDENTITIES:
            raise ValueError("bounded model probes require the stochastic scientific family")

    contract = _contract_or_full(variant_contract)
    descriptor_tuple = tuple(descriptors)
    plan = build_rank_origin_plan(
        descriptor_tuple,
        world_size=world_size,
        effective_batch=effective_global_batch,
    )
    if not 0 <= rank < world_size:
        raise ValueError("rank is outside the configured world")
    device = _model_device(model)
    candidate = getattr(model, "module", model)
    joint = getattr(candidate, "joint_model", candidate)
    if not isinstance(joint, nn.Module):
        raise TypeError("validation model does not expose the joint Q/A/G/F tree")
    precision = str(getattr(candidate, "precision", "fp32"))
    amp_enabled = precision == "bf16-mixed" and device.type == "cuda"
    was_training = model.training
    stochastic = bool(getattr(joint.proposal, "is_stochastic", False))
    sampling_namespace = getattr(joint.proposal, "sampling_namespace", None)
    cpu_rng = torch.random.get_rng_state() if stochastic else None
    cuda_rng = torch.cuda.get_rng_state(device) if stochastic and device.type == "cuda" else None
    if stochastic:
        joint.proposal.sampling_namespace = "dev/v1"
    local_stochastic = {name: 0.0 for name in _STOCHASTIC_STAT_FIELDS} if stochastic else None
    model.eval()
    local_numerators = {name: 0.0 for name in _BRANCHES}
    local_denominators = {name: 0 for name in _BRANCHES}
    local_reads = 0
    model_probe = None
    seen: set[int] = set()
    materialized_iterator = iter_materialized_rank_slices(
        dataset,
        plan,
        rank,
        prepare_trajectory=lambda item: _prepare_for_variant(item, contract),
    )
    zero_motion = tuple(
        sorted(
            (
                index
                for index, descriptor in enumerate(descriptor_tuple)
                if descriptor.origin_count == 0 and plan.owner_for(index) == rank
            ),
            key=lambda index: (descriptor_tuple[index].trajectory_key, index),
        )
    )
    try:
        with torch.no_grad():
            for update_index in range(plan.update_count):
                actual_index, rows = next(materialized_iterator)
                if actual_index != update_index:
                    raise RuntimeError("validation materialization order drifted from its plan")
                for row in rows:
                    if row.trajectory_index not in seen:
                        seen.add(row.trajectory_index)
                        local_reads += 1
                batches = _rank_update_batches(
                    rows=rows,
                    plan=plan,
                    rank=rank,
                    update_index=update_index,
                    q_goal_views_by_trajectory=q_goal_views_by_trajectory,
                    horizon=horizon,
                    microbatch_per_rank=microbatch_per_rank,
                    variant_contract=contract,
                )
                if update_index == 0 and zero_motion:
                    terminal_batches: list[JointBatch] = []
                    for trajectory_index in zero_motion:
                        descriptor = descriptor_tuple[trajectory_index]
                        item = _prepare_for_variant(
                            dataset[descriptor.dataset_index], contract
                        )
                        seen.add(trajectory_index)
                        local_reads += 1
                        terminal_batches.append(
                            materialize_joint_batch(
                                item,
                                origin_indices=torch.empty(0, dtype=torch.int64),
                                q_origin_indices=torch.empty(0, dtype=torch.int64),
                                goal_indices=torch.empty(0, dtype=torch.int64),
                                horizon=horizon,
                                context_size=4,
                                include_terminal_stop=True,
                                loss_weights=contract.loss_weights,
                            )
                        )
                    batches.append(
                        _concatenate_joint_batches(
                            tuple(
                                _apply_context_variant(batch, contract)
                                for batch in terminal_batches
                            )
                        )
                    )
                for batch in batches:
                    moved = _move_joint_batch(batch, device)
                    if model_probe_config is not None and model_probe is None and (moved.current_grid.shape[0] or moved.terminal_grid.shape[0]):
                        from j2j.context4_probes import bounded_model_probe
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                            model_probe = bounded_model_probe(joint, moved, **model_probe_config,
                                                             sampling_namespace="dev-probe/v1")
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=amp_enabled,
                    ):
                        result = joint_numerators(
                            joint, moved, qg_objective=getattr(candidate, "qg_objective", "marginal"),
                            loss_weights=contract.loss_weights
                        )
                    if local_stochastic is not None:
                        _accumulate_stochastic_statistics(local_stochastic, result.stochastic_statistics)
                    for name in _BRANCHES:
                        numerator = result.numerators[name]
                        if not bool(torch.isfinite(numerator.detach())):
                            raise FloatingPointError(f"validation {name} numerator is non-finite")
                        local_numerators[name] += float(numerator.detach())
                        local_denominators[name] += int(result.denominators[name])
    finally:
        model.train(was_training)
        if stochastic:
            joint.proposal.sampling_namespace = sampling_namespace
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)

    expected_owned = {index for index, owner in enumerate(plan.owners) if owner == rank}
    if seen != expected_owned:
        raise RuntimeError("validation did not materialize every rank-owned trajectory exactly once")
    global_numerators = {
        name: _distributed_sum_float(value, world_size=world_size, device=device)
        for name, value in local_numerators.items()
    }
    global_denominators = {
        name: _distributed_sum_int(value, world_size=world_size, device=device)
        for name, value in local_denominators.items()
    }
    losses = {
        name: (
            global_numerators[name] / global_denominators[name]
            if global_denominators[name]
            else 0.0
        )
        for name in _BRANCHES
    }
    losses["total"] = weighted_joint_loss_value(
        losses,
        weights=contract.loss_weights,
    )
    return ValidationReceipt(
        numerators=global_numerators,
        denominators=global_denominators,
        losses=losses,
        source_reads=_distributed_sum_int(local_reads, world_size=world_size, device=device),
        stochastic_statistics=_global_stochastic_statistics(local_stochastic, world_size=world_size, device=device),
        model_probe=model_probe,
    )


def successful_optimizer_update(
    loss: Tensor,
    parameters: Iterable[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    *,
    max_grad_norm: float,
) -> OptimizerUpdate:
    """Advance optimizer/scheduler only after a finite backward and clip."""

    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("optimizer update needs trainable parameters")
    if not isinstance(loss, Tensor) or loss.numel() != 1:
        raise TypeError("optimizer loss must be a scalar tensor")
    if not math.isfinite(float(max_grad_norm)) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    optimizer.zero_grad(set_to_none=True)
    if not bool(torch.isfinite(loss.detach())):
        return OptimizerUpdate(False, 0.0, 0.0, 0.0, "non-finite loss")
    loss.backward()
    if any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in parameter_tuple
    ):
        optimizer.zero_grad(set_to_none=True)
        return OptimizerUpdate(False, math.inf, math.inf, 0.0, "non-finite gradient")
    pre_clip = _gradient_norm(parameter_tuple)
    torch.nn.utils.clip_grad_norm_(parameter_tuple, max_norm=max_grad_norm)
    post_clip = _gradient_norm(parameter_tuple)
    if not math.isfinite(pre_clip) or not math.isfinite(post_clip):
        optimizer.zero_grad(set_to_none=True)
        return OptimizerUpdate(False, pre_clip, post_clip, 0.0, "non-finite clipped gradient")
    optimizer.step()
    step = getattr(scheduler, "step", None)
    if not callable(step):
        raise TypeError("scheduler must expose step()")
    step()
    scale = 1.0 if pre_clip == 0.0 else min(1.0, max_grad_norm / pre_clip)
    return OptimizerUpdate(True, pre_clip, post_clip, scale)


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    successful_updates: int
    identity: Mapping[str, str]


def _scratch_metadata_mode(*values: object | None) -> bool:
    present = tuple(value is not None for value in values)
    if any(present) and not all(present):
        raise ValueError("scratch checkpoint V2 metadata must be provided all-or-none")
    return all(present)


def _validated_training_contract(value: object) -> dict[str, str]:
    contract = _mapping(value, "scratch training contract")
    protocol_by_id = {
        RAW_TRAINING_CONTRACT_ID: RAW_PROTOCOL_SHA256,
        STOCHASTIC_TRAINING_CONTRACT_ID: STOCHASTIC_PROTOCOL_SHA256,
        STABLE_STOCHASTIC_TRAINING_CONTRACT_ID: STABLE_STOCHASTIC_PROTOCOL_SHA256,
        RECURRENT_STOCHASTIC_TRAINING_CONTRACT_ID: RECURRENT_STOCHASTIC_PROTOCOL_SHA256,
        SCRATCH_TRAINING_CONTRACT_ID: SCRATCH_OPTIMIZATION_PROTOCOL_SHA256,
    }
    contract_id = contract.get("id")
    if contract_id not in protocol_by_id:
        raise ValueError("scratch checkpoint training contract is invalid")
    expected = {
        "id": contract_id,
        "optimization_protocol_sha256": protocol_by_id[contract_id],
    }
    if dict(contract) != expected:
        raise ValueError("scratch checkpoint training contract is invalid")
    return expected


def _validated_initialization_identity(value: object) -> dict[str, object]:
    identity = _mapping(value, "scratch initialization identity")
    required = {
        "mode",
        "schema",
        "global_seed",
        "constructor_state_sha256",
        "imported_trainable_keys",
        "trainable_checkpoint_sources",
    }
    if set(identity) != required:
        raise ValueError("scratch checkpoint initialization identity fields are invalid")
    if identity.get("mode") != SCRATCH_INITIALIZATION_MODE:
        raise ValueError("scratch checkpoint initialization mode is invalid")
    if identity.get("schema") != SCRATCH_INITIALIZATION_SCHEMA:
        raise ValueError("scratch checkpoint initialization schema is invalid")
    if identity.get("global_seed") != 3072:
        raise ValueError("scratch checkpoint initialization seed is invalid")
    constructor_sha = _require_sha(
        identity.get("constructor_state_sha256"), "constructor state"
    )
    if identity.get("imported_trainable_keys") != []:
        raise ValueError("scratch checkpoint must not import trainable keys")
    if identity.get("trainable_checkpoint_sources") != []:
        raise ValueError("scratch checkpoint must not name trainable sources")
    return {
        "mode": SCRATCH_INITIALIZATION_MODE,
        "schema": SCRATCH_INITIALIZATION_SCHEMA,
        "global_seed": 3072,
        "constructor_state_sha256": constructor_sha,
        "imported_trainable_keys": [],
        "trainable_checkpoint_sources": [],
    }


def _optimizer_group_identity(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    optimizer_partition: OptimizerPartitionReceipt,
) -> Mapping[str, object]:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("scratch checkpoint optimizer must be exactly AdamW")
    if not isinstance(optimizer_partition, OptimizerPartitionReceipt):
        raise TypeError("scratch checkpoint requires an optimizer partition receipt")
    if (
        optimizer_partition.scheme != SCRATCH_OPTIMIZER_SCHEME
        or optimizer_partition.ordered_group_names != ("q", "agf")
        or not optimizer_partition.partition_validated
        or tuple(optimizer_partition.groups) != ("q", "agf")
    ):
        raise ValueError("scratch checkpoint optimizer partition is invalid")
    if len(optimizer.param_groups) != 2:
        raise ValueError("scratch checkpoint optimizer must have two groups")

    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    serialized_groups: dict[str, object] = {}
    seen: set[int] = set()
    for index, name in enumerate(("q", "agf")):
        receipt = optimizer_partition.groups[name]
        fqns = tuple(receipt.ordered_fqns)
        if receipt.fqn_sha256 != _canonical_json_sha256(list(fqns)):
            raise ValueError("scratch checkpoint optimizer FQN digest is invalid")
        if any(fqn not in named_parameters for fqn in fqns):
            raise ValueError("scratch checkpoint optimizer FQN is missing from model")
        expected_parameters = tuple(named_parameters[fqn] for fqn in fqns)
        group = optimizer.param_groups[index]
        if set(group) not in {
            _ADAMW_BASE_PARAM_GROUP_KEYS,
            _ADAMW_PARAM_GROUP_KEYS,
        }:
            raise ValueError("scratch checkpoint AdamW group schema drifted")
        if group.get("group_name") != name:
            raise ValueError("scratch checkpoint optimizer group name/order drifted")
        actual_parameters = tuple(group["params"])
        if tuple(map(id, actual_parameters)) != tuple(map(id, expected_parameters)):
            raise ValueError("scratch checkpoint optimizer group FQN order drifted")
        identifiers = {id(parameter) for parameter in actual_parameters}
        if seen.intersection(identifiers):
            raise ValueError("scratch checkpoint optimizer groups overlap")
        seen.update(identifiers)
        if float(group.get("weight_decay", math.nan)) != 1e-3:
            raise ValueError("scratch checkpoint AdamW weight_decay drifted")
        if tuple(group.get("betas", ())) != (0.9, 0.999):
            raise ValueError("scratch checkpoint AdamW betas drifted")
        if float(group.get("eps", math.nan)) != 1e-8:
            raise ValueError("scratch checkpoint AdamW eps drifted")
        if "initial_lr" in group and float(group["initial_lr"]) != float(
            receipt.peak_lr
        ):
            raise ValueError("scratch checkpoint AdamW initial LR drifted")
        for field, expected_value in _ADAMW_BEHAVIOR_DEFAULTS:
            observed_value = group.get(field)
            if (
                type(observed_value) is not type(expected_value)
                or observed_value != expected_value
            ):
                raise ValueError(
                    f"scratch checkpoint AdamW {field} behavior drifted"
                )
        serialized_groups[name] = {
            "ordered_fqns": list(fqns),
            "fqn_sha256": receipt.fqn_sha256,
            "peak_lr": float(receipt.peak_lr),
            "clip_norm": float(receipt.clip_norm),
        }
    expected_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if seen != expected_ids:
        raise ValueError("scratch checkpoint optimizer groups do not cover model")

    scheduler_state_fn = getattr(scheduler, "state_dict", None)
    if not callable(scheduler_state_fn):
        raise TypeError("scheduler must expose state_dict()")
    scheduler_state = scheduler_state_fn()
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("scratch checkpoint scheduler state is invalid")
    base_lrs = scheduler_state.get("base_lrs")
    current_lrs = scheduler_state.get("_last_lr")
    if not isinstance(base_lrs, (list, tuple)) or len(base_lrs) != 2:
        raise ValueError("scratch checkpoint scheduler base LR state is invalid")
    if not isinstance(current_lrs, (list, tuple)) or len(current_lrs) != 2:
        raise ValueError("scratch checkpoint scheduler current LR state is invalid")
    expected_base = tuple(
        float(optimizer_partition.groups[name].peak_lr) for name in ("q", "agf")
    )
    observed_base = tuple(float(value) for value in base_lrs)
    observed_current = tuple(float(value) for value in current_lrs)
    optimizer_current = tuple(float(group["lr"]) for group in optimizer.param_groups)
    if observed_base != expected_base:
        raise ValueError("scratch checkpoint scheduler base LR disagrees with partition")
    if observed_current != optimizer_current:
        raise ValueError("scratch checkpoint scheduler current LR disagrees with optimizer")
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (*observed_base, *observed_current)
    ):
        raise ValueError("scratch checkpoint scheduler LR state must be finite")
    return {
        "scheme": SCRATCH_OPTIMIZER_SCHEME,
        "ordered_group_names": ["q", "agf"],
        "partition_validated": True,
        "groups": serialized_groups,
        "adamw": {
            "name": "AdamW",
            "weight_decay": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "scheduler_lrs": {
            "base": dict(zip(("q", "agf"), observed_base, strict=True)),
            "current": dict(zip(("q", "agf"), observed_current, strict=True)),
        },
    }


def _validated_rank_rng_ledger(value: object, *, rank: int) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("scratch checkpoint rank RNG ledger is missing")
    if not 0 <= rank < len(value):
        raise ValueError("scratch checkpoint rank RNG ledger excludes this rank")
    validated: list[Mapping[str, object]] = []
    for expected_rank, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "rank",
            "torch_rng_state",
            "cuda_rng_state",
        }:
            raise ValueError("scratch checkpoint rank RNG schema is invalid")
        if raw.get("rank") != expected_rank:
            raise ValueError("scratch checkpoint rank RNG order is invalid")
        cpu_state = raw.get("torch_rng_state")
        if (
            not isinstance(cpu_state, Tensor)
            or cpu_state.dtype != torch.uint8
            or cpu_state.device.type != "cpu"
            or cpu_state.ndim != 1
        ):
            raise ValueError("scratch checkpoint CPU RNG state is invalid")
        try:
            torch.Generator(device="cpu").set_state(cpu_state)
        except RuntimeError as exc:
            raise ValueError("scratch checkpoint CPU RNG state is unloadable") from exc
        cuda_state = raw.get("cuda_rng_state")
        if cuda_state is not None:
            if (
                not isinstance(cuda_state, Tensor)
                or cuda_state.dtype != torch.uint8
                or cuda_state.device.type != "cpu"
                or cuda_state.ndim != 1
            ):
                raise ValueError("scratch checkpoint CUDA RNG state is invalid")
            if not torch.cuda.is_available():
                raise ValueError("scratch checkpoint requires CUDA RNG state")
            try:
                torch.Generator(device=torch.device("cuda", torch.cuda.current_device())).set_state(
                    cuda_state
                )
            except RuntimeError as exc:
                raise ValueError("scratch checkpoint CUDA RNG state is unloadable") from exc
        validated.append(raw)
    return tuple(validated)


def _preflight_model_state(model: nn.Module, value: object) -> Mapping[str, Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("resume checkpoint model state is invalid")
    expected = model.state_dict()
    if set(value) != set(expected):
        raise ValueError("resume checkpoint model keys mismatch")
    for name, target in expected.items():
        observed = value[name]
        if not isinstance(observed, Tensor):
            raise ValueError(f"resume checkpoint model tensor {name} is invalid")
        if observed.shape != target.shape:
            raise ValueError(f"resume checkpoint model shape {name} mismatch")
        if observed.dtype != target.dtype:
            raise ValueError(f"resume checkpoint model dtype {name} mismatch")
    return value  # type: ignore[return-value]


def _preflight_v2_optimizer_scheduler(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    optimizer_partition: OptimizerPartitionReceipt,
    observed_group_identity: object,
    optimizer_state: object,
    scheduler_state: object,
    successful_updates: int,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if (
        isinstance(successful_updates, bool)
        or not isinstance(successful_updates, int)
        or successful_updates < 1
    ):
        raise ValueError(
            "scratch checkpoint successful-update frontier must be positive"
        )
    expected_group_identity = _optimizer_group_identity(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        optimizer_partition=optimizer_partition,
    )
    observed = _mapping(observed_group_identity, "optimizer group identity")
    for name in (
        "scheme",
        "ordered_group_names",
        "partition_validated",
        "groups",
        "adamw",
    ):
        if observed.get(name) != expected_group_identity[name]:
            raise ValueError(f"scratch checkpoint optimizer group identity {name} mismatch")
    observed_lrs = _mapping(
        observed.get("scheduler_lrs"), "optimizer group scheduler_lrs"
    )
    expected_lrs = _mapping(
        expected_group_identity["scheduler_lrs"], "expected scheduler_lrs"
    )
    if observed_lrs.get("base") != expected_lrs["base"]:
        raise ValueError("scratch checkpoint optimizer base LR identity mismatch")
    current_by_name = _mapping(
        observed_lrs.get("current"), "optimizer current LR identity"
    )
    if tuple(current_by_name) != ("q", "agf"):
        raise ValueError("scratch checkpoint optimizer current LR names are invalid")
    current_lrs = tuple(float(current_by_name[name]) for name in ("q", "agf"))
    if not all(math.isfinite(value) and value >= 0.0 for value in current_lrs):
        raise ValueError("scratch checkpoint optimizer current LRs are invalid")

    optimizer_payload = _mapping(optimizer_state, "optimizer state")
    if set(optimizer_payload) != {"state", "param_groups"}:
        raise ValueError("scratch checkpoint optimizer state schema is invalid")
    payload_groups = optimizer_payload["param_groups"]
    if not isinstance(payload_groups, (list, tuple)) or len(payload_groups) != 2:
        raise ValueError("scratch checkpoint optimizer group state is invalid")
    parameter_by_saved_id: dict[object, nn.Parameter] = {}
    for index, name in enumerate(("q", "agf")):
        raw_group = payload_groups[index]
        if not isinstance(raw_group, Mapping):
            raise ValueError("scratch checkpoint optimizer group state is invalid")
        target_group = optimizer.param_groups[index]
        if set(raw_group) != set(target_group):
            raise ValueError("scratch checkpoint optimizer group schema mismatch")
        if raw_group.get("group_name") != name:
            raise ValueError("scratch checkpoint optimizer group state order mismatch")
        saved_ids = raw_group.get("params")
        target_parameters = tuple(optimizer.param_groups[index]["params"])
        if not isinstance(saved_ids, (list, tuple)) or len(saved_ids) != len(
            target_parameters
        ):
            raise ValueError("scratch checkpoint optimizer group parameter count mismatch")
        for saved_id, parameter in zip(saved_ids, target_parameters, strict=True):
            if saved_id in parameter_by_saved_id:
                raise ValueError("scratch checkpoint optimizer parameter ID overlaps")
            parameter_by_saved_id[saved_id] = parameter
        if tuple(raw_group.get("betas", ())) != (0.9, 0.999):
            raise ValueError("scratch checkpoint optimizer beta state mismatch")
        if float(raw_group.get("weight_decay", math.nan)) != 1e-3:
            raise ValueError("scratch checkpoint optimizer weight_decay state mismatch")
        if float(raw_group.get("eps", math.nan)) != 1e-8:
            raise ValueError("scratch checkpoint optimizer eps state mismatch")
        for field, _default_value in _ADAMW_BEHAVIOR_DEFAULTS:
            observed_value = raw_group.get(field)
            target_value = target_group.get(field)
            if (
                type(observed_value) is not type(target_value)
                or observed_value != target_value
            ):
                raise ValueError(
                    f"scratch checkpoint optimizer {field} state mismatch"
                )
        if float(raw_group.get("initial_lr", math.nan)) != float(
            expected_lrs["base"][name]
        ):
            raise ValueError("scratch checkpoint optimizer initial LR state mismatch")
        if float(raw_group.get("lr", math.nan)) != current_lrs[index]:
            raise ValueError("scratch checkpoint optimizer current LR state mismatch")

    raw_state = optimizer_payload["state"]
    if not isinstance(raw_state, Mapping) or set(raw_state) != set(
        parameter_by_saved_id
    ):
        raise ValueError(
            "scratch checkpoint AdamW parameter state is incomplete"
        )
    for saved_id, state in raw_state.items():
        if not isinstance(state, Mapping) or set(state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError("scratch checkpoint AdamW moment state is invalid")
        parameter = parameter_by_saved_id[saved_id]
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(moment_name)
            if not isinstance(moment, Tensor) or moment.shape != parameter.shape:
                raise ValueError("scratch checkpoint AdamW moment shape mismatch")
            if moment.dtype != parameter.dtype:
                raise ValueError("scratch checkpoint AdamW moment dtype mismatch")
            if not bool(torch.isfinite(moment).all()):
                raise ValueError("scratch checkpoint AdamW moment is non-finite")
        step = state.get("step")
        if (
            not isinstance(step, Tensor)
            or step.numel() != 1
            or not step.is_floating_point()
        ):
            raise ValueError("scratch checkpoint AdamW step state is invalid")
        step_value = float(step.detach().cpu().item())
        if (
            not math.isfinite(step_value)
            or not step_value.is_integer()
            or int(step_value) != successful_updates
        ):
            raise ValueError(
                "scratch checkpoint AdamW step disagrees with update frontier"
            )

    scheduler_payload = _mapping(scheduler_state, "scheduler state")
    current_scheduler = getattr(scheduler, "state_dict", None)
    if not callable(current_scheduler):
        raise TypeError("scheduler must expose state_dict()")
    current_schema = current_scheduler()
    if not isinstance(current_schema, Mapping) or set(scheduler_payload) != set(
        current_schema
    ):
        raise ValueError("scratch checkpoint scheduler state schema mismatch")
    immutable_scheduler_fields = (
        "warmup_steps",
        "max_steps",
        "warmup_start_lr",
        "eta_min",
        "verbose",
        "_get_lr_called_within_step",
    )
    if any(
        scheduler_payload.get(name) != current_schema.get(name)
        for name in immutable_scheduler_fields
    ):
        raise ValueError("scratch checkpoint scheduler immutable state mismatch")
    last_epoch = scheduler_payload.get("last_epoch")
    step_count = scheduler_payload.get("_step_count")
    if (
        isinstance(last_epoch, bool)
        or not isinstance(last_epoch, int)
        or last_epoch != successful_updates
        or isinstance(step_count, bool)
        or not isinstance(step_count, int)
        or step_count != successful_updates + 1
    ):
        raise ValueError(
            "scratch checkpoint scheduler chronology disagrees with update frontier"
        )
    base_lrs = scheduler_payload.get("base_lrs")
    last_lrs = scheduler_payload.get("_last_lr")
    if not isinstance(base_lrs, (list, tuple)) or tuple(map(float, base_lrs)) != tuple(
        float(expected_lrs["base"][name]) for name in ("q", "agf")
    ):
        raise ValueError("scratch checkpoint scheduler base LR state mismatch")
    if not isinstance(last_lrs, (list, tuple)) or tuple(map(float, last_lrs)) != current_lrs:
        raise ValueError("scratch checkpoint scheduler current LR state mismatch")
    warmup_steps = current_schema.get("warmup_steps")
    max_steps = current_schema.get("max_steps")
    warmup_start_lr = current_schema.get("warmup_start_lr")
    eta_min = current_schema.get("eta_min")
    if (
        isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or warmup_steps < 1
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps <= warmup_steps
        or successful_updates > max_steps
        or isinstance(warmup_start_lr, bool)
        or not isinstance(warmup_start_lr, (int, float))
        or not math.isfinite(float(warmup_start_lr))
        or isinstance(eta_min, bool)
        or not isinstance(eta_min, (int, float))
        or not math.isfinite(float(eta_min))
    ):
        raise ValueError("scratch checkpoint scheduler admitted recipe is invalid")
    recomputed_lrs: list[float] = []
    for name in ("q", "agf"):
        base_lr = float(expected_lrs["base"][name])
        if successful_updates < warmup_steps:
            value = float(warmup_start_lr) + (
                base_lr - float(warmup_start_lr)
            ) * successful_updates / warmup_steps
        else:
            value = float(eta_min) + (base_lr - float(eta_min)) * (
                1.0
                + math.cos(
                    math.pi
                    * (successful_updates - warmup_steps)
                    / (max_steps - warmup_steps)
                )
            ) / 2.0
        recomputed_lrs.append(value)
    if tuple(recomputed_lrs) != current_lrs:
        raise ValueError(
            "scratch checkpoint scheduler LR disagrees with update frontier"
        )
    return optimizer_payload, scheduler_payload


def save_epoch_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    epoch: int,
    successful_updates: int,
    plan_sha256: str,
    config_sha256: str,
    code_sha256: str,
    next_batch_sha256: str,
    rng_state_by_rank: Sequence[Mapping[str, object]] | None = None,
    variant_identity: Mapping[str, str] | None = None,
    training_contract: Mapping[str, str] | None = None,
    initialization_identity: Mapping[str, object] | None = None,
    optimizer_partition: OptimizerPartitionReceipt | None = None,
) -> str:
    destination = Path(path)
    scratch_v2 = _scratch_metadata_mode(
        training_contract,
        initialization_identity,
        optimizer_partition,
    )
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    if isinstance(successful_updates, bool) or not isinstance(successful_updates, int) or successful_updates < 0:
        raise ValueError("checkpoint successful_updates must be non-negative")
    identity = {
        "plan_sha256": _require_sha(plan_sha256, "plan"),
        "config_sha256": _require_sha(config_sha256, "config"),
        "code_sha256": _require_sha(code_sha256, "code"),
        "next_batch_sha256": _require_sha(next_batch_sha256, "next batch"),
    }
    scheduler_state = getattr(scheduler, "state_dict", None)
    if not callable(scheduler_state):
        raise TypeError("scheduler must expose state_dict()")
    resolved_scheduler_state = scheduler_state()
    payload = {
        "schema": (
            "J2J_CONTEXT4_EPOCH_CHECKPOINT_V2"
            if scratch_v2
            else "J2J_CONTEXT4_EPOCH_CHECKPOINT_V1"
        ),
        "identity": identity,
        "epoch": epoch,
        "successful_updates": successful_updates,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": resolved_scheduler_state,
        "torch_rng_state": torch.random.get_rng_state(),
        # V2 owns an explicit rank-bound RNG ledger below.  Avoid asking rank 0
        # to touch every visible CUDA device merely to populate the legacy copy.
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if not scratch_v2 and torch.cuda.is_available()
            else None
        ),
    }
    if scratch_v2:
        if (
            training_contract is None
            or initialization_identity is None
            or optimizer_partition is None
        ):
            raise RuntimeError("validated scratch checkpoint metadata is missing")
        payload["training_contract"] = _validated_training_contract(
            training_contract
        )
        payload["initialization_identity"] = _validated_initialization_identity(
            initialization_identity
        )
        payload["optimizer_group_identity"] = _optimizer_group_identity(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_partition=optimizer_partition,
        )
    if variant_identity is not None:
        variant = resolve_variant_contract(variant_identity).identity
        if variant == FULL_IDENTITY:
            raise ValueError("Full checkpoint must not carry a native variant identity")
        payload["variant_identity"] = variant_identity_dict(variant)
    if scratch_v2:
        payload["rng_state_by_rank"] = _validated_rank_rng_ledger(
            rng_state_by_rank,
            rank=0,
        )
    elif rng_state_by_rank is not None:
        states = tuple(rng_state_by_rank)
        if not states:
            raise ValueError("distributed RNG state ledger must be nonempty")
        for expected_rank, state in enumerate(states):
            if not isinstance(state, Mapping) or state.get("rank") != expected_rank:
                raise ValueError("distributed RNG state ledger rank order is invalid")
            if not isinstance(state.get("torch_rng_state"), Tensor):
                raise ValueError("distributed CPU RNG state is invalid")
            cuda_state = state.get("cuda_rng_state")
            if cuda_state is not None and not isinstance(cuda_state, Tensor):
                raise ValueError("distributed CUDA RNG state is invalid")
        payload["rng_state_by_rank"] = states
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError("checkpoint temporary path already exists")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return _sha256_file(destination)


def load_epoch_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    expected_identity: Mapping[str, str],
    rank: int = 0,
    expected_variant_identity: Mapping[str, str] | None = None,
    expected_training_contract: Mapping[str, str] | None = None,
    expected_initialization_identity: Mapping[str, object] | None = None,
    expected_optimizer_partition: OptimizerPartitionReceipt | None = None,
    expected_updates_per_epoch: int | None = None,
) -> ResumeState:
    scratch_v2 = _scratch_metadata_mode(
        expected_training_contract,
        expected_initialization_identity,
        expected_optimizer_partition,
    )
    source = Path(path)
    if not source.is_file():
        raise ValueError("resume checkpoint is missing")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    expected_schema = (
        "J2J_CONTEXT4_EPOCH_CHECKPOINT_V2"
        if scratch_v2
        else "J2J_CONTEXT4_EPOCH_CHECKPOINT_V1"
    )
    if not isinstance(payload, Mapping) or payload.get("schema") != expected_schema:
        raise ValueError("resume checkpoint schema is invalid")
    expected = {
        name: _require_sha(expected_identity.get(name), name)
        for name in ("plan_sha256", "config_sha256", "code_sha256", "next_batch_sha256")
    }
    observed = payload.get("identity")
    if observed != expected:
        raise ValueError("resume checkpoint identity mismatch")
    observed_variant = payload.get("variant_identity")
    if expected_variant_identity is None:
        if observed_variant is not None:
            raise ValueError("Full resume checkpoint must not carry a native variant identity")
    else:
        expected_contract = resolve_variant_contract(expected_variant_identity)
        if expected_contract.identity == FULL_IDENTITY:
            raise ValueError("Full resume identity must remain exact-derived")
        if not isinstance(observed_variant, Mapping):
            raise ValueError("control resume checkpoint native variant identity is missing")
        observed_contract = resolve_variant_contract(observed_variant)
        if observed_contract.identity != expected_contract.identity:
            raise ValueError("resume checkpoint native variant identity mismatch")
    model_state = payload.get("model_state")
    optimizer_state = payload.get("optimizer_state")
    scheduler_state = payload.get("scheduler_state")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping) or not isinstance(scheduler_state, Mapping):
        raise ValueError("resume checkpoint state mappings are invalid")
    epoch = payload.get("epoch")
    updates = payload.get("successful_updates")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("resume checkpoint epoch is invalid")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("resume checkpoint update count is invalid")

    if scratch_v2:
        if expected_training_contract is None:
            raise RuntimeError("scratch resume training contract is missing")
        if expected_initialization_identity is None:
            raise RuntimeError("scratch resume initialization identity is missing")
        if expected_optimizer_partition is None:
            raise RuntimeError("scratch resume optimizer partition is missing")
        if expected_updates_per_epoch is None:
            raise ValueError(
                "scratch resume expected_updates_per_epoch is missing"
            )
        resolved_updates_per_epoch = _positive_int(
            expected_updates_per_epoch,
            "scratch resume expected_updates_per_epoch",
        )
        if epoch < 1 or updates != epoch * resolved_updates_per_epoch:
            raise ValueError(
                "scratch resume epoch/update frontier disagrees with the admitted plan"
            )
        expected_training = _validated_training_contract(
            expected_training_contract
        )
        expected_initialization = _validated_initialization_identity(
            expected_initialization_identity
        )
        if payload.get("training_contract") != expected_training:
            raise ValueError("scratch resume training contract mismatch")
        if payload.get("initialization_identity") != expected_initialization:
            raise ValueError("scratch resume initialization identity mismatch")
        model_state = _preflight_model_state(model, model_state)
        optimizer_state, scheduler_state = _preflight_v2_optimizer_scheduler(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            optimizer_partition=expected_optimizer_partition,
            observed_group_identity=payload.get("optimizer_group_identity"),
            optimizer_state=optimizer_state,
            scheduler_state=scheduler_state,
            successful_updates=updates,
        )
        distributed_rng = _validated_rank_rng_ledger(
            payload.get("rng_state_by_rank"),
            rank=rank,
        )
        selected_rng: Mapping[str, object] | None = distributed_rng[rank]
    else:
        if expected_updates_per_epoch is not None:
            raise ValueError(
                "legacy resume must not carry scratch updates-per-epoch identity"
            )
        selected_rng = None
        distributed_rng = payload.get("rng_state_by_rank")
        if distributed_rng is not None:
            if (
                not isinstance(distributed_rng, (list, tuple))
                or not 0 <= rank < len(distributed_rng)
            ):
                raise ValueError(
                    "resume distributed RNG ledger is invalid for this rank"
                )
            candidate = distributed_rng[rank]
            if not isinstance(candidate, Mapping) or candidate.get("rank") != rank:
                raise ValueError("resume distributed RNG rank binding is invalid")
            selected_rng = candidate

    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    load_scheduler = getattr(scheduler, "load_state_dict", None)
    if not callable(load_scheduler):
        raise TypeError("scheduler must expose load_state_dict()")
    load_scheduler(scheduler_state)
    rng = (
        selected_rng.get("torch_rng_state")
        if selected_rng is not None
        else payload.get("torch_rng_state")
    )
    if not isinstance(rng, Tensor):
        raise ValueError("resume checkpoint CPU RNG state is invalid")
    torch.random.set_rng_state(rng)
    cuda_rng = (
        selected_rng.get("cuda_rng_state")
        if selected_rng is not None
        else payload.get("cuda_rng_state_all")
    )
    if cuda_rng is not None and selected_rng is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint requires CUDA RNG state")
        if not isinstance(cuda_rng, Tensor):
            raise ValueError("resume checkpoint local CUDA RNG state is invalid")
        torch.cuda.set_rng_state(cuda_rng)
    elif cuda_rng is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint requires CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_rng)
    return ResumeState(epoch=epoch, successful_updates=updates, identity=expected)


_EXPECTED_TRAIN_LEDGER = {
    "train_trajectories": 8_004,
    "factual_origins": 640_313,
    "q_goal_view_occurrences": 1_218_429,
    "active_q_future_blocks": 4_825_787,
}
_EXPECTED_MAIN_PARAMETERS = 120_758_809


def _expected_train_ledger(config: Mapping[str, Any]) -> dict[str, int]:
    ledger = dict(_EXPECTED_TRAIN_LEDGER)
    if _variant_contract_from_config(config).identity in RECURRENT_STOCHASTIC_IDENTITIES:
        ledger["active_q_future_blocks"] = ledger["q_goal_view_occurrences"]
    return ledger


def _training_horizons(model_config: Mapping[str, Any]) -> tuple[int, int, int]:
    """Separate deployment, unchanged goal sampling, and supervised target length."""
    deployment = _positive_int(model_config.get("horizon"), "model horizon")
    sampling = _positive_int(model_config.get("goal_sampling_horizon", deployment), "goal sampling horizon")
    target = _positive_int(model_config.get("training_q_horizon", deployment), "training Q horizon")
    return deployment, sampling, target


def supervised_q_future_blocks(goals_by_trajectory: Mapping[int, Sequence[tuple[int, int]]], horizon: int) -> int:
    """Count actual supervised blocks without altering the retained goal pairs."""
    horizon = _positive_int(horizon, "supervised Q horizon")
    return sum(min(horizon, goal - origin) for pairs in goals_by_trajectory.values()
               for origin, goal in pairs)



@dataclass(frozen=True)
class TrainingRunReceipt:
    run_kind: str
    rank: int
    world_size: int
    start_epoch: int
    completed_epoch: int
    successful_updates: int
    plan_sha256: str
    config_sha256: str
    code_sha256: str
    final_checkpoint_path: str
    final_checkpoint_sha256: str
    stopped_early: bool = False


def _reject_secret_fields(value: object, *, path: str = "config") -> None:
    forbidden = {
        "password",
        "passwd",
        "pwd",
        "token",
        "api_key",
        "apikey",
        "access_token",
        "proxy",
        "proxy_url",
        "proxy_password",
    }
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in forbidden:
                raise ValueError(f"{path} must not contain credential field {key!r}")
            _reject_secret_fields(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secret_fields(child, path=f"{path}[{index}]")
    elif isinstance(value, str) and "://" in value and "@" in value:
        raise ValueError(f"{path} must not contain an authenticated URL")


def _required_path(section: Mapping[str, Any], name: str, *, directory: bool) -> Path:
    raw = section.get(name)
    if not isinstance(raw, (str, os.PathLike)) or not str(raw):
        raise ValueError(f"config path {name} must be declared")
    path = Path(raw)
    if directory and not path.is_dir():
        raise ValueError(f"config directory {name} is missing")
    if not directory and not path.is_file():
        raise ValueError(f"config file {name} is missing")
    return path


def _runtime_sections(
    config: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any]]:
    data = _mapping(config.get("data"), "data")
    warm_raw = config.get("warm_start")
    warm = None if warm_raw is None else _mapping(warm_raw, "warm_start")
    ledger = _mapping(config.get("ledger"), "ledger")
    return data, warm, ledger


def _admit_runtime_paths(config: Mapping[str, Any]) -> Mapping[str, Path]:
    """Check all caller paths and bytes before cache or checkpoint loading."""

    data, warm, ledger = _runtime_sections(config)
    identity = _mapping(config.get("identity"), "identity")
    paths: dict[str, Path] = {
        "canonical_manifest": _required_path(data, "canonical_manifest", directory=False),
        "source_catalog": _required_path(data, "source_catalog", directory=False),
        "z32_cache_dir": _required_path(data, "z32_cache_dir", directory=True),
    }
    scratch = config.get("initialization") is not None
    if scratch:
        if warm is not None:
            raise ValueError("scratch runtime must not admit warm checkpoint paths")
    else:
        if warm is None:
            raise ValueError("legacy runtime requires warm_start paths")
        paths["q_checkpoint"] = _required_path(
            warm, "q_checkpoint", directory=False
        )
        paths["agf_checkpoint"] = _required_path(
            warm, "agf_checkpoint", directory=False
        )
    cache_manifest = paths["z32_cache_dir"] / "manifest.json"
    if not cache_manifest.is_file():
        raise ValueError("Z32 cache manifest is missing")
    checks: list[tuple[Path, object, str]] = [
        (paths["canonical_manifest"], identity["source_manifest_sha256"], "source manifest"),
        (cache_manifest, identity["cache_manifest_sha256"], "cache manifest"),
    ]
    if not scratch:
        checks.extend(
            (
                (
                    paths["q_checkpoint"],
                    identity["q_warm_sha256"],
                    "Q warm checkpoint",
                ),
                (
                    paths["agf_checkpoint"],
                    identity["agf_warm_sha256"],
                    "AGF warm checkpoint",
                ),
            )
        )
    for path, expected_raw, label in checks:
        expected = _require_sha(expected_raw, label)
        if _sha256_file(path) != expected:
            raise ValueError(f"{label} SHA-256 mismatch")
    expected_parent = _require_sha(
        data.get("expected_parent_manifest_sha256"),
        "cache parent manifest",
    )
    expected_identities = data.get("expected_identities")
    if not isinstance(expected_identities, Mapping) or not expected_identities:
        raise ValueError("config data.expected_identities must be a nonempty mapping")
    require_production = data.get("require_production_eligible", True)
    if type(require_production) is not bool or require_production is not True:
        raise ValueError("formal context4 data must require production eligibility")
    if type(data.get("pin_memory", True)) is not bool:
        raise ValueError("config data.pin_memory must be boolean")
    for name, expected in _expected_train_ledger(config).items():
        if ledger.get(name) != expected:
            raise ValueError(f"config train ledger {name} does not match the frozen design")
    return {**paths, "cache_manifest": cache_manifest}


def _descriptor_projection(dataset: object) -> tuple[TrajectoryDescriptor, ...]:
    project = getattr(dataset, "batch_descriptors", None)
    if not callable(project):
        raise TypeError("categorical dataset must expose batch_descriptors()")
    rows = tuple(project())
    descriptors: list[TrajectoryDescriptor] = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ValueError("categorical batch descriptor must have three fields")
        dataset_index, trajectory_key, origin_count = row
        descriptors.append(
            TrajectoryDescriptor(
                dataset_index=dataset_index,
                trajectory_key=trajectory_key,
                origin_count=origin_count,
            )
        )
    if not descriptors:
        raise ValueError("categorical dataset descriptor projection is empty")
    return tuple(descriptors)


def _code_identity() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "j2j" / "context4.py",
        root / "j2j" / "context4_data.py",
        root / "j2j" / "context4_objective.py",
        root / "j2j" / "context4_variants.py",
        root / "j2j" / "context4_rollout.py",
        root / "j2j" / "context4_metrics.py",
        root / "j2j" / "context4_training.py",
        root / "j2j" / "spatial_intact.py",
        root / "j2j" / "proposal" / "losses.py",
        root / "j2j" / "proposal" / "initialization.py",
        root / "j2j" / "proposal" / "model.py",
        root / "j2j" / "proposal" / "blocks.py",
        root / "j2j" / "proposal" / "future.py",
        root / "j2j" / "proposal" / "stochastic.py",
        root / "j2j" / "proposal" / "sampling.py",
        root / "j2j" / "proposal" / "stability.py",
        root / "j2j" / "context4_probes.py",
        root / "j2j" / "context4_forensics.py",
        root / "j2j" / "data" / "dataset.py",
        root / "j2j" / "data" / "source.py",
        root / "j2j" / "encoding" / "cache.py",
        root / "j2j" / "encoding" / "croco224.py",
        root / "j2j" / "encoding" / "dinov3_256.py",
        root / "module.py",
        root / "scripts" / "train_j2j_context4.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"code identity input is missing: {path.name}")
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def _scientific_config_identity(config: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(config))
    runtime = value.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("resume_checkpoint", None)
    return _canonical_json_sha256(value)


def _write_atomic_json(path: Path, value: Mapping[str, object], *, allow_identical: bool = False) -> None:
    _reject_secret_fields(value, path="receipt")
    encoded = (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if allow_identical and path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to overwrite run identity file {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _parse_cpu_affinity(specification: str) -> tuple[int, ...]:
    values: set[int] = set()
    for raw_piece in specification.split(","):
        piece = raw_piece.strip()
        if not piece:
            raise ValueError("CPU affinity contains an empty range")
        if "-" in piece:
            bounds = piece.split("-")
            if len(bounds) != 2 or not all(value.isdigit() for value in bounds):
                raise ValueError("CPU affinity range is invalid")
            start, stop = (int(value) for value in bounds)
            if start > stop:
                raise ValueError("CPU affinity range must be ascending")
            values.update(range(start, stop + 1))
        elif piece.isdigit():
            values.add(int(piece))
        else:
            raise ValueError("CPU affinity entry is invalid")
    if not values:
        raise ValueError("CPU affinity set must be nonempty")
    return tuple(sorted(values))


def _configure_cpu_runtime(runtime: Mapping[str, Any], *, local_rank: int, local_world_size: int) -> tuple[int, ...]:
    mode = runtime.get("cpu_affinity")
    if not isinstance(mode, str) or not mode:
        raise ValueError("runtime cpu_affinity must be a nonempty string")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if mode == "split-current":
        if not affinity:
            raise RuntimeError("split-current CPU affinity is unsupported on this platform")
        if not 0 <= local_rank < local_world_size:
            raise ValueError("LOCAL_RANK is outside LOCAL_WORLD_SIZE")
        start = len(affinity) * local_rank // local_world_size
        stop = len(affinity) * (local_rank + 1) // local_world_size
        assigned = affinity[start:stop]
        if not assigned:
            raise RuntimeError("CPU affinity has fewer cores than local ranks")
        os.sched_setaffinity(0, assigned)
        affinity = assigned
    elif mode not in {"caller-bound", "launcher"}:
        requested = list(_parse_cpu_affinity(mode))
        if affinity and not set(requested).issubset(affinity):
            raise ValueError("requested CPU affinity escapes the caller-visible CPU set")
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("explicit CPU affinity is unsupported on this platform")
        os.sched_setaffinity(0, requested)
        affinity = requested
    threads = runtime.get("torch_threads_per_rank")
    if threads is not None:
        threads = _positive_int(threads, "runtime torch_threads_per_rank")
        torch.set_num_threads(threads)
    return tuple(affinity)


def _distributed_runtime(config: Mapping[str, Any]) -> tuple[int, int, int, torch.device, bool]:
    training = _mapping(config.get("training"), "training")
    runtime = _mapping(config.get("runtime"), "runtime")
    world_size = int(training["world_size"])
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if not 0 <= rank < world_size:
        raise ValueError("RANK is outside the configured world")
    _configure_cpu_runtime(
        runtime,
        local_rank=local_rank,
        local_world_size=local_world_size,
    )
    device_kind = runtime.get("device", "cuda")
    if device_kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("context4 CUDA runtime is unavailable")
        if not 0 <= local_rank < torch.cuda.device_count():
            raise RuntimeError("LOCAL_RANK has no visible CUDA device")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif device_kind == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError("runtime device must be cuda or cpu")
    created = False
    if world_size > 1:
        if not torch.distributed.is_available():
            raise RuntimeError("distributed package is unavailable")
        if not torch.distributed.is_initialized():
            timeout_seconds = _positive_int(
                runtime.get(
                    "process_group_timeout_seconds",
                    runtime.get("timeout_seconds", 1800),
                ),
                "runtime process_group_timeout_seconds",
            )
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(seconds=timeout_seconds),
            )
            created = True
        if torch.distributed.get_world_size() != world_size or torch.distributed.get_rank() != rank:
            raise RuntimeError("initialized process group disagrees with torchrun environment")
    elif torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_world_size() != 1 or torch.distributed.get_rank() != rank:
            raise RuntimeError("preinitialized process group disagrees with single-rank config")
    return rank, local_rank, world_size, device, created


def _fit_assessment(
    train_history: Sequence[float],
    dev_history: Sequence[float],
) -> str:
    if len(train_history) < 2 or len(dev_history) < 2:
        return "insufficient_history"
    if train_history[-1] < train_history[-2] and dev_history[-1] > dev_history[-2]:
        return "possible_overfit_signal"
    if train_history[-1] >= train_history[-2] and dev_history[-1] >= dev_history[-2]:
        return "no_fit_progress_signal"
    return "no_overfit_signal"


def run_context4_training(
    *,
    config: Mapping[str, Any],
    output_root: str | os.PathLike[str],
    qualification_update_limit: int | None = None,
) -> TrainingRunReceipt | QualificationRunReceipt:
    """Run one resolved LR-pilot or formal context-four training identity."""

    if not isinstance(config, Mapping):
        raise TypeError("context4 training config must be a mapping")
    _reject_secret_fields(config)
    actual_world = int(os.environ.get("WORLD_SIZE", "1"))
    admitted = validate_context4_config(config, actual_world_size=actual_world)
    variant_contract = _variant_contract_from_config(admitted)
    resolved_variant = variant_identity_dict(variant_contract.identity)
    training = _mapping(admitted["training"], "training")
    runtime = _mapping(admitted["runtime"], "runtime")
    data = _mapping(admitted["data"], "data")
    model_config = _mapping(admitted["model"], "model")
    identity = _mapping(admitted["identity"], "identity")
    run_root = Path(output_root)
    scratch = admitted.get("initialization") is not None
    resume_path = runtime.get("resume_checkpoint")
    if qualification_update_limit is not None:
        if type(qualification_update_limit) is not int or qualification_update_limit <= 0:
            raise ValueError("qualification limit must be a positive integer")
        if resume_path is not None or not scratch:
            raise ValueError("qualification requires a fresh scratch run without resume")
    if run_root.exists() and not run_root.is_dir():
        raise ValueError("output_root must be a directory path")
    output_is_new_or_empty = not run_root.exists() or not any(run_root.iterdir())
    if scratch and resume_path is not None and output_is_new_or_empty:
        raise ValueError(
            "scratch initial launch must not use a resume checkpoint"
        )
    if resume_path is None and run_root.is_dir() and any(run_root.iterdir()):
        raise FileExistsError("fresh context4 run requires an empty output_root")
    paths = _admit_runtime_paths(admitted)

    rank = 0
    world_size = actual_world
    created_process_group = False
    try:
        rank, local_rank, world_size, device, created_process_group = _distributed_runtime(admitted)
        seed = _positive_int(model_config.get("global_seed", 3072), "model global_seed", allow_zero=True)
        torch.manual_seed(seed + rank)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed + rank)

        # Reuse the admitted project source/cache bridge verbatim.  This runner
        # contributes only rank ownership and joint objective scheduling.
        from j2j.data.dataset import CategoricalTrajectoryDataset
        from j2j.data.source import open_released_streamvln_source
        from j2j.encoding.cache import open_cache_store
        from stable_pretraining.optim import create_scheduler

        cache_store = open_cache_store(
            paths["z32_cache_dir"],
            expected_parent_manifest_sha256=str(data["expected_parent_manifest_sha256"]),
            expected_identities=dict(_mapping(data["expected_identities"], "data.expected_identities")),
            require_stage=str(data.get("require_stage", "Z32")),
            require_training_eligible=True,
            require_production_eligible=True,
        )
        train_source = open_released_streamvln_source(
            canonical_manifest=paths["canonical_manifest"],
            source_catalog=paths["source_catalog"],
            cache_store=cache_store,
            expected_manifest_sha256=str(identity["source_manifest_sha256"]),
            partition="project-train",
            require_production_eligible=True,
        )
        validation_enabled = bool(training.get("validation_enabled", True))
        dev_source = open_released_streamvln_source(
            canonical_manifest=paths["canonical_manifest"],
            source_catalog=paths["source_catalog"],
            cache_store=cache_store,
            expected_manifest_sha256=str(identity["source_manifest_sha256"]),
            partition="project-dev",
            require_production_eligible=True,
        ) if validation_enabled else None
        stochastic = variant_contract.identity in STOCHASTIC_IDENTITIES
        raw = variant_contract.identity in RAW_IDENTITIES
        dataset_options = {
            "spatial_shape": (int(model_config["grid_side"]) ** 2, int(model_config["latent_dim"]))
        } if raw else {}
        if raw and tuple(cache_store.manifest.get("spatial_shape", (36, 768))) != dataset_options["spatial_shape"]:
            raise ValueError("raw cache and configured model grid shapes disagree")
        train_dataset = CategoricalTrajectoryDataset(train_source, cache_store, **dataset_options)
        dev_dataset = CategoricalTrajectoryDataset(dev_source, cache_store, **dataset_options) if validation_enabled else None
        train_descriptors = _descriptor_projection(train_dataset)
        dev_descriptors = _descriptor_projection(dev_dataset) if validation_enabled else ()
        horizon, goal_sampling_horizon, training_q_horizon = _training_horizons(model_config)
        train_goals, train_goal_counts = build_q_goal_views(train_descriptors, horizon=goal_sampling_horizon)
        dev_goals = build_q_goal_views(dev_descriptors, horizon=goal_sampling_horizon)[0] if validation_enabled else {}
        observed_ledger = {
            "train_trajectories": len(train_descriptors),
            "factual_origins": sum(value.origin_count for value in train_descriptors),
            "q_goal_view_occurrences": train_goal_counts["q_occurrences"],
            "active_q_future_blocks": supervised_q_future_blocks(train_goals, training_q_horizon),
        }
        if observed_ledger != _expected_train_ledger(admitted):
            raise ValueError(
                f"released project-train ledger mismatch: observed={observed_ledger}"
            )

        effective_batch = int(training["effective_global_batch"])
        plan = build_rank_origin_plan(
            train_descriptors,
            world_size=world_size,
            effective_batch=effective_batch,
        )
        expected_updates = (
            observed_ledger["factual_origins"] + effective_batch - 1
        ) // effective_batch
        if plan.update_count != expected_updates:
            raise ValueError(
                "project-train update count is inconsistent with its factual ledger and effective batch"
            )
        plan_sha = _plan_identity(plan)
        _qualification_limit(qualification_update_limit, update_count=plan.update_count,
                             epoch=1, successful_updates=0, world_size=world_size)
        checkpoint_identity = {
            "plan_sha256": plan_sha,
            "config_sha256": _scientific_config_identity(admitted),
            "code_sha256": _code_identity(),
            "next_batch_sha256": _next_batch_identity(plan, train_goals),
        }

        joint_model = Context4SpatialJointModel(
            global_seed=seed,
            modes=int(model_config["modes"]),
            horizon=horizon,
            latent_dim=int(model_config["latent_dim"]),
            grid_side=int(model_config["grid_side"]),
            hidden_dim=int(model_config.get("hidden_dim", 768)),
            heads=int(model_config.get("heads", 16)),
            ffn_dim=int(model_config.get("ffn_dim", 2048)),
            proposal_depth=int(model_config.get("proposal_depth", 6)),
            actor_depth=int(model_config.get("actor_depth", 3)),
            forward_depth=int(model_config.get("forward_depth", 6)),
            dropout=float(model_config.get("dropout", 0.1)),
            **({name: model_config[name] for name in (
                "attention_backend", "activation_checkpointing", "memory_norm", "prediction_norm"
            )} if raw else {}),
            **({name: model_config[name] for name in (
                "proposal_architecture", "trajectory_latent_dim", "kl_beta",
                "sigma_epsilon", "posterior_residual_init_std"
            )} if stochastic else {}),
            **({"stability": model_config["stability"]} if variant_contract.identity in STABLE_STOCHASTIC_IDENTITIES else {}),
            **({"training_q_horizon": training_q_horizon} if variant_contract.identity in RECURRENT_STOCHASTIC_IDENTITIES else {}),
            **({"retain_activation_blocks": model_config["retain_activation_blocks"]}
               if "retain_activation_blocks" in model_config else {}),
            **({"training_qkv_normalization_backend": model_config["training_qkv_normalization_backend"]}
               if "training_qkv_normalization_backend" in model_config else {}),
        )
        constructor_state_sha256 = _tensor_state_sha256(joint_model)
        scratch_training_contract: Mapping[str, str] | None = None
        scratch_initialization_identity: Mapping[str, object] | None = None
        if scratch:
            scratch_training_contract = _validated_training_contract(
                admitted.get("training_contract")
            )
            scratch_initialization_identity = _validated_initialization_identity(
                {
                    "mode": SCRATCH_INITIALIZATION_MODE,
                    "schema": SCRATCH_INITIALIZATION_SCHEMA,
                    "global_seed": seed,
                    "constructor_state_sha256": constructor_state_sha256,
                    "imported_trainable_keys": [],
                    "trainable_checkpoint_sources": [],
                }
            )
        parameter_count = sum(parameter.numel() for parameter in joint_model.parameters() if parameter.requires_grad)
        expected_parameters = (135235064 if variant_contract.identity in RECURRENT_STOCHASTIC_IDENTITIES else
                               148827992 if variant_contract.identity == STABLE_STOCHASTIC_IDENTITY else
                               148827672 if stochastic else _EXPECTED_MAIN_PARAMETERS)
        if int(model_config["modes"]) == 4 and horizon == 4 and parameter_count != expected_parameters:
            raise RuntimeError("main context4 model parameter count drifted from the reviewed design")
        objective_model = Context4ObjectiveModule(
            joint_model,
            loss_weights=variant_contract.loss_weights,
            precision=str(runtime.get("precision", "bf16-mixed")),
            **({"qg_objective": training["qg_objective"]} if raw else {}),
        )
        objective_model.to(device)

        warm_receipt: WarmStartReceipt | None = None
        if resume_path is None and not scratch:
            warm_receipt = load_model_only_warm_start(
                joint_model,
                q_checkpoint=str(paths["q_checkpoint"]),
                expected_q_sha256=str(identity["q_warm_sha256"]),
                agf_checkpoint=str(paths["agf_checkpoint"]),
                expected_agf_sha256=str(identity["agf_warm_sha256"]),
            )

        optimizer_config = _mapping(training["optimizer"], "training optimizer")
        optimizer_partition: OptimizerPartitionReceipt | None = None
        scratch_gradient_clip_norms: Mapping[str, float] | None = None
        if scratch:
            optimization_config = _mapping(
                training["optimization"], "training optimization"
            )
            optimizer, optimizer_partition = build_q_agf_optimizer(
                joint_model,
                optimization=optimization_config,
                optimizer_config=optimizer_config,
                **({"training_contract": scratch_training_contract} if raw else {}),
            )
            scratch_gradient_clip_norms = {
                name: float(value)
                for name, value in _mapping(
                    optimization_config["gradient_clip_norms"],
                    "training optimization gradient_clip_norms",
                ).items()
            }
        else:
            optimizer = torch.optim.AdamW(
                objective_model.parameters(),
                lr=float(training["learning_rate"]),
                weight_decay=float(optimizer_config["weight_decay"]),
                betas=(
                    float(optimizer_config["betas"][0]),
                    float(optimizer_config["betas"][1]),
                ),
                eps=float(optimizer_config["eps"]),
            )
        if scratch and resume_path is None and optimizer.state:
            raise RuntimeError("scratch optimizer state must be empty at launch")
        total_updates = int(training["epochs"]) * plan.update_count
        scheduler = create_scheduler(
            optimizer,
            {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(float(training["warmup_fraction"]) * total_updates)),
                "max_steps": total_updates,
                "warmup_start_lr": 0.0,
                "eta_min": 0.0,
            },
            module=None,
        )

        if resume_path is None:
            start_epoch = 1
            successful_updates = 0
        else:
            if not isinstance(resume_path, (str, os.PathLike)) or not Path(resume_path).is_file():
                raise ValueError("runtime resume_checkpoint is missing")
            resumed = load_epoch_checkpoint(
                resume_path,
                model=joint_model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_identity=checkpoint_identity,
                rank=rank,
                expected_variant_identity=(
                    None
                    if variant_contract.identity == FULL_IDENTITY
                    else resolved_variant
                ),
                expected_training_contract=(
                    scratch_training_contract if scratch else None
                ),
                expected_initialization_identity=(
                    scratch_initialization_identity if scratch else None
                ),
                expected_optimizer_partition=(
                    optimizer_partition if scratch else None
                ),
                expected_updates_per_epoch=(
                    plan.update_count if scratch else None
                ),
            )
            start_epoch = resumed.epoch + 1
            successful_updates = resumed.successful_updates
            if start_epoch > int(training["epochs"]):
                raise ValueError("resume checkpoint is already at or beyond the configured final epoch")

        if world_size > 1:
            ddp_kwargs: dict[str, object] = {
                "broadcast_buffers": True,
                "find_unused_parameters": False,
                "gradient_as_bucket_view": True,
            }
            if device.type == "cuda":
                ddp_kwargs.update({"device_ids": [local_rank], "output_device": local_rank})
            train_model: nn.Module = DistributedDataParallel(objective_model, **ddp_kwargs)
        else:
            train_model = objective_model

        workers = int(runtime["workers"])
        prefetch = int(runtime["prefetch"])
        pin_memory = bool(data.get("pin_memory", device.type == "cuda"))
        if type(data.get("pin_memory", pin_memory)) is not bool:
            raise ValueError("config data.pin_memory must be boolean")

        if rank == 0:
            run_root.mkdir(parents=True, exist_ok=True)
            run_identity = {
                "schema": "J2J_CONTEXT4_RUN_IDENTITY_V1",
                "experiment_id": str(admitted["experiment_id"]),
                "resolved_variant_identity": resolved_variant,
                "run_kind": training["run_kind"],
                "disposable_weights": training["run_kind"] == "lr_pilot",
                "world_size": world_size,
                "parameter_count": parameter_count,
                **checkpoint_identity,
                "source_catalog_sha256": _sha256_file(paths["source_catalog"]),
            }
            if scratch:
                if (
                    scratch_training_contract is None
                    or scratch_initialization_identity is None
                ):
                    raise RuntimeError("scratch run identity metadata is missing")
                run_identity["training_contract"] = dict(scratch_training_contract)
                run_identity["initialization"] = {
                    **dict(scratch_initialization_identity),
                    "optimizer_state_empty": resume_path is None
                    and not bool(optimizer.state),
                    "cursors": {
                        "epoch": 0,
                        "successful_updates": 0,
                        "data_frontier": 0,
                    },
                }
                if optimizer_partition is None:
                    raise RuntimeError("scratch optimizer partition receipt is missing")
                run_identity["optimization"] = {
                    "scheme": optimizer_partition.scheme,
                    "ordered_group_names": list(
                        optimizer_partition.ordered_group_names
                    ),
                    "partition_validated": optimizer_partition.partition_validated,
                    "groups": {
                        name: {
                            "ordered_fqns": list(group.ordered_fqns),
                            "fqn_sha256": group.fqn_sha256,
                            "peak_lr": group.peak_lr,
                            "clip_norm": group.clip_norm,
                        }
                        for name, group in optimizer_partition.groups.items()
                    },
                }
            else:
                run_identity["warm_start"] = (
                    {
                        "q_source_sha256": warm_receipt.q_source_sha256,
                        "agf_source_sha256": warm_receipt.agf_source_sha256,
                        "imported_keys": list(warm_receipt.imported_keys),
                        "ignored_keys": list(warm_receipt.ignored_keys),
                        "final_tensor_state_sha256": warm_receipt.final_tensor_state_sha256,
                    }
                    if warm_receipt is not None
                    else {"mode": "exact_resume", "checkpoint": str(resume_path)}
                )
            if qualification_update_limit is not None:
                run_identity.update(execution_kind="qualification",
                                    qualification_update_limit=qualification_update_limit,
                                    forensic_not_resume=True, disposable_weights=True)
            identity_path = run_root / "run_identity.json"
            resolved_path = run_root / "resolved_config.json"
            if resume_path is None:
                _write_atomic_json(resolved_path, admitted)
                _write_atomic_json(identity_path, run_identity)
            else:
                if not identity_path.is_file() or not resolved_path.is_file():
                    raise ValueError("exact resume requires the original run identity and resolved config")
                try:
                    original_identity = json.loads(identity_path.read_text(encoding="utf-8"))
                    original_config = json.loads(resolved_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError("original run identity/config is unreadable") from exc
                if not isinstance(original_identity, Mapping) or not isinstance(original_config, Mapping):
                    raise ValueError("original run identity/config is invalid")
                if _scientific_config_identity(original_config) != checkpoint_identity["config_sha256"]:
                    raise ValueError("original resolved config disagrees with resume identity")
                for name, expected in checkpoint_identity.items():
                    if original_identity.get(name) != expected:
                        raise ValueError("original run identity disagrees with the resume checkpoint")
        if world_size > 1:
            torch.distributed.barrier()

        store = TrainingMetricsStore(run_root)
        train_history: list[float] = []
        dev_history: list[float] = []
        final_checkpoint_path = ""
        final_checkpoint_sha = ""
        completed_epoch = start_epoch - 1
        epoch1_q_gradient_median: float | None = None
        stopped_early = False
        gate_identity: Mapping[str, str] | None = None
        if scratch:
            if scratch_training_contract is None:
                raise RuntimeError("scratch gate training contract is missing")
            gate_identity = {
                "training_contract_id": scratch_training_contract["id"],
                "optimization_protocol_sha256": scratch_training_contract[
                    "optimization_protocol_sha256"
                ],
                **checkpoint_identity,
            }
            if resume_path is not None:
                completed_gate_epoch = min(start_epoch - 1, 2)
                for gate_epoch in range(1, completed_gate_epoch + 1):
                    admitted_gate = _load_canonical_q_gradient_gate(
                        output_root=run_root,
                        epoch=gate_epoch,
                        world_size=world_size,
                        identity=gate_identity,
                        epoch1_median=epoch1_q_gradient_median,
                    )
                    if gate_epoch == 1:
                        epoch1_q_gradient_median = float(
                            admitted_gate["epoch_median"]
                        )
        # The admitted dataset and first-use order are deterministic. Keep
        # persistent workers (and their successfully verified cache shards)
        # across epochs, but start each ordered iterator only after its gates.
        train_loader: DataLoader | None = None
        dev_loader: DataLoader | None = None
        anomaly_capture = None
        if "anomaly_forensics" in training:
            from j2j.context4_forensics import FiniteGradientCapture
            anomaly_capture = FiniteGradientCapture(run_root / "data" / "gradient_forensics",
                                                    **training["anomaly_forensics"])
        for epoch in range(start_epoch, int(training["epochs"]) + 1):
            if train_loader is None or workers == 0:
                train_loader = _build_rank_dataloader(
                    train_dataset,
                    plan,
                    rank=rank,
                    workers=workers,
                    prefetch=prefetch,
                    pin_memory=pin_memory,
                    seed=seed + epoch * 10_000 + rank,
                    variant_contract=variant_contract,
                )
            train_proxy = _OrderedDataLoaderProxy(train_loader)
            train_receipt = run_context4_epoch(
                dataset=train_proxy,
                descriptors=train_descriptors,
                q_goal_views_by_trajectory=train_goals,
                model=train_model,
                optimizer=optimizer,
                scheduler=scheduler,
                output_root=run_root,
                rank=rank,
                world_size=world_size,
                effective_global_batch=effective_batch,
                microbatch_per_rank=int(training["microbatch_per_rank"]),
                accumulation_steps=int(training["accumulation_steps"]),
                horizon=training_q_horizon,
                max_grad_norm=(
                    None
                    if scratch
                    else float(training["gradient_clip_norm"])
                ),
                gradient_clip_norms=(
                    scratch_gradient_clip_norms if scratch else None
                ),
                optimizer_partition=(optimizer_partition if scratch else None),
                training_contract=(
                    scratch_training_contract if scratch else None
                ),
                initialization_identity=(
                    scratch_initialization_identity if scratch else None
                ),
                epoch=epoch,
                successful_updates=successful_updates,
                checkpoint_identity=checkpoint_identity,
                diagnostic_interval_updates=int(
                    training["diagnostic_interval_updates"]
                ),
                model_probe_config=training.get("model_probe"),
                anomaly_capture=anomaly_capture,
                variant_contract=variant_contract,
                **({"qualification_update_limit": qualification_update_limit} if qualification_update_limit is not None else {}),
            )
            if isinstance(train_receipt, QualificationRunReceipt):
                from j2j.context4_forensics import _cpu_copy, _save_exclusive, _agree_error
                rng_states = _gather_rng_states(rank=rank, world_size=world_size, device=device)
                error = None
                if rank == 0:
                    try:
                        state = {"schema": "j2j.qualification_state.v1", "forensic_not_resume": True,
                                 "epoch_completed": False, "epoch": epoch,
                                 "successful_updates": train_receipt.successful_updates,
                                 "next_update_index": train_receipt.successful_updates,
                                 "model": _cpu_copy(joint_model.state_dict()),
                                 "optimizer": _cpu_copy(optimizer.state_dict()),
                                 "scheduler": _cpu_copy(scheduler.state_dict()),
                                 "rng_state_by_rank": rng_states, "identity": checkpoint_identity,
                                 "variant_identity": resolved_variant}
                        state_identity = _save_exclusive(run_root / "qualification_state.pt", state)
                        _write_atomic_json(run_root / "qualification.json", {
                            "schema": "j2j.qualification_result.v1", "status": "WINDOW_COMPLETED_PENDING_AUDIT",
                            "forensic_not_resume": True, "epoch_completed": False,
                            "budget_updates": qualification_update_limit, "training_schedule_updates": total_updates,
                            "variant_identity": resolved_variant, "identity": checkpoint_identity,
                            "state": state_identity, "receipt": asdict(train_receipt)})
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                _agree_error(error, world_size)
                return train_receipt
            train_proxy.assert_exhausted()
            successful_updates = train_receipt.successful_updates
            train_history.append(float(train_receipt.losses["total"]))
            completed_epoch = epoch
            if rank == 0:
                final_checkpoint_path = train_receipt.checkpoint_path
                final_checkpoint_sha = train_receipt.checkpoint_sha256
            if validation_enabled:
                if dev_loader is None or workers == 0:
                    dev_plan = build_rank_origin_plan(
                        dev_descriptors,
                        world_size=world_size,
                        effective_batch=effective_batch,
                    )
                    dev_loader = _build_rank_dataloader(
                        dev_dataset,
                        dev_plan,
                        rank=rank,
                        workers=workers,
                        prefetch=prefetch,
                        pin_memory=pin_memory,
                        seed=seed + 1_000_000 + epoch * 10_000 + rank,
                        variant_contract=variant_contract,
                    )
                dev_proxy = _OrderedDataLoaderProxy(dev_loader)
                validation = evaluate_context4(
                    dataset=dev_proxy,
                    descriptors=dev_descriptors,
                    q_goal_views_by_trajectory=dev_goals,
                    model=train_model,
                    rank=rank,
                    world_size=world_size,
                    effective_global_batch=effective_batch,
                    microbatch_per_rank=int(training["microbatch_per_rank"]),
                    horizon=training_q_horizon,
                    variant_contract=variant_contract,
                    model_probe_config=training.get("model_probe"),
                )
                dev_proxy.assert_exhausted()
                if validation.model_probe is not None:
                    store.write_model_probe({"rank": rank, "phase": "dev", "epoch": epoch,
                        "successful_update": successful_updates, "variant_identity": resolved_variant,
                        "model_probe": validation.model_probe})
                dev_history.append(float(validation.losses["total"]))
                if rank == 0:
                    store.write_validation(
                        {
                            "epoch": epoch,
                            "variant_identity": resolved_variant,
                            "train": dict(train_receipt.losses),
                            "dev": dict(validation.losses),
                            "dev_numerator": dict(validation.numerators),
                            "dev_denominator": dict(validation.denominators),
                            **({"stochastic_q": _stochastic_probability_record(
                                validation.stochastic_statistics, kl_beta=joint_model.kl_beta,
                                sampling_namespace="dev/v1")}
                               if validation.stochastic_statistics is not None else {}),
                            "generalization_gap": dev_history[-1] - train_history[-1],
                            "fit_assessment": _fit_assessment(train_history, dev_history),
                            "source_reads": validation.source_reads,
                        }
                    )
            if scratch and epoch in (1, 2):
                if gate_identity is None:
                    raise RuntimeError("scratch early-gate identity is missing")
                local_gate_summary = summarize_q_gradient_epoch(
                    epoch=epoch,
                    q_pre_clip_grad_norms=train_receipt.q_pre_clip_grad_norms,
                )
                gate_receipt = synchronize_q_gradient_early_gate(
                    local_summary=local_gate_summary,
                    epoch1_median=epoch1_q_gradient_median,
                    rank=rank,
                    world_size=world_size,
                    output_root=run_root,
                    identity=gate_identity,
                )
                if epoch == 1:
                    canonical_epoch_median = gate_receipt.get("epoch_median")
                    if canonical_epoch_median is None and world_size == 1:
                        canonical_epoch_median = local_gate_summary.epoch_median
                    if (
                        isinstance(canonical_epoch_median, bool)
                        or not isinstance(canonical_epoch_median, (int, float))
                        or not math.isfinite(float(canonical_epoch_median))
                        or float(canonical_epoch_median) < 0.0
                    ):
                        raise RuntimeError(
                            "epoch 1 early gate omitted its canonical median"
                        )
                    epoch1_q_gradient_median = float(canonical_epoch_median)
                if gate_receipt.get("status") == "STOPPED_EARLY":
                    stopped_early = True
                    break
        if world_size > 1:
            torch.distributed.barrier()
        return TrainingRunReceipt(
            run_kind=str(training["run_kind"]),
            rank=rank,
            world_size=world_size,
            start_epoch=start_epoch,
            completed_epoch=completed_epoch,
            successful_updates=successful_updates,
            plan_sha256=plan_sha,
            config_sha256=checkpoint_identity["config_sha256"],
            code_sha256=checkpoint_identity["code_sha256"],
            final_checkpoint_path=final_checkpoint_path,
            final_checkpoint_sha256=final_checkpoint_sha,
            stopped_early=stopped_early,
        )
    except Exception as exc:
        if rank == 0:
            try:
                _write_atomic_json(
                    run_root / "data" / "training_metrics" / "raw" / "failure.json",
                    {
                        "schema": "J2J_CONTEXT4_FAILURE_V1",
                        "status": "FAILED_CLOSED",
                        "exception_type": type(exc).__name__,
                        "variant_identity": resolved_variant,
                    },
                )
            except Exception:
                # Never mask the scientific/runtime failure with a secondary
                # receipt write failure.
                pass
        raise
    finally:
        if created_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
