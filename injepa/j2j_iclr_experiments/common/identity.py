"""Checkpoint-bound scientific and variant identities."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
import warnings

from j2j.context4_training import _scientific_config_identity
from j2j.context4_variants import (
    FULL_IDENTITY,
    RAW_IDENTITY,
    STOCHASTIC_IDENTITY,
    STABLE_STOCHASTIC_IDENTITY,
    RECURRENT_STOCHASTIC_IDENTITY,
    RECURRENT_STOCHASTIC_IDENTITIES,
    STABLE_STOCHASTIC_IDENTITIES,
    STOCHASTIC_IDENTITIES,
    RAW_IDENTITIES,
    resolve_variant_contract,
    variant_identity_dict,
)


FULL_V1_VARIANT_ID = "context4_no_selector_spatial_joint_v1"
FULL_V1_DESIGN_SHA256 = (
    "06e614c0bcdef6ef7fd99d7961670bf726faa62850ab705262284e3b460d1474"
)
FULL_V1_FORMAL_PROTOCOL_SHA256 = (
    "592fcc6b54f7715ed95cb569a361903e42ede446063c3baea4828092e8a53ccb"
)
CURRENT_MODEL_ARCHITECTURE = MappingProxyType(
    {
        "context_size": 4,
        "selector_enabled": False,
        "modes": 4,
        "horizon": 4,
        "latent_dim": 768,
        "grid_side": 6,
        "hidden_dim": 768,
        "heads": 16,
        "ffn_dim": 2048,
        "proposal_depth": 6,
        "actor_depth": 3,
        "forward_depth": 6,
        "dropout": 0.1,
    }
)
CURRENT_TRAINABLE_PARAMETERS = 120_758_809
STOCHASTIC_TRAINABLE_PARAMETERS = 148_827_672
STABLE_STOCHASTIC_TRAINABLE_PARAMETERS = 148_827_992
RECURRENT_STOCHASTIC_TRAINABLE_PARAMETERS = 135_235_064


def _validate_native_model(sidecar, model):
    contract = resolve_variant_contract(sidecar.get("variant_identity"))
    if contract.identity not in RAW_IDENTITIES:
        raise ValueError("native architecture requires a registered RAW scientific family")
    if sidecar.get("experiment_id") != contract.identity.variant_id:
        raise ValueError("native model experiment and variant identities differ")
    from j2j.context4_training import _validated_training_contract
    training_contract = _validated_training_contract(sidecar.get("training_contract"))
    if training_contract["optimization_protocol_sha256"] != contract.identity.transform_spec_sha:
        raise ValueError("native model training protocol differs from its variant")
    if sidecar.get("initialization") != {"mode": "scratch_trainable", "schema": "j2j_context4_trainable_init_v1"}:
        raise ValueError("native model requires exact scratch initialization")
    data = sidecar.get("data")
    if not isinstance(data, Mapping) or data.get("require_stage") != "RAW32":
        raise ValueError("native model requires RAW32 training data")
    coordinates = data.get("expected_identities")
    if not isinstance(coordinates, Mapping) or coordinates.get("whitening_sha256", "missing") is not None:
        raise ValueError("native model forbids whitening")
    training = sidecar.get("training")
    qg = "posterior_sample" if contract.identity in STOCHASTIC_IDENTITIES else "posterior_weighted"
    if not isinstance(training, Mapping) or training.get("qg_objective") != qg:
        raise ValueError("native QG objective differs from its scientific family")
    source_identity = sidecar.get("identity", {})
    if (source_identity.get("design_sha256") != contract.identity.transform_spec_sha or
        source_identity.get("formal_protocol_sha256") != contract.identity.transform_spec_sha):
        raise ValueError("native model source protocol identity mismatch")
    extra = {"representation", "attention_backend", "activation_checkpointing", "memory_norm", "prediction_norm"}
    if "training_qkv_normalization_backend" in model:
        from j2j.proposal.stability import validate_training_qkv_normalization_backend
        validate_training_qkv_normalization_backend(
            model["training_qkv_normalization_backend"],
            stable=contract.identity in STABLE_STOCHASTIC_IDENTITIES)
        extra.add("training_qkv_normalization_backend")
    if "retain_activation_blocks" in model:
        from j2j.proposal.blocks import validate_retained_activation_blocks
        if not isinstance(model["retain_activation_blocks"], list):
            raise TypeError("sidecar retain_activation_blocks must be a JSON list")
        validate_retained_activation_blocks(
            model["retain_activation_blocks"], activation_checkpointing=model.get("activation_checkpointing"),
            depths={"proposal.future.blocks": model.get("proposal_depth", 0),
                    "forward_core.blocks": model.get("forward_depth", 0),
                    "actor.blocks": model.get("actor_depth", 0)})
        extra.add("retain_activation_blocks")
    stochastic_values = {"proposal_architecture": "stochastic_divided", "trajectory_latent_dim": 128,
                         "kl_beta": 0.05, "sigma_epsilon": 1e-4, "posterior_residual_init_std": 1e-3}
    if contract.identity in RECURRENT_STOCHASTIC_IDENTITIES:
        stochastic_values.update(proposal_architecture="stochastic_recurrent",
                                 training_q_horizon=1, goal_sampling_horizon=4)
    if contract.identity in STOCHASTIC_IDENTITIES:
        extra |= set(stochastic_values)
    if contract.identity in STABLE_STOCHASTIC_IDENTITIES:
        from dataclasses import asdict
        from j2j.proposal.stability import StabilityConfig
        controls = model.get("stability")
        parsed = StabilityConfig.from_mapping(controls)
        if dict(controls) != asdict(StabilityConfig()) or asdict(parsed) != dict(controls):
            raise ValueError("v2 sidecar requires complete frozen stability controls")
        extra.add("stability")
    if set(model) != {"global_seed", *CURRENT_MODEL_ARCHITECTURE, *extra}:
        raise ValueError("native model architecture fields are incomplete or unknown")
    for field, expected in CURRENT_MODEL_ARCHITECTURE.items():
        if field in {"grid_side", "modes", "horizon"}:
            if type(model[field]) is not int or model[field] <= 0:
                raise ValueError(f"native model {field} must be a positive integer")
        elif type(model[field]) is not type(expected) or model[field] != expected:
            raise ValueError(f"native model {field} differs from the reviewed architecture")
    if type(model["global_seed"]) is not int or model["global_seed"] != 3072:
        raise ValueError("native scratch model seed must match its training contract")
    if model["grid_side"] != 24:
        warnings.warn("Non-default native grid; encoder/cache/model coordinates must match.", UserWarning)
    if (model["representation"] != "raw" or model["memory_norm"] is not True or
        model["prediction_norm"] != "layernorm" or model["attention_backend"] not in {"math", "auto"} or
        type(model["activation_checkpointing"]) is not bool):
        raise ValueError("native model representation/normalization/execution fields are invalid")
    if contract.identity in STOCHASTIC_IDENTITIES:
        for field, expected in stochastic_values.items():
            if type(model[field]) is not type(expected) or model[field] != expected:
                raise ValueError(f"stochastic model {field} differs from its frozen first-version factor")
    return model


def scientific_config_sha256(config: Mapping[str, object]) -> str:
    """Return the exact training-side scientific-config projection digest."""
    if not isinstance(config, Mapping):
        raise TypeError("scientific config must be a mapping")
    return _scientific_config_identity(config)


def validate_current_model_architecture(
    training_sidecar: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Require the one current Context4 model while keeping seed parameterized."""

    model = training_sidecar.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("current model architecture must be a mapping")
    if training_sidecar.get("experiment_id") in {identity.variant_id for identity in RAW_IDENTITIES}:
        return _validate_native_model(training_sidecar, model)
    expected_fields = {"global_seed", *CURRENT_MODEL_ARCHITECTURE}
    if set(model) != expected_fields:
        raise ValueError("current model architecture fields are incomplete or unknown")
    seed = model.get("global_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("current model.global_seed must be a non-negative integer")
    for field, expected in CURRENT_MODEL_ARCHITECTURE.items():
        observed = model.get(field)
        if field == "dropout":
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or float(observed) != float(expected)
            ):
                raise ValueError(
                    f"current model architecture requires frozen {field}={expected!r}"
                )
        elif type(observed) is not type(expected) or observed != expected:
            raise ValueError(
                f"current model architecture requires frozen {field}={expected!r}"
            )
    return model


def _native_variant_identity(
    sidecar: Mapping[str, Any], experiment_id: str
) -> Mapping[str, str]:
    native = sidecar.get("variant_identity")
    if not isinstance(native, Mapping):
        raise ValueError(
            "trained control requires its frozen native variant identity; "
            "Full-v1 exact derivation is forbidden"
        )
    contract = resolve_variant_contract(native)
    if contract.identity == FULL_IDENTITY:
        raise ValueError("Full-v1 may not use a native variant identity")
    if contract.identity.variant_id != experiment_id:
        raise ValueError("native variant identity must match the training experiment_id")
    return MappingProxyType(variant_identity_dict(contract.identity))


def derive_variant_identity(training_sidecar: Mapping[str, object]) -> Mapping[str, str]:
    """Derive frozen Full or validate one exact trained-control identity.

    Full-v1 is the sole exact-derivation case. It is accepted only when all
    three checkpoint-bound source fields exactly match the frozen protocol.
    """
    if not isinstance(training_sidecar, Mapping):
        raise TypeError("training sidecar must be a mapping")
    experiment_id = training_sidecar.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("training sidecar variant experiment_id is missing")
    validate_current_model_architecture(training_sidecar)

    if experiment_id != FULL_V1_VARIANT_ID:
        return _native_variant_identity(training_sidecar, experiment_id)

    if "variant_identity" in training_sidecar:
        raise ValueError(
            "Full-v1 variant identity must come only from its exact source fields"
        )
    source_identity = training_sidecar.get("identity")
    if not isinstance(source_identity, Mapping):
        raise TypeError("Full-v1 training identity must be a mapping")
    if source_identity.get("design_sha256") != FULL_V1_DESIGN_SHA256:
        raise ValueError("Full-v1 design identity does not match the frozen design")
    if (
        source_identity.get("formal_protocol_sha256")
        != FULL_V1_FORMAL_PROTOCOL_SHA256
    ):
        raise ValueError("Full-v1 protocol identity does not match the frozen protocol")
    return MappingProxyType(variant_identity_dict(FULL_IDENTITY))


__all__ = [
    "CURRENT_MODEL_ARCHITECTURE",
    "CURRENT_TRAINABLE_PARAMETERS",
    "FULL_V1_DESIGN_SHA256",
    "FULL_V1_FORMAL_PROTOCOL_SHA256",
    "FULL_V1_VARIANT_ID",
    "derive_variant_identity",
    "scientific_config_sha256",
    "validate_current_model_architecture",
]
