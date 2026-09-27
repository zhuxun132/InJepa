"""Strict, model-only Context4 evaluation checkpoint loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from j2j.authority import sha256_file
from j2j.spatial_intact import Context4SpatialJointModel
from j2j.context4_variants import (
    FULL_IDENTITY,
    RAW_IDENTITY,
    STOCHASTIC_IDENTITY,
    STABLE_STOCHASTIC_IDENTITY,
    RECURRENT_STOCHASTIC_IDENTITY,
    RECURRENT_STOCHASTIC_IDENTITIES,
    RAW_IDENTITIES,
    resolve_variant_contract,
    variant_identity_dict,
)

from .identity import (
    CURRENT_TRAINABLE_PARAMETERS,
    STOCHASTIC_TRAINABLE_PARAMETERS,
    STABLE_STOCHASTIC_TRAINABLE_PARAMETERS,
    RECURRENT_STOCHASTIC_TRAINABLE_PARAMETERS,
    derive_variant_identity,
    scientific_config_sha256,
    validate_current_model_architecture,
)


_CHECKPOINT_SCHEMA = "J2J_CONTEXT4_EPOCH_CHECKPOINT_V1"
_CHECKPOINT_IDENTITY_FIELDS = {
    "plan_sha256",
    "config_sha256",
    "code_sha256",
    "next_batch_sha256",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LoadedEvalCheckpoint:
    """Evaluation-only model and immutable admitted provenance."""

    model: nn.Module
    training_config: Mapping[str, object]
    variant_identity: Mapping[str, str]
    provenance: Mapping[str, object]


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _verify_file(path: str | Path, expected_sha256: str, *, name: str) -> Path:
    expected = _require_sha256(f"expected {name} SHA", expected_sha256)
    resolved = Path(path)
    observed = sha256_file(resolved)
    if observed != expected:
        raise ValueError(
            f"{name} bytes/SHA mismatch: expected {expected}, observed {observed}"
        )
    return resolved


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"training sidecar contains duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_sidecar(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"training sidecar contains non-finite JSON value: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("training sidecar is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError("training sidecar root must be a mapping")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _model_kwargs(training_config: Mapping[str, Any]) -> dict[str, object]:
    model = validate_current_model_architecture(training_config)

    kwargs = {
        "global_seed": model["global_seed"],
        "modes": model["modes"],
        "horizon": model["horizon"],
        "latent_dim": model["latent_dim"],
        "grid_side": model["grid_side"],
        "hidden_dim": model["hidden_dim"],
        "heads": model["heads"],
        "ffn_dim": model["ffn_dim"],
        "proposal_depth": model["proposal_depth"],
        "actor_depth": model["actor_depth"],
        "forward_depth": model["forward_depth"],
        "dropout": float(model["dropout"]),
    }
    for name in ("attention_backend", "activation_checkpointing", "memory_norm", "prediction_norm",
                 "proposal_architecture", "trajectory_latent_dim", "kl_beta", "sigma_epsilon",
                 "posterior_residual_init_std", "stability", "retain_activation_blocks",
                 "training_qkv_normalization_backend", "training_q_horizon"):
        if name in model:
            kwargs[name] = model[name]
    return kwargs


def _exact_final_identity(
    training_config: Mapping[str, Any], provenance: Mapping[str, Any]
) -> dict[str, int]:
    training = training_config.get("training")
    ledger = training_config.get("ledger")
    if not isinstance(training, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError(
            "exact-final checkpoint requires training and ledger sidecar mappings"
        )
    if training.get("run_kind") != "formal":
        raise ValueError("exact-final checkpoint requires a formal training run")
    epochs = training.get("epochs")
    effective_batch = training.get("effective_global_batch")
    factual_origins = ledger.get("factual_origins")
    if any(
        type(value) is not int or value <= 0
        for value in (epochs, effective_batch, factual_origins)
    ):
        raise ValueError(
            "exact-final epoch/update derivation is absent from the training sidecar"
        )
    expected_updates = math.ceil(factual_origins / effective_batch) * epochs
    if (
        provenance.get("checkpoint_epoch") != epochs
        or provenance.get("successful_updates") != expected_updates
    ):
        raise ValueError(
            "checkpoint is not exact-final: expected epoch "
            f"{epochs} and {expected_updates} successful updates"
        )
    return {
        "epoch": epochs,
        "successful_updates": expected_updates,
        "effective_global_batch": effective_batch,
        "factual_origins": factual_origins,
    }


def require_exact_final_checkpoint(loaded: Any) -> Mapping[str, int]:
    """Validate one loaded checkpoint against its own formal sidecar ledger."""

    training_config = getattr(loaded, "training_config", None)
    provenance = getattr(loaded, "provenance", None)
    if not isinstance(training_config, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("exact-final checkpoint provenance is absent")
    return MappingProxyType(_exact_final_identity(training_config, provenance))


def _completed_epoch_identity(training_config, provenance):
    training = training_config.get("training", {})
    ledger = training_config.get("ledger", {})
    if not isinstance(training, Mapping) or not isinstance(ledger, Mapping):
        raise ValueError("completed epoch requires training and ledger mappings")
    epochs = training.get("epochs")
    batch = training.get("effective_global_batch")
    origins = ledger.get("factual_origins")
    epoch = provenance.get("checkpoint_epoch")
    updates = provenance.get("successful_updates")
    if training.get("run_kind") != "formal" or any(
        type(value) is not int or value <= 0
        for value in (epochs, batch, origins, epoch, updates)
    ):
        raise ValueError("completed epoch requires a formal run and positive integer counters")
    if epoch > epochs or updates != math.ceil(origins / batch) * epoch:
        raise ValueError("checkpoint is not a completed epoch of its training ledger")
    return {"epoch": epoch, "successful_updates": updates,
            "effective_global_batch": batch, "factual_origins": origins}


def require_completed_epoch_checkpoint(loaded: Any) -> Mapping[str, int]:
    """Admit a completed formal-training epoch for explicit diagnostics only."""
    training = getattr(loaded, "training_config", None)
    provenance = getattr(loaded, "provenance", None)
    if not isinstance(training, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("completed epoch checkpoint provenance is absent")
    return MappingProxyType(_completed_epoch_identity(training, provenance))


def _checkpoint_variant_identity(
    payload: Mapping[str, Any],
    sidecar_identity: Mapping[str, str],
) -> Mapping[str, str] | None:
    sidecar_contract = resolve_variant_contract(sidecar_identity)
    if sidecar_contract.identity == FULL_IDENTITY:
        if "variant_identity" in payload:
            raise ValueError("Full checkpoint must not carry a native variant identity")
        return None
    native = payload.get("variant_identity")
    if not isinstance(native, Mapping):
        raise ValueError("control checkpoint native variant identity tuple is missing")
    checkpoint_contract = resolve_variant_contract(native)
    if checkpoint_contract.identity != sidecar_contract.identity:
        raise ValueError("checkpoint and training sidecar variant identity tuples disagree")
    return MappingProxyType(variant_identity_dict(checkpoint_contract.identity))


def load_eval_checkpoint(
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    training_resolved_config_path: str | Path,
    expected_training_sidecar_sha256: str,
    expected_training_code_sha256: str,
    *,
    device: str | torch.device = "cpu",
    checkpoint_admission: str = "exact_final",
) -> LoadedEvalCheckpoint:
    """Verify an admitted bundle and strict-load only its model state.

    Optimizer, scheduler, and serialized RNG fields are intentionally ignored.
    The caller's CPU RNG state is restored even if model construction fails.
    """
    training_code_sha = _require_sha256(
        "expected training code SHA", expected_training_code_sha256
    )

    # Both raw byte identities are established before parsing or constructing.
    checkpoint_file = _verify_file(
        checkpoint_path, expected_checkpoint_sha256, name="checkpoint"
    )
    sidecar_file = _verify_file(
        training_resolved_config_path,
        expected_training_sidecar_sha256,
        name="training sidecar",
    )

    training_config = _load_sidecar(sidecar_file)
    payload = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    checkpoint_schema = payload.get("schema")
    if checkpoint_schema not in {_CHECKPOINT_SCHEMA, "J2J_CONTEXT4_EPOCH_CHECKPOINT_V2"}:
        raise ValueError("checkpoint schema is not an admitted Context4 epoch schema")
    checkpoint_identity = payload.get("identity")
    if not isinstance(checkpoint_identity, Mapping):
        raise TypeError("checkpoint identity must be a mapping")
    if set(checkpoint_identity) != _CHECKPOINT_IDENTITY_FIELDS:
        raise ValueError("checkpoint must contain exactly four identity fields")
    for key in sorted(_CHECKPOINT_IDENTITY_FIELDS):
        _require_sha256(f"checkpoint identity.{key}", checkpoint_identity.get(key))

    observed_config_sha = scientific_config_sha256(training_config)
    if checkpoint_identity["config_sha256"] != observed_config_sha:
        raise ValueError("checkpoint config SHA does not match the training sidecar")
    if checkpoint_identity["code_sha256"] != training_code_sha:
        raise ValueError("checkpoint training code SHA does not match admission")

    variant_identity = derive_variant_identity(training_config)
    native_raw = variant_identity["variant_id"] in {identity.variant_id for identity in RAW_IDENTITIES}
    if native_raw and checkpoint_schema != "J2J_CONTEXT4_EPOCH_CHECKPOINT_V2":
        raise ValueError("native RAW training requires its scratch V2 checkpoint identity")
    if checkpoint_schema == "J2J_CONTEXT4_EPOCH_CHECKPOINT_V2":
        from j2j.context4_training import _validated_training_contract, _validated_initialization_identity
        training_contract = _validated_training_contract(training_config.get("training_contract"))
        if _validated_training_contract(payload.get("training_contract")) != training_contract:
            raise ValueError("checkpoint training contract differs from its source sidecar")
        initialization = _validated_initialization_identity(payload.get("initialization_identity"))
        if initialization["global_seed"] != training_config["model"]["global_seed"]:
            raise ValueError("checkpoint initialization seed differs from source model")
    checkpoint_variant_identity = _checkpoint_variant_identity(
        payload,
        variant_identity,
    )
    kwargs = _model_kwargs(training_config)
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping):
        raise TypeError("checkpoint model_state must be a mapping")
    epoch = payload.get("epoch")
    successful_updates = payload.get("successful_updates")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("checkpoint epoch must be a non-negative integer")
    if type(successful_updates) is not int or successful_updates < 0:
        raise ValueError("checkpoint successful_updates must be a non-negative integer")
    counters = {"checkpoint_epoch": epoch, "successful_updates": successful_updates}
    completed_identity = None
    if checkpoint_admission == "exact_final":
        final_identity = _exact_final_identity(training_config, counters)
    elif checkpoint_admission == "completed_epoch":
        completed_identity = _completed_epoch_identity(training_config, counters)
        final_identity = (_exact_final_identity(training_config, counters)
                          if epoch == training_config["training"]["epochs"] else None)
    else:
        raise ValueError("unknown checkpoint admission policy")

    # Context4 initializes under its own fork_rng, but preserve this API-level
    # invariant even for alternate/test module constructors.
    with torch.random.fork_rng(devices=[]):
        model = Context4SpatialJointModel(**kwargs)
        model.load_state_dict(model_state, strict=True)
        model.to(device)
        model.eval()

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected_parameters = (RECURRENT_STOCHASTIC_TRAINABLE_PARAMETERS if variant_identity["variant_id"] in {v.variant_id for v in RECURRENT_STOCHASTIC_IDENTITIES} else
                           STABLE_STOCHASTIC_TRAINABLE_PARAMETERS if variant_identity["variant_id"] == STABLE_STOCHASTIC_IDENTITY.variant_id else
                           STOCHASTIC_TRAINABLE_PARAMETERS if variant_identity["variant_id"] == STOCHASTIC_IDENTITY.variant_id
                           else CURRENT_TRAINABLE_PARAMETERS)
    if trainable_parameters != expected_parameters:
        raise ValueError(
            "current model trainable parameter count must be exactly "
            f"{expected_parameters:,}; observed {trainable_parameters:,}"
        )

    first_parameter = next(model.parameters(), None)
    loaded_device = (
        str(first_parameter.device)
        if first_parameter is not None
        else str(torch.device(device))
    )

    provenance_record: dict[str, object] = {
            "schema": checkpoint_schema,
            "load_status": "STRICT_MODEL_ONLY_LOADED",
            "strict_model_state": True,
            "optimizer_scheduler_rng_restored": False,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "training_sidecar_sha256": expected_training_sidecar_sha256,
            "training_code_sha256": training_code_sha,
            "checkpoint_identity": dict(checkpoint_identity),
            "checkpoint_epoch": epoch,
            "successful_updates": successful_updates,
            "exact_final_identity": final_identity,
            "completed_epoch_identity": completed_identity,
            "trainable_parameters": trainable_parameters,
            "loaded_device": loaded_device,
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "evaluator_source_sha256": {
                "checkpoint.py": sha256_file(Path(__file__)),
                "identity.py": sha256_file(Path(__file__).with_name("identity.py")),
            },
        }
    if checkpoint_variant_identity is not None:
        provenance_record["checkpoint_variant_identity"] = dict(
            checkpoint_variant_identity
        )
    provenance = _freeze(provenance_record)
    assert isinstance(provenance, Mapping)
    return LoadedEvalCheckpoint(
        model=model,
        training_config=_freeze(training_config),  # type: ignore[arg-type]
        variant_identity=variant_identity,
        provenance=provenance,
    )


__all__ = [
    "Context4SpatialJointModel",
    "LoadedEvalCheckpoint",
    "load_eval_checkpoint",
    "require_exact_final_checkpoint",
]
