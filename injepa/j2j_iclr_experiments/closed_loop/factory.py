"""Strict Context4 construction seam for the mature ImageGoal episode runner."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any

import torch

from j2j.context4_variants import resolve_variant_contract
from j2j.evaluation.vjepa_grid_encoder import build_vjepa_grid_encoder
from j2j_iclr_experiments.common.checkpoint import (
    load_eval_checkpoint,
    require_exact_final_checkpoint,
)
from j2j_iclr_experiments.common.artifacts import (
    canonical_mapping_sha256,
    file_identity,
    load_json_file_identity,
    to_plain_json,
)
from j2j_iclr_experiments.common.config import PREREGISTERED_ACTIVE_PREFIXES
from j2j_iclr_experiments.local_habitat.capability import (
    validate_formal_reset_replay_capability,
)
from j2j_iclr_experiments.online import Context4Planner, Context4PolicyAdapter
from j2j_iclr_experiments.online.stop import validate_stop_calibration_receipt

from .decision import (
    ANALYSIS_SCHEMA,
    resolve_variant_decision_row,
    validate_analysis_source_manifest,
    validate_five_row_run_decision,
)


class Context4ClosedLoopBlocked(RuntimeError):
    """A formal closed-loop artifact or capability gate is still open."""

    status = "BLOCKED"


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_IDENTITY = re.compile(r"[0-9a-f]{40}")
_COORDINATE_FIELDS = {
    "vjepa_source_commit",
    "vjepa_source_tree",
    "checkpoint_sha256",
    "preprocess_sha256",
    "pool_sha256",
    "whitening_sha256",
}
_TRAINING_COORDINATE_FIELDS = _COORDINATE_FIELDS - {"vjepa_source_tree"}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} must be a mapping")
    return value


def _verified_json(
    path_value: object,
    expected_sha256: object,
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(path_value, (str, Path)) or not str(path_value):
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} path is absent")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} is unavailable: {path}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} SHA-256 is absent")
    try:
        observed = file_identity(path, name=name)
    except (OSError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: {name} live file identity is invalid"
        ) from exc
    if observed["sha256"] != expected_sha256:
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} SHA-256 mismatch")
    try:
        payload, identity = load_json_file_identity(observed, name=name)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} is not valid JSON") from exc
    return payload, to_plain_json(identity)


def _verified_json_reference(reference: object, *, name: str):
    """Accept the canonical file identity as well as a path/SHA reference."""
    reference = _mapping(reference, name=name)
    if set(reference) not in ({"path", "sha256"}, {"path", "sha256", "bytes"}):
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} has noncanonical identity fields")
    if "bytes" in reference and (type(reference["bytes"]) is not int or reference["bytes"] <= 0):
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} bytes must be a positive integer")
    payload, identity = _verified_json(reference.get("path"), reference.get("sha256"), name=name)
    if "bytes" in reference and reference["bytes"] != identity["bytes"]:
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} file size mismatch")
    return payload, identity


def _validate_capability(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        validated = validate_formal_reset_replay_capability(receipt)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: formal Habitat reset-replay capability is invalid: {exc}"
        ) from exc
    if not isinstance(validated, Mapping):
        raise Context4ClosedLoopBlocked(
            "BLOCKED: formal Habitat reset-replay capability validator returned no evidence"
        )
    return validated


def _coordinate_identity(value: object, *, name: str) -> dict[str, str]:
    raw = _mapping(value, name=name)
    if set(raw) != _COORDINATE_FIELDS:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: {name} visual coordinate identity fields are incomplete"
        )
    result = {field: str(raw[field]) for field in _COORDINATE_FIELDS}
    if _GIT_IDENTITY.fullmatch(result["vjepa_source_commit"]) is None:
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} source commit is invalid")
    if _GIT_IDENTITY.fullmatch(result["vjepa_source_tree"]) is None:
        raise Context4ClosedLoopBlocked(f"BLOCKED: {name} source tree is invalid")
    for field in _COORDINATE_FIELDS - {
        "vjepa_source_commit",
        "vjepa_source_tree",
    }:
        if _SHA256.fullmatch(result[field]) is None:
            raise Context4ClosedLoopBlocked(
                f"BLOCKED: {name} {field} identity is invalid"
            )
    return result


def _close_visual_coordinate_identity(
    *,
    loaded: Any,
    encoder_provenance: Mapping[str, Any],
    stop_receipt: Mapping[str, Any],
) -> dict[str, str]:
    data = _mapping(loaded.training_config.get("data"), name="training sidecar data")
    expected = _mapping(
        data.get("expected_identities"),
        name="training sidecar data.expected_identities",
    )
    missing = _TRAINING_COORDINATE_FIELDS - set(expected)
    if missing:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: training visual coordinate identity is incomplete"
        )
    training = {field: str(expected[field]) for field in _TRAINING_COORDINATE_FIELDS}
    runtime = _coordinate_identity(
        encoder_provenance.get("coordinate_identity"),
        name="runtime encoder",
    )
    stop_provenance = _mapping(
        stop_receipt.get("provenance"), name="STOP calibration provenance"
    )
    stop = _coordinate_identity(
        stop_provenance.get("visual_coordinate_identity"),
        name="STOP calibration",
    )
    if {field: runtime[field] for field in _TRAINING_COORDINATE_FIELDS} != training:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: training and runtime V-JEPA visual coordinate identities differ"
        )
    if stop != runtime:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: runtime and STOP visual coordinate identities differ"
        )

    load_ledger = _mapping(
        encoder_provenance.get("load_ledger"), name="runtime encoder load ledger"
    )
    if (
        load_ledger.get("load_status") != "STRICT_FROZEN_VJEPA2_LOADED"
        or load_ledger.get("source_commit") != runtime["vjepa_source_commit"]
        or load_ledger.get("source_tree") != runtime["vjepa_source_tree"]
        or load_ledger.get("checkpoint_sha256") != runtime["checkpoint_sha256"]
    ):
        raise Context4ClosedLoopBlocked(
            "BLOCKED: runtime V-JEPA strict-load ledger differs from coordinate identity"
        )
    runtime_checkpoint = _mapping(
        encoder_provenance.get("vjepa_checkpoint"), name="runtime V-JEPA checkpoint"
    )
    runtime_whitening = _mapping(
        encoder_provenance.get("whitening"), name="runtime whitening"
    )
    stop_artifacts = _mapping(
        stop_provenance.get("artifacts"), name="STOP calibration artifacts"
    )
    stop_checkpoint = _mapping(
        stop_artifacts.get("vjepa_checkpoint"), name="STOP V-JEPA checkpoint"
    )
    stop_whitening = _mapping(
        stop_artifacts.get("whitening"), name="STOP whitening"
    )
    if any(
        (
            runtime_checkpoint.get("sha256") != runtime["checkpoint_sha256"],
            stop_checkpoint.get("sha256") != runtime["checkpoint_sha256"],
            runtime_whitening.get("sha256") != runtime["whitening_sha256"],
            stop_whitening.get("sha256") != runtime["whitening_sha256"],
        )
    ):
        raise Context4ClosedLoopBlocked(
            "BLOCKED: V-JEPA checkpoint or whitening artifact identity drifted"
        )
    return runtime


def _current_ranker_identity(*, k_active: int, h_active: int) -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1] / "offline" / "rankers.py"
    return {
        "source": file_identity(source, name="current ranker source"),
        "full_deployment": "full",
        "no_f_deployment": "no_f",
        "k_active": k_active,
        "h_active": h_active,
        "tie_break": {
            "full": [
                "F_endpoint_goal_distance",
                "negative_log_mass",
                "global_k",
                "one_based_h",
            ],
            "no_f": [
                "Q_endpoint_goal_distance",
                "negative_log_mass",
                "global_k",
                "one_based_h",
            ],
        },
    }


def _current_candidate_contract(*, k_active: int, h_active: int) -> dict[str, Any]:
    count = k_active * h_active
    return {
        "motion_actions": ["FWD", "LEFT", "RIGHT"],
        "ordinary_stop_forbidden": True,
        "retain_all_finite": True,
        "full_count": count,
        "no_f_count": count,
        "local_methods": ["Full", "NoG-enumerate"],
        "no_g_tree_nodes": sum(3**h for h in range(1, h_active + 1)),
    }


def _admit_frozen_decision(
    *,
    section: Mapping[str, Any],
    loaded: Any,
    final_identity: Mapping[str, Any],
    checkpoint_config: Mapping[str, Any],
    deployment: str,
    row_id: str,
    k_active: int,
    h_active: int,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    """Admit analysis and decision bytes before STOP/capability/encoder work."""

    analysis_config = _mapping(
        section.get("analysis_manifest"), name="context4.analysis_manifest"
    )
    analysis_payload, analysis_identity = _verified_json_reference(
        analysis_config,
        name="analysis manifest",
    )
    protocol_sha = analysis_payload.get("protocol_sha256")
    try:
        validated_analysis = validate_analysis_source_manifest(
            analysis_payload,
            expected_protocol_sha256=protocol_sha,
            project_root=Path(__file__).resolve().parents[2],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: analysis manifest is invalid: {exc}"
        ) from exc
    if validated_analysis.get("schema") != ANALYSIS_SCHEMA:
        raise Context4ClosedLoopBlocked("BLOCKED: analysis manifest schema drift")

    decision_config = _mapping(section.get("decision"), name="context4.decision")
    decision_payload, decision_identity = _verified_json_reference(
        decision_config,
        name="five-row decision",
    )
    if decision_payload.get("protocol_sha256") != protocol_sha:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: decision and analysis protocol identities differ"
        )
    variant = _mapping(loaded.variant_identity, name="checkpoint variant identity")
    model = _mapping(loaded.training_config.get("model"), name="training sidecar model")
    training_seed = model.get("global_seed")
    if type(training_seed) is not int or training_seed < 0:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: checkpoint native training seed is absent"
        )
    checkpoint_identity = file_identity(
        checkpoint_config.get("path"), name="Context4 checkpoint"
    )
    sidecar_identity = file_identity(
        checkpoint_config.get("training_resolved_config_path"),
        name="Context4 training sidecar",
    )
    candidates = decision_payload.get("checkpoint_set")
    if not isinstance(candidates, list):
        raise Context4ClosedLoopBlocked("BLOCKED: decision checkpoint_set is invalid")
    matching = [
        row
        for row in candidates
        if isinstance(row, Mapping)
        and row.get("variant_id") == variant.get("variant_id")
        and row.get("training_seed") == training_seed
        and row.get("checkpoint") == checkpoint_identity
        and row.get("training_sidecar") == sidecar_identity
        and row.get("training_code_sha256")
        == checkpoint_config.get("training_code_sha256")
        and row.get("epoch") == final_identity.get("epoch")
        and row.get("exact_final") is True
    ]
    if len(matching) != 1:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: decision has no unique native-seed exact checkpoint entry"
        )
    checkpoint_entry = to_plain_json(matching[0])
    checkpoint_id = checkpoint_entry["checkpoint_id"]
    ranker = _current_ranker_identity(k_active=k_active, h_active=h_active)
    candidate = _current_candidate_contract(k_active=k_active, h_active=h_active)
    # The factory can validate the frozen matrix and every live artifact now.
    # The mature launcher later replaces the preflight/ledger/Habitat values
    # with its independently resolved live launch identities before Habitat.
    live = {
        "protocol_sha256": protocol_sha,
        "row_id": row_id,
        "deployment": deployment,
        "training_seed": training_seed,
        "checkpoint_id": checkpoint_id,
        "checkpoint_entry": checkpoint_entry,
        "episode_ledger": decision_payload.get("episode_ledger"),
        "stop_receipt": decision_payload.get("stop_receipt"),
        "preflight_receipt": decision_payload.get("preflight_receipt"),
        "habitat_scientific_identity": decision_payload.get(
            "habitat_scientific_identity"
        ),
        "ranker_identity": ranker,
        "candidate_contract": candidate,
        "analysis_manifest": analysis_identity,
    }
    try:
        validated_decision = validate_five_row_run_decision(
            decision_payload,
            expected_protocol_sha256=str(protocol_sha),
            expected_deployment=deployment,
            live_identity=live,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: five-row decision is invalid: {exc}"
        ) from exc
    admission = {
        "protocol_sha256": str(protocol_sha),
        "row_id": row_id,
        "deployment": deployment,
        "training_seed": training_seed,
        "checkpoint_id": checkpoint_id,
        "checkpoint_entry": checkpoint_entry,
        "ranker_identity": ranker,
        "candidate_contract": candidate,
    }
    return validated_decision, analysis_identity, {
        "decision": decision_identity,
        "analysis_manifest": analysis_identity,
        "admission": admission,
    }


def build_context4_adapter(
    config: Mapping[str, Any],
    checkpoint: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    *,
    checkpoint_loader=load_eval_checkpoint,
    encoder_builder=build_vjepa_grid_encoder,
) -> Context4PolicyAdapter:
    """Build one current Context4 adapter; the mature runner owns all episodes."""

    section = _mapping(config.get("context4"), name="config.context4")
    checkpoint_config = _mapping(section.get("checkpoint"), name="context4.checkpoint")
    runtime = _mapping(section.get("runtime", {}), name="context4.runtime")
    device = torch.device(str(runtime.get("device", "cpu")))
    try:
        loaded = checkpoint_loader(
            checkpoint or checkpoint_config.get("path"),
            checkpoint_sha256 or checkpoint_config.get("sha256"),
            checkpoint_config.get("training_resolved_config_path"),
            checkpoint_config.get("training_resolved_config_sha256"),
            checkpoint_config.get("training_code_sha256"),
            device=device,
        )
    except Exception as exc:
        raise Context4ClosedLoopBlocked(f"BLOCKED: checkpoint admission failed: {exc}") from exc
    try:
        final_identity = require_exact_final_checkpoint(loaded)
    except (TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(f"BLOCKED: {exc}") from exc
    variant_identity = _mapping(
        loaded.variant_identity,
        name="checkpoint variant identity",
    )
    try:
        variant_contract = resolve_variant_contract(variant_identity)
    except (KeyError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: current variant identity is not admitted: {exc}"
        ) from exc
    requested = _mapping(section.get("model"), name="context4.model")
    for field in ("k_active", "h_active"):
        value = requested.get(field)
        if type(value) is not int or value not in PREREGISTERED_ACTIVE_PREFIXES:
            raise Context4ClosedLoopBlocked(
                "BLOCKED: active K/H is outside the preregistered {1,2,4} set"
            )
    k_active = int(requested["k_active"])
    h_active = int(requested["h_active"])
    deployment = str(section.get("deployment", "full"))
    if deployment not in {"full", "no_f"}:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: current closed-loop deployment must be full or no_f"
        )
    try:
        row_id = resolve_variant_decision_row(
            variant_identity,
            deployment=deployment,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: variant/deployment row admission failed: {exc}"
        ) from exc

    if "decision" not in section or "analysis_manifest" not in section:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: both decision and analysis manifest identities are required"
        )
    effective_checkpoint = dict(checkpoint_config)
    if checkpoint is not None:
        effective_checkpoint["path"] = checkpoint
    if checkpoint_sha256 is not None:
        effective_checkpoint["sha256"] = checkpoint_sha256
    try:
        frozen_decision, _analysis_identity, frozen_artifacts = _admit_frozen_decision(
            section=section,
            loaded=loaded,
            final_identity=final_identity,
            checkpoint_config=effective_checkpoint,
            deployment=deployment,
            row_id=row_id,
            k_active=k_active,
            h_active=h_active,
        )
    except Context4ClosedLoopBlocked:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: frozen decision admission failed: {exc}"
        ) from exc

    stop = _mapping(section.get("stop"), name="context4.stop")
    if stop.get("mode") != "calibrated":
        raise Context4ClosedLoopBlocked("BLOCKED: formal STOP receipt is required")
    stop_payload, stop_identity = _verified_json(
        stop.get("receipt"), stop.get("receipt_sha256"), name="STOP receipt"
    )
    try:
        from j2j_iclr_experiments.online.stop_admission import (
            admit_stop_attestation, stop_admission_config_sha256, validate_stop_once,
        )
        stop_attestation_identity = None
        if stop.get("attestation") is not None:
            attestation_config = _mapping(stop["attestation"], name="STOP attestation")
            attestation_payload, stop_attestation_identity = _verified_json(
                attestation_config.get("path"), attestation_config.get("sha256"), name="STOP attestation"
            )
            stop_admission = admit_stop_attestation(stop_payload, attestation_payload, config_sha256=stop_admission_config_sha256(config))
        else:
            stop_admission = validate_stop_once(stop_payload)
        validated_stop = stop_admission.receipt
    except (TypeError, ValueError) as exc:
        raise Context4ClosedLoopBlocked(
            f"BLOCKED: STOP receipt is not protocol-complete: {exc}"
        ) from exc
    stop_provenance = _mapping(
        validated_stop.get("provenance"), name="STOP calibration provenance"
    )
    if (
        stop_provenance.get("protocol_sha256")
        != frozen_artifacts["admission"]["protocol_sha256"]
    ):
        raise Context4ClosedLoopBlocked(
            "BLOCKED: five-row decision protocol differs from the formal STOP protocol"
        )

    local = _mapping(section.get("local"), name="context4.local")
    capability_payload, capability_identity = _verified_json(
        local.get("capability_receipt"),
        local.get("capability_receipt_sha256"),
        name="Habitat reset-replay capability",
    )
    validated_capability = _validate_capability(capability_payload)
    scientific_identity = _mapping(
        validated_capability.get("scientific_identity"),
        name="Habitat reset-replay scientific identity",
    )
    if set(scientific_identity) != {"evaluation_config", "sensor_config"}:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: Habitat reset-replay scientific identity fields are incomplete"
        )
    if to_plain_json(frozen_decision["stop_receipt"]) != stop_identity:
        raise Context4ClosedLoopBlocked(
            "BLOCKED: five-row decision STOP receipt identity drift"
        )
    observed_habitat = {
        "sha256": canonical_mapping_sha256(scientific_identity)
    }
    if (
        to_plain_json(frozen_decision["habitat_scientific_identity"])
        != observed_habitat
    ):
        raise Context4ClosedLoopBlocked(
            "BLOCKED: five-row decision Habitat scientific identity drift"
        )

    model_identity = _mapping(
        loaded.training_config.get("model"), name="training sidecar model"
    )
    k_model = model_identity.get("modes")
    h_model = model_identity.get("horizon")
    if type(k_model) is not int or type(h_model) is not int:
        raise Context4ClosedLoopBlocked("BLOCKED: checkpoint K/H identity is invalid")
    visual = dict(_mapping(section.get("visual"), name="context4.visual"))
    try:
        encode_grid, encoder_provenance = encoder_builder(**visual, device=device)
    except Exception as exc:
        raise Context4ClosedLoopBlocked(f"BLOCKED: shared V-JEPA encoder failed: {exc}") from exc
    coordinate_identity = _close_visual_coordinate_identity(
        loaded=loaded,
        encoder_provenance=_mapping(
            encoder_provenance, name="shared V-JEPA encoder provenance"
        ),
        stop_receipt=stop_payload,
    )

    planner = Context4Planner(
        model=loaded.model,
        deployment=deployment,
        k_model=k_model,
        h_model=h_model,
        active_k=k_active,
        active_h=h_active,
        stop_mode="calibrated",
        stop_threshold=float(stop_payload["threshold"]),
        variant_contract=variant_contract,
    )
    online = _mapping(section.get("online"), name="context4.online")
    runtime_identity = {
        "checkpoint_sha256": str(loaded.provenance["checkpoint_sha256"]),
        "stop_receipt": stop_identity,
        "habitat_reset_replay_capability": capability_identity,
        "habitat_reset_replay_scientific_identity": json.loads(
            json.dumps(
                scientific_identity,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
        "visual_coordinate_identity": coordinate_identity,
    }
    if stop_attestation_identity is not None:
        runtime_identity["stop_attestation"] = stop_attestation_identity
    runtime_identity.update(to_plain_json(frozen_artifacts))
    return Context4PolicyAdapter(
        encode_grid=encode_grid,
        planner=planner,
        provenance={
            "variant_identity": dict(loaded.variant_identity),
            "checkpoint_provenance": dict(loaded.provenance),
            "encoder_provenance": dict(encoder_provenance),
            "runtime_identity": runtime_identity,
        },
        stop_mode="calibrated",
        stop_receipt=stop_payload,
        stop_admission=stop_admission,
        max_diagnostics=int(online.get("max_diagnostics", 128)),
    )


def build_context4_diagnostic_adapter(*, model, encode_grid, encoder_provenance,
                                     active_k, active_h, sampling_seed=None,
                                     sampling_namespace="diagnostic/v1", max_diagnostics=128):
    """Compose native factual mechanics without granting formal STOP admission."""
    proposal = model.proposal
    if not bool(getattr(proposal, "is_stochastic", False)):
        raise ValueError("native diagnostic composition requires stochastic Q")
    if any(module.training for module in model.modules()):
        raise ValueError("diagnostic planning requires every model module in eval mode")
    shape = (model.forward_core.grid_side ** 2, model.forward_core.latent_dim)
    coordinates = _mapping(encoder_provenance.get("coordinate_identity"), name="native diagnostic coordinates")
    from j2j.evaluation.vjepa_grid_encoder import VJEPA_NATIVE_POOL_IDENTITY_SHA256
    if (coordinates.get("representation") != "raw" or
        coordinates.get("whitening_sha256", "missing") is not None or
        coordinates.get("pool_sha256") != VJEPA_NATIVE_POOL_IDENTITY_SHA256 or
        tuple(coordinates.get("spatial_shape", ())) != shape or
        proposal.future.spatial_position.shape[0] != shape[0] or
        proposal.input_proj.in_features != shape[1]):
        raise ValueError("diagnostic encoder and model native coordinates differ")

    def checked_encoder(image):
        grid = encode_grid(image)
        if not isinstance(grid, torch.Tensor) or tuple(grid.shape) != shape:
            raise ValueError("diagnostic encoder output differs from declared native grid")
        if not bool(torch.isfinite(grid).all()):
            raise FloatingPointError("diagnostic encoder output must be finite")
        return grid.detach().clone()

    planner = Context4Planner(model=model, deployment="full", k_model=proposal.default_samples,
        h_model=getattr(proposal, "default_horizon", proposal.trained_horizon), active_k=active_k, active_h=active_h,
        sampling_seed=sampling_seed, sampling_namespace=sampling_namespace, stop_mode="never_stop")
    return Context4PolicyAdapter(encode_grid=checked_encoder, planner=planner,
        provenance={"evidence_tier": "bounded_runtime_diagnostic", "formal_admitted": False,
                    "encoder_provenance": dict(encoder_provenance),
                    "sampling_seed": planner.sampling_seed, "sampling_namespace": sampling_namespace,
                    "active_k": active_k, "active_h": active_h},
        stop_mode="never_stop", max_diagnostics=max_diagnostics)


__all__ = ["Context4ClosedLoopBlocked", "build_context4_adapter", "build_context4_diagnostic_adapter"]
