"""Native recurrent composition using the existing planner and factual adapter."""
import torch
import math

from j2j_iclr_experiments.online.planner import Context4Planner
from j2j_iclr_experiments.online.adapter import Context4PolicyAdapter
from .raw_stop import close_raw_visual_coordinate_identity, validate_raw_stop_calibration_receipt


def build_recurrent_adapter(*, loaded, encode_grid, encoder_provenance,
        stop_receipt, expected_protocol_sha256, active_k, active_h,
        sampling_seed, namespace, deployment="full", max_diagnostics=128,
        diagnostic_stop_threshold=None, diagnostic_reach_radius=None,
        whole_branch_ranking=None, execution_interval=1, stagnation_guard=None, run_tier=None):
    """Compose a loaded model; the run launcher admits its final checkpoint/ledger."""
    if type(execution_interval) is not int or execution_interval < 1:
        raise ValueError("execution_interval must be a positive integer")
    if deployment == "goal_gf":
        from j2j_iclr_experiments.offline.whole_branch_rank import validate_whole_branch_options
        if (run_tier != "diagnostic" or diagnostic_reach_radius is None
                or execution_interval != 1 or whole_branch_ranking is None
                or stagnation_guard is not None):
            raise ValueError("goal_gf requires diagnostic reach-only whole-branch risk0 ext1")
        options = validate_whole_branch_options(whole_branch_ranking)
        if options["risk_weight"] != 0 or options.get("score_mode", "l1_consistency") != "l1_consistency":
            raise ValueError("goal_gf requires matched normalized L1 endpoint-only risk0")
    if execution_interval > 1 and (deployment != "full" or whole_branch_ranking is None
            or diagnostic_reach_radius is None or active_h < execution_interval):
        raise ValueError("sequence execution requires full whole-branch reach-only H>=interval")
    if diagnostic_reach_radius is not None:
        if (isinstance(diagnostic_reach_radius, bool)
                or not isinstance(diagnostic_reach_radius, (int, float))
                or not math.isfinite(diagnostic_reach_radius) or diagnostic_reach_radius <= 0):
            raise ValueError("diagnostic reach radius must be positive and finite")
        if diagnostic_stop_threshold is not None:
            raise ValueError("reach-only and diagnostic STOP threshold are mutually exclusive")
    model, training = loaded.model, loaded.training_config
    if ("encoder_contract" in training
            or "encoder_family" in training.get("data", {}).get("expected_identities", {})):
        raise ValueError("unsupported visual encoder identity")
    if any(module.training for module in model.modules()):
        raise ValueError("recurrent evaluation requires every module in eval mode")
    proposal = model.proposal
    if not getattr(proposal, "is_stochastic", False) or not getattr(proposal, "is_recurrent", False):
        raise ValueError("native recurrent factory requires the recurrent stochastic model")
    if stop_receipt is None:
        raise ValueError("V-JEPA requires its admitted STOP coordinate receipt")
    if stop_receipt.get("protocol_sha256") != expected_protocol_sha256:
        raise ValueError("RAW STOP protocol differs from the configured experiment")
    validate_raw_stop_calibration_receipt(stop_receipt)
    coordinate = close_raw_visual_coordinate_identity(
        training_expected_identities=training["data"]["expected_identities"],
        encoder_provenance=encoder_provenance,
        stop_coordinate_identity=stop_receipt["visual_coordinate_identity"])
    shape = (training["model"]["grid_side"] ** 2, training["model"]["latent_dim"])
    if (tuple(coordinate["spatial_shape"]) != shape
            or (model.forward_core.grid_side ** 2, model.forward_core.latent_dim) != shape):
        raise ValueError("RAW training/model/encoder grid shapes differ")

    def checked_encoder(rgb):
        grid = encode_grid(rgb)
        if not isinstance(grid, torch.Tensor) or tuple(grid.shape) != shape:
            raise ValueError("native encoder output shape differs")
        if not bool(torch.isfinite(grid).all()):
            raise FloatingPointError("native encoder returned a nonfinite grid")
        return grid.detach().clone()

    threshold = float(stop_receipt["threshold"])
    if diagnostic_stop_threshold is not None:
        if (isinstance(diagnostic_stop_threshold, bool)
                or not isinstance(diagnostic_stop_threshold, (int, float))
                or not math.isfinite(diagnostic_stop_threshold) or diagnostic_stop_threshold < 0):
            raise ValueError("diagnostic STOP threshold must be finite and nonnegative")
        threshold = float(diagnostic_stop_threshold)
    threshold_policy = {"calibrated_threshold": float(stop_receipt["threshold"]),
        "effective_threshold": threshold,
        "kind": "calibrated" if diagnostic_stop_threshold is None else "diagnostic_override"}
    stop_mode = "calibrated"
    if diagnostic_reach_radius is not None:
        stop_mode, threshold = "never_stop", None
        threshold_policy.update(kind="diagnostic_reach_only", effective_threshold=None,
                                diagnostic_reach_radius=float(diagnostic_reach_radius))
    planner = Context4Planner(model=model, deployment=deployment,
        k_model=training["model"]["modes"], h_model=training["model"]["horizon"],
        active_k=active_k, active_h=active_h, sampling_seed=sampling_seed,
        sampling_namespace=namespace, stop_mode=stop_mode,
        stop_threshold=threshold, whole_branch_ranking=whole_branch_ranking,
        stagnation_guard=stagnation_guard)
    return Context4PolicyAdapter(encode_grid=checked_encoder, planner=planner,
        stop_mode=stop_mode, stop_receipt=stop_receipt, max_diagnostics=max_diagnostics,
        execution_interval=execution_interval,
        provenance={
            **({"candidate_source": "goal_gf_categorical", "action_temperature": 1.0,
                "q_metrics_status": "not_applicable", "q_prior_sampled": False}
               if deployment == "goal_gf" else {}),
            "variant_identity": dict(loaded.variant_identity),
            "stop_threshold_policy": threshold_policy,
            "checkpoint_provenance": dict(loaded.provenance),
            "encoder_provenance": dict(encoder_provenance),
            "protocol_sha256": expected_protocol_sha256,
            "visual_coordinate_identity": coordinate, "deployment": deployment,
            "active_k": active_k, "active_h": active_h,
            "sampling_seed": sampling_seed, "sampling_namespace": namespace})
