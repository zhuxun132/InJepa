"""Bounded factual prediction and input-sensitivity diagnostics, with no updates."""

from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch.nn import functional as F

from j2j.context4_objective import q_to_g_logits
from j2j.proposal.sampling import keyed_normal


def validate_probe_config(config):
    if not isinstance(config, Mapping) or set(config) != {"max_origins", "max_q_occurrences", "prior_samples"}:
        raise ValueError("model probe requires explicit origin, occurrence and sample budgets")
    for name, value in config.items():
        minimum = 2 if name == "prior_samples" else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"model probe {name} must be an integer >= {minimum}")
    return dict(config)


def _finite(value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("bounded model probe produced nonfinite values")
    return value


def _list(value):
    return _finite(value).detach().cpu().tolist()


def _tv(left, right):
    return (left - right).abs().sum(-1) * 0.5


def bounded_model_probe(model, batch, *, max_origins, max_q_occurrences,
                        prior_samples, sampling_namespace):
    """Probe a deterministic batch prefix; interventions have no counterfactual truth.

    Values are rank-local small-sample diagnostics, never population dev metrics.
    Every module's original mode and the caller RNG survive even failed probes.
    """
    validate_probe_config(dict(max_origins=max_origins, max_q_occurrences=max_q_occurrences,
                               prior_samples=prior_samples))
    if not isinstance(sampling_namespace, str) or not sampling_namespace:
        raise ValueError("probe sampling namespace must be nonempty")
    if not bool(getattr(model.proposal, "is_stochastic", False)):
        raise ValueError("bounded prior probe requires the stochastic model")
    modes = [(module, module.training) for module in model.modules()]
    namespace = model.proposal.sampling_namespace
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            return _probe(model, batch, max_origins=max_origins,
                          max_q_occurrences=max_q_occurrences, prior_samples=prior_samples,
                          sampling_namespace=sampling_namespace)
    finally:
        for module, training in modes:
            module.training = training
        model.proposal.sampling_namespace = namespace


def _probe(model, batch, *, max_origins, max_q_occurrences, prior_samples, sampling_namespace):
    n = min(max_origins, batch.current_grid.shape[0])
    qn = min(max_q_occurrences, batch.q_goal_grid.shape[0])
    tn = min(max_origins, batch.terminal_grid.shape[0])
    result = {"schema": "j2j.bounded_model_probe.v1", "sampling_namespace": sampling_namespace,
              "counts": dict(origins=n, q_occurrences=qn, terminals=tn, prior_samples=prior_samples),
              "g": {}, "f": {"rows": []}, "prior": {name: [] for name in (
                  "sample_keys", "latent_pair_mse", "first_grid_pair_mae",
                  "first_action_probability_tv", "first_action_agreement", "posterior_first_grid_mae",
                  "prior_oracle_first_grid_mae", "prior_average_first_grid_mae")}}
    views = (
        ("local", batch.current_grid[:n], batch.local_intent[:n], batch.previous_raw4[:n], batch.action_ids[:n], batch.origin_indices[:n]),
        ("goal", batch.current_grid[:n], batch.goal_intent[:n], batch.previous_raw4[:n], batch.action_ids[:n], batch.origin_indices[:n]),
        ("terminal", batch.terminal_grid[:tn], batch.terminal_intent[:tn], batch.terminal_previous_raw4[:tn],
         batch.terminal_action_ids[:tn], torch.arange(tn, device=batch.current_grid.device)),
    )
    for view, current, intent, previous, labels, origins in views:
        record = {"rows": []}
        result["g"][view] = record
        if len(current) and not len(intent):
            record["unavailable_reason"] = "intent_pruned_by_loss_ablation"
            continue
        logits = _finite(model.actor_logits(current, intent, previous).float()) if len(current) else None
        if logits is not None:
            probability = logits.softmax(-1)
            nll = _finite(F.cross_entropy(logits, labels, reduction="none"))
            record["rows"] = [dict(origin_index=int(origin), label=int(label), prediction=int(pred), nll=float(loss))
                              for origin, label, pred, loss in zip(origins.cpu(), labels.cpu(),
                              probability.argmax(-1).cpu(), nll.cpu())]
        if view == "terminal":
            for index, row in enumerate(record["rows"]):
                row.update(origin_index=None, terminal_row_index=index)
            continue  # Zero STOP intent has no distinct valid intent donor.
        record.update(intent_swap_probability_tv=[], previous_swap_probability_tv=[],
                      intent_swap_input_changed=[], previous_swap_input_changed=[], unavailable_reason=None)
        if n < 2:
            record["unavailable_reason"] = "no_origins" if n == 0 else "needs_at_least_two_origins"
            if n:
                for field in ("intent_swap_probability_tv", "previous_swap_probability_tv"):
                    record[field] = [None]
                record["intent_swap_input_changed"] = [False]
                record["previous_swap_input_changed"] = [False]
            continue
        for name, original in (("intent", intent), ("previous", previous)):
            donor = original.roll(1, 0)
            altered = _finite(model.actor_logits(current, donor if name == "intent" else intent,
                                                 donor if name == "previous" else previous).float()).softmax(-1)
            record[f"{name}_swap_probability_tv"] = _list(_tv(probability, altered))
            record[f"{name}_swap_input_changed"] = (donor != original).flatten(1).any(-1).cpu().tolist()
    if n:
        grids, outgoing, valid = batch.context_grid[:n], batch.context_outgoing_raw4[:n], batch.context_valid[:n]
        action = batch.action_ids[:n]
        if bool(((action < 1) | (action > 3)).any()):
            raise ValueError("F probes require factual motion primitives")
        prediction = _finite(model.predict_next_grid(grids, outgoing, valid).float())
        rotated = action.remainder(3) + 1
        changed = outgoing.clone()
        changed[:, -1] = F.one_hot(rotated, num_classes=4).to(changed)
        alternative = _finite(model.predict_next_grid(grids, changed, valid).float())
        factual = (prediction - batch.next_grid[:n].float()).square().mean((-2, -1))
        persistence = (batch.current_grid[:n].float() - batch.next_grid[:n].float()).square().mean((-2, -1))
        sensitivity = (prediction - alternative).square().mean((-2, -1))
        result["f"]["rows"] = [dict(origin_index=int(batch.origin_indices[i]), action_id=int(action[i]),
                                   rotated_action_id=int(rotated[i]), factual_mse=a, persistence_mse=b,
                                   action_rotated_prediction_mse=c)
                                  for i, (a, b, c) in enumerate(zip(_list(factual), _list(persistence), _list(sensitivity)))]
    if qn:
        rows = batch.q_origin_row[:qn]
        record = batch.context_grid.index_select(0, rows)
        raw = batch.context_incoming_raw4.index_select(0, rows)
        incoming = model.embed_actions(raw.reshape(-1, 4)).detach().float().reshape(qn, raw.shape[1], -1)
        facts = (record, incoming, batch.context_age.index_select(0, rows), batch.context_type.index_select(0, rows),
                 batch.context_valid.index_select(0, rows), batch.q_goal_grid[:qn])
        keys = batch.q_sample_keys[:qn]
        if len(keys) != qn:
            raise ValueError("bounded prior diagnostics require canonical occurrence keys")
        proposal = model.proposal
        prior = proposal.sample_prior(*facts, samples=prior_samples, horizon=proposal.trained_horizon,
                                      sample_keys=keys, seed=proposal.global_seed, namespace=sampling_namespace)
        _finite(prior.tape)
        noise = keyed_normal(keys, seed=proposal.global_seed, namespace=sampling_namespace + "/posterior",
                             samples=1, latent_dim=proposal.trajectory_latent_dim, device=record.device, dtype=torch.float32)
        posterior = proposal.posterior_forward(*facts, batch.q_target_grid[:qn], batch.q_active_h[:qn],
                                                horizon=proposal.trained_horizon, sample_noise=noise)
        _finite(posterior.tape)
        probability = _finite(q_to_g_logits(model.actor, model, batch.current_grid.index_select(0, rows),
                              prior.tape[:, :, 0], batch.previous_raw4.index_select(0, rows)).float()).softmax(-1)
        errors = (prior.tape[:, :, 0].float() - batch.q_target_grid[:qn, None, 0].float()).abs().mean((-2, -1))
        result["prior"] = dict(sample_keys=[key.hex() for key in keys],
            latent_pair_mse=_list((prior.trajectory_latent[:, 0] - prior.trajectory_latent[:, 1]).square().mean(-1)),
            first_grid_pair_mae=_list((prior.tape[:, 0, 0].float() - prior.tape[:, 1, 0].float()).abs().mean((-2, -1))),
            first_action_probability_tv=_list(_tv(probability[:, 0], probability[:, 1])),
            first_action_agreement=(probability[:, 0].argmax(-1) == probability[:, 1].argmax(-1)).cpu().tolist(),
            posterior_first_grid_mae=_list((posterior.tape[:, 0, 0].float() - batch.q_target_grid[:qn, 0].float()).abs().mean((-2, -1))),
            prior_oracle_first_grid_mae=_list(errors.min(1).values), prior_average_first_grid_mae=_list(errors.mean(1)))
        config = getattr(proposal, "stability", None)
        if config is not None:
            from j2j.proposal.stability import BoundedLayerNorm, BoundedMultiheadAttention
            gaussian = {}
            for name, value, bound in (
                ("prior_mean", prior.prior_mean, config.prior_mean_bound),
                ("prior_log_scale", .5 * prior.prior_logvar, config.prior_log_scale_bound),
                ("posterior_standardized_delta", posterior.posterior_standardized_delta, config.posterior_mean_delta_bound),
                ("posterior_log_scale_ratio", posterior.posterior_log_scale_ratio, config.posterior_log_scale_ratio_bound),
            ):
                gaussian[name] = float((_finite(value).abs() > .95 * bound).float().mean())
            attentions = [m for m in proposal.modules() if isinstance(m, BoundedMultiheadAttention)]
            temperatures = torch.cat([m.effective_temperature for m in attentions])
            saturated = torch.cat([m.effective_temperature > .95 * config.attention_temperature_max_factor * math.sqrt(m.head_dim)
                                   for m in attentions])
            gains = torch.cat([m.effective_weight.flatten() for m in proposal.modules() if isinstance(m, BoundedLayerNorm)])
            result["stability"] = {
                "gaussian_saturation_fraction": gaussian,
                "temperature_saturation_scope": "upper_bound_only; inspect min for lower saturation",
                "temperature": {"min": float(temperatures.min()), "max": float(temperatures.max()),
                                "saturation_fraction": float(saturated.float().mean())},
                "effective_ln_gain_max": float(gains.abs().max()),
                "decoder_residual_scales": [m.residual_scale for m in proposal.future.blocks],
                "readout_residual_scales": {"condition": proposal.condition_readout.residual_scale,
                                            "future": proposal.future_readout.residual_scale},
            }
    return result
