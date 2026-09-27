"""Exact five-branch loss arithmetic for context-four joint training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from j2j.proposal.losses import proper_mixture_terms, conditional_elbo_terms
from j2j.proposal.sampling import keyed_normal
from j2j.context4_variants import FULL_LOSS_WEIGHTS, NO_QG_LOSS_WEIGHTS


def _encode_actions(action_encoder, raw4: Tensor) -> Tensor:
    if hasattr(action_encoder, "embed_actions"):
        return action_encoder.embed_actions(raw4)
    try:
        result = action_encoder(raw4)
    except (RuntimeError, ValueError):
        result = action_encoder(raw4[:, None]).squeeze(1)
    if result.ndim == 3 and result.shape[1] == 1:
        result = result.squeeze(1)
    if result.ndim != 2 or result.shape[0] != raw4.shape[0]:
        raise ValueError("action encoder must return [batch, embedding]")
    return result


def q_to_g_logits(
    actor,
    action_encoder,
    current_grid: Tensor,
    first_steps: Tensor,
    previous_raw4: Tensor,
) -> Tensor:
    """Evaluate every proposal mode at the detached Q-to-G seam."""

    if current_grid.ndim != 3 or first_steps.ndim != 4:
        raise ValueError("current_grid/first_steps must be [B,S,D]/[B,K,S,D]")
    batch, modes, spatial, latent = first_steps.shape
    if current_grid.shape != (batch, spatial, latent):
        raise ValueError("Q first-step coordinates do not match current_grid")
    if previous_raw4.shape != (batch, 4):
        raise ValueError("previous_raw4 must have shape [B,4]")
    current = current_grid.detach()[:, None].expand(-1, modes, -1, -1)
    intent = first_steps.detach() - current
    previous = _encode_actions(action_encoder, previous_raw4)
    previous = previous[:, None].expand(-1, modes, -1)
    logits = actor(
        current.reshape(batch * modes, spatial, latent),
        intent.reshape(batch * modes, spatial, latent),
        previous.reshape(batch * modes, -1),
    )
    if logits.shape != (batch * modes, 4):
        raise ValueError("actor must return four logits for every Q mode")
    return logits.reshape(batch, modes, 4)


def qg_marginal_nll(logits: Tensor, labels: Tensor) -> Tensor:
    """Uniform-mode categorical marginal in the required float32 log domain."""

    if logits.ndim != 3 or logits.shape[-1] != 4:
        raise ValueError("logits must have shape [batch,modes,4]")
    if labels.dtype != torch.int64 or labels.shape != (logits.shape[0],):
        raise ValueError("labels must be int64 [batch]")
    modes = logits.shape[1]
    if modes < 1:
        raise ValueError("at least one proposal mode is required")
    chosen = torch.log_softmax(logits.float(), dim=-1).gather(
        -1,
        labels[:, None, None].expand(-1, modes, 1),
    ).squeeze(-1)
    result = -(torch.logsumexp(chosen, dim=1) - math.log(modes))
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Q-to-G marginal NLL is non-finite")
    return result


def qg_posterior_weighted_nll(logits: Tensor, labels: Tensor, responsibility: Tensor) -> Tensor:
    """Action CE weighted by detached Q posterior on the factual whole tape."""
    if logits.ndim != 3 or logits.shape[-1] != 4 or logits.shape[1] < 1:
        raise ValueError("logits must have shape [batch,modes,4]")
    if labels.dtype != torch.int64 or labels.shape != (logits.shape[0],):
        raise ValueError("labels must be int64 [batch]")
    if responsibility.shape != logits.shape[:2]:
        raise ValueError("responsibility must have shape [batch,modes]")
    weights = responsibility.detach().float()
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("responsibility must be finite and non-negative")
    if not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-5, rtol=1e-5):
        raise ValueError("responsibility rows must sum to one")
    chosen = torch.log_softmax(logits.float(), dim=-1).gather(
        -1, labels[:, None, None].expand(-1, logits.shape[1], 1)
    ).squeeze(-1)
    result = -(weights * chosen).sum(-1)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("Q-to-G posterior weighted NLL is non-finite")
    return result


def normalize_q_occurrence_nll(
    raw_occurrence_nll: Tensor,
    active_h: Tensor,
    *,
    spatial_tokens: int,
    latent_dim: int,
) -> Tensor:
    if raw_occurrence_nll.ndim != 1 or active_h.ndim != 2:
        raise ValueError("Q occurrence NLL and active_h ranks are inconsistent")
    if raw_occurrence_nll.shape[0] != active_h.shape[0] or active_h.dtype != torch.bool:
        raise ValueError("Q occurrence NLL and active_h rows are inconsistent")
    if spatial_tokens <= 0 or latent_dim <= 0:
        raise ValueError("spatial_tokens and latent_dim must be positive")
    denominator = active_h.sum(dim=1).to(torch.float32) * spatial_tokens * latent_dim
    if bool((denominator <= 0).any()):
        raise ValueError("every Q occurrence must contain an active horizon")
    return raw_occurrence_nll.float() / denominator


def branch_denominators(
    *,
    q_occurrences: int,
    transitions: int,
    terminals: int,
    spatial_tokens: int,
    latent_dim: int,
) -> Mapping[str, int]:
    values = (q_occurrences, transitions, terminals, spatial_tokens, latent_dim)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("branch counts must be non-negative integers")
    if spatial_tokens < 1 or latent_dim < 1:
        raise ValueError("spatial and latent dimensions must be positive")
    return {
        "q": q_occurrences,
        "f": transitions * spatial_tokens * latent_dim,
        "g_local": transitions,
        "g_goal": transitions + terminals,
        "qg": q_occurrences,
    }


def scaled_local_branch(
    local_numerator: Tensor,
    *,
    global_denominator: int | Tensor,
    world_size: int,
) -> Tensor:
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be positive")
    denominator = torch.as_tensor(
        global_denominator,
        dtype=torch.float32,
        device=local_numerator.device,
    ).detach()
    if denominator.numel() != 1 or not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise ValueError("global denominator must be finite and positive")
    return local_numerator.float() * float(world_size) / denominator


def graph_connected_zero(parameters: Iterable[nn.Parameter]) -> Tensor:
    values = tuple(parameters)
    if not values:
        raise ValueError("at least one parameter is required for a connected zero")
    zero = values[0].sum() * 0.0
    for parameter in values[1:]:
        zero = zero + parameter.sum() * 0.0
    return zero


def factual_sse_numerator(predicted: Tensor, factual_target: Tensor) -> Tensor:
    if predicted.shape != factual_target.shape:
        raise ValueError("forward prediction and factual target shapes differ")
    return (predicted.float() - factual_target.detach().float()).square().sum()


def weighted_joint_loss(
    *,
    q: Tensor,
    f: Tensor,
    g_local: Tensor,
    g_goal: Tensor,
    qg: Tensor,
    weights: Mapping[str, float] | None = None,
) -> Tensor:
    values = (q, f, g_local, g_goal, qg)
    if any(not isinstance(value, Tensor) or value.numel() != 1 for value in values):
        raise TypeError("every joint loss branch must be a scalar tensor")
    return _weighted_joint_sum(
        q.float(),
        f.float(),
        g_local.float(),
        g_goal.float(),
        qg.float(),
        weights=weights,
    )


def _loss_weight_values(
    weights: Mapping[str, float] | None,
) -> tuple[float, float, float, float, float]:
    resolved = FULL_LOSS_WEIGHTS if weights is None else weights
    if not isinstance(resolved, Mapping) or set(resolved) != {
        "q", "f", "g_local", "g_goal", "qg"
    }:
        raise ValueError("joint loss weights must contain exactly five branches")
    values: list[float] = []
    for name in ("q", "f", "g_local", "g_goal", "qg"):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"joint loss weight {name} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"joint loss weight {name} must be finite and non-negative")
        values.append(float(value))
    result = tuple(values)
    admitted = {
        tuple(float(FULL_LOSS_WEIGHTS[name]) for name in ("q", "f", "g_local", "g_goal", "qg")),
        tuple(float(NO_QG_LOSS_WEIGHTS[name]) for name in ("q", "f", "g_local", "g_goal", "qg")),
    }
    for branch in ("qg", "g_local", "g_goal"):
        admitted.add(tuple(0.0 if name == branch else float(FULL_LOSS_WEIGHTS[name])
                           for name in ("q", "f", "g_local", "g_goal", "qg")))
    if result not in admitted:
        raise ValueError("joint loss weights are not an admitted Context4 loss identity")
    return result  # type: ignore[return-value]


def _weighted_joint_sum(q, f, g_local, g_goal, qg, *, weights):
    q_weight, f_weight, local_weight, goal_weight, qg_weight = _loss_weight_values(
        weights
    )
    if (q_weight, f_weight, local_weight, goal_weight, qg_weight) == (
        1.0,
        1.0,
        0.1,
        0.05,
        0.1,
    ):
        # Preserve the exact pre-control Full expression and operation order.
        return q + f + 0.1 * g_local + 0.05 * g_goal + 0.1 * qg
    return q + f + local_weight * g_local + goal_weight * g_goal + qg_weight * qg


def weighted_joint_loss_value(
    branches: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compose already-normalized metric scalars with the training weights."""

    if not isinstance(branches, Mapping) or set(branches) != {
        "q", "f", "g_local", "g_goal", "qg"
    }:
        raise ValueError("joint metric branches must contain exactly five losses")
    values = tuple(float(branches[name]) for name in ("q", "f", "g_local", "g_goal", "qg"))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("joint metric losses must be finite")
    return float(_weighted_joint_sum(*values, weights=weights))


@dataclass(frozen=True)
class JointNumerators:
    numerators: Mapping[str, Tensor]
    denominators: Mapping[str, int]
    counts: Mapping[str, int]
    diagnostic_tensors: Mapping[str, Tensor] | None = None
    stochastic_statistics: Mapping[str, Tensor] | None = None


def _action_logits(model, z: Tensor, intent: Tensor, previous_raw4: Tensor) -> Tensor:
    if hasattr(model, "action_logits"):
        return model.action_logits(z, intent, previous_raw4)
    if hasattr(model, "actor_logits"):
        return model.actor_logits(z, intent, previous_raw4)
    previous = _encode_actions(model.action_encoder, previous_raw4)
    return model.actor(z, intent, previous)


def _predict_next(model, history_grid: Tensor, outgoing_raw4: Tensor, valid: Tensor) -> Tensor:
    if hasattr(model, "predict_next"):
        return model.predict_next(history_grid, outgoing_raw4, valid)
    if hasattr(model, "predict_next_grid"):
        return model.predict_next_grid(history_grid, outgoing_raw4, valid)
    action = model.action_encoder(outgoing_raw4)
    return model.pred_proj(model.forward_core(history_grid, action, valid))


def _owner_zero(*modules: nn.Module) -> Tensor:
    return graph_connected_zero(
        parameter for module in modules for parameter in module.parameters()
    )


def joint_numerators(model, batch, *, collect_diagnostics: bool = False,
                     qg_objective: str = "marginal",
                     loss_weights: Mapping[str, float] | None = None) -> JointNumerators:
    """Run the five branches once from one factual ``JointBatch``."""
    if not isinstance(collect_diagnostics, bool):
        raise TypeError("collect_diagnostics must be a bool")
    if qg_objective not in {"marginal", "posterior_weighted", "posterior_sample"}:
        raise ValueError("unknown QG objective")
    _, _, local_weight, goal_weight, qg_weight = _loss_weight_values(loss_weights)
    stochastic = bool(getattr(model.proposal, "is_stochastic", False))
    stochastic_statistics = None
    if stochastic != (qg_objective == "posterior_sample"):
        raise ValueError("stochastic Q requires the posterior_sample QG objective")
    transitions = int(batch.current_grid.shape[0])
    q_occurrences = int(batch.q_goal_grid.shape[0])
    terminals = int(batch.terminal_grid.shape[0])
    if transitions:
        spatial, latent = batch.current_grid.shape[1:]
    elif terminals:
        spatial, latent = batch.terminal_grid.shape[1:]
    else:
        raise ValueError("joint batch must contain a transition or terminal")

    if q_occurrences:
        q_rows = batch.q_origin_row
        q_context_grid = batch.context_grid.index_select(0, q_rows)
        q_context_incoming = batch.context_incoming_raw4.index_select(0, q_rows)
        if hasattr(model, "embed_actions"):
            incoming_embedding = model.embed_actions(q_context_incoming)
        else:
            incoming_embedding = model.action_encoder(q_context_incoming)
        # The shared action embedder is owned by F/G/QG.  Q consumes its
        # representation as fixed context metadata, so L_Q must stop here.
        incoming_embedding = incoming_embedding.detach().float()
        proposal_inputs = (
            q_context_grid,
            incoming_embedding,
            batch.context_age.index_select(0, q_rows),
            batch.context_type.index_select(0, q_rows),
            batch.context_valid.index_select(0, q_rows),
            batch.q_goal_grid,
        )
        if stochastic:
            keys = getattr(batch, "q_sample_keys", ())
            if len(keys) != q_occurrences:
                raise ValueError("stochastic Q requires one factual occurrence key per target")
            noise = keyed_normal(keys, seed=model.proposal.global_seed,
                namespace=model.proposal.sampling_namespace, samples=1,
                latent_dim=model.proposal.trajectory_latent_dim,
                device=q_context_grid.device, dtype=torch.float32)
            proposal = model.proposal.posterior_forward(
                *proposal_inputs, batch.q_target_grid, batch.q_active_h,
                horizon=batch.q_target_grid.shape[1], sample_noise=noise)
            relative_kwargs = {}
            if getattr(model.proposal, "stability", None) is not None:
                for field in ("posterior_standardized_delta", "posterior_log_scale_ratio"):
                    value = getattr(proposal, field, None)
                    if value is None:
                        raise ValueError("stable posterior requires both relative Gaussian fields")
                    relative_kwargs[field] = value
            q_terms = conditional_elbo_terms(
                proposal.tape, batch.q_target_grid, batch.q_active_h,
                proposal.posterior_mean, proposal.posterior_logvar,
                proposal.prior_mean, proposal.prior_logvar, kl_beta=model.kl_beta,
                **relative_kwargs)
            q_num = q_terms.occurrence_loss.sum()
            with torch.no_grad():
                stochastic_statistics = {
                    "occurrences": q_num.new_tensor(q_occurrences),
                    "reconstruction_mean_sum": q_terms.reconstruction_mean.detach().sum(),
                    "reconstruction_sum": q_terms.reconstruction_sum.detach().sum(),
                    "target_scalars": q_terms.scalar_count.detach().sum(),
                    "kl_total_sum": q_terms.kl_total.detach().sum(),
                    "kl_mean_sum": q_terms.kl_mean.detach().sum(),
                    "exact_nelbo_per_scalar_sum": q_terms.exact_nelbo_per_scalar.detach().sum(),
                    "prior_mean_square_sum": proposal.prior_mean.detach().square().sum(),
                    "posterior_mean_square_sum": proposal.posterior_mean.detach().square().sum(),
                    "prior_variance_sum": proposal.prior_logvar.detach().exp().sum(),
                    "posterior_variance_sum": proposal.posterior_logvar.detach().exp().sum(),
                    "latent_dimensions": q_num.new_tensor(proposal.prior_mean.numel()),
                }
        else:
            proposal = model.proposal(*proposal_inputs, batch.q_active_h)
            q_terms = proper_mixture_terms(
                proposal.tape, proposal.log_mass, batch.q_target_grid, batch.q_active_h)
            q_num = q_terms.occurrence_nll.sum()
        qg_logits = None
        qg_labels = batch.action_ids.index_select(0, q_rows)
        if qg_weight:
            encoder_for_qg = model if hasattr(model, "embed_actions") else model.action_encoder
            qg_logits = q_to_g_logits(
                model.actor,
                encoder_for_qg,
                batch.current_grid.index_select(0, q_rows),
                proposal.tape[:, :, 0],
                batch.previous_raw4.index_select(0, q_rows),
            )
            qg_labels = batch.action_ids.index_select(0, q_rows)
            qg_num = F.cross_entropy(qg_logits[:, 0].float(), qg_labels, reduction="sum") if stochastic else (
                qg_posterior_weighted_nll(qg_logits, qg_labels, q_terms.responsibility)
                if qg_objective == "posterior_weighted"
                else qg_marginal_nll(qg_logits, qg_labels)
            ).sum()
        else:
            qg_num = _owner_zero(model.action_encoder, model.actor)
        modes = int(proposal.tape.shape[1])
        diagnostic_tensors = (
            {
                "q_responsibility": q_terms.responsibility.detach(),
                "q_tape": proposal.tape.detach(),
                "q_active_h": batch.q_active_h.detach(),
                "qg_logits": qg_logits.detach(),
                "qg_labels": qg_labels.detach(),
            }
            if collect_diagnostics and not stochastic and qg_weight
            else None
        )
    else:
        q_num = _owner_zero(model.proposal)
        qg_num = _owner_zero(model.action_encoder, model.actor)
        modes = 0
        diagnostic_tensors = None

    if transitions:
        predicted = _predict_next(
            model,
            batch.context_grid,
            batch.context_outgoing_raw4,
            batch.context_valid,
        )
        f_num = factual_sse_numerator(predicted, batch.next_grid)
        g_local_num = _owner_zero(model.action_encoder, model.actor)
        g_goal_num = _owner_zero(model.action_encoder, model.actor)
        if local_weight:
            local_logits = _action_logits(model, batch.current_grid, batch.local_intent, batch.previous_raw4)
            g_local_num = F.cross_entropy(local_logits.float(), batch.action_ids, reduction="sum")
        if goal_weight:
            goal_logits = _action_logits(model, batch.current_grid, batch.goal_intent, batch.previous_raw4)
            g_goal_num = F.cross_entropy(goal_logits.float(), batch.action_ids, reduction="sum")
    else:
        f_num = _owner_zero(model.action_encoder, model.forward_core, model.pred_proj)
        g_local_num = _owner_zero(model.action_encoder, model.actor)
        g_goal_num = _owner_zero(model.action_encoder, model.actor)
    if terminals and goal_weight:
        terminal_logits = _action_logits(
            model,
            batch.terminal_grid,
            batch.terminal_intent,
            batch.terminal_previous_raw4,
        )
        g_goal_num = g_goal_num + F.cross_entropy(
            terminal_logits.float(), batch.terminal_action_ids, reduction="sum"
        )
    denominators = branch_denominators(
        q_occurrences=q_occurrences,
        transitions=transitions,
        terminals=terminals,
        spatial_tokens=spatial,
        latent_dim=latent,
    )
    return JointNumerators(
        numerators={
            "q": q_num,
            "f": f_num,
            "g_local": g_local_num,
            "g_goal": g_goal_num,
            "qg": qg_num,
        },
        denominators=denominators,
        counts={
            "transitions": transitions,
            "q_occurrences": q_occurrences,
            "terminals": terminals,
            "qg_mode_rows": q_occurrences * modes if qg_weight else 0,
        },
        diagnostic_tensors=diagnostic_tensors,
        stochastic_statistics=stochastic_statistics,
    )
