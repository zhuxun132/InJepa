"""Proper whole-tape Laplace-mixture objective for proposal tapes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor


def diagonal_gaussian_kl(
    q_mean: Tensor, q_logvar: Tensor, p_mean: Tensor, p_logvar: Tensor,
) -> Tensor:
    """Analytic KL(q || p), summing latent dimensions per occurrence.

    Probability arithmetic is at least float32, independent of decoder autocast;
    float64 inputs retain float64 precision for analytic checks.
    """
    values = (q_mean, q_logvar, p_mean, p_logvar)
    if not all(isinstance(value, Tensor) for value in values):
        raise TypeError("Gaussian parameters must be tensors")
    if q_mean.ndim != 2 or min(q_mean.shape) < 1:
        raise ValueError("Gaussian parameters must have nonempty shape [B,L]")
    if any(value.shape != q_mean.shape for value in values):
        raise ValueError("Gaussian parameter shapes must match")
    if any(not value.is_floating_point() for value in values):
        raise TypeError("Gaussian parameters must be floating point")
    if any(value.device != q_mean.device for value in values):
        raise ValueError("Gaussian parameters must share one device")
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("Gaussian parameters must be finite")
    dtype = torch.float64 if any(v.dtype == torch.float64 for v in values) else torch.float32
    with torch.autocast(device_type=q_mean.device.type, enabled=False):
        qm, qv, pm, pv = (v.to(dtype=dtype) for v in values)
        delta = qv - pv
        result = 0.5 * (torch.expm1(delta) - delta
                        + (qm - pm).square() * (-pv).exp()).sum(-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Gaussian KL overflowed")
    return result


def diagonal_gaussian_relative_kl(delta: Tensor, log_scale_ratio: Tensor) -> Tensor:
    """KL in standardized posterior coordinates, without subtracting means."""
    if not isinstance(delta, Tensor) or not isinstance(log_scale_ratio, Tensor):
        raise TypeError("relative Gaussian parameters must be tensors")
    if delta.ndim != 2 or min(delta.shape) < 1 or delta.shape != log_scale_ratio.shape:
        raise ValueError("relative Gaussian parameters require matching nonempty [B,L]")
    if delta.device != log_scale_ratio.device or not delta.is_floating_point() or not log_scale_ratio.is_floating_point():
        raise ValueError("relative Gaussian parameters require floating point on one device")
    if not bool(torch.isfinite(delta).all() & torch.isfinite(log_scale_ratio).all()):
        raise ValueError("relative Gaussian parameters must be finite")
    dtype = torch.float64 if torch.float64 in (delta.dtype, log_scale_ratio.dtype) else torch.float32
    with torch.autocast(device_type=delta.device.type, enabled=False):
        d, twice_r = delta.to(dtype), 2 * log_scale_ratio.to(dtype)
        result = .5 * (torch.expm1(twice_r) - twice_r + d.square()).sum(-1)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("relative Gaussian KL overflowed")
    return result


@dataclass(frozen=True)
class ConditionalELBOTerms:
    loss: Tensor
    occurrence_loss: Tensor
    reconstruction_sum: Tensor
    scalar_count: Tensor
    kl_total: Tensor
    reconstruction_mean: Tensor
    kl_mean: Tensor
    exact_nelbo_per_scalar: Tensor


def conditional_elbo_terms(
    pred: Tensor, target: Tensor, future_label_valid: Tensor,
    q_mean: Tensor, q_logvar: Tensor, p_mean: Tensor, p_logvar: Tensor,
    *, laplace_scale: float = 2**-0.5, kl_beta: float = 0.05,
    posterior_standardized_delta: Tensor | None = None,
    posterior_log_scale_ratio: Tensor | None = None,
) -> ConditionalELBOTerms:
    """One-posterior-sample rate/distortion loss and separate exact NELBO.

    KL is per trajectory, reconstruction per valid scalar, then occurrences
    have equal weight. This is not a finite-mixture marginal likelihood.
    """
    if pred.ndim != 5 or pred.shape[1] != 1 or min(pred.shape) < 1:
        raise ValueError("conditional prediction must have shape [B,1,H,M,D]")
    batch, _, horizon, spatial, latent = pred.shape
    if target.shape != (batch, horizon, spatial, latent):
        raise ValueError("conditional target shape must match prediction")
    if future_label_valid.shape != (batch, horizon) or future_label_valid.dtype != torch.bool:
        raise ValueError("future labels require boolean [B,H] validity")
    if not bool(future_label_valid[:, 0].all()) or bool(((~future_label_valid[:, :-1]) & future_label_valid[:, 1:]).any()):
        raise ValueError("future labels require a nonempty contiguous prefix")
    if any(value.device != pred.device for value in (target, future_label_valid, q_mean, q_logvar, p_mean, p_logvar)):
        raise ValueError("conditional objective inputs must share one device")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("conditional predictions/targets must be floating point")
    if q_mean.shape[0] != batch:
        raise ValueError("one conditional Gaussian is required per occurrence")
    for name, value in (("laplace_scale", laplace_scale), ("kl_beta", kl_beta)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    dtype = torch.float64 if any(v.dtype == torch.float64 for v in (pred, target, q_mean, q_logvar, p_mean, p_logvar)) else torch.float32
    with torch.autocast(device_type=pred.device.type, enabled=False):
        mask = future_label_valid[:, :, None, None]
        safe_pred = pred[:, 0].to(dtype).masked_fill(~mask, 0)
        safe_target = target.detach().to(dtype).masked_fill(~mask, 0)
        if not bool(torch.isfinite(safe_pred).all() & torch.isfinite(safe_target).all()):
            raise ValueError("valid conditional targets/predictions must be finite")
        terms = (safe_pred - safe_target).abs() / float(laplace_scale) + math.log(2 * float(laplace_scale))
        reconstruction_sum = terms.masked_fill(~mask, 0).sum((1, 2, 3))
        scalar_count = future_label_valid.sum(-1) * spatial * latent
        reconstruction_mean = reconstruction_sum / scalar_count
        if (posterior_standardized_delta is None) != (posterior_log_scale_ratio is None):
            raise ValueError("relative Gaussian fields must be provided together")
        if posterior_standardized_delta is None:
            kl_total = diagonal_gaussian_kl(q_mean, q_logvar, p_mean, p_logvar)
        else:
            delta, ratio = posterior_standardized_delta, posterior_log_scale_ratio
            if delta.shape != q_mean.shape or delta.device != pred.device:
                raise ValueError("relative Gaussian fields differ from absolute parameters")
            if any(v.shape != q_mean.shape or not bool(torch.isfinite(v).all()) for v in (q_mean, q_logvar, p_mean, p_logvar)):
                raise ValueError("absolute Gaussian parameters are inconsistent or nonfinite")
            kl_total = diagonal_gaussian_relative_kl(delta, ratio)
            with torch.no_grad():
                if not torch.allclose(q_mean, p_mean + (.5 * p_logvar).exp() * delta, rtol=2e-6, atol=2e-7):
                    raise ValueError("posterior standardized displacement identity mismatch")
                if not torch.allclose(q_logvar, p_logvar + 2 * ratio, rtol=2e-6, atol=2e-7):
                    raise ValueError("posterior relative scale identity mismatch")
        kl_mean = kl_total / q_mean.shape[1]
        occurrence_loss = reconstruction_mean + float(kl_beta) * kl_mean
        exact_nelbo = reconstruction_mean + kl_total / scalar_count
    return ConditionalELBOTerms(occurrence_loss.mean(), occurrence_loss, reconstruction_sum,
                               scalar_count, kl_total, reconstruction_mean, kl_mean, exact_nelbo)


@dataclass(frozen=True)
class ProperMixtureTerms:
    loss: Tensor
    occurrence_nll: Tensor
    component_log_likelihood: Tensor
    log_responsibility: Tensor
    responsibility: Tensor
    active_h_count: Tensor


def _validate_mixture_inputs(
    pred: Tensor,
    log_mass: Tensor,
    target: Tensor,
    active_h: Tensor,
    laplace_scale: float,
) -> tuple[int, int, int, int, int]:
    if not all(
        isinstance(value, Tensor)
        for value in (pred, log_mass, target, active_h)
    ):
        raise TypeError("mixture inputs must be tensors")
    if pred.ndim != 5:
        raise ValueError("pred must have shape [B,K,H,M,D]")
    batch, modes, horizon, spatial, latent = pred.shape
    if any(size <= 0 for size in pred.shape):
        raise ValueError("every pred axis must be nonempty")
    if target.shape != (batch, horizon, spatial, latent):
        raise ValueError("target shape is inconsistent with pred")
    if log_mass.shape != (batch, modes):
        raise ValueError("log_mass shape is inconsistent with pred")
    if active_h.shape != (batch, horizon):
        raise ValueError("active_h shape is inconsistent with pred")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must be floating-point tensors")
    if log_mass.dtype != torch.float32:
        raise TypeError("log_mass must have dtype float32")
    if active_h.dtype != torch.bool:
        raise TypeError("active_h must have dtype bool")
    if len({pred.device, log_mass.device, target.device, active_h.device}) != 1:
        raise ValueError("all mixture inputs must share one device")
    if isinstance(laplace_scale, bool) or not isinstance(laplace_scale, Real):
        raise TypeError("laplace_scale must be a real scalar")
    scale_value = float(laplace_scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("laplace_scale must be finite and positive")

    if not bool(active_h[:, 0].all()):
        raise ValueError("every row must have a nonempty active prefix")
    if horizon > 1 and bool(((~active_h[:, :-1]) & active_h[:, 1:]).any()):
        raise ValueError("active_h must be a contiguous prefix")
    if not bool(torch.isfinite(log_mass).all()):
        raise ValueError("log_mass must be finite")
    if modes == 1:
        if int(torch.count_nonzero(log_mass)) != 0:
            raise ValueError("K=1 requires exact zero log_mass")
    else:
        residual = torch.logsumexp(log_mass, dim=1).abs()
        tolerance = 8 * torch.finfo(torch.float32).eps
        if bool((residual > tolerance).any()):
            raise ValueError("log_mass must already be normalized")

    pred_active = active_h[:, None, :, None, None].expand_as(pred)
    target_active = active_h[:, :, None, None].expand_as(target)
    if not bool(torch.isfinite(pred.masked_select(pred_active)).all()):
        raise ValueError("active pred entries must be finite")
    if not bool(torch.isfinite(target.masked_select(target_active)).all()):
        raise ValueError("active target entries must be finite")
    return batch, modes, horizon, spatial, latent


def _proper_mixture_kernel(
    pred: Tensor,
    log_mass: Tensor,
    target: Tensor,
    active_h: Tensor,
    *,
    laplace_scale: float = 2**-0.5,
) -> ProperMixtureTerms:
    batch, modes, horizon, spatial, latent = _validate_mixture_inputs(
        pred,
        log_mass,
        target,
        active_h,
        laplace_scale,
    )
    del batch, modes, horizon

    with torch.autocast(device_type=pred.device.type, enabled=False):
        pred32 = pred.float()
        target32 = target.detach().float()
        pred_mask = active_h[:, None, :, None, None]
        target_mask = active_h[:, :, None, None]
        safe_pred = pred32.masked_fill(~pred_mask, 0.0)
        safe_target = target32.masked_fill(~target_mask, 0.0)[:, None]
        difference = safe_pred - safe_target
        if not bool(torch.isfinite(difference).all()):
            raise FloatingPointError("valid Laplace residual overflowed")

        active_h_count = active_h.sum(dim=1).to(dtype=torch.int64)
        denominator = active_h_count.double() * spatial * latent
        # Accumulate in double without materializing a double future tape.
        # Center BEFORE adding the small log prior, so neither prior credit
        # nor backward normalization subtracts rounded million-scale values.
        absolute_sum = difference.abs().masked_fill(~pred_mask, 0.0).sum(
            dim=(2, 3, 4), dtype=torch.float64
        )
        component64 = -absolute_sum / float(laplace_scale) - denominator[:, None] * math.log(2.0 * float(laplace_scale))
        offset = component64.amax(dim=1, keepdim=True).detach()
        joint = component64 - offset + log_mass.double()
        log_evidence = offset.squeeze(1) + torch.logsumexp(joint, dim=1)
        occurrence_nll = (-log_evidence / denominator).float()
        component_log_likelihood = component64.float()
        log_responsibility = torch.log_softmax(joint, dim=1).float()
        responsibility = log_responsibility.exp()
        loss = occurrence_nll.mean()

    return ProperMixtureTerms(
        loss=loss,
        occurrence_nll=occurrence_nll,
        component_log_likelihood=component_log_likelihood,
        log_responsibility=log_responsibility,
        responsibility=responsibility,
        active_h_count=active_h_count,
    )


def proper_mixture_terms(
    pred: Tensor,
    log_mass: Tensor,
    target: Tensor,
    active_h: Tensor,
    *,
    laplace_scale: float = 2**-0.5,
) -> ProperMixtureTerms:
    return _proper_mixture_kernel(
        pred,
        log_mass,
        target,
        active_h,
        laplace_scale=laplace_scale,
    )


def proper_mixture_nll(
    pred: Tensor,
    log_mass: Tensor,
    target: Tensor,
    active_h: Tensor,
    *,
    laplace_scale: float = 2**-0.5,
) -> Tensor:
    return _proper_mixture_kernel(
        pred,
        log_mass,
        target,
        active_h,
        laplace_scale=laplace_scale,
    ).loss
