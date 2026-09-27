"""Operation-defined tensor metrics for one-pass Context4 diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def _require_finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains non-finite values")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def expected_calibration_error(
    confidence: Tensor,
    correct: Tensor,
    *,
    bins: int,
) -> float:
    """Equal-width ECE with ``[left,right)`` bins and a right-closed last bin."""

    bins = _positive_int(bins, "bins")
    if not isinstance(confidence, Tensor) or not confidence.is_floating_point():
        raise TypeError("confidence must be a floating tensor")
    if not isinstance(correct, Tensor) or correct.dtype != torch.bool:
        raise TypeError("correct must be a boolean tensor")
    if confidence.shape != correct.shape or confidence.numel() == 0:
        raise ValueError("confidence and correct must have one nonempty shared shape")
    _require_finite("confidence", confidence)
    if bool(((confidence < 0) | (confidence > 1)).any()):
        raise ValueError("confidence must lie in [0,1]")

    flat_confidence = confidence.detach().double().reshape(-1)
    flat_correct = correct.detach().double().reshape(-1)
    # floor maps exact 1.0 to ``bins``; clamping puts it in the last bin.
    bin_index = torch.floor(flat_confidence * bins).to(torch.int64).clamp_max(bins - 1)
    total = float(flat_confidence.numel())
    result = 0.0
    for index in range(bins):
        mask = bin_index == index
        count = int(mask.sum())
        if count:
            accuracy = float(flat_correct[mask].mean())
            mean_confidence = float(flat_confidence[mask].mean())
            result += (count / total) * abs(accuracy - mean_confidence)
    return result


def cosine_distance(prediction: Tensor, target: Tensor, *, epsilon: float) -> Tensor:
    """Cosine distance after flattening exactly the final full-grid axes."""

    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise TypeError("epsilon must be a real scalar")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
        raise TypeError("prediction and target must be tensors")
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share [...,spatial,latent]")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")
    if prediction.device != target.device:
        raise ValueError("prediction and target must share one device")
    _require_finite("prediction", prediction)
    _require_finite("target", target)
    # Metrics are off-graph.  Float64 plus the mathematical cosine bounds
    # prevents identical float32 grids from producing a small negative
    # distance when the product of two rounded norms undershoots the dot.
    pred = prediction.detach().double().flatten(start_dim=prediction.ndim - 2)
    truth = target.detach().double().flatten(start_dim=target.ndim - 2)
    numerator = (pred * truth).sum(dim=-1)
    denominator = pred.norm(dim=-1) * truth.norm(dim=-1)
    cosine = (numerator / denominator.clamp_min(epsilon)).clamp(-1.0, 1.0)
    return 1.0 - cosine


def finite_rate(values: Tensor, *, eligible: Tensor) -> dict[str, int | float | None]:
    """Count finite eligible rows while keeping unavailable rows explicit."""

    if not isinstance(values, Tensor) or not isinstance(eligible, Tensor):
        raise TypeError("values and eligible must be tensors")
    if eligible.dtype != torch.bool or eligible.shape != values.shape:
        raise ValueError("eligible must be boolean with the same shape as values")
    eligible_rows = int(eligible.sum())
    unavailable_rows = int((~eligible).sum())
    finite = torch.isfinite(values) if values.is_floating_point() else torch.ones_like(eligible)
    finite_rows = int((finite & eligible).sum())
    return {
        "finite_rows": finite_rows,
        "eligible_rows": eligible_rows,
        "unavailable_rows": unavailable_rows,
        "finite_rate": finite_rows / eligible_rows if eligible_rows else None,
    }


def grid_error_metrics(prediction: Tensor, target: Tensor, *, epsilon: float) -> dict[str, Tensor]:
    """Return per-row full-grid MSE, MAE and cosine distance."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share [...,spatial,latent]")
    _require_finite("prediction", prediction)
    _require_finite("target", target)
    difference = prediction.float() - target.float()
    return {
        "mse": difference.square().mean(dim=(-1, -2)),
        "mae": difference.abs().mean(dim=(-1, -2)),
        "cosine_distance": cosine_distance(prediction, target, epsilon=epsilon),
    }


def categorical_summary(logits: Tensor, labels: Tensor, *, bins: int) -> dict[str, Any]:
    """Four-class NLL/confusion/macro metrics without majority-class hiding."""

    if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("logits must have shape [rows,4]")
    if not isinstance(labels, Tensor) or labels.dtype != torch.int64:
        raise TypeError("labels must be an int64 tensor")
    if labels.shape != (logits.shape[0],) or labels.numel() == 0:
        raise ValueError("labels must align with a nonempty logits batch")
    _require_finite("logits", logits)
    if bool(((labels < 0) | (labels > 3)).any()):
        raise ValueError("labels must be in STOP/FWD/LEFT/RIGHT id range")
    log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    confidence, prediction = probabilities.max(dim=-1)
    confusion = torch.zeros((4, 4), dtype=torch.int64, device=labels.device)
    confusion.index_put_((labels, prediction), torch.ones_like(labels), accumulate=True)
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for class_id in range(4):
        tp = int(confusion[class_id, class_id])
        predicted_count = int(confusion[:, class_id].sum())
        actual_count = int(confusion[class_id].sum())
        p = tp / predicted_count if predicted_count else 0.0
        r = tp / actual_count if actual_count else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2.0 * p * r / (p + r) if p + r else 0.0)
    return {
        "nll": float(
            -log_probabilities.gather(1, labels[:, None]).squeeze(1).mean()
        ),
        "accuracy": float((prediction == labels).float().mean()),
        "macro_f1": sum(f1) / 4.0,
        "precision": tuple(precision),
        "recall": tuple(recall),
        "confusion": confusion.detach().cpu().tolist(),
        "ece": expected_calibration_error(confidence, prediction == labels, bins=bins),
    }


__all__ = [
    "categorical_summary",
    "cosine_distance",
    "expected_calibration_error",
    "finite_rate",
    "grid_error_metrics",
]
