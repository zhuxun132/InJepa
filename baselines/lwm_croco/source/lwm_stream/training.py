"""Differentiable calls through the unchanged official LWM modules."""

import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F


def _mask(tensor, mask, prefix=False):
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != tensor.shape or mask.device != tensor.device):
        raise ValueError("mask must be boolean with matching shape and device")
    if prefix and torch.any(mask[..., 1:] & ~mask[..., :-1]):
        raise ValueError("trajectory validity must be a prefix")
    return mask


def _finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"nonfinite active {name}")


def _actions(actions, valid_mask):
    if actions.ndim != 4 or actions.shape[-1] != 3 or not actions.is_floating_point():
        raise ValueError("actions must have floating shape (B,M,T,3)")
    if min(actions.shape[:3]) < 1:
        raise ValueError("empty observation, candidate or time axis")
    mask = (torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
            if valid_mask is None else _mask(actions[..., 0], valid_mask, prefix=True))
    _finite(actions[mask], "action")
    return actions.masked_fill(~mask[..., None], 0), mask


def _tokens(tokens, length=None):
    if (tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[0] < 1
            or tokens.shape[1] < 2 or (length is not None and tokens.shape[1] != length)):
        raise ValueError("tokens must have nonempty int64 shape (B,max_len)")
    if torch.any(tokens < 0) or torch.any(tokens > 66) or torch.any(tokens[:, 0] != 64):
        raise ValueError("invalid token vocabulary or missing initial BOS")
    eos = tokens == 65
    if torch.any(eos.sum(dim=1) != 1) or torch.any(tokens[:, 1:] == 64):
        raise ValueError("expected exactly one EOS and one initial BOS")
    after_eos = eos.cumsum(dim=1) - eos.to(torch.long) > 0
    if torch.any(tokens[after_eos] != 66) or torch.any((tokens == 66) & ~after_eos):
        raise ValueError("PAD only occurs after EOS; all post-EOS tokens must be PAD")
    return tokens[:, 1:] != 66


class WorldModelTraining(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, now, goal, actions_m, valid_mask=None):
        """Same source tensor graph, allowing per-observation candidate sets."""
        actions, _ = _actions(actions_m, valid_mask)
        b, m, t, _ = actions.shape
        if (t != self.model.num_a or now.shape[0] != b or goal.shape[0] != b
                or now.device != goal.device or now.device != actions.device):
            raise ValueError("image/action batch, horizon or device mismatch")
        fusion = self.model.cro_proj(self.model.encode_images(now, goal))
        memory = fusion.repeat_interleave(m, dim=0)
        # The original inference ABI multiplies *all* three coordinates by .1.
        action_features = self.model.acton_encoder((actions * .1).reshape(b * m, t, 3))
        action_features = self.model.position(action_features)
        causal = nn.Transformer.generate_square_subsequent_mask(t, device=actions.device)
        predicted = self.model.decoder1(tgt=action_features, memory=memory,
                                       tgt_mask=causal, tgt_is_causal=True)
        return self.model.output_sim(predicted).squeeze(-1).reshape(b, m, t)


class PolicyTraining(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, now, goal, tokens):
        _tokens(tokens, self.model.max_len)
        if (now.shape[0] != tokens.shape[0] or goal.shape[0] != tokens.shape[0]
                or now.device != tokens.device or goal.device != tokens.device):
            raise ValueError("image/token batch or device mismatch")
        fusion = self.model.cro_proj(self.model.encode_images(now, goal))
        return self.model.waypoint_decoder(fusion, tokens)


@torch.no_grad()
def log_distance_targets(actions_m, goal_xy_m, epsilon_m, valid_mask):
    """Negated paper compatibility targets in metre space, without label grads."""
    if (isinstance(epsilon_m, bool) or not isinstance(epsilon_m, numbers.Real)
            or not math.isfinite(epsilon_m) or epsilon_m <= 0):
        raise ValueError("epsilon_m must be finite and positive")
    actions, mask = _actions(actions_m, valid_mask)
    b, m, t, _ = actions.shape
    if goal_xy_m.shape != (b, 2) or goal_xy_m.device != actions.device:
        raise ValueError("goal XY must have matching (B,2) shape and device")
    # At least fp32 for log targets under an autocast training caller.
    dtype = torch.float64 if actions.dtype == torch.float64 else torch.float32
    goals = goal_xy_m[:, None, None, :].expand(b, m, t, 2)[mask].to(dtype)
    _finite(goals, "goal XY")
    xy = actions[..., :2][mask].to(dtype)
    distances = torch.linalg.vector_norm(xy - goals, dim=-1)
    labels = torch.zeros((b, m, t), dtype=dtype, device=actions.device)
    labels[mask] = torch.log(distances + epsilon_m)
    return labels


def masked_mse_terms(pred, target, mask):
    """Additive loss statistics; select before arithmetic to exclude NaN padding."""
    if pred.shape != target.shape or pred.device != target.device:
        raise ValueError("prediction and target must have matching shape/device")
    _mask(pred, mask)
    active_pred, active_target = pred[mask], target.detach()[mask]
    _finite(active_pred, "prediction")
    _finite(active_target, "target")
    if active_pred.dtype in (torch.float16, torch.bfloat16):
        active_pred = active_pred.float()
    return (active_pred - active_target).square().sum(), mask.sum(dtype=torch.int64)


def cross_entropy_terms(logits, tokens, pad_token=66):
    if pad_token != 66:
        raise ValueError("official LWM uses PAD=66")
    active = _tokens(tokens)
    if (logits.shape != (*active.shape, 67) or logits.device != tokens.device):
        raise ValueError("logits must align with the 67-way next-token targets")
    scores = logits[active]
    _finite(scores, "policy logits")
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    numerator = F.cross_entropy(scores, tokens[:, 1:][active], reduction="sum")
    return numerator, active.sum(dtype=torch.int64)
