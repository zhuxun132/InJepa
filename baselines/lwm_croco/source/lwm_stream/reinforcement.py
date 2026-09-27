"""LWM Eq14–16 and likelihoods through the unchanged official decoder."""
import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F

from .training import _finite


def _scalar(value, name, lower=0., upper=None, inclusive=False):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or (value < lower if inclusive else value <= lower)
            or (upper is not None and value >= upper)):
        raise ValueError(f"invalid {name}")


def _rollout_mask(tokens, length=None):
    if (tokens.ndim != 3 or tokens.dtype != torch.long or min(tokens.shape) < 1
            or tokens.shape[-1] < 2
            or (length is not None and tokens.shape[-1] != length)):
        raise ValueError("tokens must be int64 (B,G,max_len)")
    if (torch.any(tokens < 0) or torch.any(tokens > 66)
            or torch.any(tokens[..., 0] != 64) or torch.any(tokens[..., 1:] == 64)):
        raise ValueError("invalid rollout token vocabulary or BOS")
    eos = tokens == 65
    if torch.any(eos.sum(-1) > 1):
        raise ValueError("multiple EOS tokens")
    after = eos.cumsum(-1) - eos.long() > 0
    if torch.any(tokens[after] != 66) or torch.any((tokens == 66) & ~after):
        raise ValueError("PAD must follow EOS, and all post-EOS tokens must be PAD")
    return tokens[..., 1:] != 66


def _compute_float(tensor):
    return tensor if tensor.dtype == torch.float64 else tensor.float()


class PolicyLikelihood(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, now, goal, tokens, *, temperature):
        _scalar(temperature, "temperature")
        if any(module.training for module in self.model.modules()):
            raise ValueError("RL likelihood requires eval dropout mode, with gradients enabled")
        mask = _rollout_mask(tokens, self.model.max_len)
        b, g, length = tokens.shape
        if (now.shape[0] != b or goal.shape[0] != b
                or now.device != tokens.device or goal.device != tokens.device):
            raise ValueError("image/token batch or device mismatch")
        features = self.model.cro_proj(self.model.encode_images(now, goal))
        logits = self.model.waypoint_decoder(
            features.repeat_interleave(g, dim=0), tokens.reshape(b * g, length))
        logits = _compute_float(logits).reshape(b, g, length - 1, 67)
        _finite(logits[mask], "rollout logits")
        # Match the original roll_out finite logit mask BEFORE temperature.
        logits = logits.masked_fill(~mask[..., None], 0).clone()
        logits[..., [64, 66]] = -1e9
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        _finite(log_probs[mask], "rollout log probabilities")
        log_probs = log_probs.masked_fill(~mask[..., None], 0)
        selected = log_probs.gather(-1, tokens[..., 1:, None]).squeeze(-1)
        return {"log_probs": log_probs, "token_log_probs": selected,
                "sequence_log_probs": selected.sum(-1), "valid_mask": mask}


@torch.no_grad()
def group_advantages(rewards):
    if (rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2
            or not rewards.is_floating_point()):
        raise ValueError("rewards must be floating (B,G), G>=2")
    _finite(rewards, "rewards")
    # Scaling before centering also protects finite double extremes.
    values = rewards.double()
    scale = values.abs().amax(-1, keepdim=True)
    scaled = values / torch.where(scale > 0, scale, torch.ones_like(scale))
    centered = scaled - scaled.mean(-1, keepdim=True)
    std = centered.square().mean(-1, keepdim=True).sqrt()
    result = centered / torch.where(std > 0, std, torch.ones_like(std))
    return result if rewards.dtype == torch.float64 else result.float()


def grpo_terms(current_log_probs, reference_log_probs, tokens, advantages, *, beta, clip_epsilon):
    _scalar(beta, "beta", inclusive=True)
    _scalar(clip_epsilon, "clip_epsilon", upper=1.)
    mask = _rollout_mask(tokens)
    expected = (*mask.shape, 67)
    if (current_log_probs.shape != expected or reference_log_probs.shape != expected
            or not current_log_probs.is_floating_point()
            or not reference_log_probs.is_floating_point()
            or current_log_probs.device != tokens.device
            or reference_log_probs.device != tokens.device
            or advantages.shape != tokens.shape[:2] or advantages.device != tokens.device
            or not advantages.is_floating_point()):
        raise ValueError("probability/advantage shapes, floating types or devices mismatch")
    _finite(advantages, "advantages")
    current = _compute_float(current_log_probs[mask])
    reference = _compute_float(reference_log_probs.detach()[mask])
    for probabilities in (current, reference):
        _finite(probabilities, "active log distribution")
        if not torch.allclose(probabilities.logsumexp(-1),
                              torch.zeros_like(probabilities[..., 0]), atol=1e-5, rtol=0):
            raise ValueError("active log probabilities must be normalized")
    targets = tokens[..., 1:][mask].unsqueeze(-1)
    differences = (current.gather(-1, targets) - reference.gather(-1, targets)).squeeze(-1)
    sequence_differences = differences.new_zeros(mask.shape).masked_scatter(mask, differences).sum(-1)
    # Literal Eq16 min(r,clip(r)): upper cap before exp prevents overflow.
    ratio = torch.exp(torch.clamp_max(sequence_differences, math.log1p(clip_epsilon)))
    prefix_kl = (current.exp() * (current - reference)).sum(-1)
    sequence_kl = prefix_kl.new_zeros(mask.shape).masked_scatter(mask, prefix_kl).sum(-1)
    total = (-ratio * advantages.detach() + beta * sequence_kl).sum()
    _finite(total, "GRPO loss")
    return total, torch.tensor(advantages.numel(), device=tokens.device, dtype=torch.int64)
