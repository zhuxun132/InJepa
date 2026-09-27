"""Original LWM pseudo-label and reinforcement sampling connections."""
import numbers

import torch

from .training import WorldModelTraining, _finite
from .reinforcement import _scalar, _rollout_mask


def _eval(model):
    if any(module.training for module in model.modules()):
        raise ValueError("stage data generation requires eval dropout mode")


def _abi(wm, tokenizer, policy=None):
    if (tokenizer.K != 64 or tokenizer.max_len != wm.num_a + 2
            or (tokenizer.BOS, tokenizer.EOS, tokenizer.PAD) != (64, 65, 66)
            or tuple(tokenizer.centers_tensor.shape) != (64, 2)
            or (policy is not None and (policy.max_len != tokenizer.max_len
                                       or policy.num_a != wm.num_a))):
        raise ValueError("incompatible official tokenizer, policy and WM ABI")
    _finite(tokenizer.centers_tensor, "tokenizer centers")


@torch.no_grad()
def pseudo_labels(wm, tokenizer, now, goal, candidates_m):
    _eval(wm)
    _abi(wm, tokenizer)
    if (candidates_m.ndim != 3 or candidates_m.shape[0] < 1
            or candidates_m.shape[1:] != (wm.num_a, 3)
            or not candidates_m.is_floating_point()):
        raise ValueError("candidate codebook must be floating (K,T,3) in metres")
    actions = candidates_m.unsqueeze(0).expand(now.shape[0], -1, -1, -1)
    scores = WorldModelTraining(wm)(now, goal, actions)
    _finite(scores, "candidate scores")
    best = scores.flatten(1).argmin(-1)
    candidates, points = best // wm.num_a, best % wm.num_a
    tokens = torch.stack([tokenizer.encode_tensor(candidates_m[candidates[i], :points[i]+1, :2])
                          for i in range(now.shape[0])])
    return {"tokens": tokens, "candidate_indices": candidates, "point_indices": points}


@torch.no_grad()
def sample_rewards(policy, wm, tokenizer, now, goal, *, num_sample, temperature):
    _eval(policy)
    _eval(wm)
    _abi(wm, tokenizer, policy)
    _scalar(temperature, "temperature")
    if isinstance(num_sample, bool) or not isinstance(num_sample, numbers.Integral) or num_sample < 1:
        raise ValueError("num_sample must be a positive integer")
    tokens = policy.roll_out(now, goal, num_sample=num_sample, temperature=temperature)
    _rollout_mask(tokens, policy.max_len)
    b, g, length = tokens.shape
    if b != now.shape[0] or g != num_sample:
        raise ValueError("rollout batch or group differs from requested shape")
    actions, lengths = tokenizer.batch_decode(tokens.reshape(b * g, length))
    actions = actions.reshape(b, g, wm.num_a, 3)
    scores = WorldModelTraining(wm)(now, goal, actions)
    # Original get_reward: an empty rollout is scored at its zero-padded first step.
    endpoints = (lengths - 1).clamp_min(0).reshape(b, g, 1)
    rewards = -scores.gather(-1, endpoints).squeeze(-1)
    _finite(rewards, "rollout rewards")
    return {"tokens": tokens, "actions_m": actions,
            "lengths": lengths.reshape(b, g), "rewards": rewards}
