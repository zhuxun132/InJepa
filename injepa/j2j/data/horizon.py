"""Construction of causally aligned factual future tapes."""

from __future__ import annotations

import torch
from torch import Tensor


def build_factual_tape(
    z: Tensor, *, t: int, horizon: int
) -> tuple[Tensor, Tensor]:
    """Build factual ``z[t+1:]`` targets and an exact active-horizon mask."""
    if not isinstance(z, Tensor) or z.ndim < 1:
        raise ValueError("z must be a Tensor with a time dimension")
    if type(t) is not int or type(horizon) is not int:
        raise ValueError("t and horizon must be integers")

    terminal = z.size(0) - 1
    if horizon < 1 or not 0 <= t < terminal:
        raise ValueError("invalid factual origin or horizon")

    active_count = min(horizon, terminal - t)
    target = z.new_zeros((horizon, *z.shape[1:]))
    active = torch.zeros(horizon, dtype=torch.bool, device=z.device)
    successor = z[t + 1 : t + 1 + active_count].detach()
    target[:active_count].copy_(successor)
    active[:active_count] = True
    return target, active
