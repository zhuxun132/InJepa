"""Small, deterministic helpers for effective-batch gradient accumulation.

The official RAE model and loss remain untouched.  This module only defines
the arithmetic that maps a paper *global* batch to a per-rank microbatch and
the update boundaries used by the thin training-loop adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
import numbers


def require_positive_int(value: object, *, name: str) -> int:
    """Return an exact positive integer, rejecting bool/ fractional values."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


# Private compatibility alias for the original helper name used in this
# module.  Callers that validate runtime/receipt inputs should use the public
# function so fractional values cannot be truncated by ``int(...)``.
_positive_int = require_positive_int


@dataclass(frozen=True)
class AccumulationPlan:
    """Resolved effective-batch arithmetic for one DDP training run."""

    global_batch_size: int
    world_size: int
    accumulation_steps: int
    microbatch_size: int

    @property
    def microbatch_global_size(self) -> int:
        return self.world_size * self.microbatch_size

    @property
    def effective_global_batch_size(self) -> int:
        return self.microbatch_global_size * self.accumulation_steps


def make_accumulation_plan(
    *,
    global_batch_size: int,
    world_size: int,
    accumulation_steps: int = 1,
) -> AccumulationPlan:
    """Resolve a per-rank microbatch while preserving an exact global batch.

    The divisibility check is intentional: silently rounding would alter the
    paper's effective batch and optimizer schedule.
    """

    total = _positive_int(global_batch_size, name="global_batch_size")
    world = _positive_int(world_size, name="world_size")
    steps = _positive_int(accumulation_steps, name="gradient_accumulation_steps")
    denominator = world * steps
    if total % denominator:
        raise ValueError(
            "global batch must be divisible by world_size * gradient accumulation "
            f"(global={total}, world_size={world}, accumulation_steps={steps})"
        )
    microbatch = total // denominator
    if microbatch <= 0:
        raise ValueError("resolved per-rank microbatch must be positive")
    return AccumulationPlan(total, world, steps, microbatch)


def accumulation_update_count(num_microbatches: int, accumulation_steps: int) -> int:
    """Return complete optimizer groups; an incomplete tail is not an update."""

    count = _positive_int(num_microbatches, name="num_microbatches")
    steps = _positive_int(accumulation_steps, name="gradient_accumulation_steps")
    return count // steps


def scaled_loss_value(loss: float, accumulation_steps: int) -> float:
    """Scale a scalar loss exactly once per microbatch group."""

    steps = _positive_int(accumulation_steps, name="gradient_accumulation_steps")
    return float(loss) / steps


__all__ = [
    "AccumulationPlan",
    "accumulation_update_count",
    "make_accumulation_plan",
    "require_positive_int",
    "scaled_loss_value",
]
