"""RAE-NWM context-four adapter for factual ImageNav observations.

The append operation is a direct tensor-level adaptation of the MIT-licensed
RAE-NWM source at commit ``0219ce41c44d515f86719dd763c1efe7c7f72519``
(``infer.py:164,200-202``).  The validity mask below is the sole ImageNav
episode-start adaptation: it represents a causal prefix without copying a
frame or inventing an action.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


CONTEXT_SIZE = 4
RAE_NWM_COMMIT = "0219ce41c44d515f86719dd763c1efe7c7f72519"


@dataclass(frozen=True)
class FactualContext:
    """One right-aligned context made only from recorded trajectory facts."""

    record_grid: Tensor
    incoming_raw4: Tensor
    outgoing_raw4: Tensor
    record_age: Tensor
    record_type: Tensor
    record_valid: Tensor
    full_context_ready: bool


def _require_context_size(context_size: object) -> int:
    if isinstance(context_size, bool) or not isinstance(context_size, int):
        raise TypeError("context_size must be an integer")
    if context_size != CONTEXT_SIZE:
        raise ValueError("context_size must equal the RAE-NWM source value 4")
    return context_size


def append_context_latent(
    curr_latents: Tensor,
    new_latent: Tensor,
    *,
    context_size: int = CONTEXT_SIZE,
) -> Tensor:
    """Append one dense latent and retain RAE-NWM's last-four context.

    This deliberately preserves the three source operations from RAE-NWM
    MIT commit ``0219ce41c44d515f86719dd763c1efe7c7f72519``,
    ``infer.py:164,200-202``.  It returns new storage and never mutates the
    factual input buffer.
    """

    _require_context_size(context_size)
    if not isinstance(curr_latents, Tensor) or not isinstance(new_latent, Tensor):
        raise TypeError("curr_latents and new_latent must be tensors")
    if curr_latents.ndim < 2 or new_latent.shape != curr_latents.shape[:1] + curr_latents.shape[2:]:
        raise ValueError("new_latent must match curr_latents with the context axis removed")
    if curr_latents.device != new_latent.device or curr_latents.dtype != new_latent.dtype:
        raise ValueError("context and appended latent must share dtype and device")

    # Direct RAE-NWM tensor path (MIT, fixed commit above).
    input_latents = curr_latents[:, -context_size:]
    curr_latents = torch.cat((input_latents, new_latent.unsqueeze(1)), dim=1)
    return curr_latents[:, -context_size:]


def build_factual_context(
    grids: Tensor,
    incoming_raw4: Tensor,
    outgoing_raw4: Tensor,
    *,
    origin: int,
    context_size: int = CONTEXT_SIZE,
) -> FactualContext:
    """Build the masked causal context ending at ``origin`` from real rows."""

    context_size = _require_context_size(context_size)
    if not all(isinstance(value, Tensor) for value in (grids, incoming_raw4, outgoing_raw4)):
        raise TypeError("grids and aligned actions must be tensors")
    if grids.ndim != 3 or grids.shape[0] < 1:
        raise ValueError("grids must have shape [frames, spatial, latent]")
    if incoming_raw4.shape != (grids.shape[0], 4):
        raise ValueError("incoming_raw4 must have shape [frames, 4]")
    if outgoing_raw4.shape != (grids.shape[0], 4):
        raise ValueError("outgoing_raw4 must have shape [frames, 4]")
    if grids.dtype != torch.float32 or incoming_raw4.dtype != torch.float32 or outgoing_raw4.dtype != torch.float32:
        raise TypeError("factual grids and raw actions must be float32")
    if grids.device != incoming_raw4.device or grids.device != outgoing_raw4.device:
        raise ValueError("factual grids and raw actions must share one device")
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise TypeError("origin must be an integer")
    if origin < 0 or origin >= grids.shape[0]:
        raise ValueError("origin is outside the factual trajectory")
    if not bool(torch.isfinite(grids).all()):
        raise ValueError("factual grids must be finite")
    if not bool(torch.isfinite(incoming_raw4).all()) or not bool(torch.isfinite(outgoing_raw4).all()):
        raise ValueError("factual actions must be finite")

    start = max(0, origin - context_size + 1)
    count = origin - start + 1
    destination = slice(context_size - count, context_size)
    source = slice(start, origin + 1)

    record_grid = grids.new_zeros((context_size, *grids.shape[1:]))
    incoming = incoming_raw4.new_zeros((context_size, 4))
    outgoing = outgoing_raw4.new_zeros((context_size, 4))
    valid = torch.zeros(context_size, dtype=torch.bool, device=grids.device)
    ages = torch.zeros(context_size, dtype=torch.int64, device=grids.device)
    kinds = torch.zeros(context_size, dtype=torch.int64, device=grids.device)

    record_grid[destination] = grids[source].detach()
    incoming[destination] = incoming_raw4[source].detach()
    outgoing[destination] = outgoing_raw4[source].detach()
    valid[destination] = True
    ages[destination] = torch.arange(count - 1, -1, -1, dtype=torch.int64, device=grids.device)
    kinds[destination] = 1
    kinds[-1] = 2

    return FactualContext(
        record_grid=record_grid.detach(),
        incoming_raw4=incoming.detach(),
        outgoing_raw4=outgoing.detach(),
        record_age=ages,
        record_type=kinds,
        record_valid=valid,
        full_context_ready=count == context_size,
    )
