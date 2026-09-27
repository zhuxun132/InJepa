"""Pure, non-mutating one-factor interventions for offline diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from typing import Any

import torch
from torch import Tensor

from j2j.context4_variants import (
    CONTEXT1_IDENTITY,
    mean_repeat_grid as _shared_mean_repeat_grid,
    resolve_variant_contract,
)


_SINGLE_FACTOR_ARMS = frozenset(
    {
        "context_suffix_1",
        "context_suffix_2",
        "context_suffix_4",
        "mean_repeat_input",
        "fixed_permutation_input",
        "goal_permutation",
        "intent_permutation",
        "previous_action_permutation",
        "action_permutation",
    }
)


def mean_repeat_grid(grid: Tensor) -> Tensor:
    """Forward to the shared trained-control MeanRepeat operator."""

    return _shared_mean_repeat_grid(grid)


def fixed_token_permutation(grid: Tensor, permutation: Tensor) -> Tensor:
    """Apply one fixed bijection to the spatial axis without changing values."""

    if not isinstance(grid, Tensor) or grid.ndim < 2:
        raise TypeError("grid must be a tensor ending in [spatial,latent]")
    if not isinstance(permutation, Tensor) or permutation.dtype != torch.int64:
        raise TypeError("permutation must be an int64 tensor")
    if permutation.ndim != 1 or permutation.numel() != grid.shape[-2]:
        raise ValueError("permutation must contain one index per spatial token")
    expected = torch.arange(permutation.numel(), device=permutation.device)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError("permutation must be a bijection over the spatial axis")
    return grid.index_select(-2, permutation.to(device=grid.device)).clone()


def _zero_context_prefix(value: Tensor, *, axis: int, masked: int) -> Tensor:
    result = value.clone()
    if masked:
        selection = [slice(None)] * result.ndim
        selection[axis] = slice(0, masked)
        result[tuple(selection)] = 0
    return result


def apply_context_suffix(context: Mapping[str, Any], *, length: int) -> dict[str, Any]:
    """Keep the last ``length`` context slots and zero every earlier field."""

    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("context suffix length must be a positive integer")
    if length == 1:
        return dict(
            resolve_variant_contract(CONTEXT1_IDENTITY).context_transform(context)
        )
    required = {
        "record_grid": -3,
        "incoming_raw4": -2,
        "outgoing_raw4": -2,
        "record_age": -1,
        "record_type": -1,
        "record_valid": -1,
    }
    missing = sorted(set(required) - set(context))
    if missing:
        raise KeyError(f"context is missing fields: {missing}")
    tensors = {name: context[name] for name in required}
    if any(not isinstance(value, Tensor) for value in tensors.values()):
        raise TypeError("context fields must be tensors")
    context_size = int(tensors["record_valid"].shape[-1])
    if length > context_size:
        raise ValueError("context suffix length exceeds the materialized context")
    for name, axis in required.items():
        value = tensors[name]
        if value.ndim < abs(axis) or value.shape[axis] != context_size:
            raise ValueError(f"{name} has an inconsistent context axis")
    if tensors["record_valid"].dtype != torch.bool:
        raise TypeError("record_valid must be boolean")

    result = copy.deepcopy(dict(context))
    masked = context_size - length
    for name, axis in required.items():
        result[name] = _zero_context_prefix(tensors[name], axis=axis, masked=masked)
    return result


def build_single_factor_arm_plan(arms: Sequence[str]) -> tuple[str, ...]:
    """Return baseline plus unique one-factor arms, never a Cartesian product."""

    if isinstance(arms, (str, bytes)) or not isinstance(arms, Sequence):
        raise TypeError("arms must be a sequence of names")
    result = ["baseline"]
    seen = {"baseline"}
    for arm in arms:
        if not isinstance(arm, str) or not arm:
            raise TypeError("arm names must be nonempty strings")
        if arm == "baseline":
            continue
        if "+" in arm:
            raise ValueError("Cartesian intervention arms are forbidden")
        if arm not in _SINGLE_FACTOR_ARMS:
            raise ValueError(f"unknown single-factor arm {arm!r}")
        if arm in seen:
            raise ValueError(f"duplicate arm {arm!r}")
        seen.add(arm)
        result.append(arm)
    return tuple(result)


def build_cyclic_donor_plan(rows: Sequence[Mapping[str, Any]]) -> dict[Any, Any | None]:
    """Choose deterministic no-self donors strictly within each trajectory."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    groups: dict[tuple[Any, Any], list[Any]] = defaultdict(list)
    observed: set[Any] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("donor rows must be mappings")
        for field in ("row_key", "building", "trajectory"):
            if field not in row:
                raise KeyError(f"donor row is missing {field}")
        key = row["row_key"]
        if key in observed:
            raise ValueError("row_key values must be unique")
        observed.add(key)
        groups[(row["building"], row["trajectory"])].append(key)

    result: dict[Any, Any | None] = {}
    for group in sorted(groups, key=lambda value: (str(value[0]), str(value[1]))):
        keys = sorted(groups[group], key=str)
        if len(keys) < 2:
            result[keys[0]] = None
            continue
        for index, key in enumerate(keys):
            result[key] = keys[(index + 1) % len(keys)]
    return result


__all__ = [
    "apply_context_suffix",
    "build_cyclic_donor_plan",
    "build_single_factor_arm_plan",
    "fixed_token_permutation",
    "mean_repeat_grid",
]
