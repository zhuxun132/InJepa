"""Deterministic spatial and age sinusoidal position encodings."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def fixed_2d_sincos(*, dim: int, grid_side: int) -> Tensor:
    """Return the official row-major V-JEPA 2D sine/cosine table."""

    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if isinstance(grid_side, bool) or not isinstance(grid_side, int):
        raise TypeError("grid_side must be an integer")
    if dim <= 0 or dim % 4 != 0:
        raise ValueError("dim must be positive and divisible by four")
    if grid_side <= 0:
        raise ValueError("grid_side must be positive")

    quarter = dim // 4
    frequencies = 10000.0 ** (
        -np.arange(quarter, dtype=np.float64) / float(quarter)
    )
    coordinates = np.arange(grid_side, dtype=np.float64)
    row_phase = np.outer(coordinates, frequencies)
    column_phase = np.outer(coordinates, frequencies)
    row_table = np.concatenate((np.sin(row_phase), np.cos(row_phase)), axis=1)
    column_table = np.concatenate(
        (np.sin(column_phase), np.cos(column_phase)), axis=1
    )
    table = np.concatenate(
        (
            np.repeat(row_table[:, None, :], grid_side, axis=1),
            np.repeat(column_table[None, :, :], grid_side, axis=0),
        ),
        axis=2,
    ).reshape(grid_side * grid_side, dim)
    return torch.from_numpy(np.ascontiguousarray(table, dtype="<f4"))


def age_sincos(age: Tensor, *, dim: int) -> Tensor:
    """Encode non-negative integer ages using CPU float64 arithmetic."""

    if not isinstance(age, Tensor):
        raise TypeError("age must be a Tensor")
    if age.dtype != torch.int64:
        raise TypeError("age must have dtype int64")
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("dim must be an integer")
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("dim must be positive and even")
    if bool((age < 0).any()):
        raise ValueError("age values must be non-negative")

    age64 = age.detach().cpu().numpy().astype(np.float64, copy=False)
    frequencies = 10000.0 ** (
        -(2.0 * np.arange(dim // 2, dtype=np.float64)) / float(dim)
    )
    phase = age64[..., None] * frequencies
    output = np.empty((*age64.shape, dim), dtype=np.float64)
    output[..., 0::2] = np.sin(phase)
    output[..., 1::2] = np.cos(phase)
    return torch.from_numpy(np.ascontiguousarray(output, dtype="<f4"))
