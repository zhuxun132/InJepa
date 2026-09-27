"""Canonical four-action adapter for ImageNav."""

from __future__ import annotations

from enum import IntEnum

import torch
from torch import Tensor


class ActionId(IntEnum):
    """Canonical categorical ImageNav action identifiers."""

    STOP = 0
    FWD = 1
    LEFT = 2
    RIGHT = 3


class Raw4Adapter:
    """Convert between categorical actions and INTACT's raw four-vector ABI."""

    @staticmethod
    def encode(action_id: ActionId) -> Tensor:
        """Return the exact float32 one-hot vector for ``action_id``."""
        if not isinstance(action_id, ActionId):
            raise ValueError("action_id must be an ActionId")
        encoded = torch.zeros(4, dtype=torch.float32)
        encoded[int(action_id)] = 1.0
        return encoded

    @staticmethod
    def encode_bos() -> Tensor:
        """Return the all-zero raw4 beginning-of-sequence marker."""
        return torch.zeros(4, dtype=torch.float32)

    @staticmethod
    def decode_logits(logits: Tensor) -> Tensor:
        """Decode logits using the frozen FWD, LEFT, RIGHT, STOP tie order."""
        if not isinstance(logits, Tensor) or logits.ndim < 1 or logits.shape[-1] != 4:
            raise ValueError("logits must be a Tensor with last dimension 4")

        priority = torch.tensor([1, 2, 3, 0], dtype=torch.long, device=logits.device)
        priority_winner = logits.index_select(-1, priority).argmax(dim=-1)
        return priority[priority_winner]
