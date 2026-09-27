"""Thin V9 policy adapter for the Habitat evaluation runner.

The adapter owns no model math.  Callers provide the already-constructed V9
inference callable (which should use the official V-JEPA/INTACT objects); this
module only normalizes the RGB-only runner ABI and records provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any, Callable

from .adapters import _canonical_action


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V9PolicyAdapter:
    """Wrap an existing V9 one-step inference function without reimplementing it."""

    provenance = "j2j_v9_existing_intact_controller"

    def __init__(
        self,
        act_fn: Callable[..., Any],
        *,
        reset_fn: Callable[..., Any] | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        preprocess_fn: Callable[..., Any] | None = None,
    ) -> None:
        if not callable(act_fn):
            raise TypeError("V9 act_fn must be callable")
        self.act_fn = act_fn
        self.reset_fn = reset_fn
        self.preprocess_fn = preprocess_fn
        if checkpoint is None:
            raise ValueError("V9 adapter requires an admitted checkpoint path")
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"V9 checkpoint does not exist: {checkpoint_path}")
        if checkpoint_sha256 is None or str(checkpoint_sha256).strip() == "":
            raise ValueError("V9 checkpoint SHA-256 is required")
        actual_sha = _sha256_file(checkpoint_path)
        if str(checkpoint_sha256).lower() != actual_sha:
            raise ValueError(
                f"V9 checkpoint SHA-256 mismatch: expected {checkpoint_sha256!r}, got {actual_sha!r}"
            )
        self.checkpoint = str(checkpoint_path)
        self.checkpoint_sha256 = actual_sha
        self.calls = 0

    def reset(self, goal_rgb: Any) -> None:
        self.calls = 0
        if self.reset_fn is not None:
            self.reset_fn(goal_rgb)

    def act(self, current_rgb: Any, goal_rgb: Any, history: Any) -> Any:
        self.calls += 1
        current, goal = current_rgb, goal_rgb
        if self.preprocess_fn is not None:
            current, goal = self.preprocess_fn(current_rgb, goal_rgb)
        result = self.act_fn(current, goal, history)
        # V9's runner ABI is categorical.  Do not silently convert continuous
        # commands here; an accidental NWM result should fail loudly.
        if isinstance(result, Mapping) and "action_id" in result:
            _canonical_action(result["action_id"])
        else:
            _canonical_action(result)
        return result

    def close(self) -> None:
        return None

    def provenance_record(self) -> dict[str, Any]:
        return {
            "adapter": self.__class__.__name__,
            "provenance": self.provenance,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "calls": self.calls,
        }


__all__ = ["V9PolicyAdapter"]
