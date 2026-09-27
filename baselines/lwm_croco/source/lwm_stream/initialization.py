"""Strict loading of the original CroCo backbone, leaving LWM heads fresh."""

import hashlib
import os
import re

import torch


def _identity(stat):
    # An unlinked-but-open original is still the hashed object; ctime/nlink may
    # change when another file replaces its pathname without changing its bytes.
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def initialize_croco(model, checkpoint_path, expected_sha256, expected_kwargs):
    """Validate everything before copying any pretrained backbone values."""
    if not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("an explicit lowercase SHA-256 is required")
    with open(checkpoint_path, "rb") as file:
        before = os.fstat(file.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("CroCo asset SHA-256 mismatch")
        file.seek(0)
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        if _identity(before) != _identity(os.fstat(file.fileno())):
            raise ValueError("CroCo asset changed while reading")

    if not isinstance(checkpoint, dict) or set(checkpoint) != {"model", "croco_kwargs"}:
        raise ValueError("expected original CroCo model/kwargs archive")
    if checkpoint["croco_kwargs"] != expected_kwargs:
        raise ValueError("CroCo architecture kwargs mismatch")
    state = checkpoint["model"]
    destination = model.croco.state_dict()
    if not isinstance(state, dict) or set(state) != set(destination):
        raise ValueError("CroCo state keys mismatch; no navigation checkpoint fallback")
    for name, tensor in state.items():
        current = destination[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                or tensor.shape != current.shape or tensor.dtype != current.dtype):
            raise ValueError(f"CroCo tensor shape/dtype/layout mismatch: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite CroCo tensor: {name}")

    model.croco.load_state_dict(state, strict=True, assign=False)
    return {"status": "CROCO_STRICT_LOAD_PASS", "sha256": expected_sha256,
            "bytes": before.st_size, "state_keys": len(state),
            "state_numel": sum(t.numel() for t in state.values()),
            "loaded_module": "croco", "croco_kwargs": dict(expected_kwargs),
            "other_modules_loaded": False}
