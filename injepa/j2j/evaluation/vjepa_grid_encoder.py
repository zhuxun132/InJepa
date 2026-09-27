"""Shared evaluation-only RGB to frozen V-JEPA spatial-grid encoder."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import math
import warnings
from typing import Any

import numpy as np
import torch
from torch import Tensor

from j2j.authority import sha256_file
from j2j.encoding.cache import apply_whitening, load_whitening
from j2j.encoding.vjepa2 import (
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_TREE,
    encode_images,
    load_frozen_vjepa2,
)


_TOKENS = 36
_FEATURES = 768
_HEX = frozenset("0123456789abcdef")
# SHA-256 identities of the two already-verified current encoding contracts:
# VJEPA_PREPROCESS_CLOSURE_SERVER_R1.json and VJEPA_POOL_CLOSURE_SERVER_R1.json.
VJEPA_PREPROCESS_IDENTITY_SHA256 = (
    "da5f73a48fb8f71dee05045b829f1a288a431c8d42a5e76d5614f34a0614e7a3"
)
VJEPA_POOL_IDENTITY_SHA256 = (
    "026792674d2e1d55f4f925f852c640cd810e7495c62f22864031fd34b044d4ef"
)
VJEPA_NATIVE_POOL_IDENTITY_SHA256 = "815079d56d54eea6ad8f1a2a5ace5bdc6384217bc8d10477ec3b1a9f67c8a22c"
_LOAD_LEDGER_FIELDS = (
    "source_root",
    "source_commit",
    "source_tree",
    "checkpoint",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "builder_file",
    "preprocessor_file",
    "raw_key_count",
    "normalized_key_count",
    "model_key_count",
    "missing_keys",
    "unexpected_keys",
    "shape_mismatch",
)


def _expected_sha256(value: str, *, label: str) -> str:
    expected = str(value).strip().lower()
    if len(expected) != 64 or any(character not in _HEX for character in expected):
        raise ValueError(f"{label} SHA-256 must be 64 hexadecimal characters")
    return expected


def _verified_asset(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if expected_bytes is not None:
        if isinstance(expected_bytes, bool) or int(expected_bytes) <= 0:
            raise ValueError(f"{label} expected bytes must be positive")
        if resolved.stat().st_size != int(expected_bytes):
            raise ValueError(
                f"{label} byte count mismatch: "
                f"{resolved.stat().st_size} != {int(expected_bytes)}"
            )
    expected = _expected_sha256(expected_sha256, label=label)
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected!r}, got {actual!r}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": actual,
    }


def _canonical_rgb(image: Any) -> Tensor:
    if isinstance(image, Tensor):
        tensor = image.detach().cpu()
        if tensor.dtype is not torch.uint8:
            raise TypeError("RGB input must use uint8 sensor values")
    else:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise TypeError("RGB input must use uint8 sensor values")
        tensor = torch.from_numpy(np.ascontiguousarray(array))

    if tensor.ndim != 3:
        raise ValueError("RGB input must have shape [3,H,W] or [H,W,3]")
    if tensor.shape[0] == 3:
        chw = tensor
    elif tensor.shape[-1] == 3:
        chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError("RGB input must have shape [3,H,W] or [H,W,3]")
    if chw.shape[1] <= 0 or chw.shape[2] <= 0:
        raise ValueError("RGB input must have shape [3,H,W] or [H,W,3]")
    return chw.contiguous()


def _canonical_grid(value: Any, *, label: str) -> np.ndarray:
    grid = np.asarray(value)
    if grid.shape != (_TOKENS, _FEATURES):
        raise ValueError(f"{label} must have shape [36,768]")
    result = np.asarray(grid, dtype="<f4", order="C")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _strict_load_ledger(
    encoder: Any,
    *,
    source_root: Path,
    checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    raw = getattr(encoder, "load_ledger", None)
    if raw is None or any(not hasattr(raw, field) for field in _LOAD_LEDGER_FIELDS):
        raise ValueError("V-JEPA encoder did not expose its complete strict-load ledger")
    ledger = {field: getattr(raw, field) for field in _LOAD_LEDGER_FIELDS}
    if Path(str(ledger["source_root"])).expanduser().resolve() != source_root:
        raise ValueError("V-JEPA load ledger source root mismatch")
    if ledger["source_commit"] != OFFICIAL_SOURCE_COMMIT:
        raise ValueError("V-JEPA load ledger source commit mismatch")
    if ledger["source_tree"] != OFFICIAL_SOURCE_TREE:
        raise ValueError("V-JEPA load ledger source tree mismatch")
    expected_checkpoint = {
        "path": str(Path(str(ledger["checkpoint"])).expanduser().resolve()),
        "bytes": ledger["checkpoint_bytes"],
        "sha256": ledger["checkpoint_sha256"],
    }
    if expected_checkpoint != dict(checkpoint_identity):
        raise ValueError("V-JEPA load ledger checkpoint identity mismatch")
    counts = tuple(ledger[name] for name in (
        "raw_key_count",
        "normalized_key_count",
        "model_key_count",
    ))
    if any(type(value) is not int or value <= 0 for value in counts) or len(set(counts)) != 1:
        raise ValueError("V-JEPA load ledger key census is not exact")
    for field in ("missing_keys", "unexpected_keys", "shape_mismatch"):
        values = tuple(ledger[field])
        if values:
            raise ValueError(f"V-JEPA load ledger {field} is nonempty")
        ledger[field] = values
    for field in ("builder_file", "preprocessor_file"):
        value = Path(str(ledger[field])).expanduser().resolve()
        if not value.is_relative_to(source_root):
            raise ValueError(f"V-JEPA load ledger {field} escaped the verified source")
        ledger[field] = str(value)
    ledger["source_root"] = str(source_root)
    ledger["checkpoint"] = checkpoint_identity["path"]
    ledger["load_status"] = "STRICT_FROZEN_VJEPA2_LOADED"
    return ledger


def build_vjepa_grid_encoder(
    *,
    source_root: str | Path,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    whitening: str | Path,
    whitening_sha256: str,
    device: str | torch.device,
) -> tuple[Callable[[Any], Tensor], dict[str, Any]]:
    """Build the single shared uint8 RGB -> whitened ``[36,768]`` seam."""

    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"V-JEPA source root is not a directory: {root}")
    checkpoint_identity = _verified_asset(
        checkpoint,
        checkpoint_sha256,
        label="V-JEPA checkpoint",
        expected_bytes=checkpoint_bytes,
    )
    whitening_identity = _verified_asset(
        whitening,
        whitening_sha256,
        label="whitening artifact",
    )

    encoder = load_frozen_vjepa2(
        source_root=source_root,
        checkpoint=checkpoint_identity["path"],
        expected_sha256=checkpoint_identity["sha256"],
        expected_bytes=checkpoint_identity["bytes"],
    )
    encoder.to(device=torch.device(device)).eval().requires_grad_(False)
    load_ledger = _strict_load_ledger(
        encoder,
        source_root=root,
        checkpoint_identity=checkpoint_identity,
    )
    transform = load_whitening(
        whitening_identity["path"],
        expected_sha256=whitening_identity["sha256"],
    )

    def encode(image: Any) -> Tensor:
        rgb = _canonical_rgb(image).unsqueeze(0)
        encoded = encode_images(encoder, rgb)
        raw_grid = encoded.grid
        if not isinstance(raw_grid, Tensor) or tuple(raw_grid.shape) != (
            1,
            _TOKENS,
            _FEATURES,
        ):
            raise ValueError("V-JEPA encoded grid must have shape [1,36,768]")
        grid = _canonical_grid(
            raw_grid[0].detach().cpu().numpy(),
            label="V-JEPA encoded grid",
        )
        whitened = _canonical_grid(
            apply_whitening(grid, transform),
            label="whitened V-JEPA grid",
        )
        result = torch.from_numpy(whitened).detach().cpu().contiguous()
        result.requires_grad_(False)
        return result

    provenance = {
        "vjepa_checkpoint": checkpoint_identity,
        "whitening": whitening_identity,
        "encoder_source_root": str(root),
        "load_ledger": load_ledger,
        "coordinate_identity": {
            "vjepa_source_commit": load_ledger["source_commit"],
            "vjepa_source_tree": load_ledger["source_tree"],
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "preprocess_sha256": VJEPA_PREPROCESS_IDENTITY_SHA256,
            "pool_sha256": VJEPA_POOL_IDENTITY_SHA256,
            "whitening_sha256": whitening_identity["sha256"],
        },
    }
    return encode, provenance


def build_native_vjepa_grid_encoder(
    *, source_root: str | Path, checkpoint: str | Path,
    checkpoint_sha256: str, checkpoint_bytes: int, device: str | torch.device,
    expected_spatial_shape: tuple[int, int] = (576, 768),
) -> tuple[Callable[[Any], Tensor], dict[str, Any]]:
    """Strict frozen native encoder, without pooling or whitening.

    Shape is a declared coordinate identity; this function never resizes tokens
    to make an incompatible encoder/cache/model appear compatible.
    """
    shape = tuple(expected_spatial_shape)
    if len(shape) != 2 or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("native spatial shape must contain two positive integers")
    if shape[1] != _FEATURES or math.isqrt(shape[0]) ** 2 != shape[0]:
        raise ValueError("native spatial shape requires a square token grid with width768")
    if shape != (576, 768):
        warnings.warn(f"Non-default native grid shape {shape}; encoder/cache/model must match.", UserWarning)
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"V-JEPA source root is not a directory: {root}")
    checkpoint_identity = _verified_asset(checkpoint, checkpoint_sha256,
                                         label="V-JEPA checkpoint", expected_bytes=checkpoint_bytes)
    encoder = load_frozen_vjepa2(source_root=source_root, checkpoint=checkpoint_identity["path"],
        expected_sha256=checkpoint_identity["sha256"], expected_bytes=checkpoint_identity["bytes"])
    encoder.to(device=torch.device(device)).eval().requires_grad_(False)
    load_ledger = _strict_load_ledger(encoder, source_root=root, checkpoint_identity=checkpoint_identity)

    def encode(image: Any) -> Tensor:
        rgb = _canonical_rgb(image).unsqueeze(0)
        with torch.no_grad():
            raw = encode_images(encoder, rgb, pool_kernel_size=None).grid
        if not isinstance(raw, Tensor) or tuple(raw.shape) != (1, *shape):
            raise ValueError(f"native V-JEPA grid must match [1,{shape[0]},{shape[1]}]")
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("native V-JEPA grid must be finite")
        result = raw[0].detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("native V-JEPA grid overflows canonical float32 coordinates")
        return result

    provenance = {
        "vjepa_checkpoint": checkpoint_identity, "whitening": None,
        "encoder_source_root": str(root), "load_ledger": load_ledger,
        "coordinate_identity": {
            "vjepa_source_commit": load_ledger["source_commit"],
            "vjepa_source_tree": load_ledger["source_tree"],
            "checkpoint_sha256": checkpoint_identity["sha256"],
            "preprocess_sha256": VJEPA_PREPROCESS_IDENTITY_SHA256,
            "pool_sha256": VJEPA_NATIVE_POOL_IDENTITY_SHA256,
            "whitening_sha256": None, "representation": "raw", "spatial_shape": shape,
        },
    }
    return encode, provenance


__all__ = [
    "VJEPA_POOL_IDENTITY_SHA256",
    "VJEPA_PREPROCESS_IDENTITY_SHA256",
    "build_vjepa_grid_encoder",
    "build_native_vjepa_grid_encoder",
]
