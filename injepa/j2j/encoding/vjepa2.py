"""Source-faithful frozen V-JEPA 2.1 image encoder adapter."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import os
import math
import warnings
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F


OFFICIAL_SOURCE_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"
OFFICIAL_SOURCE_TREE = "dd6cfc1e792158510b983d827cb2e84f47fd5706"
OFFICIAL_CHECKPOINT_BYTES = 1_664_223_428
_CHECKPOINT_PREFIX = "module.backbone."


@dataclass(frozen=True)
class SpatialEncoding:
    """Configured spatial tokens and an auxiliary diagnostic token mean."""

    grid: Tensor
    global_: Tensor


@dataclass(frozen=True)
class _LoadLedger:
    source_root: str
    source_commit: str
    source_tree: str
    checkpoint: str
    checkpoint_bytes: int
    checkpoint_sha256: str
    builder_file: str
    preprocessor_file: str
    raw_key_count: int
    normalized_key_count: int
    model_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatch: tuple[str, ...]


def _git(source_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _validate_source(source_root: Path) -> Path:
    root = source_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"V-JEPA source root is not a directory: {root}")
    try:
        commit = _git(root, "rev-parse", "HEAD")
        tree = _git(root, "rev-parse", "HEAD^{tree}")
        status = _git(root, "status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot verify V-JEPA source identity: {root}") from exc
    if commit != OFFICIAL_SOURCE_COMMIT:
        raise ValueError(f"unexpected V-JEPA source commit: {commit}")
    if tree != OFFICIAL_SOURCE_TREE:
        raise ValueError(f"unexpected V-JEPA source tree: {tree}")
    if status:
        raise ValueError("V-JEPA source root must be clean")
    return root


def _module_is_inside(module: ModuleType, source_root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    try:
        return Path(module_file).resolve().is_relative_to(source_root)
    except (OSError, RuntimeError):
        return False


@contextmanager
def _official_modules(source_root: Path):
    original_sys_path = list(sys.path)
    source = str(source_root)
    sys.path.insert(0, source)
    try:
        backbones = importlib.import_module("src.hub.backbones")
        preprocessor = importlib.import_module("evals.hub.preprocessor")
        if not _module_is_inside(backbones, source_root):
            raise ImportError(
                "V-JEPA builder was imported outside the verified source root"
            )
        if not _module_is_inside(preprocessor, source_root):
            raise ImportError(
                "V-JEPA preprocessor was imported outside the verified source root"
            )
        yield backbones, preprocessor
    finally:
        sys.path[:] = original_sys_path


def _hash_open_checkpoint(
    checkpoint: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> tuple[Any, str]:
    if expected_bytes <= 0:
        raise ValueError("expected checkpoint bytes must be positive")
    if len(expected_sha256) != 64:
        raise ValueError("expected checkpoint SHA-256 must contain 64 hex characters")
    try:
        int(expected_sha256, 16)
    except ValueError as exc:
        raise ValueError("expected checkpoint SHA-256 is not hexadecimal") from exc

    handle = checkpoint.open("rb")
    try:
        actual_bytes = os.fstat(handle.fileno()).st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"checkpoint byte count mismatch: {actual_bytes} != {expected_bytes}"
            )
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256.lower():
            raise ValueError("checkpoint SHA-256 mismatch")
        handle.seek(0)
        return handle, actual_sha256
    except BaseException:
        handle.close()
        raise


def _normalize_ema_state(
    raw_state: Mapping[object, object],
    model_state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for raw_key, value in raw_state.items():
        if not isinstance(raw_key, str):
            raise ValueError("ema_encoder keys must be strings")
        if not raw_key.startswith(_CHECKPOINT_PREFIX):
            raise ValueError(f"ema_encoder key lacks official prefix: {raw_key}")
        key = raw_key[len(_CHECKPOINT_PREFIX) :]
        if not key or _CHECKPOINT_PREFIX in key:
            raise ValueError(f"ema_encoder key has a repeated prefix: {raw_key}")
        if key in normalized:
            raise ValueError(f"ema_encoder normalization collision: {key}")
        if not isinstance(value, Tensor):
            raise TypeError(f"ema_encoder value is not a tensor: {raw_key}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"ema_encoder tensor is non-finite: {raw_key}")
        normalized[key] = value

    missing = sorted(set(model_state).difference(normalized))
    unexpected = sorted(set(normalized).difference(model_state))
    shape_mismatch = sorted(
        key
        for key in set(model_state).intersection(normalized)
        if tuple(model_state[key].shape) != tuple(normalized[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "ema_encoder does not exactly match the official encoder: "
            f"missing={missing}, unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}"
        )
    return normalized


class _FrozenVJEPA2(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        preprocessor: Callable[[list[Tensor]], object],
        ledger: _LoadLedger,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.load_ledger = ledger
        self.train(False)

    def train(self, mode: bool = True) -> "_FrozenVJEPA2":
        super().train(False)
        return self


def load_frozen_vjepa2(
    *,
    source_root: str | Path,
    checkpoint: str | Path,
    expected_sha256: str,
    expected_bytes: int = OFFICIAL_CHECKPOINT_BYTES,
) -> nn.Module:
    """Load only the official checkpoint's EMA encoder under the frozen ABI."""

    root = _validate_source(Path(source_root))
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    handle, actual_sha256 = _hash_open_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
    )
    try:
        with _official_modules(root) as (backbones, preprocessor_module):
            built = backbones.vjepa2_1_vit_base_384(pretrained=False)
            if not isinstance(built, tuple) or len(built) != 2:
                raise TypeError(
                    "official V-JEPA builder must return (encoder, predictor)"
                )
            encoder, _predictor = built
            if not isinstance(encoder, nn.Module):
                raise TypeError(
                    "official V-JEPA builder did not return an encoder module"
                )
            preprocessor = preprocessor_module.vjepa2_preprocessor(crop_size=384)
        outer = torch.load(
            handle,
            map_location="cpu",
            weights_only=True,
        )
    finally:
        handle.close()

    if not isinstance(outer, Mapping):
        raise TypeError("V-JEPA checkpoint must be a mapping")
    raw_state = outer.get("ema_encoder")
    if not isinstance(raw_state, Mapping):
        raise TypeError("V-JEPA checkpoint must contain a mapping ema_encoder")

    model_state = encoder.state_dict()
    normalized = _normalize_ema_state(raw_state, model_state)
    incompatible = encoder.load_state_dict(normalized, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict V-JEPA encoder load returned incompatible keys")

    encoder.requires_grad_(False)
    encoder.eval()
    if any(parameter.dtype is not torch.float32 for parameter in encoder.parameters()):
        raise TypeError("V-JEPA encoder parameters must be float32")
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("V-JEPA encoder parameters must remain frozen")

    ledger = _LoadLedger(
        source_root=str(root),
        source_commit=OFFICIAL_SOURCE_COMMIT,
        source_tree=OFFICIAL_SOURCE_TREE,
        checkpoint=str(checkpoint_path),
        checkpoint_bytes=expected_bytes,
        checkpoint_sha256=actual_sha256,
        builder_file=str(Path(backbones.__file__).resolve()),
        preprocessor_file=str(Path(preprocessor_module.__file__).resolve()),
        raw_key_count=len(raw_state),
        normalized_key_count=len(normalized),
        model_key_count=len(model_state),
        missing_keys=(),
        unexpected_keys=(),
        shape_mismatch=(),
    )
    return _FrozenVJEPA2(encoder, preprocessor, ledger)


def _validate_rgb(rgb_uint8: Tensor) -> None:
    if not isinstance(rgb_uint8, Tensor):
        raise TypeError("RGB input must be a torch tensor")
    if rgb_uint8.device.type != "cpu":
        raise ValueError("RGB input must reside on CPU")
    if rgb_uint8.dtype is not torch.uint8:
        raise TypeError("RGB input must have dtype torch.uint8")
    if rgb_uint8.ndim != 4:
        raise ValueError("RGB input must have shape [B,3,H,W]")
    batch, channels, height, width = rgb_uint8.shape
    if batch <= 0 or channels != 3 or height <= 0 or width <= 0:
        raise ValueError("RGB input must be non-empty [B,3,H,W]")
    if not rgb_uint8.is_contiguous():
        raise ValueError("RGB input must be contiguous")


def _encoder_device(model: _FrozenVJEPA2) -> torch.device:
    parameters = tuple(model.encoder.parameters())
    if not parameters:
        raise ValueError("V-JEPA encoder has no parameters")
    devices = {parameter.device for parameter in parameters}
    if len(devices) != 1:
        raise ValueError("V-JEPA encoder parameters span multiple devices")
    if any(parameter.dtype is not torch.float32 for parameter in parameters):
        raise TypeError("V-JEPA encoder parameters must be float32")
    if any(parameter.requires_grad for parameter in parameters):
        raise RuntimeError("V-JEPA encoder parameters must remain frozen")
    return next(iter(devices))


def _preprocess_batch(model: _FrozenVJEPA2, rgb_uint8: Tensor) -> Tensor:
    samples: list[Tensor] = []
    with torch.autocast(device_type="cpu", enabled=False):
        for image in rgb_uint8:
            transformed = model.preprocessor([image])
            if not isinstance(transformed, (list, tuple)) or len(transformed) != 1:
                raise ValueError("official preprocessor must return one transformed frame")
            sample = transformed[0]
            if not isinstance(sample, Tensor):
                raise TypeError("official preprocessor output must be a tensor")
            if sample.shape != (3, 1, 384, 384):
                raise ValueError(
                    "official preprocessor output must have shape [3,1,384,384]"
                )
            if sample.device.type != "cpu" or sample.dtype is not torch.float32:
                raise TypeError("official preprocessor output must be CPU float32")
            if not bool(torch.isfinite(sample).all()):
                raise ValueError("official preprocessor output must be finite")
            samples.append(sample.contiguous())
    return torch.stack(samples, dim=0)


def encode_images(
    model: nn.Module, rgb_uint8: Tensor, *, pool_kernel_size: int | None = 4,
) -> SpatialEncoding:
    """Encode images; ``None`` preserves the encoder's native spatial tokens."""

    if not isinstance(model, _FrozenVJEPA2):
        raise TypeError("model must be returned by load_frozen_vjepa2")
    _validate_rgb(rgb_uint8)
    model.train(False)
    device = _encoder_device(model)

    with torch.inference_mode():
        preprocessed = _preprocess_batch(model, rgb_uint8).to(
            device=device,
            dtype=torch.float32,
        )
        if preprocessed.dtype is not torch.float32:
            raise TypeError("encoder input must be float32")
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = model.encoder(preprocessed)

        if not isinstance(tokens, Tensor):
            raise TypeError("V-JEPA encoder output must be a tensor")
        if tokens.ndim != 3 or tokens.shape[0] != rgb_uint8.shape[0] or min(tokens.shape[1:]) < 1:
            raise ValueError("V-JEPA encoder output must have shape [B,spatial,latent]")
        if tokens.shape[2] != 768:
            raise ValueError("V-JEPA ViT-B checkpoint requires latent width 768")
        if tuple(tokens.shape[1:]) != (576, 768):
            warnings.warn(f"Non-default native V-JEPA grid {tuple(tokens.shape[1:])}; consumers must use its declared shape.", UserWarning)
        if tokens.dtype is not torch.float32:
            raise TypeError("V-JEPA encoder output must be float32")
        if tokens.device != device:
            raise ValueError("V-JEPA encoder output is on the wrong device")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("V-JEPA encoder output must be finite")

        if pool_kernel_size is None:
            grid = tokens.contiguous()
        else:
            if type(pool_kernel_size) is not int or pool_kernel_size <= 0:
                raise ValueError("pool_kernel_size must be a positive integer or None")
            side = math.isqrt(tokens.shape[1])
            if side * side != tokens.shape[1] or side % pool_kernel_size:
                raise ValueError("spatial pooling requires a square grid divisible by its kernel")
            spatial = tokens.reshape(-1, side, side, tokens.shape[2]).permute(0, 3, 1, 2)
            pooled = F.avg_pool2d(spatial, kernel_size=pool_kernel_size, stride=pool_kernel_size)
            grid = pooled.flatten(2).transpose(1, 2).contiguous()
        global_ = grid.mean(dim=1, dtype=torch.float32)
        if grid.dtype is not torch.float32 or global_.dtype is not torch.float32:
            raise TypeError("spatial encoding must be float32")
        if not bool(torch.isfinite(grid).all()) or not bool(torch.isfinite(global_).all()):
            raise ValueError("spatial encoding must be finite")
        return SpatialEncoding(grid=grid, global_=global_)
