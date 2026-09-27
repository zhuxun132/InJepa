"""Identity and offline-load checks for the fixed official RAE assets.

The RAE model code is intentionally untouched.  This module only verifies the
files that its unchanged stage-1 configuration will consume and constructs a
local Hugging Face snapshot identity so a missing file cannot trigger the
upstream encoder's network fallback.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .config_guard import sha256_file


HF_REVISION = "a1d738ccfa7ae170945f210395d99dde8adb1805"
ASSET_SHA256: dict[str, str] = {
    "stage1_config": "b28412264fc56b812c652c93403627bf11e00c131f4b0a6fc83f5b4339fc63a5",
    "decoder_config": "0c9845cd65131b4074fbac77f7d36fdcdca71f455e75ae9ba169b34b8e1e3614",
    "decoder": "5fedf7c9660476a709e122cef18385c917532914a23f034388e1c1a52bde2be6",
    "stats": "84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59",
    "hf_config": "6af60aa760138fc90db0ba37b7701730b140b9bf1742514412d03050e72bf7e0",
    "hf_processor": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
    "hf_weights": "7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362",
}


def _require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing RAE {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"RAE {label} SHA mismatch: got {actual}, expected {expected}")
    return actual


def _snapshot_root(hf_home: str | Path, revision: str = HF_REVISION) -> Path:
    home = Path(hf_home).resolve()
    return home / "hub" / "models--facebook--dinov2-with-registers-base" / "snapshots" / revision


def verify_rae_assets(
    repo_root: str | Path,
    *,
    hf_home: str | Path | None = None,
    hf_snapshot_root: str | Path | None = None,
    expected_revision: str = HF_REVISION,
) -> dict[str, Any]:
    """Verify all fixed decoder/stats/config/HF bytes and local snapshot refs."""

    root = Path(repo_root).resolve()
    if expected_revision != HF_REVISION:
        raise ValueError(f"only pinned DINOv2 snapshot {HF_REVISION} is admitted")
    derived_snapshot = _snapshot_root(hf_home, expected_revision) if hf_home is not None else None
    if hf_snapshot_root is not None:
        snapshot = Path(hf_snapshot_root).resolve()
        if derived_snapshot is not None and snapshot != derived_snapshot:
            raise ValueError(
                "hf_snapshot_root must equal the pinned snapshot below hf_home; "
                "verifying one cache while training reads another is forbidden"
            )
    elif derived_snapshot is not None:
        snapshot = derived_snapshot
    else:
        raise ValueError("a dedicated hf_home or hf_snapshot_root is required for offline RAE loading")
    stage1 = root / "RAE" / "configs" / "stage1" / "pretrained" / "DINOv2-B.yaml"
    decoder_config = root / "RAE" / "configs" / "decoder" / "ViTXL" / "config.json"
    decoder = root / "models" / "decoders" / "dinov2" / "wReg_base" / "ViTXL_n08" / "model.pt"
    stats = root / "models" / "stats" / "dinov2" / "wReg_base" / "imagenet1k" / "stat.pt"
    paths = {
        "stage1_config": stage1,
        "decoder_config": decoder_config,
        "decoder": decoder,
        "stats": stats,
    }
    hashes = {label: _require_hash(path, ASSET_SHA256[label], label) for label, path in paths.items()}

    # Bind the loader's relative paths and encoder ID to the exact stage-1
    # config.  This avoids a valid but unrelated decoder being accepted under
    # the expected filename.
    try:
        import yaml

        stage_config = yaml.safe_load(stage1.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"could not parse fixed RAE stage-1 config: {stage1}") from exc
    params = (((stage_config or {}).get("stage_1") or {}).get("params") or {})
    if params.get("encoder_config_path") != "facebook/dinov2-with-registers-base":
        raise RuntimeError("fixed stage-1 config has an unexpected DINOv2 encoder ID")
    if params.get("pretrained_decoder_path") != "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt":
        raise RuntimeError("fixed stage-1 config has an unexpected decoder path")
    if params.get("normalization_stat_path") != "models/stats/dinov2/wReg_base/imagenet1k/stat.pt":
        raise RuntimeError("fixed stage-1 config has an unexpected stats path")

    refs = snapshot.parent.parent / "refs" / "main"
    if not refs.is_file() or refs.read_text(encoding="utf-8").strip() != expected_revision:
        raise RuntimeError(f"HF snapshot refs/main is not pinned to {expected_revision}: {refs}")
    hf_paths = {
        "hf_config": snapshot / "config.json",
        "hf_processor": snapshot / "preprocessor_config.json",
        "hf_weights": snapshot / "model.safetensors",
    }
    hashes.update({label: _require_hash(path, ASSET_SHA256[label], label) for label, path in hf_paths.items()})

    return {
        "revision": expected_revision,
        "files": {label: {"path": str(path), "sha256": hashes[label]} for label, path in {**paths, **hf_paths}.items()},
        "snapshot_root": str(snapshot),
        "offline_required": True,
    }


def configure_offline_hf_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return a copy with network fallback disabled for Transformers/HF Hub."""

    result = {str(key): str(value) for key, value in environment.items()}
    result["HF_HUB_OFFLINE"] = "1"
    result["TRANSFORMERS_OFFLINE"] = "1"
    result["HF_DATASETS_OFFLINE"] = "1"
    return result


__all__ = [
    "ASSET_SHA256",
    "HF_REVISION",
    "configure_offline_hf_environment",
    "verify_rae_assets",
]
