import hashlib
from pathlib import Path

import pytest


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_offline_environment_disables_hf_network_fallback():
    from rae_stream.assets import configure_offline_hf_environment

    result = configure_offline_hf_environment({"HF_HOME": "/tmp/hf"})
    assert result["HF_HUB_OFFLINE"] == "1"
    assert result["TRANSFORMERS_OFFLINE"] == "1"
    assert result["HF_DATASETS_OFFLINE"] == "1"


def test_asset_gate_binds_snapshot_and_all_files(tmp_path: Path, monkeypatch):
    import rae_stream.assets as assets

    root = tmp_path / "repo"
    stage = root / "RAE" / "configs" / "stage1" / "pretrained" / "DINOv2-B.yaml"
    stage_bytes = (
        "stage_1:\n  params:\n    encoder_config_path: facebook/dinov2-with-registers-base\n"
        "    pretrained_decoder_path: models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt\n"
        "    normalization_stat_path: models/stats/dinov2/wReg_base/imagenet1k/stat.pt\n"
    ).encode()
    decoder_config = root / "RAE" / "configs" / "decoder" / "ViTXL" / "config.json"
    decoder = root / "models" / "decoders" / "dinov2" / "wReg_base" / "ViTXL_n08" / "model.pt"
    stats = root / "models" / "stats" / "dinov2" / "wReg_base" / "imagenet1k" / "stat.pt"
    snapshot = (
        tmp_path
        / "hub"
        / "models--facebook--dinov2-with-registers-base"
        / "snapshots"
        / assets.HF_REVISION
    )
    refs = snapshot.parent.parent / "refs" / "main"
    refs.parent.mkdir(parents=True)
    refs.write_text(assets.HF_REVISION + "\n", encoding="utf-8")
    values = {
        "stage1_config": _write(stage, stage_bytes),
        "decoder_config": _write(decoder_config, b"{}"),
        "decoder": _write(decoder, b"decoder"),
        "stats": _write(stats, b"stats"),
        "hf_config": _write(snapshot / "config.json", b"config"),
        "hf_processor": _write(snapshot / "preprocessor_config.json", b"processor"),
        "hf_weights": _write(snapshot / "model.safetensors", b"weights"),
    }
    monkeypatch.setattr(assets, "ASSET_SHA256", values)
    result = assets.verify_rae_assets(root, hf_snapshot_root=snapshot)
    assert result["revision"] == assets.HF_REVISION
    assert result["files"]["hf_weights"]["sha256"] == values["hf_weights"]

    (snapshot / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hf_weights SHA"):
        assets.verify_rae_assets(root, hf_snapshot_root=snapshot)


def test_asset_gate_rejects_cache_snapshot_mismatch(tmp_path: Path, monkeypatch):
    import rae_stream.assets as assets

    # The exact bytes are irrelevant here; the mismatch must be rejected
    # before any model loader can choose a different cache.
    monkeypatch.setattr(assets, "ASSET_SHA256", {key: "0" * 64 for key in assets.ASSET_SHA256})
    with pytest.raises(ValueError, match="hf_snapshot_root"):
        assets.verify_rae_assets(
            tmp_path,
            hf_home=tmp_path / "hf",
            hf_snapshot_root=tmp_path / "other-snapshot",
        )
