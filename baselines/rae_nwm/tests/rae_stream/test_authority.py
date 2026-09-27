import json
from pathlib import Path

import pytest

from rae_stream.authority import verify_code_authority
from rae_stream.config_guard import OFFICIAL_COMMIT, canonical_digest, sha256_file


def _write_authority(tmp_path: Path, *, source: bytes = b"source") -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "entry.py"
    target.write_bytes(source)
    payload = {
        "schema": "rae_stream_code_authority_v1",
        "official_commit": OFFICIAL_COMMIT,
        "files": {"entry.py": sha256_file(target)},
    }
    payload["payload_sha256"] = canonical_digest(payload)
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return authority, sha256_file(authority)


def test_external_authority_binds_source_bytes(tmp_path: Path, monkeypatch):
    import rae_stream.authority as module

    authority, digest = _write_authority(tmp_path)
    monkeypatch.setattr(module, "_runtime_paths", lambda root: {"entry.py"})
    result = verify_code_authority(tmp_path / "repo", authority, expected_sha256=digest)
    assert result["sha256"] == digest
    (tmp_path / "repo" / "entry.py").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source SHA"):
        verify_code_authority(tmp_path / "repo", authority, expected_sha256=digest)


def test_external_authority_rejects_checkout_bytecode(tmp_path: Path, monkeypatch):
    authority, digest = _write_authority(tmp_path)
    monkeypatch.setattr(__import__("rae_stream.authority", fromlist=["_runtime_paths"]), "_runtime_paths", lambda root: {"entry.py"})
    cache = tmp_path / "repo" / "__pycache__"
    cache.mkdir()
    (cache / "entry.cpython-313.pyc").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="bytecode"):
        verify_code_authority(tmp_path / "repo", authority, expected_sha256=digest)


def test_external_authority_rejects_symlink_file(tmp_path: Path, monkeypatch):
    import rae_stream.authority as module

    authority, digest = _write_authority(tmp_path)
    real = tmp_path / "authority.real.json"
    authority.rename(real)
    authority.symlink_to(real)
    monkeypatch.setattr(module, "_runtime_paths", lambda root: {"entry.py"})
    with pytest.raises((ValueError, FileNotFoundError), match="symlink|regular"):
        verify_code_authority(tmp_path / "repo", authority, expected_sha256=digest)


def test_code_authority_must_be_outside_mutable_checkout(tmp_path: Path, monkeypatch):
    import rae_stream.authority as module

    authority, digest = _write_authority(tmp_path)
    inside = tmp_path / "repo" / "receipts" / "authority.json"
    inside.parent.mkdir()
    inside.write_bytes(authority.read_bytes())
    monkeypatch.setattr(module, "_runtime_paths", lambda root: {"entry.py"})
    with pytest.raises(ValueError, match="outside.*checkout|external"):
        verify_code_authority(tmp_path / "repo", inside, expected_sha256=digest)


def test_runtime_surface_preserves_unicode_path_bytes(monkeypatch, tmp_path: Path):
    """Authority enumeration must parse only Git's actual record terminator."""

    import subprocess
    import rae_stream.authority as module

    unicode_path = "rae_stream/config_guard.py\u2028"

    def fake_check_output(command, **kwargs):
        assert kwargs.get("text") is True
        if "--others" in command:
            return unicode_path + "\0"
        return "README.md\0"

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)
    paths = module._runtime_paths(tmp_path)
    assert unicode_path in paths
    assert unicode_path + "\0" not in paths
