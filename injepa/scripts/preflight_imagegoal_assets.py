#!/usr/bin/env python3
"""Preflight published assets and the frozen Habitat ImageGoal contract.

This is an evaluation-only command.  It never downloads, extracts, rewrites,
or feeds observations into training.  A non-zero exit means the runtime must
not be started.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - launch environment issue
    raise SystemExit("PyYAML is required for evaluation preflight") from exc


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from j2j.evaluation import (  # noqa: E402
    EXPECTED_SCENE_IDS,
    canonical_episode_key,
    canonical_scene_id,
    validate_imagegoal_config,
)


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _resolved_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the fully expanded scientific config before launch asset overrides."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_yaml(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"recursive config defaults include: {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("evaluation config must contain a mapping")
    defaults = value.pop("defaults", [])
    if not isinstance(defaults, list):
        raise TypeError("config.defaults must be a list when present")
    merged: dict[str, Any] = {}
    for item in defaults:
        if isinstance(item, str):
            name = item
        elif isinstance(item, Mapping):
            # Hydra group entries (e.g. launcher: local) do not contribute to
            # the ImageGoal contract; _self_ is the current file.
            if "_self_" in item:
                continue
            continue
        else:
            continue
        if name == "_self_":
            continue
        candidate = path.parent / f"{name}.yaml"
        if candidate.is_file():
            merged = _deep_merge(merged, _load_yaml(candidate, seen))
    return _deep_merge(merged, value)


def _directory_fingerprint(root: Path) -> str:
    """Deterministically bind a scene-root tree to its receipt."""
    digest = hashlib.sha256()
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big")); digest.update(rel)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _load_episodes(path: Path) -> tuple[list[dict[str, Any]], str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        payload = json.load(handle)
    if isinstance(payload, dict):
        episodes = payload.get("episodes")
    else:
        episodes = payload
    if not isinstance(episodes, list) or not all(isinstance(item, dict) for item in episodes):
        raise ValueError("episode ledger must be a JSON list or {'episodes': list}")
    return episodes, sha256_file(path)


def _git_head(path: Path) -> str | None:
    if not path.exists() or not path.is_dir():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _scene_id_from_value(value: Any, *, expected_scene_ids: tuple[str, ...] = EXPECTED_SCENE_IDS) -> str:
    """Resolve one official MP3D scene token from a ledger/path value."""
    return canonical_scene_id(value, expected_scene_ids=expected_scene_ids)


def validate_episode_ledger(
    episodes: Any,
    *,
    expected_count: int = 495,
    expected_scene_ids: tuple[str, ...] = EXPECTED_SCENE_IDS,
) -> dict[str, Any]:
    """Validate count and exact scene coverage of the supplied ledger.

    The returned scene IDs and ``episode_keys`` are resolved from every
    episode, not inferred from a filename or from a count.  Habitat's bare
    ``episode_id`` is only scene-local, so uniqueness is enforced on the
    canonical ``(scene_id, episode_id)`` pair.  Unknown scenes and a missing
    frozen scene are both hard errors before Habitat is constructed.
    """

    if not isinstance(episodes, list):
        raise TypeError("episode ledger must be a list")
    if len(episodes) != expected_count:
        raise ValueError(
            f"episode ledger has {len(episodes)} episodes, expected {expected_count}"
        )
    scene_counts = {scene: 0 for scene in expected_scene_ids}
    # Habitat's MP3D episode IDs are assigned from zero independently for each
    # scene.  The stable ledger key is therefore the canonical scene token and
    # the normalized bare episode ID, not the bare ID by itself.
    episode_keys: dict[tuple[str, str], int] = {}
    episode_key_rows: list[list[str]] = []
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TypeError(f"episode ledger entry {index} must be a mapping")
        if "scene_id" not in episode:
            raise ValueError(f"episode ledger entry {index} has no scene_id")
        episode_identity = episode.get("episode_id", episode.get("id"))
        if episode_identity is None or not str(episode_identity).strip():
            raise ValueError(f"episode ledger entry {index} has no episode identity")
        key = canonical_episode_key(
            str(episode["scene_id"]),
            episode_identity,
            expected_scene_ids=expected_scene_ids,
        )
        scene, identity = key
        previous_index = episode_keys.get(key)
        if previous_index is not None:
            raise ValueError(
                f"duplicate episode identity {key!r} at rows "
                f"{previous_index} and {index}"
            )
        episode_keys[key] = index
        episode_key_rows.append([scene, identity])
        scene_counts[scene] += 1
    missing = [scene for scene, count in scene_counts.items() if count == 0]
    if missing:
        raise ValueError("episode ledger is missing frozen scenes: " + ", ".join(missing))
    return {
        "count": len(episodes),
        "scene_ids": list(expected_scene_ids),
        "scene_counts": scene_counts,
        "episode_keys": episode_key_rows,
    }


def _scene_member_kind(path_name: str) -> str | None:
    lower = path_name.lower()
    if lower.endswith(".glb"):
        return "glb"
    if ".navmesh" in lower or lower.endswith(".navmesh"):
        return "navmesh"
    return None


def _scene_token_in_path(path_name: str, expected_scene_ids: tuple[str, ...]) -> str | None:
    normalized = path_name.replace("\\", "/")
    matches = [scene for scene in expected_scene_ids if scene in normalized]
    return matches[0] if len(matches) == 1 else None


def _validate_scene_members(
    names: Any,
    *,
    expected_scene_ids: tuple[str, ...] = EXPECTED_SCENE_IDS,
) -> dict[str, Any]:
    members = {
        scene: {"glb": [], "navmesh": []} for scene in expected_scene_ids
    }
    for name in names:
        kind = _scene_member_kind(str(name))
        if kind is None:
            continue
        scene = _scene_token_in_path(str(name), expected_scene_ids)
        if scene is not None:
            members[scene][kind].append(str(name))
    missing = {
        scene: [kind for kind, values in kinds.items() if not values]
        for scene, kinds in members.items()
        if any(not values for values in kinds.values())
    }
    if missing:
        details = "; ".join(
            f"{scene}: {','.join(kinds)}" for scene, kinds in sorted(missing.items())
        )
        raise ValueError("required GLB/navmesh scene members are missing: " + details)
    return {
        "scene_ids": list(expected_scene_ids),
        "members": members,
    }


def validate_scene_root(
    root: Path | str,
    *,
    expected_scene_ids: tuple[str, ...] = EXPECTED_SCENE_IDS,
) -> dict[str, Any]:
    """Require a GLB and navmesh member for every frozen scene."""

    root = Path(root).expanduser()
    if not root.is_dir():
        raise ValueError(f"scene_root is not a directory: {root}")
    names = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()]
    if not names:
        raise ValueError(f"scene_root has no scene files: {root}")
    result = _validate_scene_members(names, expected_scene_ids=expected_scene_ids)
    result["path"] = str(root)
    return result


def validate_scene_archive(
    archive: Path | str,
    *,
    expected_scene_ids: tuple[str, ...] = EXPECTED_SCENE_IDS,
) -> dict[str, Any]:
    """Validate required GLB/navmesh members inside an MP3D ZIP archive."""

    archive = Path(archive)
    with zipfile.ZipFile(archive, "r") as handle:
        return _validate_scene_members(handle.namelist(), expected_scene_ids=expected_scene_ids)


def validate_habitat_revisions(
    lab_head: str | None,
    sim_head: str | None,
    expected_lab: Any,
    expected_sim: Any,
) -> dict[str, str]:
    """Require exact source identities for the Lab/Sim pair."""

    values = {
        "habitat_lab": (lab_head, expected_lab),
        "habitat_sim": (sim_head, expected_sim),
    }
    result: dict[str, str] = {}
    errors: list[str] = []
    for label, (actual, expected) in values.items():
        if expected is None or str(expected).strip() == "":
            errors.append(f"{label} expected revision is missing")
        elif actual is None or str(actual).strip() == "":
            errors.append(f"{label} git revision is unavailable")
        elif str(actual) != str(expected):
            errors.append(f"{label} revision mismatch: expected {expected!r}, got {actual!r}")
        else:
            result[f"{label}_revision"] = str(actual)
    if errors:
        raise ValueError("; ".join(errors))
    return result


def _probe_habitat_runtime_identity(
    lab_root: Path | None = None,
    sim_root: Path | None = None,
) -> dict[str, Any]:
    """Import and report the actual Lab/Sim runtime identity.

    Source roots are temporarily prepended so an installed unrelated Habitat
    package cannot satisfy preflight by accident.  The process exits after the
    receipt, so restoring ``sys.path`` is sufficient isolation.
    """

    inserted: list[str] = []
    for root in (lab_root, sim_root):
        if root is None:
            continue
        for candidate in (root, root / "habitat-lab", root / "habitat-sim"):
            if candidate.is_dir() and str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
                inserted.append(str(candidate))
    importlib.invalidate_caches()
    try:
        habitat = importlib.import_module("habitat")
        habitat_sim = importlib.import_module("habitat_sim")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Habitat-Lab/SIM 0.2.4 runtime is not importable") from exc
    lab_version = getattr(habitat, "__version__", None)
    sim_version = getattr(habitat_sim, "__version__", None)
    if sim_version is None:
        try:
            from importlib import metadata as importlib_metadata

            sim_version = importlib_metadata.version("habitat-sim")
        except Exception:
            sim_version = None
    if lab_version is None or sim_version is None:
        raise RuntimeError(
            f"Habitat runtime identity is incomplete (lab={lab_version!r}, sim={sim_version!r})"
        )
    if str(lab_version) != "0.2.4" or str(sim_version) != "0.2.4":
        raise RuntimeError(
            f"Habitat runtime must be Lab/Sim 0.2.4 (lab={lab_version!r}, sim={sim_version!r})"
        )
    return {
        "habitat_lab_version": str(lab_version),
        "habitat_sim_version": str(sim_version),
        "habitat_file": str(getattr(habitat, "__file__", "")),
        "habitat_sim_file": str(getattr(habitat_sim, "__file__", "")),
    }


def _scene_ids_in_tree(root: Path) -> set[str]:
    found: set[str] = set()
    if not root.is_dir():
        return found
    # Do not require a particular extraction layout: official archives have
    # changed one intermediate directory name across Habitat releases.
    wanted = set(EXPECTED_SCENE_IDS)
    for directory, dirs, files in __import__("os").walk(root):
        names = [*dirs, *files]
        for name in names:
            for scene in wanted:
                if scene in name:
                    found.add(scene)
        if found == wanted:
            break
    return found


def _scene_ids_in_zip(path: Path) -> set[str]:
    found: set[str] = set()
    wanted = set(EXPECTED_SCENE_IDS)
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            for scene in wanted:
                if scene in member:
                    found.add(scene)
            if found == wanted:
                break
    return found


def _check_checkpoint(path: Path | None, expected_sha256: str | None = None) -> dict[str, Any]:
    if path is None:
        return {"status": "NOT_SUPPLIED"}
    if not path.is_file():
        return {"status": "MISSING", "path": str(path)}
    if expected_sha256 is None or str(expected_sha256).strip() == "":
        return {
            "status": "IDENTITY_REQUIRED",
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    actual_sha = sha256_file(path)
    if str(expected_sha256).lower() != actual_sha:
        return {
            "status": "SHA_MISMATCH",
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "expected_sha256": str(expected_sha256),
        }
    return {
        "status": "PRESENT",
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual_sha,
        "expected_sha256": str(expected_sha256),
    }


def run_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    checks: dict[str, Any] = {}
    errors: list[str] = []
    config_path = Path(args.config).expanduser().resolve()
    resolved_config_sha256: str | None = None
    try:
        config = _load_yaml(config_path)
        resolved_config_sha256 = _resolved_config_sha256(config)
        config.setdefault("assets", {})
        assets = config["assets"]
        if not isinstance(assets, dict):
            raise ValueError("config.assets must be a mapping")
        for key, cli in (
            ("mp3d_archive", args.mp3d_archive),
            ("scene_root", args.scene_root),
            ("episodes", args.episodes),
            ("habitat_lab_root", args.habitat_lab_root),
            ("habitat_sim_root", args.habitat_sim_root),
        ):
            if cli:
                assets[key] = str(Path(cli).expanduser().resolve())
        for key, cli in (
            ("habitat_lab_revision", getattr(args, "habitat_lab_revision", None)),
            ("habitat_sim_revision", getattr(args, "habitat_sim_revision", None)),
        ):
            if cli:
                assets[key] = str(cli)
        validate_imagegoal_config(config, require_assets=True)
        checks["contract"] = {"status": "PASS"}
    except Exception as exc:
        checks["contract"] = {"status": "FAIL", "error": str(exc)}
        errors.append(f"contract: {exc}")
        config = {}
        assets = {}

    archive_value = assets.get("mp3d_archive") if isinstance(assets, dict) else None
    archive = Path(archive_value).expanduser() if archive_value else None
    archive_check: dict[str, Any] = {"status": "MISSING"}
    if archive is None or not archive.is_file():
        errors.append("mp3d archive is missing")
        if archive is not None:
            archive_check["path"] = str(archive)
    elif archive.name.endswith(".part"):
        errors.append("mp3d archive is an incomplete .part file")
        archive_check = {"status": "INCOMPLETE", "path": str(archive)}
    else:
        actual_sha = sha256_file(archive)
        archive_check = {
            "status": "PASS",
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": actual_sha,
        }
        expected_bytes = assets.get("mp3d_bytes")
        expected_sha = assets.get("mp3d_sha256")
        if expected_bytes is not None and int(expected_bytes) != archive.stat().st_size:
            errors.append("mp3d archive byte count does not match config")
        if expected_sha and str(expected_sha) != actual_sha:
            errors.append("mp3d archive SHA-256 does not match config")
        try:
            with zipfile.ZipFile(archive) as handle:
                bad = handle.testzip()
                archive_check["zip_test"] = "PASS" if bad is None else f"BAD_MEMBER:{bad}"
                archive_check["scene_ids"] = sorted(_scene_ids_in_zip(archive))
                if bad is not None:
                    errors.append(f"mp3d ZIP has corrupt member {bad}")
                if set(archive_check["scene_ids"]) != set(EXPECTED_SCENE_IDS):
                    errors.append("mp3d archive does not contain the frozen 11 scenes")
                try:
                    archive_check["scene_members"] = validate_scene_archive(archive)["members"]
                except ValueError as exc:
                    errors.append(f"mp3d archive scene members: {exc}")
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"mp3d ZIP integrity check failed: {exc}")
    checks["mp3d_archive"] = archive_check

    scene_value = assets.get("scene_root") if isinstance(assets, dict) else None
    scene_root = Path(scene_value).expanduser() if scene_value else None
    if scene_root is None or not scene_root.is_dir():
        errors.append("extracted MP3D scene_root is missing")
        checks["scene_root"] = {"status": "MISSING", "path": str(scene_root) if scene_root else None}
    else:
        try:
            scene_check = validate_scene_root(scene_root)
            checks["scene_root"] = {
                "status": "PASS",
                "path": str(scene_root),
                "scene_ids": scene_check["scene_ids"],
                "scene_members": scene_check["members"],
                "fingerprint": _directory_fingerprint(scene_root),
            }
        except ValueError as exc:
            checks["scene_root"] = {"status": "FAIL", "path": str(scene_root), "error": str(exc)}
            errors.append(f"scene_root: {exc}")

    episodes_value = assets.get("episodes") if isinstance(assets, dict) else None
    episodes_path = Path(episodes_value).expanduser() if episodes_value else None
    if episodes_path is None or not episodes_path.is_file():
        errors.append("official val episode ledger is missing")
        checks["episodes"] = {"status": "MISSING", "path": str(episodes_path) if episodes_path else None}
    else:
        try:
            episodes, actual_sha = _load_episodes(episodes_path)
            ledger_check = validate_episode_ledger(episodes, expected_count=int(config.get("episode_count", 495)))
            checks["episodes"] = {
                "status": "PASS",
                "path": str(episodes_path),
                "count": len(episodes),
                "sha256": actual_sha,
                "scene_ids": ledger_check["scene_ids"],
                "scene_counts": ledger_check["scene_counts"],
                "episode_keys": ledger_check["episode_keys"],
            }
            expected_episode_sha = assets.get("episodes_sha256")
            if expected_episode_sha and str(expected_episode_sha) != actual_sha:
                errors.append("episode ledger SHA-256 does not match config")
        except Exception as exc:
            checks["episodes"] = {"status": "FAIL", "error": str(exc)}
            errors.append(f"episodes: {exc}")

    for label, key in (("habitat_lab", "habitat_lab_root"), ("habitat_sim", "habitat_sim_root")):
        value = assets.get(key) if isinstance(assets, dict) else None
        root = Path(value).expanduser() if value else None
        head = _git_head(root) if root is not None else None
        checks[label] = {
            "status": "PASS" if root is not None and root.is_dir() else "MISSING",
            "path": str(root) if root else None,
            "git_head": head,
        }
        if root is None or not root.is_dir():
            errors.append(f"{label} root is missing")

    # A directory and a recorded git SHA are not enough: require the exact
    # pair selected by the resolved config, then probe the actual imported
    # Lab/Sim runtime identity.  This check intentionally fails on machines
    # without Habitat rather than allowing a metadata-only PASS.
    try:
        revision_check = validate_habitat_revisions(
            checks.get("habitat_lab", {}).get("git_head"),
            checks.get("habitat_sim", {}).get("git_head"),
            assets.get("habitat_lab_revision") if isinstance(assets, dict) else None,
            assets.get("habitat_sim_revision") if isinstance(assets, dict) else None,
        )
        checks["habitat_revisions"] = {"status": "PASS", **revision_check}
    except Exception as exc:
        checks["habitat_revisions"] = {"status": "FAIL", "error": str(exc)}
        errors.append(f"habitat revisions: {exc}")
    try:
        runtime_check = _probe_habitat_runtime_identity(
            Path(assets["habitat_lab_root"]).expanduser()
            if isinstance(assets, dict) and assets.get("habitat_lab_root")
            else None,
            Path(assets["habitat_sim_root"]).expanduser()
            if isinstance(assets, dict) and assets.get("habitat_sim_root")
            else None,
        )
        checks["habitat_runtime"] = {"status": "PASS", **runtime_check}
    except Exception as exc:
        checks["habitat_runtime"] = {"status": "FAIL", "error": str(exc)}
        errors.append(f"habitat runtime: {exc}")

    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else None
    expected_checkpoint_sha = getattr(args, "checkpoint_sha256", None)
    if expected_checkpoint_sha is None and isinstance(config, dict):
        expected_checkpoint_sha = (
            config.get("v9", {}).get("checkpoint_sha256")
            if isinstance(config.get("v9"), dict)
            else config.get("assets", {}).get("checkpoint_sha256")
            if isinstance(config.get("assets"), dict)
            else None
        )
    checks["checkpoint"] = _check_checkpoint(checkpoint, expected_checkpoint_sha)
    if checkpoint is not None and checks["checkpoint"]["status"] != "PRESENT":
        errors.append("requested checkpoint identity/hash is missing or mismatched")

    receipt = {
        "schema": "j2j.imagegoal.preflight.v1",
        "status": "PASS" if not errors else "FAIL",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "resolved_config_sha256": resolved_config_sha256,
        "checks": checks,
        "errors": errors,
        "training_boundary": "MP3D evaluation-only; no training/cache writes",
    }
    return receipt, (0 if not errors else 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mp3d-archive", required=True)
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--habitat-lab-root", required=True)
    parser.add_argument("--habitat-sim-root", required=True)
    parser.add_argument("--habitat-lab-revision")
    parser.add_argument("--habitat-sim-revision")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt, code = run_preflight(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
