"""Current-only Habitat reset/replay evidence producers.

The injectable entry point is deliberately mechanics-only. Formal evidence
has a separate schema and constructs the official Habitat 0.2.4 environment
through the same loader used by the mature ImageGoal evaluator.
"""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import inspect
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import yaml

from j2j.authority import sha256_file
from j2j.evaluation.contracts import canonical_episode_key, validate_imagegoal_config
from j2j.evaluation.habitat_runner import (
    load_habitat_dataset,
    load_habitat_environment,
)
from j2j.receipts import canonical_json_bytes


MECHANICS_SCHEMA = "J2J_HABITAT_RESET_REPLAY_MECHANICS_V1"
FORMAL_CAPABILITY_SCHEMA = "J2J_HABITAT_RESET_REPLAY_CAPABILITY_V3"
HABITAT_LAB_COMMIT = "1639e1ae732ba1e84199a1a04b79c7243c3f8586"
HABITAT_SIM_COMMIT = "f179b584bcd713c5a2a998132211e2cae881d6d1"
_OFFICIAL_ORIGINS = {
    "habitat_lab": "facebookresearch/habitat-lab",
    "habitat_sim": "facebookresearch/habitat-sim",
}
_MOTION_ACTIONS = frozenset({"FWD", "LEFT", "RIGHT"})
_ACTION_IDS = {"FWD": 1, "LEFT": 2, "RIGHT": 3}
_STATE_INPUT_FIELDS = {"rgb_bytes", "pose_bytes", "context_bytes"}
_SHA256 = re.compile(r"[0-9a-f]{64}")
HABITAT_SCIENTIFIC_PROJECTION_SCHEMA = "J2J_HABITAT_SCIENTIFIC_PROJECTION_V1"
HABITAT_SCIENTIFIC_FIELDS = (
    "habitat_version",
    "task",
    "split",
    "episode_count",
    "scene_ids",
    "agent",
    "actions",
    "rgb",
    "policy_observation_keys",
    "privileged_observation_keys",
    "max_episode_steps",
    "success_distance",
    "assets",
)


def _plain_json_value(value: object, *, name: str) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return {
            key: _plain_json_value(item, name=f"{name}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _plain_json_value(item, name=f"{name}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError(f"{name} contains a non-JSON value")


def habitat_scientific_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact current Habitat science-only config projection."""

    if not isinstance(config, Mapping):
        raise TypeError("formal Habitat evaluation config must be a mapping")
    missing = [field for field in HABITAT_SCIENTIFIC_FIELDS if field not in config]
    if missing:
        raise ValueError(
            "formal Habitat scientific projection is missing: " + ", ".join(missing)
        )
    projection = {
        field: _plain_json_value(config[field], name=field)
        for field in HABITAT_SCIENTIFIC_FIELDS
    }
    # Serialization is also the finite-number gate for every nested value.
    json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return projection


def habitat_scientific_projection_identity(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    projection = habitat_scientific_projection(config)
    encoded = json.dumps(
        projection,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": HABITAT_SCIENTIFIC_PROJECTION_SCHEMA,
        "projection": projection,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _habitat_scientific_identity(
    config: Mapping[str, Any],
    *,
    sensor_config_path: str | os.PathLike[str],
) -> dict[str, Any]:
    return {
        "evaluation_config": habitat_scientific_projection_identity(config),
        "sensor_config": _file_identity(sensor_config_path, name="sensor config"),
    }


def _validate_official_github_origin(
    origin: str,
    *,
    expected_repository: str,
) -> str:
    """Admit a credential-free canonical GitHub repository URL."""

    if not isinstance(origin, str) or not origin or origin != origin.strip():
        raise ValueError("Habitat official source origin is not canonical")
    expected_path = "/" + expected_repository.lower().strip("/")
    if expected_path.endswith(".git"):
        expected_path = expected_path[:-4]

    if origin.startswith("git@github.com:"):
        repository_path = "/" + origin[len("git@github.com:") :].rstrip("/")
        if any(marker in repository_path for marker in ("?", "#")):
            raise ValueError("Habitat official source origin is not canonical")
        if repository_path.lower().endswith(".git"):
            repository_path = repository_path[:-4]
        if repository_path.lower() != expected_path:
            raise ValueError("Habitat source origin is not the official repository")
        return origin

    parsed = urlparse(origin)
    if parsed.scheme.lower() not in {"https", "git+https", "ssh"}:
        raise ValueError("Habitat official source origin scheme is not canonical")
    try:
        username = parsed.username
        password = parsed.password
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Habitat official source origin is not canonical") from exc
    if password is not None or parsed.query or parsed.fragment or parsed.params:
        raise ValueError("Habitat official source origin is not canonical")
    if port is not None or hostname is None or hostname.lower() != "github.com":
        raise ValueError("Habitat source origin is not the official repository")
    if parsed.scheme.lower() == "ssh":
        if username != "git":
            raise ValueError("Habitat official SSH source origin is not canonical")
    elif username is not None:
        raise ValueError("Habitat official HTTPS source origin is not canonical")
    repository_path = parsed.path.rstrip("/")
    if repository_path.lower().endswith(".git"):
        repository_path = repository_path[:-4]
    if repository_path.lower() != expected_path:
        raise ValueError("Habitat source origin is not the official repository")
    return origin


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            f"Habitat source repository is not a readable Git worktree: {root}"
        ) from exc
    return completed.stdout.strip()


def _repository_identity(
    value: str | os.PathLike[str],
    *,
    name: str,
    expected_commit: str | None = None,
    official_origin_fragment: str | None = None,
) -> dict[str, Any]:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Habitat {name} source repository is unavailable")
    top = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise ValueError(f"Habitat {name} source root is not the Git toplevel")
    commit = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    clean = _git(root, "status", "--porcelain", "--untracked-files=all") == ""
    if not clean:
        raise ValueError(f"Habitat {name} source repository is not clean")
    if expected_commit is not None and commit != expected_commit:
        raise ValueError(
            f"Habitat {name} source revision is not the admitted v0.2.4 commit"
        )
    if official_origin_fragment is None:
        try:
            origin = _git(root, "config", "--get", "remote.origin.url")
        except ValueError:
            origin = ""
    else:
        origin = _git(root, "config", "--get", "remote.origin.url")
        try:
            origin = _validate_official_github_origin(
                origin,
                expected_repository=official_origin_fragment,
            )
        except ValueError as exc:
            raise ValueError(
                f"Habitat {name} source origin is not the official repository"
            ) from exc
    return {
        "root": str(root),
        "origin": origin,
        "commit": commit,
        "tree": tree,
        "clean": True,
    }


def _file_identity(value: str | os.PathLike[str], *, name: str) -> dict[str, Any]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Habitat capability {name} file is unavailable")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _module_file(module: object, *, name: str) -> Path:
    value = getattr(module, "__file__", None)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Habitat {name} live module has no source/install file")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Habitat {name} live module file is unavailable")
    return path


def _distribution_metadata_files(distribution_name: str) -> list[dict[str, Any]]:
    try:
        distribution = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError as exc:
        raise ValueError(
            f"Habitat {distribution_name} distribution metadata is unavailable"
        ) from exc
    result: list[dict[str, Any]] = []
    for relative in distribution.files or ():
        name = Path(str(relative)).name
        if name not in {"METADATA", "PKG-INFO", "direct_url.json"}:
            continue
        candidate = Path(distribution.locate_file(relative)).resolve()
        if candidate.is_file():
            result.append(
                _file_identity(candidate, name=f"{distribution_name} metadata")
            )
    if not result:
        raise ValueError(
            f"Habitat {distribution_name} has no bindable package metadata"
        )
    return sorted(result, key=lambda item: str(item["path"]))


def _habitat_sim_build_source(
    metadata_files: Sequence[Mapping[str, Any]],
    *,
    habitat_sim_source_root: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    direct_identities = [
        dict(identity)
        for identity in metadata_files
        if Path(str(identity.get("path", ""))).name == "direct_url.json"
    ]
    if len(direct_identities) != 1:
        raise ValueError(
            "Habitat-Sim installed binary needs exactly one PEP 610 direct URL build source"
        )
    direct_identity = direct_identities[0]
    direct_path = Path(str(direct_identity["path"])).resolve()
    try:
        payload = json.loads(direct_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Habitat-Sim direct URL build provenance is invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Habitat-Sim direct URL build provenance must be a mapping")
    url = payload.get("url")
    vcs = payload.get("vcs_info")
    if (
        not isinstance(url, str)
        or not url
        or url != url.strip()
        or not isinstance(vcs, Mapping)
    ):
        raise ValueError("Habitat-Sim direct URL has no VCS source binding")
    if vcs.get("vcs") != "git":
        raise ValueError("Habitat-Sim build source must use Git provenance")
    if vcs.get("commit_id") != HABITAT_SIM_COMMIT:
        raise ValueError("Habitat-Sim build source commit is not the admitted revision")
    requested_revision = vcs.get("requested_revision")
    if requested_revision is not None and (
        not isinstance(requested_revision, str)
        or re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", requested_revision) is None
    ):
        raise ValueError("Habitat-Sim requested source revision is invalid")

    parsed = urlparse(url)
    if parsed.scheme == "file":
        try:
            username = parsed.username
            password = parsed.password
        except ValueError as exc:
            raise ValueError("Habitat-Sim local build source URL is not canonical") from exc
        if (
            username is not None
            or password is not None
            or parsed.query
            or parsed.fragment
            or parsed.params
            or parsed.netloc not in {"", "localhost"}
        ):
            raise ValueError("Habitat-Sim local build source URL is not canonical")
        if habitat_sim_source_root is None:
            raise ValueError(
                "Habitat-Sim local build source has no admitted source-root binding"
            )
        source_path = Path(unquote(parsed.path)).resolve()
        admitted_source = Path(habitat_sim_source_root).expanduser().resolve()
        if source_path != admitted_source:
            raise ValueError(
                "Habitat-Sim local build source differs from the admitted source root"
            )
    else:
        try:
            _validate_official_github_origin(
                url,
                expected_repository="facebookresearch/habitat-sim",
            )
        except ValueError as exc:
            raise ValueError(
                "Habitat-Sim build source URL is not the official repository"
            ) from exc
    directory = payload.get("dir_info")
    if isinstance(directory, Mapping) and directory.get("editable") is True:
        raise ValueError("Habitat-Sim editable installs are not formal build evidence")
    return {
        "kind": "pep610_vcs",
        "vcs": "git",
        "url": url,
        "requested_revision": requested_revision,
        "commit_id": HABITAT_SIM_COMMIT,
        "direct_url": direct_identity,
    }


def habitat_runtime_identity(
    *,
    habitat_lab_source_root: str | os.PathLike[str],
    habitat_sim_install_root: str | os.PathLike[str],
    habitat_sim_source_root: str | os.PathLike[str] | None = None,
    frozen_seed: int,
) -> dict[str, Any]:
    """Derive the live runtime identity; no version value is caller supplied."""

    if type(frozen_seed) is not int or frozen_seed < 0:
        raise ValueError("formal Habitat frozen_seed must be a non-negative integer")
    hash_seed = os.environ.get("PYTHONHASHSEED")
    if hash_seed != str(frozen_seed):
        raise ValueError(
            "formal Habitat capability requires PYTHONHASHSEED to equal frozen_seed"
        )
    habitat = importlib.import_module("habitat")
    habitat_sim = importlib.import_module("habitat_sim")
    lab_version = getattr(habitat, "__version__", None)
    sim_version = getattr(habitat_sim, "__version__", None)
    if sim_version is None:
        try:
            sim_version = importlib_metadata.version("habitat-sim")
        except importlib_metadata.PackageNotFoundError:
            sim_version = None
    if str(lab_version) != "0.2.4" or str(sim_version) != "0.2.4":
        raise ValueError("formal Habitat runtime versions must both be exactly 0.2.4")

    lab_root = Path(habitat_lab_source_root).expanduser().resolve()
    sim_install = Path(habitat_sim_install_root).expanduser().resolve()
    if not sim_install.is_dir():
        raise ValueError("Habitat-Sim admitted install root is unavailable")
    lab_module = _module_file(habitat, name="Lab")
    env_class = getattr(habitat, "Env", None)
    env_source_value = inspect.getsourcefile(env_class) if env_class is not None else None
    if not isinstance(env_source_value, str):
        raise ValueError("Habitat runtime does not expose source-backed habitat.Env")
    env_source = Path(env_source_value).resolve()
    if not _is_within(lab_module, lab_root) or not _is_within(env_source, lab_root):
        raise ValueError("live Habitat-Lab module/Env source is outside admitted checkout")

    sim_module = _module_file(habitat_sim, name="Sim")
    if not _is_within(sim_module, sim_install):
        raise ValueError("live Habitat-Sim module is outside admitted install root")
    native_files = sorted(
        path.resolve() for path in sim_module.parent.rglob("*.so") if path.is_file()
    )
    if not native_files or any(not _is_within(path, sim_install) for path in native_files):
        raise ValueError("Habitat-Sim native extension identity is unavailable")
    conda_records = list((sim_install / "conda-meta").glob("habitat-sim-0.2.4-*.json"))
    if conda_records:
        from j2j_recurrent_experiments.common.conda_provenance import verify_habitat_conda_install
        build_source = verify_habitat_conda_install(
            sim_install, module_path=sim_module, native_paths=native_files,
            expected_commit=HABITAT_SIM_COMMIT,
        )
        metadata_files = [build_source["record"]]
    else:
        metadata_files = _distribution_metadata_files("habitat-sim")
        build_source = _habitat_sim_build_source(
            metadata_files,
            habitat_sim_source_root=habitat_sim_source_root,
        )
    if any(
        not _is_within(Path(str(identity["path"])), sim_install)
        for identity in metadata_files
    ):
        raise ValueError(
            "Habitat-Sim distribution metadata is outside admitted install root"
        )

    executable = Path(sys.executable).resolve()
    return {
        "python": {
            "version": platform.python_version(),
            "executable": _file_identity(executable, name="Python executable"),
            "prefix": str(Path(sys.prefix).resolve()),
            "pythonhashseed": hash_seed,
            "frozen_seed": frozen_seed,
        },
        "habitat_lab": {
            "import_name": "habitat",
            "version": str(lab_version),
            "module_file": _file_identity(lab_module, name="Habitat-Lab module"),
            "env_class": {
                "module": str(env_class.__module__),
                "qualname": str(env_class.__qualname__),
                "source_file": _file_identity(env_source, name="habitat.Env source"),
            },
        },
        "habitat_sim": {
            "import_name": "habitat_sim",
            "version": str(sim_version),
            "install_root": str(sim_install),
            "module_file": _file_identity(sim_module, name="Habitat-Sim module"),
            "native_extensions": [
                _file_identity(path, name="Habitat-Sim native extension")
                for path in native_files
            ],
            "distribution_metadata": metadata_files,
            "build_source": build_source,
        },
    }


def _case(value: object) -> tuple[list[str], list[str]]:
    if not isinstance(value, Mapping) or set(value) != {"episode_key", "action_prefix"}:
        raise ValueError("Habitat capability case fields are incomplete")
    episode = value["episode_key"]
    actions = value["action_prefix"]
    if (
        not isinstance(episode, (list, tuple))
        or len(episode) != 2
        or any(not isinstance(item, str) or not item for item in episode)
    ):
        raise ValueError("Habitat capability episode key is invalid")
    if (
        not isinstance(actions, (list, tuple))
        or not actions
        or any(type(action) is not str or action not in _MOTION_ACTIONS for action in actions)
    ):
        raise ValueError("Habitat capability action prefix is invalid")
    return list(episode), list(actions)


def _state(value: object, *, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _STATE_INPUT_FIELDS:
        raise ValueError("Habitat executor state fields are incomplete")
    result: dict[str, Any] = {"index": index}
    for source, destination in (
        ("rgb_bytes", "rgb_sha256"),
        ("pose_bytes", "pose_sha256"),
        ("context_bytes", "context_sha256"),
    ):
        payload = value[source]
        if type(payload) is not bytes:
            raise TypeError(f"Habitat executor {source} must be exact bytes")
        result[destination] = hashlib.sha256(payload).hexdigest()
    return result


def _run_prefix(
    executor: object, episode: Sequence[str], actions: Sequence[str]
) -> list[dict[str, Any]]:
    reset = getattr(executor, "reset", None)
    step = getattr(executor, "step", None)
    if not callable(reset) or not callable(step):
        raise TypeError("Habitat capability executor must expose reset and step")
    states = [_state(reset(list(episode)), index=0)]
    for index, action in enumerate(actions, start=1):
        states.append(_state(step(action), index=index))
    return states


def _measure(
    executor: object, cases: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise ValueError("Habitat capability cases must be a nonempty sequence")
    produced_cases: list[dict[str, Any]] = []
    action_count = 0
    state_pair_count = 0
    failure_count = 0
    for raw_case in cases:
        episode, actions = _case(raw_case)
        reference = _run_prefix(executor, episode, actions)
        replay = _run_prefix(executor, episode, actions)
        if reference != replay:
            failure_count += sum(
                left != right for left, right in zip(reference, replay, strict=True)
            )
        produced_cases.append(
            {
                "episode_key": episode,
                "action_prefix": actions,
                "reference_states": reference,
                "replay_states": replay,
            }
        )
        action_count += len(actions)
        state_pair_count += len(reference)
    return produced_cases, {
        "case_count": len(produced_cases),
        "action_count": action_count,
        "state_pair_count": state_pair_count,
        "failure_count": failure_count,
    }


def _atomic_write_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Habitat capability receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def produce_reset_replay_capability(
    *,
    executor: object,
    cases: Sequence[Mapping[str, Any]],
    habitat_lab_source_root: str | os.PathLike[str],
    habitat_sim_source_root: str | os.PathLike[str],
    sensor_config_path: str | os.PathLike[str],
    evaluation_config_path: str | os.PathLike[str],
    episode_ledger_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Exercise an injected executor, producing non-formal mechanics only."""

    repositories = {
        "habitat_lab": _repository_identity(
            habitat_lab_source_root, name="mechanics Habitat-Lab"
        ),
        "habitat_sim": _repository_identity(
            habitat_sim_source_root, name="mechanics Habitat-Sim"
        ),
    }
    artifacts = {
        "sensor_config": _file_identity(sensor_config_path, name="sensor config"),
        "evaluation_config": _file_identity(
            evaluation_config_path, name="evaluation config"
        ),
        "episode_ledger": _file_identity(
            episode_ledger_path, name="episode ledger"
        ),
        "mechanics_code": _file_identity(Path(__file__), name="mechanics code"),
    }
    produced_cases, census = _measure(executor, cases)
    receipt = {
        "schema": MECHANICS_SCHEMA,
        "status": "MECHANICS_ONLY" if census["failure_count"] == 0 else "MISMATCH",
        "repositories": repositories,
        "artifacts": artifacts,
        "cases": produced_cases,
        "census": census,
    }
    destination = Path(output_path).expanduser().resolve()
    _atomic_write_once(destination, receipt)
    return receipt


def _array_bytes(value: object, *, name: str) -> bytes:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"formal Habitat {name} must be a numeric array")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"formal Habitat {name} must be finite")
    descriptor = canonical_json_bytes(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    )
    return descriptor + array.tobytes(order="C")


def _rotation_array(value: object) -> np.ndarray:
    if hasattr(value, "real") and hasattr(value, "imag"):
        imag = np.asarray(getattr(value, "imag"), dtype=np.float64).reshape(-1)
        return np.concatenate(([float(getattr(value, "real"))], imag))
    fields = [getattr(value, name, None) for name in ("w", "x", "y", "z")]
    if all(item is not None for item in fields):
        return np.asarray(fields, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


class _CanonicalHabitat024ReplayExecutor:
    """Thin fixed-prefix adapter over the mature official Habitat loader."""

    def __init__(self, env: object, *, frozen_seed: int) -> None:
        habitat = importlib.import_module("habitat")
        env_class = getattr(habitat, "Env", None)
        if env_class is None or type(env) is not env_class:
            raise TypeError("formal replay executor requires exact habitat.Env")
        episodes = getattr(env, "episodes", None)
        if not isinstance(episodes, list) or not episodes:
            raise ValueError("formal Habitat environment has no admitted episodes")
        self.env = env
        self.frozen_seed = frozen_seed
        self._episodes: dict[tuple[str, str], object] = {}
        for episode in episodes:
            key = canonical_episode_key(
                getattr(episode, "scene_id", None),
                getattr(episode, "episode_id", getattr(episode, "id", None)),
            )
            if key in self._episodes:
                raise ValueError("formal Habitat episode ledger has duplicate composite keys")
            self._episodes[key] = episode
        self._episode_key: tuple[str, str] | None = None
        self._actions: list[str] = []
        self._observation: Mapping[str, Any] | None = None
        self._rgb_history: list[np.ndarray] = []
        self._goal_rgb: np.ndarray | None = None

    def _snapshot(self) -> dict[str, bytes]:
        observation = self._observation
        if not isinstance(observation, Mapping) or "rgb" not in observation:
            raise ValueError("formal Habitat observation has no RGB sensor value")
        simulator = getattr(self.env, "sim", None)
        state_getter = getattr(simulator, "get_agent_state", None)
        if not callable(state_getter):
            raise ValueError("formal Habitat simulator exposes no agent state")
        agent_state = state_getter()
        rgb_bytes = _array_bytes(observation["rgb"], name="RGB")
        position = np.asarray(
            getattr(agent_state, "position", None), dtype=np.float64
        ).reshape(-1)
        rotation = _rotation_array(getattr(agent_state, "rotation", None))
        if position.size != 3 or rotation.size != 4:
            raise ValueError("formal Habitat agent pose has invalid dimensions")
        pose_bytes = (
            _array_bytes(position, name="position")
            + _array_bytes(rotation, name="rotation")
        )
        context_bytes = canonical_json_bytes(
            {
                "episode_key": list(self._episode_key or ()),
                "step": len(self._actions),
                "action_prefix": list(self._actions),
                "rgb_sha256": hashlib.sha256(rgb_bytes).hexdigest(),
                "pose_sha256": hashlib.sha256(pose_bytes).hexdigest(),
            }
        )
        return {
            "rgb_bytes": rgb_bytes,
            "pose_bytes": pose_bytes,
            "context_bytes": context_bytes,
        }

    def reset(self, episode_key: Sequence[str]) -> dict[str, bytes]:
        key = canonical_episode_key(episode_key[0], episode_key[1])
        try:
            episode = self._episodes[key]
        except KeyError as exc:
            raise ValueError("formal Habitat capability case is absent from ledger") from exc
        seed = getattr(self.env, "seed", None)
        if not callable(seed):
            raise ValueError("formal Habitat environment exposes no deterministic seed")
        seed(self.frozen_seed)
        setattr(self.env, "current_episode", episode)
        observation = self.env.reset()
        live = getattr(self.env, "current_episode", None)
        live_key = canonical_episode_key(
            getattr(live, "scene_id", None),
            getattr(live, "episode_id", getattr(live, "id", None)),
        )
        if live_key != key:
            raise RuntimeError("formal Habitat reset returned the wrong ledger episode")
        if not isinstance(observation, Mapping):
            raise TypeError("formal Habitat reset observation must be a mapping")
        self._episode_key = key
        self._actions = []
        self._observation = observation
        self._rgb_history = [np.array(observation["rgb"], copy=True)]
        goal_value = observation.get("imagegoal", observation.get("goal_rgb"))
        if goal_value is None:
            raise ValueError("formal Habitat reset observation has no ImageGoal RGB")
        self._goal_rgb = np.array(goal_value, copy=True)
        return self._snapshot()

    def step(self, action: str) -> dict[str, bytes]:
        if action not in _ACTION_IDS:
            raise ValueError("formal Habitat replay action is not a motion primitive")
        if bool(getattr(self.env, "episode_over", False)):
            raise RuntimeError("formal Habitat episode ended before replay prefix")
        observation = self.env.step(_ACTION_IDS[action])
        if not isinstance(observation, Mapping):
            raise TypeError("formal Habitat step observation must be a mapping")
        self._actions.append(action)
        self._observation = observation
        self._rgb_history.append(np.array(observation["rgb"], copy=True))
        return self._snapshot()

    def reset_and_replay(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Reset one ledger state and verify every factual identity axis."""

        from j2j.adapter import ActionId, Raw4Adapter
        from j2j_iclr_experiments.online.adapter import _axes_sha256
        from .branch import _validate_state_identity

        if not isinstance(state, Mapping):
            raise TypeError("formal branch state must be a mapping")
        identity = _validate_state_identity(state.get("identity"))
        episode_key = (str(identity["scene"]), str(identity["episode"]))
        snapshots = [self.reset(episode_key)]
        for action in identity["executed_action_prefix"]:
            snapshots.append(self.step(str(action)))
        factual = identity["factual_steps"]
        if len(snapshots) != len(factual):
            raise RuntimeError("formal replay factual prefix length drifted")
        goal_rgb = self._goal_rgb
        if goal_rgb is None or identity["goal_identity"]["rgb_sha256"] != hashlib.sha256(
            _array_bytes(goal_rgb, name="goal RGB")
        ).hexdigest():
            raise RuntimeError("formal reset/replay goal RGB identity mismatch")
        actions = tuple(str(action) for action in identity["executed_action_prefix"])
        for index, (expected, snapshot) in enumerate(
            zip(factual, snapshots, strict=True)
        ):
            expected_pose = _array_bytes(
                np.asarray(expected["position"], dtype=np.float64), name="position"
            ) + _array_bytes(
                np.asarray(expected["rotation"], dtype=np.float64), name="rotation"
            )
            observed = {
                "rgb_sha256": hashlib.sha256(snapshot["rgb_bytes"]).hexdigest(),
                "pose_sha256": hashlib.sha256(snapshot["pose_bytes"]).hexdigest(),
                "context_sha256": hashlib.sha256(snapshot["context_bytes"]).hexdigest(),
            }
            if expected.get("rgb_sha256") != observed["rgb_sha256"]:
                raise RuntimeError("formal reset/replay rgb_sha256 mismatch")
            if hashlib.sha256(expected_pose).hexdigest() != observed["pose_sha256"]:
                raise RuntimeError("formal reset/replay pose identity mismatch")
            if expected.get("context_sha256") != observed["context_sha256"]:
                raise RuntimeError("formal reset/replay context_sha256 mismatch")
            incoming = (
                Raw4Adapter.encode_bos()
                if index == 0
                else Raw4Adapter.encode(ActionId[actions[index - 1]])
            )
            outgoing = (
                None
                if index == len(actions)
                else Raw4Adapter.encode(ActionId[actions[index]])
            )
            if expected.get("action_axes_sha256") != _axes_sha256(
                incoming, outgoing
            ):
                raise RuntimeError("formal reset/replay action axes identity mismatch")
        return copy.deepcopy(dict(identity))

    def factual_inputs(self) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        """Return bounded transient RGB arrays after a verified replay."""

        if not self._rgb_history or self._goal_rgb is None:
            raise RuntimeError("formal replay has no verified factual state")
        return (
            tuple(np.array(value, copy=True) for value in self._rgb_history),
            np.array(self._goal_rgb, copy=True),
        )

    def _distance(self) -> float:
        metrics = getattr(self.env, "get_metrics", None)
        values = metrics() if callable(metrics) else None
        if not isinstance(values, Mapping):
            raise RuntimeError("formal Habitat DistanceToGoal is unavailable")
        distance = values.get("distance_to_goal", values.get("distance"))
        if isinstance(distance, Mapping):
            distance = distance.get("distance", distance.get("value"))
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            raise RuntimeError("formal Habitat DistanceToGoal is unavailable")
        number = float(distance)
        if not np.isfinite(number):
            raise RuntimeError("formal Habitat DistanceToGoal is nonfinite")
        return number

    def execute_prefix(self, actions: Sequence[str]) -> Mapping[str, Any]:
        """Execute one frozen motion prefix from the just-replayed real state."""

        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence) or not actions:
            raise ValueError("formal candidate action prefix must be nonempty")
        start = self._distance()
        for index, action in enumerate(actions):
            if bool(getattr(self.env, "episode_over", False)):
                raise RuntimeError("formal Habitat episode ended before candidate prefix")
            self.step(str(action))
            if index + 1 < len(actions) and bool(getattr(self.env, "episode_over", False)):
                raise RuntimeError("formal Habitat episode ended before candidate prefix")
        end = self._distance()
        if self._observation is None:
            raise RuntimeError("formal candidate produced no endpoint observation")
        rgb = np.array(self._observation["rgb"], copy=True)
        rgb_sha = hashlib.sha256(_array_bytes(rgb, name="RGB")).hexdigest()
        return {
            "start_distance": start,
            "end_distance": end,
            "actual_progress": start - end,
            "endpoint_rgb": rgb,
            "endpoint_rgb_sha256": rgb_sha,
            "endpoint_key": rgb_sha,
        }

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


def build_formal_replay_executor(
    env: object,
    *,
    frozen_seed: int,
    capability_receipt: Mapping[str, Any],
) -> _CanonicalHabitat024ReplayExecutor:
    """Construct the sole formal replay executor after live capability checks."""

    validate_formal_reset_replay_capability(capability_receipt)
    return _CanonicalHabitat024ReplayExecutor(env, frozen_seed=frozen_seed)


def _read_evaluation_config(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError("formal Habitat evaluation config is not valid YAML") from exc
    if not isinstance(value, Mapping):
        raise ValueError("formal Habitat evaluation config root must be a mapping")
    return validate_imagegoal_config(value, require_assets=True)


def _executor_provenance() -> dict[str, Any]:
    import j2j.evaluation.habitat_runner as habitat_runner

    return {
        "module_name": __name__,
        "class_qualname": _CanonicalHabitat024ReplayExecutor.__qualname__,
        "module_file": _file_identity(Path(__file__), name="formal executor module"),
        "constructor_code": _file_identity(
            Path(habitat_runner.__file__), name="mature Habitat constructor code"
        ),
    }


def _verify_file_identity(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ValueError(f"formal Habitat {name} file identity is incomplete")
    observed = _file_identity(str(value.get("path", "")), name=name)
    if observed != dict(value):
        raise ValueError(f"formal Habitat {name} live file identity drifted")
    return observed


def _validate_formal_cases(
    cases: object, census: object
) -> None:
    if not isinstance(cases, list) or not cases:
        raise ValueError("formal Habitat capability has no replay cases")
    action_count = 0
    state_pair_count = 0
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != {
            "episode_key",
            "action_prefix",
            "reference_states",
            "replay_states",
        }:
            raise ValueError("formal Habitat capability case fields are incomplete")
        episode, actions = _case(
            {
                "episode_key": case["episode_key"],
                "action_prefix": case["action_prefix"],
            }
        )
        if list(case["episode_key"]) != episode or list(case["action_prefix"]) != actions:
            raise ValueError("formal Habitat capability case is not canonical")
        reference = case["reference_states"]
        replay = case["replay_states"]
        if (
            not isinstance(reference, list)
            or not isinstance(replay, list)
            or len(reference) != len(actions) + 1
            or len(replay) != len(actions) + 1
        ):
            raise ValueError("formal Habitat capability must contain N+1 states")
        for index, (left, right) in enumerate(zip(reference, replay, strict=True)):
            expected_fields = {
                "index",
                "rgb_sha256",
                "pose_sha256",
                "context_sha256",
            }
            if (
                not isinstance(left, Mapping)
                or not isinstance(right, Mapping)
                or set(left) != expected_fields
                or set(right) != expected_fields
                or left.get("index") != index
                or right.get("index") != index
            ):
                raise ValueError("formal Habitat capability state identity is invalid")
            for field in ("rgb_sha256", "pose_sha256", "context_sha256"):
                if (
                    not isinstance(left.get(field), str)
                    or not isinstance(right.get(field), str)
                    or _SHA256.fullmatch(str(left[field])) is None
                    or _SHA256.fullmatch(str(right[field])) is None
                ):
                    raise ValueError("formal Habitat capability state hash is invalid")
            if dict(left) != dict(right):
                raise ValueError("formal Habitat reset/replay state identity mismatch")
        action_count += len(actions)
        state_pair_count += len(reference)
    expected_census = {
        "case_count": len(cases),
        "action_count": action_count,
        "state_pair_count": state_pair_count,
        "failure_count": 0,
    }
    if not isinstance(census, Mapping) or dict(census) != expected_census:
        raise ValueError("formal Habitat capability census does not match evidence")


def validate_formal_reset_replay_capability(
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Recompute all live V3 provenance before a formal consumer admits it."""

    required = {
        "schema",
        "status",
        "runtime",
        "repositories",
        "artifacts",
        "scientific_identity",
        "executor_provenance",
        "cases",
        "census",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise ValueError("formal Habitat capability fields are incomplete")
    if receipt.get("schema") != FORMAL_CAPABILITY_SCHEMA:
        raise ValueError("retired/nonformal Habitat capability schema is rejected")
    if receipt.get("status") != "PASS":
        raise ValueError("formal Habitat capability did not pass exactly")
    repositories = receipt["repositories"]
    if not isinstance(repositories, Mapping) or set(repositories) != {
        "habitat_lab",
        "habitat_sim",
    }:
        raise ValueError("formal Habitat repository provenance is incomplete")
    if any(
        not isinstance(repositories[name], Mapping)
        for name in ("habitat_lab", "habitat_sim")
    ):
        raise ValueError("formal Habitat repository provenance is incomplete")
    expected_repositories = {
        "habitat_lab": _repository_identity(
            str(repositories["habitat_lab"].get("root", "")),
            name="Habitat-Lab",
            expected_commit=HABITAT_LAB_COMMIT,
            official_origin_fragment=_OFFICIAL_ORIGINS["habitat_lab"],
        ),
        "habitat_sim": _repository_identity(
            str(repositories["habitat_sim"].get("root", "")),
            name="Habitat-Sim",
            expected_commit=HABITAT_SIM_COMMIT,
            official_origin_fragment=_OFFICIAL_ORIGINS["habitat_sim"],
        ),
    }
    if {name: dict(repositories[name]) for name in repositories} != expected_repositories:
        raise ValueError("formal Habitat repository live identity drifted")

    artifacts = receipt["artifacts"]
    required_artifacts = {
        "sensor_config",
        "evaluation_config_source",
        "episode_ledger",
        "producer_code",
        "constructor_code",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_artifacts:
        raise ValueError("formal Habitat capability artifact provenance is incomplete")
    for name in sorted(required_artifacts - {"evaluation_config_source"}):
        _verify_file_identity(artifacts[name], name=name)
    evaluation_source = artifacts["evaluation_config_source"]
    if (
        not isinstance(evaluation_source, Mapping)
        or set(evaluation_source) != {"path"}
        or not isinstance(evaluation_source.get("path"), str)
        or not evaluation_source.get("path")
    ):
        raise ValueError(
            "formal Habitat evaluation config source path is incomplete"
        )
    evaluation_path = Path(str(evaluation_source["path"])).expanduser().resolve()
    if (
        str(evaluation_path) != evaluation_source["path"]
        or not evaluation_path.is_file()
    ):
        raise ValueError("formal Habitat evaluation config source is unavailable")
    current_executor = _executor_provenance()
    if dict(artifacts["producer_code"]) != current_executor["module_file"]:
        raise ValueError("formal Habitat producer code identity drifted")
    if dict(artifacts["constructor_code"]) != current_executor["constructor_code"]:
        raise ValueError("formal Habitat constructor code identity drifted")
    executor = receipt["executor_provenance"]
    if not isinstance(executor, Mapping) or dict(executor) != current_executor:
        raise ValueError("formal Habitat executor provenance is not canonical")

    runtime = receipt["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("formal Habitat runtime provenance is incomplete")
    python = runtime.get("python")
    sim = runtime.get("habitat_sim")
    if not isinstance(python, Mapping) or not isinstance(sim, Mapping):
        raise ValueError("formal Habitat runtime provenance is incomplete")
    frozen_seed = python.get("frozen_seed")
    install_root = sim.get("install_root")
    expected_runtime = habitat_runtime_identity(
        habitat_lab_source_root=expected_repositories["habitat_lab"]["root"],
        habitat_sim_source_root=expected_repositories["habitat_sim"]["root"],
        habitat_sim_install_root=str(install_root),
        frozen_seed=frozen_seed,
    )
    if dict(runtime) != expected_runtime:
        raise ValueError("formal Habitat live runtime/import identity drifted")
    live_evaluation = _read_evaluation_config(str(evaluation_path))
    expected_scientific_identity = _habitat_scientific_identity(
        live_evaluation,
        sensor_config_path=str(artifacts["sensor_config"]["path"]),
    )
    scientific_identity = receipt["scientific_identity"]
    if (
        not isinstance(scientific_identity, Mapping)
        or set(scientific_identity) != {"evaluation_config", "sensor_config"}
        or dict(scientific_identity) != expected_scientific_identity
    ):
        raise ValueError(
            "formal Habitat scientific evaluation config projection or sensor identity drifted"
        )
    _validate_formal_cases(receipt["cases"], receipt["census"])
    return receipt


def produce_formal_reset_replay_capability(
    *,
    cases: Sequence[Mapping[str, Any]],
    habitat_lab_source_root: str | os.PathLike[str],
    habitat_sim_source_root: str | os.PathLike[str],
    habitat_sim_install_root: str | os.PathLike[str] | None = None,
    sensor_config_path: str | os.PathLike[str],
    evaluation_config_path: str | os.PathLike[str],
    episode_ledger_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    frozen_seed: int = 0,
    executor: object | None = None,
) -> dict[str, Any]:
    """Produce V3 only from the official loader; injected executors are rejected."""

    if executor is not None:
        raise ValueError("formal Habitat capability forbids an injected executor")
    if habitat_sim_install_root is None:
        raise ValueError("formal Habitat-Sim admitted install root is required")
    repositories = {
        "habitat_lab": _repository_identity(
            habitat_lab_source_root,
            name="Habitat-Lab",
            expected_commit=HABITAT_LAB_COMMIT,
            official_origin_fragment=_OFFICIAL_ORIGINS["habitat_lab"],
        ),
        "habitat_sim": _repository_identity(
            habitat_sim_source_root,
            name="Habitat-Sim",
            expected_commit=HABITAT_SIM_COMMIT,
            official_origin_fragment=_OFFICIAL_ORIGINS["habitat_sim"],
        ),
    }
    runtime = habitat_runtime_identity(
        habitat_lab_source_root=habitat_lab_source_root,
        habitat_sim_source_root=habitat_sim_source_root,
        habitat_sim_install_root=habitat_sim_install_root,
        frozen_seed=frozen_seed,
    )
    evaluation = _read_evaluation_config(evaluation_config_path)
    assets = evaluation.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("formal Habitat evaluation config has no asset mapping")
    ledger_path = Path(episode_ledger_path).expanduser().resolve()
    configured_ledger = Path(str(assets.get("episodes", ""))).expanduser().resolve()
    if ledger_path != configured_ledger:
        raise ValueError("formal Habitat episode ledger differs from evaluation config")
    dataset = load_habitat_dataset(
        str(ledger_path), scenes_dir=str(assets.get("scene_root"))
    )
    env = load_habitat_environment(
        config_path=str(Path(sensor_config_path).expanduser().resolve()),
        dataset=dataset,
        evaluation_contract=evaluation,
    )
    replay = _CanonicalHabitat024ReplayExecutor(env, frozen_seed=frozen_seed)
    try:
        produced_cases, census = _measure(replay, cases)
    finally:
        replay.close()
    executor_provenance = _executor_provenance()
    artifacts = {
        "sensor_config": _file_identity(sensor_config_path, name="sensor config"),
        "evaluation_config_source": {
            "path": str(Path(evaluation_config_path).expanduser().resolve())
        },
        "episode_ledger": _file_identity(ledger_path, name="episode ledger"),
        "producer_code": _file_identity(Path(__file__), name="producer code"),
        "constructor_code": executor_provenance["constructor_code"],
    }
    scientific_identity = _habitat_scientific_identity(
        evaluation,
        sensor_config_path=sensor_config_path,
    )
    receipt = {
        "schema": FORMAL_CAPABILITY_SCHEMA,
        "status": "PASS" if census["failure_count"] == 0 else "BLOCKED",
        "runtime": runtime,
        "repositories": repositories,
        "artifacts": artifacts,
        "scientific_identity": scientific_identity,
        "executor_provenance": executor_provenance,
        "cases": produced_cases,
        "census": census,
    }
    validate_formal_reset_replay_capability(receipt)
    destination = Path(output_path).expanduser().resolve()
    _atomic_write_once(destination, receipt)
    return receipt


__all__ = [
    "FORMAL_CAPABILITY_SCHEMA",
    "HABITAT_SCIENTIFIC_FIELDS",
    "HABITAT_SCIENTIFIC_PROJECTION_SCHEMA",
    "HABITAT_LAB_COMMIT",
    "HABITAT_SIM_COMMIT",
    "MECHANICS_SCHEMA",
    "build_formal_replay_executor",
    "habitat_scientific_projection",
    "habitat_scientific_projection_identity",
    "habitat_runtime_identity",
    "produce_formal_reset_replay_capability",
    "produce_reset_replay_capability",
    "validate_formal_reset_replay_capability",
]
