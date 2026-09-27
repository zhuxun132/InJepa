"""Runtime-safe launcher helpers for the unchanged official RAE ``train.py``."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

from .accumulation import make_accumulation_plan, require_positive_int
from .config_guard import (
    PAPER_PROFILE_NAME,
    ACCUMULATION_SOURCE_VARIANT,
    training_profile_name,
    OFFICIAL_COMMIT,
    OFFICIAL_CHECKOUT_PROFILE,
    OFFICIAL_BASELINE_RESTRICTED_SHA256,
    OFFICIAL_RESTRICTED_SHA256,
    ALLOWED_MODIFIED_CONFIG_SHA256,
    OFFICIAL_BASE_CONFIG_SHA256,
    STREAMVLN_REVISION,
    assert_no_fixed_runtime_paths,
    assert_fresh_training_output,
    assert_data_overlay,
    assert_runtime_paper_config,
    assert_training_profile,
    canonical_digest,
    converter_bundle_sha256,
    converter_closure_digest_map,
    load_yaml_mapping,
    paper_batch_mapping,
    runtime_paper_config,
    sha256_file,
    verify_converter_source,
    verify_upstream_source,
)
from .assets import configure_offline_hf_environment, verify_rae_assets
from .authority import verify_code_authority


OFFICIAL_TRAIN_CLI_DEFAULTS: dict[str, int] = {
    # These values mirror the official README reproduction command.  The
    # paper appendix supplies the 50-epoch budget; the README supplies the
    # explicit seed/checkpoint/evaluation/precision flags.
    "--epochs": 50,
    "--global-seed": 42,
    "--log-every": 100,
    "--ckpt-every": 5000,
    "--eval-every": 1000,
    "--bfloat16": 1,
    "--torch-compile": 1,
}
SAFE_CHILD_ENV_OVERRIDES = frozenset(
    {
        "WANDB_MODE",
        "WANDB_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    }
)
IMPORT_SHADOW_ENV_KEYS = frozenset(
    {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "LD_PRELOAD"}
)
# The B1 launcher owns one official ``train.py`` job.  For world size > 1 it
# invokes the upstream PyTorch torchrun module and supplies a generated YAML
# whose per-rank batch is exactly 96/world.  Inherited rendezvous variables are
# scrubbed so the child cannot join an unrelated job; torchrun adds its own
# rank variables after spawn.
DISTRIBUTED_ENV_KEYS = frozenset(
    {
        "RANK",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_MAX_RESTARTS",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
    }
)
CANONICAL_RESERVATION_FILENAME = ".rae_stream_gpu.lock"
ACTIVE_RESERVATION_FILENAME = ".rae_stream_gpu.active.json"
SYSTEM_NVIDIA_SMI_DIRS = tuple(
    Path(path)
    for path in ("/usr/bin", "/usr/local/bin", "/bin", "/sbin", "/usr/sbin")
)
# Source bytes that define the already-completed manifest validation pass.
# The pinned receipt may predate the accumulation-only train.py adapter, but
# these validator/data semantics must remain byte-identical when the receipt
# is reused at formal launch.
VALIDATION_RECEIPT_AUTHORITY_FILES = (
    "scripts/validate_rae_stream.py",
    "rae_stream/config_guard.py",
    "rae_stream/annotations.py",
    "rae_stream/authority.py",
    "rae_stream/geometry.py",
    "rae_stream/manifest.py",
    "rae_stream/materialize.py",
)

# The production validation receipt was generated before the reviewed
# accumulation adapter was added.  ``config_guard.py`` consequently has one
# deliberate, source-reviewed runtime-profile transition alongside the
# train.py transition.  Keep this allow-list exact: no other validator/data
# closure byte may drift while reusing the receipt without another RGB pass.
VALIDATION_RECEIPT_CLOSURE_TRANSITIONS = {
    "rae_stream/config_guard.py": {
        "from_sha256": "fdad9cabf0afca5f80ab914b8a3820b7593671051a664253427a7915e9058695",
        "to_sha256": "fa085002dfac2b25b6d8ba3f5eb43a3ca15b346cf32573948cd5f4a2be07440c",
        "reason": "accumulation_runtime_profile_guard",
    },
}


def verify_validation_receipt_closure(
    recorded_files: Mapping[str, Any],
    current_files: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Compare the source closure behind a pinned validation receipt.

    Most closure files must be byte-identical.  The sole exception is the
    reviewed ``config_guard.py`` runtime-profile extension, whose exact old
    and new hashes are pinned above.  Returning the transition map makes the
    exception auditable in the formal launch receipt instead of silently
    treating two source identities as equal.
    """

    if not isinstance(recorded_files, Mapping) or not isinstance(current_files, Mapping):
        raise ValueError("pinned validation receipt code-authority file maps are malformed")
    transitions: dict[str, dict[str, str]] = {}
    for relative in VALIDATION_RECEIPT_AUTHORITY_FILES:
        recorded = recorded_files.get(relative)
        current = current_files.get(relative)
        if recorded == current:
            if recorded is None:
                raise ValueError(f"pinned validation receipt validation-closure {relative} is missing")
            continue
        allowed = VALIDATION_RECEIPT_CLOSURE_TRANSITIONS.get(relative)
        if (
            allowed is None
            or recorded != allowed["from_sha256"]
            or current != allowed["to_sha256"]
        ):
            raise ValueError(
                f"pinned validation receipt validation-closure {relative} differs from current bytes"
            )
        transitions[relative] = dict(allowed)
    return transitions


def assert_formal_revision(value: str | None) -> str:
    """Return the one admitted StreamVLN revision or reject an override."""

    revision = STREAMVLN_REVISION if value is None else str(value)
    if revision != STREAMVLN_REVISION:
        raise ValueError(
            f"formal RAE-stream requires fixed StreamVLN revision {STREAMVLN_REVISION}; got {revision}"
        )
    return revision


def _validate_official_train_args(extra_args: Sequence[str] | None) -> list[str]:
    """Accept only explicit CLI values equal to upstream checkout defaults."""

    if isinstance(extra_args, (str, bytes)):
        raise TypeError("extra_args must be a sequence of argv tokens")
    tokens = [str(token) for token in (extra_args or ())]
    normalized: dict[str, int] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected official train argv token {token!r}")
        if "=" in token:
            option, value_text = token.split("=", 1)
            consumed = 1
        else:
            option = token
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"protected official train option {option!r} requires its fixed value")
            value_text = tokens[index + 1]
            consumed = 2
        if option == "--config" or option not in OFFICIAL_TRAIN_CLI_DEFAULTS:
            raise ValueError(f"protected or unsupported official train option {option!r}")
        if option in normalized:
            raise ValueError(f"duplicate protected official train option {option!r}")
        try:
            value = int(value_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"protected official train option {option!r} must be an integer") from exc
        expected = OFFICIAL_TRAIN_CLI_DEFAULTS[option]
        if value != expected:
            raise ValueError(
                f"protected official train option {option!r}={value} changes paper default {expected}"
            )
        normalized[option] = value
        index += consumed
    return tokens


def build_child_environment(
    overrides: Mapping[str, str] | None,
    *,
    gpu_index: int | None = None,
    gpu_indices: Sequence[int] | None = None,
    hf_home: str | Path | None = None,
) -> dict[str, str]:
    """Build a child environment without import-path/code injection routes.

    ``gpu_indices`` contains physical NVIDIA indices in the order assigned to
    torchrun local ranks.  ``CUDA_VISIBLE_DEVICES`` remaps them to logical
    ``cuda:0..N-1`` as required by the unchanged upstream ``distributed.py``.
    The singular ``gpu_index`` argument remains a compatibility shorthand for
    a one-process launch.
    """

    if gpu_indices is not None and gpu_index is not None:
        raise ValueError("provide gpu_indices or gpu_index, not both")
    if gpu_indices is None:
        if gpu_index is None:
            raise ValueError("one GPU index or a non-empty GPU index list is required")
        gpu_indices = [gpu_index]
    try:
        normalized_indices = [int(index) for index in gpu_indices]
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU indices must be integers") from exc
    if not normalized_indices or any(index < 0 for index in normalized_indices):
        raise ValueError("GPU indices must be a non-empty list of nonnegative integers")
    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("GPU indices must be unique")

    environment = os.environ.copy()
    for key in IMPORT_SHADOW_ENV_KEYS:
        environment.pop(key, None)
    for key in DISTRIBUTED_ENV_KEYS:
        environment.pop(key, None)
    if overrides is not None:
        invalid = sorted(str(key) for key in overrides if str(key) not in SAFE_CHILD_ENV_OVERRIDES)
        if invalid:
            raise ValueError("unsafe child environment override(s): " + ", ".join(invalid))
        environment.update({str(key): str(value) for key, value in overrides.items()})
    if hf_home is not None:
        home = Path(hf_home).resolve()
        hub = (home / "hub").resolve()
        # Do not let an inherited/overridden cache take precedence over the
        # hash-verified snapshot.  Explicitly supplied values are accepted
        # only when they resolve to this same dedicated root.
        expected = {
            "HF_HOME": str(home),
            "HF_HUB_CACHE": str(hub),
            "TRANSFORMERS_CACHE": str(hub),
        }
        for key, value in expected.items():
            if overrides is not None and key in overrides and Path(str(overrides[key])).resolve() != Path(value):
                raise ValueError(f"{key} override does not match the dedicated verified HF cache")
            environment[key] = value
        environment = configure_offline_hf_environment(environment)
    else:
        # Formal launch always supplies hf_home; this branch remains useful
        # for pure unit/test argv helpers but still prevents inherited network
        # endpoints from silently changing a later child.
        for key in ("HF_ENDPOINT", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
            environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in normalized_indices)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def python_executable_identity(value: str) -> dict[str, str]:
    """Hash the interpreter target while preserving its lexical entrypoint."""

    text = str(value)
    candidate = shutil.which(text) if os.sep not in text else str(Path(text).expanduser())
    if not candidate:
        raise FileNotFoundError(f"Python executable not found: {text}")
    lexical = Path(candidate).absolute()
    resolved = lexical.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"Python executable is not an executable file: {resolved}")
    # A venv's ``bin/python`` is commonly a symlink to a shared interpreter.
    # Passing the resolved target to Popen bypasses the venv prefix and its
    # site-packages (including torch).  Keep the lexical venv path for exec,
    # but hash the resolved regular file for identity.
    return {"path": str(lexical), "sha256": sha256_file(resolved)}


def runtime_environment_identity(python_path: str | Path) -> dict[str, Any]:
    """Capture the dependency/path surface used by the exact child interpreter."""

    # Preserve a venv's lexical entrypoint for the same reason as
    # ``python_executable_identity``: resolving the symlink before invoking
    # Python drops the venv prefix and its site-packages.
    executable = Path(str(python_path)).absolute()
    probe = r'''
import hashlib, importlib.metadata, json, pathlib, sys, sysconfig
names = ("torch", "torchvision", "transformers", "numpy", "Pillow", "PyYAML", "torchdiffeq", "decord", "lpips", "evo")
packages = {}
for name in names:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
pth = {}
for base in (sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")):
    if not base:
        continue
    for path in sorted(pathlib.Path(base).glob("*.pth")):
        pth[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps({"python_version": sys.version, "sys_path": [str(path) for path in sys.path], "packages": packages, "pth_files": pth}, sort_keys=True))
'''
    child_env = dict(os.environ)
    for key in IMPORT_SHADOW_ENV_KEYS | DISTRIBUTED_ENV_KEYS:
        child_env.pop(key, None)
    child_env["PYTHONNOUSERSITE"] = "1"
    try:
        output = subprocess.check_output(
            [str(executable), "-I", "-c", probe],
            text=True,
            stderr=subprocess.STDOUT,
            env=child_env,
        )
        value = json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not inspect runtime dependency surface for {executable}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("packages"), dict):
        raise RuntimeError("runtime dependency probe returned malformed JSON")
    value["executable"] = str(executable)
    return value


def executable_identity(value: str) -> dict[str, str]:
    """Resolve and record a helper executable used by a formal gate."""

    text = str(value)
    candidate = shutil.which(text) if os.sep not in text else str(Path(text).expanduser())
    if not candidate:
        raise FileNotFoundError(f"executable not found: {text}")
    resolved = Path(candidate).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"executable is not runnable: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _is_root_owned_system_executable(path: str | Path) -> bool:
    """Return whether *path* is a canonical root-owned system executable."""

    candidate = Path(path)
    try:
        lexical = candidate.absolute()
        resolved = candidate.resolve(strict=True)
        lexical_stat = lexical.stat()
        resolved_stat = resolved.stat()
    except OSError:
        return False
    if lexical.name != "nvidia-smi" or resolved.name != "nvidia-smi":
        return False
    if lexical_stat.st_uid != 0 or resolved_stat.st_uid != 0:
        return False
    if not stat.S_ISREG(lexical_stat.st_mode) or not stat.S_ISREG(resolved_stat.st_mode):
        return False
    if not os.access(resolved, os.X_OK):
        return False
    return any(lexical.parent == directory for directory in SYSTEM_NVIDIA_SMI_DIRS)


def nvidia_smi_identity(
    value: str,
    *,
    expected_sha256: str | None = None,
    formal: bool = False,
) -> dict[str, Any]:
    """Resolve the trusted system ``nvidia-smi`` helper and bind its bytes."""

    identity = executable_identity(value)
    trusted_system = _is_root_owned_system_executable(identity["path"])
    if formal and not trusted_system:
        raise ValueError(
            "formal GPU discovery requires a root-owned system nvidia-smi, not a wrapper"
        )
    if formal and expected_sha256 is None:
        raise ValueError("formal GPU discovery requires an external nvidia-smi SHA")
    if expected_sha256 is not None:
        expected = str(expected_sha256)
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
            raise ValueError("nvidia-smi SHA must be a 64-character hexadecimal digest")
        if identity["sha256"] != expected:
            raise RuntimeError(
                f"nvidia-smi SHA mismatch: got {identity['sha256']}, expected {expected}"
            )
    identity["system_root_owned"] = bool(trusted_system)
    return identity


def canonical_reservation_path(root: str | Path, requested: str | Path | None) -> Path:
    """Return the one reservation pathname allowed for formal launches."""

    root_path = Path(root).resolve()
    expected = root_path / CANONICAL_RESERVATION_FILENAME
    if requested is None:
        return expected
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    candidate = candidate.absolute()
    if candidate != expected:
        raise ValueError(
            f"formal launch requires the canonical reservation lock {expected}"
        )
    return expected


def active_reservation_path(root: str | Path) -> Path:
    """Return the private marker used to reject concurrent formal children."""

    return Path(root).resolve() / ACTIVE_RESERVATION_FILENAME


def _pid_is_live_train(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        command = proc_cmdline.read_bytes()
    except OSError:
        # A live process with an unavailable /proc entry is conservatively
        # treated as active; the operator can inspect/remove the exact stale
        # marker after verifying the PID.
        return True
    return b"train.py" in command


def _guard_active_reservation(root: Path) -> Path:
    """Reject another live official child, or remove one exact stale marker."""

    marker = active_reservation_path(root)
    if marker.exists() or marker.is_symlink():
        if marker.is_symlink():
            raise ValueError(f"active reservation marker is a symlink: {marker}")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            pid = int(payload.get("pid", -1)) if isinstance(payload, Mapping) else -1
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pid = -1
        if _pid_is_live_train(pid):
            raise RuntimeError(
                f"another RAE-stream formal training child is active (pid={pid}); refusing duplicate launch"
            )
        # This is the one exact private stale marker, never a broad cleanup.
        marker.unlink(missing_ok=True)
    return marker


@dataclass(frozen=True)
class GPUInfo:
    index: int
    free_mib: int
    util: int | None
    name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "free_mib": self.free_mib,
            "util": self.util,
            "name": self.name,
        }


def build_train_argv(
    *,
    repo_root: str | Path,
    config: str | Path,
    extra_args: Sequence[str] | None = None,
    python_executable: str = "python",
    world_size: int = 1,
) -> list[str]:
    """Build an argv that invokes the official script, never a copied loop.

    For multiple ranks the launcher uses the standard ``torch.distributed.run``
    module from the same interpreter.  No training code is reimplemented and
    no topology is embedded in the source; ``world_size`` is a runtime value.
    """

    root = Path(repo_root)
    if not root.is_dir():
        raise ValueError(f"repo_root is not a directory: {root}")
    config_text = str(config)
    if not config_text:
        raise ValueError("config path is required")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    args = _validate_official_train_args(extra_args)
    # Upstream ``train.py`` has historical defaults (300 epochs, seed 0,
    # checkpoint/eval intervals different from the released reproduction
    # command).  Fill every omitted protected option explicitly so a formal
    # launch cannot accidentally fall back to those non-paper defaults.
    supplied_names = {
        str(token).split("=", 1)[0]
        for token in args
        if str(token).startswith("--")
    }
    for option, value in OFFICIAL_TRAIN_CLI_DEFAULTS.items():
        if option not in supplied_names:
            args.extend([option, str(value)])
            supplied_names.add(option)
    if any("\x00" in token for token in args) or "\x00" in config_text:
        raise ValueError("argv contains NUL")
    if world_size == 1:
        prefix = [str(python_executable)]
    else:
        prefix = [
            str(python_executable),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(world_size),
        ]
    return [*prefix, "train.py", "--config", config_text, *args]


def _eligible_gpu_candidates(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_free_mib: int = 4096,
    max_util: int = 25,
    allow_unknown_util: bool = False,
    require_apps_census: bool = False,
) -> list[dict[str, Any]]:
    """Normalize and filter a GPU inventory without selecting a topology."""

    if isinstance(min_free_mib, bool) or not isinstance(min_free_mib, int) or min_free_mib < 0:
        raise ValueError("min_free_mib must be nonnegative")
    if isinstance(max_util, bool) or not isinstance(max_util, int) or max_util < 0 or max_util > 100:
        raise ValueError("max_util must be in [0,100]")
    candidates: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for row in rows:
        try:
            index = int(row["index"])
            free_raw = row.get("free_mib", row.get("memory.free", row.get("memory_free_mib")))
            free = int(str(free_raw).strip().strip("[]").strip())
            raw_util = row.get("util", row.get("utilization.gpu", row.get("utilization", 100)))
            util_text = str(raw_util).strip().strip("[]").strip().upper() if raw_util is not None else ""
            if raw_util is None or util_text in {"N/A", "NA", "UNKNOWN", ""}:
                util = None
            else:
                util = int(util_text)
            name = str(row.get("name", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed GPU inventory row: {row!r}") from exc
        if index < 0 or index in seen_indices or free < 0 or (util is not None and (util < 0 or util > 100)):
            if index in seen_indices:
                raise ValueError(f"duplicate GPU inventory index: {index}")
            raise ValueError(f"malformed GPU inventory values: {row!r}")
        seen_indices.add(index)
        util_ok = util is not None and util <= max_util
        if util is None and allow_unknown_util:
            util_ok = True
        apps = row.get("apps")
        if isinstance(apps, Sequence) and not isinstance(apps, (str, bytes)) and len(apps) > 0:
            util_ok = False
        if require_apps_census:
            census_ok = (
                isinstance(row.get("uuid"), str)
                and bool(str(row.get("uuid")).strip())
                and row.get("apps_query_status") == "ok"
                and isinstance(apps, Sequence)
                and not isinstance(apps, (str, bytes))
                and len(apps) == 0
            )
            util_ok = util is not None and util_ok and census_ok
        if free >= min_free_mib and util_ok:
            candidate = dict(row)
            candidate.update({"index": index, "free_mib": free, "util": util, "name": name})
            if util is None:
                candidate["utilization_unknown"] = True
                candidate["unknown_util_admitted"] = bool(allow_unknown_util)
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: int(item["index"]))


def choose_free_gpu(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_free_mib: int = 4096,
    max_util: int = 25,
    allow_unknown_util: bool = False,
    require_apps_census: bool = False,
) -> dict[str, Any]:
    """Select the lowest-index GPU meeting a live resource threshold."""

    candidates = _eligible_gpu_candidates(
        rows,
        min_free_mib=min_free_mib,
        max_util=max_util,
        allow_unknown_util=allow_unknown_util,
        require_apps_census=require_apps_census,
    )
    if not candidates:
        suffix = " (unknown utilization is fail-closed)" if not allow_unknown_util else ""
        raise RuntimeError(
            f"no GPU meets runtime quota (free>={int(min_free_mib)} MiB, util<={int(max_util)}%){suffix}"
        )
    return candidates[0]


def choose_free_gpus(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
    indices: Sequence[int] | None = None,
    min_free_mib: int = 4096,
    max_util: int = 25,
    allow_unknown_util: bool = False,
    require_apps_census: bool = False,
) -> list[dict[str, Any]]:
    """Select an ordered, non-overlapping set of quota-eligible GPUs."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("GPU count must be a positive integer")
    candidates = _eligible_gpu_candidates(
        rows,
        min_free_mib=min_free_mib,
        max_util=max_util,
        allow_unknown_util=allow_unknown_util,
        require_apps_census=require_apps_census,
    )
    by_index = {int(row["index"]): row for row in candidates}
    if indices is not None:
        try:
            requested = [int(index) for index in indices]
        except (TypeError, ValueError) as exc:
            raise ValueError("GPU indices must be integers") from exc
        if len(requested) != count or len(set(requested)) != len(requested) or any(index < 0 for index in requested):
            raise ValueError("GPU indices must be unique and match the requested GPU count")
        missing = [index for index in requested if index not in by_index]
        if missing:
            raise RuntimeError(f"requested GPU(s) are not quota-eligible: {missing}")
        return [dict(by_index[index]) for index in requested]
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} GPU(s) meet runtime quota; {count} required")
    return [dict(row) for row in candidates[:count]]


def resolve_world_size(
    requested: int | None,
    *,
    eligible_count: int,
    gradient_accumulation_steps: int = 1,
) -> int:
    """Resolve a runtime world size while preserving the effective batch."""

    if isinstance(eligible_count, bool) or not isinstance(eligible_count, int) or eligible_count <= 0:
        raise ValueError("eligible_count must be a positive integer")
    try:
        steps = require_positive_int(
            gradient_accumulation_steps, name="gradient_accumulation_steps"
        )
        make_accumulation_plan(
            global_batch_size=int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]),
            world_size=1,
            accumulation_steps=steps,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("gradient_accumulation_steps must be a positive divisor of the paper batch") from exc
    divisors = [
        divisor
        for divisor in range(1, int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]) + 1)
        if int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]) % (divisor * steps) == 0
    ]
    if requested is None:
        return max(divisor for divisor in divisors if divisor <= eligible_count)
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("world_size must be a positive integer")
    if requested not in divisors:
        raise ValueError(
            f"world_size {requested} does not divide paper global batch with accumulation={steps}"
        )
    if requested > eligible_count:
        raise RuntimeError(f"world_size {requested} requires {requested} eligible GPUs; found {eligible_count}")
    return requested


def _assert_homogeneous_gpu_set(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject a mixed hardware set whose per-rank memory/throughput is unknown."""

    if not rows:
        raise ValueError("GPU set must not be empty")
    names = {str(row.get("name", "")).strip() for row in rows}
    if len(names) > 1:
        raise RuntimeError(f"selected GPUs have mixed models; refusing an unreviewed topology: {sorted(names)}")
    totals = {
        int(row["total_mib"])
        for row in rows
        if row.get("total_mib") is not None
    }
    if len(totals) > 1:
        raise RuntimeError(
            "selected GPUs have different total memory; refusing an unreviewed topology"
        )


def _parse_gpu_rows(output: str) -> list[dict[str, Any]]:
    """Parse both the legacy four-column and extended inventory formats."""

    rows: list[dict[str, Any]] = []
    def clean(value: str) -> str:
        # Some nvidia-smi wrappers render unavailable values as ``[N/A]``;
        # normalize the decoration before numeric parsing while retaining the
        # original row/name for provenance where possible.
        return value.strip().strip("[]").strip()

    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        # The current query emits index,uuid,free,used,total,util,name.  Keep
        # accepting index,free,util,name for small fake/older nvidia-smi tools.
        try:
            if len(parts) >= 7:
                index_text, uuid, free_text, used_text, total_text, util_text = parts[:6]
                name = ",".join(parts[6:]).strip()
            elif len(parts) == 6:
                index_text, free_text, used_text, total_text, util_text, name = parts
                uuid = None
            elif len(parts) == 4:
                index_text, free_text, util_text, name = parts
                used_text = total_text = None
                uuid = None
            else:
                raise ValueError
            index = int(clean(index_text))
            free_text = clean(free_text)
            util_text = clean(util_text)
            used_text = clean(used_text) if used_text is not None else None
            total_text = clean(total_text) if total_text is not None else None
            free = int(free_text)
            used = None if used_text is None or used_text.upper() in {"N/A", "NA", "UNKNOWN", ""} else int(used_text)
            total = None if total_text is None or total_text.upper() in {"N/A", "NA", "UNKNOWN", ""} else int(total_text)
            util = None if util_text.upper() in {"N/A", "NA", "UNKNOWN", ""} else int(util_text)
        except ValueError as exc:
            raise RuntimeError(f"unexpected nvidia-smi numeric row: {line!r}") from exc
        if index < 0 or free < 0 or (used is not None and used < 0) or (total is not None and total < 0):
            raise RuntimeError(f"unexpected nvidia-smi memory row: {line!r}")
        if util is not None and not 0 <= util <= 100:
            raise RuntimeError(f"unexpected nvidia-smi utilization row: {line!r}")
        row: dict[str, Any] = {"index": index, "free_mib": free, "util": util, "name": name}
        if uuid:
            row["uuid"] = uuid
        if used is not None:
            row["used_mib"] = used
        if total is not None:
            row["total_mib"] = total
        rows.append(row)
    return rows


def _query_compute_apps(executable: str, rows: list[dict[str, Any]]) -> None:
    """Best-effort read-only process census attached to GPU inventory rows.

    Some driver/container combinations expose ``utilization.gpu=N/A``.  The
    launcher never infers idleness from that value; this extra census is
    recorded when available and its failure is itself preserved in the
    receipt.  A failed auxiliary query does not mutate or terminate jobs.
    """

    if not any("uuid" in row for row in rows):
        for row in rows:
            row["apps"] = None
            row["apps_query_status"] = "unavailable_without_uuid"
        return
    command = [
        executable,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        for row in rows:
            row["apps"] = None
            row["apps_query_status"] = f"error:{type(exc).__name__}"
        return
    inventory_uuids = [str(row["uuid"]) for row in rows if "uuid" in row]
    if len(inventory_uuids) != len(rows) or len(set(inventory_uuids)) != len(inventory_uuids):
        for row in rows:
            row["apps"] = None
            row["apps_query_status"] = "malformed"
        return
    by_uuid: dict[str, list[dict[str, Any]]] = {uuid: [] for uuid in inventory_uuids}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            # Preserve a malformed census as unknown rather than claiming no
            # processes.  The primary memory/utilization gate remains active.
            for row in rows:
                row["apps"] = None
                row["apps_query_status"] = "malformed"
            return
        uuid, pid, process_name, used_memory = parts
        try:
            valid_pid = int(pid) > 0
        except (TypeError, ValueError):
            valid_pid = False
        if uuid not in by_uuid or not valid_pid or not process_name:
            for row in rows:
                row["apps"] = None
                row["apps_query_status"] = "malformed"
            return
        by_uuid[uuid].append(
            {"pid": pid, "process_name": process_name, "used_memory": used_memory}
        )
    for row in rows:
        uuid = str(row.get("uuid", ""))
        row["apps"] = by_uuid.get(uuid, [])
        row["apps_query_status"] = "ok"


def query_gpus(*, executable: str = "nvidia-smi", include_apps: bool = False) -> list[dict[str, Any]]:
    """Read a live inventory without reserving, killing, or mutating jobs."""

    command = [
        executable,
        "--query-gpu=index,uuid,memory.free,memory.used,memory.total,utilization.gpu,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("nvidia-smi GPU inventory failed") from exc
    rows = _parse_gpu_rows(output)
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise RuntimeError("nvidia-smi returned a duplicate GPU inventory index")
    uuids = [str(row["uuid"]) for row in rows if "uuid" in row]
    if uuids and (len(uuids) != len(rows) or len(uuids) != len(set(uuids))):
        raise RuntimeError("nvidia-smi returned a missing or duplicate GPU UUID")
    if include_apps:
        _query_compute_apps(executable, rows)
    return rows


def recheck_gpu_inventory(
    initial_rows: Iterable[Mapping[str, Any]],
    query: Any,
    *,
    min_free_mib: int = 4096,
    max_util: int = 25,
    allow_unknown_util: bool = False,
    require_apps_census: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Re-read GPU quota and reject a changed eligible-card decision.

    ``initial_rows`` is the pre-reservation snapshot.  ``query`` must perform
    a fresh read while the caller's cooperative reservation is held.  Keeping
    this helper pure makes the race gate testable without CUDA or a real
    ``nvidia-smi`` binary.
    """

    initial = choose_free_gpu(
        initial_rows,
        min_free_mib=min_free_mib,
        max_util=max_util,
        allow_unknown_util=allow_unknown_util,
        require_apps_census=require_apps_census,
    )
    fresh_rows = list(query())
    fresh = choose_free_gpu(
        fresh_rows,
        min_free_mib=min_free_mib,
        max_util=max_util,
        allow_unknown_util=allow_unknown_util,
        require_apps_census=require_apps_census,
    )
    if int(initial["index"]) != int(fresh["index"]):
        raise RuntimeError(
            "eligible GPU changed between inventory snapshots: "
            f"initial={initial!r}, fresh={fresh!r}"
        )
    initial_uuid = initial.get("uuid")
    fresh_uuid = fresh.get("uuid")
    if initial_uuid is not None and fresh_uuid is not None and str(initial_uuid) != str(fresh_uuid):
        raise RuntimeError(
            "GPU UUID changed between inventory snapshots: "
            f"initial={initial_uuid!r}, fresh={fresh_uuid!r}"
        )
    return initial, fresh, fresh_rows


def recheck_gpu_inventory_set(
    initial_rows: Iterable[Mapping[str, Any]],
    query: Any,
    *,
    world_size: int,
    gpu_indices: Sequence[int] | None = None,
    min_free_mib: int = 4096,
    max_util: int = 25,
    allow_unknown_util: bool = False,
    require_apps_census: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-read and compare a complete multi-GPU reservation set."""

    initial = choose_free_gpus(
        initial_rows,
        count=world_size,
        indices=gpu_indices,
        min_free_mib=min_free_mib,
        max_util=max_util,
        allow_unknown_util=allow_unknown_util,
        require_apps_census=require_apps_census,
    )
    fresh_rows = list(query())
    initial_ids = [int(row["index"]) for row in initial]
    try:
        fresh = choose_free_gpus(
            fresh_rows,
            count=world_size,
            indices=initial_ids,
            min_free_mib=min_free_mib,
            max_util=max_util,
            allow_unknown_util=allow_unknown_util,
            require_apps_census=require_apps_census,
        )
    except RuntimeError as exc:
        # Prefer a stable race-gate diagnostic over leaking the lower-level
        # "not quota-eligible" wording when a reserved member disappeared.
        try:
            fresh_any = choose_free_gpus(
                fresh_rows,
                count=world_size,
                min_free_mib=min_free_mib,
                max_util=max_util,
                allow_unknown_util=allow_unknown_util,
                require_apps_census=require_apps_census,
            )
        except Exception:
            raise
        fresh_ids_any = [int(row["index"]) for row in fresh_any]
        if fresh_ids_any != initial_ids:
            raise RuntimeError(
                "eligible GPU set changed between inventory snapshots: "
                f"initial={initial_ids!r}, fresh={fresh_ids_any!r}"
            ) from exc
        raise
    fresh_ids = [int(row["index"]) for row in fresh]
    if initial_ids != fresh_ids:
        raise RuntimeError(
            "eligible GPU set changed between inventory snapshots: "
            f"initial={initial_ids!r}, fresh={fresh_ids!r}"
        )
    initial_uuids = [str(row.get("uuid", "")) for row in initial]
    fresh_uuids = [str(row.get("uuid", "")) for row in fresh]
    if any(initial_uuids) and initial_uuids != fresh_uuids:
        raise RuntimeError(
            "GPU UUID set changed between inventory snapshots: "
            f"initial={initial_uuids!r}, fresh={fresh_uuids!r}"
        )
    return initial, fresh, fresh_rows


def verify_child_started(
    process: Any,
    *,
    wait_seconds: float = 5.0,
    poll_interval_seconds: float = 0.2,
) -> bool:
    """Require a spawned child to remain alive for a bounded grace period."""

    if not callable(getattr(process, "poll", None)):
        raise TypeError("child process must expose poll()")
    try:
        wait = float(wait_seconds)
        interval = float(poll_interval_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("child startup timing must be numeric") from exc
    if not math.isfinite(wait) or wait < 0.0:
        raise ValueError("wait_seconds must be finite and nonnegative")
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("poll_interval_seconds must be finite and positive")
    deadline = time.monotonic() + wait
    while True:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"training child pid={getattr(process, 'pid', None)} exited before startup verification "
                f"with return code {return_code}"
            )
        now = time.monotonic()
        if now >= deadline:
            return True
        time.sleep(min(interval, max(0.0, deadline - now)))


@contextmanager
def _gpu_reservation(path: str | Path):
    """Hold a cooperative inter-launcher lock for the query→spawn window."""

    # Keep the lexical path here: resolving before open would follow a
    # dangling symlink and create the lock outside the reviewed checkout.
    # Formal callers validate the parent path first; O_NOFOLLOW closes the
    # remaining final-component replacement race at the actual open.
    lock_path = Path(path).absolute()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - Linux server path
        raise RuntimeError("RAE-stream GPU reservation requires a POSIX fcntl lock") from exc
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_NONBLOCK
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(str(lock_path), flags, 0o600)
    except OSError as exc:
        raise RuntimeError(f"reservation lock must be a non-symlink regular file: {lock_path}") from exc
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode):
            raise RuntimeError(f"reservation lock must be a regular file: {lock_path}")
        handle = os.fdopen(descriptor, "a+")
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield lock_path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _load_validation_module(repo_root: Path) -> Any:
    script = repo_root / "scripts" / "validate_rae_stream.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    name = f"rae_stream_validation_{sha256_file(script)[:12]}"
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load validation module from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_prepare_layout_identity(
    receipt_path: str | Path,
    *,
    expected_sha256: str,
    official_root: str | Path,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, int]]:
    """Recheck the cheap lstat map from a pinned prepare receipt.

    This helper is intentionally separate from the full RGB validator so the
    launcher can close the final lock→spawn race without a second multi-GB
    frame/hash pass.
    """

    root = Path(official_root).resolve()
    path = _validate_owned_runtime_path(root, receipt_path, label="prepare receipt")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("prepare receipt must stay below the RAE-stream root") from exc
    if ".." in relative.parts or path.is_symlink():
        raise ValueError("prepare receipt must be a canonical regular file")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise ValueError("prepare receipt contains a symlink parent")
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(expected_sha256)
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
        raise ValueError("prepare receipt SHA must be a 64-character hexadecimal digest")
    _, actual = _stable_sha256_owned_file(root, path, label="prepare receipt")
    if actual != expected:
        raise RuntimeError("prepare receipt SHA changed before spawn")
    if payload is None:
        _, payload, _ = _read_pinned_json_receipt(
            root,
            path,
            expected_sha256=expected,
            label="prepare receipt",
        )
    if not isinstance(payload, Mapping):
        raise ValueError("prepare receipt must be a JSON object")
    module = _load_validation_module(root)
    return dict(module._verify_immutable_path_identities(payload, root))


def _read_pinned_json_receipt(
    root: Path,
    value: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[Path, dict[str, Any], str]:
    """Read one checkout-owned JSON receipt from a single pinned byte snapshot."""

    path = _validate_owned_runtime_path(root, value, label=label)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    if path.resolve() != path:
        raise ValueError(f"{label} must be canonical and contain no symlink components: {path}")
    expected = str(expected_sha256)
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected.lower()):
        raise ValueError(f"{label} SHA must be a 64-character hexadecimal digest")
    before = os.stat(path, follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(f"{label} must be a regular non-symlink: {path}") from exc
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError(f"{label} changed before it was opened")
            data = handle.read()
            after_read = os.fstat(handle.fileno())
    except Exception:
        raise
    after_path = os.stat(path, follow_symlinks=False)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened, field) != getattr(after_read, field) for field in identity_fields):
        raise RuntimeError(f"{label} changed while it was being read")
    if any(getattr(opened, field) != getattr(after_path, field) for field in identity_fields):
        raise RuntimeError(f"{label} was replaced while it was being read")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise RuntimeError(f"{label} SHA mismatch: got {actual}, expected {expected}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} root must be a JSON object")
    return path, dict(payload), actual


def _stable_sha256_owned_file(
    root: Path, value: str | Path, *, label: str
) -> tuple[Path, str]:
    """Hash one checkout-owned regular file from a stable descriptor snapshot."""

    path = _validate_owned_runtime_path(root, value, label=label)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(f"{label} must be a canonical regular file")
    before = os.stat(path, follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(f"{label} must be a regular non-symlink: {path}") from exc
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError(f"{label} changed before it was opened")
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after_read = os.fstat(handle.fileno())
    except Exception:
        raise
    after_path = os.stat(path, follow_symlinks=False)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened, field) != getattr(after_read, field) for field in fields):
        raise RuntimeError(f"{label} changed while it was being read")
    if any(getattr(opened, field) != getattr(after_path, field) for field in fields):
        raise RuntimeError(f"{label} was replaced while it was being read")
    return path, digest.hexdigest()


def verify_pinned_validation_receipt(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    config: Mapping[str, Any],
    expected_revision: str | None,
    prepare_receipt_path: str | Path,
    prepare_receipt_sha256: str,
    authority_path: str | Path,
    authority_sha256: str,
    validation_receipt_path: str | Path,
    validation_receipt_sha256: str,
) -> dict[str, Any]:
    """Admit a prior full validation pass with only cheap immutable rechecks.

    The full RGB/manifest scan is performed by ``validate_rae_stream.py`` once
    during data validation.  Formal launch reuses its externally pinned
    receipt, rechecks the manifest/prepare bytes and immutable lstat map, and
    independently verifies the current converter/source authority.  This keeps
    the launch race gate while avoiding a second multi-gigabyte image pass.
    """

    root = Path(repo_root).resolve()
    assert_formal_revision(expected_revision)
    receipt_file, payload, actual_receipt_sha = _read_pinned_json_receipt(
        root,
        validation_receipt_path,
        expected_sha256=validation_receipt_sha256,
        label="validation receipt",
    )
    if payload.get("status") != "PASS" or payload.get("production_eligible") is not True:
        raise ValueError("pinned validation receipt is not a production PASS")
    if payload.get("claim_boundary") != "FORMAL_DATA_VALIDATION_ONLY":
        raise ValueError("pinned validation receipt has an incompatible claim boundary")

    manifest = _validate_owned_runtime_path(root, manifest_path, label="manifest")
    if not manifest.is_file() or manifest.is_symlink() or manifest.resolve() != manifest:
        raise ValueError("manifest must be a canonical regular file")
    declared_manifest = payload.get("manifest_path")
    if not isinstance(declared_manifest, str) or Path(declared_manifest).resolve() != manifest:
        raise ValueError("pinned validation receipt belongs to a different manifest path")
    _, manifest_sha = _stable_sha256_owned_file(root, manifest, label="manifest")
    if payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("pinned validation receipt manifest SHA does not match current bytes")

    prepare = _validate_owned_runtime_path(root, prepare_receipt_path, label="prepare receipt")
    if not prepare.is_file() or prepare.is_symlink() or prepare.resolve() != prepare:
        raise ValueError("prepare receipt must be a canonical regular file")
    declared_prepare = payload.get("prepare_receipt_path")
    if not isinstance(declared_prepare, str) or Path(declared_prepare).resolve() != prepare:
        raise ValueError("pinned validation receipt belongs to a different prepare receipt")
    _, actual_prepare_sha = _stable_sha256_owned_file(root, prepare, label="prepare receipt")
    if actual_prepare_sha != str(prepare_receipt_sha256):
        raise RuntimeError("prepare receipt changed after the pinned validation pass")
    if payload.get("prepare_receipt_sha256") != actual_prepare_sha:
        raise ValueError("pinned validation receipt prepare SHA does not match current bytes")

    # The validation receipt is produced by an earlier, full data pass.  Bind
    # that pass to the exact production-preparation ABI as well; checking only
    # the outer prepare-file digest would otherwise admit a forged/minimal
    # prepare payload.  Read it through the same descriptor- and identity-
    # checked path used for the validation receipt, without rescanning RGB.
    _, prepare_payload, _ = _read_pinned_json_receipt(
        root,
        prepare,
        expected_sha256=str(prepare_receipt_sha256),
        label="prepare receipt",
    )
    if prepare_payload.get("status") != "PASS" or prepare_payload.get("production_eligible") is not True:
        raise ValueError("pinned validation receipt is not backed by a production PASS prepare receipt")
    declared_prepare_self_hash = prepare_payload.get("receipt_sha256")
    prepare_without_self_hash = dict(prepare_payload)
    prepare_without_self_hash.pop("receipt_sha256", None)
    if declared_prepare_self_hash != canonical_digest(prepare_without_self_hash):
        raise ValueError("prepare receipt self-hash is missing or invalid")
    if prepare_payload.get("manifest_path") != str(manifest):
        raise ValueError("prepare receipt belongs to a different manifest path")
    if prepare_payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("prepare receipt manifest SHA does not match current bytes")
    # These are the fixed loader/data ABI values used by the unchanged
    # official dataset class.  They are not model or optimizer modifications.
    for field, expected in (
        ("min_length", 68),
        ("formal_min_length", 68),
        ("context_size", 4),
        ("len_traj_pred", 64),
    ):
        if prepare_payload.get(field) != expected:
            raise ValueError(f"prepare receipt {field} does not match the formal data ABI")
    if prepare_payload.get("converter_version") != "rae_stream_converter_v2":
        raise ValueError("prepare receipt converter version does not match the reviewed converter")
    for field in ("immutable_data_tree", "immutable_split_lists", "immutable_manifest"):
        if prepare_payload.get(field) is not True:
            raise ValueError(f"prepare receipt lacks immutable {field} guarantee")
    immutable_sources = prepare_payload.get("immutable_source_trees")
    if immutable_sources != {"R2R": True, "RxR": True}:
        raise ValueError("prepare receipt lacks immutable R2R/RxR source trees")

    data_root, split_root = _dataset_paths_from_config(root, config)
    for field, expected in (("data_root", data_root), ("split_root", split_root)):
        declared = payload.get(field)
        if not isinstance(declared, str) or Path(declared).resolve() != expected:
            raise ValueError(f"pinned validation receipt {field} does not match the resolved config")
    if prepare_payload.get("data_root") != str(data_root) or prepare_payload.get("split_root") != str(split_root):
        raise ValueError("prepare receipt dataset paths do not match the resolved config")
    declared_parent_dirs = prepare_payload.get("immutable_parent_dirs")
    if declared_parent_dirs != {str(data_root.parent): True}:
        raise ValueError("prepare receipt generated-data parent identity is incomplete")
    split_hashes = prepare_payload.get("split_list_sha256")
    if not isinstance(split_hashes, Mapping):
        raise ValueError("prepare receipt lacks split-list hashes")
    actual_split_hashes = {
        split: _stable_sha256_owned_file(
            root, split_root / split / "traj_names.txt", label=f"{split} traj_names.txt"
        )[1]
        for split in ("train", "test")
    }
    if dict(split_hashes) != actual_split_hashes:
        raise ValueError("prepare receipt split-list hashes do not match current bytes")

    source_entries = prepare_payload.get("sources")
    if not isinstance(source_entries, list):
        raise ValueError("prepare receipt lacks effective source entries")
    source_by_dataset: dict[str, Mapping[str, Any]] = {}
    for entry in source_entries:
        if not isinstance(entry, Mapping):
            raise ValueError("prepare receipt source entry is malformed")
        dataset_name = str(entry.get("dataset", ""))
        effective = entry.get("effective_source_root")
        if dataset_name not in {"R2R", "RxR"} or dataset_name in source_by_dataset:
            raise ValueError("prepare receipt source dataset set is not exactly R2R/RxR")
        if entry.get("source_mode") != "tar-selected" or entry.get("source_revision") != STREAMVLN_REVISION:
            raise ValueError(f"prepare receipt {dataset_name} source identity is not the fixed release")
        if not isinstance(effective, str) or not effective:
            raise ValueError(f"prepare receipt {dataset_name} source root is invalid")
        effective_path = _validate_owned_runtime_path(
            root, effective, label=f"{dataset_name} effective source root"
        )
        if not effective_path.is_dir():
            raise ValueError(f"prepare receipt {dataset_name} source root is not a directory")
        source_by_dataset[dataset_name] = entry
    if set(source_by_dataset) != {"R2R", "RxR"}:
        raise ValueError("prepare receipt source dataset set is incomplete")
    expected_source_parents = {
        str(Path(str(entry["effective_source_root"])).resolve().parent): True
        for entry in source_by_dataset.values()
    }
    if prepare_payload.get("immutable_source_parent_dirs") != expected_source_parents:
        raise ValueError("prepare receipt source-parent identity is incomplete")
    raw_identity_map = prepare_payload.get("immutable_path_identities")
    if not isinstance(raw_identity_map, Mapping) or not raw_identity_map:
        raise ValueError("prepare receipt lacks immutable path identities")
    expected_identity_paths = {
        str(path.resolve())
        for path in {
            data_root,
            data_root.parent,
            data_root.parent.parent,
            split_root,
            split_root.parent,
            manifest,
            *(split_root / split for split in ("train", "test")),
            *(split_root / split / "traj_names.txt" for split in ("train", "test")),
            *(Path(str(entry["effective_source_root"])).resolve() for entry in source_by_dataset.values()),
            *(Path(str(entry["effective_source_root"])).resolve().parent for entry in source_by_dataset.values()),
        }
    }
    normalized_identity_paths = {
        str(_validate_owned_runtime_path(root, raw_path, label="immutable path identity").resolve())
        for raw_path in raw_identity_map
    }
    if normalized_identity_paths != expected_identity_paths:
        raise ValueError("prepare receipt immutable path identity set does not match the expected layout")
    for field in ("rows", "train_trajectories", "test_trajectories", "frame_count_materialized"):
        try:
            require_positive_int(payload.get(field), name=f"validation receipt {field}")
        except ValueError as exc:
            raise ValueError(f"pinned validation receipt has invalid {field}") from exc
    try:
        decode_samples = require_positive_int(
            payload.get("image_decode_samples"), name="validation receipt image_decode_samples"
        )
    except ValueError as exc:
        raise ValueError("pinned validation receipt image decode sample count is invalid") from exc
    if decode_samples != 8:
        raise ValueError("pinned validation receipt must contain the formal 8 image decode samples")
    if payload.get("converter_identity_checked") is not True:
        raise ValueError("pinned validation receipt lacks converter identity verification")

    current_converter_sha = verify_converter_source(root)
    current_converter_source = converter_closure_digest_map(root)
    current_converter_bundle = converter_bundle_sha256(current_converter_source)
    if payload.get("converter_sha256") != current_converter_sha:
        raise ValueError("pinned validation receipt converter SHA does not match current bytes")
    if payload.get("converter_source_sha256") != current_converter_source:
        raise ValueError("pinned validation receipt converter closure does not match current bytes")
    if payload.get("converter_bundle_sha256") != current_converter_bundle:
        raise ValueError("pinned validation receipt converter bundle does not match current bytes")
    for field, current_value in (
        ("converter_sha256", current_converter_sha),
        ("converter_source_sha256", current_converter_source),
        ("converter_bundle_sha256", current_converter_bundle),
    ):
        if prepare_payload.get(field) != current_value:
            raise ValueError(f"prepare receipt {field} does not match current converter bytes")

    recorded_source = payload.get("official_source")
    if not isinstance(recorded_source, Mapping) or recorded_source.get("commit") != OFFICIAL_COMMIT:
        raise ValueError("pinned validation receipt is not bound to the admitted official commit")
    current_source = verify_upstream_source(root)
    current_authority = verify_code_authority(
        root,
        authority_path,
        expected_sha256=str(authority_sha256),
    )
    # The old validation receipt may legitimately predate the reviewed
    # train.py accumulation adapter, but immutable model/planner/config bytes
    # must agree.  Record the deliberate train-source transition explicitly;
    # never silently equate the two source identities.
    recorded_files = recorded_source.get("files")
    current_files = current_source.get("files")
    if not isinstance(recorded_files, Mapping) or not isinstance(current_files, Mapping):
        raise ValueError("pinned validation receipt lacks the official source file map")
    for relative in ("models.py", "planning_eval.py"):
        if recorded_files.get(relative) != OFFICIAL_BASELINE_RESTRICTED_SHA256[relative]:
            raise ValueError(f"pinned validation receipt {relative} hash is not the official baseline")
        if current_files.get(relative) != OFFICIAL_RESTRICTED_SHA256[relative]:
            raise ValueError(f"current official source {relative} hash is not admitted")
    recorded_train = recorded_files.get("train.py")
    if recorded_train not in {
        OFFICIAL_BASELINE_RESTRICTED_SHA256["train.py"],
        OFFICIAL_RESTRICTED_SHA256["train.py"],
    }:
        raise ValueError("pinned validation receipt train.py hash is not a reviewed source variant")
    if recorded_source.get("data_config_files") != current_source.get("data_config_files"):
        raise ValueError("pinned validation receipt data/config source map differs from current bytes")
    recorded_authority = payload.get("code_authority")
    if not isinstance(recorded_authority, Mapping):
        raise ValueError("pinned validation receipt lacks its code-authority provenance")
    if recorded_authority.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("pinned validation receipt code authority has an incompatible commit")
    recorded_authority_files = recorded_authority.get("files")
    current_authority_files = current_authority.get("files")
    closure_transitions = verify_validation_receipt_closure(
        recorded_authority_files, current_authority_files
    )
    for relative in ("models.py", "planning_eval.py"):
        if recorded_authority_files.get(relative) != current_authority_files.get(relative):
            raise ValueError(f"pinned validation receipt code-authority {relative} differs from current bytes")
    if recorded_train != current_files.get("train.py"):
        train_transition = "baseline_validation_to_reviewed_accumulation_source"
    else:
        train_transition = "same_reviewed_source"
    immutable_layout = verify_prepare_layout_identity(
        prepare,
        expected_sha256=str(prepare_receipt_sha256),
        official_root=root,
    )
    result = dict(payload)
    result.update(
        {
            "validation_mode": "PINNED_RECEIPT_WITH_IMMUTABLE_LAYOUT_RECHECK",
            "validation_receipt_path": str(receipt_file),
            "validation_receipt_sha256": actual_receipt_sha,
            "validated_receipt_official_source": recorded_source,
            "validated_receipt_code_authority": payload.get("code_authority"),
            "validated_receipt_train_source_sha256": recorded_train,
            "current_train_source_sha256": current_files.get("train.py"),
            "validation_source_transition": train_transition,
            "current_official_source": current_source,
            "current_code_authority": current_authority,
            "validation_receipt_closure_transitions": closure_transitions,
            "immutable_layout_rechecked": True,
            "immutable_layout_identity_count": len(immutable_layout),
        }
    )
    return result


def validate_manifest_for_training(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    config: Mapping[str, Any] | None = None,
    expected_revision: str | None = None,
    prepare_receipt_path: str | Path | None = None,
    prepare_receipt_sha256: str | None = None,
    authority_path: str | Path | None = None,
    authority_sha256: str | None = None,
    validation_receipt_path: str | Path | None = None,
    validation_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the full non-mutating manifest gate required before formal train."""

    root = Path(repo_root).resolve()
    fixed_revision = assert_formal_revision(expected_revision)
    manifest = Path(manifest_path).resolve()
    if prepare_receipt_path is None:
        raise ValueError("formal training requires the prepare receipt path")
    if (validation_receipt_path is None) != (validation_receipt_sha256 is None):
        raise ValueError("validation receipt path and SHA must be supplied together")
    if validation_receipt_path is not None:
        if authority_path is None or authority_sha256 is None:
            raise ValueError("pinned validation receipt requires the external code authority")
        if config is None:
            config = load_yaml_mapping(root / "config" / "rae_stream.yaml")
        return verify_pinned_validation_receipt(
            repo_root=root,
            manifest_path=manifest,
            config=config,
            expected_revision=fixed_revision,
            prepare_receipt_path=prepare_receipt_path,
            prepare_receipt_sha256=str(prepare_receipt_sha256),
            authority_path=authority_path,
            authority_sha256=str(authority_sha256),
            validation_receipt_path=validation_receipt_path,
            validation_receipt_sha256=str(validation_receipt_sha256),
        )
    if config is None:
        config = load_yaml_mapping(root / "config" / "rae_stream.yaml")
    data_root, split_root = _dataset_paths_from_config(root, config)
    module = _load_validation_module(root)
    receipt = module.validate_dataset(
        data_root=data_root,
        split_root=split_root,
        manifest_path=manifest,
        official_root=root,
        expected_revision=fixed_revision,
        allow_nonproduction=False,
        require_production=True,
        prepare_receipt_path=prepare_receipt_path,
        prepare_receipt_sha256=prepare_receipt_sha256,
        authority_path=authority_path,
        authority_sha256=authority_sha256,
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("status") != "PASS"
        or receipt.get("production_eligible") is not True
        or receipt.get("claim_boundary") != "FORMAL_DATA_VALIDATION_ONLY"
    ):
        raise RuntimeError("manifest validation did not return a formal production PASS")
    return dict(receipt)


def _dataset_paths_from_config(root: Path, config: Mapping[str, Any]) -> tuple[Path, Path]:
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {"rae_stream"}:
        raise ValueError("training config must contain only datasets.rae_stream")
    entry = datasets.get("rae_stream")
    if not isinstance(entry, Mapping):
        raise ValueError("datasets.rae_stream must be a mapping")
    data_value = entry.get("data_folder")
    train_value = entry.get("train")
    test_value = entry.get("test")
    if not all(isinstance(value, str) and value for value in (data_value, train_value, test_value)):
        raise ValueError("datasets.rae_stream requires data_folder/train/test paths")
    data = Path(data_value)
    train = Path(train_value)
    test = Path(test_value)
    data_root = (data if data.is_absolute() else root / data).resolve()
    train_root = (train if train.is_absolute() else root / train).resolve()
    test_root = (test if test.is_absolute() else root / test).resolve()
    if train_root.parent != test_root.parent:
        raise ValueError("datasets.rae_stream train/test must share one split root")
    split_root = train_root.parent
    return data_root, split_root


def cpu_snapshot() -> dict[str, Any]:
    affinity = None
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "affinity": affinity,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _with_self_hash(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Add a canonical digest over a receipt payload (excluding the digest)."""

    result = dict(value)
    result.pop(field, None)
    result[field] = canonical_digest(result)
    return result


def verify_memory_preflight(
    path: str | Path,
    *,
    expected_sha256: str,
    world_size: int,
    gradient_accumulation_steps: int = 1,
    config_sha256: str,
    gpu_indices: Sequence[int],
    gpu_uuids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify a bounded official one-step memory probe receipt.

    Formal training is not allowed to infer batch feasibility from free
    ``nvidia-smi`` memory alone.  The receipt must come from the unchanged
    official model/training ABI, include an optimizer step under the released
    BF16/compile flags, and bind the exact derived YAML, world size,
    accumulation factor, and physical GPU list that will be launched.
    """

    receipt_path = Path(path).resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    actual_sha = sha256_file(receipt_path)
    expected = str(expected_sha256)
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected.lower()):
        raise ValueError("memory preflight SHA must be a 64-character hexadecimal digest")
    if actual_sha != expected:
        raise RuntimeError(
            f"memory preflight SHA mismatch: got {actual_sha}, expected {expected}"
        )
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("memory preflight receipt is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("memory preflight receipt root must be an object")
    if payload.get("receipt_sha256") != canonical_digest({k: v for k, v in payload.items() if k != "receipt_sha256"}):
        raise ValueError("memory preflight receipt self-hash is missing or invalid")
    if payload.get("status") != "PASS" or payload.get("probe_kind") != "official_train_step_memory":
        raise ValueError("memory preflight did not report an official one-step PASS")
    if payload.get("official_commit") != OFFICIAL_COMMIT:
        raise ValueError("memory preflight official commit does not match the reviewed checkout")
    source_hashes = payload.get("official_source_sha256")
    if not isinstance(source_hashes, Mapping) or any(
        str(source_hashes.get(name)) != digest for name, digest in OFFICIAL_RESTRICTED_SHA256.items()
    ):
        raise ValueError("memory preflight source hashes do not match the reviewed official files")
    official_source = payload.get("official_source")
    if not isinstance(official_source, Mapping) or official_source.get(
        "training_source_variant"
    ) != ACCUMULATION_SOURCE_VARIANT:
        raise ValueError(
            "memory preflight official_source is not the reviewed accumulation variant"
        )
    try:
        requested_world = require_positive_int(world_size, name="world_size")
        requested_accumulation = require_positive_int(
            gradient_accumulation_steps, name="gradient_accumulation_steps"
        )
        make_accumulation_plan(
            global_batch_size=int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]),
            world_size=requested_world,
            accumulation_steps=requested_accumulation,
        )
        payload_world = require_positive_int(payload["world_size"], name="payload world_size")
        payload_local = require_positive_int(
            payload["per_rank_batch_size"], name="payload per_rank_batch_size"
        )
        payload_accumulation = require_positive_int(
            payload.get("gradient_accumulation_steps", 1),
            name="payload gradient_accumulation_steps",
        )
        expected_profile = training_profile_name(requested_accumulation)
        payload_profile = payload.get("training_profile")
        if requested_accumulation > 1 and payload_profile != expected_profile:
            raise ValueError(
                "accumulation memory preflight must identify the RAE-stream-accumulation profile"
            )
        if payload_profile is not None and payload_profile != expected_profile:
            raise ValueError("memory preflight training profile does not match accumulation")
        paper_profile = payload.get("paper_training_profile", PAPER_PROFILE_NAME)
        if requested_accumulation > 1 and "paper_training_profile" not in payload:
            raise ValueError(
                "accumulation memory preflight must record paper_training_profile"
            )
        if paper_profile != PAPER_PROFILE_NAME:
            raise ValueError("memory preflight paper training profile is not paper-reproduction-v1")
        expected_eval_semantics = (
            "physical_microbatch" if requested_accumulation > 1 else "paper_per_rank_batch"
        )
        if requested_accumulation > 1 and any(
            field not in payload
            for field in ("evaluation_batch_size_per_rank", "evaluation_batch_semantics")
        ):
            raise ValueError("accumulation memory preflight must record evaluation batch semantics")
        if "evaluation_batch_size_per_rank" in payload:
            try:
                payload_eval_batch = require_positive_int(
                    payload["evaluation_batch_size_per_rank"],
                    name="payload evaluation_batch_size_per_rank",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("memory preflight evaluation batch is invalid") from exc
        if "evaluation_batch_size_per_rank" in payload and payload_eval_batch != payload_local:
            raise ValueError("memory preflight evaluation batch is inconsistent")
        if "evaluation_batch_semantics" in payload and payload["evaluation_batch_semantics"] != expected_eval_semantics:
            raise ValueError("memory preflight evaluation batch semantics are inconsistent")
        if payload_accumulation != requested_accumulation:
            raise ValueError("accumulation factor does not match requested launch")
        payload_config = {
            "batch_size": payload_local,
            "gradient_accumulation_steps": payload_accumulation,
        }
        if "effective_global_batch_size" in payload:
            payload_config["effective_global_batch_size"] = payload["effective_global_batch_size"]
        mapping = paper_batch_mapping(
            payload_config,
            world_size=payload_world,
            gradient_accumulation_steps=payload_accumulation,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"memory preflight world/per-rank batch mapping is invalid: {exc}"
        ) from exc
    try:
        payload_global = require_positive_int(
            payload.get("global_batch_size"), name="payload global_batch_size"
        )
        payload_world_value = require_positive_int(payload.get("world_size"), name="payload world_size")
        payload_local_value = require_positive_int(
            payload.get("per_rank_batch_size"), name="payload per_rank_batch_size"
        )
        payload_accumulation_value = require_positive_int(
            payload.get("gradient_accumulation_steps", 1),
            name="payload gradient_accumulation_steps",
        )
    except ValueError as exc:
        raise ValueError("memory preflight batch fields must be exact positive integers") from exc
    if payload_global != mapping["global_batch_size"]:
        raise ValueError("memory preflight global batch is not 96")
    if (
        payload_world_value != requested_world
        or payload_local_value != mapping["per_rank_batch_size"]
        or payload_accumulation_value != requested_accumulation
    ):
        raise ValueError(
            "memory preflight world/per-rank batch/accumulation does not match the requested launch"
        )
    if requested_accumulation > 1 and any(
        field not in payload
        for field in ("gradient_accumulation_steps", "microbatch_global_size", "effective_global_batch_size")
    ):
        raise ValueError(
            "accumulation memory preflight must record microbatch/effective-batch fields"
        )
    runtime_metadata = payload.get("runtime_config")
    if requested_accumulation > 1 and not isinstance(runtime_metadata, Mapping):
        raise ValueError(
            "accumulation memory preflight must record runtime_config metadata"
        )
    if runtime_metadata is not None:
        if not isinstance(runtime_metadata, Mapping):
            raise ValueError("memory preflight runtime_config must be an object")
        expected_profile = training_profile_name(requested_accumulation)
        if runtime_metadata.get("config_sha256") != str(config_sha256):
            raise ValueError("memory preflight runtime_config hash does not match resolved config")
        if runtime_metadata.get("training_profile") != expected_profile:
            raise ValueError("memory preflight runtime_config profile does not match accumulation")
        if runtime_metadata.get("paper_training_profile") != PAPER_PROFILE_NAME:
            raise ValueError("memory preflight runtime_config paper profile is invalid")
        if requested_accumulation > 1 and runtime_metadata.get("derived") is not True:
            raise ValueError("accumulation memory preflight runtime_config must be derived")
        try:
            nested_accumulation = require_positive_int(
                runtime_metadata.get("gradient_accumulation_steps"),
                name="runtime_config gradient_accumulation_steps",
            )
            nested_microbatch = require_positive_int(
                runtime_metadata.get("microbatch_size"),
                name="runtime_config microbatch_size",
            )
            nested_microbatch_global = require_positive_int(
                runtime_metadata.get("microbatch_global_size"),
                name="runtime_config microbatch_global_size",
            )
            nested_effective = require_positive_int(
                runtime_metadata.get("effective_global_batch_size"),
                name="runtime_config effective_global_batch_size",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("memory preflight runtime_config batch fields are invalid") from exc
        if (
            nested_accumulation != requested_accumulation
            or nested_microbatch != mapping["per_rank_batch_size"]
            or nested_microbatch_global != mapping["microbatch_global_size"]
            or nested_effective != mapping["effective_global_batch_size"]
        ):
            raise ValueError("memory preflight runtime_config batch mapping is inconsistent")
        try:
            nested_eval_batch = require_positive_int(
                runtime_metadata.get("evaluation_batch_size_per_rank"),
                name="runtime_config evaluation_batch_size_per_rank",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("memory preflight runtime_config evaluation batch is invalid") from exc
        expected_eval_semantics = (
            "physical_microbatch" if requested_accumulation > 1 else "paper_per_rank_batch"
        )
        if nested_eval_batch != mapping["per_rank_batch_size"] or runtime_metadata.get(
            "evaluation_batch_semantics"
        ) != expected_eval_semantics:
            raise ValueError("memory preflight runtime_config evaluation batch semantics are inconsistent")
        nested_mapping = runtime_metadata.get("batch_mapping")
        if not isinstance(nested_mapping, Mapping):
            raise ValueError("memory preflight runtime_config batch_mapping is missing")
        for field, expected_value in mapping.items():
            try:
                actual_value = require_positive_int(
                    nested_mapping.get(field), name=f"runtime_config batch_mapping {field}"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("memory preflight runtime_config batch_mapping is invalid") from exc
            if actual_value != expected_value:
                raise ValueError("memory preflight runtime_config batch_mapping is inconsistent")
    try:
        payload_effective = require_positive_int(
            payload.get("effective_global_batch_size", mapping.get("global_batch_size")),
            name="payload effective_global_batch_size",
        )
    except ValueError as exc:
        raise ValueError("memory preflight effective global batch must be an exact integer") from exc
    if payload_effective != 96:
        raise ValueError("memory preflight effective global batch is not 96")
    if "microbatch_global_size" in payload:
        try:
            payload_microbatch_global = require_positive_int(
                payload["microbatch_global_size"], name="payload microbatch_global_size"
            )
        except ValueError as exc:
            raise ValueError("memory preflight microbatch global size must be an exact integer") from exc
        if payload_microbatch_global != mapping["microbatch_global_size"]:
            raise ValueError("memory preflight microbatch global size is inconsistent")
    try:
        requested_indices = [int(index) for index in gpu_indices]
        receipt_indices = [int(index) for index in payload["gpu_indices"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("memory preflight GPU indices are invalid") from exc
    if receipt_indices != requested_indices or len(receipt_indices) != requested_world:
        raise ValueError("memory preflight GPU list does not match the selected launch set")
    if gpu_uuids is not None:
        try:
            expected_uuids = [str(uuid) for uuid in gpu_uuids]
            receipt_uuids = [str(uuid) for uuid in payload["gpu_uuids"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("memory preflight GPU UUIDs are invalid") from exc
        if receipt_uuids != expected_uuids or len(receipt_uuids) != requested_world:
            raise ValueError("memory preflight GPU UUIDs do not match the selected launch set")
    if str(payload.get("config_sha256")) != str(config_sha256):
        raise ValueError("memory preflight was run against a different resolved config")
    try:
        steps = require_positive_int(
            payload["probe_steps"], name="memory preflight probe_steps"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("memory preflight probe_steps is invalid") from exc
    if steps < 1:
        raise ValueError("memory preflight probe_steps must be at least one")
    if payload.get("optimizer_step") is not True or payload.get("oom") is not False:
        raise ValueError("memory preflight must include a finite optimizer step with no OOM")
    for field in ("free_mib_before", "peak_memory_allocated_mib"):
        values = payload.get(field)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != requested_world:
            raise ValueError(f"memory preflight {field} must contain one value per rank")
        try:
            if any(float(value) <= 0 or not math.isfinite(float(value)) for value in values):
                raise ValueError(f"memory preflight {field} contains non-positive values")
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("memory preflight"):
                raise
            raise ValueError(f"memory preflight {field} contains non-numeric values") from exc
    try:
        safety_margin = float(payload["safety_margin_mib"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("memory preflight safety_margin_mib is invalid") from exc
    if not math.isfinite(safety_margin) or safety_margin < 0:
        raise ValueError("memory preflight safety margin must be finite and nonnegative")
    free_values = [float(value) for value in payload["free_mib_before"]]
    peak_values = [float(value) for value in payload["peak_memory_allocated_mib"]]
    if any(peak + safety_margin > free for peak, free in zip(peak_values, free_values)):
        raise ValueError(
            "memory preflight peak plus safety margin exceeds measured free memory"
        )
    if int(payload.get("bfloat16", -1)) != 1 or int(payload.get("torch_compile", -1)) != 1:
        raise ValueError("memory preflight must use the paper/README BF16 and compile flags")
    result = dict(payload)
    result["path"] = str(receipt_path)
    result["sha256"] = actual_sha
    return result


def assert_memory_preflight_headroom(
    preflight: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Close the preflight-to-spawn memory race under the reservation lock.

    ``verify_memory_preflight`` proves that the exact model path once fit on
    each selected rank.  This second, intentionally cheap check compares the
    latest locked inventory to that receipt before Popen; a different job may
    have consumed memory after the probe, so the generic 4096-MiB quota alone
    is not sufficient evidence for the paper batch.
    """

    if not isinstance(preflight, Mapping):
        raise TypeError("memory preflight must be a mapping")
    if not isinstance(selected_rows, Sequence) or isinstance(selected_rows, (str, bytes)):
        raise TypeError("selected GPU rows must be a sequence")
    try:
        world = int(preflight["world_size"])
        margin = float(preflight["safety_margin_mib"])
        peaks = list(preflight["peak_memory_allocated_mib"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("memory preflight headroom fields are incomplete") from exc
    if world <= 0 or len(selected_rows) != world or len(peaks) != world:
        raise ValueError("memory preflight headroom rank count does not match the selected set")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("memory preflight safety margin must be finite and nonnegative")
    try:
        peak_values = [float(value) for value in peaks]
        current_free = [float(row["free_mib"]) for row in selected_rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selected GPU rows or preflight peaks are non-numeric") from exc
    if any(not math.isfinite(value) or value < 0 for value in peak_values):
        raise ValueError("memory preflight peaks must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in current_free):
        raise ValueError("current GPU free memory must be finite and positive")
    required = [peak + margin for peak in peak_values]
    failures = [
        (rank, free, need)
        for rank, (free, need) in enumerate(zip(current_free, required))
        if free < need
    ]
    if failures:
        raise RuntimeError(
            "current free GPU memory is below the verified preflight requirement: "
            + ", ".join(f"rank {rank}: free={free:.1f} MiB < required={need:.1f} MiB" for rank, free, need in failures)
        )
    return {
        "status": "PASS",
        "world_size": world,
        "current_free_mib": current_free,
        "required_mib": required,
        "safety_margin_mib": margin,
    }


def _validate_owned_runtime_path(
    root: Path,
    value: str | Path,
    *,
    label: str,
    require_new: bool = False,
    require_regular: bool = False,
) -> Path:
    """Require a runtime receipt/log/lock path inside the dedicated checkout."""

    raw = Path(value).expanduser()
    path = raw if raw.is_absolute() else root / raw
    # Reject symlink components before resolving the path; otherwise a path
    # such as ``receipts -> /other`` would pass a superficial prefix check.
    path = path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay below the RAE-stream repo root") from exc
    if ".." in relative.parts:
        raise ValueError(f"{label} contains traversal components")
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink parent: {current}")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    if require_regular and path.exists():
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                raise ValueError(f"{label} must be a regular file: {path}")
        except OSError as exc:
            raise ValueError(f"{label} cannot be inspected safely: {path}") from exc
    if require_new and path.exists():
        raise RuntimeError(f"{label} already exists; refusing duplicate formal launch: {path}")
    return path


def _assert_formal_config_paths(root: Path, config: str | Path, base: str | Path | None) -> Path:
    """Keep formal train tied to the reviewed template or derived YAML.

    A derived config is admitted only below the dedicated receipts namespace;
    its semantic equality to ``config/rae_stream.yaml`` is checked separately
    by :func:`assert_runtime_paper_config` before a child is spawned.
    """

    expected_config = (root / "config" / "rae_stream.yaml").resolve()
    raw_config = Path(config if Path(config).is_absolute() else root / config)
    actual_config = raw_config.resolve()
    if actual_config != expected_config:
        try:
            relative = actual_config.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"formal training requires the dedicated overlay or derived config below {root / 'receipts'}") from exc
        if len(relative.parts) < 3 or relative.parts[:2] != ("receipts", "runtime_configs"):
            raise ValueError(f"formal training requires the dedicated overlay or receipts/runtime_configs derived YAML: {expected_config}")
        if not actual_config.is_file() or actual_config.is_symlink():
            raise ValueError(f"derived runtime config must be a regular file: {actual_config}")
    expected_base = (root / "config" / "raenwm.yaml").resolve()
    if base is not None:
        actual_base = Path(base if Path(base).is_absolute() else root / base).resolve()
        if actual_base != expected_base:
            raise ValueError(f"formal training requires the untouched base config {expected_base}")
    return actual_config


def materialize_runtime_config(
    repo_root: str | Path,
    *,
    template_path: str | Path = "config/rae_stream.yaml",
    world_size: int,
    gradient_accumulation_steps: int = 1,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Create (or verify) the deterministic per-rank YAML for a DDP run.

    The checked-in world-one template is used directly only for the original
    accumulation-free profile.  Any multi-rank or accumulation-enabled run
    receives a deterministic file under ``receipts/runtime_configs`` with the
    derived microbatch/accumulation fields.  The returned metadata is suitable
    for a durable launch receipt.
    """

    root = Path(repo_root).resolve()
    template_file = Path(template_path)
    if not template_file.is_absolute():
        template_file = root / template_file
    template_file = template_file.resolve()
    expected_template = (root / "config" / "rae_stream.yaml").resolve()
    if template_file != expected_template:
        raise ValueError(f"runtime config template must be {expected_template}")
    template = load_yaml_mapping(template_file)
    assert_training_profile(template, epochs=int(OFFICIAL_CHECKOUT_PROFILE["epochs"]), world_size=1)
    steps = require_positive_int(
        gradient_accumulation_steps, name="gradient_accumulation_steps"
    )
    resolved_world_size = require_positive_int(world_size, name="world_size")
    plan = make_accumulation_plan(
        global_batch_size=int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]),
        world_size=resolved_world_size,
        accumulation_steps=steps,
    )
    resolved = runtime_paper_config(
        template,
        world_size=resolved_world_size,
        gradient_accumulation_steps=steps,
    )
    mapping = paper_batch_mapping(
        resolved,
        world_size=resolved_world_size,
        gradient_accumulation_steps=steps,
    )
    template_sha = sha256_file(template_file)
    if resolved_world_size == 1 and steps == 1:
        return template_file, resolved, {
            "template_path": str(template_file),
            "template_sha256": template_sha,
            "derived": False,
            "config_sha256": template_sha,
            "batch_mapping": mapping,
            "training_profile": training_profile_name(steps),
            "paper_training_profile": PAPER_PROFILE_NAME,
            "gradient_accumulation_steps": steps,
            "microbatch_size": plan.microbatch_size,
            "microbatch_global_size": plan.microbatch_global_size,
            "effective_global_batch_size": plan.effective_global_batch_size,
            "evaluation_batch_size_per_rank": plan.microbatch_size,
            "evaluation_batch_semantics": (
                "physical_microbatch" if steps > 1 else "paper_per_rank_batch"
            ),
        }
    runtime_dir = root / "receipts" / "runtime_configs"
    runtime_file = runtime_dir / (
        f"rae_stream_{template_sha[:16]}_world{resolved_world_size}_accum{steps}.yaml"
    )
    _validate_owned_runtime_path(root, runtime_file, label="derived runtime config")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("PyYAML is required to write the derived runtime config") from exc
    encoded = yaml.safe_dump(resolved, sort_keys=False, default_flow_style=False).encode("utf-8")
    if runtime_file.exists():
        if runtime_file.is_symlink() or runtime_file.read_bytes() != encoded:
            raise RuntimeError(f"derived runtime config already exists with different bytes: {runtime_file}")
    else:
        temporary = runtime_file.with_name(f".{runtime_file.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, runtime_file)
        finally:
            temporary.unlink(missing_ok=True)
    # Reparse the bytes that the child will consume and prove only the derived
    # batch differs from the reviewed template.
    parsed = load_yaml_mapping(runtime_file)
    assert_runtime_paper_config(
        template,
        parsed,
        world_size=resolved_world_size,
        gradient_accumulation_steps=steps,
        epochs=50,
    )
    return runtime_file, parsed, {
        "template_path": str(template_file),
        "template_sha256": template_sha,
        "derived": True,
        "config_sha256": sha256_file(runtime_file),
        "path": str(runtime_file),
        "batch_mapping": mapping,
        "training_profile": training_profile_name(steps),
        "paper_training_profile": PAPER_PROFILE_NAME,
        "gradient_accumulation_steps": steps,
        "microbatch_size": plan.microbatch_size,
        "microbatch_global_size": plan.microbatch_global_size,
        "effective_global_batch_size": plan.effective_global_batch_size,
        "evaluation_batch_size_per_rank": plan.microbatch_size,
        "evaluation_batch_semantics": (
            "physical_microbatch" if steps > 1 else "paper_per_rank_batch"
        ),
    }


def resolved_training_identity(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    manifest_path: str | Path | None = None,
    expected_commit: str | None = None,
    base_config_path: str | Path | None = None,
    prepare_receipt_path: str | Path | None = None,
    prepare_receipt_sha256: str | None = None,
    hf_home: str | Path | None = None,
    hf_snapshot_root: str | Path | None = None,
    authority_path: str | Path | None = None,
    authority_sha256: str | None = None,
    nvidia_smi_sha256: str | None = None,
    python_executable: str | None = None,
    world_size: int = 1,
    gradient_accumulation_steps: int = 1,
    diagnostic_inference: bool = False,
) -> dict[str, Any]:
    """Verify source/config identity before a child process is created."""

    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config = load_yaml_mapping(config_file)
    accumulation_steps = require_positive_int(
        gradient_accumulation_steps, name="gradient_accumulation_steps"
    )
    resolved_world_size = require_positive_int(world_size, name="world_size")
    # Resolve the effective arithmetic up front so the identity cannot bind a
    # template/config to a different microbatch later in the launch window.
    make_accumulation_plan(
        global_batch_size=int(OFFICIAL_CHECKOUT_PROFILE["global_batch_size"]),
        world_size=resolved_world_size,
        accumulation_steps=accumulation_steps,
    )
    template_file = (root / "config" / "rae_stream.yaml").resolve()
    if config_file.resolve() == template_file:
        # The checked-in file is the paper-total-batch template.  For a
        # multi-rank plan, derive the prospective mapping without pretending
        # that template batch=96 is a per-rank value.
        assert_training_profile(config, epochs=OFFICIAL_CHECKOUT_PROFILE["epochs"], world_size=1)
        runtime_config_info = {
            "template_path": str(template_file),
            "template_sha256": sha256_file(template_file),
            "derived": False,
            "config_sha256": sha256_file(config_file),
        }
        prospective = runtime_paper_config(
            config,
            world_size=resolved_world_size,
            gradient_accumulation_steps=accumulation_steps,
        )
        batch_mapping = paper_batch_mapping(
            prospective,
            world_size=resolved_world_size,
            gradient_accumulation_steps=accumulation_steps,
        )
        overlay_config = config
    else:
        if not config_file.is_relative_to(root / "receipts" / "runtime_configs"):
            raise ValueError("resolved config must be the checked-in overlay or a receipts/runtime_configs derived YAML")
        template = load_yaml_mapping(template_file)
        assert_runtime_paper_config(
            template,
            config,
            world_size=resolved_world_size,
            gradient_accumulation_steps=accumulation_steps,
            epochs=50,
        )
        runtime_config_info = {
            "template_path": str(template_file),
            "template_sha256": sha256_file(template_file),
            "derived": True,
            "config_sha256": sha256_file(config_file),
            "path": str(config_file),
        }
        batch_mapping = paper_batch_mapping(
            config,
            world_size=resolved_world_size,
            gradient_accumulation_steps=accumulation_steps,
        )
        overlay_config = template
    # Static config must remain relocatable; runtime absolute paths are passed
    # only by the launcher/official process environment.
    assert_no_fixed_runtime_paths(config)
    experiment_dir = assert_fresh_training_output(root, config)
    # Compare the RAE-stream overlay against the untouched official config at
    # the top-level semantic mapping.  This prevents a copied config from
    # drifting in a model/optimizer/transport field while comments/ordering
    # remain visually plausible.
    base_file: Path | None
    if base_config_path is None:
        candidate = root / "config" / "raenwm.yaml"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        base_file = candidate
    else:
        candidate = Path(base_config_path)
        base_file = candidate if candidate.is_absolute() else root / candidate
        base_file = base_file.resolve()
        if not base_file.is_file():
            raise FileNotFoundError(base_file)
    overlay_changed: list[str] | None = None
    base_config: dict[str, Any] | None = None
    if base_file is not None:
        actual_base_sha = sha256_file(base_file)
        if actual_base_sha != OFFICIAL_BASE_CONFIG_SHA256:
            raise RuntimeError(
                "official base config SHA mismatch; refusing a drifted overlay comparison: "
                f"got {actual_base_sha}, expected {OFFICIAL_BASE_CONFIG_SHA256}"
            )
        base_config = load_yaml_mapping(base_file)
        if base_file.resolve() == config_file.resolve():
            raise ValueError("RAE-stream launcher requires a dedicated overlay, not config/raenwm.yaml")
        overlay_changed = assert_data_overlay(base_config, overlay_config)
    source = verify_upstream_source(root, expected_commit=expected_commit or "0219ce41c44d515f86719dd763c1efe7c7f72519",
        **({"diagnostic_inference": True} if diagnostic_inference else {}))
    assets = None
    if hf_home is not None or hf_snapshot_root is not None:
        assets = verify_rae_assets(
            root,
            hf_home=hf_home,
            hf_snapshot_root=hf_snapshot_root,
        )
    authority = None
    if authority_path is not None or authority_sha256 is not None:
        if authority_path is None or authority_sha256 is None:
            raise ValueError("authority_path and authority_sha256 must be supplied together")
        authority = verify_code_authority(root, authority_path, expected_sha256=str(authority_sha256))
    converter_sha = verify_converter_source(root)
    converter_source_sha256 = converter_closure_digest_map(root)
    converter_bundle_digest = converter_bundle_sha256(converter_source_sha256)
    identity: dict[str, Any] = {
        "repo_root": str(root),
        "training_profile": training_profile_name(accumulation_steps),
        "paper_training_profile": PAPER_PROFILE_NAME,
        "paper_batch_mapping": batch_mapping,
        "gradient_accumulation_steps": accumulation_steps,
        "evaluation_batch_size_per_rank": batch_mapping["per_rank_batch_size"],
        "evaluation_batch_semantics": (
            "physical_microbatch" if accumulation_steps > 1 else "paper_per_rank_batch"
        ),
        "paper_training_values": {
            "epochs": int(OFFICIAL_CHECKOUT_PROFILE["epochs"]),
            "lr": float(OFFICIAL_CHECKOUT_PROFILE["lr"]),
            "final_lr": float(OFFICIAL_CHECKOUT_PROFILE["final_lr"]),
            "lr_schedule": str(OFFICIAL_CHECKOUT_PROFILE["lr_schedule"]),
            "weight_decay": float(OFFICIAL_CHECKOUT_PROFILE["weight_decay"]),
        },
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "config": config,
        "runtime_config": runtime_config_info,
        "official_source": source,
        "converter_sha256": converter_sha,
        "converter_source_sha256": converter_source_sha256,
        "converter_bundle_sha256": converter_bundle_digest,
        "cpu": cpu_snapshot(),
        "experiment_dir": str(experiment_dir),
    }
    if assets is not None:
        identity["rae_assets"] = assets
    if authority is not None:
        identity["code_authority"] = authority
    if python_executable is not None:
        executable_info = python_executable_identity(python_executable)
        identity["python_executable"] = executable_info
        identity["runtime_environment"] = runtime_environment_identity(executable_info["path"])
    if base_file is not None:
        identity["base_config_path"] = str(base_file.resolve())
        identity["base_config_sha256"] = actual_base_sha
        identity["overlay_changed_keys"] = overlay_changed
    if manifest_path is not None:
        manifest = Path(manifest_path).resolve()
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        identity["manifest_path"] = str(manifest)
        identity["manifest_sha256"] = sha256_file(manifest)
    if prepare_receipt_path is not None:
        receipt = Path(prepare_receipt_path).resolve()
        if not receipt.is_file():
            raise FileNotFoundError(receipt)
        identity["prepare_receipt_path"] = str(receipt)
        actual_prepare_sha = sha256_file(receipt)
        if prepare_receipt_sha256 is not None and actual_prepare_sha != str(prepare_receipt_sha256):
            raise RuntimeError(
                f"prepare receipt SHA mismatch: got {actual_prepare_sha}, expected {prepare_receipt_sha256}"
            )
        identity["prepare_receipt_sha256"] = actual_prepare_sha
        if prepare_receipt_sha256 is not None:
            identity["prepare_receipt_sha256_expected"] = str(prepare_receipt_sha256)
    # The official loader reads traj_names.txt directly, while creating its
    # own writable index cache beside it.  Bind the immutable list bytes into
    # the launch identity so a replacement between validation and spawn is
    # detected without making the cache directory readonly.
    try:
        _, split_root = _dataset_paths_from_config(root, config)
        split_hashes = {
            split: sha256_file(split_root / split / "traj_names.txt")
            for split in ("train", "test")
        }
    except (FileNotFoundError, ValueError):
        if manifest_path is not None:
            raise
    else:
        identity["split_list_sha256"] = split_hashes
    if nvidia_smi_sha256 is not None:
        identity["nvidia_smi_sha256_expected"] = str(nvidia_smi_sha256)
    identity["resolved_identity_sha256"] = canonical_digest(identity)
    return identity


def launch_official_train(
    *,
    repo_root: str | Path,
    config: str | Path,
    extra_args: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    gpu_rows: Iterable[Mapping[str, Any]] | None = None,
    gpu_index: int | None = None,
    gpu_indices: Sequence[int] | None = None,
    world_size: int | None = None,
    gradient_accumulation_steps: int = 1,
    min_free_mib: int = 4096,
    max_util: int = 25,
    allow_unknown_util: bool = False,
    python_executable: str = "python",
    nvidia_smi_executable: str = "nvidia-smi",
    log_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    prepare_receipt_path: str | Path | None = None,
    prepare_receipt_sha256: str | None = None,
    expected_revision: str | None = None,
    base_config_path: str | Path | None = None,
    hf_home: str | Path | None = None,
    hf_snapshot_root: str | Path | None = None,
    authority_path: str | Path | None = None,
    authority_sha256: str | None = None,
    nvidia_smi_sha256: str | None = None,
    memory_preflight_path: str | Path | None = None,
    memory_preflight_sha256: str | None = None,
    validation_receipt_path: str | Path | None = None,
    validation_receipt_sha256: str | None = None,
    manifest_validation: Mapping[str, Any] | None = None,
    reservation_path: str | Path | None = None,
    startup_wait_seconds: float = 5.0,
    startup_poll_seconds: float = 0.2,
) -> subprocess.Popen[Any]:
    """Run the official training script after all local gates pass.

    The launcher may select one or more healthy GPUs at runtime.  For a
    multi-GPU run it derives a per-rank YAML and invokes official torchrun;
    the synchronized effective global batch remains exactly 96.  Accumulation
    is an explicit runtime memory adaptation; model/loss/optimizer code is
    otherwise unchanged.
    """

    root = Path(repo_root).resolve()
    accumulation_steps = require_positive_int(
        gradient_accumulation_steps, name="gradient_accumulation_steps"
    )
    if manifest_validation is not None:
        raise ValueError(
            "manifest_validation injection is forbidden; launch_official_train validates the manifest itself"
        )
    if manifest_path is None:
        raise ValueError("formal training requires --manifest and a validated production dataset")
    if prepare_receipt_path is None:
        raise ValueError("formal training requires --prepare-receipt")
    if prepare_receipt_sha256 is None:
        raise ValueError("formal training requires --prepare-receipt-sha256")
    if hf_home is None:
        raise ValueError("formal training requires a dedicated --hf-home for verified offline RAE assets")
    if authority_path is None or authority_sha256 is None:
        raise ValueError("formal training requires the externally pinned code authority")
    if receipt_path is None:
        raise ValueError("formal training requires --receipt so startup identity is durable")
    if log_path is None:
        raise ValueError("formal training requires --log so child output is durable")
    if memory_preflight_path is None or memory_preflight_sha256 is None:
        raise ValueError(
            "formal training requires a matching one-step --memory-preflight receipt"
        )
    if validation_receipt_path is None or validation_receipt_sha256 is None:
        raise ValueError(
            "formal training requires a pinned production validation receipt"
        )
    # Resolve and reject a symlink/traversal receipt before taking the first
    # identity snapshot.  The path is part of that snapshot below and must be
    # available before any field is recorded.
    validation_receipt_file = _validate_owned_runtime_path(
        root, validation_receipt_path, label="validation receipt"
    )
    template_config_path = (root / "config" / "rae_stream.yaml").resolve()
    requested_config_path = Path(config if Path(config).is_absolute() else root / config).resolve()
    if requested_config_path != template_config_path:
        raise ValueError(
            "formal launch accepts the checked-in config/rae_stream.yaml template; "
            "the launcher creates any per-rank derived YAML"
        )
    _assert_formal_config_paths(root, template_config_path, base_config_path)
    fixed_revision = assert_formal_revision(expected_revision)
    if int(min_free_mib) < 4096:
        raise ValueError("formal GPU quota cannot be relaxed below 4096 MiB")
    if int(max_util) > 25:
        raise ValueError("formal GPU quota cannot be relaxed above 25% utilization")
    if allow_unknown_util:
        raise ValueError(
            "formal launch never admits unknown GPU utilization; repair telemetry instead"
        )
    checked_extra_args = _validate_official_train_args(extra_args)
    python_identity = python_executable_identity(python_executable)
    nvidia_identity = nvidia_smi_identity(
        nvidia_smi_executable,
        expected_sha256=nvidia_smi_sha256,
        formal=True,
    )
    initial_identity = resolved_training_identity(
        repo_root=root,
        config_path=template_config_path,
        manifest_path=manifest_path,
        prepare_receipt_path=prepare_receipt_path,
        prepare_receipt_sha256=prepare_receipt_sha256,
        base_config_path=base_config_path,
        hf_home=hf_home,
        hf_snapshot_root=hf_snapshot_root,
        authority_path=authority_path,
        authority_sha256=authority_sha256,
        nvidia_smi_sha256=nvidia_smi_sha256,
        python_executable=python_identity["path"],
        world_size=1,
        gradient_accumulation_steps=accumulation_steps,
    )
    # Do not accept a caller-supplied "PASS" mapping as a substitute for the
    # validator.  The CLI used to validate once and pass that mapping here,
    # which left this public helper vulnerable to a forged receipt.  Validate
    # exactly once at this ownership boundary instead; the CLI consumes the
    # resulting child receipt after spawn.
    manifest_validation = validate_manifest_for_training(
            repo_root=root,
            manifest_path=manifest_path,
            config=initial_identity["config"],
            expected_revision=fixed_revision,
            prepare_receipt_path=prepare_receipt_path,
            prepare_receipt_sha256=prepare_receipt_sha256,
            authority_path=authority_path,
            authority_sha256=authority_sha256,
            validation_receipt_path=validation_receipt_path,
            validation_receipt_sha256=validation_receipt_sha256,
        )
    if manifest_validation.get("status") != "PASS":
        raise RuntimeError("manifest validation gate did not return PASS")
    # Bind the receipt to the exact file passed to the child.  This catches a
    # stale receipt after a manifest was replaced between validation and launch.
    validated_path = manifest_validation.get("manifest_path")
    if validated_path is not None and Path(str(validated_path)).resolve() != Path(manifest_path).resolve():
        raise RuntimeError("manifest validation receipt belongs to a different path")
    validated_sha = manifest_validation.get("manifest_sha256")
    actual_sha = sha256_file(manifest_path)
    if validated_sha is not None and str(validated_sha) != actual_sha:
        raise RuntimeError("manifest changed after validation; refusing to start training")
    initial_identity["manifest_validation"] = dict(manifest_validation)
    initial_identity["manifest_sha256"] = actual_sha
    initial_identity["validation_receipt_path"] = str(validation_receipt_file)
    initial_identity["validation_receipt_sha256"] = str(
        manifest_validation["validation_receipt_sha256"]
    )
    initial_identity["python_executable"] = python_identity
    initial_identity["train_extra_args"] = list(checked_extra_args)
    initial_identity["resolved_identity_sha256"] = canonical_digest(
        {
            key: value
            for key, value in initial_identity.items()
            if key != "resolved_identity_sha256"
        }
    )
    try:
        wait_value = float(startup_wait_seconds)
        poll_value = float(startup_poll_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("startup timing must be numeric") from exc
    if not math.isfinite(wait_value) or wait_value < 0.0:
        raise ValueError("startup_wait_seconds must be finite and nonnegative")
    if not math.isfinite(poll_value) or poll_value <= 0.0:
        raise ValueError("startup_poll_seconds must be finite and positive")

    rows = list(gpu_rows) if gpu_rows is not None else query_gpus(
        executable=nvidia_identity["path"], include_apps=True
    )
    # Resolve topology only from the live inventory.  No physical index or
    # process count is embedded in code; an explicit list is honored in its
    # supplied order, otherwise the largest divisor of the paper batch that
    # fits the currently eligible cards is selected.
    eligible_rows = _eligible_gpu_candidates(
        rows,
        min_free_mib=min_free_mib,
        max_util=max_util,
        allow_unknown_util=allow_unknown_util,
        require_apps_census=True,
    )
    if gpu_indices is not None and gpu_index is not None:
        raise ValueError("provide gpu_indices or gpu_index, not both")
    requested_indices: list[int] | None = None
    if gpu_indices is not None:
        try:
            requested_indices = [int(index) for index in gpu_indices]
        except (TypeError, ValueError) as exc:
            raise ValueError("GPU indices must be integers") from exc
        if not requested_indices or len(set(requested_indices)) != len(requested_indices):
            raise ValueError("GPU indices must be a non-empty unique list")
        inferred_world = len(requested_indices)
    elif gpu_index is not None:
        try:
            requested_indices = [int(gpu_index)]
        except (TypeError, ValueError) as exc:
            raise ValueError("gpu_index must be an integer") from exc
        inferred_world = 1
    else:
        inferred_world = None
    if world_size is None:
        resolved_world = (
            resolve_world_size(
                inferred_world,
                eligible_count=len(eligible_rows),
                gradient_accumulation_steps=accumulation_steps,
            )
            if inferred_world is not None
            else resolve_world_size(
                None,
                eligible_count=len(eligible_rows),
                gradient_accumulation_steps=accumulation_steps,
            )
        )
    else:
        resolved_world = resolve_world_size(
            world_size,
            eligible_count=len(eligible_rows),
            gradient_accumulation_steps=accumulation_steps,
        )
        if inferred_world is not None and resolved_world != inferred_world:
            raise ValueError("world_size must match the supplied GPU list")
    initial_selected_set = choose_free_gpus(
        rows,
        count=resolved_world,
        indices=requested_indices,
        min_free_mib=min_free_mib,
        max_util=max_util,
        allow_unknown_util=allow_unknown_util,
        require_apps_census=True,
    )
    _assert_homogeneous_gpu_set(initial_selected_set)
    lock = canonical_reservation_path(root, reservation_path)
    _validate_owned_runtime_path(root, lock, label="reservation lock", require_regular=True)
    # Existence is checked again under the reservation lock below.  The first
    # check only rejects symlink/traversal paths before any GPU query.
    receipt_file = _validate_owned_runtime_path(root, receipt_path, label="receipt")
    log_file_requested = _validate_owned_runtime_path(root, log_path, label="log")
    memory_preflight_file = _validate_owned_runtime_path(
        root, memory_preflight_path, label="memory preflight receipt"
    )
    active_marker = _validate_owned_runtime_path(
        root, active_reservation_path(root), label="active reservation marker"
    )
    with _gpu_reservation(lock) as held_lock:
        # Serialize all formal launchers through one canonical lock.  These
        # checks are deliberately inside the lock so two callers cannot both
        # pass a pre-existing-path test and then overwrite each other's
        # receipt/log.  The active marker also rejects distinct receipt paths
        # while an earlier official child is still alive.
        receipt_file = _validate_owned_runtime_path(
            root, receipt_file, label="receipt", require_new=True
        )
        log_file = _validate_owned_runtime_path(root, log_file_requested, label="log", require_new=True)
        _guard_active_reservation(root)
        # Re-read the checkout-owned validation receipt under the same lock
        # used for the GPU query.  A replacement after the initial data gate
        # must never be paired with the child command.
        _, _, final_validation_receipt_sha = _read_pinned_json_receipt(
            root,
            validation_receipt_file,
            expected_sha256=validation_receipt_sha256,
            label="validation receipt",
        )
        if final_validation_receipt_sha != str(
            manifest_validation["validation_receipt_sha256"]
        ):
            raise RuntimeError("validation receipt changed before spawn")
        _, locked_prepare_payload, _ = _read_pinned_json_receipt(
            root,
            prepare_receipt_path,
            expected_sha256=str(prepare_receipt_sha256),
            label="prepare receipt",
        )
        # Re-hash the manifest while holding the same cooperative lock.  The
        # pre-lock digest above is only a provenance snapshot; binding that
        # stale value into the spawn identity would leave a replacement
        # window between validation and the final source/config check.
        _, locked_manifest_sha = _stable_sha256_owned_file(
            root, manifest_path, label="manifest"
        )
        if locked_manifest_sha != str(manifest_validation.get("manifest_sha256", "")):
            raise RuntimeError("manifest changed before spawn")
        actual_sha = locked_manifest_sha
        # Re-read while holding the cooperative lock.  The first snapshot is
        # retained for provenance; the second is the actual pre-spawn gate.
        initial, selected, fresh_rows = recheck_gpu_inventory_set(
            rows,
            lambda: query_gpus(executable=nvidia_identity["path"], include_apps=True),
            world_size=resolved_world,
            gpu_indices=[int(row["index"]) for row in initial_selected_set],
            min_free_mib=min_free_mib,
            max_util=max_util,
            allow_unknown_util=allow_unknown_util,
            require_apps_census=True,
        )
        _assert_homogeneous_gpu_set(selected)
        selected_indices = [int(row["index"]) for row in selected]
        runtime_config_path, runtime_config, runtime_config_info = materialize_runtime_config(
            root,
            template_path=template_config_path,
            world_size=resolved_world,
            gradient_accumulation_steps=accumulation_steps,
        )
        identity = resolved_training_identity(
            repo_root=root,
            config_path=runtime_config_path,
            manifest_path=manifest_path,
            prepare_receipt_path=prepare_receipt_path,
            prepare_receipt_sha256=prepare_receipt_sha256,
            base_config_path=base_config_path,
            hf_home=hf_home,
            hf_snapshot_root=hf_snapshot_root,
            authority_path=authority_path,
            authority_sha256=authority_sha256,
            nvidia_smi_sha256=nvidia_smi_sha256,
            python_executable=python_identity["path"],
            world_size=resolved_world,
            gradient_accumulation_steps=accumulation_steps,
        )
        if str(identity.get("manifest_sha256", "")) != actual_sha:
            raise RuntimeError("manifest changed after the locked validation snapshot")
        identity["runtime_config"] = runtime_config_info
        identity["manifest_validation"] = dict(manifest_validation)
        identity["validation_receipt_path"] = str(validation_receipt_file)
        identity["validation_receipt_sha256"] = str(
            manifest_validation["validation_receipt_sha256"]
        )
        identity["python_executable"] = python_identity
        identity["train_extra_args"] = list(checked_extra_args)
        # The derived YAML changes only the per-rank batch.  Reuse the already
        # completed production manifest validation after proving its dataset
        # paths are identical; do not rescan millions of RGB records merely to
        # validate a scalar batch decomposition.
        if _dataset_paths_from_config(root, initial_identity["config"]) != _dataset_paths_from_config(root, runtime_config):
            raise RuntimeError("derived runtime config changed the validated dataset paths")
        memory_preflight = verify_memory_preflight(
            memory_preflight_file,
            expected_sha256=str(memory_preflight_sha256),
            world_size=resolved_world,
            gradient_accumulation_steps=accumulation_steps,
            config_sha256=str(identity["config_sha256"]),
            gpu_indices=selected_indices,
            gpu_uuids=[str(row.get("uuid", "")) for row in selected],
        )
        memory_headroom = assert_memory_preflight_headroom(memory_preflight, selected)
        identity["memory_preflight"] = memory_preflight
        identity["memory_preflight_headroom"] = memory_headroom
        identity["resource_gate_mode"] = (
            "UNKNOWN_UTIL_EXPLICIT_OVERRIDE_WITH_EMPTY_COMPUTE_CENSUS"
            if any(row.get("util") is None for row in selected)
            else "MEASURED_UTIL_WITH_EMPTY_COMPUTE_CENSUS"
        )
        identity["resource_gate_unknown_util_is_not_idle_measurement"] = any(
            row.get("util") is None for row in selected
        )
        # Recheck the reviewed converter after the final inventory read and
        # while holding the reservation.  This closes the validation→spawn
        # TOCTOU window without touching the official trainer source.
        converter_sha = verify_converter_source(root)
        converter_source_sha256 = converter_closure_digest_map(root)
        converter_bundle_digest = converter_bundle_sha256(converter_source_sha256)
        if (
            str(identity.get("converter_sha256")) != converter_sha
            or identity.get("converter_source_sha256") != converter_source_sha256
            or str(identity.get("converter_bundle_sha256")) != converter_bundle_digest
        ):
            raise RuntimeError("RAE-stream converter changed after identity validation")
        # Re-resolve the complete source/config identity at the last safe
        # boundary.  This catches edits to the overlay, untouched base YAML,
        # or restricted official files after the initial validation and before
        # the child receives its argv.
        final_identity = resolved_training_identity(
            repo_root=root,
            config_path=runtime_config_path,
            manifest_path=manifest_path,
            prepare_receipt_path=prepare_receipt_path,
            prepare_receipt_sha256=prepare_receipt_sha256,
            base_config_path=base_config_path,
            hf_home=hf_home,
            hf_snapshot_root=hf_snapshot_root,
            authority_path=authority_path,
            authority_sha256=authority_sha256,
            nvidia_smi_sha256=nvidia_smi_sha256,
            python_executable=python_identity["path"],
            world_size=resolved_world,
            gradient_accumulation_steps=accumulation_steps,
        )
        if str(final_identity.get("manifest_sha256", "")) != actual_sha:
            raise RuntimeError("manifest changed during final source/config identity resolution")
        final_identity["memory_preflight"] = memory_preflight
        final_identity["memory_preflight_headroom"] = memory_headroom
        final_identity["manifest_validation"] = dict(manifest_validation)
        final_identity["python_executable"] = python_identity
        final_identity["train_extra_args"] = list(checked_extra_args)
        for key in (
            "official_source",
            "converter_sha256",
            "converter_source_sha256",
            "converter_bundle_sha256",
            "manifest_sha256",
            "split_list_sha256",
            "prepare_receipt_sha256",
            "prepare_receipt_sha256_expected",
            "nvidia_smi_sha256_expected",
            "rae_assets",
            "code_authority",
            "python_executable",
            "runtime_environment",
        ):
            if final_identity.get(key) != initial_identity.get(key):
                raise RuntimeError(f"RAE-stream identity changed before spawn: {key}")
        identity["spawn_identity"] = final_identity
        identity["nvidia_smi"] = nvidia_identity
        identity["distributed_env_policy"] = (
            "torchrun_standalone_env_scrubbed" if resolved_world > 1 else "single_process_env_scrubbed"
        )
        identity["world_size"] = resolved_world
        identity["physical_gpu_indices"] = selected_indices
        identity["logical_rank_gpu_map"] = {
            str(rank): int(index) for rank, index in enumerate(selected_indices)
        }
        # A full manifest/source digest pass already happened before the
        # reservation.  Recheck the pinned lstat map here, immediately before
        # opening the log/spawning the official child, so a writable ancestor
        # cannot swap the validated data tree in the remaining race window.
        identity["prepare_layout_identity"] = verify_prepare_layout_identity(
            prepare_receipt_path,
            expected_sha256=str(prepare_receipt_sha256),
            official_root=root,
            payload=locked_prepare_payload,
        )
        # The layout check proves the immutable inode map; this final stable
        # byte digest also catches an in-place manifest mutation immediately
        # before the child receives its argv.
        _, final_manifest_sha = _stable_sha256_owned_file(
            root, manifest_path, label="manifest"
        )
        if final_manifest_sha != actual_sha:
            raise RuntimeError("manifest changed immediately before spawn")
        _, final_train_split_sha = _stable_sha256_owned_file(
            root,
            _dataset_paths_from_config(root, runtime_config)[1] / "train" / "traj_names.txt",
            label="train traj_names.txt",
        )
        _, final_test_split_sha = _stable_sha256_owned_file(
            root,
            _dataset_paths_from_config(root, runtime_config)[1] / "test" / "traj_names.txt",
            label="test traj_names.txt",
        )
        expected_split_sha = locked_prepare_payload.get("split_list_sha256")
        if not isinstance(expected_split_sha, Mapping):
            raise RuntimeError("prepare receipt split-list hashes are missing before spawn")
        if {
            "train": final_train_split_sha,
            "test": final_test_split_sha,
        } != dict(expected_split_sha):
            raise RuntimeError("trajectory split list changed immediately before spawn")
        # Upstream auto-resume is checked again at the last pre-spawn boundary.
        assert_fresh_training_output(root, final_identity["config"])
        child_env = build_child_environment(env, gpu_indices=selected_indices, hf_home=hf_home)
        # Set externally, never in model code.  Physical indices are mapped to
        # logical local ranks by CUDA_VISIBLE_DEVICES for torchrun.
        argv = build_train_argv(
            repo_root=root,
            config=runtime_config_path,
            extra_args=checked_extra_args,
            python_executable=python_identity["path"],
            world_size=resolved_world,
        )
        stdout_handle = None
        log_file_path: str | None = None
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # ``x`` prevents accidental mixing with an old log even if a caller
        # bypasses the path preflight between snapshots.
        stdout_handle = log_file.open("x", encoding="utf-8", buffering=1)
        log_file_path = str(log_file)
        # Arm the durable receipt before Popen.  If the final post-start write
        # fails, the child is terminated and this record remains evidence that
        # no unreceipted formal process was intentionally left running.
        armed_receipt = dict(identity)
        armed_receipt.update(
            {
                "status": "FORMAL_TRAINING_ARMED",
                "argv": argv if "argv" in locals() else None,
                "gpu_initial": initial,
                "gpu": selected,
                "gpu_inventory_initial": rows,
                "gpu_inventory_rechecked": fresh_rows,
                "reservation_lock": str(held_lock),
                "receipt_path": str(receipt_file),
                "log_path": log_file_path,
                "active_reservation_path": str(active_marker),
            }
        )
        # Build argv before arming so the exact child command is durable.
        if armed_receipt.get("argv") is None:
            armed_receipt["argv"] = build_train_argv(
                repo_root=root,
                config=runtime_config_path,
                extra_args=checked_extra_args,
                python_executable=python_identity["path"],
                world_size=resolved_world,
            )
        armed_receipt = _with_self_hash(armed_receipt, "receipt_sha256")
        try:
            _atomic_json(receipt_file, armed_receipt)
        except Exception:
            stdout_handle.close()
            raise
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(root),
                env=child_env,
                stdout=stdout_handle,
                stderr=subprocess.STDOUT if stdout_handle is not None else None,
                text=True,
                start_new_session=True,
            )
        except Exception:
            if stdout_handle is not None:
                stdout_handle.close()
            raise
        if stdout_handle is not None:
            # The child inherited the descriptor; closing our copy avoids a
            # file handle leak while preserving the log stream.
            stdout_handle.close()
        try:
            _atomic_json(
                active_marker,
                {
                    "status": "ACTIVE",
                    "pid": int(process.pid),
                    "gpu": selected,
                    "world_size": resolved_world,
                    "physical_gpu_indices": selected_indices,
                    "receipt_path": str(receipt_file),
                    "started_at_epoch": time.time(),
                },
            )
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            raise
        try:
            verify_child_started(
                process,
                wait_seconds=wait_value,
                poll_interval_seconds=poll_value,
            )
        except Exception as exc:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            if receipt_file is not None:
                failed = dict(identity)
                failed.update(
                    {
                        "status": "FORMAL_TRAINING_START_FAILED",
                        "argv": argv,
                        "cuda_visible_devices": child_env["CUDA_VISIBLE_DEVICES"],
                        "gpu_initial": initial,
                        "gpu": selected,
                        "gpu_inventory_initial": rows,
                        "gpu_inventory_rechecked": fresh_rows,
                        "reservation_lock": str(held_lock),
                        "pid": getattr(process, "pid", None),
                        "cwd": str(root),
                        "log_path": log_file_path,
                        "active_reservation_path": str(active_marker),
                        "error": str(exc),
                    }
                )
                _atomic_json(receipt_file, _with_self_hash(failed, "receipt_sha256"))
            try:
                active_marker.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        receipt = dict(identity)
        receipt.update(
            {
                "status": "FORMAL_TRAINING_STARTED",
                "startup_verified": True,
                "startup_wait_seconds": wait_value,
                "argv": argv,
                "cuda_visible_devices": child_env["CUDA_VISIBLE_DEVICES"],
                "gpu_initial": initial,
                "gpu": selected,
                "gpu_inventory_initial": rows,
                "gpu_inventory_rechecked": fresh_rows,
                "reservation_lock": str(held_lock),
                "pid": process.pid,
                "cwd": str(root),
                "log_path": log_file_path,
                "active_reservation_path": str(active_marker),
            }
        )
        try:
            setattr(process, "rae_stream_launch_receipt", receipt)
        except Exception:
            pass
        receipt = _with_self_hash(receipt, "receipt_sha256")
        try:
            _atomic_json(receipt_file, receipt)
        except Exception:
            # Never leave a formally launched child without a durable final
            # identity.  The process belongs to this helper, so terminating it
            # is safe and avoids a duplicate retry ambiguity.
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            try:
                active_marker.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        return process


__all__ = [
    "GPUInfo",
    "build_train_argv",
    "build_child_environment",
    "assert_formal_revision",
    "python_executable_identity",
    "runtime_environment_identity",
    "executable_identity",
    "nvidia_smi_identity",
    "canonical_reservation_path",
    "active_reservation_path",
    "choose_free_gpu",
    "choose_free_gpus",
    "resolve_world_size",
    "query_gpus",
    "recheck_gpu_inventory",
    "recheck_gpu_inventory_set",
    "verify_prepare_layout_identity",
    "verify_validation_receipt_closure",
    "verify_pinned_validation_receipt",
    "verify_child_started",
    "cpu_snapshot",
    "validate_manifest_for_training",
    "resolved_training_identity",
    "materialize_runtime_config",
    "launch_official_train",
    "verify_memory_preflight",
    "assert_memory_preflight_headroom",
]
