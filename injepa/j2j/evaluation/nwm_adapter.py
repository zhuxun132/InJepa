"""Explicit wrapper boundary for the official NWM planning chain.

No NWM implementation is copied here.  The caller must inject the verified
official preprocessing/planner callable and an admitted checkpoint identity.
This prevents a geometric toy adapter from being mistaken for an NWM result.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import hashlib
import inspect
from pathlib import Path
import subprocess
import re
from typing import Any, Callable

from .adapters import NWMContinuousActionAdapter


_CHAIN_ROLE_ALIASES = {
    "planner": "planner",
    "planner_symbol": "planner",
    "preprocess": "preprocessing",
    "preprocessing": "preprocessing",
    "preprocess_symbol": "preprocessing",
    "preprocessing_symbol": "preprocessing",
}


def _coerce_official_image(image: Any) -> Any:
    """Adapt a Habitat RGB array to the PIL input expected by NWM ``transform``.

    Habitat's RGB sensor returns an ``H×W×3`` ``uint8`` NumPy array, whereas
    the verified NWM transform begins with a PIL-only aspect-ratio crop.  PIL
    images (and any other input type accepted by a caller-provided official
    transform) are left untouched.  The conversion is intentionally narrow:
    non-RGB arrays and non-``uint8`` arrays are rejected instead of silently
    changing channel order or numeric range.
    """

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - official runtime dependency
        raise RuntimeError(
            "Pillow is required to bridge Habitat RGB arrays to official NWM transform"
        ) from exc
    if isinstance(image, Image.Image):
        return image

    # Import NumPy lazily so the adapter's legacy pair-transform path does not
    # acquire an eager image dependency merely by being imported.
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - official runtime dependency
        raise RuntimeError(
            "NumPy is required to bridge Habitat RGB arrays to official NWM transform"
        ) from exc
    if not isinstance(image, np.ndarray):
        return image
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            "official NWM RGB input must be an HxWx3 array before PIL conversion"
        )
    if image.dtype != np.uint8:
        raise TypeError(
            "official NWM RGB NumPy input must use uint8 sensor values"
        )
    return Image.fromarray(np.ascontiguousarray(image))


def _snapshot_factual_image(image: Any) -> Any:
    """Take a small immutable-at-boundary snapshot of one real RGB frame.

    Habitat wrappers are allowed to reuse a sensor buffer.  Keeping that
    mutable object in the context queue would silently rewrite an earlier
    factual timestep.  Only the concrete image/tensor types are copied; other
    caller values retain the legacy pass-through behavior.
    """

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - NumPy is an official runtime dependency
        np = None
    if np is not None and isinstance(image, np.ndarray):
        return np.array(image, copy=True)

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is an official runtime dependency
        Image = None
    if Image is not None and isinstance(image, Image.Image):
        return image.copy()

    # Avoid importing torch merely for legacy step-mode adapters.  The module
    # name check is sufficient to distinguish tensors from arbitrary objects
    # that happen to expose a ``clone`` method.
    if str(type(image).__module__).split(".", 1)[0] == "torch":
        clone = getattr(image, "clone", None)
        if callable(clone):
            return clone()
    return image


class OfficialUnaryPreprocess:
    """Call an official one-image transform while retaining its identity.

    The checked-in NWM transform is a ``torchvision.Compose`` object, not a
    two-argument Python function.  This tiny wrapper lets an external factory
    apply that object independently to the current and goal images.  The
    factory must provide a source descriptor for the exact official
    ``misc.py:transform`` object; the descriptor is checked against the
    verified repository by :func:`_callable_source_identity`.
    """

    __nwm_unary__ = True

    def __init__(self, transform: Callable[[Any], Any], source_identity: Mapping[str, Any]):
        if not callable(transform):
            raise TypeError("official unary preprocess transform must be callable")
        if not isinstance(source_identity, Mapping):
            raise TypeError("official unary preprocess source_identity must be a mapping")
        self.transform = transform
        self.__nwm_source_identity__ = dict(source_identity)

    def __call__(self, image: Any) -> Any:
        return self.transform(_coerce_official_image(image))


def _normalize_symbol_descriptor(value: Any, *, role: str) -> dict[str, Any]:
    """Normalize one expected official symbol descriptor.

    A symbol is deliberately identified by its import module and qualified
    name, rather than merely by the file containing it.  This is what makes a
    same-file wrapper fail the admission check while still allowing the real
    ``planning_eval``/``datasets`` symbols to be selected parametrically.
    ``relative_path`` and ``sha256`` are optional refinements; the observed
    callable identity always records both for the receipt.
    """

    if isinstance(value, str):
        raw = value.strip()
        if ":" not in raw:
            raise ValueError(
                f"official NWM expected {role} symbol must use module:qualname"
            )
        module, qualname = raw.split(":", 1)
        descriptor: dict[str, Any] = {"module": module, "qualname": qualname}
    elif isinstance(value, Mapping):
        descriptor = dict(value)
        symbol = descriptor.get("symbol")
        if symbol is not None:
            if not isinstance(symbol, str) or ":" not in symbol:
                raise ValueError(
                    f"official NWM expected {role} symbol must use module:qualname"
                )
            symbol_module, symbol_qualname = symbol.strip().split(":", 1)
            if "module" in descriptor and str(descriptor["module"]) != symbol_module:
                raise ValueError(f"official NWM expected {role} module disagrees with symbol")
            if "qualname" in descriptor and str(descriptor["qualname"]) != symbol_qualname:
                raise ValueError(f"official NWM expected {role} qualname disagrees with symbol")
            descriptor["module"] = symbol_module
            descriptor["qualname"] = symbol_qualname
    else:
        raise TypeError(f"official NWM expected {role} symbol must be a mapping or string")

    module = str(descriptor.get("module", "")).strip()
    qualname = str(descriptor.get("qualname", "")).strip()
    if not module or not qualname or module.startswith("<") or qualname.startswith("<"):
        raise ValueError(
            f"official NWM expected {role} symbol requires nonempty module and qualname"
        )
    result: dict[str, Any] = {
        "module": module,
        "qualname": qualname,
        "symbol": f"{module}:{qualname}",
    }
    relative_path = descriptor.get("relative_path")
    if relative_path is None:
        # ``path`` is accepted only as a repository-relative declaration.  An
        # absolute path would make an identity machine-specific.
        relative_path = descriptor.get("path")
    if relative_path is not None:
        relative_path = str(relative_path).strip()
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError(
                f"official NWM expected {role} relative_path must be nonempty and relative"
            )
        result["relative_path"] = relative_path
    if descriptor.get("sha256") is not None:
        digest = str(descriptor["sha256"]).lower().strip()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"official NWM expected {role} sha256 must be 64 hexadecimal characters")
        result["sha256"] = digest
    return result


def _normalize_chain_identity(value: Any) -> dict[str, Any]:
    """Require an explicit, parameterized planner/preprocessing chain identity."""

    if not isinstance(value, Mapping):
        raise ValueError(
            "official NWM expected_chain_identity must declare planner and preprocessing symbols"
        )
    result: dict[str, Any] = {}
    for raw_role, canonical_role in _CHAIN_ROLE_ALIASES.items():
        if canonical_role in result:
            continue
        if raw_role in value:
            result[canonical_role] = _normalize_symbol_descriptor(
                value[raw_role], role=canonical_role
            )
    for role in ("planner", "preprocessing"):
        if role not in result:
            raise ValueError(
                f"official NWM expected_chain_identity missing {role} symbol"
            )
    chain_name = value.get("chain")
    if chain_name is not None:
        chain_name = str(chain_name).strip()
        if not chain_name:
            raise ValueError("official NWM chain identity name must be nonempty")
        result["chain"] = chain_name
    return result


def _assert_symbol_matches(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, role: str
) -> None:
    """Compare every declared expected field with observed callable identity."""

    for key in ("module", "qualname", "relative_path", "sha256"):
        if key in expected and str(actual.get(key, "")) != str(expected[key]):
            raise ValueError(
                f"official NWM {role} symbol/chain identity mismatch for {key}: "
                f"expected {expected[key]!r}, got {actual.get(key)!r}"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_file_identity(
    path: str | Path | None,
    expected_sha256: str | None,
    *,
    label: str,
) -> tuple[str, str]:
    if path is None or str(path).strip() == "":
        raise ValueError(f"official NWM {label} path is required")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"official NWM {label} does not exist: {resolved}")
    if expected_sha256 is None or str(expected_sha256).strip() == "":
        raise ValueError(f"official NWM {label} SHA-256 is required")
    actual = _sha256_file(resolved)
    if str(expected_sha256).lower() != actual:
        raise ValueError(
            f"official NWM {label} SHA-256 mismatch: expected {expected_sha256!r}, got {actual!r}"
        )
    return str(resolved), actual


def _declared_source_identity(
    fn: Callable[..., Any],
    declared: Mapping[str, Any],
    *,
    role: str,
    repository: Path,
    source_commit: str,
) -> dict[str, str]:
    """Validate a descriptor attached to an official callable object.

    ``misc.transform`` is a ``Compose`` instance, so Python cannot locate its
    implementation with ``inspect.getsourcefile(transform)``.  An external
    factory may attach the identity of the owning official symbol through
    :class:`OfficialUnaryPreprocess`; this helper still verifies the declared
    repository-relative file and its exact blob bytes before accepting it.
    """

    module = str(declared.get("module", "")).strip()
    qualname = str(declared.get("qualname", "")).strip()
    relative = str(
        declared.get("relative_path", declared.get("path", ""))
    ).strip()
    digest = str(declared.get("sha256", "")).lower().strip()
    declared_commit = str(declared.get("source_commit", source_commit)).strip()
    if not module or not qualname or not relative:
        raise ValueError(f"official NWM {role} declared source identity is incomplete")
    if Path(relative).is_absolute() or relative.startswith("../"):
        raise ValueError(f"official NWM {role} declared source path must be relative")
    if declared_commit.lower() != source_commit.lower():
        raise ValueError(f"official NWM {role} declared source commit mismatch")
    source_path = (repository / relative).resolve()
    try:
        source_path.relative_to(repository.resolve())
    except ValueError as exc:
        raise ValueError(f"official NWM {role} declared source is outside repository") from exc
    if not source_path.is_file():
        raise FileNotFoundError(f"official NWM {role} declared source does not exist: {source_path}")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"official NWM {role} declared source SHA-256 is required")
    actual_sha = _sha256_file(source_path)
    if actual_sha != digest:
        raise ValueError(
            f"official NWM {role} declared source SHA-256 mismatch: "
            f"expected {digest!r}, got {actual_sha!r}"
        )
    try:
        subprocess.check_output(
            ["git", "-C", str(repository), "ls-files", "--error-unmatch", "--", relative],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        committed_blob = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", f"{source_commit}:{relative}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        working_blob = subprocess.check_output(
            ["git", "-C", str(repository), "hash-object", str(source_path)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"official NWM {role} declared source is not tracked at source_commit") from exc
    if committed_blob != working_blob:
        raise ValueError(f"official NWM {role} declared source differs from source_commit")
    return {
        "path": str(source_path),
        "relative_path": relative,
        "module": module,
        "qualname": qualname,
        "symbol": f"{module}:{qualname}",
        "sha256": actual_sha,
        "source_commit": source_commit,
    }


def _callable_source_identity(
    fn: Callable[..., Any] | None,
    *,
    role: str,
    repository: Path,
    source_commit: str,
) -> dict[str, str]:
    """Prove an injected callable is tracked source from the verified worktree."""
    if fn is None:
        raise ValueError(f"official NWM {role} callable is required")
    declared = getattr(fn, "__nwm_source_identity__", None)
    if declared is not None:
        if not isinstance(declared, Mapping):
            raise TypeError(f"official NWM {role} declared source identity must be a mapping")
        return _declared_source_identity(
            fn,
            declared,
            role=role,
            repository=repository,
            source_commit=source_commit,
        )
    if getattr(fn, "__name__", None) == "<lambda>":
        raise TypeError(f"official NWM {role} lambda has no admissible provenance")
    try:
        source_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
    except (OSError, TypeError) as exc:
        raise TypeError(f"official NWM {role} source is not locatable") from exc
    if not source_file:
        raise TypeError(f"official NWM {role} source is not locatable")
    source_path = Path(source_file).expanduser().resolve()
    try:
        relative = source_path.relative_to(repository)
    except ValueError as exc:
        raise ValueError(f"official NWM {role} source is outside verified repository") from exc
    rel = relative.as_posix()
    try:
        subprocess.check_output(
            ["git", "-C", str(repository), "ls-files", "--error-unmatch", "--", rel],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        committed_blob = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", f"{source_commit}:{rel}"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"official NWM {role} source is not tracked at source_commit") from exc
    file_sha = _sha256_file(source_path)
    if committed_blob != subprocess.check_output(
        ["git", "-C", str(repository), "hash-object", str(source_path)], text=True,
        stderr=subprocess.DEVNULL,
    ).strip():
        raise ValueError(f"official NWM {role} source file differs from source_commit")
    return {
        "path": str(source_path),
        "relative_path": rel,
        "module": str(getattr(fn, "__module__", "")),
        "qualname": str(getattr(fn, "__qualname__", "")),
        "symbol": f"{getattr(fn, '__module__', '')}:{getattr(fn, '__qualname__', '')}",
        "sha256": file_sha,
        "source_commit": source_commit,
    }


def _require_official_remote(repository: Path) -> None:
    try:
        remote = subprocess.check_output(
            ["git", "-C", str(repository), "remote", "get-url", "origin"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().lower()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("official NWM repository origin is unavailable") from exc
    normalized = remote.removesuffix("/").removesuffix(".git")
    if normalized not in {"https://github.com/facebookresearch/nwm", "git@github.com:facebookresearch/nwm"}:
        raise ValueError("official NWM repository origin is not facebookresearch/nwm")


class NWMOfficialChainAdapter:
    """Wrap official NWM preprocess → rollout/CEM → continuous command output."""

    provenance = "official_nwm_repository_preprocess_rollout_planner"

    def __init__(
        self,
        planner_fn: Callable[..., Any],
        *,
        action_adapter: NWMContinuousActionAdapter | None = None,
        preprocess_fn: Callable[..., Any] | None = None,
        checkpoint: str | Path | None = None,
        checkpoint_sha256: str | None = None,
        source_commit: str | None = None,
        official_config: str | Path | None = None,
        official_config_sha256: str | None = None,
        official_repository: str | Path | None = None,
        expected_chain_identity: Mapping[str, Any] | None = None,
        stop_rule: Mapping[str, Any] | None = None,
        planner_mode: str = "step",
        official_dataset_name: str | None = None,
        official_output_dir: str | Path | None = None,
        official_horizon: int = 8,
        official_context_size: int = 4,
        waypoint_spacing_m: float = 1.0,
    ) -> None:
        if not callable(planner_fn):
            raise TypeError("official NWM planner_fn must be callable")
        if source_commit is None or not re.fullmatch(r"[0-9a-fA-F]{40}", str(source_commit).strip()):
            raise ValueError("official NWM adapter requires a full 40-character source commit")
        if official_repository is None or str(official_repository).strip() == "":
            raise ValueError("official NWM adapter requires an official repository path")
        repository_path = Path(official_repository).expanduser().resolve()
        if not repository_path.is_dir():
            raise FileNotFoundError(f"official NWM repository does not exist: {repository_path}")
        try:
            repository_head = subprocess.check_output(
                ["git", "-C", str(repository_path), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError("official NWM repository is not a git worktree") from exc
        if repository_head.lower() != str(source_commit).strip().lower():
            raise ValueError(
                f"official NWM repository HEAD mismatch: expected {source_commit!r}, got {repository_head!r}"
            )
        source_commit = repository_head.lower()
        _require_official_remote(repository_path)
        chain_identity = _normalize_chain_identity(expected_chain_identity)
        planner_identity = _callable_source_identity(
            planner_fn, role="planner", repository=repository_path, source_commit=repository_head
        )
        preprocess_identity = _callable_source_identity(
            preprocess_fn, role="preprocess", repository=repository_path, source_commit=repository_head
        )
        _assert_symbol_matches(
            planner_identity, chain_identity["planner"], role="planner"
        )
        _assert_symbol_matches(
            preprocess_identity, chain_identity["preprocessing"], role="preprocessing"
        )
        checkpoint_path, checkpoint_digest = _require_file_identity(
            checkpoint, checkpoint_sha256, label="checkpoint"
        )
        config_path, config_digest = _require_file_identity(
            official_config, official_config_sha256, label="config"
        )
        self.planner_fn = planner_fn
        self.preprocess_fn = preprocess_fn
        self.action_adapter = action_adapter or NWMContinuousActionAdapter()
        self.checkpoint = checkpoint_path
        self.checkpoint_sha256 = checkpoint_digest
        self.checkpoint_identity = {
            "path": checkpoint_path,
            "bytes": Path(checkpoint_path).stat().st_size,
            "sha256": checkpoint_digest,
        }
        self.source_commit = str(source_commit)
        self.official_repository = str(repository_path)
        self.official_config = config_path
        self.official_config_sha256 = config_digest
        self.planner_identity = planner_identity
        self.preprocess_identity = preprocess_identity
        self.chain_identity = chain_identity
        self.stop_rule = dict(stop_rule or {})
        if planner_mode not in {"step", "official_batch"}:
            raise ValueError("planner_mode must be 'step' or 'official_batch'")
        if isinstance(official_horizon, bool) or not isinstance(official_horizon, int) or official_horizon <= 0:
            raise ValueError("official_horizon must be a positive integer")
        if isinstance(official_context_size, bool) or not isinstance(official_context_size, int) or official_context_size <= 0:
            raise ValueError("official_context_size must be a positive integer")
        spacing = float(waypoint_spacing_m)
        if not __import__("math").isfinite(spacing) or spacing <= 0.0:
            raise ValueError("waypoint_spacing_m must be positive and finite")
        if planner_mode == "official_batch":
            if official_dataset_name is None or not str(official_dataset_name).strip():
                raise ValueError("official_dataset_name is required for official_batch planner")
            if official_output_dir is None or not str(official_output_dir).strip():
                raise ValueError("official_output_dir is required for official_batch planner")
        self.planner_mode = planner_mode
        self.official_dataset_name = (
            str(official_dataset_name).strip() if official_dataset_name is not None else None
        )
        self.official_output_dir = (
            str(Path(official_output_dir).expanduser().resolve())
            if official_output_dir is not None
            else None
        )
        self.official_horizon = official_horizon
        self.official_context_size = official_context_size
        self.waypoint_spacing_m = spacing
        self.calls = 0
        # The Habitat runner appends the post-step observation to its public
        # history before the next policy call.  Keep our own bounded queue of
        # the RGB actually received at each call so the reset frame is not
        # lost and the runner's current frame is never appended twice.
        self._factual_rgb = deque(maxlen=self.official_context_size)
        self.last_command: tuple[float, float, float] | None = None
        self.last_plan: dict[str, Any] | None = None

    def reset(self, goal_rgb: Any) -> None:
        self.calls = 0
        self._factual_rgb.clear()
        self.last_command = None
        self.last_plan = None
        reset = getattr(self.planner_fn, "reset", None)
        if callable(reset):
            reset(goal_rgb)

    def _preprocess_pair(self, current_rgb: Any, goal_rgb: Any) -> tuple[Any, Any]:
        """Apply either the legacy pair transform or official unary transform."""

        if self.preprocess_fn is None:
            return current_rgb, goal_rgb
        if bool(getattr(self.preprocess_fn, "__nwm_unary__", False)):
            return self.preprocess_fn(current_rgb), self.preprocess_fn(goal_rgb)
        try:
            signature = inspect.signature(self.preprocess_fn)
            signature.bind(current_rgb, goal_rgb)
        except (TypeError, ValueError):
            # A one-argument callable (including torchvision.Compose) is the
            # official form.  Do not retry a failed transform invocation here;
            # signature binding is performed before calling it.
            return self.preprocess_fn(current_rgb), self.preprocess_fn(goal_rgb)
        return self.preprocess_fn(current_rgb, goal_rgb)

    @staticmethod
    def _tensor_first_xy(plan: Any) -> tuple[float, float]:
        """Extract the first row from a returned ``[B,H,2]`` cumulative plan."""

        value = plan
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        shape = getattr(plan, "shape", None)
        ndim = len(shape) if shape is not None else None
        if ndim == 3:
            value = value[0][0]
        elif ndim == 2:
            value = value[0]
        elif ndim is None:
            # Nested Python lists are accepted for the mock/factory contract.
            if len(value) == 0:
                raise ValueError("official NWM planner returned an empty action plan")
            first = value[0]
            if len(first) == 0:
                raise ValueError("official NWM planner returned an empty action plan")
            value = first[0] if isinstance(first[0], (list, tuple)) else first
        else:
            raise ValueError("official NWM planner action plan must have rank 2 or 3")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("official NWM planner first action must contain exactly two coordinates")
        return float(value[0]), float(value[1])

    def _command_from_official_plan(self, output: Any) -> tuple[float, float, float]:
        """Recover one continuous command from official cumulative plan output."""

        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError("official NWM planner must return (pred_actions, pred_yaw)")
        pred_actions, pred_yaw = output
        del pred_yaw  # official value is the complete-horizon yaw sum
        raw_dx, raw_dy = self._tensor_first_xy(pred_actions)
        # ``pred_actions`` is cumulative in NWM waypoint units.  Its first row
        # is therefore the first increment; only xy needs the registered metric
        # waypoint scale.  The official first yaw is atan2 of that increment.
        import math

        command = (
            raw_dx * self.waypoint_spacing_m,
            raw_dy * self.waypoint_spacing_m,
            math.atan2(raw_dy, raw_dx),
        )
        self.last_plan = {
            "output_type": "official_cumulative_xy_plus_total_yaw",
            "first_xy_waypoint_units": [raw_dx, raw_dy],
            "waypoint_spacing_m": self.waypoint_spacing_m,
            "first_yaw_rad": command[2],
            "horizon": self.official_horizon,
        }
        return command

    @staticmethod
    def _command_from_output(output: Any) -> Any:
        if isinstance(output, Mapping):
            for key in ("continuous_action", "command", "action_continuous"):
                if key in output:
                    return output[key]
            if all(key in output for key in ("dx", "dy", "dyaw")):
                return output
        return output

    def _official_inputs(self, current: Any, goal: Any, history: Any) -> tuple[Any, ...]:
        """Build official planner tensors from factual RGB history."""

        import torch

        # ``history`` is intentionally not replayed here.  In the Habitat
        # runner it already contains the RGB returned by the preceding
        # ``env.step``; replaying it would duplicate the current frame and
        # discard the reset frame.  The adapter queue is populated only from
        # real ``current`` observations crossing this call boundary.
        del history
        self._factual_rgb.append(_snapshot_factual_image(current))
        factual_frames = list(self._factual_rgb)
        if not factual_frames:
            raise ValueError("official NWM context cannot be empty")
        # At reset there are fewer than four factual frames.  Repeating the
        # first real frame is an explicit context initialization, not imagined
        # data.  Subsequent cycles always use only the newest real records.
        factual_frames = [factual_frames[0]] * (
            self.official_context_size - len(factual_frames)
        ) + factual_frames
        context = torch.stack([self.preprocess_fn(frame) for frame in factual_frames])
        target = self.preprocess_fn(goal)
        if context.ndim != 4:
            raise ValueError("official NWM preprocessor must return [3,H,W] tensors")
        if target.ndim != 3:
            raise ValueError("official NWM goal preprocessor must return [3,H,W] tensor")
        context = context.unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
        idxs = torch.tensor([self.calls - 1], dtype=torch.long)
        gt_actions = torch.zeros(
            (1, self.official_horizon, 3), dtype=context.dtype, device=context.device
        )
        return (
            self.official_output_dir,
            self.official_dataset_name,
            idxs,
            context,
            target,
            gt_actions,
            self.official_horizon,
        )

    def act(self, current_rgb: Any, goal_rgb: Any, history: Any):
        self.calls += 1
        if self.planner_mode == "official_batch":
            # The official planner consumes a four-frame tensor and a batched
            # goal, rather than the runner's three-argument step ABI.
            output = self.planner_fn(*self._official_inputs(current_rgb, goal_rgb, history))
            command = self._command_from_official_plan(output)
        else:
            current, goal = self._preprocess_pair(current_rgb, goal_rgb)
            output = self.planner_fn(current, goal, history)
            command = self._command_from_output(output)
        # External stop is deliberately fixed by this adapter's recorded rule;
        # it cannot inspect Habitat success/measurements during act().
        # ``None`` delegates the decision to the decoder's configured external
        # rule (including command_threshold).  Passing False explicitly would
        # bypass ``should_stop`` and silently disable that threshold.
        stop = True if bool(self.stop_rule.get("always_stop", False)) else None
        stop_threshold = self.stop_rule.get("command_threshold")
        action = self.action_adapter.decode(
            command,
            stop=stop,
            stop_threshold=stop_threshold,
            stop_reason=self.stop_rule.get("reason"),
        )
        self.last_command = self.action_adapter.conversions[-1].command
        return action

    def close(self) -> None:
        close = getattr(self.planner_fn, "close", None)
        if callable(close):
            close()

    def provenance_record(self) -> dict[str, Any]:
        return {
            "adapter": self.__class__.__name__,
            "provenance": self.provenance,
            "official_source_commit": self.source_commit,
            "official_repository": self.official_repository,
            "checkpoint": dict(self.checkpoint_identity),
            "checkpoint_sha256": self.checkpoint_sha256,
            "official_config": self.official_config,
            "official_config_sha256": self.official_config_sha256,
            "preprocessing": dict(self.preprocess_identity),
            "planner": dict(self.planner_identity),
            "chain_identity": {
                key: dict(value) if isinstance(value, Mapping) else value
                for key, value in self.chain_identity.items()
            },
            "planner_adapter": {
                "mode": self.planner_mode,
                "dataset_name": self.official_dataset_name,
                "context_size": self.official_context_size,
                "horizon": self.official_horizon,
                "waypoint_spacing_m": self.waypoint_spacing_m,
                "first_output_rule": (
                    "pred_actions[:,0] and atan2(first_y,first_x)"
                    if self.planner_mode == "official_batch"
                    else None
                ),
                "last_plan": dict(self.last_plan) if self.last_plan is not None else None,
            },
            "stop_rule": dict(self.stop_rule),
            "action_conversion": self.action_adapter.conversion_summary(),
        }


# Short alias used by a few evaluation configs.
NWMPolicyAdapter = NWMOfficialChainAdapter

__all__ = ["NWMOfficialChainAdapter", "NWMPolicyAdapter", "OfficialUnaryPreprocess"]
