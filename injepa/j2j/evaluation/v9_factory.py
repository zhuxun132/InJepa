"""Checkpoint-backed V9 ImageGoal evaluation seam.

This module is deliberately an orchestration layer.  It reuses the existing
``jepa.JEPA``, ``ProposalJEPA``, factual-memory helpers, and ``J2JController``
without copying their math.  The factory is strict about model assets: a
missing proposal/Q/S checkpoint is reported as ``V9_QTS_UNAVAILABLE`` rather
than being replaced by a random or Direct policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import inspect
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from j2j.adapter import ActionId, Raw4Adapter
from j2j.controller import ControlStatus, ControlThresholds, J2JController
from j2j.data.keys import frame_key
from j2j.memory import (
    FactualMemory,
    FactualRecord,
    MemoryConfig,
    PreEvictionView,
    start_memory,
    update_after_observation,
)
from j2j.memory_objective import FactualViewKey, collate_factual_views
from j2j.proposal import ProposalJEPA

from .v9_adapter import V9PolicyAdapter


_TRAJECTORY_DOMAIN = b"J2J_EVAL_TRAJECTORY_V1\x00"
_PROPOSAL_FIELDS = (
    "latent_dim",
    "grid_side",
    "hidden_dim",
    "heads",
    "ffn_dim",
    "selector_depth",
    "future_depth",
    "dropout",
    "modes",
    "horizon",
    "global_seed",
)
_PROPOSAL_ARCHITECTURE_FIELDS = (
    "latent_dim",
    "grid_side",
    "hidden_dim",
    "heads",
    "ffn_dim",
    "selector_depth",
    "future_depth",
    "dropout",
)


class V9FactoryError(RuntimeError):
    """A malformed or incompatible V9 evaluation asset."""


class V9FactoryPending(V9FactoryError):
    """A required future-stage asset has not been admitted yet."""

    def __init__(self, status: str, message: str) -> None:
        self.status = str(status)
        super().__init__(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(path: str | Path | None, expected: str | None, *, label: str) -> dict[str, Any]:
    if path is None or not str(path).strip():
        raise ValueError(f"{label} path is required")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if expected is None or not str(expected).strip():
        raise ValueError(f"{label} SHA-256 is required")
    expected_hex = str(expected).lower().strip()
    if len(expected_hex) != 64 or any(c not in "0123456789abcdef" for c in expected_hex):
        raise ValueError(f"{label} SHA-256 must be 64 hexadecimal characters")
    actual = _sha256(resolved)
    if actual != expected_hex:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_hex!r}, got {actual!r}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": actual}


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _v9_mapping(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("v9")
    if not isinstance(value, Mapping):
        raise V9FactoryPending("V9_CONFIG_UNAVAILABLE", "config.v9 mapping is required")
    return value


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def _state_mapping(payload: Any, *, label: str) -> Mapping[str, Tensor]:
    """Extract a state dict without accepting opaque serialized modules."""

    if not isinstance(payload, Mapping):
        raise V9FactoryError(f"{label} checkpoint must contain a mapping")
    candidates: list[Mapping[str, Any]] = []
    for key in ("state_dict", "model_state", "ema", "weights"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    candidates.append(payload)
    for candidate in candidates:
        if candidate and all(isinstance(key, str) and isinstance(value, Tensor) for key, value in candidate.items()):
            return candidate  # type: ignore[return-value]
    raise V9FactoryError(f"{label} checkpoint has no tensor state mapping")


def _proposal_state(payload: Any) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise V9FactoryError("V9 proposal checkpoint must contain a mapping")
    for key in ("proposal_state", "proposal", "state_dict", "model_state"):
        value = payload.get(key)
        if isinstance(value, Mapping) and value:
            candidate = value
            if all(isinstance(k, str) and isinstance(v, Tensor) for k, v in candidate.items()):
                return candidate  # type: ignore[return-value]
    candidate = _state_mapping(payload, label="V9 proposal")
    # Some generic wrappers save the proposal under a ``proposal.`` prefix.
    if candidate and all(key.startswith("proposal.") for key in candidate):
        return {key.removeprefix("proposal."): value for key, value in candidate.items()}
    return candidate


def _proposal_config(config: Mapping[str, Any]) -> dict[str, Any]:
    v9 = _v9_mapping(config)
    raw_marker = v9.get("proposal", config.get("proposal"))
    raw = raw_marker if raw_marker is not None else {}
    values = dict(_mapping(raw, name="v9.proposal")) if raw else {}
    if values:
        # A resolved V9 evaluation identity must bind every architecture field
        # explicitly.  Otherwise a historical partial mapping could silently
        # acquire the current defaults and load a checkpoint under a new ABI.
        missing = [name for name in _PROPOSAL_ARCHITECTURE_FIELDS if name not in values]
        if missing:
            raise V9FactoryError(
                "v9.proposal must declare a complete architecture; "
                f"missing {', '.join(missing)}"
            )
    # ``proposal.modes`` is the canonical K namespace. Accept the historical
    # top-level alias only when it is absent from the nested mapping, and
    # reject disagreement instead of silently loading a different mode axis.
    nested_modes = values.get("modes")
    top_modes = v9.get("modes", config.get("modes"))
    if nested_modes is not None and top_modes is not None:
        if type(nested_modes) is not int or type(top_modes) is not int or nested_modes != top_modes:
            raise V9FactoryError("v9.proposal.modes and top-level modes disagree")
    if nested_modes is None and top_modes is not None:
        values["modes"] = top_modes
    defaults: dict[str, Any] = {
        "latent_dim": 768,
        "grid_side": 6,
        # Keep evaluation's fallback aligned with the formal H4 production
        # recipe.  Explicit resolved/ablation configs still override each
        # field below, so compact toy checkpoints remain loadable when their
        # architecture is declared explicitly.
        "hidden_dim": 768,
        "heads": 16,
        "ffn_dim": 2048,
        "selector_depth": 3,
        "future_depth": 6,
        "dropout": 0.1,
        # K is a run-level positive-integer hyperparameter.  Keep the H4 main
        # recipe as the fallback when an older resolved config omits it; a
        # supplied config value still takes precedence below.
        "modes": 4,
        "horizon": v9.get("horizon", 4),
        "global_seed": config.get("seed", 0),
    }
    defaults.update({key: values[key] for key in _PROPOSAL_FIELDS if key in values})
    # Keep constructor types explicit when values came from YAML/OmegaConf.
    for key in _PROPOSAL_FIELDS:
        if key in ("dropout",):
            defaults[key] = float(defaults[key])
        else:
            defaults[key] = int(defaults[key])
    if defaults["latent_dim"] != 768:
        raise V9FactoryError(
            "v9.proposal latent_dim must remain 768 for the admitted V-JEPA ABI"
        )
    if defaults["grid_side"] != 6:
        raise V9FactoryError(
            "v9.proposal grid_side must remain 6 for the admitted V-JEPA ABI"
        )
    for name in ("hidden_dim", "heads", "ffn_dim", "selector_depth", "future_depth", "modes", "horizon"):
        if defaults[name] <= 0:
            raise V9FactoryError(f"v9.proposal {name} must be positive")
    if defaults["hidden_dim"] % defaults["heads"] != 0:
        raise V9FactoryError("v9.proposal hidden_dim must be divisible by heads")
    if not 0.0 <= defaults["dropout"] < 1.0:
        raise V9FactoryError("v9.proposal dropout must be in [0, 1)")
    return defaults


def _thresholds(config: Mapping[str, Any], horizon: int) -> ControlThresholds:
    v9 = _v9_mapping(config)
    raw = v9.get("thresholds", config.get("thresholds"))
    if raw is None:
        raise V9FactoryPending("V9_THRESHOLDS_UNAVAILABLE", "V9 control thresholds are required")
    values = _mapping(raw, name="v9.thresholds")
    try:
        path = torch.tensor(list(values["path"]), dtype=torch.float32)
        endpoint = torch.tensor(list(values["endpoint"]), dtype=torch.float32)
        stop = float(values["stop"])
    except (KeyError, TypeError, ValueError) as exc:
        raise V9FactoryError("V9 thresholds must provide path, endpoint, and stop") from exc
    if path.numel() != horizon or endpoint.numel() != horizon:
        raise ValueError("V9 threshold horizon does not match ProposalJEPA horizon")
    if not bool(torch.isfinite(path).all().item()) or not bool(torch.isfinite(endpoint).all().item()):
        raise ValueError("V9 path/endpoint thresholds must be finite")
    return ControlThresholds(path=path, endpoint=endpoint, stop=stop)


def _memory_config(config: Mapping[str, Any]) -> MemoryConfig:
    raw = _v9_mapping(config).get("memory", config.get("memory"))
    if raw is None:
        raise V9FactoryPending("V9_MEMORY_CONFIG_UNAVAILABLE", "V9 factual-memory config is required")
    values = _mapping(raw, name="v9.memory")
    try:
        capacity = int(values["capacity"])
        recent_window = int(values.get("recent_window", 2))
    except (KeyError, TypeError, ValueError) as exc:
        raise V9FactoryError("V9 memory config requires integer capacity") from exc
    return MemoryConfig(capacity=capacity, recent_window=recent_window)


def _model_from_config(config: Mapping[str, Any]) -> nn.Module:
    """Instantiate the already-resolved official INTACT model config."""

    v9 = _v9_mapping(config)
    model_path = v9.get("model_config") or v9.get("resolved_model_config")
    assets = config.get("assets")
    if model_path is None and isinstance(assets, Mapping):
        model_path = assets.get("v9_model_config")
    if model_path is None:
        raise V9FactoryPending(
            "V9_MODEL_CONFIG_UNAVAILABLE",
            "a resolved official INTACT model config is required",
        )
    path = Path(str(model_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V9 model config does not exist: {path}")
    try:
        from omegaconf import OmegaConf
        import hydra
    except ModuleNotFoundError as exc:  # pragma: no cover - server runtime dependency
        raise V9FactoryPending(
            "V9_RUNTIME_DEPENDENCY_UNAVAILABLE",
            "Hydra/OmegaConf is required for the official INTACT model config",
        ) from exc
    loaded = OmegaConf.load(path)
    model_cfg = loaded.get("model") if isinstance(loaded, Mapping) and loaded.get("model") is not None else loaded
    try:
        model = hydra.utils.instantiate(model_cfg)
    except Exception as exc:
        raise V9FactoryError(f"failed to instantiate official INTACT model from {path}") from exc
    if not isinstance(model, nn.Module):
        raise TypeError("official INTACT model config did not instantiate an nn.Module")
    return model


def _default_component_loader(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    proposal_checkpoint: Path,
    device: torch.device,
) -> Mapping[str, Any]:
    model = _model_from_config(config)
    model.load_state_dict(_state_mapping(_torch_load(checkpoint), label="V9 A/G/F"), strict=True)
    proposal = ProposalJEPA(**_proposal_config(config))
    proposal.load_state_dict(_proposal_state(_torch_load(proposal_checkpoint)), strict=True)
    model.to(device=device).eval().requires_grad_(False)
    proposal.to(device=device).eval().requires_grad_(False)
    return {"intact_model": model, "proposal": proposal}


def _invoke_component_loader(
    loader: Callable[..., Any],
    *,
    config: Mapping[str, Any],
    checkpoint: Path,
    proposal_checkpoint: Path,
    device: torch.device,
) -> Mapping[str, Any]:
    if not callable(loader):
        raise TypeError("component_loader must be callable")
    try:
        signature = inspect.signature(loader)
    except (TypeError, ValueError):
        result = loader(config, checkpoint, proposal_checkpoint, device)
    else:
        candidates = (
            ((), {"config": config, "checkpoint": checkpoint, "proposal_checkpoint": proposal_checkpoint, "device": device}),
            ((config, checkpoint, proposal_checkpoint, device), {}),
            ((config, checkpoint, proposal_checkpoint), {}),
            ((config, checkpoint), {}),
            ((config,), {}),
            ((), {}),
        )
        for args, kwargs in candidates:
            try:
                signature.bind(*args, **kwargs)
            except TypeError:
                continue
            result = loader(*args, **kwargs)
            break
        else:
            raise TypeError("component_loader has no supported signature")
    if not isinstance(result, Mapping):
        raise TypeError("component_loader must return a mapping")
    return result


def _rgb_tensor(image: Any) -> Tensor:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - numpy is runtime dependency
        raise V9FactoryPending("V9_RUNTIME_DEPENDENCY_UNAVAILABLE", "NumPy is required for RGB evaluation") from exc
    if isinstance(image, Tensor):
        tensor = image.detach().cpu()
        if tensor.dtype != torch.uint8:
            raise TypeError("V9 RGB tensor must use uint8 sensor values")
        if tensor.ndim == 3 and tensor.shape[0] == 3:
            return tensor.contiguous()
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            return tensor.permute(2, 0, 1).contiguous()
        raise ValueError("V9 RGB tensor must be [3,H,W] or [H,W,3]")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("V9 RGB input must have shape [H,W,3]")
    if array.dtype != np.uint8:
        raise TypeError("V9 RGB input must use uint8 sensor values")
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()


def _default_encoder_fn(config: Mapping[str, Any], device: torch.device) -> tuple[Callable[[Any], Tensor], dict[str, Any]]:
    v9 = _v9_mapping(config)
    visual = v9.get("visual", config.get("visual", {}))
    visual = _mapping(visual, name="v9.visual")
    source_root = visual.get("source_root")
    checkpoint = visual.get("checkpoint")
    checkpoint_sha = visual.get("checkpoint_sha256")
    whitening = visual.get("whitening") or visual.get("whitening_path")
    whitening_sha = visual.get("whitening_sha256")
    if not all(value is not None and str(value).strip() for value in (source_root, checkpoint, checkpoint_sha, whitening, whitening_sha)):
        raise V9FactoryPending(
            "V9_VISUAL_ASSETS_UNAVAILABLE",
            "V-JEPA source/checkpoint and training whitening identities are required",
        )
    from j2j.encoding.cache import apply_whitening, load_whitening
    from j2j.encoding.vjepa2 import encode_images, load_frozen_vjepa2

    visual_checkpoint = _verified_file(checkpoint, checkpoint_sha, label="V-JEPA checkpoint")
    whitening_path = _verified_file(whitening, whitening_sha, label="V9 whitening artifact")
    expected_bytes = int(visual.get("checkpoint_bytes", 1_664_223_428))
    encoder = load_frozen_vjepa2(
        source_root=source_root,
        checkpoint=visual_checkpoint["path"],
        expected_sha256=visual_checkpoint["sha256"],
        expected_bytes=expected_bytes,
    ).to(device=device).eval()
    transform = load_whitening(whitening_path["path"], expected_sha256=whitening_path["sha256"])

    def encode(image: Any) -> Tensor:
        rgb = _rgb_tensor(image)[None]
        encoded = encode_images(encoder, rgb)
        grid = encoded.grid[0].detach().cpu().numpy().astype("<f4", copy=True)
        # Whitening is intentionally the same fixed CPU transform used by the
        # released cache; no evaluation observation is written back to cache.
        z32 = apply_whitening(grid, transform)
        return torch.from_numpy(z32).contiguous()

    return encode, {
        "vjepa_checkpoint": visual_checkpoint,
        "whitening": whitening_path,
        "encoder_source_root": str(Path(str(source_root)).expanduser().resolve()),
    }


def _validate_grid(grid: Tensor, *, spatial: int, latent: int) -> Tensor:
    if not isinstance(grid, Tensor) or tuple(grid.shape) != (spatial, latent):
        raise ValueError(f"encoded V9 RGB grid must have shape [{spatial},{latent}]")
    if grid.dtype != torch.float32:
        grid = grid.to(dtype=torch.float32)
    grid = grid.detach().cpu().contiguous()
    if not bool(torch.isfinite(grid).all().item()):
        raise ValueError("encoded V9 RGB grid must be finite")
    return grid


class _V9Runtime:
    """Stateful factual-memory shell around the existing stateless controller."""

    def __init__(
        self,
        *,
        intact_model: nn.Module,
        proposal: nn.Module,
        controller: J2JController,
        encode_fn: Callable[[Any], Tensor],
        memory_config: MemoryConfig,
        device: torch.device,
        spatial: int,
        latent: int,
        identity: Mapping[str, Any],
    ) -> None:
        self.intact_model = intact_model
        self.proposal = proposal
        self.controller = controller
        self.encode_fn = encode_fn
        self.memory_config = memory_config
        self.device = device
        self.spatial = spatial
        self.latent = latent
        self.identity = dict(identity)
        self.memory: FactualMemory | None = None
        self.goal_grid: Tensor | None = None
        self.trajectory_key: bytes | None = None
        self.pending_action: ActionId | None = None

    def _encode(self, image: Any) -> Tensor:
        return _validate_grid(self.encode_fn(image), spatial=self.spatial, latent=self.latent)

    def reset(self, goal_rgb: Any) -> None:
        goal_grid = self._encode(goal_rgb)
        # This key is an episode-local identity only; it contains no predicted
        # state and is never used as a training/cache key.
        try:
            goal_bytes = _rgb_tensor(goal_rgb).numpy().tobytes(order="C")
        except Exception:
            goal_bytes = repr(goal_rgb).encode("utf-8")
        self.trajectory_key = hashlib.sha256(_TRAJECTORY_DOMAIN + goal_bytes).digest()
        self.goal_grid = goal_grid
        self.memory = None
        self.pending_action = None

    def _record(self, grid: Tensor, *, time: int, incoming: Tensor) -> FactualRecord:
        if self.trajectory_key is None:
            raise RuntimeError("V9 adapter must be reset before acting")
        return FactualRecord(
            grid=grid,
            incoming_raw4=incoming,
            time=time,
            frame_key=frame_key(self.trajectory_key, time),
            trajectory_key=self.trajectory_key,
        )

    @staticmethod
    def _view_key(state: FactualMemory | PreEvictionView, *, kind: str, deleted: bytes | None = None) -> FactualViewKey:
        if isinstance(state, FactualMemory):
            records = state.bank + state.recent + (state.current,)
        else:
            records = state.pool + state.recent + (state.current,)
        return FactualViewKey(
            trajectory_key=state.current.trajectory_key,
            origin_time=state.current.time,
            current_frame_key=state.current.frame_key,
            ordered_record_frame_keys=tuple(record.frame_key for record in records),
            view_kind=kind,
            deleted_frame_key=deleted,
        )

    def _score_pool(self, view: PreEvictionView) -> Tensor:
        key = self._view_key(view, kind="full")
        facts = collate_factual_views(
            ((key, view),), self.intact_model.action_encoder,
            device=self.device, precision="32-true",
        )
        score = self.proposal.score_records(
            facts.record_grid,
            facts.incoming_action_embedding,
            facts.record_age,
            facts.record_type,
            facts.record_valid,
        )
        if tuple(score.shape) != (1, len(view.pool) + len(view.recent) + 1):
            raise ValueError("V9 selector returned an unexpected record score shape")
        return score[0, : len(view.pool)].detach().to(device="cpu", dtype=torch.float32)

    def _advance(self, grid: Tensor) -> None:
        if self.memory is None:
            self.memory = start_memory(
                self._record(grid, time=0, incoming=Raw4Adapter.encode_bos()),
                self.memory_config,
            )
            return
        if self.pending_action is None:
            raise RuntimeError("V9 runtime received a new observation without a prior motion action")
        next_time = self.memory.current.time + 1
        observed = self._record(
            grid,
            time=next_time,
            incoming=Raw4Adapter.encode(self.pending_action),
        )
        scorer = self._score_pool if self.memory_config.capacity > 0 else None
        self.memory = update_after_observation(self.memory, observed, score_fn=scorer).memory
        self.pending_action = None

    def _facts_and_history(self) -> tuple[Any, Tensor, Tensor]:
        assert self.memory is not None and self.goal_grid is not None
        key = self._view_key(self.memory, kind="steady")
        facts = collate_factual_views(
            ((key, self.memory),), self.intact_model.action_encoder,
            device=self.device, precision="32-true",
        )
        records = self.memory.bank + self.memory.recent + (self.memory.current,)
        history_size = int(self.intact_model.predictor.pos_embedding.size(1))
        history = records[-history_size:]
        embedding_history = torch.stack([record.grid.mean(dim=0) for record in history])[None].to(self.device)
        raw4_history = torch.stack([record.incoming_raw4 for record in history])[None].to(self.device)
        return facts, embedding_history, raw4_history

    @torch.inference_mode()
    def act(self, current_rgb: Any, goal_rgb: Any, history: Any) -> ActionId:
        del goal_rgb, history
        if self.goal_grid is None or self.trajectory_key is None:
            raise RuntimeError("V9 adapter must be reset before acting")
        self._advance(self._encode(current_rgb))
        facts, embedding_history, raw4_history = self._facts_and_history()
        result = self.controller.step(
            facts=facts,
            goal_grid=self.goal_grid.to(self.device).unsqueeze(0),
            embedding_history=embedding_history,
            raw4_history=raw4_history,
        )
        if len(result.status) != 1:
            raise RuntimeError("V9 controller must run one evaluation row at a time")
        status = result.status[0]
        if status == ControlStatus.STOP:
            self.pending_action = None
            return ActionId.STOP
        if status != ControlStatus.ACTION or int(result.action_id[0].item()) < 1:
            raise RuntimeError(f"V9 controller produced no verified motion action: {status.value}")
        action = ActionId(int(result.action_id[0].item()))
        self.pending_action = action
        return action

    def close(self) -> None:
        self.memory = None
        self.goal_grid = None
        self.pending_action = None


class V9RuntimeAdapter(V9PolicyAdapter):
    """Runner-facing adapter backed by factual RGB encoding and J2JController."""

    provenance = "j2j_v9_official_vjepa_intact_controller"

    def __init__(self, runtime: _V9Runtime, *, checkpoint: Mapping[str, Any], proposal_checkpoint: Mapping[str, Any]) -> None:
        self.runtime = runtime
        super().__init__(
            runtime.act,
            reset_fn=runtime.reset,
            checkpoint=checkpoint["path"],
            checkpoint_sha256=checkpoint["sha256"],
        )
        self._proposal_checkpoint = dict(proposal_checkpoint)

    def close(self) -> None:
        self.runtime.close()

    def provenance_record(self) -> dict[str, Any]:
        record = super().provenance_record()
        record.update({
            "provenance": self.provenance,
            "proposal_checkpoint": self._proposal_checkpoint,
            "runtime_identity": self.runtime.identity,
        })
        return record


def build_v9_adapter(
    config: Mapping[str, Any],
    checkpoint: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    *,
    component_loader: Callable[..., Any] | None = None,
    encode_fn: Callable[[Any], Tensor] | None = None,
    device: str | torch.device | None = None,
) -> V9RuntimeAdapter:
    """Build a strict V9 adapter from final A/G/F and Q/S assets.

    ``component_loader`` and ``encode_fn`` are dependency-injection seams for
    deterministic unit tests or an already-loaded server runtime.  The normal
    path uses the official resolved INTACT config and V-JEPA/whitening loader.
    They do not alter controller math or provide a substitute for missing
    proposal weights.
    """

    config = _mapping(config, name="config")
    v9 = _v9_mapping(config)
    agf_identity = _verified_file(
        checkpoint or v9.get("checkpoint"),
        checkpoint_sha256 or v9.get("checkpoint_sha256"),
        label="V9 A/G/F checkpoint",
    )
    proposal_path = v9.get("proposal_checkpoint") or v9.get("q2_checkpoint")
    proposal_sha = v9.get("proposal_checkpoint_sha256") or v9.get("q2_checkpoint_sha256")
    if proposal_path is None or proposal_sha is None or not str(proposal_path).strip():
        raise V9FactoryPending(
            "V9_QTS_UNAVAILABLE",
            "final Q2/S2 Proposal checkpoint and SHA-256 are required before V9 evaluation",
        )
    proposal_identity = _verified_file(proposal_path, proposal_sha, label="V9 Q/S proposal checkpoint")
    runtime_device = torch.device(device or v9.get("device", "cpu"))
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise V9FactoryPending("V9_RUNTIME_DEVICE_UNAVAILABLE", f"requested V9 device is unavailable: {runtime_device}")

    components = (
        _invoke_component_loader(
            component_loader,
            config=config,
            checkpoint=Path(agf_identity["path"]),
            proposal_checkpoint=Path(proposal_identity["path"]),
            device=runtime_device,
        )
        if component_loader is not None
        else _default_component_loader(
            config,
            checkpoint=Path(agf_identity["path"]),
            proposal_checkpoint=Path(proposal_identity["path"]),
            device=runtime_device,
        )
    )
    intact_model = components.get("intact_model")
    proposal = components.get("proposal")
    if not isinstance(intact_model, nn.Module) or not isinstance(proposal, nn.Module):
        raise TypeError("component_loader must return nn.Module values for intact_model and proposal")
    intact_model.to(runtime_device).eval().requires_grad_(False)
    proposal.to(runtime_device).eval().requires_grad_(False)
    if not hasattr(intact_model, "action_encoder") or not hasattr(intact_model, "predictor"):
        raise TypeError("V9 INTACT model must expose action_encoder and predictor")
    latent: int | None = None
    # Proposal's public spatial/horizon objects are the authoritative shape
    # contract; avoid guessing latent dimensions from private layer names.
    try:
        spatial = int(proposal.future.spatial_position.shape[0])
        latent = int(proposal.input_proj.in_features)
        horizon = int(proposal.future.horizon_embedding.num_embeddings)
    except (AttributeError, TypeError, ValueError):
        # A dependency-injected test/server shell may expose the same public
        # contract as scalar attributes while delegating the real ProposalJEPA
        # object elsewhere.  The production loader always takes the branch
        # above, so this fallback does not weaken the official shape check.
        try:
            spatial = int(getattr(proposal, "spatial"))
            latent = int(getattr(proposal, "latent"))
            horizon = int(getattr(proposal, "horizon"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise TypeError("V9 proposal does not expose the ProposalJEPA shape contract") from exc
    thresholds = _thresholds(config, horizon)
    memory_config = _memory_config(config)
    encoder_identity: dict[str, Any] = {}
    if encode_fn is None:
        encode_fn, encoder_identity = _default_encoder_fn(config, runtime_device)
    if not callable(encode_fn):
        raise TypeError("V9 encode_fn must be callable")
    controller = J2JController(
        intact_model=intact_model,
        proposal=proposal,
        thresholds=thresholds,
    )
    identity = {
        "horizon": horizon,
        "spatial_tokens": spatial,
        "latent_dim": latent,
        "memory": {"capacity": memory_config.capacity, "recent_window": memory_config.recent_window},
        "thresholds": {
            "path": [float(value) for value in thresholds.path.tolist()],
            "endpoint": [float(value) for value in thresholds.endpoint.tolist()],
            "stop": float(thresholds.stop),
        },
        "encoder": encoder_identity,
    }
    runtime = _V9Runtime(
        intact_model=intact_model,
        proposal=proposal,
        controller=controller,
        encode_fn=encode_fn,
        memory_config=memory_config,
        device=runtime_device,
        spatial=spatial,
        latent=latent,
        identity=identity,
    )
    return V9RuntimeAdapter(
        runtime,
        checkpoint=agf_identity,
        proposal_checkpoint=proposal_identity,
    )


__all__ = [
    "V9FactoryError",
    "V9FactoryPending",
    "V9RuntimeAdapter",
    "build_v9_adapter",
]
