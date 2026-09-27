"""Canonical public scientific identities for the J2J training runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
from importlib import metadata
import json
import math
import platform
import re
import struct
from typing import Any

import torch
from torch import Tensor, nn

from j2j.adapter import ActionId, Raw4Adapter


_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SCHEDULER_CONSTRUCTOR_FQN = (
    "stable_pretraining.optim.lr_scheduler.create_scheduler"
)


def canonical_json_bytes(value: object) -> bytes:
    """Encode a public record as compact, deterministic UTF-8 JSON plus LF."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _tensor_raw_bytes(tensor: Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous()
    return cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def public_state_dict_sha256(
    module_or_state_dict: nn.Module | Mapping[str, Tensor],
) -> str:
    """Hash the public state-dict tensor ABI without inspecting object internals."""
    if isinstance(module_or_state_dict, nn.Module):
        state = module_or_state_dict.state_dict()
    elif isinstance(module_or_state_dict, Mapping):
        state = module_or_state_dict
    else:
        raise TypeError("expected a module or public state-dict mapping")

    if any(not isinstance(name, str) for name in state):
        raise TypeError("state-dict keys must be strings")

    payload = bytearray(b"J2J_PUBLIC_STATE_DICT_V1\x00")
    for name in sorted(state, key=lambda item: item.encode("utf-8")):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError("state-dict values must be tensors")
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("utf-8")
        raw = _tensor_raw_bytes(tensor)
        payload.extend(struct.pack("<I", len(name_bytes)))
        payload.extend(name_bytes)
        payload.extend(struct.pack("<I", len(dtype_bytes)))
        payload.extend(dtype_bytes)
        payload.extend(struct.pack("<I", tensor.ndim))
        for dimension in tensor.shape:
            payload.extend(struct.pack("<Q", dimension))
        payload.extend(struct.pack("<Q", len(raw)))
        payload.extend(raw)
    return hashlib.sha256(payload).hexdigest()


def _project_state_tree(value: object) -> Any:
    if isinstance(value, Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "raw_sha256": hashlib.sha256(_tensor_raw_bytes(value)).hexdigest(),
        }
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical state mappings require string keys")
            projected[key] = _project_state_tree(child)
        return projected
    if isinstance(value, (list, tuple)):
        return [_project_state_tree(child) for child in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical state scalars must be finite")
        return value
    raise TypeError("unsupported canonical state value")


def canonical_state_tree_sha256(value: object) -> str:
    """Hash the supported public builtin/Tensor state-tree projection."""
    return hashlib.sha256(canonical_json_bytes(_project_state_tree(value))).hexdigest()


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_revision(name: str, value: object) -> None:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 40-hex revision")


def _dataclass_sha256(value: object, *, domain: str) -> str:
    record = {field.name: getattr(value, field.name) for field in fields(value)}
    return canonical_state_tree_sha256({"domain": domain, **record})


@dataclass(frozen=True)
class DependencyEnvironment:
    """Public dependency versions and source digests affecting numerics."""

    python_version: str
    torch_version: str
    cuda_runtime_version: str | None
    stable_pretraining_version: str
    environment_lock_sha256: str
    stable_pretraining_module_sha256: str
    stable_pretraining_scheduler_sha256: str
    scheduler_constructor_fqn: str
    exclude_bias_norm: bool

    def __post_init__(self) -> None:
        for name in (
            "python_version",
            "torch_version",
            "stable_pretraining_version",
        ):
            _require_nonempty_string(name, getattr(self, name))
        if self.cuda_runtime_version is not None:
            _require_nonempty_string("cuda_runtime_version", self.cuda_runtime_version)
        for name in (
            "environment_lock_sha256",
            "stable_pretraining_module_sha256",
            "stable_pretraining_scheduler_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if self.scheduler_constructor_fqn != _SCHEDULER_CONSTRUCTOR_FQN:
            raise ValueError("scheduler_constructor_fqn does not match the public factory")
        if self.exclude_bias_norm is not False:
            raise ValueError("exclude_bias_norm must be false for the single AdamW group")

    def sha256(self) -> str:
        return _dataclass_sha256(
            self, domain="J2J_DEPENDENCY_ENVIRONMENT_V1"
        )


@dataclass(frozen=True)
class NumericalRuntime:
    """Public numerical runtime fields used for same-runtime comparisons."""

    python_version: str
    torch_version: str
    stable_pretraining_version: str
    device_type: str
    precision: str
    autocast_enabled: bool
    autocast_dtype: str | None
    rank: int
    world_size: int
    logical_cuda_device_count: int
    deterministic_algorithms: bool
    cudnn_benchmark: bool | None
    cudnn_deterministic: bool | None
    matmul_allow_tf32: bool | None
    cudnn_allow_tf32: bool | None
    environment_ledger_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "python_version",
            "torch_version",
            "stable_pretraining_version",
        ):
            _require_nonempty_string(name, getattr(self, name))
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("device_type must be cpu or cuda")
        if self.precision not in {"32-true", "bf16-mixed"}:
            raise ValueError("precision must be 32-true or bf16-mixed")
        expected_autocast = self.precision == "bf16-mixed"
        expected_dtype = str(torch.bfloat16) if expected_autocast else None
        if self.autocast_enabled is not expected_autocast:
            raise ValueError("autocast_enabled does not match precision")
        if self.autocast_dtype != expected_dtype:
            raise ValueError("autocast_dtype does not match precision")
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if type(self.world_size) is not int or self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if (
            type(self.logical_cuda_device_count) is not int
            or self.logical_cuda_device_count < 0
        ):
            raise ValueError("logical_cuda_device_count must be non-negative")
        if type(self.deterministic_algorithms) is not bool:
            raise ValueError("deterministic_algorithms must be bool")
        cuda_fields = (
            self.cudnn_benchmark,
            self.cudnn_deterministic,
            self.matmul_allow_tf32,
            self.cudnn_allow_tf32,
        )
        if self.device_type == "cpu":
            if self.logical_cuda_device_count != 0 or any(
                value is not None for value in cuda_fields
            ):
                raise ValueError("CPU runtime must not report CUDA numeric fields")
        elif any(type(value) is not bool for value in cuda_fields):
            raise ValueError("CUDA runtime flags must be bool")
        _require_sha256(
            "environment_ledger_sha256", self.environment_ledger_sha256
        )

    @classmethod
    def capture(
        cls,
        *,
        device_type: str,
        precision: str,
        rank: int,
        world_size: int,
        logical_cuda_device_count: int,
        environment_ledger_sha256: str,
    ) -> "NumericalRuntime":
        if precision == "32-true":
            autocast_enabled = False
            autocast_dtype = None
        elif precision == "bf16-mixed":
            autocast_enabled = True
            autocast_dtype = str(torch.bfloat16)
        else:
            raise ValueError("precision must be 32-true or bf16-mixed")

        if device_type == "cpu":
            cuda_fields: tuple[bool | None, ...] = (None, None, None, None)
        elif device_type == "cuda":
            cuda_fields = (
                bool(torch.backends.cudnn.benchmark),
                bool(torch.backends.cudnn.deterministic),
                bool(torch.backends.cuda.matmul.allow_tf32),
                bool(torch.backends.cudnn.allow_tf32),
            )
        else:
            raise ValueError("device_type must be cpu or cuda")

        return cls(
            python_version=platform.python_version(),
            torch_version=str(torch.__version__),
            stable_pretraining_version=metadata.version("stable-pretraining"),
            device_type=device_type,
            precision=precision,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
            rank=rank,
            world_size=world_size,
            logical_cuda_device_count=logical_cuda_device_count,
            deterministic_algorithms=(
                torch.are_deterministic_algorithms_enabled()
            ),
            cudnn_benchmark=cuda_fields[0],
            cudnn_deterministic=cuda_fields[1],
            matmul_allow_tf32=cuda_fields[2],
            cudnn_allow_tf32=cuda_fields[3],
            environment_ledger_sha256=environment_ledger_sha256,
        )

    def sha256(self) -> str:
        return _dataclass_sha256(self, domain="J2J_NUMERICAL_RUNTIME_V1")


@dataclass(frozen=True)
class ScientificIdentity:
    """Public identities whose changes alter the scientific interpretation."""

    intact_base_commit: str
    design_bundle_sha256: str
    j2j_code_commit: str
    j2j_code_ledger_sha256: str
    resolved_config_sha256: str
    environment_ledger_sha256: str
    numerical_runtime_sha256: str
    streamvln_revision: str
    data_manifest_sha256: str
    split_ledger_sha256: str
    cache_manifest_sha256: str
    action_encoder_artifact_sha256: str
    action_encoder_state_sha256: str
    raw4_adapter_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "intact_base_commit",
            "j2j_code_commit",
            "streamvln_revision",
        ):
            _require_revision(name, getattr(self, name))
        for name in (
            "design_bundle_sha256",
            "j2j_code_ledger_sha256",
            "resolved_config_sha256",
            "environment_ledger_sha256",
            "numerical_runtime_sha256",
            "data_manifest_sha256",
            "split_ledger_sha256",
            "cache_manifest_sha256",
            "action_encoder_artifact_sha256",
            "action_encoder_state_sha256",
            "raw4_adapter_sha256",
        ):
            _require_sha256(name, getattr(self, name))

    def sha256(self) -> str:
        return _dataclass_sha256(self, domain="J2J_SCIENTIFIC_IDENTITY_V1")


def _raw4_values(tensor: Tensor) -> list[int]:
    return [int(value) for value in tensor.tolist()]


def _raw4_tie_order() -> list[str]:
    logits = torch.zeros(4, dtype=torch.float32)
    order: list[str] = []
    for _ in ActionId:
        action = ActionId(int(Raw4Adapter.decode_logits(logits).item()))
        order.append(action.name)
        logits[int(action)] = float("-inf")
    return order


def raw4_adapter_ledger() -> dict[str, object]:
    """Return the canonical ledger derived from the existing adapter API."""
    ledger: dict[str, object] = {"BOS": _raw4_values(Raw4Adapter.encode_bos())}
    for action in ActionId:
        ledger[action.name] = {
            "action_id": int(action),
            "raw4": _raw4_values(Raw4Adapter.encode(action)),
        }
    ledger["exact_logit_tie_order"] = _raw4_tie_order()
    return ledger


def raw4_adapter_sha256() -> str:
    """Hash the canonical ledger of the existing raw-four adapter."""
    return hashlib.sha256(canonical_json_bytes(raw4_adapter_ledger())).hexdigest()
