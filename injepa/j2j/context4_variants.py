"""Exact single-factor contracts for the four current Context4 variants."""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

import copy
import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import re
from types import MappingProxyType
from typing import Any

from torch import Tensor

from j2j.context4_data import PreparedJointTrajectory


FULL_VARIANT_ID = "context4_no_selector_spatial_joint_v1"
RAW_VARIANT_ID = "context4_raw_spatial_joint_v1"
RAW_PROTOCOL_SHA256 = "77709038a6e3075f93ffec910f33bfdc85a549943d47fdf09d93737ac24f776a"
STOCHASTIC_VARIANT_ID = "context4_stochastic_divided_raw_v1"
STOCHASTIC_PROTOCOL_SHA256 = "1e2bb42d6e37326e96a513931840289f70f247fac1cdfc0b7f80383db7c24114"
STABLE_STOCHASTIC_VARIANT_ID = "context4_stochastic_divided_raw_v2"
STABLE_STOCHASTIC_PROTOCOL_SHA256 = "38fd507b65210707b83001efcdcbe6f111edb6d40dd0da621bcebc336cbecc1c"
RECURRENT_STOCHASTIC_VARIANT_ID = "context4_stochastic_recurrent_raw_v1"
RECURRENT_STOCHASTIC_PROTOCOL_SHA256 = "d432ce2924283d74b08bd25f06e7e2c5dfe5017cebdfaaab0aec570eca3820d5"
FULL_TRANSFORM_SPEC_SHA256 = (
    "06e614c0bcdef6ef7fd99d7961670bf726faa62850ab705262284e3b460d1474"
)
FULL_LOSS_SPEC_SHA256 = (
    "592fcc6b54f7715ed95cb569a361903e42ede446063c3baea4828092e8a53ccb"
)
CONTEXT1_TRANSFORM_SPEC_SHA256 = (
    "efd4094e4441e72a28567a203b09bd841c822e4be1d75e82a5d1de60b6c8803b"
)
MEAN_REPEAT_TRANSFORM_SPEC_SHA256 = (
    "7c343fae66a4c9d296eb7c414251faed09b271c1227bef1676832b65994fc778"
)
NO_QG_LOSS_SPEC_SHA256 = (
    "0fc5703019c3345c003cc65986086a14a7da807718cbe966dbaab24316cd4575"
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY_FIELDS = frozenset(
    {"variant_id", "transform_spec_sha", "loss_spec_sha"}
)


@dataclass(frozen=True)
class VariantIdentity:
    variant_id: str
    transform_spec_sha: str
    loss_spec_sha: str


@dataclass(frozen=True)
class VariantContract:
    identity: VariantIdentity
    prepared_trajectory_transform: Callable[
        [PreparedJointTrajectory], PreparedJointTrajectory
    ]
    context_transform: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    runtime_grid_transform: Callable[[Tensor], Tensor]
    loss_weights: Mapping[str, float]


FULL_IDENTITY = VariantIdentity(
    FULL_VARIANT_ID,
    FULL_TRANSFORM_SPEC_SHA256,
    FULL_LOSS_SPEC_SHA256,
)
RAW_IDENTITY = VariantIdentity(RAW_VARIANT_ID, RAW_PROTOCOL_SHA256, RAW_PROTOCOL_SHA256)
STOCHASTIC_IDENTITY = VariantIdentity(STOCHASTIC_VARIANT_ID, STOCHASTIC_PROTOCOL_SHA256, STOCHASTIC_PROTOCOL_SHA256)
STABLE_STOCHASTIC_IDENTITY = VariantIdentity(STABLE_STOCHASTIC_VARIANT_ID, STABLE_STOCHASTIC_PROTOCOL_SHA256, STABLE_STOCHASTIC_PROTOCOL_SHA256)
RECURRENT_STOCHASTIC_IDENTITY = VariantIdentity(RECURRENT_STOCHASTIC_VARIANT_ID, RECURRENT_STOCHASTIC_PROTOCOL_SHA256, RECURRENT_STOCHASTIC_PROTOCOL_SHA256)
E12_ABLATION_IDENTITIES = MappingProxyType({
    branch: VariantIdentity(
        f"e12_recurrent_no_{branch}_v1", RECURRENT_STOCHASTIC_PROTOCOL_SHA256,
        hashlib.sha256(f"E12 loss ablation v1: remove {branch}; remaining weights unchanged; skip disabled actor; preserve data ledger".encode()).hexdigest(),
    ) for branch in ("qg", "g_local", "g_goal")
})
CROCO224_IDENTITY = VariantIdentity("e12_croco224_full_v1", RECURRENT_STOCHASTIC_PROTOCOL_SHA256, RECURRENT_STOCHASTIC_PROTOCOL_SHA256)
DINOV3_256_IDENTITY = VariantIdentity("e12_dinov3_256_full_v1", RECURRENT_STOCHASTIC_PROTOCOL_SHA256, RECURRENT_STOCHASTIC_PROTOCOL_SHA256)
RECURRENT_STOCHASTIC_IDENTITIES = frozenset({RECURRENT_STOCHASTIC_IDENTITY, *E12_ABLATION_IDENTITIES.values(), CROCO224_IDENTITY, DINOV3_256_IDENTITY})
STABLE_STOCHASTIC_IDENTITIES = frozenset({STABLE_STOCHASTIC_IDENTITY, *RECURRENT_STOCHASTIC_IDENTITIES})
STOCHASTIC_IDENTITIES = frozenset({STOCHASTIC_IDENTITY, *STABLE_STOCHASTIC_IDENTITIES})
RAW_IDENTITIES = frozenset({RAW_IDENTITY, *STOCHASTIC_IDENTITIES})
CONTEXT1_IDENTITY = VariantIdentity(
    "context1",
    CONTEXT1_TRANSFORM_SPEC_SHA256,
    FULL_LOSS_SPEC_SHA256,
)
MEAN_REPEAT_IDENTITY = VariantIdentity(
    "mean_repeat",
    MEAN_REPEAT_TRANSFORM_SPEC_SHA256,
    FULL_LOSS_SPEC_SHA256,
)
NO_QG_IDENTITY = VariantIdentity(
    "no_qg",
    FULL_TRANSFORM_SPEC_SHA256,
    NO_QG_LOSS_SPEC_SHA256,
)
_IDENTITIES = (
    FULL_IDENTITY,
    RAW_IDENTITY,
    STOCHASTIC_IDENTITY,
    STABLE_STOCHASTIC_IDENTITY,
    RECURRENT_STOCHASTIC_IDENTITY,
    CONTEXT1_IDENTITY,
    MEAN_REPEAT_IDENTITY,
    NO_QG_IDENTITY,
    *E12_ABLATION_IDENTITIES.values(),
    CROCO224_IDENTITY,
    DINOV3_256_IDENTITY,
)
_IDENTITY_BY_VALUES = {
    (value.variant_id, value.transform_spec_sha, value.loss_spec_sha): value
    for value in _IDENTITIES
}

FULL_LOSS_WEIGHTS = MappingProxyType(
    {"q": 1.0, "f": 1.0, "g_local": 0.1, "g_goal": 0.05, "qg": 0.1}
)
NO_QG_LOSS_WEIGHTS = MappingProxyType({**FULL_LOSS_WEIGHTS, "qg": 0.0})


def _identity_values(value: object) -> tuple[str, str, str]:
    if isinstance(value, VariantIdentity):
        return value.variant_id, value.transform_spec_sha, value.loss_spec_sha
    if not isinstance(value, Mapping):
        raise TypeError("variant identity must be a mapping or VariantIdentity")
    if set(value) != _IDENTITY_FIELDS:
        raise ValueError("variant identity must contain exactly three fields")
    result = tuple(value.get(name) for name in (
        "variant_id", "transform_spec_sha", "loss_spec_sha"
    ))
    if not isinstance(result[0], str) or not result[0]:
        raise ValueError("variant_id must be a nonempty string")
    for name, digest in zip(
        ("transform_spec_sha", "loss_spec_sha"), result[1:], strict=True
    ):
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"variant identity {name} must be a lowercase SHA-256")
    return result  # type: ignore[return-value]


def variant_identity_dict(identity: VariantIdentity) -> dict[str, str]:
    if not isinstance(identity, VariantIdentity):
        raise TypeError("identity must be a VariantIdentity")
    return {
        "variant_id": identity.variant_id,
        "transform_spec_sha": identity.transform_spec_sha,
        "loss_spec_sha": identity.loss_spec_sha,
    }


def mean_repeat_grid(grid: Tensor) -> Tensor:
    """Repeat each grid's spatial mean over the frozen 36-token axis."""

    if not isinstance(grid, Tensor):
        raise TypeError("grid must be a tensor")
    if grid.ndim < 2 or grid.shape[-2] < 1:
        raise ValueError("grid must end in nonempty [spatial,latent] axes")
    if not grid.is_floating_point():
        raise TypeError("grid must be floating point")
    return grid.mean(dim=-2, keepdim=True).repeat_interleave(
        int(grid.shape[-2]), dim=-2
    )


def _identity_grid(grid: Tensor) -> Tensor:
    if not isinstance(grid, Tensor):
        raise TypeError("grid must be a tensor")
    return grid.clone()


def _prepared_with_identity(
    prepared: PreparedJointTrajectory,
    *,
    transform_spec_sha: str,
) -> PreparedJointTrajectory:
    if not isinstance(prepared, PreparedJointTrajectory):
        raise TypeError("prepared trajectory transform requires PreparedJointTrajectory")
    observed = prepared.transform_spec_sha
    if observed == transform_spec_sha:
        return prepared
    if observed is not None:
        raise ValueError("prepared trajectory already carries a different transform")
    return replace(
        prepared,
        grids=prepared.grids.clone(),
        action_ids=prepared.action_ids.clone(),
        incoming_raw4=prepared.incoming_raw4.clone(),
        outgoing_raw4=prepared.outgoing_raw4.clone(),
        transform_spec_sha=transform_spec_sha,
    )


def _prepared_full(prepared: PreparedJointTrajectory) -> PreparedJointTrajectory:
    return _prepared_with_identity(
        prepared,
        transform_spec_sha=FULL_TRANSFORM_SPEC_SHA256,
    )


def _prepared_raw(prepared: PreparedJointTrajectory) -> PreparedJointTrajectory:
    return _prepared_with_identity(prepared, transform_spec_sha=RAW_PROTOCOL_SHA256)


def _prepared_stochastic(prepared: PreparedJointTrajectory) -> PreparedJointTrajectory:
    return _prepared_with_identity(prepared, transform_spec_sha=STOCHASTIC_PROTOCOL_SHA256)


def _prepared_stable_stochastic(prepared: PreparedJointTrajectory) -> PreparedJointTrajectory:
    return _prepared_with_identity(prepared, transform_spec_sha=STABLE_STOCHASTIC_PROTOCOL_SHA256)


def _prepared_recurrent_stochastic(prepared: PreparedJointTrajectory) -> PreparedJointTrajectory:
    return _prepared_with_identity(prepared, transform_spec_sha=RECURRENT_STOCHASTIC_PROTOCOL_SHA256)


def _prepared_context1(prepared: PreparedJointTrajectory) -> PreparedJointTrajectory:
    return _prepared_with_identity(
        prepared,
        transform_spec_sha=CONTEXT1_TRANSFORM_SPEC_SHA256,
    )


def _prepared_mean_repeat(
    prepared: PreparedJointTrajectory,
) -> PreparedJointTrajectory:
    if not isinstance(prepared, PreparedJointTrajectory):
        raise TypeError("prepared trajectory transform requires PreparedJointTrajectory")
    observed = prepared.transform_spec_sha
    if observed == MEAN_REPEAT_TRANSFORM_SPEC_SHA256:
        return prepared
    if observed is not None:
        raise ValueError("prepared trajectory already carries a different transform")
    return replace(
        prepared,
        grids=mean_repeat_grid(prepared.grids).detach(),
        action_ids=prepared.action_ids.clone(),
        incoming_raw4=prepared.incoming_raw4.clone(),
        outgoing_raw4=prepared.outgoing_raw4.clone(),
        transform_spec_sha=MEAN_REPEAT_TRANSFORM_SPEC_SHA256,
    )


def _prepared_no_qg(prepared: PreparedJointTrajectory) -> PreparedJointTrajectory:
    return _prepared_with_identity(
        prepared,
        transform_spec_sha=FULL_TRANSFORM_SPEC_SHA256,
    )


_CONTEXT_SCHEMAS = (
    {
        "context_grid": -3,
        "context_incoming_raw4": -2,
        "context_outgoing_raw4": -2,
        "context_age": -1,
        "context_type": -1,
        "context_valid": -1,
    },
    {
        "record_grid": -3,
        "incoming_raw4": -2,
        "outgoing_raw4": -2,
        "record_age": -1,
        "record_type": -1,
        "record_valid": -1,
    },
    {
        "grids": -3,
        "incoming": -2,
        "outgoing": -2,
        "ages": -1,
        "kinds": -1,
        "valid": -1,
    },
)


def _clone_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        raise TypeError("context must be a mapping")
    return {
        str(name): value.clone() if isinstance(value, Tensor) else copy.deepcopy(value)
        for name, value in context.items()
    }


def _identity_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    return _clone_context(context)


def _context1(context: Mapping[str, Any]) -> Mapping[str, Any]:
    result = _clone_context(context)
    schema = next(
        (candidate for candidate in _CONTEXT_SCHEMAS if set(candidate).issubset(result)),
        None,
    )
    if schema is None:
        raise KeyError("context is missing the six fixed-four fields")
    context_size: int | None = None
    for name, axis in schema.items():
        value = result[name]
        if not isinstance(value, Tensor):
            raise TypeError(f"context field {name} must be a tensor")
        if value.ndim < abs(axis):
            raise ValueError(f"context field {name} rank is invalid")
        size = int(value.shape[axis])
        if context_size is None:
            context_size = size
        elif size != context_size:
            raise ValueError("context fields disagree on the context axis")
    if context_size != 4:
        raise ValueError("Context1 requires a fixed four-slot context")
    for name, axis in schema.items():
        value = result[name]
        selection = [slice(None)] * value.ndim
        selection[axis] = slice(0, 3)
        value[tuple(selection)] = 0
    return result


def _contract(identity: VariantIdentity) -> VariantContract:
    if identity in RAW_IDENTITIES:
        prepared = (_prepared_recurrent_stochastic if identity in RECURRENT_STOCHASTIC_IDENTITIES else
                    _prepared_stable_stochastic if identity == STABLE_STOCHASTIC_IDENTITY else
                    _prepared_stochastic if identity == STOCHASTIC_IDENTITY else _prepared_raw)
        context = _identity_context
        runtime = _identity_grid
        weights = FULL_LOSS_WEIGHTS
    elif identity == CONTEXT1_IDENTITY:
        prepared = _prepared_context1
        context = _context1
        runtime = _identity_grid
        weights = FULL_LOSS_WEIGHTS
    elif identity == MEAN_REPEAT_IDENTITY:
        prepared = _prepared_mean_repeat
        context = _identity_context
        runtime = mean_repeat_grid
        weights = FULL_LOSS_WEIGHTS
    elif identity == NO_QG_IDENTITY:
        prepared = _prepared_no_qg
        context = _identity_context
        runtime = _identity_grid
        weights = NO_QG_LOSS_WEIGHTS
    else:
        prepared = _prepared_full
        context = _identity_context
        runtime = _identity_grid
        weights = FULL_LOSS_WEIGHTS
    for branch, ablation_identity in E12_ABLATION_IDENTITIES.items():
        if identity == ablation_identity:
            weights = MappingProxyType({**FULL_LOSS_WEIGHTS, branch: 0.0})
    return VariantContract(identity, prepared, context, runtime, weights)


def resolve_variant_contract(identity: object) -> VariantContract:
    """Resolve one exact allowlisted identity without touching RNG or runtime state."""

    values = _identity_values(identity)
    try:
        resolved = _IDENTITY_BY_VALUES[values]
    except KeyError as exc:
        raise ValueError("variant identity is not one of the four frozen rows") from exc
    return _contract(resolved)


__all__ = [
    "CONTEXT1_IDENTITY",
    "FULL_IDENTITY",
    "FULL_LOSS_SPEC_SHA256",
    "FULL_LOSS_WEIGHTS",
    "FULL_TRANSFORM_SPEC_SHA256",
    "FULL_VARIANT_ID",
    "MEAN_REPEAT_IDENTITY",
    "NO_QG_IDENTITY",
    "VariantContract",
    "VariantIdentity",
    "mean_repeat_grid",
    "resolve_variant_contract",
    "variant_identity_dict",
]
