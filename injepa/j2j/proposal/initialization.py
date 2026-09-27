"""Per-FQN deterministic initialization for ProposalJEPA."""

from __future__ import annotations

import hashlib
import math
import struct
import unicodedata

import torch
from torch import nn


_SEMANTIC_ROOT = "j2j.proposal.ProposalJEPA"


def _semantic_fqn(relative_key: str) -> str:
    return unicodedata.normalize("NFC", f"{_SEMANTIC_ROOT}.{relative_key}")


def _fqn_seed(global_seed: int, semantic_fqn: str) -> int:
    normalized = unicodedata.normalize("NFC", semantic_fqn).encode("utf-8")
    preimage = (
        b"J2J_Q_INIT_V1\x00"
        + struct.pack("<Q", global_seed)
        + struct.pack("<Q", len(normalized))
        + normalized
    )
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") % (2**63)


def _initialization_rule(relative_key: str) -> tuple[str, int | None]:
    if relative_key in {
        "selector.type_embedding.weight",
        "future.type_embedding.weight",
    }:
        return "type_normal", None
    if relative_key.endswith((".bias", "in_proj_bias")):
        return "zero", None
    if any(
        f".{name}.weight" in f".{relative_key}"
        for name in ("self_norm", "cross_norm", "ffn_norm", "final_norm")
    ):
        return "one", None

    pieces = relative_key.split(".")
    if len(pieces) >= 5 and pieces[0] == "selector" and pieces[1] == "blocks":
        layer = int(pieces[2]) + 1
        tail = ".".join(pieces[3:])
        if tail in {"self_attn.out_proj.weight", "ffn.fc2.weight"}:
            return "selector_scaled", layer
    if len(pieces) >= 5 and pieces[0] == "future" and pieces[1] == "blocks":
        layer = int(pieces[2]) + 1
        tail = ".".join(pieces[3:])
        if tail in {
            "self_attn.out_proj.weight",
            "cross_attn.out_proj.weight",
            "ffn.fc2.weight",
        }:
            return "future_scaled", layer
    return "trunc_normal", None


def initialize_proposal_(proposal: nn.Module, *, global_seed: int) -> None:
    """Initialize every parameter once from its semantic FQN-local generator."""

    ordered = sorted(
        proposal.named_parameters(),
        key=lambda item: _semantic_fqn(item[0]).encode("utf-8"),
    )
    with torch.no_grad():
        for relative_key, parameter in ordered:
            rule, layer = _initialization_rule(relative_key)
            if rule == "zero":
                parameter.zero_()
                continue
            if rule == "one":
                parameter.fill_(1.0)
                continue

            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _fqn_seed(global_seed, _semantic_fqn(relative_key))
            )
            if rule == "type_normal":
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=1e-6,
                    generator=generator,
                )
            else:
                nn.init.trunc_normal_(
                    parameter,
                    mean=0.0,
                    std=0.02,
                    a=-2.0,
                    b=2.0,
                    generator=generator,
                )
                if rule == "selector_scaled":
                    assert layer is not None
                    parameter.div_(math.sqrt(2 * layer))
                elif rule == "future_scaled":
                    assert layer is not None
                    parameter.div_(math.sqrt(3 * layer))
