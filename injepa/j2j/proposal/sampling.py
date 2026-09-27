"""Stateless occurrence/candidate noise; never consumes the dropout RNG."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
from torch import Tensor


def keyed_normal(
    sample_keys: Sequence[bytes], *, seed: int, namespace: str, samples: int,
    latent_dim: int, sample_offset: int = 0, device, dtype,
) -> Tensor:
    """Return [B,K,L] from independent SHA256-addressed CPU substreams.

    Candidate index is part of the address, so batching/order/K do not change
    any existing sample. CPU float64 is the canonical draw; device/dtype are
    transport choices. Duplicate keys intentionally replay the same sample.
    """
    for name, value, minimum in (("seed", seed, 0), ("samples", samples, 1),
                                 ("latent_dim", latent_dim, 1),
                                 ("sample_offset", sample_offset, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be nonempty text")
    if isinstance(sample_keys, (bytes, str)) or not isinstance(sample_keys, Sequence):
        raise TypeError("sample_keys must be a sequence of byte strings")
    if not sample_keys or any(not isinstance(key, bytes) or not key for key in sample_keys):
        raise ValueError("sample_keys must contain nonempty byte strings")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("sampling dtype must be floating point")
    domain = b"j2j.keyed-normal.v1"
    def framed(value: bytes) -> bytes:
        return len(value).to_bytes(8, "big") + value
    prefix = domain + framed(str(seed).encode()) + framed(namespace.encode("utf-8"))
    rows = []
    for key in sample_keys:
        candidates = []
        for index in range(sample_offset, sample_offset + samples):
            address = prefix + framed(key) + framed(str(index).encode())
            sub_seed = int.from_bytes(hashlib.sha256(address).digest()[:8], "big")
            generator = torch.Generator(device="cpu").manual_seed(sub_seed)
            candidates.append(torch.randn(latent_dim, generator=generator, dtype=torch.float64))
        rows.append(torch.stack(candidates))
    return torch.stack(rows).to(device=device, dtype=dtype)
