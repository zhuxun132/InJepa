#!/usr/bin/env python3
"""Thin CLI for the parameterized context-four spatial joint runner."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch


def _claim_torchrun_local_cuda() -> torch.Tensor | None:
    raw_local_rank = os.environ.get("LOCAL_RANK")
    if raw_local_rank is None or not torch.cuda.is_available():
        return None
    device = torch.device("cuda", int(raw_local_rank))
    torch.cuda.set_device(device)
    return torch.empty(1, dtype=torch.uint8, device=device)


_EARLY_CUDA_CLAIM = _claim_torchrun_local_cuda()

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import j2j.context4_training as training


def _load_config(path: Path) -> Mapping[str, object]:
    raw = path.read_text(encoding="utf-8")
    value = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(value, Mapping):
        raise TypeError("context4 config root must be a mapping")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    admitted = training.validate_context4_config(
        _load_config(args.config),
        actual_world_size=world_size,
    )
    training.run_context4_training(config=admitted, output_root=args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
