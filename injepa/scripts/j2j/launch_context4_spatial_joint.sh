#!/usr/bin/env bash
set -Eeuo pipefail

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the caller-selected GPU set}"
: "${NPROC_PER_NODE:?set NPROC_PER_NODE to the number of visible training ranks}"
: "${CONFIG_PATH:?set CONFIG_PATH to a fully resolved context4 config}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to the new run output directory}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$PROJECT_ROOT"
exec torchrun \
  --standalone \
  --nproc-per-node="$NPROC_PER_NODE" \
  scripts/train_j2j_context4.py \
  --config "$CONFIG_PATH" \
  --output-root "$OUTPUT_ROOT"
