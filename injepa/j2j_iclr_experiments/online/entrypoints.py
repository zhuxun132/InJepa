"""Protocol-complete, fail-closed STOP calibration producer."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from j2j.authority import sha256_file
from j2j.receipts import canonical_json_bytes

from .stop import (
    derive_stop_calibration,
    load_stop_pair_ledger,
    recompute_cache_bound_stop_rows,
    validate_stop_calibration_receipt,
    validate_stop_calibration_provenance,
)

def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"BLOCKED: STOP {name} must be a mapping")
    return value


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"BLOCKED: STOP {name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"BLOCKED: STOP {name} root must be a mapping")
    return value


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"BLOCKED: STOP identity file is unavailable: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _load_provenance(
    *,
    stop: Mapping[str, Any],
    statistics: Mapping[str, Any],
    ledger_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path_value = stop.get("calibration_provenance_path")
    expected_sha = stop.get("calibration_provenance_sha256")
    if not path_value or not expected_sha:
        raise RuntimeError(
            "BLOCKED: STOP calibration provenance path/SHA is required"
        )
    path = Path(str(path_value)).expanduser().resolve()
    identity = _file_identity(path)
    if identity["sha256"] != expected_sha:
        raise ValueError("BLOCKED: STOP calibration provenance SHA mismatch")
    provenance = _read_json(path, name="calibration provenance")
    expected_bootstrap = {
        "cluster_unit": "building",
        "replicates": int(statistics["replicates"]),
        "bit_generator": "PCG64",
        "seed": int(statistics["seed"]),
        "ci_level": float(statistics["ci_level"]),
        "quantile_method": "linear",
    }
    try:
        validate_stop_calibration_provenance(
            provenance,
            ledger_identity=ledger_identity,
            expected_bootstrap=expected_bootstrap,
            expected_numpy_version=np.__version__,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BLOCKED: STOP calibration provenance is invalid: {exc}") from exc
    return provenance, identity


def _write_atomic_once(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite STOP receipt: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def run_stop_calibration(config: Mapping[str, Any]) -> int:
    stop = _mapping(config.get("stop"), name="config.stop")
    statistics = _mapping(config.get("statistics"), name="config.statistics")
    ledger_value = stop.get("calibration_ledger_path")
    expected_sha = stop.get("calibration_ledger_sha256")
    if not ledger_value or not expected_sha:
        raise RuntimeError(
            "BLOCKED: STOP calibration ledger path/SHA is absent; "
            "never_stop remains PARTIAL"
        )
    ledger_path = Path(str(ledger_value)).expanduser().resolve()
    ledger_identity = _file_identity(ledger_path)
    if ledger_identity["sha256"] != expected_sha:
        raise ValueError("BLOCKED: STOP calibration ledger SHA mismatch")
    provenance, provenance_identity = _load_provenance(
        stop=stop,
        statistics=statistics,
        ledger_identity=ledger_identity,
    )

    rows = load_stop_pair_ledger(ledger_path)
    rows, census_value = recompute_cache_bound_stop_rows(
        rows, provenance=provenance
    )
    census = dict(census_value)
    if census.get("partition") != "project-dev":
        raise ValueError("BLOCKED: STOP threshold selection requires project-dev")
    derived = derive_stop_calibration(
        rows,
        census=census,
        bootstrap_parameters=_mapping(provenance["bootstrap"], name="bootstrap"),
    )
    receipt = {
        "schema": "STOP_CALIBRATION_V1",
        "status": "FORMAL",
        "interpretation": "released-data location-match transfer proxy",
        "calibration_provenance": provenance_identity,
        "provenance": provenance,
        "ledger": ledger_identity,
        **derived,
    }
    validate_stop_calibration_receipt(receipt)
    destination = (
        Path(str(_mapping(config.get("output"), name="config.output")["root"]))
        / "stop_calibration"
        / "STOP_CALIBRATION_V1.json"
    )
    _write_atomic_once(destination, receipt)
    return 0


__all__ = ["run_stop_calibration"]
