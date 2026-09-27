"""Single-pass, bounded qualitative records for the current Context4 suite."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any

from j2j.receipts import canonical_json_bytes
from j2j_iclr_experiments.common.config import QUALITATIVE_REQUIRED_STRATA


_G_VIEWS = frozenset({"local", "goal", "terminal"})
QUALITATIVE_STRATA = QUALITATIVE_REQUIRED_STRATA


def _stratum(row: Mapping[str, Any]) -> str | None:
    if row.get("available", True) is False or row.get("arm") != "baseline":
        return None
    component = row.get("component")
    view = row.get("view")
    if component == "g" and view in _G_VIEWS and type(row.get("correct")) is bool:
        outcome = "correct" if row["correct"] else "incorrect"
        return f"g:{view}:{outcome}"
    if component == "ranking" and view == "full" and type(row.get("winner")) is bool:
        outcome = "winner" if row["winner"] else "nonwinner"
        return f"ranking:full:{outcome}"
    return None


def _scalar_projection(
    row: Mapping[str, Any],
    *,
    stratum: str,
) -> dict[str, str | int | float | bool | None]:
    """Drop tensors/containers and retain only finite scalar evidence."""

    result: dict[str, str | int | float | bool | None] = {"stratum": stratum}
    for key, value in row.items():
        if not isinstance(key, str):
            raise TypeError("qualitative record keys must be strings")
        if value is None or isinstance(value, (str, bool)) or type(value) is int:
            result[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("qualitative records must contain finite scalars")
            result[key] = value
        # Tensor/array/container fields are intentionally not retained.
    return result


class StreamingQualitativeCollector:
    """Keep canonical-hash-min records under global and per-stratum caps."""

    def __init__(self, *, limit: int, per_stratum_limit: int) -> None:
        for name, value in (("limit", limit), ("per_stratum_limit", per_stratum_limit)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.limit = limit
        self.per_stratum_limit = per_stratum_limit
        self._records: list[
            tuple[tuple[str, bytes], dict[str, str | int | float | bool | None]]
        ] = []

    @property
    def resident_record_count(self) -> int:
        return len(self._records)

    def consider(self, raw: Mapping[str, Any]) -> None:
        if not isinstance(raw, Mapping):
            raise TypeError("qualitative rows must be mappings")
        stratum = _stratum(raw)
        if stratum is None:
            return
        record = _scalar_projection(raw, stratum=stratum)
        payload = canonical_json_bytes(record)
        key = (hashlib.sha256(payload).hexdigest(), payload)

        same_stratum = [
            (index, existing_key)
            for index, (existing_key, existing) in enumerate(self._records)
            if existing["stratum"] == stratum
        ]
        if len(same_stratum) >= self.per_stratum_limit:
            worst_index, worst_key = max(same_stratum, key=lambda item: item[1])
            if key >= worst_key:
                return
            self._records.pop(worst_index)
        self._records.append((key, record))
        while len(self._records) > self.limit:
            by_stratum: dict[str, list[int]] = {}
            for index, (_existing_key, existing) in enumerate(self._records):
                by_stratum.setdefault(str(existing["stratum"]), []).append(index)
            removable = [
                index
                for indices in by_stratum.values()
                for index in sorted(indices, key=lambda value: self._records[value][0])[1:]
            ]
            if not removable:
                raise ValueError(
                    "qualitative limit cannot retain one record for every observed stratum"
                )
            worst_index = max(removable, key=lambda index: self._records[index][0])
            self._records.pop(worst_index)

    def records(self) -> tuple[dict[str, str | int | float | bool | None], ...]:
        return tuple(
            record
            for _key, record in sorted(
                self._records,
                key=lambda item: (str(item[1]["stratum"]), item[0]),
            )
        )


__all__ = ["QUALITATIVE_STRATA", "StreamingQualitativeCollector"]
