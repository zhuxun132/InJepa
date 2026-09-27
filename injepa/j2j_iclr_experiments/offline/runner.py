"""Single-materialization traversal core for all offline one-factor arms."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from j2j.context4_data import prepare_joint_trajectory

from .transforms import build_single_factor_arm_plan


def _default_trajectory_key(item: Any, ordinal: int) -> Any:
    source_item = getattr(item, "source_item", None)
    trajectory = getattr(source_item, "canonical_trajectory", None)
    key = getattr(trajectory, "canonical_trajectory_key", None)
    if key is not None:
        return key
    if isinstance(item, Mapping) and "trajectory" in item:
        return item["trajectory"]
    return ordinal


def _rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        result = tuple(value)
        if all(isinstance(row, Mapping) for row in result):
            return result
    raise TypeError("evaluate_fn must return a mapping or iterable of mappings")


def run_one_pass(
    trajectories: Iterable[Any],
    *,
    arms: Sequence[str],
    prepare_fn: Callable[[Any], Any] = prepare_joint_trajectory,
    evaluate_fn: Callable[[Any, str], Any],
    trajectory_key_fn: Callable[[Any, int], Any] | None = None,
    trajectory_budget: int | None = None,
    row_sink: Callable[[Mapping[str, Any]], None] | None = None,
    retain_rows: bool = True,
    population_ordinal_by_key: Mapping[Any, int] | None = None,
) -> dict[str, Any]:
    """Prepare each base trajectory once, then evaluate baseline + one factors.

    ``row_sink`` permits incremental persistence.  With ``retain_rows=False``
    the traversal never retains completed scalar rows in memory.
    """

    if not callable(prepare_fn) or not callable(evaluate_fn):
        raise TypeError("prepare_fn and evaluate_fn must be callable")
    if trajectory_key_fn is not None and not callable(trajectory_key_fn):
        raise TypeError("trajectory_key_fn must be callable")
    if row_sink is not None and not callable(row_sink):
        raise TypeError("row_sink must be callable")
    if not isinstance(retain_rows, bool):
        raise TypeError("retain_rows must be boolean")
    if population_ordinal_by_key is not None and not isinstance(
        population_ordinal_by_key, Mapping
    ):
        raise TypeError("population_ordinal_by_key must be a mapping")
    if trajectory_budget is not None and (
        isinstance(trajectory_budget, bool)
        or not isinstance(trajectory_budget, int)
        or trajectory_budget <= 0
    ):
        raise ValueError("trajectory_budget must be a positive integer")

    arm_plan = build_single_factor_arm_plan(tuple(arm for arm in arms if arm != "baseline"))
    supplied = tuple(arms)
    if supplied != arm_plan:
        raise ValueError("arms must equal baseline plus unique single-factor arms")

    retained: list[Mapping[str, Any]] = []
    observed_key_set: set[Any] = set()
    observed_keys: list[Any] = []
    trajectory_count = 0
    emitted_rows = 0
    evaluations = 0
    unavailable_rows = 0
    for ordinal, item in enumerate(trajectories):
        if trajectory_budget is not None and trajectory_count >= trajectory_budget:
            break
        key = (
            trajectory_key_fn(item, ordinal)
            if trajectory_key_fn is not None
            else _default_trajectory_key(item, ordinal)
        )
        if key in observed_key_set:
            raise ValueError("a base trajectory may be materialized at most once")
        observed_key_set.add(key)
        observed_keys.append(key)
        if population_ordinal_by_key is None:
            population_ordinal = None
        else:
            if key not in population_ordinal_by_key:
                raise ValueError("trajectory key is absent from the global population plan")
            population_ordinal = population_ordinal_by_key[key]
            if type(population_ordinal) is not int or population_ordinal < 0:
                raise ValueError("global population ordinal must be non-negative")
        prepared = prepare_fn(item)
        trajectory_count += 1
        emission_ordinal = 0
        for arm in arm_plan:
            evaluations += 1
            for row in _rows(evaluate_fn(prepared, arm)):
                emitted = dict(row)
                if population_ordinal is not None:
                    emitted["population_ordinal"] = population_ordinal
                    emitted["emission_ordinal"] = emission_ordinal
                emission_ordinal += 1
                if emitted.get("available") is False or emitted.get("status") == "UNAVAILABLE":
                    unavailable_rows += 1
                if row_sink is not None:
                    row_sink(emitted)
                if retain_rows:
                    retained.append(emitted)
                emitted_rows += 1

    return {
        "rows": retained,
        "observed_trajectory_keys": tuple(observed_keys),
        "counts": {
            "trajectories": trajectory_count,
            "materializations": trajectory_count,
            "evaluations": evaluations,
            "rows": emitted_rows,
            "unavailable_rows": unavailable_rows,
        },
        "arms": arm_plan,
    }


__all__ = ["run_one_pass"]
