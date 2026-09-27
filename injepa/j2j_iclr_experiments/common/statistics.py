"""Protocol-exact aggregation and resampling for Context4 experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


_CLOSED_LOOP_METRICS = ("success", "spl")


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _probability(name: str, value: object, *, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    lower_ok = number > 0.0 if strict else number >= 0.0
    upper_ok = number < 1.0 if strict else number <= 1.0
    if not math.isfinite(number) or not lower_ok or not upper_ok:
        boundary = "between zero and one" if strict else "in [0,1]"
        raise ValueError(f"{name} must be finite and {boundary}")
    return number


def _finite_value(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric and finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"row {key!r} must be a non-empty string")
    return value


def _percentile_interval(
    values: np.ndarray, *, ci_level: float
) -> tuple[float, float]:
    tail = (1.0 - ci_level) / 2.0
    low, high = np.quantile(values, (tail, 1.0 - tail), method="linear")
    return float(low), float(high)


def offline_hierarchical_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    replicates: int,
    seed: int,
    ci_level: float,
) -> dict[str, object]:
    """Aggregate row -> trajectory -> building and bootstrap both clusters."""
    replicate_count = _positive_integer("replicates", replicates)
    random_seed = _seed(seed)
    level = _probability("ci_level", ci_level, strict=True)
    if not isinstance(value_key, str) or not value_key:
        raise ValueError("value_key must be a non-empty string")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("offline rows must be a non-empty sequence")

    seed_presence = ["training_seed" in row for row in rows if isinstance(row, Mapping)]
    if seed_presence and any(seed_presence) and not all(seed_presence):
        raise ValueError("offline rows may not mix labeled and unlabeled training seeds")
    has_training_seeds = bool(seed_presence and all(seed_presence))
    grouped_by_seed: dict[
        int | None, dict[str, dict[str, list[float]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every offline row must be a mapping")
        training_seed: int | None = None
        if has_training_seeds:
            candidate_seed = row.get("training_seed")
            if type(candidate_seed) is not int:
                raise TypeError("offline training_seed must be an integer, not bool")
            training_seed = candidate_seed
        building = _required_string(row, "building")
        trajectory = _required_string(row, "trajectory")
        grouped_by_seed[training_seed][building][trajectory].append(
            _finite_value(f"row {value_key}", row.get(value_key))
        )

    seed_order = sorted(grouped_by_seed) if has_training_seeds else [None]
    buildings_by_seed: dict[int | None, list[str]] = {}
    trajectory_means_by_seed: dict[int | None, dict[str, np.ndarray]] = {}
    seed_estimates: dict[int | None, float] = {}
    for training_seed in seed_order:
        grouped = grouped_by_seed[training_seed]
        buildings = sorted(grouped)
        buildings_by_seed[training_seed] = buildings
        trajectory_means_by_seed[training_seed] = {
            building: np.asarray(
                [
                    float(np.mean(grouped[building][trajectory]))
                    for trajectory in sorted(grouped[building])
                ],
                dtype=np.float64,
            )
            for building in buildings
        }
        seed_estimates[training_seed] = float(
            np.mean(
                [
                    float(
                        np.mean(trajectory_means_by_seed[training_seed][building])
                    )
                    for building in buildings
                ]
            )
        )
    estimate = float(np.mean([seed_estimates[item] for item in seed_order]))

    rng = np.random.Generator(np.random.PCG64(random_seed))
    bootstrap = np.empty(replicate_count, dtype=np.float64)
    for replicate in range(replicate_count):
        sampled_seed_estimates: list[float] = []
        for training_seed in seed_order:
            buildings = buildings_by_seed[training_seed]
            sampled_buildings = rng.integers(0, len(buildings), size=len(buildings))
            sampled_building_means: list[float] = []
            for building_index in sampled_buildings:
                values = trajectory_means_by_seed[training_seed][
                    buildings[int(building_index)]
                ]
                sampled_trajectories = rng.integers(0, len(values), size=len(values))
                sampled_building_means.append(
                    float(np.mean(values[sampled_trajectories]))
                )
            sampled_seed_estimates.append(float(np.mean(sampled_building_means)))
        bootstrap[replicate] = float(np.mean(sampled_seed_estimates))
    ci_low, ci_high = _percentile_interval(bootstrap, ci_level=level)

    unique_buildings = {
        building
        for buildings in buildings_by_seed.values()
        for building in buildings
    }
    result: dict[str, object] = {
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "row_count": len(rows),
        "trajectory_count": sum(
            len(grouped_by_seed[training_seed][building])
            for training_seed in seed_order
            for building in buildings_by_seed[training_seed]
        ),
        "building_count": len(unique_buildings),
        "replicates": replicate_count,
        "seed": random_seed,
        "ci_level": level,
        "quantile_method": "linear",
        "degenerate_replicates": bool(np.all(bootstrap == bootstrap[0])),
    }
    if has_training_seeds:
        result["seed_estimates"] = {
            int(training_seed): seed_estimates[training_seed]
            for training_seed in seed_order
            if training_seed is not None
        }
        result["seed_count"] = len(seed_order)
    return result


def closed_loop_point_estimates(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    """Micro-average episodes within seed, then weight training seeds equally."""
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("closed-loop rows must be a non-empty sequence")
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    seen: set[tuple[str, int, str, str]] = set()
    checkpoints: dict[tuple[str, int], str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every closed-loop row must be a mapping")
        variant = _required_string(row, "variant")
        training_seed = row.get("training_seed")
        if type(training_seed) is not int:
            raise TypeError("training_seed must be an integer, not bool")
        checkpoint = _required_string(row, "checkpoint_sha")
        building = _required_string(row, "building")
        episode = _required_string(row, "episode_key")
        row_key = (variant, training_seed, building, episode)
        if row_key in seen:
            raise ValueError("closed-loop ledger contains a duplicate episode row")
        seen.add(row_key)
        group_key = (variant, training_seed)
        if group_key in checkpoints and checkpoints[group_key] != checkpoint:
            raise ValueError("one variant/training-seed group has multiple checkpoints")
        checkpoints[group_key] = checkpoint
        for metric in _CLOSED_LOOP_METRICS:
            _probability(metric, row.get(metric))
        grouped[group_key].append(row)

    variants = sorted({variant for variant, _ in grouped})
    result: dict[tuple[str, str], dict[str, object]] = {}
    for variant in variants:
        seeds = sorted(seed for candidate, seed in grouped if candidate == variant)
        for metric in _CLOSED_LOOP_METRICS:
            seed_estimates = {
                seed: float(
                    np.mean(
                        [float(row[metric]) for row in grouped[(variant, seed)]]
                    )
                )
                for seed in seeds
            }
            result[(variant, metric)] = {
                "estimate": float(np.mean(list(seed_estimates.values()))),
                "seed_estimates": seed_estimates,
                "seed_count": len(seeds),
                "episode_count": sum(len(grouped[(variant, seed)]) for seed in seeds),
            }
    return result


def stable_holm(
    p_values: Mapping[tuple[str, str], float], *, alpha: float = 0.05
) -> list[dict[str, object]]:
    """Apply Holm step-down correction with a fully stable tie order."""
    family_alpha = _probability("alpha", alpha, strict=True)
    if not isinstance(p_values, Mapping) or not p_values:
        raise ValueError("p_values must be a non-empty mapping")
    validated: list[tuple[float, str, str]] = []
    for key, p_value in p_values.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(part, str) or not part for part in key)
        ):
            raise TypeError("p-value keys must be (baseline_id, metric_id) strings")
        validated.append((_probability("p", p_value), key[0], key[1]))
    validated.sort(key=lambda item: (item[0], item[1], item[2]))

    total = len(validated)
    still_rejecting = True
    adjusted_so_far = 0.0
    result: list[dict[str, object]] = []
    for index, (p_value, baseline, metric) in enumerate(validated):
        threshold = family_alpha / (total - index)
        rejected = still_rejecting and p_value <= threshold
        if not rejected:
            still_rejecting = False
        adjusted_so_far = max(adjusted_so_far, (total - index) * p_value)
        result.append(
            {
                "baseline_id": baseline,
                "metric_id": metric,
                "p": p_value,
                "holm_rank": index + 1,
                "holm_threshold": threshold,
                "holm_adjusted_p": min(1.0, adjusted_so_far),
                "rejected": rejected,
            }
        )
    return result


def _formal_group(
    grouped: Mapping[tuple[str, int], list[Mapping[str, object]]],
    *,
    variant: str,
    seeds: tuple[int, ...],
    episode_keys: tuple[tuple[str, str], ...],
) -> bool:
    for training_seed in seeds:
        rows = grouped.get((variant, training_seed), [])
        observed_keys = [
            (row.get("building"), row.get("episode_key")) for row in rows
        ]
        if observed_keys != list(episode_keys):
            return False
        if any(row.get("status") != "FORMAL" for row in rows):
            return False
        checkpoints = {row.get("checkpoint_sha") for row in rows}
        if len(checkpoints) != 1 or not isinstance(next(iter(checkpoints)), str):
            return False
        for row in rows:
            try:
                for metric in _CLOSED_LOOP_METRICS:
                    _probability(metric, row.get(metric))
            except (TypeError, ValueError):
                return False
    return True


def paired_comparison_family(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_variants: Sequence[str],
    expected_seeds: Sequence[int],
    expected_episode_keys: Sequence[tuple[str, str]],
    replicates: int,
    seed: int,
    alpha: float,
    seed_field: str = "evaluation_seed",
    ci_level: float = 0.95,
) -> dict[str, object]:
    """Reuse paired within-building episode resampling for named comparisons."""
    replicate_count = _positive_integer("replicates", replicates)
    random_seed = _seed(seed)
    family_alpha = _probability("alpha", alpha, strict=True)
    level = _probability("ci_level", ci_level, strict=True)
    if seed_field not in ("evaluation_seed", "training_seed"):
        raise ValueError("comparison seed field must identify evaluation or training seeds")
    variants = tuple(expected_variants)
    seeds = tuple(expected_seeds)
    episode_keys = tuple(expected_episode_keys)
    if len(variants) < 2 or len(set(variants)) != len(variants):
        raise ValueError("paired family requires a reference and unique controls")
    if any(not isinstance(variant, str) or not variant for variant in variants):
        raise TypeError("expected variants must be non-empty strings")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(item) is not int for item in seeds):
        raise ValueError("expected seeds must be unique integers")
    if (
        not episode_keys
        or len(set(episode_keys)) != len(episode_keys)
        or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or any(not isinstance(part, str) or not part for part in key)
            for key in episode_keys
        )
    ):
        raise ValueError("expected episode keys must be unique (building, episode) strings")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("confirmatory rows must be a sequence")

    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("every confirmatory row must be a mapping")
        variant = row.get("variant")
        training_seed = row.get(seed_field)
        if variant not in variants:
            raise ValueError("confirmatory row has an unexpected variant")
        if training_seed not in seeds or type(training_seed) is not int:
            raise ValueError(f"confirmatory row has an unexpected {seed_field}")
        grouped[(variant, training_seed)].append(row)

    strata: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in episode_keys:
        strata[key[0]].append(key)
    rng = np.random.Generator(np.random.PCG64(random_seed))
    resampled_ledgers: list[tuple[tuple[str, str], ...]] = []
    for _ in range(replicate_count):
        sampled: list[tuple[str, str]] = []
        for building in sorted(strata):
            keys = strata[building]
            indices = rng.integers(0, len(keys), size=len(keys))
            sampled.extend(keys[int(index)] for index in indices)
        resampled_ledgers.append(tuple(sampled))

    reference = variants[0]
    reference_formal = _formal_group(
        grouped, variant=reference, seeds=seeds, episode_keys=episode_keys
    )
    comparisons: dict[tuple[str, str], dict[str, object]] = {}
    p_values: dict[tuple[str, str], float] = {}
    for baseline in variants[1:]:
        baseline_formal = _formal_group(
            grouped, variant=baseline, seeds=seeds, episode_keys=episode_keys
        )
        formal = reference_formal and baseline_formal
        for metric in _CLOSED_LOOP_METRICS:
            key = (baseline, metric)
            if not formal:
                comparisons[key] = {
                    "status": "NONFORMAL",
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p": 1.0,
                    "reason": "incomplete, reordered, nonfinite, or non-FORMAL fixed ledger",
                }
                p_values[key] = 1.0
                continue

            reference_values = {
                training_seed: {
                    (str(row["building"]), str(row["episode_key"])): float(row[metric])
                    for row in grouped[(reference, training_seed)]
                }
                for training_seed in seeds
            }
            baseline_values = {
                training_seed: {
                    (str(row["building"]), str(row["episode_key"])): float(row[metric])
                    for row in grouped[(baseline, training_seed)]
                }
                for training_seed in seeds
            }
            per_seed = [
                float(
                    np.mean(
                        [
                            reference_values[training_seed][episode_key]
                            - baseline_values[training_seed][episode_key]
                            for episode_key in episode_keys
                        ]
                    )
                )
                for training_seed in seeds
            ]
            estimate = float(np.mean(per_seed))
            bootstrap = np.asarray(
                [
                    float(
                        np.mean(
                            [
                                float(
                                    np.mean(
                                        [
                                            reference_values[training_seed][episode_key]
                                            - baseline_values[training_seed][episode_key]
                                            for episode_key in sampled_ledger
                                        ]
                                    )
                                )
                                for training_seed in seeds
                            ]
                        )
                    )
                    for sampled_ledger in resampled_ledgers
                ],
                dtype=np.float64,
            )
            ci_low, ci_high = _percentile_interval(bootstrap, ci_level=level)
            p_value = float((1 + int(np.count_nonzero(bootstrap <= 0.0))) / (replicate_count + 1))
            comparisons[key] = {
                "status": "FORMAL",
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p": p_value,
                "degenerate_replicates": bool(np.all(bootstrap == bootstrap[0])),
            }
            p_values[key] = p_value

    holm_rows = stable_holm(p_values, alpha=family_alpha)
    for holm in holm_rows:
        key = (str(holm["baseline_id"]), str(holm["metric_id"]))
        comparisons[key].update(
            {
                "holm_rank": holm["holm_rank"],
                "holm_threshold": holm["holm_threshold"],
                "holm_adjusted_p": holm["holm_adjusted_p"],
                "rejected": holm["rejected"],
            }
        )
    return {
        "reference_variant": reference,
        "comparisons": comparisons,
        "holm": holm_rows,
        "family_size": len(comparisons),
        "alpha": family_alpha,
        "replicates": replicate_count,
        "seed": random_seed,
        "ci_level": level,
        "seed_field": seed_field,
        "quantile_method": "linear",
    }


def confirmatory_family(rows, *, expected_variants, expected_seeds,
        expected_episode_keys, replicates, seed, alpha):
    """Preserve the historical Full-vs-four-controls eight-test contract."""
    variants = tuple(expected_variants)
    if len(variants) != 5 or len(set(variants)) != 5:
        raise ValueError("confirmatory family requires Full plus four unique controls")
    result = paired_comparison_family(rows, expected_variants=variants,
        expected_seeds=expected_seeds, expected_episode_keys=expected_episode_keys,
        replicates=replicates, seed=seed, alpha=alpha,
        seed_field="training_seed", ci_level=0.95)
    result.pop("seed_field")
    return result


__all__ = [
    "closed_loop_point_estimates",
    "confirmatory_family",
    "paired_comparison_family",
    "offline_hierarchical_summary",
    "stable_holm",
]
