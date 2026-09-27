"""Offline-only diagnostics and deployment rankers for Context4 experiments."""

from .efficiency import benchmark_decision, no_g_tree_nodes, throughput_summary
from .execution import (
    ExecutionIdentity,
    PopulationPlan,
    ValidatedRankUnion,
    build_population_plan,
    iter_rank_owned_trajectories,
    iter_validated_scalar_shards,
    managed_process_group,
    project_dataset_descriptors,
    validate_rank_union,
)
from .metrics import (
    categorical_summary,
    cosine_distance,
    expected_calibration_error,
    finite_rate,
    grid_error_metrics,
)
from .qualitative import QUALITATIVE_STRATA, StreamingQualitativeCollector
from .rankers import (
    RankerResult,
    rank_full,
    rank_no_f,
    rank_proposal_only,
    rank_posthoc_f,
    rank_no_g,
)
from .runner import run_one_pass
from .transforms import (
    apply_context_suffix,
    build_cyclic_donor_plan,
    build_single_factor_arm_plan,
    fixed_token_permutation,
    mean_repeat_grid,
)

__all__ = [
    "apply_context_suffix",
    "benchmark_decision",
    "build_cyclic_donor_plan",
    "build_population_plan",
    "build_single_factor_arm_plan",
    "categorical_summary",
    "cosine_distance",
    "expected_calibration_error",
    "ExecutionIdentity",
    "finite_rate",
    "fixed_token_permutation",
    "grid_error_metrics",
    "iter_rank_owned_trajectories",
    "iter_validated_scalar_shards",
    "managed_process_group",
    "mean_repeat_grid",
    "no_g_tree_nodes",
    "PopulationPlan",
    "project_dataset_descriptors",
    "QUALITATIVE_STRATA",
    "RankerResult",
    "rank_full",
    "rank_no_f",
    "rank_proposal_only",
    "rank_posthoc_f",
    "rank_no_g",
    "run_one_pass",
    "StreamingQualitativeCollector",
    "throughput_summary",
    "ValidatedRankUnion",
    "validate_rank_union",
]
