"""Construct the unchanged official RAE planner for the RAE-stream adapter.

This module is deliberately a factory, not a second planner.  It imports the
checkout's ``planning_eval.WM_Planning_Evaluator`` and supplies the complete
``argparse.Namespace`` that the upstream class reads.  Keeping the namespace
construction here makes the Habitat entry point reproducible while leaving
CEM, latent rollout, ODE sampling, and checkpoint loading in the official
source unchanged.
"""

from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Iterator

from .habitat_policy import OFFICIAL_PLANNER_ROLLOUT_STEPS, OfficialRAEPlannerBackend


# These are the registered paper/official-planner settings.  They are not
# training hyperparameters and are intentionally not inferred from results.
OFFICIAL_PLANNER_PROFILE: dict[str, Any] = {
    "datasets": "rae_stream",
    "num_samples": 120,
    "topk": 3,
    "opt_steps": 1,
    "rollout_stride": 1,
    "num_repeat_eval": 1,
    "traj_sampler": "curve",
    "score_type": "dino",
}


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """Temporarily provide the relative-config ABI expected by upstream RAE."""

    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_planner_args(
    *,
    repo_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    eval_num_workers: int = 0,
    eval_batch_size: int = 1,
) -> Namespace:
    """Build every attribute read by ``WM_Planning_Evaluator``.

    The two evaluation loader controls default to a single-process/single-item
    loader because the Habitat adapter calls ``generate_actions`` directly and
    never invokes the evaluator's dataset-wide ``evaluate`` loop.  They do not
    alter the official model or CEM path.
    """

    root = Path(repo_root).resolve()
    config = Path(config_path)
    if not config.is_absolute():
        config = root / config
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    if not root.is_dir():
        raise FileNotFoundError(root)
    if isinstance(eval_num_workers, bool) or int(eval_num_workers) < 0:
        raise ValueError("eval_num_workers must be nonnegative")
    if isinstance(eval_batch_size, bool) or int(eval_batch_size) <= 0:
        raise ValueError("eval_batch_size must be positive")

    values: dict[str, Any] = {
        # Core evaluator/config ABI.
        "exp": str(config.resolve()),
        "ckp": "unused-with-explicit-checkpoint",
        "datasets": OFFICIAL_PLANNER_PROFILE["datasets"],
        "output_dir": str(output.resolve()),
        "save_preds": False,
        "num_workers": int(eval_num_workers),
        "batch_size": int(eval_batch_size),
        "checkpoint_path": str(checkpoint.resolve()),
        "subset_items": 0,
        "subset_seed": 0,
        "run_tag": "rae_stream_habitat",
        # Official planner controls.
        "num_samples": int(OFFICIAL_PLANNER_PROFILE["num_samples"]),
        "rollout_stride": int(OFFICIAL_PLANNER_PROFILE["rollout_stride"]),
        "topk": int(OFFICIAL_PLANNER_PROFILE["topk"]),
        "opt_steps": int(OFFICIAL_PLANNER_PROFILE["opt_steps"]),
        "num_repeat_eval": int(OFFICIAL_PLANNER_PROFILE["num_repeat_eval"]),
        "prior_mix": 0.0,
        "backtrack_allow": 0.0,
        "prior_beta": 0.8,
        "traj_sampler": OFFICIAL_PLANNER_PROFILE["traj_sampler"],
        "plot": False,
        "plot_topn": 0,
        "score_type": OFFICIAL_PLANNER_PROFILE["score_type"],
    }
    return Namespace(**values)


def _import_planning_module(repo_root: Path) -> Any:
    """Import ``planning_eval`` from this checkout, rejecting stale modules."""

    root_text = str(repo_root)
    sibling_names = (
        "planning_eval",
        "datasets",
        "infer",
        "misc",
        "evaluate",
        "distributed",
        "models",
        "diffusion",
        "RAE",
    )

    def assert_origins() -> None:
        for module_name, module in list(sys.modules.items()):
            if not any(module_name == name or module_name.startswith(name + ".") for name in sibling_names):
                continue
            origin = getattr(getattr(module, "__spec__", None), "origin", None)
            if origin in (None, "built-in", "frozen"):
                # The official RAE/ directory is a PEP420 namespace package.
                # Bind every search location to its exact package directory;
                # do not invent an __init__.py or accept a foreign namespace.
                locations = getattr(getattr(module, '__spec__', None), 'submodule_search_locations', None)
                if locations is not None:
                    expected = (repo_root / Path(*module_name.split('.'))).resolve()
                    paths = [Path(location).resolve() for location in locations]
                    if not paths or any(path != expected or not path.is_dir() for path in paths):
                        raise RuntimeError(f'official namespace {module_name} has foreign or missing search locations')
                    expected.relative_to(repo_root)
                    continue
                if module_name in sibling_names:
                    raise RuntimeError(f"official module {module_name} has no file origin")
                continue
            try:
                Path(origin).resolve().relative_to(repo_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"official module {module_name} already loaded from outside {repo_root}: {origin}"
                ) from exc

    assert_origins()
    # A pre-existing module from another RAE checkout could silently mix model
    # code and configs.  Refuse that rather than trying to mutate its globals.
    existing = sys.modules.get("planning_eval")
    if existing is not None:
        origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if origin is None or Path(origin).resolve().parent != repo_root:
            raise RuntimeError(f"planning_eval already loaded from outside {repo_root}: {origin}")
        return existing
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    module = importlib.import_module("planning_eval")
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    if origin is None or Path(origin).resolve().parent != repo_root:
        raise RuntimeError(f"planning_eval resolved outside official RAE root: {origin}")
    assert_origins()
    return module


def create_official_backend(
    *,
    repo_root: str | Path,
    config: str | Path,
    checkpoint: str | Path,
    checkpoint_sha256: str | None = None,
    runner_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    eval_num_workers: int = 0,
    eval_batch_size: int = 1,
    encoder_batch_size: int = 16,
    model_batch_size: int = 16,
    inference_memory_mode: str = "batched_endpoint",
) -> OfficialRAEPlannerBackend:
    """Initialize one official evaluator and wrap it in the policy ABI.

    The caller must use the returned object for all episodes in a process;
    constructing it per episode would repeat checkpoint/model initialization.
    ``runner_root`` and ``checkpoint_sha256`` are accepted for a stable factory
    ABI and provenance validation by the outer script; the official evaluator
    itself only needs the RAE root/config/checkpoint.
    """

    if inference_memory_mode not in ("official", "batched_endpoint"):
        raise ValueError("invalid inference memory mode")
    del runner_root  # provenance is recorded by run_rae_stream_habitat.py
    root = Path(repo_root).resolve()
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = root / config_path
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = root / checkpoint_path
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if checkpoint_sha256 is not None:
        import hashlib

        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != str(checkpoint_sha256):
            raise RuntimeError(f"checkpoint SHA mismatch in official factory: got {actual}, expected {checkpoint_sha256}")

    if output_dir is None:
        output_dir = root / "runs" / "habitat_planner"
    args = build_planner_args(
        repo_root=root,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_num_workers=eval_num_workers,
        eval_batch_size=eval_batch_size,
    )
    # planning_eval imports config files relative to cwd and imports sibling
    # modules by name.  Keep that upstream ABI intact only during import and
    # one-time construction; all subsequent calls use loaded objects.
    with _working_directory(root):
        planning_module = _import_planning_module(root)
        evaluator = planning_module.WM_Planning_Evaluator(args)
    if inference_memory_mode == "batched_endpoint":
        from .inference_batching import install_encoder_batching, make_batched_model, install_endpoint_only_sampler
        install_encoder_batching(evaluator.rae, batch_size=encoder_batch_size)
        evaluator.model_without_ddp = make_batched_model(evaluator.model_without_ddp, batch_size=model_batch_size)
        install_endpoint_only_sampler(evaluator.sampler)
    return OfficialRAEPlannerBackend(
        evaluator,
        transform=planning_module.transform,
        torch_module=planning_module.torch,
        len_traj_pred=OFFICIAL_PLANNER_ROLLOUT_STEPS,
        dataset_name="rae_stream",
    )


__all__ = [
    "OFFICIAL_PLANNER_PROFILE",
    "build_planner_args",
    "create_official_backend",
]
