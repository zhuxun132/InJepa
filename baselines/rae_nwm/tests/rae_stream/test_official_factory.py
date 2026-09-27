from argparse import Namespace
from pathlib import Path
import importlib.util
import math
import sys
import types

import pytest
import yaml

from rae_stream.official_factory import build_planner_args


def test_rae_stream_planner_prior_uses_official_terminal_bias_units() -> None:
    """The upstream planner multiplies its third sample component by pi."""

    root = Path(__file__).resolve().parents[2]
    values = yaml.safe_load((root / "config" / "data_hyperparams_plan.yaml").read_text())
    prior = values["rae_stream"]
    assert prior["var_scale"][:2] == pytest.approx([1.0 / 64.0, 1.0 / 64.0])
    assert prior["var_scale"][2] == pytest.approx((math.pi / 12.0) / math.pi)


def test_build_planner_args_is_official_and_relocatable(tmp_path: Path) -> None:
    config = tmp_path / "config" / "rae_stream.yaml"
    checkpoint = tmp_path / "checkpoint.pth.tar"
    output = tmp_path / "runs" / "planner"

    args = build_planner_args(
        repo_root=tmp_path,
        config_path=config,
        checkpoint_path=checkpoint,
        output_dir=output,
    )

    assert isinstance(args, Namespace)
    assert args.exp == str(config.resolve())
    assert args.checkpoint_path == str(checkpoint.resolve())
    assert args.output_dir == str(output.resolve())
    assert args.datasets == "rae_stream"
    assert args.num_samples == 120
    assert args.topk == 3
    assert args.opt_steps == 1
    assert args.rollout_stride == 1
    assert args.num_repeat_eval == 1
    assert args.traj_sampler == "curve"
    assert args.score_type == "dino"
    # The factory must provide every attribute read by the unchanged official
    # WM_Planning_Evaluator instead of relying on argparse side effects.
    for name in (
        "save_preds",
        "plot",
        "plot_topn",
        "num_workers",
        "batch_size",
        "subset_items",
        "subset_seed",
        "run_tag",
        "ckp",
        "prior_mix",
        "backtrack_allow",
        "prior_beta",
    ):
        assert hasattr(args, name)


def test_habitat_cli_has_builtin_official_factory() -> None:
    from scripts.run_rae_stream_habitat import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--runner-root",
            "/runner",
            "--habitat-config",
            "/habitat.yaml",
            "--episodes-path",
            "/episodes.json.gz",
            "--rae-root",
            "/rae",
            "--checkpoint",
            "/rae/checkpoint.pth.tar",
        ]
    )
    assert args.planner_factory == "rae_stream.official_factory:create_official_backend"
    assert args.rae_config == Path("config/rae_stream.yaml")


def test_habitat_runner_existing_module_must_match_requested_root(tmp_path: Path, monkeypatch) -> None:
    from scripts.run_rae_stream_habitat import _load_runner

    expected = tmp_path / "j2j" / "evaluation" / "habitat_runner.py"
    expected.parent.mkdir(parents=True)
    expected.write_text("# fixture\n", encoding="utf-8")
    module = types.ModuleType("j2j.evaluation.habitat_runner")
    module.__spec__ = importlib.util.spec_from_file_location(module.__name__, expected)
    module.run_imagegoal_episode = lambda *args, **kwargs: None
    module.load_habitat_environment = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert _load_runner(tmp_path) is module

    outside = tmp_path / "outside_runner.py"
    outside.write_text("# outside\n", encoding="utf-8")
    module.__spec__ = importlib.util.spec_from_file_location(module.__name__, outside)
    with pytest.raises(RuntimeError, match="outside"):
        _load_runner(tmp_path)
