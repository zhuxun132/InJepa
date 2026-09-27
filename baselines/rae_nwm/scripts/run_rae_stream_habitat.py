#!/usr/bin/env python3
"""Run RAE-stream through the existing V9 Habitat ImageGoal runner.

No simulator or planner implementation is copied here.  The runner and its
strict RGB-only firewall are loaded dynamically from ``--runner-root``.  A
factory seam is required for a real official planner so an incompatible
checkout fails closed instead of silently falling back to a different CEM.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


def _bootstrap_repo() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _bootstrap_repo()
from rae_stream.action_decoder import RAEStreamActionDecoder  # noqa: E402
from rae_stream.habitat_policy import (  # noqa: E402
    OFFICIAL_PLANNER_ROLLOUT_STEPS,
    OfficialRAEPlannerBackend,
    RAEStreamPolicy,
)
from rae_stream.launcher import resolved_training_identity  # noqa: E402


DEFAULT_PLANNER_FACTORY = "rae_stream.official_factory:create_official_backend"


def _load_symbol(spec: str, roots: Sequence[Path] = ()) -> Any:
    """Load ``module:attribute`` or a Python file symbol without shell eval."""

    if ":" not in spec:
        raise ValueError("factory must be MODULE:ATTRIBUTE")
    module_name, attribute = spec.split(":", 1)
    if not module_name or not attribute:
        raise ValueError("factory must be MODULE:ATTRIBUTE")
    for root in roots:
        root_text = str(root.resolve())
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
    if module_name.endswith(".py") and Path(module_name).is_file():
        path = Path(module_name).resolve()
        generated_name = f"rae_stream_factory_{hashlib.sha256(str(path).encode()).hexdigest()[:12]}"
        module_spec = importlib.util.spec_from_file_location(generated_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"could not load factory file {path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(f"factory attribute not found: {spec}") from exc


def _invoke_factory(factory: Callable[..., Any], **kwargs: Any) -> Any:
    if not callable(factory):
        raise TypeError("factory must be callable")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)
    accepted = {name: value for name, value in kwargs.items() if name in signature.parameters}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        accepted = kwargs
    try:
        signature.bind(**accepted)
    except TypeError as exc:
        raise TypeError(f"factory cannot satisfy RAE-stream factory ABI: {exc}") from exc
    return factory(**accepted)


class _FakeBackend:
    def __init__(self, command: Sequence[float] = (0.0, 0.0, 0.0)) -> None:
        if len(command) != 3:
            raise ValueError("fake command must have three values")
        self.command = tuple(float(value) for value in command)
        self.calls = 0

    def plan(self, context: Any, goal: Any) -> tuple[float, float, float]:
        self.calls += 1
        return self.command


def _load_runner(runner_root: Path) -> Any:
    root = runner_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    expected_origin = (root / "j2j" / "evaluation" / "habitat_runner.py").resolve()
    existing = sys.modules.get("j2j.evaluation.habitat_runner")
    if existing is not None:
        origin = getattr(getattr(existing, "__spec__", None), "origin", None)
        if origin is None or Path(origin).resolve() != expected_origin:
            raise RuntimeError(
                "existing V9 Habitat runner is loaded from outside --runner-root: "
                f"{origin!r} (expected {expected_origin})"
            )
        module = existing
    else:
        try:
            module = importlib.import_module("j2j.evaluation.habitat_runner")
        except Exception as exc:
            raise RuntimeError(f"could not import existing V9 Habitat runner from {root}: {exc}") from exc
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin is None or Path(origin).resolve() != expected_origin:
            raise RuntimeError(
                "V9 Habitat runner resolved outside --runner-root: "
                f"{origin!r} (expected {expected_origin})"
            )
    for symbol in ("run_imagegoal_episode", "load_habitat_environment"):
        if not callable(getattr(module, symbol, None)):
            raise RuntimeError(f"existing runner lacks required symbol {symbol}")
    return module


def _verify_checkpoint(path: Path, expected_sha: str | None) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"checkpoint SHA mismatch: got {actual}, expected {expected_sha}")
    return actual


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _video_recorder(path, **kwargs):
    from j2j.evaluation.video import FirstPersonVideoRecorder
    return FirstPersonVideoRecorder(path, **kwargs)


def _continuous_habitat_config(config_path):
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import ActionConfig
    from omegaconf import open_dict, read_write
    config = get_config(str(config_path))
    with read_write(config), open_dict(config):
        config.habitat.task.actions.rae_continuous = ActionConfig(type='RAEContinuousVelocityAction')
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--habitat-config", type=Path, required=True)
    parser.add_argument("--episodes-path", type=Path, required=True)
    parser.add_argument("--scenes-dir", type=Path)
    parser.add_argument("--rae-root", type=Path, required=True)
    parser.add_argument(
        "--rae-config",
        type=Path,
        default=Path("config/rae_stream.yaml"),
        help="RAE-stream overlay passed to the official planner (relative to --rae-root)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="trained official RAE checkpoint (required for a real run; optional for --fake/--dry-run)",
    )
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument(
        "--planner-factory",
        default=DEFAULT_PLANNER_FACTORY,
        help="MODULE:factory returning a backend or official evaluator",
    )
    parser.add_argument("--fake", action="store_true", help="use an explicit zero-command fake backend for contract smoke")
    parser.add_argument("--fake-command", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument("--decode-max-distance", type=float)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-integration-steps", type=int, default=None)
    parser.add_argument("--diagnostic-reach-radius", type=float, default=None)
    parser.add_argument("--episode-indices", nargs="+", type=int)
    parser.add_argument("--planner-output-dir", type=Path)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-episodes", type=int, default=0)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--step-trace-dir", type=Path)
    parser.add_argument("--episode-results-dir", type=Path)
    parser.add_argument('--control-mode', choices=('discrete', 'continuous'), default='discrete')
    parser.add_argument('--encoder-batch-size', type=int, default=16)
    parser.add_argument('--model-batch-size', type=int, default=16)
    parser.add_argument('--inference-memory-mode', choices=('official', 'batched_endpoint'),
                        default='batched_endpoint',
                        help='official skips all memory wrappers; batch sizes apply only to batched_endpoint')
    parser.add_argument('--control-dt', type=float, default=1.)
    parser.add_argument('--control-max-translation', type=float, default=.25)
    parser.add_argument('--control-max-rotation', type=float, default=math.pi/12)
    parser.add_argument('--control-stop-translation-speed', type=float, default=.025)
    parser.add_argument('--control-stop-rotation-speed', type=float, default=math.pi/180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        integration_steps = args.diagnostic_integration_steps
        if integration_steps is not None:
            if type(integration_steps) is not int or integration_steps <= 0:
                raise ValueError('diagnostic integration steps must be a positive integer')
            if not args.diagnostic:
                raise ValueError('integration step override requires --diagnostic')
        memory_mode = getattr(args, 'inference_memory_mode', 'batched_endpoint')
        if memory_mode not in ('official', 'batched_endpoint'):
            raise ValueError('invalid inference memory mode')
        reach_radius = getattr(args, 'diagnostic_reach_radius', None)
        if reach_radius is not None:
            if (isinstance(reach_radius, bool) or not isinstance(reach_radius, (int, float))
                    or not math.isfinite(reach_radius) or reach_radius <= 0):
                raise ValueError('diagnostic reach radius must be finite and positive')
            if not args.diagnostic:
                raise ValueError('reach radius requires --diagnostic')
        if args.encoder_batch_size <= 0:
            raise ValueError('encoder-batch-size must be positive')
        if args.model_batch_size <= 0:
            raise ValueError('model-batch-size must be positive')
        control_parameters = dict(dt=args.control_dt, max_translation=args.control_max_translation,
            max_rotation=args.control_max_rotation, translation_stop_speed=args.control_stop_translation_speed,
            rotation_stop_speed=args.control_stop_rotation_speed)
        if args.control_mode == 'continuous':
            if not args.diagnostic:
                raise ValueError('continuous control currently requires --diagnostic')
            from rae_stream.continuous_control import decode_continuous_command
            decode_continuous_command([0., 0., 0.], **control_parameters)
        if args.episodes <= 0 or args.max_steps <= 0:
            raise ValueError("episodes and max-steps must be positive")
        if args.video_episodes < 0 or not math.isfinite(args.video_fps) or args.video_fps <= 0:
            raise ValueError("invalid video budget or fps")
        if args.video_episodes and args.video_dir is None:
            raise ValueError("video recording requires --video-dir")
        if (args.episode_indices is not None or args.step_trace_dir is not None) and not args.diagnostic:
            raise ValueError("selected episodes and scalar traces require --diagnostic")
        selected_rows = None
        if args.episode_indices is not None:
            indices = args.episode_indices
            if len(indices) != args.episodes or len(set(indices)) != len(indices) or min(indices) < 0:
                raise ValueError("episode-indices must be unique and match --episodes")
            opener = gzip.open if args.episodes_path.suffix == ".gz" else open
            with opener(args.episodes_path, "rt") as handle:
                ledger = json.load(handle)["episodes"]
            if max(indices) >= len(ledger):
                raise ValueError("selected episode is outside ledger")
            selected_rows = [ledger[index] for index in indices]
        if args.checkpoint is None and not (args.fake or args.dry_run):
            raise ValueError("--checkpoint is required for a real Habitat run")
        checkpoint_sha: str | None = None
        if args.checkpoint is not None:
            checkpoint_path = args.checkpoint.resolve()
            # A planning/fake contract run must be able to validate wiring
            # before a formal checkpoint exists.  If a checkpoint is supplied
            # we still hash it, so the receipt remains identity-bound.
            if checkpoint_path.is_file():
                checkpoint_sha = _verify_checkpoint(checkpoint_path, args.checkpoint_sha256)
            elif not (args.fake or args.dry_run):
                raise FileNotFoundError(checkpoint_path)
        runner = None
        plan = {
            "runner_root": str(args.runner_root.resolve()),
            "habitat_config": str(args.habitat_config.resolve()),
            "episodes_path": str(args.episodes_path.resolve()),
            "scenes_dir": str(args.scenes_dir.resolve()) if args.scenes_dir else None,
            "rae_root": str(args.rae_root.resolve()),
            "rae_config": str(
                (args.rae_config if args.rae_config.is_absolute() else args.rae_root / args.rae_config).resolve()
            ),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_sha256": checkpoint_sha,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "decoder": "rae_stream_fixed_codebook_v1" if args.control_mode == 'discrete' else None,
            "control_mode": args.control_mode,
            "encoder_batch_size": args.encoder_batch_size,
            "model_batch_size": args.model_batch_size,
            "inference_memory_mode": memory_mode,
            "control_parameters": control_parameters if args.control_mode == 'continuous' else None,
            "diagnostic": args.diagnostic,
            "diagnostic_integration_steps": integration_steps,
            "diagnostic_reach_radius": reach_radius,
            "episode_indices": args.episode_indices,
            "episodes_sha256": (_verify_checkpoint(args.episodes_path, None)
                                if args.episodes_path.is_file() else None),
        }
        official_identity = None
        if not (args.fake or args.dry_run):
            # A real comparison is identity-bound: the fixed official commit,
            # restricted source hashes, and data-only overlay must all pass
            # before importing Habitat or constructing the planner.  This is
            # intentionally the same gate as formal training, so an external
            # checkout/config cannot be labelled RAE-stream by accident.
            if args.planner_factory != DEFAULT_PLANNER_FACTORY:
                raise ValueError(
                    "real RAE-stream evaluation requires the in-tree official planner factory; "
                    "an external factory is not an admitted comparison identity"
                )
            resolved_config = args.rae_config if args.rae_config.is_absolute() else args.rae_root / args.rae_config
            official_identity = resolved_training_identity(
                repo_root=args.rae_root.resolve(),
                config_path=resolved_config.resolve(),
                base_config_path=(args.rae_root / "config" / "raenwm.yaml").resolve(),
                **({"diagnostic_inference": True} if args.diagnostic and integration_steps is not None else {}),
            )
            plan["official_identity"] = official_identity
        else:
            plan["official_identity"] = None
        if args.dry_run:
            plan["status"] = "PLAN_ONLY"
            print(json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2))
            return 0

        runner = _load_runner(args.runner_root)

        if args.fake:
            backend: Any = _FakeBackend(args.fake_command)
        elif args.planner_factory:
            inference_config=(args.rae_config if args.rae_config.is_absolute() else args.rae_root / args.rae_config).resolve()
            if integration_steps is not None:
                from rae_stream.diagnostic_inference import write_inference_config
                planner_output=(args.planner_output_dir or args.rae_root / "runs" / "habitat_planner").resolve()
                plan["inference_config"]=write_inference_config(inference_config,planner_output / "diagnostic_inference.yaml",integration_steps)
                inference_config=Path(plan["inference_config"]["path"])
                print(json.dumps({"event":"DIAGNOSTIC_INFERENCE_CONFIG_BOUND",**plan["inference_config"]}),flush=True)
            factory = _load_symbol(args.planner_factory, roots=(args.rae_root, args.runner_root))
            backend = _invoke_factory(
                factory,
                repo_root=args.rae_root.resolve(),
                config=inference_config,
                checkpoint=args.checkpoint.resolve(),
                checkpoint_sha256=checkpoint_sha,
                runner_root=args.runner_root.resolve(),
                habitat_config=args.habitat_config.resolve(),
                output_dir=(args.planner_output_dir or args.rae_root / "runs" / "habitat_planner").resolve(),
                encoder_batch_size=args.encoder_batch_size,
                model_batch_size=args.model_batch_size,
                inference_memory_mode=memory_mode,
            )
            if callable(getattr(backend, "generate_actions", None)):
                # The factory may return an already initialized official
                # WM_Planning_Evaluator.  Import its module's transform and
                # torch exactly once, then leave CEM/ODE execution upstream.
                module_name = getattr(backend.__class__, "__module__", "")
                planning_module = sys.modules.get(module_name)
                if planning_module is None:
                    raise RuntimeError("official evaluator module is not import-visible for callback capture")
                transform = getattr(planning_module, "transform", None)
                torch_module = getattr(planning_module, "torch", None)
                if transform is None or torch_module is None:
                    raise RuntimeError("official evaluator module lacks transform/torch ABI")
                backend = OfficialRAEPlannerBackend(
                    backend,
                    transform=transform,
                    torch_module=torch_module,
                    len_traj_pred=OFFICIAL_PLANNER_ROLLOUT_STEPS,
                    dataset_name="rae_stream",
                )
            if not callable(getattr(backend, "plan", None)):
                raise TypeError("planner factory must return backend.plan(...) or official evaluator")

        decoder = RAEStreamActionDecoder(
            spacing=0.25,
            action_stats={"min": [-64, -64], "max": [64, 64]},
            max_distance=args.decode_max_distance,
        )
        policy = RAEStreamPolicy(backend, decoder=decoder, context_size=4, mode=args.control_mode)
        environment_config = {'config_path': str(args.habitat_config.resolve())}
        control_events = []
        def continuous_handler(command):
            event = decode_continuous_command(command, **control_parameters)
            control_events.append(event)
            return {key: event[key] for key in ('name', 'is_stop', 'habitat_payload')}
        if args.control_mode == 'continuous':
            from rae_stream.habitat_task import register_continuous_action
            register_continuous_action()
            habitat_config = _continuous_habitat_config(args.habitat_config.resolve())
            environment_config = {'config': habitat_config}
        env = runner.load_habitat_environment(
            **environment_config,
            episodes_path=str(args.episodes_path.resolve()),
            scenes_dir=str(args.scenes_dir.resolve()) if args.scenes_dir else None,
            **({"episode_indices": args.episode_indices} if args.episode_indices is not None else {}),
        )
        receipts: list[dict[str, Any]] = []
        try:
            for episode_index in range(args.episodes):
                global_index = args.episode_indices[episode_index] if args.episode_indices is not None else episode_index
                expected = {}
                if selected_rows is not None:
                    row = selected_rows[episode_index]
                    expected = {"expected_episode_id": str(row["episode_id"]),
                                "expected_scene_id": Path(row["scene_id"]).stem}
                recorder = None
                if episode_index < args.video_episodes:
                    recorder = _video_recorder(
                        args.video_dir / f"episode_{global_index:06d}.mp4", fps=args.video_fps,
                        episode_key=[expected.get("expected_scene_id"), expected.get("expected_episode_id")],
                        identities={"checkpoint_sha256": checkpoint_sha, "ledger_index": global_index,
                                    "episode_ledger_sha256": plan["episodes_sha256"]})
                trace = None
                if args.step_trace_dir is not None:
                    args.step_trace_dir.mkdir(parents=True, exist_ok=True)
                    trace = (args.step_trace_dir / f"episode_{global_index:06d}.steps.jsonl").open("x")
                def observe_frame(event):
                    if recorder is not None:
                        recorder(event)
                    if trace is not None and event["phase"] == "step":
                        decision = policy.last_decision
                        metric_delta = (control_events[-1]['metric_delta'] if args.control_mode == 'continuous'
                            else decoder.events[-1].metric_delta) if decision.continuous_command is not None else None
                        evidence = {"ledger_index": global_index, "step": event["step"],
                            "action": event["action"], "continuous_command": decision.continuous_command,
                            "metric_delta": metric_delta, "timing": "post_action",
                            "distance_to_goal": float(env.get_metrics()["distance_to_goal"])}
                        if args.control_mode == 'continuous':
                            evidence['execution'] = control_events[-1]
                        trace.write(json.dumps(evidence, allow_nan=False) + "\n")
                        trace.flush()
                try:
                    result = runner.run_imagegoal_episode(
                        env, policy, max_steps=args.max_steps, success_distance=args.success_distance,
                        **({'diagnostic_reach_radius': reach_radius} if reach_radius is not None else {}),
                        **({'continuous_action_handler': continuous_handler} if args.control_mode == 'continuous' else {}),
                        **expected, **({"frame_observer": observe_frame} if recorder is not None or trace is not None else {}))
                finally:
                    if trace is not None:
                        trace.close()
                if recorder is not None:
                    result["first_person_video"] = recorder.close()
                if integration_steps is not None and not args.fake:
                    observed=getattr(getattr(backend,"evaluator",None),"last_ode_num_steps",None)
                    if observed is not None and observed != integration_steps:
                        raise RuntimeError("actual sampler integration steps mismatch")
                    result["observed_ode_num_steps"]=observed
                    plan["observed_ode_num_steps"]=observed
                result["ledger_index"] = global_index
                if args.episode_results_dir is not None:
                    args.episode_results_dir.mkdir(parents=True, exist_ok=True)
                    with (args.episode_results_dir / f"episode_{global_index:06d}.json").open("x") as handle:
                        json.dump(result, handle, allow_nan=False)
                receipts.append(result)
                print(json.dumps({"event": "episode_complete", "ledger_index": global_index,
                                  "success": result.get("success"), "num_steps": result.get("num_steps"),
                                  "observed_ode_num_steps": result.get("observed_ode_num_steps")}), flush=True)
        finally:
            policy.close()
            close = getattr(env, "close", None)
            if callable(close):
                close()
        output = {
            "status": "SMOKE_PASS" if args.fake else "HABITAT_EVALUATION_COMPLETED",
            "claim_boundary": (
                "TRAINING_MECHANICS_ONLY_NOT_NAVIGATION_PERFORMANCE"
                if args.fake
                else "HABITAT_EVALUATION_ONLY"
            ),
            "formal_evaluation_eligible": bool(not args.fake and not args.diagnostic),
            "fake_backend": bool(args.fake),
            "identity": plan,
            "episodes_completed": len(receipts),
            "episode_receipts": receipts,
            "action_conversion": (decoder.conversion_summary() if args.control_mode == 'discrete' else
                {'interface': 'habitat_sim.physics.VelocityControl', 'parameters': control_parameters,
                 'events': control_events, 'count': len(control_events), 'primitive_quantization': False}),
            "policy_backend_calls": getattr(backend, "calls", None),
        }
        if args.receipt is not None:
            _atomic_json(args.receipt.resolve(), output)
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        if args.diagnostic:
            import traceback
            traceback.print_exc()
        print(f"run_rae_stream_habitat: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
