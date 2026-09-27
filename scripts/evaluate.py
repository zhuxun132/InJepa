"""Run a prepared configuration through its original evaluation entrypoint."""
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from release_common import METHODS, ROOT, checked_ledger, read_json, resolve, resolve_fields, write_json


def plan(args):
    config = copy.deepcopy(read_json(args.config))
    method = args.method
    is_injepa = config.get('schema') == 'J2J_RECURRENT_RAW_CLOSED_LOOP_CONFIG_V1'
    if (method == 'injepa') != is_injepa:
        raise ValueError('method does not match the configuration schema')
    if config.get('release_method', method) != method:
        raise ValueError('method does not match this prepared configuration')
    if args.episodes is not None and not 1 <= args.episodes <= 150:
        raise ValueError('episodes must be between 1 and 150')
    if args.seed is not None and args.seed < 0:
        raise ValueError('seed must be nonnegative')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Choose a new result directory: {output}')
    # Config lives alongside the create-once result directory; the original
    # runners remain responsible for creating the result directory themselves.
    resolved_path = output.with_name(output.name + '.config.json')
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(ROOT / 'injepa')
    cwd = ROOT
    if is_injepa:
        deployment = config.get('deployment')
        if deployment not in {'f_feedback_q', 'proposal_only', 'posthoc_f', 'goal_gf'}:
            raise ValueError('unsupported InJepa deployment')
        script = 'evaluate_vjepa_full.py' if deployment == 'f_feedback_q' else 'evaluate_vjepa_variants.py'
        entry = ROOT / 'injepa/scripts' / script
        runpy.run_path(str(entry))['validate_export_scope'](config)
        habitat = config['habitat']
        ledger, _ = checked_ledger(habitat['episode_ledger'], habitat['episode_ledger_sha256'])
        habitat['episode_ledger'] = str(ledger)
        resolve_fields(habitat, ['sensor_config', 'evaluation_config', 'scene_root', 'capability_receipt'])
        resolve_fields(config['checkpoint'], ['path', 'training_resolved_config_path'])
        resolve_fields(config['visual'], ['source_root', 'checkpoint'])
        resolve_fields(config['stop'], ['receipt'])
        key = 'diagnostic_episode_indices'
        if args.seed is not None:
            config['evaluation_seed'] = args.seed
        if args.device is not None:
            config['device'] = args.device
        config['video']['directory'] = str(output / 'videos')
        command = [sys.executable, str(entry), '--config', str(resolved_path), '--output', str(output / 'result.json')]
    else:
        ledger, _ = checked_ledger(config['episodes_path'], config['ledger_sha256'])
        config['episodes_path'] = str(ledger)
        resolve_fields(config, ['sensor_config', 'scenes_dir', 'runner_root', 'runner_manifest',
            'continuous_task_file', 'official_root', 'diffusion_root', 'checkpoint', 'wm_checkpoint',
            'policy_checkpoint', 'release_manifest', 'evaluation_manifest', 'training_config', 'rae_root'])
        key = 'episode_indices'
        if args.seed is not None:
            config['seed'] = args.seed
        if args.device is not None:
            config['device'] = args.device
        config['output'] = str(output)
        if method == 'lwm_croco':
            env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'baselines/lwm_evaluation'), str(ROOT / 'injepa')])
            command = [sys.executable, '-m', 'lwm_official_eval.cli', '--config', str(resolved_path), '--run']
        elif method == 'lwm_vjepa':
            source = ROOT / 'baselines/lwm_vjepa/source'
            env['PYTHONPATH'] = os.pathsep.join(map(str, [source, source / 'official', ROOT / 'baselines/lwm_evaluation', ROOT / 'injepa']))
            command = [sys.executable, str(ROOT / 'baselines/lwm_vjepa/scripts/run_clean150.py'), '--config', str(resolved_path)]
        elif method == 'nomad':
            if args.device not in (None, 'cuda', 'cuda:0'):
                raise ValueError('NoMaD uses CUDA; select the GPU with CUDA_VISIBLE_DEVICES')
            command = [sys.executable, str(ROOT / 'baselines/nomad/nomad_eval.py'), '--config', str(resolved_path)]
        else:
            command = []  # Expanded below after selecting the episode prefix.

    selected = config.get(key, list(range(150)))
    if (not isinstance(selected, list) or not selected or selected != sorted(set(selected))
            or any(type(i) is not int or not 0 <= i < 150 for i in selected)):
        raise ValueError('episode indices must be a sorted unique Clean150 subset')
    if args.episodes is not None:
        if args.episodes > len(selected):
            raise ValueError('episodes exceeds the configured subset')
        selected = selected[:args.episodes]
    config[key] = selected
    if is_injepa:
        config['diagnostic']['episode_budget'] = len(selected)
    if method == 'rae_nwm':
        runtime = Path(config['rae_root'])
        cwd = runtime
        env['PYTHONPATH'] = os.pathsep.join([str(runtime), str(ROOT / 'injepa')])
        env['DIAGNOSTIC_SEED'] = str(config['seed'])
        env['PYTHONHASHSEED'] = str(config['seed'])
        command = [sys.executable, str(runtime / 'scripts/official_bootstrap.py'),
            str(runtime / 'scripts/run_rae_stream_habitat.py'),
            '--runner-root', config['runner_root'], '--habitat-config', config['sensor_config'],
            '--episodes-path', config['episodes_path'], '--scenes-dir', config['scenes_dir'],
            '--rae-root', str(runtime), '--checkpoint', config['checkpoint'],
            '--checkpoint-sha256', config['checkpoint_sha256'],
            '--episodes', str(len(selected)), '--episode-indices', *map(str, selected),
            '--max-steps', str(config['max_steps']), '--success-distance', str(config['reach_radius']),
            '--diagnostic', '--diagnostic-reach-radius', str(config['reach_radius']),
            '--diagnostic-integration-steps', str(config['integration_steps']),
            '--control-mode', 'continuous', '--control-dt', str(config['dt']),
            '--control-max-translation', str(config['max_translation']),
            '--control-max-rotation', str(config['max_yaw']),
            '--control-stop-translation-speed', '0', '--control-stop-rotation-speed', '0',
            '--planner-output-dir', str(output / 'planner'), '--episode-results-dir', str(output / 'episodes'),
            '--step-trace-dir', str(output / 'traces'), '--receipt', str(output / 'result.json')]
    inherited_path = os.environ.get('PYTHONPATH')
    if inherited_path:
        env['PYTHONPATH'] += os.pathsep + inherited_path
    return {'command': command, 'resolved_config': config,
            'config_path': str(resolved_path), 'cwd': str(cwd)}, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device')
    parser.add_argument('--dry-run', action='store_true', help='Print the command and resolved configuration')
    args = parser.parse_args()
    result, env = plan(args)
    if args.dry_run:
        print(json.dumps(result, indent=2))
        return
    write_json(result['config_path'], result['resolved_config'])
    subprocess.run(result['command'], cwd=result['cwd'], env=env, check=True)


if __name__ == '__main__':
    main()
