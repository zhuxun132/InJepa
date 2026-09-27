"""Run a prepared configuration through its original evaluation entrypoint."""
import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from release_common import METHODS, ROOT, checked_ledger, read_json, resolve_fields, write_json


def plan(args):
    config = copy.deepcopy(read_json(args.config))
    method = args.method
    if method != 'injepa' or config.get('schema') != 'J2J_RECURRENT_RAW_CLOSED_LOOP_CONFIG_V1':
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
    deployment = config.get('deployment')
    if deployment not in {'f_feedback_q', 'proposal_only', 'posthoc_f'}:
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

    selected = config.get(key, list(range(150)))
    if (not isinstance(selected, list) or not selected or selected != sorted(set(selected))
            or any(type(i) is not int or not 0 <= i < 150 for i in selected)):
        raise ValueError('episode indices must be a sorted unique Clean150 subset')
    if args.episodes is not None:
        if args.episodes > len(selected):
            raise ValueError('episodes exceeds the configured subset')
        selected = selected[:args.episodes]
    config[key] = selected
    config['diagnostic']['episode_budget'] = len(selected)
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
