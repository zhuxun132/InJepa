"""Bind Clean150, local assets and original evaluator configurations."""
import argparse
import json
import math
from pathlib import Path
import sys

from release_common import METHODS, ROOT, checked, checked_ledger, identity, read_json, sha256, write_json


def require(args, *names):
    for name in names:
        if getattr(args, name, None) is None:
            raise ValueError(f'--{name.replace("_", "-")} is required for {args.method}')


def bind_environment(args):
    import yaml
    archive = getattr(args, 'mp3d_archive', None)
    if archive is not None and not archive.is_file():
        raise FileNotFoundError(archive)
    scene_root = args.scene_root.expanduser().resolve()
    ledger = ROOT / 'data/clean150/episodes150.json.gz'
    expected = read_json(ROOT / 'data/clean150/evaluation.template.json')['assets']['episodes_sha256']
    ledger, episodes = checked_ledger(ledger, expected)
    scenes = sorted({episode['scene_id'] for episode in episodes})
    for scene in scenes:
        path = scene_root / scene
        if not path.is_file():
            raise FileNotFoundError(f'MP3D scene: {path}')
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    sensor = yaml.safe_load((ROOT / 'data/clean150/sensor.template.yaml').read_text())
    sensor['habitat']['dataset']['scenes_dir'] = str(scene_root)
    sensor['habitat']['dataset']['data_path'] = str(ledger)
    sensor_path = output / 'sensor.yaml'
    sensor_path.write_text(yaml.safe_dump(sensor, sort_keys=False))
    evaluation = read_json(ROOT / 'data/clean150/evaluation.template.json')
    evaluation['assets'].update(episodes=str(ledger), episodes_sha256=sha256(ledger), scene_root=str(scene_root))
    if archive is not None:
        evaluation['assets']['mp3d_archive'] = str(archive.resolve())
    evaluation_path = output / 'evaluation.json'
    write_json(evaluation_path, evaluation)
    return output, sensor_path, evaluation_path, ledger, episodes


def prepare_injepa(args):
    require(args, 'vjepa_source', 'vjepa_checkpoint', 'habitat_lab_source', 'habitat_sim_source', 'habitat_prefix', 'mp3d_archive')
    weights = read_json(ROOT / 'weights/manifest.json')['items']
    for item in weights:
        checked(item['path'], item['sha256'])
        checked(item['training_config'], item['training_config_sha256'])
    visual = read_json(ROOT / 'data/encoder_assets.json')['variants']['VJEPA']
    checked(args.vjepa_checkpoint.resolve(), visual['weight_sha256'])
    output, sensor, evaluation, ledger, episodes = bind_environment(args)
    sys.path.insert(0, str(ROOT / 'injepa'))
    from j2j_recurrent_experiments.closed_loop.raw_stop import validate_raw_stop_calibration_receipt
    from j2j_iclr_experiments.local_habitat.capability import produce_formal_reset_replay_capability

    # Rebind the measured dev artifacts and recompute the original statistics.
    # A runtime capability is measured below on this installation.
    calibration = read_json(ROOT / 'data/calibration/calibration.template.json')
    calibration['producer_receipt'] = identity(ROOT / 'data/calibration/producer_receipt.json')
    calibration['ledger'] = identity(ROOT / 'data/calibration/pair_ledger.jsonl')
    validate_raw_stop_calibration_receipt(calibration)
    calibration_path = output / 'calibration.json'
    write_json(calibration_path, calibration)
    capability_path = output / 'capability.json'
    cases = [{'episode_key': [episode['scene_id'], str(episode['episode_id'])],
              'action_prefix': ['FWD', 'LEFT', 'RIGHT']} for episode in episodes[:2]]
    produce_formal_reset_replay_capability(
        cases=cases, habitat_lab_source_root=args.habitat_lab_source.resolve(),
        habitat_sim_source_root=args.habitat_sim_source.resolve(),
        habitat_sim_install_root=args.habitat_prefix.resolve(),
        sensor_config_path=sensor, evaluation_config_path=evaluation,
        episode_ledger_path=ledger, output_path=capability_path, frozen_seed=0)
    paths = []
    for variant in ['full', 'p_only', 'posthoc_f', 'goal_gf']:
        cfg = read_json(ROOT / f'injepa/configs/evaluate_vjepa_{variant}.template.json')
        cfg['release_method'] = 'injepa'
        cfg['device'] = args.device
        cfg['checkpoint']['path'] = str(ROOT / 'weights/injepa/epoch_0012.pt')
        cfg['checkpoint']['training_resolved_config_path'] = str(ROOT / 'weights/injepa/e12_training_config.json')
        cfg['visual'].update(source_root=str(args.vjepa_source.resolve()), checkpoint=str(args.vjepa_checkpoint.resolve()))
        cfg['stop'] = {'receipt': str(calibration_path), 'receipt_sha256': sha256(calibration_path)}
        cfg['habitat'].update(sensor_config=str(sensor), sensor_config_sha256=sha256(sensor),
            evaluation_config=str(evaluation), evaluation_config_sha256=sha256(evaluation),
            episode_ledger=str(ledger), episode_ledger_sha256=sha256(ledger),
            scene_root=str(args.scene_root.resolve()), capability_receipt=str(capability_path),
            capability_receipt_sha256=sha256(capability_path))
        destination = output / f'{variant}.json'
        write_json(destination, cfg)
        paths.append(str(destination))
    return paths


def source_manifest(root):
    return {str(path.relative_to(root)): {'bytes': path.stat().st_size, 'sha256': sha256(path)}
            for path in sorted(root.rglob('*.py')) if '__pycache__' not in path.parts}


def prepare_baseline(args):
    method = args.method
    if method == 'lwm_croco':
        require(args, 'wm_checkpoint', 'policy_checkpoint')
    elif method == 'lwm_vjepa':
        require(args, 'training_config')
    else:
        require(args, 'checkpoint')
    if method == 'rae_nwm':
        require(args, 'rae_root')
    for name in ['checkpoint', 'wm_checkpoint', 'policy_checkpoint', 'training_config']:
        path = getattr(args, name, None)
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    output, sensor, _, ledger, _ = bind_environment(args)
    common = dict(release_method=method, sensor_config=str(sensor), sensor_sha256=sha256(sensor),
        episodes_path=str(ledger), ledger_sha256=sha256(ledger), scenes_dir=str(args.scene_root.resolve()),
        runner_root=str(ROOT / 'injepa'), device=args.device, seed=0, episode_indices=list(range(150)),
        output=str(output / 'results'), continuous_task_file=str(ROOT / 'baselines/rae_nwm/rae_stream/habitat_task.py'),
        continuous_task_sha256=sha256(ROOT / 'baselines/rae_nwm/rae_stream/habitat_task.py'))
    if method.startswith('lwm_'):
        cfg = read_json(ROOT / f'baselines/{method}/configs/evaluate.template.json')
        runner_manifest = output / 'runner_manifest.json'
        write_json(runner_manifest, source_manifest(ROOT / 'injepa'))
        cfg.update(runner_manifest=str(runner_manifest), runner_manifest_sha256=sha256(runner_manifest))
        if method == 'lwm_croco':
            official = ROOT / 'baselines/lwm_croco/source/official'
            cfg.update(official_root=str(official), wm_checkpoint=str(args.wm_checkpoint.resolve()),
                       policy_checkpoint=str(args.policy_checkpoint.resolve()))
            sources = list(official.rglob('*.py')) + list((official / 'tokenizer').glob('*.json'))
            sources += [args.wm_checkpoint.resolve(), args.policy_checkpoint.resolve()]
            assets = output / 'assets.json'
            write_json(assets, {'assets': [identity(path) for path in sorted(set(sources))]})
            cfg['release_manifest'] = str(assets)
        else:
            training = read_json(args.training_config)
            training_output = Path(training['output'])
            stages = {}
            files = [args.training_config.resolve()]
            for stage in ['wm', 'rl']:
                receipt_path = training_output / stage / 'COMPLETE.json'
                receipt = read_json(receipt_path)
                if receipt['status'] != 'COMPLETE' or receipt['progress']['epoch'] != training[stage]['epochs']:
                    raise ValueError(f'Complete the {stage} training stage first')
                checkpoint = (receipt_path.parent / receipt['checkpoint']).resolve()
                if checkpoint.parent != receipt_path.parent.resolve():
                    raise ValueError('Checkpoint must belong to its training stage')
                checked(checkpoint, receipt['sha256'])
                stages[stage] = receipt['sha256']
                files.extend([checkpoint, receipt_path])
            assets = output / 'assets.json'
            files.extend((ROOT / 'baselines/lwm_vjepa/source').rglob('*.py'))
            write_json(assets, {'assets': [identity(path) for path in sorted(set(files))]})
            cfg.update(training_config=str(args.training_config.resolve()), evaluation_manifest=str(assets),
                       wm_sha256=stages['wm'], policy_sha256=stages['rl'])
    elif method == 'nomad':
        cfg = read_json(ROOT / 'baselines/nomad/evaluate.template.json')
        cfg.update(official_root=str(ROOT / 'baselines/nomad/official'),
                   diffusion_root=str(ROOT / 'baselines/nomad/diffusion_dependency'),
                   checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint))
    else:
        runtime = args.rae_root.resolve()
        for relative in ['scripts/official_bootstrap.py', 'scripts/run_rae_stream_habitat.py', 'planning_eval.py']:
            if not (runtime / relative).is_file():
                raise FileNotFoundError(runtime / relative)
        baseline_settings = read_json(ROOT / 'data/baseline_settings.json')
        settings = baseline_settings['RAE-NWM']
        protocol = baseline_settings['common_evaluation']
        cfg = dict(rae_root=str(runtime), checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=sha256(args.checkpoint), max_steps=protocol['max_environment_steps'],
                   reach_radius=protocol['arrival_geodesic_distance_m_less_than'],
                   integration_steps=settings['planning']['euler_points_evaluated'],
                   dt=settings['controller']['integration_seconds'],
                   max_translation=settings['controller']['max_displacement_m'],
                   max_yaw=math.radians(settings['controller']['max_yaw_degrees']))
    cfg.update(common)
    path = output / 'evaluate.json'
    write_json(path, cfg)
    return [str(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    for name in ['mp3d-archive', 'vjepa-source', 'vjepa-checkpoint', 'habitat-lab-source', 'habitat-sim-source',
                 'habitat-prefix', 'checkpoint', 'wm-checkpoint', 'policy-checkpoint', 'training-config', 'rae-root']:
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    paths = prepare_injepa(args) if args.method == 'injepa' else prepare_baseline(args)
    print(json.dumps({'method': args.method, 'configurations': paths}, indent=2))


if __name__ == '__main__':
    main()
