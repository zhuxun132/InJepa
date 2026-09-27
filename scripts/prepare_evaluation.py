"""Bind Clean150, local assets and original evaluator configurations."""
import argparse
import json
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
    for variant in ['full', 'p_only', 'posthoc_f']:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--scene-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    for name in ['mp3d-archive', 'vjepa-source', 'vjepa-checkpoint', 'habitat-lab-source', 'habitat-sim-source',
                 'habitat-prefix']:
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    paths = prepare_injepa(args)
    print(json.dumps({'method': args.method, 'configurations': paths}, indent=2))


if __name__ == '__main__':
    main()
