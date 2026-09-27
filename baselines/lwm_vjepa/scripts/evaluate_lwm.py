#!/usr/bin/env python3
"""LWM policy adapter on the existing Habitat ImageGoal loop and metrics."""
import argparse
import json
from pathlib import Path
import site
import sys

from lwm_stream.entry import checked, construct_model, load_completed, preflight, read_json, scientific_identity, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--habitat-config', type=Path, required=True)
    p.add_argument('--habitat-site', type=Path, required=True)
    p.add_argument('--episodes', type=Path)
    p.add_argument('--scenes', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--episode-index', type=int, action='append')
    p.add_argument('--num-sample', type=int, default=32)
    p.add_argument('--device', default='cuda')
    p.add_argument('--check-runtime', action='store_true')
    a = p.parse_args()
    # Append the existing Habitat environment read-only; the dedicated LWM torch stays first.
    site.addsitedir(str(a.habitat_site))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/existing_habitat_runner'))
    import habitat
    import habitat_sim
    from habitat.config.default import get_config
    from habitat.config import read_write
    from habitat.config.default_structured_configs import StopActionConfig, VelocityControlActionConfig
    from omegaconf import OmegaConf
    from j2j.evaluation.habitat_runner import load_habitat_environment, run_imagegoal_episode
    from j2j.evaluation.metrics import aggregate_episode_metrics
    from lwm_stream.navigation import LWMNavigationAdapter, VelocityHandler
    cfg = get_config(str(a.habitat_config))
    velocity = VelocityControlActionConfig()
    with read_write(cfg):
        cfg.habitat.task.actions = {'stop': StopActionConfig(), 'velocity_control': velocity}
    handler = VelocityHandler(linear_range=velocity.lin_vel_range, angular_range_deg=velocity.ang_vel_range,
                              time_step=velocity.time_step, min_abs_linear=velocity.min_abs_lin_speed,
                              min_abs_angular_deg=velocity.min_abs_ang_speed)
    if str(habitat.__version__) != '0.2.4' or str(habitat_sim.__version__) != '0.2.4':
        raise ValueError('requires the existing official Habitat 0.2.4 pair')
    if a.check_runtime:
        print(json.dumps({'status': 'RUNTIME_CONFIG_PASS', 'habitat': habitat.__version__,
                          'habitat_sim': habitat_sim.__version__, 'velocity': OmegaConf.to_container(OmegaConf.structured(velocity)),
                          'straight_probe': handler({'continuous_action': {'waypoint_m': [.2, 0.], 'stop': False}})}, indent=2))
        return
    if not all((a.episodes, a.scenes, a.output)):
        p.error('evaluation requires --episodes, --scenes and a new --output')
    config = read_json(a.config)
    admission = preflight(config, 'rl')
    directory = Path(config['output']) / 'rl'
    receipt = read_json(directory / 'COMPLETE.json')
    if receipt['status'] != 'COMPLETE' or receipt['scientific'] != scientific_identity(config, 'rl', admission):
        raise ValueError('completed matching RL checkpoint required')
    checkpoint = directory / receipt['checkpoint']
    if checkpoint.resolve().parent != directory.resolve():
        raise ValueError('RL checkpoint must be inside its stage directory')
    checked(checkpoint, receipt['sha256'])
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    from lwm import ActionTokenizer
    policy, wm = construct_model(config, 'policy'), construct_model(config, 'wm')
    load_completed(policy, dict(path=str(checkpoint), **{k: receipt[k] for k in ('sha256', 'scientific')}), config['rl']['epochs'])
    load_completed(wm, admission['predecessors']['wm'], config['wm']['epochs'])
    device = torch.device(a.device)
    transform = None
    if 'vision' in config:
        from lwm_stream.vjepa_frontend import VJEPAImageTransform
        transform = VJEPAImageTransform(config['vision']['source_root'])
    adapter = LWMNavigationAdapter(policy.to(device), wm.to(device),
                                    ActionTokenizer(Path(config['codebooks']) / 'action_centers.json'),
                                    device=device, num_sample=a.num_sample, transform=transform)
    env = load_habitat_environment(config=cfg, episodes_path=str(a.episodes), scenes_dir=str(a.scenes),
                                   episode_indices=a.episode_index)
    write_json(a.output / 'IDENTITY.json', {'rl': receipt['sha256'], 'wm': admission['predecessors']['wm']['sha256'],
               'episodes_sha256': sha256(a.episodes), 'episode_indices': a.episode_index,
               'habitat_config': OmegaConf.to_container(cfg, resolve=True), 'num_sample': a.num_sample,
               'controller': 'provisional first-waypoint nonholonomic native VelocityAction'})
    results = []
    try:
        for _ in range(len(env.episodes)):
            result = run_imagegoal_episode(env, adapter, continuous_action_handler=handler,
                    max_steps=int(cfg.habitat.environment.max_episode_steps),
                    success_distance=float(cfg.habitat.task.measurements.success.success_distance))
            results.append(result)
            with (a.output / 'episodes.jsonl').open('a') as stream:
                stream.write(json.dumps(result) + '\n')
        write_json(a.output / 'METRICS.json', aggregate_episode_metrics(results))
    finally:
        env.close()


if __name__ == '__main__':
    main()
