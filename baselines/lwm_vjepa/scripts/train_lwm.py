#!/usr/bin/env python3
"""Run one admitted LWM stage; launch with python or externally configured torchrun."""
import argparse
import json
import os
from pathlib import Path

from lwm_stream.entry import preflight, run_training, run_pseudo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('wm', 'pseudo', 'il', 'rl'))
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--smoke-updates', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--micro-batch', type=int)
    parser.add_argument('--cpu-threads', type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    original_output = Path(config['output']).resolve()
    if args.output:
        config['output'] = str(args.output)
    if args.smoke_updates is not None and (
            args.smoke_updates < 1 or args.output is None or args.output.resolve() == original_output
            or args.stage == 'pseudo'):
        parser.error('smoke requires positive updates and a separate --output; pseudo is generated once after WM')
    if args.stage == 'pseudo' and args.resume:
        parser.error('pseudo admits completed cache reuse only, not a partial training resume')
    for argument, key in ((args.workers, 'workers'), (args.micro_batch, 'micro_batch')):
        if argument is not None:
            config[key] = argument
    admission = preflight(config, args.stage)
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    if args.preflight:
        if rank == 0:
            print(json.dumps({'status': 'PREFLIGHT_PASS', 'stage': args.stage, **admission}, indent=2))
        return
    import torch
    import torch.distributed as dist
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
    device = torch.device('cpu')
    if args.device == 'cuda':
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        if args.stage == 'pseudo':
            result = run_pseudo(config, admission, device, rank=rank, world_size=world)
        else:
            result = run_training(config, args.stage, admission, device, rank=rank, world_size=world,
                                  resume=args.resume, smoke_updates=args.smoke_updates)
        if rank == 0:
            print(json.dumps(result, indent=2))
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
