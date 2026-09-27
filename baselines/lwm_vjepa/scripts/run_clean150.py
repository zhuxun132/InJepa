"""V2 factory; unchanged V1 clean150 planner, controller and Habitat runner."""
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
from bridge import ImageTransform, ModelABI

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    from lwm_official_eval.backend import validate_assets, RGBBackend
    validate_assets(cfg['evaluation_manifest'])
    from lwm_stream.entry import read_json,preflight,scientific_identity,checked,construct_model,load_completed,write_json
    from lwm_stream.vjepa_frontend import VJEPAImageTransform
    from lwm import ActionTokenizer
    from lwm_official_eval.adapter import OfficialLWMPlanner
    from lwm_official_eval.cli import run
    torch.set_num_threads(cfg['cpu_threads'])
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    config=read_json(cfg['training_config'])
    admission=preflight(config,'rl')
    directory=Path(config['output'])/'rl'
    receipt=read_json(directory/'COMPLETE.json')
    if receipt['status']!='COMPLETE' or receipt['scientific']!=scientific_identity(config,'rl',admission):
        raise ValueError('final RL scientific identity mismatch')
    if receipt['progress']['epoch']!=config['rl']['epochs'] or receipt['progress']['sampler_offset']!=0:
        raise ValueError('incomplete RL budget')
    checkpoint=directory/receipt['checkpoint']
    if checkpoint.resolve().parent!=directory.resolve():raise ValueError('checkpoint outside RL directory')
    checked(checkpoint,receipt['sha256'])
    if receipt['sha256']!=cfg['policy_sha256'] or admission['predecessors']['wm']['sha256']!=cfg['wm_sha256']:
        raise ValueError('evaluation checkpoint selection mismatch')
    policy,wm=construct_model(config,'policy'),construct_model(config,'wm')
    load_completed(policy,dict(path=str(checkpoint),sha256=receipt['sha256'],scientific=receipt['scientific']),config['rl']['epochs'])
    load_completed(wm,admission['predecessors']['wm'],config['wm']['epochs'])
    device=torch.device(cfg['device'])
    policy.to(device).eval();wm.to(device).eval()
    planner=OfficialLWMPlanner(policy=ModelABI(policy),world_model=ModelABI(wm),
        tokenizer=ActionTokenizer(Path(config['codebooks'])/'action_centers.json'),
        mode=cfg['mode'],num_samples=cfg['num_samples'],temperature=cfg['temperature'])
    backend=RGBBackend(planner,ImageTransform(VJEPAImageTransform(config['vision']['source_root'])),device=device)
    write_json(Path(a.config).parent/'LOADED.json',{'status':'STRICT_LOADED','rl':receipt,'wm':admission['predecessors']['wm'],
        'execution_mode':cfg['execution_mode'],'gpu':torch.cuda.get_device_name(0)})
    print('STRICT_LOADED; starting frozen clean150 runner',flush=True)
    run(cfg,backend)

if __name__=='__main__':main()
