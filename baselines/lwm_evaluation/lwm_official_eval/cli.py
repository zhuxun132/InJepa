"""Preparation and closed-loop wiring to the existing Habitat runner."""
from .adapter import EmptyPlanError


def finalize_empty_plan(error,steps,metrics):
    if not isinstance(error,EmptyPlanError):raise TypeError('typed empty-plan error required')
    if type(steps) is not int or steps<0:raise ValueError('actual step count required')
    import numbers,math
    scalar_metrics={}
    for key in ('distance_to_goal','start_distance','path_length'):
        value=metrics.get(key)
        if isinstance(value,numbers.Real) and math.isfinite(value):scalar_metrics[key]=float(value)
    return {**scalar_metrics,'num_steps':steps,'steps':steps,'arrival_success':0.,'reach_spl':0.,
            'success':0.,'spl':0.,'termination_reason':'EMPTY_PLAN','stop_called':False,
            'empty_plan':error.result}


def build_backend(config):
    """Load pinned assets, then call original released constructors strictly."""
    from pathlib import Path
    import importlib.util,sys,torch,random,numpy as np
    from .backend import validate_assets,RGBBackend
    from .adapter import OfficialLWMPlanner
    admitted=set(validate_assets(config['release_manifest']).values())
    root=Path(config['official_root']).resolve()
    required={root/'inference.py',root/'tokenizer/zed2_64.json',root/'tokenizer/zed2_traj.json',
              Path(config['wm_checkpoint']).resolve(),Path(config['policy_checkpoint']).resolve()}
    required.update(p.resolve() for p in (root/'lwm').rglob('*.py'))
    if not required <= admitted:
        raise ValueError('loaded source/checkpoint/codebook path not admitted by release manifest')
    for name,module in list(sys.modules.items()):
        if name=='lwm' or name.startswith('lwm.'):
            origin=getattr(module,'__file__',None)
            if origin is not None and not Path(origin).resolve().is_relative_to(root):
                raise RuntimeError('LWM already imported from another source')
    sys.path.insert(0,str(root))
    spec=importlib.util.spec_from_file_location('released_lwm_inference',root/'inference.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    device=torch.device(config['device'])
    wm=official.build_world_model(config['wm_checkpoint'],device)
    policy=official.build_policy(config['policy_checkpoint'],device) if config['mode']=='policy_wm' else None
    tokenizer=official.ActionTokenizer(str(root/'tokenizer/zed2_64.json'))
    codebook=official.load_kmeans_trajectories(str(root/'tokenizer/zed2_traj.json'),normalize=False).to(device)
    planner=OfficialLWMPlanner(policy=policy,world_model=wm,tokenizer=tokenizer,
        mode=config['mode'],num_samples=config['num_samples'],temperature=config['temperature'],codebook_metric=codebook)
    return RGBBackend(planner,official.get_image_transform(),device=device)


def build_policy(config, backend):
    from .adapter import Policy
    mode = config.get('execution_mode', 'first_waypoint')
    if mode == 'first_waypoint':
        return Policy(backend)
    if mode == 'full_sequence':
        from .full_sequence import FullSequencePolicy
        return FullSequencePolicy(backend, **{k: config[k] for k in ('dt','max_translation','max_yaw')})
    raise ValueError('unknown execution_mode')


def run(config,backend):
    """One real control action per decision through the already reviewed runner."""
    import sys,importlib,importlib.util,json,gzip,hashlib
    from pathlib import Path
    from .adapter import Policy
    from .controller import waypoint_to_velocity
    root=Path(config['runner_root']).resolve()
    manifest_path=Path(config['runner_manifest'])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=config['runner_manifest_sha256']:
        raise ValueError('frozen runner manifest SHA mismatch')
    runner_files=json.loads(manifest_path.read_text())
    if not runner_files:raise ValueError('empty runner source manifest')
    for relative,entry in runner_files.items():
        source_file=root/relative
        if source_file.stat().st_size!=entry['bytes'] or hashlib.sha256(source_file.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('frozen runner source mismatch: '+relative)
    from habitat.config.default import get_config
    from habitat.config.default_structured_configs import ActionConfig
    from omegaconf import open_dict,read_write
    root=Path(config['runner_root']).resolve();sys.path.insert(0,str(root))
    runner=importlib.import_module('j2j.evaluation.habitat_runner')
    assert Path(runner.__file__).resolve()==root/'j2j/evaluation/habitat_runner.py'
    # Reuse only the simulator task seam, no RAE policy/model/decoder imports.
    task_file=Path(config['continuous_task_file']).resolve()
    assert hashlib.sha256(task_file.read_bytes()).hexdigest()==config['continuous_task_sha256']
    spec=importlib.util.spec_from_file_location('existing_continuous_task',task_file)
    task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task);task.register_continuous_action()
    sensor=Path(config['sensor_config']);ledger=Path(config['episodes_path'])
    assert hashlib.sha256(sensor.read_bytes()).hexdigest()==config['sensor_sha256']
    assert hashlib.sha256(ledger.read_bytes()).hexdigest()==config['ledger_sha256']
    with gzip.open(ledger,'rt') as f:episodes=json.load(f)['episodes']
    indices=config['episode_indices'];assert len(indices)==len(set(indices)) and all(0<=i<len(episodes) for i in indices)
    habitat_config=get_config(str(sensor))
    with read_write(habitat_config),open_dict(habitat_config):
        habitat_config.habitat.task.actions.rae_continuous=ActionConfig(type='RAEContinuousVelocityAction')
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    for name in ['episodes','traces','videos']:(out/name).mkdir()
    env=runner.load_habitat_environment(config=habitat_config,episodes_path=str(ledger),
        scenes_dir=config['scenes_dir'],episode_indices=indices)
    policy=build_policy(config,backend);results=[]
    try:
        from j2j.evaluation.video import FirstPersonVideoRecorder
        for index in indices:
            row=episodes[index];scene=Path(row['scene_id']).stem;identity=[scene,str(row['episode_id'])]
            video=FirstPersonVideoRecorder(out/'videos'/f'episode_{index:06d}.mp4',fps=config['video_fps'],
                episode_key=identity,identities={'ledger_index':index,'episode_ledger_sha256':config['ledger_sha256']})
            executed=[];frames={'steps':0};reset_hashes={}
            with (out/'traces'/f'episode_{index:06d}.steps.jsonl').open('x') as trace:
                def handler(raw):
                    control=waypoint_to_velocity(raw['continuous_action'],dt=config['dt'],
                        max_translation=config['max_translation'],max_yaw=config['max_yaw'])
                    executed.append(control)
                    return {'name':'CONTINUOUS','is_stop':False,'habitat_payload':{'action':'rae_continuous',
                        'action_args':{k:control[k] for k in ['linear_velocity','angular_velocity','time_step']}}}
                def observer(event):
                    video(event)
                    if event['phase']=='reset':
                        reset_hashes.update({k:hashlib.sha256(event[k].tobytes()).hexdigest() for k in ['rgb','imagegoal']})
                    if event['phase']=='step':
                        frames['steps']=event['step']
                        trace.write(json.dumps({'ledger_index':index,'step':event['step'],'plan':policy.last_plan,
                            'control':executed[-1],'distance_to_goal':float(env.get_metrics()['distance_to_goal'])},allow_nan=False)+'\n');trace.flush()
                try:
                    result=runner.run_imagegoal_episode(env,policy,max_steps=config['max_steps'],success_distance=config['reach_radius'],
                        diagnostic_reach_radius=config['reach_radius'],expected_episode_id=str(row['episode_id']),expected_scene_id=scene,
                        continuous_action_handler=handler,frame_observer=observer)
                except EmptyPlanError as error:
                    result=finalize_empty_plan(error,frames['steps'],env.get_metrics())
                finally:
                    video_record=video.close()
            result.update(ledger_index=index,ledger_episode_id=str(row['episode_id']),ledger_scene_id=scene,
                first_person_video=video_record,reset_rgb_sha256=reset_hashes)
            (out/'episodes'/f'episode_{index:06d}.json').write_text(json.dumps(result,allow_nan=False));results.append(result)
            print(json.dumps({'event':'episode_complete','ledger_index':index,'arrival_success':result['arrival_success'],'steps':result['num_steps']}),flush=True)
    finally:
        policy.close();env.close()
    n=len(results)
    final={'completed':n,'target':len(indices),'arrival_SR':sum(x['arrival_success'] for x in results)/n,
        'reach_SPL':sum(x['reach_spl'] for x in results)/n,'episodes':results}
    (out/'RESULTS.json').write_text(json.dumps(final,allow_nan=False))


def main():
    import argparse,json,time,torch
    from pathlib import Path
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run',action='store_true',help='Launch closed-loop evaluation; default only loads and prepares')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    torch.set_num_threads(config['cpu_threads'])
    start=time.perf_counter();backend=build_backend(config)
    print(json.dumps({'event':'PRETRAINED_MODELS_STRICT_LOADED','device':config['device'],
        'seconds':time.perf_counter()-start,'mode':config['mode']}),flush=True)
    if args.run:run(config,backend)


if __name__=='__main__':main()
