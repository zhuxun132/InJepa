"""Official NoMaD goal-conditioned policy; ROS transport replaced by Habitat."""
import ast,json,time,sys,hashlib,random,argparse,importlib.util
from pathlib import Path
import numpy as np
import torch,yaml
from PIL import Image
from bridge import control_payload,update_history

def functions(path,names,namespace):
 tree=ast.parse(Path(path).read_text())
 nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
 return namespace

class Policy:
 def __init__(self,c):
  self.c=c;root=Path(c['official_root']);sys.path[:0]=[str(root/'train'),c['diffusion_root']]
  from vint_train.models.nomad.nomad import NoMaD,DenseNetwork
  from vint_train.models.nomad.nomad_vint import NoMaD_ViNT,replace_bn_with_gn
  from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
  from torchvision import transforms
  import torchvision.transforms.functional as TF
  import typing
  from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
  p=yaml.safe_load((root/'train/config/nomad.yaml').read_text())
  state=torch.load(c['checkpoint'],map_location='cpu',weights_only=True)
  # Published checkpoint is the authority for its positional context capacity.
  pe=state['vision_encoder.positional_encoding.pos_enc']
  p['context_size']=int(pe.shape[1])-2
  ns=dict(torch=torch,nn=torch.nn,NoMaD=NoMaD,DenseNetwork=DenseNetwork,NoMaD_ViNT=NoMaD_ViNT,
   replace_bn_with_gn=replace_bn_with_gn,ConditionalUnet1D=ConditionalUnet1D,transforms=transforms,
   TF=TF,PILImage=Image,np=np,**{k:getattr(typing,k) for k in ['List','Tuple','Dict','Optional']})
  functions(root/'deployment/src/utils.py',['load_model','transform_images','to_numpy'],ns)
  self.model=ns['load_model'](c['checkpoint'],p,torch.device('cuda')).eval()
  self.model.load_state_dict(state,strict=True)
  self.transform=ns['transform_images'];self.p=p
  stats=yaml.safe_load((root/'train/vint_train/data/data_config.yaml').read_text())['action_stats']
  ns.update(ACTION_STATS={k:np.array(v) for k,v in stats.items()},from_numpy=lambda x:torch.from_numpy(x).float())
  functions(root/'train/vint_train/training/train_utils.py',['unnormalize_data','get_action'],ns)
  self.get_action=ns['get_action']
  robot=yaml.safe_load((root/'deployment/config/robot.yaml').read_text());self.dt=1/robot['frame_rate'];self.scale=robot['max_v']/robot['frame_rate']
  ns.update(DT=self.dt,MAX_V=robot['max_v'],MAX_W=robot['max_w'],EPS=1e-8,Tuple=typing.Tuple)
  functions(root/'deployment/src/pd_controller.py',['clip_angle','pd_controller'],ns);self.controller=ns['pd_controller']
  self.scheduler=DDPMScheduler(num_train_timesteps=p['num_diffusion_iters'],beta_schedule='squaredcos_cap_v2',clip_sample=True,prediction_type='epsilon')
  self.history=[];self.last_plan=None
  print(json.dumps({'event':'STRICT_LOADED','context_size':p['context_size'],'parameters':sum(x.numel() for x in self.model.parameters()),'diffusion_steps':p['num_diffusion_iters']}),flush=True)
 def reset(self,goal_rgb):self.history=[];self.last_plan=None
 @torch.inference_mode()
 def act(self,current_rgb,goal_rgb,factual_history):
  start=time.perf_counter()
  self.history=update_history(self.history,Image.fromarray(current_rgb.copy()),self.p['context_size']+1)
  obs=self.transform(self.history,self.p['image_size'],center_crop=False).cuda()
  goal=self.transform(Image.fromarray(goal_rgb),self.p['image_size'],center_crop=False).cuda()
  cond=self.model('vision_encoder',obs_img=obs,goal_img=goal,input_goal_mask=torch.zeros(1,dtype=torch.long,device='cuda'))
  action=torch.randn((self.c['num_samples'],self.p['len_traj_pred'],2),device='cuda')
  self.scheduler.set_timesteps(self.p['num_diffusion_iters'])
  for k in self.scheduler.timesteps:
   noise=self.model('noise_pred_net',sample=action,timestep=k,global_cond=cond.repeat(self.c['num_samples'],1))
   action=self.scheduler.step(model_output=noise,timestep=k,sample=action).prev_sample
  paths=self.get_action(action).cpu().numpy()
  if not np.isfinite(paths).all():raise ValueError('nonfinite official output')
  waypoint=paths[0,self.c['waypoint_index']]*self.scale
  v,w=self.controller(waypoint)
  self.last_plan={'waypoint':waypoint.tolist(),'winner':0,'planning_seconds':time.perf_counter()-start,'v':float(v),'w':float(w)}
  return {'continuous_action':control_payload(v,w,self.c['execution_seconds'])}
 def close(self):pass

def main():
 a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--smoke',action='store_true');args=a.parse_args()
 c=json.loads(Path(args.config).read_text());torch.set_num_threads(c['cpu_threads'])
 random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed'])
 assert hashlib.sha256(Path(c['checkpoint']).read_bytes()).hexdigest()==c['checkpoint_sha256']
 policy=Policy(c)
 if args.smoke:
  im=np.zeros((480,640,3),dtype=np.uint8);policy.reset(im)
  for _ in range(3):print(policy.act(im,im,[]),flush=True)
  print('SMOKE_PASS',torch.cuda.max_memory_allocated(),flush=True);return
 sys.path.insert(0,c['runner_root'])
 from j2j.evaluation import habitat_runner as runner
 from habitat.config.default import get_config
 from habitat.config.default_structured_configs import ActionConfig
 from omegaconf import open_dict,read_write
 spec=importlib.util.spec_from_file_location('nomad_continuous_task',c['continuous_task_file']);task=importlib.util.module_from_spec(spec);spec.loader.exec_module(task);task.register_continuous_action()
 import gzip
 ledger=Path(c['episodes_path']);assert hashlib.sha256(ledger.read_bytes()).hexdigest()==c['ledger_sha256']
 with gzip.open(ledger,'rt') as f:episodes=json.load(f)['episodes']
 cfg=get_config(c['sensor_config'])
 with read_write(cfg),open_dict(cfg):cfg.habitat.task.actions.rae_continuous=ActionConfig(type='RAEContinuousVelocityAction')
 out=Path(c['output']);out.mkdir(parents=True,exist_ok=False);(out/'episodes').mkdir();(out/'traces').mkdir();(out/'config.json').write_text(json.dumps(c,indent=2))
 env=runner.load_habitat_environment(config=cfg,episodes_path=str(ledger),scenes_dir=c['scenes_dir'],episode_indices=c['episode_indices'])
 try:
  for index in c['episode_indices']:
   row=episodes[index];scene=Path(row['scene_id']).stem
   with (out/'traces'/f'{index:06d}.jsonl').open('x') as trace:
    def handler(raw):return {'name':'CONTINUOUS','is_stop':False,'habitat_payload':{'action':'rae_continuous','action_args':raw['continuous_action']}}
    def observer(event):
     if event['phase']=='step':
      trace.write(json.dumps({'step':event['step'],'plan':policy.last_plan,'distance_to_goal':float(env.get_metrics()['distance_to_goal'])})+'\n');trace.flush()
    result=runner.run_imagegoal_episode(env,policy,max_steps=c['max_steps'],success_distance=c['reach_radius'],diagnostic_reach_radius=c['reach_radius'],expected_episode_id=str(row['episode_id']),expected_scene_id=scene,continuous_action_handler=handler,frame_observer=observer)
   result.update(ledger_index=index,ledger_scene_id=scene,ledger_episode_id=str(row['episode_id']))
   (out/'episodes'/f'{index:06d}.json').write_text(json.dumps(result,allow_nan=False))
   print(json.dumps({'event':'episode_complete','index':index,**{k:result[k] for k in ['arrival_success','reach_spl','num_steps']}}),flush=True)
 finally:env.close()
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
