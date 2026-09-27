"""Final CLI admits scientific predecessors before native models or CUDA."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source')); sys.path.insert(0,str(HERE/'source/official'))
from lwm_stream.entry import preflight,scientific_identity


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value): path.write_text(json.dumps(value)); return sha(path)

@pytest.fixture
def chain(tmp_path,monkeypatch):
    cfg=json.loads((HERE/'configs/lwm_stream_train.json').read_text())
    books=tmp_path/'codebooks'; books.mkdir(); output=tmp_path/'runs'; output.mkdir()
    cfg.update(codebooks=str(books),output=str(output),croco=str(tmp_path/'croco.pth'))
    torch.save({'model':{},'croco_kwargs':{}},cfg['croco']); cfg['croco_sha256']=sha(cfg['croco'])
    status={'status':'CPU_CODEBOOKS_READY','partitions':{},'books':{},'input_sha256':{},'parameters':{'n_clusters':64}}
    hashes={'croco':cfg['croco_sha256']}
    for partition in ('wm_il','rl','dev'):
        path=books/f'{partition}.jsonl'
        row={'source':'r2r','video':f'{partition}_r2r_1','split':'dev' if partition=='dev' else 'train','pose_key':partition,'keyframe_indices':[0,1]}
        path.write_text(json.dumps(row)+'\n'); hashes[partition]=sha(path)
        status['partitions'][partition]={'rows':1,'physical_trajectories':1,'windows':1,'sha256':sha(path)}
    for name,shape in [('action_centers',(64,2)),('trajectory_centers',(64,63,3))]:
        path=books/f'{name}.json'; dump(path,np.zeros(shape).tolist()); hashes[name]=sha(path)
        status['books'][name]={'shape':list(shape),'sha256':sha(path),'samples':64,'weight_sum':64.}
    dump(books/'STATUS.json',status)
    completed={}
    def complete_stage(stage,admission):
        directory=output/stage; directory.mkdir(); checkpoint=directory/'checkpoint-final.pt'
        scientific=scientific_identity(cfg,stage,admission)
        progress={'epoch':cfg[stage]['epochs'],'update':cfg[stage]['epochs'],'sampler_offset':0}
        payload={'format':'LWM_STREAM_LOCAL_STATE_V1','model':{'weight':torch.ones(1)},'optimizer':{'state':{},'param_groups':[]},'parameter_names':[],
                 'identity':{'scientific':scientific,'execution':{'world_size':1}},
                 'progress':{'training':progress,'rank_rng':[]}}
        torch.save(payload,checkpoint)
        completed[stage]={'status':'COMPLETE','checkpoint':checkpoint.name,'sha256':sha(checkpoint),'scientific':scientific,'progress':progress}
        dump(directory/'COMPLETE.json',completed[stage])
    complete_stage('wm',{'hashes':hashes,'predecessors':{},'pseudo':None})
    pseudo=output/'pseudo'; pseudo.mkdir(); tokens=np.full((1,65),66,dtype=np.int16); tokens[0,:3]=[64,2,65]; np.save(pseudo/'tokens.npy',tokens)
    ready={'status':'READY','tokens':'tokens.npy','sha256':sha(pseudo/'tokens.npy'),
           'identity':{'wm':completed['wm']['sha256'],'wm_il':hashes['wm_il'],'action_centers':hashes['action_centers'],'trajectory_centers':hashes['trajectory_centers'],'seed':cfg['seed']}}
    dump(pseudo/'READY.json',ready)
    wm_predecessor={'path':str(output/'wm/checkpoint-final.pt'),'sha256':completed['wm']['sha256'],'scientific':completed['wm']['scientific']}
    complete_stage('il',{'hashes':hashes,'predecessors':{'wm':wm_predecessor},'pseudo':ready})
    # Guards are installed after writing tiny metadata fixtures, before every preflight.
    from lwm.world_model import LatentWorldModel
    from lwm.policy import ARPlusPolicy
    def forbidden(*args,**kwargs): pytest.fail('preflight must not construct native models or initialize CUDA')
    monkeypatch.setattr(LatentWorldModel,'__init__',forbidden); monkeypatch.setattr(ARPlusPolicy,'__init__',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden); monkeypatch.setattr(torch.cuda,'set_device',forbidden)
    return cfg,hashes,completed,ready

@pytest.mark.parametrize('stage',['wm','pseudo','il','rl'])
def test_valid_stage_predecessors_before_model_or_cuda(chain,stage):
    cfg,hashes,complete,ready=chain
    result=preflight(cfg,stage)
    assert isinstance(result,dict)
    for name,value in hashes.items(): assert result['hashes'][name]==value
    if stage!='wm':
        assert result['predecessors']['wm']['sha256']==complete['wm']['sha256']
        assert Path(result['predecessors']['wm']['path'])==Path(cfg['output'])/'wm/checkpoint-final.pt'
    if stage=='rl': assert result['predecessors']['il']['sha256']==complete['il']['sha256']
    if stage=='il': assert result['pseudo']==ready

@pytest.mark.parametrize('mutation',['split_sha','center_sha','codebooks_partial','croco_sha'])
def test_wm_rejects_unadmitted_data_or_initialization_before_model(chain,mutation):
    cfg,*_=chain; books=Path(cfg['codebooks'])
    if mutation=='split_sha':
        path=books/'wm_il.jsonl'; path.write_text(path.read_text()+'\n')
    elif mutation=='center_sha':
        path=books/'trajectory_centers.json'; path.write_text(path.read_text()+'\n')
    elif mutation=='codebooks_partial':
        path=books/'STATUS.json'; status=json.loads(path.read_text()); status['status']='PARTIAL'; dump(path,status)
    else: Path(cfg['croco']).write_bytes(b'changed initialization')
    with pytest.raises(ValueError): preflight(cfg,'wm')

@pytest.mark.parametrize('stage,predecessor',[('pseudo','wm'),('il','wm'),('rl','il')])
def test_smoke_partial_never_counts_as_completed_predecessor(chain,stage,predecessor):
    cfg,*_=chain; path=Path(cfg['output'])/predecessor/'COMPLETE.json'
    receipt=json.loads(path.read_text()); receipt['status']='SMOKE_PARTIAL'; dump(path,receipt)
    with pytest.raises(ValueError): preflight(cfg,stage)

@pytest.mark.parametrize('binding',['wm','wm_il','action_centers','trajectory_centers'])
def test_il_rejects_pseudo_identity_from_other_training_chain(chain,binding):
    cfg,_,_,ready=chain; changed=copy.deepcopy(ready); changed['identity'][binding]='0'*64
    dump(Path(cfg['output'])/'pseudo/READY.json',changed)
    with pytest.raises(ValueError): preflight(cfg,'il')


def test_rl_rejects_corrupted_completed_wm_weights(chain):
    cfg,*_=chain; path=Path(cfg['output'])/'wm/checkpoint-final.pt'; path.write_bytes(b'corrupted')
    with pytest.raises(ValueError): preflight(cfg,'rl')


class EntryTinyDataset(torch.utils.data.Dataset):
    def __len__(self): return 6
    def __getitem__(self,key):
        epoch,index=key
        return {'x':torch.tensor([float(index+1)]),'y':torch.tensor([float(index)*.2]),'occurrence':torch.tensor([epoch,index])}


class EntryTinyObjective:
    def __init__(self,model,seen): self.model=model; self.training_model=model; self.seen=seen; self.stage='wm'
    def set_mode(self): self.model.train()
    def terms(self,model,batch):
        self.seen.extend(tuple(v) for v in batch['occurrence'].tolist())
        loss=(model(batch['x'])-batch['y']).square().sum()
        return loss,torch.tensor(len(batch['x']),dtype=torch.int64)


def test_run_training_real_engine_smoke_is_partial_and_resume_matches_full_epochs(chain,tmp_path,monkeypatch):
    import lwm_stream.entry as entry
    from test_engine import TinyModel,nested_equal
    cfg,hashes,_,_=chain
    cfg=copy.deepcopy(cfg); cfg['wm'].update(epochs=2,global_batch=2,lr=.02,warmup_epochs=1)
    cfg.update(workers=0,micro_batch=1,checkpoint_updates=1000,log_updates=1)
    admission={'hashes':hashes,'predecessors':{},'pseudo':None}
    seen=[]
    def build_model(config,stage,admission,device):
        assert stage=='wm' and torch.device(device).type=='cpu'
        model=TinyModel()
        return model,EntryTinyObjective(model,seen)
    monkeypatch.setattr(entry,'build_model',build_model,raising=False)
    monkeypatch.setattr(entry,'build_dataset',lambda config,stage,admission:EntryTinyDataset(),raising=False)
    full_cfg=copy.deepcopy(cfg); full_cfg['output']=str(tmp_path/'full')
    entry.run_training(full_cfg,'wm',admission,torch.device('cpu'))
    full_root=Path(full_cfg['output'])/'wm'; full_receipt=json.loads((full_root/'COMPLETE.json').read_text())
    assert full_receipt['status']=='COMPLETE'
    full_state=torch.load(full_root/full_receipt['checkpoint'],weights_only=True)
    full_seen=list(seen); seen.clear()
    partial_cfg=copy.deepcopy(cfg); partial_cfg['output']=str(tmp_path/'resumed')
    entry.run_training(partial_cfg,'wm',admission,torch.device('cpu'),smoke_updates=1)
    partial_root=Path(partial_cfg['output'])/'wm'
    assert not (partial_root/'COMPLETE.json').exists()
    smoke=json.loads((partial_root/'SMOKE_PARTIAL.json').read_text()); assert smoke['status']=='SMOKE_PARTIAL'
    latest=json.loads((partial_root/'LATEST.json').read_text()); checkpoint=partial_root/latest['checkpoint']
    partial=torch.load(checkpoint,weights_only=True)
    assert partial['progress']['training']=={'epoch':0,'update':1,'sampler_offset':1}
    assert len(seen)==2
    entry.run_training(partial_cfg,'wm',admission,torch.device('cpu'),resume=str(checkpoint))
    final=json.loads((partial_root/'COMPLETE.json').read_text()); assert final['status']=='COMPLETE'
    resumed=torch.load(partial_root/final['checkpoint'],weights_only=True)
    assert resumed['progress']['training']==full_state['progress']['training']=={'epoch':2,'update':6,'sampler_offset':0}
    assert seen==full_seen and len(seen)==12 and len(set(seen))==12
    nested_equal(resumed['model'],full_state['model']); nested_equal(resumed['optimizer'],full_state['optimizer'])


@pytest.mark.parametrize('mutation',['wm_lr','data_identity','source_identity'])
def test_preflight_rejects_completed_wm_with_different_scientific_identity(chain,mutation):
    cfg,*_=chain
    if mutation=='wm_lr': cfg['wm']['lr']*=2
    else:
        path=Path(cfg['output'])/'wm/COMPLETE.json'; receipt=json.loads(path.read_text())
        if mutation=='data_identity': receipt['scientific']['hashes']['wm_il']='0'*64
        else: receipt['scientific']['source_sha256']='0'*64
        dump(path,receipt)
    with pytest.raises(ValueError): preflight(cfg,'pseudo')


@pytest.mark.parametrize('mutation',['valid','epoch','payload_scientific'])
def test_load_completed_validates_true_payload_completion_before_tiny_model_change(chain,mutation):
    from lwm_stream.entry import load_completed
    cfg,_,completed,_=chain
    path=Path(cfg['output'])/'wm/checkpoint-final.pt'
    payload=torch.load(path,weights_only=True)
    if mutation=='epoch': payload['progress']['training']['epoch']-=1
    elif mutation=='payload_scientific': payload['identity']['scientific']['settings']['lr']*=2
    torch.save(payload,path)
    predecessor={'path':str(path),'sha256':sha(path),'scientific':completed['wm']['scientific']}
    model=torch.nn.Module(); model.register_parameter('weight',torch.nn.Parameter(torch.zeros(1)))
    if mutation=='valid':
        load_completed(model,predecessor,cfg['wm']['epochs'])
        torch.testing.assert_close(model.weight,torch.ones(1))
    else:
        with pytest.raises(ValueError): load_completed(model,predecessor,cfg['wm']['epochs'])
        torch.testing.assert_close(model.weight,torch.zeros(1),rtol=0,atol=0)


class PseudoPairDataset(torch.utils.data.Dataset):
    def __len__(self): return 3
    def __getitem__(self,key):
        epoch,index=key
        assert epoch==0
        return {'now':torch.full((4,3),index+.25),'goal':torch.full((4,3),index-.5)}


def test_pseudo_real_original_labels_ready_and_reuse_without_second_model(chain,tmp_path,monkeypatch):
    import lwm_stream.entry as entry
    from test_training_seam import SmallWM
    from lwm.tokenizer import ActionTokenizer
    from lwm_stream.stage_data import pseudo_labels
    cfg,hashes,completed,_=chain; cfg=copy.deepcopy(cfg); cfg['output']=str(tmp_path/'pseudo-run')
    admission={'hashes':hashes,'predecessors':{'wm':{'path':'fixture-completed-wm','sha256':completed['wm']['sha256']}},'pseudo':None}
    dataset=PseudoPairDataset(); wm=SmallWM().eval()
    tokenizer=ActionTokenizer(Path(cfg['codebooks'])/'action_centers.json')
    candidates=torch.arange(2*63*3,dtype=torch.float32).reshape(2,63,3)/100
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg['seed'])
        expected=torch.stack([pseudo_labels(wm,tokenizer,dataset[(0,i)]['now'][None],dataset[(0,i)]['goal'][None],candidates)['tokens'][0] for i in range(len(dataset))])
    wm.encode_count=0; builder_calls=[]
    def builder(config,admitted,device):
        builder_calls.append(1); return wm,tokenizer,candidates
    monkeypatch.setattr(entry,'build_pseudo_model',builder,raising=False)
    monkeypatch.setattr(entry,'build_dataset',lambda config,stage,admitted:dataset,raising=False)
    entry.run_pseudo(cfg,admission,torch.device('cpu'))
    root=Path(cfg['output'])/'pseudo'; ready=json.loads((root/'READY.json').read_text())
    assert ready['status']=='READY' and ready['identity']==entry.pseudo_identity(cfg,admission)
    assert ready['sha256']==sha(root/'tokens.npy')
    np.testing.assert_array_equal(np.load(root/'tokens.npy',allow_pickle=False),expected.numpy())
    assert builder_calls==[1] and wm.encode_count==3
    def forbidden(*args,**kwargs): pytest.fail('complete pseudo cache must reuse without constructing WM again')
    monkeypatch.setattr(entry,'build_pseudo_model',forbidden)
    entry.run_pseudo(cfg,admission,torch.device('cpu'))
    assert json.loads((root/'READY.json').read_text())==ready


def test_pseudo_incomplete_cache_cannot_be_reused(chain,tmp_path,monkeypatch):
    import lwm_stream.entry as entry
    cfg,hashes,completed,_=chain; cfg=copy.deepcopy(cfg); cfg['output']=str(tmp_path/'incomplete')
    directory=Path(cfg['output'])/'pseudo'; directory.mkdir(parents=True)
    np.save(directory/'tokens.npy',np.full((3,65),-1,dtype=np.int16))
    admission={'hashes':hashes,'predecessors':{'wm':{'path':'fixture-wm','sha256':completed['wm']['sha256']}},'pseudo':None}
    def forbidden(*args,**kwargs): pytest.fail('incomplete cache rejection must precede WM allocation')
    monkeypatch.setattr(entry,'build_pseudo_model',forbidden,raising=False)
    with pytest.raises((ValueError,RuntimeError,FileExistsError)) as caught:
        entry.run_pseudo(cfg,admission,torch.device('cpu'))
    assert not isinstance(caught.value,NotImplementedError), 'stub is not a validated cache rejection'
    assert not (directory/'READY.json').exists()


def test_resume_never_publishes_preexisting_future_checkpoint_as_completed(chain,tmp_path,monkeypatch):
    import lwm_stream.entry as entry
    from test_engine import TinyModel
    cfg,hashes,_,_=chain; cfg=copy.deepcopy(cfg)
    cfg['wm'].update(epochs=2,global_batch=2,lr=.02,warmup_epochs=1)
    cfg.update(output=str(tmp_path/'collision'),workers=0,micro_batch=1,checkpoint_updates=1000,log_updates=1)
    admission={'hashes':hashes,'predecessors':{},'pseudo':None}; seen=[]
    def builder(config,stage,admitted,device):
        model=TinyModel(); return model,EntryTinyObjective(model,seen)
    monkeypatch.setattr(entry,'build_model',builder)
    monkeypatch.setattr(entry,'build_dataset',lambda config,stage,admitted:EntryTinyDataset())
    entry.run_training(cfg,'wm',admission,torch.device('cpu'),smoke_updates=1)
    directory=Path(cfg['output'])/'wm'; latest=json.loads((directory/'LATEST.json').read_text())
    future=directory/'update-000000006-epoch-0002.pt'; future.write_bytes(b'unrelated future checkpoint')
    with pytest.raises((ValueError,RuntimeError,FileExistsError)):
        entry.run_training(cfg,'wm',admission,torch.device('cpu'),resume=str(directory/latest['checkpoint']))
    assert not (directory/'COMPLETE.json').exists()
    assert json.loads((directory/'LATEST.json').read_text())['checkpoint']!=future.name
    assert future.read_bytes()==b'unrelated future checkpoint'


class RankRandomObjective(EntryTinyObjective):
    def terms(self,model,batch):
        import random
        factor=1.+.05*(torch.rand(())+random.random()+float(np.random.random()))
        numerator,count=super().terms(model,batch)
        return numerator*factor,count


def entry_ddp_resume_worker(rank,init_path,root_path):
    import random
    import shutil
    from datetime import timedelta
    import torch.distributed as dist
    import lwm_stream.entry as entry
    from test_engine import TinyModel,nested_equal
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_path,rank=rank,world_size=2,timeout=timedelta(seconds=30))
    try:
        root=Path(root_path); config=json.loads((root/'config.json').read_text())
        admission={'hashes':{name:'fixture-'+name for name in ('wm_il','rl','dev','action_centers','trajectory_centers','croco')},'predecessors':{},'pseudo':None}
        seen=[]; saves=[]; original_save=entry.save_checkpoint
        def builder(cfg,stage,admitted,device):
            model=TinyModel(); return model,RankRandomObjective(model,seen)
        def save(*args,**kwargs): saves.append(str(args[0])); return original_save(*args,**kwargs)
        entry.build_model=builder; entry.build_dataset=lambda *args:EntryTinyDataset(); entry.save_checkpoint=save
        full=copy.deepcopy(config); full['output']=str(root/'ddp-full')
        entry.run_training(full,'wm',admission,torch.device('cpu'),rank=rank,world_size=2)
        expected_rng=(torch.get_rng_state().clone(),random.getstate(),copy.deepcopy(np.random.get_state()))
        expected_seen=list(seen); seen.clear()
        resumed=copy.deepcopy(config); resumed['output']=str(root/'ddp-resumed')
        entry.run_training(resumed,'wm',admission,torch.device('cpu'),rank=rank,world_size=2,smoke_updates=1)
        stage=Path(resumed['output'])/'wm'
        if rank==0: shutil.copytree(stage,root/'wrong-world'/'wm')
        dist.barrier()
        latest=json.loads((stage/'LATEST.json').read_text())
        entry.run_training(resumed,'wm',admission,torch.device('cpu'),rank=rank,world_size=2,resume=str(stage/latest['checkpoint']))
        assert seen==expected_seen
        assert torch.equal(torch.get_rng_state(),expected_rng[0]) and random.getstate()==expected_rng[1]
        actual_np=np.random.get_state(); assert actual_np[0]==expected_rng[2][0] and actual_np[2:]==expected_rng[2][2:]
        np.testing.assert_array_equal(actual_np[1],expected_rng[2][1])
        assert len(saves)==(5 if rank==0 else 0)
        (root/f'rank-{rank}.json').write_text(json.dumps({'rank':rank,'save_count':len(saves),'seen':seen,'rng_exact':True}))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_real_two_rank_entry_checkpoint_rng_resume_and_changed_world_rejected(tmp_path,monkeypatch):
    import time
    import torch.multiprocessing as mp
    import lwm_stream.entry as entry
    from test_engine import TinyModel,nested_equal
    config=json.loads((HERE/'configs/lwm_stream_train.json').read_text())
    config['wm'].update(epochs=2,global_batch=3,lr=.02,warmup_epochs=1)
    config.update(workers=0,micro_batch=1,checkpoint_updates=1000,log_updates=1)
    (tmp_path/'config.json').write_text(json.dumps(config))
    processes=mp.start_processes(entry_ddp_resume_worker,args=(str(tmp_path/'gloo'),str(tmp_path)),nprocs=2,join=False,start_method='spawn')
    deadline=time.monotonic()+60
    try:
        while not processes.join(timeout=1):
            if time.monotonic()>deadline: pytest.fail('bounded two-rank entry test exceeded60s')
    finally:
        for proc in processes.processes:
            if proc.is_alive(): proc.terminate()
        for proc in processes.processes: proc.join(timeout=3)
    states=[]
    for name in ('ddp-full','ddp-resumed'):
        directory=tmp_path/name/'wm'; receipt=json.loads((directory/'COMPLETE.json').read_text())
        payload=torch.load(directory/receipt['checkpoint'],weights_only=True)
        assert payload['progress']['training']=={'epoch':2,'update':4,'sampler_offset':0}
        assert len(payload['progress']['rank_rng'])==2
        for state in payload['progress']['rank_rng']: assert set(state)=={'torch','python','numpy','cuda'}
        assert not torch.equal(payload['progress']['rank_rng'][0]['torch'],payload['progress']['rank_rng'][1]['torch'])
        states.append(payload)
    nested_equal(states[0]['model'],states[1]['model']); nested_equal(states[0]['optimizer'],states[1]['optimizer'])
    ranks=[json.loads((tmp_path/f'rank-{i}.json').read_text()) for i in range(2)]
    assert [r['save_count'] for r in ranks]==[5,0]
    assert not {tuple(v) for v in ranks[0]['seen']}&{tuple(v) for v in ranks[1]['seen']}
    assert len(ranks[0]['seen'])==8 and len(ranks[1]['seen'])==4
    wrong=copy.deepcopy(config); wrong['output']=str(tmp_path/'wrong-world')
    admission={'hashes':{name:'fixture-'+name for name in ('wm_il','rl','dev','action_centers','trajectory_centers','croco')},'predecessors':{},'pseudo':None}
    def builder(cfg,stage,admitted,device):
        model=TinyModel(); return model,RankRandomObjective(model,[])
    monkeypatch.setattr(entry,'build_model',builder); monkeypatch.setattr(entry,'build_dataset',lambda *args:EntryTinyDataset())
    stage=tmp_path/'wrong-world'/'wm'; latest=json.loads((stage/'LATEST.json').read_text()); before=(stage/'train.jsonl').read_bytes()
    with pytest.raises(ValueError,match='identity|world|rank'):
        entry.run_training(wrong,'wm',admission,torch.device('cpu'),rank=0,world_size=1,resume=str(stage/latest['checkpoint']))
    assert not (stage/'COMPLETE.json').exists() and (stage/'train.jsonl').read_bytes()==before
