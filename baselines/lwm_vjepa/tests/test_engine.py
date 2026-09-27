"""Optimizer normalization and local restart, including real CPU DDP."""
import copy
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from lwm_stream.engine import make_adam,set_warmup_lr,optimizer_update,save_checkpoint,load_checkpoint

@pytest.fixture(autouse=True)
def isolated_rng():
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    py,npstate = random.getstate(),np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(491); random.seed(491); np.random.seed(491)
        yield
    random.setstate(py); np.random.set_state(npstate); torch.set_num_threads(threads)

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[.3]]))
        self.unused = nn.Parameter(torch.tensor([.9]))
        self.frozen = nn.Parameter(torch.tensor([2.]),requires_grad=False)
    def forward(self,x):
        return x @ self.weight

def adam(model):
    return make_adam(model,lr=.02,betas=(.8,.95),eps=.05,weight_decay=0.)

def data():
    return (torch.arange(1.,6.).reshape(5,1),torch.tensor([[2.],[1.],[0.],[-1.],[3.]]),torch.tensor([True,False,False,True,True]))

def subset(batch,index):
    return tuple(value[index] for value in batch)

def terms(model,batch):
    x,y,mask = batch
    residual = model(x).squeeze(-1)[mask]-y.squeeze(-1)[mask]
    return residual.square().sum(),mask.sum(dtype=torch.int64)

def cloned_state(model):
    return {key:value.clone() for key,value in model.state_dict().items()}

def same_state(model,state):
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,state[key],rtol=0,atol=0)

def nested_equal(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: nested_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): nested_equal(x,y)
    else: assert a==b

def test_adam_explicit_hyperparameters_and_exact_trainable_ownership():
    model=TinyModel(); before=[(id(p),p.requires_grad) for p in model.parameters()]
    opt=adam(model)
    assert isinstance(opt,torch.optim.Adam)
    assert {id(p) for g in opt.param_groups for p in g['params']} == {id(model.weight),id(model.unused)}
    assert before==[(id(p),p.requires_grad) for p in model.parameters()]
    assert opt.param_groups[0]['betas']==(.8,.95)
    assert opt.param_groups[0]['eps']==.05 and opt.param_groups[0]['weight_decay']==0

@pytest.mark.parametrize('warmup,index,expected',[(4,0,.025),(4,3,.1),(4,9,.1),(0,0,.1)])
def test_warmup_first_step_endpoint_plateau(warmup,index,expected):
    opt=torch.optim.Adam(TinyModel().parameters(),lr=.7)
    got=set_warmup_lr(opt,peak_lr=.1,warmup_updates=warmup,update_index=index)
    assert got==pytest.approx(expected)
    assert all(g['lr']==pytest.approx(expected) for g in opt.param_groups)

def test_micro_sums_equal_full_batch_gradients_and_adam():
    full,micro=TinyModel(),TinyModel()
    optfull=torch.optim.Adam([full.weight,full.unused],lr=.02,betas=(.8,.95),eps=.05)
    optmicro=torch.optim.Adam([micro.weight,micro.unused],lr=.02,betas=(.8,.95),eps=.05)
    batch=data()
    num,count=terms(full,batch); (num/count).backward(); optfull.step()
    calls=[]
    def recording(model,b):
        calls.append(len(b[0])); return terms(model,b)
    result=optimizer_update(micro,optmicro,[subset(batch,slice(0,2)),subset(batch,slice(2,5))],recording)
    assert calls==[2,3]
    torch.testing.assert_close(micro.weight,full.weight)
    torch.testing.assert_close(micro.weight.grad,full.weight.grad)
    assert micro.unused.grad is None and micro.unused not in optmicro.state
    assert result['global_count']==3
    assert result['loss']==pytest.approx((num/count).item())
    assert result['gradient_norm']==pytest.approx(full.weight.grad.norm().item())

class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): return x.sum()*0+1
    @staticmethod
    def backward(ctx,grad): return torch.full((1,1),float('nan'))

@pytest.mark.parametrize('kind',['empty','bad_loss','bad_gradient'])
def test_invalid_update_never_steps(kind):
    model=TinyModel(); before=cloned_state(model)
    opt=torch.optim.Adam([model.weight,model.unused],lr=.02)
    def invalid(m,b):
        if kind=='empty': return m.weight.flatten()[:0].sum(),torch.tensor(0,dtype=torch.int64)
        if kind=='bad_loss': return m.weight.sum()*float('nan'),torch.tensor(1,dtype=torch.int64)
        return BadGradient.apply(m.weight),torch.tensor(1,dtype=torch.int64)
    with pytest.raises((ValueError,FloatingPointError)):
        optimizer_update(model,opt,[None],invalid)
    same_state(model,before); assert not opt.state

def test_update_rejects_missing_optimizer_owner():
    model=TinyModel(); before=cloned_state(model)
    opt=torch.optim.Adam([model.weight],lr=.02)
    with pytest.raises(ValueError): optimizer_update(model,opt,[data()],terms)
    same_state(model,before)

def ddp_worker(rank,init_path,output_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_path,rank=rank,world_size=2,timeout=timedelta(seconds=20))
    try:
        raw=TinyModel(); model=DDP(raw,find_unused_parameters=True)
        opt=torch.optim.Adam([raw.weight,raw.unused],lr=.02,betas=(.8,.95),eps=.05)
        batch=data()
        batches=([subset(batch,slice(0,1)),subset(batch,slice(1,3))] if rank==0 else [subset(batch,slice(3,5))])
        result=optimizer_update(model,opt,batches,terms)
        torch.save({'weight':raw.weight.detach(),'grad':raw.weight.grad,'unused_grad':raw.unused.grad,'result':result},Path(output_dir)/f'rank{rank}.pt')
    finally: dist.destroy_process_group()

def test_two_rank_gloo_unequal_micro_counts_matches_single_process(tmp_path):
    # Real ranks; last rank0 micro contributes zero, so its earlier no_sync gradient must survive.
    context=mp.start_processes(ddp_worker,args=(str(tmp_path/'gloo'),str(tmp_path)),nprocs=2,join=False,start_method='spawn')
    deadline=time.monotonic()+35
    try:
        while not context.join(timeout=1):
            if time.monotonic()>deadline: pytest.fail('bounded Gloo update exceeded35s')
    finally:
        for proc in context.processes:
            if proc.is_alive(): proc.terminate()
        for proc in context.processes: proc.join(timeout=2)
    golden=TinyModel(); opt=torch.optim.Adam([golden.weight,golden.unused],lr=.02,betas=(.8,.95),eps=.05)
    num,count=terms(golden,data()); (num/count).backward(); opt.step()
    for rank in range(2):
        got=torch.load(tmp_path/f'rank{rank}.pt',weights_only=True)
        torch.testing.assert_close(got['weight'],golden.weight)
        torch.testing.assert_close(got['grad'],golden.weight.grad)
        assert got['unused_grad'] is None
        assert got['result']['global_count']==3
        assert got['result']['loss']==pytest.approx((num/count).item())
        assert got['result']['gradient_norm']==pytest.approx(golden.weight.grad.norm().item())

def random_update(model,opt):
    scalar=random.random()+float(np.random.random())
    x=torch.randn(3,1)+scalar; y=torch.randn(3,1)
    opt.zero_grad(set_to_none=True); (model(x)-y).square().mean().backward(); opt.step()

def test_checkpoint_exact_adam_and_three_rng_continuation(tmp_path):
    model=TinyModel(); opt=torch.optim.Adam([model.weight,model.unused],lr=.01)
    random_update(model,opt)
    identity={'stage':'WM','data':'fixture-sha','source':'fixture-source'}
    progress={'epoch':2,'update':17,'sampler_offset':3}
    path=tmp_path/'step.pt'; cuda_before=torch.cuda.is_initialized()
    save_checkpoint(path,model,opt,identity=identity,progress=progress)
    assert torch.cuda.is_initialized()==cuda_before
    payload=torch.load(path,weights_only=True)
    assert payload['model'].keys()==model.state_dict().keys()
    random_update(model,opt)
    expected_model=cloned_state(model); expected_opt=copy.deepcopy(opt.state_dict())
    expected_draw=(random.random(),float(np.random.random()),torch.rand(3))
    resumed=TinyModel(); resumed_opt=torch.optim.Adam([resumed.weight,resumed.unused],lr=.3)
    got=load_checkpoint(path,resumed,resumed_opt,identity=identity)
    assert got==progress
    random_update(resumed,resumed_opt)
    same_state(resumed,expected_model); nested_equal(resumed_opt.state_dict(),expected_opt)
    draw=(random.random(),float(np.random.random()),torch.rand(3))
    assert draw[:2]==expected_draw[:2]; torch.testing.assert_close(draw[2],expected_draw[2],rtol=0,atol=0)
    assert resumed.unused not in resumed_opt.state

@pytest.mark.parametrize('mutation',['identity','shape'])
def test_checkpoint_rejection_before_model_mutation(tmp_path,mutation):
    model=TinyModel(); opt=torch.optim.Adam([model.weight,model.unused],lr=.01)
    path=tmp_path/'bad.pt'
    # Construct format with actual writer; separate direct load RED also below.
    save_checkpoint(path,model,opt,identity={'stage':'WM'},progress={'update':1})
    target=TinyModel(); target.weight.data.fill_(8); before=cloned_state(target)
    target_opt=torch.optim.Adam([target.weight,target.unused],lr=.07)
    identity={'stage':'RL'} if mutation=='identity' else {'stage':'WM'}
    if mutation=='shape':
        payload=torch.load(path,weights_only=True); payload['model']['weight']=torch.zeros(2,2); torch.save(payload,path)
    with pytest.raises(ValueError): load_checkpoint(path,target,target_opt,identity=identity)
    same_state(target,before); assert not target_opt.state

def test_load_rejects_unknown_format(tmp_path):
    path=tmp_path/'wrong.pt'; torch.save({'format':'invalid'},path)
    model=TinyModel(); opt=torch.optim.Adam([model.weight,model.unused],lr=.01)
    with pytest.raises(ValueError): load_checkpoint(path,model,opt,identity={})

def test_checkpoint_create_once_preserves_existing_bytes(tmp_path):
    path=tmp_path/'existing.pt'; path.write_bytes(b'existing artifact')
    model=TinyModel(); opt=torch.optim.Adam([model.weight,model.unused],lr=.01)
    with pytest.raises(FileExistsError): save_checkpoint(path,model,opt,identity={},progress={})
    assert path.read_bytes()==b'existing artifact'
