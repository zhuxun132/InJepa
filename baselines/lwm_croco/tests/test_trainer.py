"""Three real stage updates and exact loop/cursor checkpoint continuation."""
import copy
import json
import random
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from test_training_seam import SmallWM,SmallPolicy,inputs,tokens
from test_engine import TinyModel,terms as toy_terms,nested_equal
from lwm.tokenizer import ActionTokenizer
from lwm_stream.engine import make_adam,optimizer_update,save_checkpoint,load_checkpoint
from lwm_stream.trainer import StageObjective,train_epochs

@pytest.fixture(autouse=True)
def rng():
    threads=torch.get_num_threads(); torch.set_num_threads(1)
    py,npstate=random.getstate(),np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(710); random.seed(710); np.random.seed(710)
        yield
    random.setstate(py); np.random.set_state(npstate); torch.set_num_threads(threads)

@pytest.fixture
def tokenizer(tmp_path):
    path=tmp_path/'centers.json'; x=torch.arange(64,dtype=torch.float32)
    path.write_text(json.dumps(torch.stack([.2+x/70,torch.sin(x)/8],-1).tolist()))
    return ActionTokenizer(path)

def optimizer(model):
    return make_adam(model,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0.)

def state(model): return {k:v.clone() for k,v in model.state_dict().items()}

def assert_state(model,before):
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)

@pytest.mark.parametrize('stage',['wm','il','rl'])
def test_real_stage_adam_update_owners_modes_and_auxiliary_detach(stage,tokenizer):
    model=SmallWM() if stage=='wm' else SmallPolicy()
    now,goal,actions=inputs()
    aux=[]
    if stage=='wm':
        obj=StageObjective(stage,model,epsilon_m=.1)
        batch={'now':now,'goal':goal,'actions_m':actions,'goal_xy_m':torch.ones(2,2),'valid_mask':torch.ones(2,3,63,dtype=torch.bool)}
    elif stage=='il':
        obj=StageObjective(stage,model)
        batch={'now':now,'goal':goal,'tokens':tokens()}
    else:
        wm=SmallWM(); reference=copy.deepcopy(model); aux=[wm,reference]
        obj=StageObjective(stage,model,wm=wm,reference=reference,tokenizer=tokenizer,num_sample=2,temperature=1.,beta=.01,clip_epsilon=.2)
        batch={'now':now,'goal':goal}
    aux_before=[state(m) for m in aux]
    before=state(model)
    assert {id(p) for p in obj.training_model.parameters()}=={id(p) for p in model.parameters()}
    assert model.training == (stage!='rl')
    assert all(not p.requires_grad for p in model.croco.enc_blocks.parameters())
    metrics=optimizer_update(obj.training_model,optimizer(obj.training_model),[batch],obj.terms)
    assert metrics['global_count']>0 and np.isfinite(metrics['loss'])
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert model.croco.patch_embed.weight.grad is not None
    assert model.croco.enc_blocks.weight.grad is None
    for m,saved in zip(aux,aux_before):
        assert_state(m,saved)
        assert all(p.grad is None for p in m.parameters())
        assert all(not mod.training for mod in m.modules())
    model.train(stage=='rl'); obj.set_mode()
    assert model.training == (stage!='rl')

@pytest.mark.parametrize('alias',['trained_reference','wm_reference'])
def test_rl_rejects_parameter_aliases(tokenizer,alias):
    model=SmallPolicy(); wm=SmallWM(); ref=copy.deepcopy(model)
    if alias=='trained_reference': ref=model
    else: ref.croco=wm.croco
    with pytest.raises(ValueError):
        StageObjective('rl',model,wm=wm,reference=ref,tokenizer=tokenizer,num_sample=2,temperature=1.,beta=.01,clip_epsilon=.2)

def test_terms_rejects_different_training_wrapper():
    model=SmallPolicy(); obj=StageObjective('il',model)
    foreign=StageObjective('il',SmallPolicy())
    with pytest.raises(ValueError): obj.terms(foreign.training_model,{'now':inputs()[0],'goal':inputs()[1],'tokens':tokens()})

class LoopObjective:
    def __init__(self): self.training_model=TinyModel(); self.mode_calls=0
    def set_mode(self): self.mode_calls+=1; self.training_model.train()
    def terms(self,model,batch): return toy_terms(model,(batch['x'],batch['y'],batch['mask']))

def toy_batch():
    return {'x':torch.tensor([[1.],[2.]]),'y':torch.tensor([[2.],[0.]]),'mask':torch.tensor([True,True])}

def test_epoch_progress_warmup_and_callback_copy_order():
    obj=LoopObjective(); opt=optimizer(obj.training_model); calls=[]; events=[]
    def factory(epoch,offset):
        calls.append((epoch,offset))
        return 2, ([toy_batch()]*(i+1) for i in range(offset,2))
    def update(p,m):
        events.append(('u',dict(p),m['lr'])); p['epoch']=99
    def epoch(p): events.append(('e',dict(p))); p['update']=99
    got=train_epochs(obj,opt,factory,epochs=2,peak_lr=.04,warmup_updates=2,on_update=update,on_epoch=epoch)
    assert got=={'epoch':2,'update':4,'sampler_offset':0}
    assert calls==[(0,0),(1,0)]
    assert [item[0] for item in events]==['u','u','e','u','u','e']
    assert [item[2] for item in events if item[0]=='u']==pytest.approx([.02,.04,.04,.04])
    assert [item[1]['sampler_offset'] for item in events if item[0]=='u']==[1,2,1,2]
    assert obj.mode_calls>=2

@pytest.mark.parametrize('resume',[False,True])
def test_empty_factory_new_epoch_rejected_completed_resume_allowed(resume):
    obj=LoopObjective(); opt=optimizer(obj.training_model)
    progress={'epoch':0,'update':2,'sampler_offset':2} if resume else None
    if resume:
        got=train_epochs(obj,opt,lambda e,o:(2 if resume else 0,iter([])),epochs=1,peak_lr=.01,warmup_updates=0,progress=progress)
        assert got=={'epoch':1,'update':2,'sampler_offset':0}
    else:
        with pytest.raises(ValueError): train_epochs(obj,opt,lambda e,o:(2 if resume else 0,iter([])),epochs=1,peak_lr=.01,warmup_updates=0)

def test_failed_update_does_not_report_epoch_or_advance_input_progress():
    obj=LoopObjective(); opt=optimizer(obj.training_model)
    progress={'epoch':0,'update':0,'sampler_offset':0}; events=[]
    batch=toy_batch(); batch['y'].fill_(float('nan'))
    with pytest.raises((ValueError,FloatingPointError)):
        train_epochs(obj,opt,lambda e,o:(1,iter([[batch]])),epochs=1,peak_lr=.01,warmup_updates=0,progress=progress,on_update=lambda *a:events.append('u'),on_epoch=lambda *a:events.append('e'))
    assert progress=={'epoch':0,'update':0,'sampler_offset':0} and events==[]

class PauseRun(Exception): pass

def test_real_il_mid_epoch_checkpoint_cursor_rng_matches_continuous(tmp_path):
    base=SmallPolicy(); initial=state(base)
    full=SmallPolicy(); full.load_state_dict(initial)
    interrupted=SmallPolicy(); interrupted.load_state_dict(initial)
    def factory(epoch,offset):
        def remaining():
            for i in range(offset,3):
                value=random.random()+float(np.random.random())
                yield [{'now':torch.randn(2,4,3)+value,'goal':torch.randn(2,4,3),'tokens':tokens()}]
        return 3,remaining()
    seed_states=(torch.get_rng_state().clone(),random.getstate(),np.random.get_state())
    obj=StageObjective('il',full); opt=optimizer(obj.training_model)
    finish=train_epochs(obj,opt,factory,epochs=2,peak_lr=.001,warmup_updates=2)
    expected_model=state(full); expected_opt=copy.deepcopy(opt.state_dict())
    expected_random=(torch.rand(3),random.random(),float(np.random.random()))
    torch.set_rng_state(seed_states[0]); random.setstate(seed_states[1]); np.random.set_state(seed_states[2])
    obj2=StageObjective('il',interrupted); opt2=optimizer(obj2.training_model)
    checkpoint=tmp_path/'paused.pt'; identity={'stage':'il','source':'fixture','dataset':'fixture'}
    def pause(progress,metrics):
        if progress['update']==2:
            save_checkpoint(checkpoint,interrupted,opt2,identity=identity,progress=progress)
            raise PauseRun()
    with pytest.raises(PauseRun): train_epochs(obj2,opt2,factory,epochs=2,peak_lr=.001,warmup_updates=2,on_update=pause)
    restored=SmallPolicy(); obj3=StageObjective('il',restored); opt3=optimizer(obj3.training_model)
    cursor=load_checkpoint(checkpoint,restored,opt3,identity=identity)
    assert cursor=={'epoch':0,'update':2,'sampler_offset':2}
    actual=train_epochs(obj3,opt3,factory,epochs=2,peak_lr=.001,warmup_updates=2,progress=cursor)
    assert actual==finish=={'epoch':2,'update':6,'sampler_offset':0}
    assert_state(restored,expected_model); nested_equal(opt3.state_dict(),expected_opt)
    torch.testing.assert_close(torch.rand(3),expected_random[0],rtol=0,atol=0)
    assert (random.random(),float(np.random.random()))==expected_random[1:]


@pytest.mark.parametrize('declared,actual,committed',[(3,2,2),(1,2,1)])
def test_factory_cardinality_mismatch_cannot_fake_completion_or_extra_step(declared,actual,committed):
    obj=LoopObjective(); opt=optimizer(obj.training_model); updates=[]; epochs=[]
    def factory(e,o): return declared, iter([[toy_batch()] for _ in range(actual)])
    with pytest.raises(ValueError):
        train_epochs(obj,opt,factory,epochs=1,peak_lr=.01,warmup_updates=0,on_update=lambda p,m:updates.append(dict(p)),on_epoch=lambda p:epochs.append(dict(p)))
    assert len(updates)==committed and epochs==[]
    assert opt.state[obj.training_model.weight]['step'].item()==committed
