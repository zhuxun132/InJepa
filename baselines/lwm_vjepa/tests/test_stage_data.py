"""Three-stage data connections against official stochastic/tokenizer goldens."""
import json
import sys
from pathlib import Path
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from test_training_seam import SmallWM,SmallPolicy,inputs
from lwm.tokenizer import ActionTokenizer
from lwm_stream.stage_data import pseudo_labels,sample_rewards

@pytest.fixture(autouse=True)
def isolated_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(706)
        yield
    torch.set_num_threads(threads)

@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path/'centers.json'
    x = torch.arange(64,dtype=torch.float32)
    path.write_text(json.dumps(torch.stack([.1+x/80,torch.sin(x)/7],-1).tolist()))
    return ActionTokenizer(path)

def snapshot(model):
    return {key:value.clone() for key,value in model.state_dict().items()}

def unchanged(model,before):
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    assert all(p.grad is None for p in model.parameters())

@pytest.mark.parametrize('batch,candidates',[(1,1),(2,3)])
def test_pseudo_labels_official_argmin_inclusive_prefix_and_stochastic_tokens(tokenizer,batch,candidates):
    wm = SmallWM().eval()
    now,goal,actions = inputs(batch,candidates)
    shared = actions[0].clone().requires_grad_()
    indices,points = wm.eval_wm(now,goal,shared*.1)
    state = torch.get_rng_state().clone()
    expected = torch.stack([tokenizer.encode_tensor(shared[indices[i],:points[i]+1,:2]) for i in range(batch)])
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    wm.encode_count = 0
    before = snapshot(wm)
    got = pseudo_labels(wm,tokenizer,now,goal,shared)
    assert wm.encode_count == 1
    assert torch.equal(got['candidate_indices'],indices)
    assert torch.equal(got['point_indices'],points)
    assert torch.equal(got['tokens'],expected)
    assert got['tokens'].shape == (batch,65)
    assert torch.equal(torch.get_rng_state(),after)
    assert all(not value.requires_grad for value in got.values())
    unchanged(wm,before)

def test_sample_rewards_actual_official_rollout_decode_and_wm_golden(tokenizer):
    policy,wm = SmallPolicy().eval(),SmallWM().eval()
    now,goal,_ = inputs()
    state = torch.get_rng_state().clone()
    tok = policy.roll_out(now,goal,num_sample=2,temperature=.8)
    actions,lengths = tokenizer.batch_decode(tok.reshape(4,65))
    actions = actions.reshape(2,2,63,3)
    expected = wm.get_reward(now,goal,actions*.1,lengths)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    policy.encode_count = wm.encode_count = 0
    old_policy,old_wm = snapshot(policy),snapshot(wm)
    owners = [(id(p),p.requires_grad) for m in (policy,wm) for p in m.parameters()]
    got = sample_rewards(policy,wm,tokenizer,now,goal,num_sample=2,temperature=.8)
    assert policy.encode_count == 1 and wm.encode_count == 1
    assert torch.equal(got['tokens'],tok)
    assert torch.equal(got['lengths'],lengths.reshape(2,2))
    torch.testing.assert_close(got['actions_m'],actions)
    torch.testing.assert_close(got['rewards'],expected)
    assert torch.equal(torch.get_rng_state(),after)
    assert all(not value.requires_grad for value in got.values())
    unchanged(policy,old_policy); unchanged(wm,old_wm)
    assert owners == [(id(p),p.requires_grad) for m in (policy,wm) for p in m.parameters()]
    assert all(not module.training for m in (policy,wm) for module in m.modules())

def test_sample_reward_empty_short_truncated_no_eos_and_call_counts(tokenizer,monkeypatch):
    policy,wm = SmallPolicy().eval(),SmallWM().eval()
    now,goal,_ = inputs()
    tok = torch.full((2,2,65),66,dtype=torch.long)
    tok[:,:,0] = 64
    tok[0,0,1] = 65
    tok[0,1,1:4] = torch.tensor([3,4,65])
    tok[1,0,1:] = torch.arange(64)
    tok[1,1,1:3] = torch.tensor([9,65])
    actions,lengths = tokenizer.batch_decode(tok.reshape(4,65))
    assert lengths.tolist() == [0,2,63,1]
    expected = wm.get_reward(now,goal,actions.reshape(2,2,63,3)*.1,lengths)
    calls = {'rollout':0,'decode':0}
    def rollout(a,b,num_sample,temperature):
        assert a is now and b is goal and num_sample == 2 and temperature == 1.7
        calls['rollout'] += 1
        return tok.clone()
    original = tokenizer.batch_decode
    def decode(flat):
        calls['decode'] += 1
        assert flat.shape == (4,65)
        return original(flat)
    monkeypatch.setattr(policy,'roll_out',rollout)
    monkeypatch.setattr(tokenizer,'batch_decode',decode)
    wm.encode_count = 0
    got = sample_rewards(policy,wm,tokenizer,now,goal,num_sample=2,temperature=1.7)
    assert calls == {'rollout':1,'decode':1} and wm.encode_count == 1
    assert torch.equal(got['tokens'],tok)
    assert got['tokens'][1,0,-1] == 63 # do not remove final sampled token
    assert torch.equal(got['lengths'],lengths.reshape(2,2))
    torch.testing.assert_close(got['actions_m'],actions.reshape(2,2,63,3))
    torch.testing.assert_close(got['rewards'],expected)
    assert torch.equal(got['actions_m'][0,0],torch.zeros(63,3))

@pytest.mark.parametrize('function,owner',[('pseudo','wm'),('rewards','wm'),('rewards','policy')])
def test_stage_calls_require_eval_submodules(tokenizer,function,owner):
    wm,policy = SmallWM().eval(),SmallPolicy().eval()
    now,goal,actions = inputs()
    if owner == 'wm': wm.decoder1.layers[0].dropout.train()
    else: policy.waypoint_decoder.pos_drop.train()
    with pytest.raises(ValueError):
        if function == 'pseudo': pseudo_labels(wm,tokenizer,now,goal,actions[0])
        else: sample_rewards(policy,wm,tokenizer,now,goal,num_sample=2,temperature=1.)

@pytest.mark.parametrize('function',['pseudo','rewards'])
def test_stage_calls_reject_incompatible_tokenizer_horizon(tokenizer,function):
    wm,policy = SmallWM().eval(),SmallPolicy().eval()
    now,goal,actions = inputs()
    tokenizer.max_len = 64
    with pytest.raises(ValueError):
        if function == 'pseudo': pseudo_labels(wm,tokenizer,now,goal,actions[0])
        else: sample_rewards(policy,wm,tokenizer,now,goal,num_sample=2,temperature=1.)
