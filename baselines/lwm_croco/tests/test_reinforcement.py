"""Literal Eq16 and official policy likelihood scientific golden tests."""
import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE / 'source'))
from test_training_seam import SmallPolicy, inputs, assert_owners
from lwm_stream.reinforcement import PolicyLikelihood, group_advantages, grpo_terms

@pytest.fixture(autouse=True)
def isolated_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(416)
        yield
    torch.set_num_threads(threads)

def rollout_tokens():
    tok = torch.full((2,2,65),66,dtype=torch.long)
    tok[:,:,0] = 64
    tok[0,0,1:4] = torch.tensor([3,9,65])
    tok[0,1,1] = 65
    tok[1,0,1:5] = torch.tensor([5,8,1,65])
    tok[1,1,1:] = torch.arange(64) % 64
    return tok

def active_mask(tok):
    return tok[...,1:] != 66

def test_likelihood_matches_official_prefix_distribution_and_sequence_sum():
    model = SmallPolicy().eval()
    wrapper = PolicyLikelihood(model)
    now,goal,_ = inputs()
    tok = rollout_tokens()
    temperature = 2.3
    state = torch.get_rng_state().clone()
    result = wrapper(now,goal,tok,temperature=temperature)
    assert model.encode_count == 1
    assert torch.equal(state,torch.get_rng_state())
    mask = active_mask(tok)
    assert result['log_probs'].shape == (2,2,64,67)
    assert torch.equal(result['valid_mask'],mask)
    features = model.cro_proj(model.encode_images(now,goal))
    for b,g in [(0,0),(0,1),(1,0),(1,1)]:
        positions = torch.nonzero(mask[b,g]).flatten().tolist()
        if len(positions)>4: positions = [0,7,63]
        for t in positions:
            logits = model.waypoint_decoder.predict(features[b:b+1],tok[b,g,:t+1].unsqueeze(0),ret_all=True)
            logits[:,[64,66]] = -1e9
            golden = F.log_softmax(logits / temperature,dim=-1)[0]
            torch.testing.assert_close(result['log_probs'][b,g,t],golden,rtol=2e-5,atol=2e-5)
    selected = result['log_probs'].gather(-1,tok[...,1:,None]).squeeze(-1)
    selected = selected.masked_fill(~mask,0)
    torch.testing.assert_close(result['token_log_probs'],selected)
    torch.testing.assert_close(result['sequence_log_probs'],selected.sum(-1))
    assert torch.equal(result['log_probs'][~mask],torch.zeros_like(result['log_probs'][~mask]))
    assert torch.equal(result['token_log_probs'][~mask],torch.zeros_like(result['token_log_probs'][~mask]))

def test_likelihood_current_gradient_owners_and_future_causality():
    model = SmallPolicy().eval()
    wrapper = PolicyLikelihood(model)
    a,b,_ = inputs()
    tok = rollout_tokens()
    result = wrapper(a,b,tok,temperature=1.)
    changed = tok.clone(); changed[1,1,20:] = 37
    other = wrapper(a,b,changed,temperature=1.)
    torch.testing.assert_close(result['log_probs'][1,1,:20],other['log_probs'][1,1,:20])
    (-result['sequence_log_probs'].sum()).backward()
    assert_owners(model,wrapper)
    assert model.waypoint_decoder.embedding.weight.grad.abs().sum() > 0

@pytest.mark.parametrize('mode',['whole_model','nested_dropout'])
def test_likelihood_requires_every_module_eval(mode):
    model = SmallPolicy().eval()
    if mode == 'whole_model': model.train()
    else: model.waypoint_decoder.pos_drop.train()
    with pytest.raises(ValueError):
        PolicyLikelihood(model)(*inputs()[:2],rollout_tokens(),temperature=1.)

@pytest.mark.parametrize('mutation',['after_eos','pad_without_eos'])
def test_likelihood_rejects_invalid_rollout_termination(mutation):
    tok = rollout_tokens()
    if mutation == 'after_eos': tok[0,0,6] = 3
    else: tok[1,1,-1] = 66
    with pytest.raises(ValueError):
        PolicyLikelihood(SmallPolicy().eval())(*inputs()[:2],tok,temperature=1.)

def test_group_advantages_population_zero_variance_dtype_and_detach():
    rewards = torch.tensor([[1.,2.,3.],[8.,8.,8.],[-1e38,0,1e38]],requires_grad=True)
    result = group_advantages(rewards)
    golden = torch.tensor([[-(1.5**.5),0,1.5**.5],[0,0,0],[-(1.5**.5),0,1.5**.5]])
    torch.testing.assert_close(result,golden)
    assert result.dtype == torch.float32 and not result.requires_grad
    result64 = group_advantages(rewards.detach().double())
    assert result64.dtype == torch.float64
    torch.testing.assert_close(result64,golden.double(),rtol=1e-6,atol=1e-7)

def one_step_distributions(probability):
    p = torch.full((1,2,1,67),(1-probability)/66,dtype=torch.float64)
    p[...,65] = probability
    return p.log()

def test_literal_negative_advantage_caps_ratio_before_advantage_not_ppo():
    current = one_step_distributions(.015).requires_grad_()
    reference = one_step_distributions(.01).requires_grad_()
    tok = torch.tensor([[[64,65],[64,65]]])
    adv = torch.tensor([[-1.,0.]],dtype=torch.float64,requires_grad=True)
    total,count = grpo_terms(current,reference,tok,adv,beta=0.,clip_epsilon=.2)
    torch.testing.assert_close(total,torch.tensor(1.2,dtype=torch.float64))
    assert total.item() != pytest.approx(1.5) # conventional PPO is a different loss
    assert count.item() == 2 and count.dtype == torch.int64
    total.backward()
    assert torch.equal(current.grad,torch.zeros_like(current))
    assert reference.grad is None and adv.grad is None

def test_signed_uncapped_ratios_hand_gradient():
    current = one_step_distributions(.005).requires_grad_()
    ref = one_step_distributions(.01)
    tok = torch.tensor([[[64,65],[64,65]]])
    adv = torch.tensor([[2.,-1.]],dtype=torch.float64)
    total,_ = grpo_terms(current,ref,tok,adv,beta=0.,clip_epsilon=.2)
    torch.testing.assert_close(total,torch.tensor(-.5,dtype=torch.float64))
    total.backward()
    expected = torch.zeros_like(current)
    expected[0,0,0,65] = -1
    expected[0,1,0,65] = .5
    torch.testing.assert_close(current.grad,expected)

def test_grpo_sequence_lengths_kl_direction_nan_padding_and_partition():
    tok = torch.tensor([[[64,3,65,66],[64,7,8,65]],[[64,65,66,66],[64,1,65,66]]])
    mask = active_mask(tok)
    p = F.log_softmax(torch.randn(2,2,3,67,dtype=torch.float64),-1)
    q = F.log_softmax(torch.randn(2,2,3,67,dtype=torch.float64),-1)
    current = p.masked_fill(~mask[...,None],float('nan')).requires_grad_()
    reference = q.masked_fill(~mask[...,None],float('nan')).requires_grad_()
    adv = torch.tensor([[.2,-.8],[1.,-.4]],dtype=torch.float64,requires_grad=True)
    total,count = grpo_terms(current,reference,tok,adv,beta=.3,clip_epsilon=.2)
    lp = p.gather(-1,tok[...,1:,None]).squeeze(-1).masked_fill(~mask,0).sum(-1)
    lq = q.gather(-1,tok[...,1:,None]).squeeze(-1).masked_fill(~mask,0).sum(-1)
    ratio = torch.minimum((lp-lq).exp(),torch.tensor(1.2,dtype=torch.float64))
    kl = (p.exp()*(p-q)).sum(-1).masked_fill(~mask,0).sum(-1)
    assert (kl>=0).all()
    golden = (-ratio*adv.detach()+.3*kl).sum()
    torch.testing.assert_close(total,golden)
    assert count.item() == 4
    first,nfirst = grpo_terms(current[:1],reference[:1],tok[:1],adv[:1],beta=.3,clip_epsilon=.2)
    second,nsecond = grpo_terms(current[1:],reference[1:],tok[1:],adv[1:],beta=.3,clip_epsilon=.2)
    torch.testing.assert_close(first+second,total)
    assert nfirst+nsecond == count
    total.backward()
    assert reference.grad is None and adv.grad is None
    assert torch.isfinite(current.grad).all()
    assert torch.equal(current.grad[~mask],torch.zeros_like(current.grad[~mask]))
    assert current.grad[mask].abs().sum()>0

def test_extreme_log_ratio_capped_before_exp():
    logits = torch.zeros(1,2,1,67,dtype=torch.float64)
    logits[...,65] = -10000
    reference = F.log_softmax(logits,-1)
    current = one_step_distributions(.5).requires_grad_()
    tok = torch.tensor([[[64,65],[64,65]]])
    total,_ = grpo_terms(current,reference,tok,torch.ones(1,2),beta=0.,clip_epsilon=.2)
    torch.testing.assert_close(total,torch.tensor(-2.4,dtype=torch.float64))
    total.backward()
    assert torch.isfinite(current.grad).all()

def test_identical_reference_zero_advantage_is_zero_loss_and_gradient():
    logits = torch.randn(1,2,1,67,dtype=torch.float64,requires_grad=True)
    current = F.log_softmax(logits,-1)
    tok = torch.tensor([[[64,65],[64,65]]])
    total,_ = grpo_terms(current,current.detach(),tok,torch.zeros(1,2),beta=.7,clip_epsilon=.2)
    torch.testing.assert_close(total,torch.zeros((),dtype=torch.float64),rtol=0,atol=0)
    total.backward()
    torch.testing.assert_close(logits.grad,torch.zeros_like(logits),rtol=0,atol=1e-15)

def test_grpo_rejects_unnormalized_active_log_distribution():
    current = one_step_distributions(.01)+1
    tok = torch.tensor([[[64,65],[64,65]]])
    with pytest.raises(ValueError):
        grpo_terms(current,one_step_distributions(.01),tok,torch.ones(1,2),beta=0.,clip_epsilon=.2)
