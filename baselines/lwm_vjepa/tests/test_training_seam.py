"""Scientific seam tests: upstream golden calls, reduced resource fixtures."""
import sys
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'source'))
sys.path.insert(0, str(HERE / 'source/official'))
from lwm.world_model import LatentWorldModel
from lwm.policy import ARPlusPolicy
from lwm.trajectory_decoder import TrajectoryDecoder
from lwm.utils import SinusoidalPositionalEncoding
from lwm_stream.training import (WorldModelTraining, PolicyTraining, log_distance_targets,
                                 masked_mse_terms, cross_entropy_terms)

@pytest.fixture(autouse=True)
def isolated_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(729)
        yield
    torch.set_num_threads(threads)

class SmallCroCo(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(3, 12)
        self.enc_blocks = nn.Linear(12, 12)
        self.enc_blocks.requires_grad_(False)
        self.enc_norm = nn.LayerNorm(12)
        self.prediction_head = nn.Linear(12, 12)
    def _encode_image(self, image, do_mask=False):
        return self.enc_norm(self.enc_blocks(self.patch_embed(image))), None, None
    def _decoder(self, feat1, pos1, masks, feat2, pos2):
        return feat1 + 0.7 * feat2

class SmallWM(LatentWorldModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.croco = SmallCroCo()
        self.dim, self.num_a, self.Np = 12, 63, 4
        self.cro_proj = nn.Linear(12, 12)
        self.acton_encoder = nn.Linear(3, 12)
        self.position = SinusoidalPositionalEncoding(12, max_len=63)
        layer = nn.TransformerDecoderLayer(12, 6, 24, dropout=0.1, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(layer, 1)
        self.output_sim = nn.Linear(12, 1)
        self.encode_count = 0
    def encode_images(self, a, b):
        self.encode_count += 1
        return super().encode_images(a, b)

class SmallPolicy(ARPlusPolicy):
    def __init__(self):
        nn.Module.__init__(self)
        self.croco = SmallCroCo()
        self.cro_proj = nn.Linear(12, 12)
        self.waypoint_decoder = TrajectoryDecoder(66, 4, 65, dim=12, decoder_layer=1)
        self.max_len, self.num_a, self.PAD_token = 65, 63, 66
        self.encode_count = 0
    def encode_images(self, a, b):
        self.encode_count += 1
        return super().encode_images(a, b)

def inputs(b=2, m=3):
    return torch.randn(b, 4, 3), torch.randn(b, 4, 3), torch.randn(b, m, 63, 3)

def tokens():
    result = torch.full((2, 65), 66, dtype=torch.long)
    result[0, :5] = torch.tensor([64, 3, 9, 0, 65])
    result[1, :3] = torch.tensor([64, 11, 65])
    return result

def assert_owners(model, wrapper):
    assert {id(p) for p in model.parameters()} == {id(p) for p in wrapper.parameters()}
    assert model.croco.enc_blocks.weight.grad is None
    for module in [model.croco.patch_embed, model.croco.enc_norm, model.croco.prediction_head, model.cro_proj]:
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0

@pytest.mark.parametrize('b,m', [(1, 1), (2, 3)])
def test_wm_matches_official_per_sample_score_once_encoded(b, m):
    model = SmallWM().eval()
    now, goal, actions = inputs(b, m)
    expected = torch.cat([model.score(now[i:i+1], goal[i:i+1], actions[i] * .1) for i in range(b)])
    model.encode_count = 0
    result = WorldModelTraining(model)(now, goal, actions)
    assert result.shape == (b, m, 63)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
    assert model.encode_count == 1

def test_wm_gradient_crosses_frozen_blocks_and_preserves_owners():
    model = SmallWM().eval()
    wrapper = WorldModelTraining(model)
    wrapper(*inputs()).square().sum().backward()
    assert_owners(model, wrapper)
    assert model.acton_encoder.weight.grad.abs().sum() > 0
    assert model.decoder1.layers[0].self_attn.in_proj_weight.grad.abs().sum() > 0

def test_wm_future_cannot_change_past():
    model = WorldModelTraining(SmallWM().eval())
    a, b, x = inputs()
    changed = x.clone()
    changed[:, :, 20:] += 100
    torch.testing.assert_close(model(a,b,x)[:,:,:20], model(a,b,changed)[:,:,:20])

def test_wm_padding_nan_sanitized_and_rng_unchanged():
    model = WorldModelTraining(SmallWM().eval())
    a,b,x = inputs()
    mask = torch.arange(63).view(1,1,63).expand(2,3,63) < 7
    dirty = x.masked_fill(~mask[...,None], float('nan'))
    clean = x.masked_fill(~mask[...,None], 0)
    state = torch.get_rng_state().clone()
    got = model(a,b,dirty,valid_mask=mask)
    expected = model(a,b,clean,valid_mask=mask)
    assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(got,expected)
    assert torch.isfinite(got).all()

def test_wm_rejects_nonprefix_mask():
    a,b,x = inputs()
    mask = torch.ones(2,3,63,dtype=torch.bool)
    mask[0,0,1] = False
    with pytest.raises(ValueError):
        WorldModelTraining(SmallWM())(a,b,x,valid_mask=mask)

def test_log_target_xy_meter_units_and_sign():
    actions = torch.zeros(1,1,63,3)
    actions[...,0] = 3
    actions[...,1] = 4
    actions[...,2] = 1000
    actions.requires_grad_()
    mask = torch.ones(1,1,63,dtype=torch.bool)
    target = log_distance_targets(actions,torch.zeros(1,2),epsilon_m=.5,valid_mask=mask)
    assert not target.requires_grad
    torch.testing.assert_close(target,torch.full((1,1,63),torch.log(torch.tensor(5.5))))
    prediction = torch.linspace(-2,2,63).reshape(1,1,63)
    torch.testing.assert_close((prediction-target).square(),(-prediction-(-target)).square())
    assert prediction.argmin() == (-prediction).argmax()

def test_log_targets_ignore_invalid_nan_and_align_goal_by_batch():
    actions = torch.zeros(2,2,63,3)
    mask = torch.zeros(2,2,63,dtype=torch.bool)
    mask[:,:,:2] = True
    actions[~mask] = float('nan')
    goal = torch.tensor([[3.,4.],[0.,2.]])
    result = log_distance_targets(actions,goal,epsilon_m=.1,valid_mask=mask)
    torch.testing.assert_close(result[0,:,:2],torch.full((2,2),torch.log(torch.tensor(5.1))))
    torch.testing.assert_close(result[1,:,:2],torch.full((2,2),torch.log(torch.tensor(2.1))))
    assert torch.equal(result[~mask],torch.zeros_like(result[~mask]))

def test_masked_mse_nan_gradient_and_partition_additivity():
    pred = torch.tensor([1.,float('nan'),4.,8.],requires_grad=True)
    target = torch.tensor([3.,float('nan'),1.,2.],requires_grad=True)
    mask = torch.tensor([True,False,True,True])
    numerator,count = masked_mse_terms(pred,target,mask)
    assert numerator.item() == 49 and count.item() == 3
    assert count.dtype == torch.int64
    parts = [masked_mse_terms(pred[s],target[s],mask[s]) for s in [slice(0,2),slice(2,4)]]
    torch.testing.assert_close(numerator,parts[0][0]+parts[1][0])
    assert count == parts[0][1]+parts[1][1]
    numerator.backward()
    torch.testing.assert_close(pred.grad,torch.tensor([-4.,0.,6.,12.]))
    assert target.grad is None

def test_empty_mse_is_differentiable_zero():
    pred = torch.full((3,),float('nan'),requires_grad=True)
    total,count = masked_mse_terms(pred,pred.detach(),torch.zeros(3,dtype=torch.bool))
    assert total.item() == 0 and count.item() == 0
    total.backward()
    assert torch.equal(pred.grad,torch.zeros(3))

@pytest.mark.parametrize('training', [False,True])
def test_policy_exact_official_teacher_forcing_including_dropout(training):
    model = SmallPolicy().train(training)
    a,b,_ = inputs()
    tok = tokens()
    state = torch.get_rng_state().clone()
    expected = model.waypoint_decoder(model.cro_proj(model.encode_images(a,b)),tok)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(state)
    model.encode_count = 0
    got = PolicyTraining(model)(a,b,tok)
    torch.testing.assert_close(got,expected,rtol=0,atol=0)
    assert torch.equal(torch.get_rng_state(),after)
    assert model.training == training and model.waypoint_decoder.pos_drop.training == training
    assert model.encode_count == 1 and got.shape == (2,64,67)

def test_policy_gradient_owners_and_causality():
    model = SmallPolicy().eval()
    wrapper = PolicyTraining(model)
    a,b,_ = inputs()
    tok = tokens()
    changed = tok.clone(); changed[0,3] = 17
    out = wrapper(a,b,tok)
    torch.testing.assert_close(out[0,:3],wrapper(a,b,changed)[0,:3])
    total,_ = cross_entropy_terms(out,tok,pad_token=66)
    total.backward()
    assert_owners(model,wrapper)
    assert model.waypoint_decoder.embedding.weight.grad.abs().sum() > 0

def test_ce_shift_eos_padding_and_nan_gradient():
    tok = tokens()
    active = tok[:,1:] != 66
    logits = torch.randn(2,64,67).masked_fill(~active[...,None],float('nan')).requires_grad_()
    total,count = cross_entropy_terms(logits,tok,pad_token=66)
    expected = F.cross_entropy(logits[active],tok[:,1:][active],reduction='sum')
    torch.testing.assert_close(total,expected)
    assert count.item() == 6 and count.dtype == torch.int64
    total.backward()
    assert torch.equal(logits.grad[~active],torch.zeros_like(logits.grad[~active]))
    assert torch.isfinite(logits.grad).all()

@pytest.mark.parametrize('mutation', ['wrong_bos','after_eos','no_eos','pad_before_eos'])
def test_policy_rejects_broken_token_termination(mutation):
    tok = tokens()
    if mutation == 'wrong_bos': tok[0,0] = 1
    elif mutation == 'after_eos': tok[0,6] = 3
    elif mutation == 'no_eos': tok[0,4] = 3
    else: tok[0,2] = 66
    with pytest.raises(ValueError):
        PolicyTraining(SmallPolicy())(*inputs()[:2],tok)
