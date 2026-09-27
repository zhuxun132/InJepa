"""RGB-only best-trajectory bridge and native normalized velocity controls."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source')); sys.path.insert(0,str(HERE/'source/official'))
from lwm.preprocess import get_image_transform
import lwm_stream.navigation as navigation
from lwm_stream.navigation import LWMNavigationAdapter,VelocityHandler

class UnreadableHistory:
    def __iter__(self): raise AssertionError('history must not enter RGB-only LWM')
    def __getitem__(self,key): raise AssertionError('history must not enter RGB-only LWM')
    def __bool__(self): raise AssertionError('history must not enter RGB-only LWM')

@pytest.fixture(autouse=True)
def cpu_threads():
    old=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

@pytest.mark.parametrize('empty_winner',[False,True])
def test_adapter_uses_only_rgb_best_first_meter_waypoint_and_replans(monkeypatch,empty_winner):
    policy,wm=torch.nn.Linear(1,1),torch.nn.Linear(1,1); tokenizer=object(); calls=[]
    current=np.full((12,16,3),42,dtype=np.uint8); goal=np.full((12,16,3),91,dtype=np.uint8)
    def sampled(p,w,t,now,goal_tensor,*,num_sample,temperature):
        assert p is policy and w is wm and t is tokenizer
        assert num_sample==3 and temperature==.8
        calls.append((now.clone(),goal_tensor.clone()))
        actions=torch.zeros(1,3,63,3); actions[0,0,0]=torch.tensor([9.,9.,0.]); actions[0,2,0]=torch.tensor([2.,-1.,.5]); actions[0,2,1]=torch.tensor([99.,99.,0.])
        return {'actions_m':actions,'lengths':torch.tensor([[2,0,1]]),'rewards':torch.tensor([[.2,.9 if empty_winner else .1,.7]]),'tokens':torch.zeros(1,3,65,dtype=torch.long)}
    monkeypatch.setattr(navigation,'sample_rewards',sampled,raising=False)
    adapter=LWMNavigationAdapter(policy,wm,tokenizer,device=torch.device('cpu'),num_sample=3,temperature=.8)
    adapter.reset(goal)
    result=adapter.act(current,goal,UnreadableHistory())
    assert set(result)=={'continuous_action'}
    assert result['continuous_action']['stop'] is empty_winner
    if not empty_winner: np.testing.assert_array_equal(result['continuous_action']['waypoint_m'],[2.,-1.])
    assert not policy.training and not wm.training
    torch.testing.assert_close(calls[0][0],get_image_transform()(Image.fromarray(current))[None],rtol=0,atol=0)
    torch.testing.assert_close(calls[0][1],get_image_transform()(Image.fromarray(goal))[None],rtol=0,atol=0)
    adapter.act(current+1,goal,UnreadableHistory())
    assert len(calls)==2 and not torch.equal(calls[0][0],calls[1][0])


def handler():
    return VelocityHandler(linear_range=[0.,.5],angular_range_deg=[-60.,60.],time_step=.25,min_abs_linear=.01,min_abs_angular_deg=1.)

@pytest.mark.parametrize('point,linear,angular',[([.05,0.],-.2,0.),([0.,.1],-1.,1.),([0.,-.1],-1.,-1.),([0.,.0001],-1.,1.)])
def test_velocity_straight_and_pure_turn_normalized_native_units(point,linear,angular):
    result=handler()({'continuous_action':{'waypoint_m':point,'stop':False}})
    assert result['name']=='CONTINUOUS' and result['is_stop'] is False
    payload=result['habitat_payload']; assert payload['action']=='velocity_control'
    assert payload['action_args']['linear_velocity']==pytest.approx(linear,abs=1e-12)
    assert payload['action_args']['angular_velocity']==pytest.approx(angular,abs=1e-12)

@pytest.mark.parametrize('explicit',[False,True])
def test_velocity_low_both_speeds_or_explicit_stop_matches_native_stop(explicit):
    result=handler()({'continuous_action':{'waypoint_m':[1.,1.] if explicit else [.0001,0.],'stop':explicit}})
    assert result=={'name':'STOP','is_stop':True,'habitat_payload':{'action':'stop'}}

@pytest.mark.parametrize('point',[[float('nan'),0.],[0.,float('inf')]])
def test_velocity_nonfinite_waypoint_rejected(point):
    with pytest.raises(ValueError): handler()({'continuous_action':{'waypoint_m':point,'stop':False}})
