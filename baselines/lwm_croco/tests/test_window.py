"""Keyframe-window to original-frame geometry and short supervision masks."""
import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(HERE/'source'))
from lwm_stream.dataset import window_geometry


def base():
    positions=np.zeros((8,3)); q=np.tile([0.,0.,0.,1.],(8,1))
    positions[2]=[1,0,2]; positions[5]=[-2,0,3]; positions[7]=[-3,0,1]
    for i,yaw in [(2,np.pi/2),(5,np.pi/2+.2),(7,np.pi/2-.3)]: q[i]=[0,np.sin(yaw/2),0,np.cos(yaw/2)]
    return positions,q


def test_noncontiguous_keyframe_anchor_cumulative_units_short_mask_and_immutability():
    p,q=base(); keyframes=np.array([0,2,5,7],dtype=np.int64)
    before=(p.copy(),q.copy(),keyframes.copy())
    got=window_geometry(p,q,keyframes,1,max_future=4)
    assert got['now_frame_index']==2
    np.testing.assert_allclose(got['actions'],[[3,1,.2],[4,-1,-.3],[0,0,0],[0,0,0]],rtol=0,atol=1e-6)
    np.testing.assert_array_equal(got['valid'],[True,True,False,False])
    np.testing.assert_array_equal(got['future_frame_indices'],[5,7,-1,-1])
    assert got['actions'].dtype==np.float32 and got['valid'].dtype==np.bool_ and got['future_frame_indices'].dtype==np.int64
    for value,saved in zip((p,q,keyframes),before): np.testing.assert_array_equal(value,saved)


def test_default63_cap_uses_original_frame_indices_without_repeating_endpoint():
    p=np.zeros((160,3)); p[:,2]=-np.arange(160)*.25
    q=np.tile([0.,0.,0.,1.],(160,1)); keys=np.arange(0,160,2)
    got=window_geometry(p,q,keys,3)
    assert got['now_frame_index']==6 and got['actions'].shape==(63,3)
    np.testing.assert_array_equal(got['future_frame_indices'],np.arange(8,134,2))
    np.testing.assert_array_equal(got['valid'],np.ones(63,dtype=bool))
    np.testing.assert_allclose(got['actions'][:,0],np.arange(1,64)*.5,rtol=0,atol=0)
    np.testing.assert_array_equal(got['actions'][:,1:],np.zeros((63,2)))

@pytest.mark.parametrize('failure',['order','float_keys','range','terminal_anchor','boolean_anchor','zero_horizon'])
def test_invalid_indices_or_horizon_rejected(failure):
    p,q=base(); keys=[0,2,5]; anchor=0; horizon=63
    if failure=='order': keys=[0,5,2]
    elif failure=='float_keys': keys=[0.,2.,5.]
    elif failure=='range': keys=[0,8]
    elif failure=='terminal_anchor': anchor=2
    elif failure=='boolean_anchor': anchor=True
    else: horizon=0
    with pytest.raises(ValueError): window_geometry(p,q,keys,anchor,max_future=horizon)
