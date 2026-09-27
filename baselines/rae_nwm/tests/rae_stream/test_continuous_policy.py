import numpy as np
import pytest
from rae_stream.habitat_policy import RAEStreamPolicy


def test_continuous_policy_preserves_triplet_and_never_calls_primitive_decoder():
    seen=[]
    def backend(context,goal):seen.append(context.copy());return [.003,.007,.02]
    class Decoder:
        def decode(self,command):raise AssertionError('continuous mode called nearest primitive')
    policy=RAEStreamPolicy(backend,decoder=Decoder(),mode='continuous')
    rgb=np.ones((2,2,3),dtype=np.uint8);policy.reset(rgb)
    output=policy.act(rgb,rgb,[{'rgb':rgb*2,'action':'CONTINUOUS','order':0,'mask':True}])
    assert set(output)=={'continuous_action'}
    assert tuple(output['continuous_action'])==(.003,.007,.02)
    assert policy.last_decision.action=='CONTINUOUS'
    assert tuple(policy.last_decision.continuous_command)==(.003,.007,.02)
    assert np.all(seen[0][-2]==2) and np.all(seen[0][-1]==1)


def test_continuous_policy_rejects_primitive_backend_output():
    policy=RAEStreamPolicy(lambda context,goal:1,mode='continuous')
    rgb=np.ones((2,2,3),dtype=np.uint8);policy.reset(rgb)
    with pytest.raises((TypeError,ValueError)):policy.act(rgb,rgb,[])
