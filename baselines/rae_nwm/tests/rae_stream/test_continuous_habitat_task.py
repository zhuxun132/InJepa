"""Thin task glue delegates dynamics and collision to official Habitat APIs."""
import sys
from types import ModuleType, SimpleNamespace
import numpy as np


def test_task_uses_official_velocity_integration_and_navmesh(monkeypatch):
    registered={};calls=[]
    def module(name,**attrs):
        m=ModuleType(name);m.__dict__.update(attrs);monkeypatch.setitem(sys.modules,name,m);return m
    class Registry:
        def register_task_action(self,cls=None,**kwargs):
            def add(c):registered['action']=c;return c
            return add(cls) if cls else add
    class Base:
        def __init__(self,*args,sim=None,**kwargs):self._sim=sim
    class VelocityControl:
        def integrate_transform(self,dt,state):
            calls.append(('integrate',dt,state,self.linear_velocity,self.angular_velocity))
            assert self.controlling_lin_vel and self.lin_vel_is_local
            assert self.controlling_ang_vel and self.ang_vel_is_local
            return SimpleNamespace(translation=np.array([8.,0.,9.]),rotation='new_rotation')
    module('habitat');module('habitat.core');module('habitat.core.registry',registry=Registry())
    module('habitat.core.embodied_task',SimulatorTaskAction=Base)
    physics=module('habitat_sim.physics',VelocityControl=VelocityControl)
    module('habitat_sim',physics=physics,RigidState=lambda rotation,translation:SimpleNamespace(rotation=rotation,translation=translation))
    module('habitat_sim.utils');module('habitat_sim.utils.common',quat_to_magnum=lambda q:q,quat_from_magnum=lambda q:q)
    module('magnum',Vector3=lambda v:np.array(v))
    class Sim:
        def __init__(self):
            self.pathfinder=SimpleNamespace(try_step=self.try_step)
            self.config=SimpleNamespace(sim_cfg=SimpleNamespace(allow_sliding=True))
            self._prev_sim_obs={}
        def get_agent_state(self):return SimpleNamespace(position=np.array([1.,0.,2.]),rotation='original_rotation')
        def try_step(self,start,target):calls.append(('try_step',start,target));return np.array([7.,0.,8.])
        def get_observations_at(self,*,position,rotation,keep_agent_at_new_pose):
            calls.append(('observe',position,rotation,keep_agent_at_new_pose));return {'rgb':'real'}
    from rae_stream.habitat_task import register_continuous_action
    register_continuous_action()
    task=registered['action'](sim=Sim(),config=SimpleNamespace())
    observations=task.step(linear_velocity=[-.5,0.,-.25],angular_velocity=[0.,.2,0.],time_step=.5)
    assert observations=={'rgb':'real'}
    assert [row[0] for row in calls]==['integrate','try_step','observe']
    assert calls[0][1]==.5
    np.testing.assert_allclose(calls[0][3],[-.5,0.,-.25])
    np.testing.assert_allclose(calls[0][4],[0.,.2,0.])
    np.testing.assert_allclose(calls[1][1],[1.,0.,2.]);np.testing.assert_allclose(calls[1][2],[8.,0.,9.])
    np.testing.assert_allclose(calls[2][1],[7.,0.,8.]);assert calls[2][2]=='new_rotation';assert calls[2][3] is True
