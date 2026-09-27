"""Replay alignment plus native Habitat API assembly, no simulator runtime."""
import copy
import sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'source'))
from lwm_stream.replay import replay_episode,NativeReplayBackend


def metadata():
    record={'id':7,'video':'sceneA_r2r_7','actions':[-1,1,2,3]}
    episode={'episode_id':'7','scene_id':'mp3d/sceneA/sceneA.glb','start_position':[1.,2.,3.],'start_rotation':[0.,0.,0.,1.]}
    return record,episode,['004.jpg','002.jpg','001.jpg','003.jpg']

class ReplayFake:
    scene_id='sceneA'
    def __init__(self,initial_offset=0,bad_state=None):
        self.calls=[]; self.p=np.zeros(3); self.q=np.zeros(4); self.initial_offset=initial_offset; self.bad_state=bad_state
    def reset(self,position,quaternion_xyzw):
        self.calls.append(('reset',list(position),list(quaternion_xyzw)))
        self.p[:]=position; self.p[0]+=self.initial_offset
        self.q[:]=-2*np.asarray(quaternion_xyzw) # same rotation; raw state preserved
        if self.bad_state=='initial_rotation': self.q[:]=[0,1,0,0]
    def step(self,action):
        self.calls.append(('step',action)); self.p[0]+=.07*action
        self.q[:]=[0.,np.sin(.03*action),0.,np.cos(.03*action)]
        if self.bad_state=='position': self.p[2]=np.nan
        if self.bad_state=='quaternion': self.q[:]=0
    def state(self):
        self.calls.append(('state',)); return self.p,self.q


def test_replay_exact_action_frame_order_buffer_copy_and_actual_poses():
    record,episode,names=metadata(); before=copy.deepcopy((record,episode,names)); backend=ReplayFake()
    got=replay_episode(backend,record,episode,names,dataset_source='r2r')
    assert got['frame_names']==['001.jpg','002.jpg','003.jpg','004.jpg']
    np.testing.assert_allclose(got['positions'],[[1,2,3],[1.07,2,3],[1.21,2,3],[1.42,2,3]],rtol=0,atol=1e-15)
    assert got['positions'].dtype==got['quaternions_xyzw'].dtype==np.float64
    assert [c for c in backend.calls if c[0]=='step']==[('step',1),('step',2),('step',3)]
    assert [c[0] for c in backend.calls]==['reset','state','step','state','step','state','step','state']
    np.testing.assert_array_equal(got['quaternions_xyzw'][0],[0,0,0,-2])
    backend.p[:]=999; backend.q[:]=999
    assert got['positions'][0,0]==1 and got['quaternions_xyzw'][0,3]==-2
    assert (record,episode,names)==before

@pytest.mark.parametrize('failure',['source','scene','initial_position','initial_rotation','position_nan','zero_quaternion'])
def test_replay_rejects_wrong_identity_reset_or_invalid_real_state(failure):
    record,episode,names=metadata(); backend=ReplayFake()
    source='r2r'
    if failure=='source': source='rxr'
    elif failure=='scene': backend.scene_id='another_scene'
    elif failure=='initial_position': backend.initial_offset=.1
    elif failure=='initial_rotation': backend.bad_state='initial_rotation'
    elif failure=='position_nan': backend.bad_state='position'
    else: backend.bad_state='quaternion'
    with pytest.raises(ValueError): replay_episode(backend,record,episode,names,dataset_source=source)
    if failure in ('source','scene','initial_position','initial_rotation'): assert not any(c[0]=='step' for c in backend.calls)

@pytest.fixture
def fake_habitat(monkeypatch):
    log=SimpleNamespace(simulators=[],from_calls=[],to_calls=[],path_loaded=True)
    class Config:
        def __init__(self,**kwargs): self.__dict__.update(kwargs)
    class AgentConfiguration(Config):
        def __init__(self): self.sensor_specifications=[]
    class ActuationSpec:
        def __init__(self,amount): self.amount=amount
    class ActionSpec:
        def __init__(self,name,actuation): self.name=name; self.actuation=actuation
    class Configuration:
        def __init__(self,sim_cfg,agents): self.sim_cfg=sim_cfg; self.agents=agents
    class Quaternion:
        def __init__(self,values): self.values=np.asarray(values).copy()
    def quat_from_coeffs(values): log.from_calls.append(np.asarray(values).copy()); return Quaternion(values)
    def quat_to_coeffs(value): log.to_calls.append(value); return value.values.copy()
    class Agent:
        def __init__(self): self.state=Config(position=np.zeros(3),rotation=Quaternion([0,0,0,1])); self.calls=[]
        def set_state(self,value,reset_sensors=True): self.state=value; self.calls.append(('set_state',reset_sensors))
        def get_state(self): self.calls.append(('get_state',)); return self.state
    class Simulator:
        def __init__(self,configuration):
            self.configuration=configuration; self.pathfinder=SimpleNamespace(is_loaded=log.path_loaded); self.agent=Agent(); self.calls=[]
            self.rgb=np.arange(18,dtype=np.uint8).reshape(2,3,3); log.simulators.append(self)
        def reset(self): self.calls.append(('reset',)); return {}
        def get_agent(self,index): assert index==0; return self.agent
        def step(self,action): self.calls.append(('step',action)); return {}
        def get_sensor_observations(self):
            self.calls.append(('observations',))
            sensor=self.configuration.agents[0].sensor_specifications[0]
            return {sensor.uuid:self.rgb}
        def close(self): self.calls.append(('close',))
    habitat=ModuleType('habitat_sim'); agent=ModuleType('habitat_sim.agent'); utils=ModuleType('habitat_sim.utils'); common=ModuleType('habitat_sim.utils.common')
    habitat.SimulatorConfiguration=Config; habitat.Configuration=Configuration; habitat.Simulator=Simulator
    habitat.AgentState=Config; habitat.CameraSensorSpec=Config
    habitat.SensorType=SimpleNamespace(COLOR='color'); habitat.SensorSubType=SimpleNamespace(PINHOLE='pinhole')
    agent.AgentConfiguration=AgentConfiguration; agent.ActionSpec=ActionSpec; agent.ActuationSpec=ActuationSpec
    habitat.agent=agent; habitat.utils=utils; utils.common=common
    common.quat_from_coeffs=quat_from_coeffs; common.quat_to_coeffs=quat_to_coeffs
    for name,value in [('habitat_sim',habitat),('habitat_sim.agent',agent),('habitat_sim.utils',utils),('habitat_sim.utils.common',common)]: monkeypatch.setitem(sys.modules,name,value)
    return log


def backend_args(tmp_path):
    path=tmp_path/'sceneA.glb'; path.write_bytes(b'fake scene: API fixture only')
    return path,{'agent_height':1.5,'agent_radius':.1,'forward_step_size':.25,'turn_angle':15.,'allow_sliding':True}


def test_native_sensorless_original_config_reset_quaternion_actions_close(fake_habitat,tmp_path):
    path,kwargs=backend_args(tmp_path); backend=NativeReplayBackend(path,**kwargs)
    sim=fake_habitat.simulators[0]; cfg=sim.configuration.sim_cfg; agent=sim.configuration.agents[0]
    assert str(cfg.scene_id)==str(path) and cfg.create_renderer is False
    assert cfg.enable_physics is False and cfg.load_semantic_mesh is False and cfg.allow_sliding is True
    assert agent.height==1.5 and agent.radius==.1 and agent.sensor_specifications==[]
    assert {v.name:v.actuation.amount for v in agent.action_space.values()}=={'move_forward':.25,'turn_left':15.,'turn_right':15.}
    assert backend.scene_id=='sceneA'
    backend.reset([1,2,3],[0,0,0,1]); p,q=backend.state()
    np.testing.assert_array_equal(p,[1,2,3]); np.testing.assert_array_equal(q,[0,0,0,1])
    assert sim.calls[0]==('reset',) and sim.agent.calls[0]==('set_state',True)
    assert len(fake_habitat.from_calls)==1 and len(fake_habitat.to_calls)==1
    for action in [1,2,3]: backend.step(action)
    assert [c for c in sim.calls if c[0]=='step']==[('step','move_forward'),('step','turn_left'),('step','turn_right')]
    with pytest.raises(ValueError): backend.step(0)
    backend.close(); assert sim.calls[-1]==('close',)


def test_native_rgb_sensor_source_intrinsics_and_original_observation(fake_habitat,tmp_path):
    path,kwargs=backend_args(tmp_path)
    backend=NativeReplayBackend(path,**kwargs,render_rgb=True,gpu_device_id=2,rgb_width=640,rgb_height=480,hfov=79,camera_height=1.25)
    sim=fake_habitat.simulators[0]; cfg=sim.configuration.sim_cfg; sensor=sim.configuration.agents[0].sensor_specifications[0]
    assert cfg.create_renderer is True and cfg.gpu_device_id==2
    assert sensor.sensor_type=='color' and sensor.sensor_subtype=='pinhole'
    assert sensor.resolution==[480,640] and sensor.position==[0,1.25,0] and float(sensor.hfov)==79
    np.testing.assert_array_equal(backend.rgb(),sim.rgb)
    backend.close()

@pytest.mark.parametrize('invalid',[{'agent_height':0},{'turn_angle':float('nan')},{'allow_sliding':1},{'render_rgb':True},{'rgb_width':True}])
def test_native_bad_configuration_rejected_before_simulator(fake_habitat,tmp_path,invalid):
    path,kwargs=backend_args(tmp_path); kwargs.update(invalid)
    with pytest.raises(ValueError): NativeReplayBackend(path,**kwargs)
    assert fake_habitat.simulators==[]


def test_native_rejects_unloaded_navmesh(fake_habitat,tmp_path):
    path,kwargs=backend_args(tmp_path); fake_habitat.path_loaded=False
    with pytest.raises(ValueError): NativeReplayBackend(path,**kwargs)
