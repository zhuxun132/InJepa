"""Recover measured poses by replaying published StreamVLN actions."""
from pathlib import Path
import numbers

import numpy as np

from .data import _array, _quaternions, join_episode, validate_annotation


def replay_episode(backend, record, episode, frame_names, *, dataset_source):
    names = validate_annotation(record, frame_names)
    joined = join_episode(record, [episode], dataset_source)
    if Path(backend.scene_id).stem != Path(joined['scene_id']).stem:
        raise ValueError('backend scene differs from episode')
    backend.reset(joined['start_position'], joined['start_rotation'])
    positions, rotations = [], []
    for index, action in enumerate(record['actions']):
        if index:
            backend.step(action)
        position, rotation = backend.state()
        p = _array([position], 3, 'replayed position')[0].copy()
        q = _array([rotation], 4, 'replayed quaternion')[0].copy()
        normalized = _quaternions([q])[0]
        if not index:
            expected = _quaternions([joined['start_rotation']])[0]
            if not np.allclose(p, joined['start_position'], rtol=0, atol=1e-5):
                raise ValueError('reset position differs from episode')
            if abs(abs(np.dot(normalized, expected)) - 1) > 1e-6:
                raise ValueError('reset rotation differs from episode')
        positions.append(p)
        rotations.append(q)
    return {'positions': np.stack(positions),
            'quaternions_xyzw': np.stack(rotations), 'frame_names': names}


class NativeReplayBackend:
    def __init__(self, scene_path, *, agent_height, agent_radius,
                 forward_step_size, turn_angle, allow_sliding,
                 render_rgb=False, gpu_device_id=None, rgb_width=640,
                 rgb_height=480, hfov=79, camera_height=1.25):
        path = Path(scene_path)
        if not path.is_file():
            raise ValueError('scene file is missing')
        for value in (agent_height, agent_radius, forward_step_size,
                      turn_angle, hfov, camera_height):
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not np.isfinite(value) or value <= 0):
                raise ValueError('dimensions and motion parameters must be finite positive numbers')
        if hfov >= 180:
            raise ValueError('hfov must be less than 180 degrees')
        for value in (rgb_width, rgb_height):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError('RGB dimensions must be positive integers')
        if not isinstance(allow_sliding, bool) or not isinstance(render_rgb, bool):
            raise ValueError('sliding and rendering flags must be boolean')
        if gpu_device_id is not None and (isinstance(gpu_device_id, bool)
                or not isinstance(gpu_device_id, numbers.Integral) or gpu_device_id < 0):
            raise ValueError('GPU device must be a nonnegative integer')
        if render_rgb and gpu_device_id is None:
            raise ValueError('rendering requires an explicit GPU device')

        import habitat_sim
        from habitat_sim.utils.common import quat_from_coeffs, quat_to_coeffs

        self._habitat = habitat_sim
        self._from_coeffs, self._to_coeffs = quat_from_coeffs, quat_to_coeffs
        self.scene_id = path.stem
        self._render_rgb = render_rgb
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = str(path)
        cfg.create_renderer = render_rgb
        cfg.enable_physics = False
        cfg.load_semantic_mesh = False
        cfg.allow_sliding = allow_sliding
        if render_rgb:
            cfg.gpu_device_id = int(gpu_device_id)
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height, agent.radius = agent_height, agent_radius
        self._actions = {1: 'move_forward', 2: 'turn_left', 3: 'turn_right'}
        agent.action_space = {
            name: habitat_sim.agent.ActionSpec(name, habitat_sim.agent.ActuationSpec(amount=amount))
            for name, amount in [('move_forward', forward_step_size),
                                 ('turn_left', turn_angle), ('turn_right', turn_angle)]}
        agent.sensor_specifications = []
        if render_rgb:
            sensor = habitat_sim.CameraSensorSpec()
            sensor.uuid = 'rgb'
            sensor.sensor_type = habitat_sim.SensorType.COLOR
            sensor.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor.resolution = [int(rgb_height), int(rgb_width)]
            sensor.position = [0, camera_height, 0]
            sensor.hfov = hfov
            agent.sensor_specifications = [sensor]
        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        if not self._sim.pathfinder.is_loaded:
            self.close()
            raise ValueError('scene navmesh is not loaded')

    def reset(self, position, quaternion_xyzw):
        p = _array([position], 3, 'reset position')[0]
        q = _quaternions([quaternion_xyzw])[0]
        self._sim.reset()
        state = self._habitat.AgentState()
        state.position = p
        state.rotation = self._from_coeffs(q)
        self._sim.get_agent(0).set_state(state, reset_sensors=True)

    def state(self):
        state = self._sim.get_agent(0).get_state()
        return np.asarray(state.position).copy(), self._to_coeffs(state.rotation).copy()

    def step(self, action):
        if isinstance(action, bool) or not isinstance(action, numbers.Integral) or action not in self._actions:
            raise ValueError('only original motion action codes 1/2/3 are valid')
        self._sim.step(self._actions[action])

    def rgb(self):
        if not self._render_rgb:
            raise ValueError('RGB sensor was not enabled')
        return self._sim.get_sensor_observations()['rgb']

    def close(self):
        self._sim.close()


def group_replay_jobs(rows, *, runtime_identity, scene_identities, splits):
    import copy
    import hashlib
    import json

    if not isinstance(runtime_identity, str) or not runtime_identity:
        raise ValueError('runtime identity is required')
    scene_splits = {}
    for split in ('train', 'dev', 'final_mp3d'):
        for scene in splits[split]:
            if scene in scene_splits:
                raise ValueError('split scenes must be unique and disjoint')
            scene_splits[scene] = split
    groups, seen = {}, set()
    for row in rows:
        record, source = row['record'], row['source']
        episode = join_episode(record, [row['episode']], source)
        actions = record.get('actions', [])
        validate_annotation(record, [str(i) + '.jpg' for i in range(len(actions))])
        scene = Path(episode['scene_id']).stem
        split = scene_splits.get(scene)
        if split not in ('train', 'dev'):
            raise ValueError('replay scene is not admitted for training or development')
        asset = scene_identities.get(scene)
        if not isinstance(asset, str) or not asset:
            raise ValueError('scene asset identity is required')
        alias_key = (source, record['video'])
        if alias_key in seen:
            raise ValueError('duplicate source/video alias')
        seen.add(alias_key)
        identity = dict(runtime=runtime_identity, asset=asset, scene=scene,
                        position=episode['start_position'], rotation=episode['start_rotation'],
                        actions=actions)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
        if key not in groups:
            groups[key] = dict(key=key, scene=scene, split=split,
                               row=copy.deepcopy(row), aliases=[])
        groups[key]['aliases'].append(dict(source=source, video=record['video'],
                                          id=record['id']))
    for group in groups.values():
        group['aliases'].sort(key=lambda alias: (alias['source'], alias['video']))
    return [groups[key] for key in sorted(groups)]
