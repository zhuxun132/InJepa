"""Unit/axis adapter to official Habitat-Sim continuous velocity control.

No primitive codebook, learned controller, motion integration, or arrival oracle.
The speed STOP rule is a provisional diagnostic borrowed from Habitat-Lab;
it is not a disclosed RAE-NWM arrival detector.
"""
import math
from .action_decoder import _triplet, _finite_real, unnormalize_xy


def decode_continuous_command(command, *, dt=1.0, translation_stop_speed=.025,
                              rotation_stop_speed=math.pi/180,
                              max_translation=.25, max_rotation=math.pi/12):
    values = _triplet(command)
    params = dict(dt=dt, translation_stop_speed=translation_stop_speed,
                  rotation_stop_speed=rotation_stop_speed,
                  max_translation=max_translation, max_rotation=max_rotation)
    params = {k: _finite_real(v, k) for k, v in params.items()}
    if any(params[k] <= 0 for k in ('dt', 'max_translation', 'max_rotation')):
        raise ValueError('time and motion bounds must be positive')
    if min(params['translation_stop_speed'], params['rotation_stop_speed']) < 0:
        raise ValueError('STOP speed thresholds must be nonnegative')
    dt = params['dt']
    x, y = map(float, unnormalize_xy(values[:2]))
    yaw = values[2]
    norm = math.hypot(x, y)
    stop = norm/dt < params['translation_stop_speed'] and abs(yaw)/dt < params['rotation_stop_speed']
    scale = min(1.0, params['max_translation']/norm) if norm else 1.0
    ex, ey = x*scale, y*scale
    eyaw = max(-params['max_rotation'], min(params['max_rotation'], yaw))
    if stop:
        ex = ey = eyaw = 0.0
    linear, angular = [-ey/dt, 0.0, -ex/dt], [0.0, eyaw/dt, 0.0]
    payload = {'action': 'stop'} if stop else {
        'action': 'rae_continuous', 'action_args': {
            'linear_velocity': linear, 'angular_velocity': angular, 'time_step': dt}}
    return dict(name='STOP' if stop else 'CONTINUOUS', is_stop=bool(stop),
                habitat_payload=payload, metric_delta=[x, y, yaw],
                executed_metric_delta=[ex, ey, eyaw], linear_velocity=linear,
                angular_velocity=angular, parameters=params)
