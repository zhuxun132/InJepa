"""RGB-only original LWM inference and native Habitat velocity translation."""
import math
import numpy as np
import torch
from PIL import Image
from lwm.preprocess import get_image_transform
from .stage_data import sample_rewards


class LWMNavigationAdapter:
    def __init__(self, policy, wm, tokenizer, *, device, num_sample=32, temperature=1.):
        if isinstance(num_sample, bool) or not isinstance(num_sample, int) or num_sample < 1:
            raise ValueError('num_sample must be positive')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be positive finite')
        self.policy, self.wm, self.tokenizer = policy.eval(), wm.eval(), tokenizer
        self.device, self.num_sample, self.temperature = torch.device(device), num_sample, temperature
        self.transform = get_image_transform()

    def reset(self, goal_rgb):
        pass

    @torch.no_grad()
    def act(self, current_rgb, goal_rgb, history):
        def transform(value):
            if not isinstance(value, Image.Image):
                value = Image.fromarray(np.asarray(value))
            return self.transform(value.convert('RGB')).unsqueeze(0).to(self.device)
        result = sample_rewards(self.policy, self.wm, self.tokenizer, transform(current_rgb), transform(goal_rgb),
                                num_sample=self.num_sample, temperature=self.temperature)
        best = int(result['rewards'][0].argmax())
        stop = int(result['lengths'][0, best]) == 0
        point = [0., 0.] if stop else result['actions_m'][0, best, 0, :2].cpu().tolist()
        return {'continuous_action': {'waypoint_m': point, 'stop': stop}}


class VelocityHandler:
    def __init__(self, *, linear_range, angular_range_deg, time_step, min_abs_linear, min_abs_angular_deg):
        self.linear = tuple(float(v) for v in linear_range)
        self.angular = tuple(float(v) for v in angular_range_deg)
        values = (*self.linear, *self.angular, time_step, min_abs_linear, min_abs_angular_deg)
        if (len(self.linear) != 2 or len(self.angular) != 2 or not all(math.isfinite(v) for v in values)
                or self.linear[0] != 0 or self.linear[1] <= 0
                or not self.angular[0] < 0 < self.angular[1] or time_step <= 0
                or min_abs_linear < 0 or min_abs_angular_deg < 0):
            raise ValueError('invalid native velocity ranges, interval or STOP thresholds')
        self.dt, self.min_linear, self.min_angular = time_step, min_abs_linear, min_abs_angular_deg

    def __call__(self, raw):
        command = raw['continuous_action']
        point = command['waypoint_m']
        if len(point) != 2 or not all(math.isfinite(v) for v in point) or type(command['stop']) is not bool:
            raise ValueError('expected finite forward/left waypoint and explicit boolean stop')
        heading = math.atan2(point[1], point[0])
        linear = min(self.linear[1], max(0., math.hypot(*point) * max(math.cos(heading), 0.) / self.dt))
        angular = min(self.angular[1], max(self.angular[0], math.degrees(heading) / self.dt))
        if command['stop'] or (abs(linear) < self.min_linear and abs(angular) < self.min_angular):
            return {'name': 'STOP', 'is_stop': True, 'habitat_payload': {'action': 'stop'}}
        def normalize(value, bounds):
            return 2. * (value - bounds[0]) / (bounds[1] - bounds[0]) - 1.
        return {'name': 'CONTINUOUS', 'is_stop': False,
                'habitat_payload': {'action': 'velocity_control', 'action_args': {
                    'linear_velocity': normalize(linear, self.linear),
                    'angular_velocity': normalize(angular, self.angular)}}}
