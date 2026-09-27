"""Explicit diagnostic: execute a selected trajectory with command odometry.

Model proposals and scoring remain official. This control variant uses no actual
pose feedback: clipping is integrated, but collisions can cause open-loop drift.
"""
import copy
import math
from .adapter import Policy, _rgb
from .controller import waypoint_to_velocity


class FullSequencePolicy(Policy):
    def __init__(self, backend, *, dt, max_translation, max_yaw):
        super().__init__(backend)
        self.controls = dict(dt=dt, max_translation=max_translation, max_yaw=max_yaw)
        waypoint_to_velocity([0., 0., 0.], **self.controls)
        self._clear_sequence()

    def _clear_sequence(self):
        self._sequence = None
        self._origin_plan = None
        self._offset = self._real_step = self._planning_step = 0
        self._pose = (0., 0., 0.)

    def reset(self, goal_rgb):
        super().reset(goal_rgb)
        self._clear_sequence()

    def act(self, current_rgb, goal_rgb, factual_history):
        import numpy as np
        current, goal = _rgb(current_rgb), _rgb(goal_rgb)
        if self.goal is None:
            raise ValueError('reset required before act')
        if not np.array_equal(self.goal, goal):
            raise ValueError('goal changed without reset')
        replanned = self._sequence is None
        if replanned:
            plan = self.backend.plan(current, goal)
            try:
                length = plan['valid_length']
                sequence = copy.deepcopy(plan['metric_trajectory'])
                if type(length) is not int or length < 1 or length != len(sequence):
                    raise ValueError('valid trajectory length mismatch')
                for waypoint in sequence:
                    waypoint_to_velocity(waypoint, **self.controls)
            except (KeyError, TypeError) as error:
                raise ValueError('complete valid metric trajectory required') from error
            pose, offset, origin = (0., 0., 0.), 0, self._real_step
        else:
            plan, sequence = self._origin_plan, self._sequence
            pose, offset, origin = self._pose, self._offset, self._origin_step
        x, y, yaw = pose
        tx, ty, tyaw = sequence[offset]
        cosine, sine = math.cos(yaw), math.sin(yaw)
        dx, dy = tx - x, ty - y
        # Match the old first-waypoint command exactly. Later headings use the
        # principal relative angle in the current command-integrated frame.
        relative = list(sequence[0]) if offset == 0 else [
            cosine * dx + sine * dy, -sine * dx + cosine * dy,
            math.atan2(math.sin(tyaw-yaw), math.cos(tyaw-yaw))]
        control = waypoint_to_velocity(relative, **self.controls)
        ex, ey, eyaw = control['executed_metric_waypoint']
        next_pose = (x + cosine*ex - sine*ey, y + sine*ex + cosine*ey, yaw + eyaw)
        diagnostic = dict(requested_interval='full_valid_trajectory',
            length=len(sequence), offset=offset, replanned=replanned,
            plan_origin_real_step=origin, real_step=self._real_step,
            planning_step=self._planning_step if replanned else self._planning_step-1,
            pose_source='integrated_clipped_commands_no_oracle',
            nominal_pose_before=list(pose), nominal_pose_after=list(next_pose),
            plan_origin_waypoint=list(sequence[offset]), local_command=relative)
        # Commit only after plan, coordinate transform and controls validate.
        self.last_plan = dict(copy.deepcopy(plan), execution=diagnostic)
        self._origin_plan, self._origin_step = copy.deepcopy(plan), origin
        self._offset, self._pose = offset + 1, next_pose
        self._sequence = sequence if offset + 1 < len(sequence) else None
        self._real_step += 1
        self._planning_step += int(replanned)
        return {'continuous_action': relative}

    def close(self):
        super().close()
        self._clear_sequence()
