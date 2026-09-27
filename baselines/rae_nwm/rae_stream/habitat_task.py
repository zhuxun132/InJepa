"""Connect Habitat-Lab task actions to Habitat-Sim's official VelocityControl.

The stock Lab VelocityAction exposes forward speed only. This task seam exposes
the existing Sim vector interface; integration and NavMesh collision filtering
remain upstream. Collision bookkeeping follows Habitat-Lab v0.2.4 nav.py
(Meta Platforms, MIT license). It implements no planner or arrival detector.
"""
import numpy as np


def register_continuous_action():
    from habitat.core.registry import registry
    from habitat.core.embodied_task import SimulatorTaskAction
    from habitat_sim.physics import VelocityControl
    from habitat_sim import RigidState
    from habitat_sim.utils.common import quat_to_magnum, quat_from_magnum

    @registry.register_task_action
    class RAEContinuousVelocityAction(SimulatorTaskAction):
        def step(self, *, linear_velocity, angular_velocity, time_step, **kwargs):
            linear = np.asarray(linear_velocity, dtype=np.float64)
            angular = np.asarray(angular_velocity, dtype=np.float64)
            if linear.shape != (3,) or angular.shape != (3,):
                raise ValueError('velocity vectors must have three components')
            if not np.isfinite(linear).all() or not np.isfinite(angular).all():
                raise ValueError('velocity vectors must be finite')
            if isinstance(time_step, bool) or not np.isfinite(time_step) or time_step <= 0:
                raise ValueError('time_step must be positive and finite')
            if linear[1] != 0 or angular[0] != 0 or angular[2] != 0:
                raise ValueError('only planar translation and yaw are admitted')
            control = VelocityControl()
            control.controlling_lin_vel = control.controlling_ang_vel = True
            control.lin_vel_is_local = control.ang_vel_is_local = True
            control.linear_velocity = linear
            control.angular_velocity = angular
            before = self._sim.get_agent_state()
            after = control.integrate_transform(
                float(time_step), RigidState(quat_to_magnum(before.rotation), before.position))
            pathfinder = self._sim.pathfinder
            step = pathfinder.try_step if self._sim.config.sim_cfg.allow_sliding else pathfinder.try_step_no_sliding
            position = step(before.position, after.translation)
            observations = self._sim.get_observations_at(
                position=position, rotation=quat_from_magnum(after.rotation),
                keep_agent_at_new_pose=True)
            requested = np.asarray(after.translation) - before.position
            actual = np.asarray(position) - before.position
            self._sim._prev_sim_obs['collided'] = bool(actual.dot(actual) + 1e-5 < requested.dot(requested))
            return observations

    return RAEContinuousVelocityAction
