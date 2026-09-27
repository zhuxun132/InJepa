"""Opt-in thin extension of official ImageGoalSensor; default class untouched."""
import math
from .core import final_approach_rotation, vector3


def _pose(episode):
    info = episode.info
    if not isinstance(info, dict) or info.get('goal_view_mode') not in ('approach_aligned_v1', 'approach_yaw_repair_v1', 'approach_yaw_repair_v2'):
        raise ValueError('explicit approach-aligned episode metadata required')
    anchor = vector3(info.get('goal_approach_anchor'))
    goal = vector3(episode.goals[0].position)
    arc = info.get('approach_anchor_arc_m', 1.)
    if type(arc) not in (int, float) or not math.isfinite(arc) or arc <= 0:
        raise ValueError('invalid approach arc length')
    if not 0 < math.dist(anchor, goal) <= arc + 1e-5:
        raise ValueError('goal anchor chord exceeds declared approach arc')
    rotation = info.get('goal_image_rotation')
    if (not isinstance(rotation, (list, tuple)) or len(rotation) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in rotation)
            or not math.isclose(sum(v*v for v in rotation), 1., abs_tol=1e-6)):
        raise ValueError('goal rotation must be a finite unit xyzw quaternion')
    expected = final_approach_rotation([anchor, goal], arc)
    mode = info['goal_view_mode']
    offset = 0
    if mode in ('approach_yaw_repair_v1', 'approach_yaw_repair_v2'):
        bound = 90 if mode == 'approach_yaw_repair_v1' else 180
        offset = info.get('goal_view_yaw_offset_deg')
        if type(offset) not in (int, float) or not math.isfinite(offset) or not -bound <= offset <= bound:
            raise ValueError('repair yaw offset must be finite in [-%s,%s]' % (bound,bound))
        yaw = 2 * math.atan2(expected[1], expected[3]) + math.radians(offset)
        expected = [0., math.sin(yaw/2), 0., math.cos(yaw/2)]
    if min(sum((a-b)**2 for a, b in zip(rotation, expected)),
           sum((a+b)**2 for a, b in zip(rotation, expected))) > 1e-10:
        raise ValueError('goal rotation differs from final path approach')
    return goal, list(rotation), (tuple(anchor), arc, mode, offset)


def make_view_aligned_sensor_class(official_base):
    class ViewAlignedImageGoalSensor(official_base):
        def get_observation(self, *args, episode, **kwargs):
            goal, rotation, path = _pose(episode)  # Validate BEFORE inherited cache hit.
            key = (episode.scene_id, episode.episode_id)
            identity = (tuple(goal), tuple(rotation), path)
            if (getattr(self, '_aligned_episode_key', None) == key
                    and self._aligned_pose_identity != identity):
                raise ValueError('same episode identity changed path/goal pose')
            result = super().get_observation(*args, episode=episode, **kwargs)
            self._aligned_episode_key, self._aligned_pose_identity = key, identity
            return result

        def _get_pointnav_episode_image_goal(self, episode):
            goal, rotation, _ = _pose(episode)
            observation = self._sim.get_observations_at(
                position=goal, rotation=rotation, keep_agent_at_new_pose=False)
            return observation[self._rgb_sensor_uuid]
    return ViewAlignedImageGoalSensor


def register_sensor():
    from habitat.core.registry import registry
    from habitat.tasks.nav.nav import ImageGoalSensor
    name = 'ViewAlignedImageGoalSensor'
    existing = registry.get_sensor(name)
    if existing is not None:
        if existing.__module__ != __name__:
            raise ValueError('conflicting view-aligned sensor registry owner')
        return existing
    cls = make_view_aligned_sensor_class(ImageGoalSensor)
    registry.register_sensor(name=name)(cls)
    return cls
