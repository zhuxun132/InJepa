import math

import numpy as np
import pytest

from rae_stream.geometry import (
    ACTION_BOS,
    ACTION_FWD,
    ACTION_LEFT,
    ACTION_RIGHT,
    action_to_twist,
    integrate_actions,
    inverse_pose,
    relative_pose,
    wrap_angle,
)


def test_single_forward_uses_world_x_axis():
    poses = integrate_actions([-1, ACTION_FWD])
    np.testing.assert_allclose(poses[-1], [0.25, 0.0, 0.0], atol=1e-7)


def test_left_then_forward_rotates_forward_vector():
    poses = integrate_actions([ACTION_BOS, ACTION_LEFT, ACTION_FWD])
    expected = [0.25 * math.cos(math.pi / 12), 0.25 * math.sin(math.pi / 12), math.pi / 12]
    np.testing.assert_allclose(poses[-1], expected, atol=1e-7)


def test_right_turn_has_no_translation_and_wraps():
    poses = integrate_actions([-1, ACTION_RIGHT])
    np.testing.assert_allclose(poses[-1], [0.0, 0.0, -math.pi / 12], atol=1e-7)
    assert math.isclose(wrap_angle(3 * math.pi), -math.pi)


def test_relative_pose_and_inverse_are_group_consistent():
    poses = integrate_actions([-1, ACTION_LEFT, ACTION_FWD, ACTION_RIGHT])
    start, goal = poses[1], poses[3]
    rel = relative_pose(start, goal)
    recovered = inverse_pose(start, rel)
    np.testing.assert_allclose(recovered, goal, atol=1e-7)


def test_bos_must_be_first_and_stop_is_rejected():
    assert action_to_twist(ACTION_BOS) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="STOP"):
        integrate_actions([-1, 0])
    with pytest.raises(ValueError, match="BOS"):
        integrate_actions([ACTION_FWD, ACTION_BOS])
