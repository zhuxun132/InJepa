"""Scientific contracts for LWM's StreamVLN pose sidecars; no simulator needed."""
import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))


@pytest.fixture
def api():
    try:
        return importlib.import_module("lwm_stream.data")
    except ModuleNotFoundError as exc:
        if exc.name in {"lwm_stream", "lwm_stream.data"}:
            pytest.fail("RED: lwm_stream.data scientific replay contracts are not implemented", pytrace=False)
        raise


def record(actions=None):
    return {"id": 7, "video": "images/sceneA_r2r_7", "instructions": ["go"],
            "actions": [-1, 1, 2] if actions is None else actions}


def episode(scene="sceneA", eid="7"):
    return {"episode_id": eid, "scene_id": f"mp3d/{scene}/{scene}.glb",
            "start_position": [0., 0., 0.], "start_rotation": [0., 0., 0., 1.]}


def quats(*angles):
    return np.array([[0., np.sin(a / 2), 0., np.cos(a / 2)] for a in angles])


@pytest.mark.parametrize("names,expected", [
    (["002.jpg", "000.jpg", "001.jpg"], ["000.jpg", "001.jpg", "002.jpg"]),
    (["003.jpg", "001.jpg", "002.jpg"], ["001.jpg", "002.jpg", "003.jpg"]),
])
def test_annotation_preserves_real_frame_numbers_and_bos_alignment(api, names, expected):
    rec = record()
    before = copy.deepcopy(rec), names[:]
    assert list(api.validate_annotation(rec, names)) == expected
    assert (rec, names) == before


@pytest.mark.parametrize("actions,names", [
    ([], []), ([1, 1, 2], ["0.jpg", "1.jpg", "2.jpg"]),
    ([-1, -1, 2], ["0.jpg", "1.jpg", "2.jpg"]),
    ([-1, 4, 2], ["0.jpg", "1.jpg", "2.jpg"]),
    ([-1, 0, 2], ["0.jpg", "1.jpg", "2.jpg"]),
    ([-1, 1, 0], ["0.jpg", "1.jpg", "2.jpg"]),
    ([-1, True, 2], ["0.jpg", "1.jpg", "2.jpg"]),
    ([-1, 1.5, 2], ["0.jpg", "1.jpg", "2.jpg"]),
    ([-1, 1, 2], ["0.jpg", "1.jpg"]),
    ([-1, 1, 2], ["0.jpg", "1.jpg", "3.jpg"]),
    ([-1, 1, 2], ["2.jpg", "3.jpg", "4.jpg"]),
    ([-1, 1, 2], ["0.jpg", "1.jpg", "01.jpg"]),
    ([-1, 1, 2], ["0.jpg", "1.jpg", "invalid.jpg"]),
])
def test_annotation_rejects_alignment_and_action_mutations(api, actions, names):
    with pytest.raises(ValueError):
        api.validate_annotation(record(actions), names)


def test_annotation_bos_only_is_one_reset_frame(api):
    assert list(api.validate_annotation(record([-1]), ["001.jpg"])) == ["001.jpg"]


def test_join_matches_scene_and_id_without_mutating_inputs(api):
    target = episode()
    rows = [episode("sceneB"), episode(eid="8"), target]
    rec = record()
    before = copy.deepcopy((rec, rows))
    assert api.join_episode(rec, rows, "r2r") == target
    assert (rec, rows) == before


@pytest.mark.parametrize("rows,source", [
    ([], "r2r"), ([episode("sceneB")], "r2r"),
    ([episode(eid="8")], "r2r"), ([episode(), episode()], "r2r"),
    ([episode()], "rxr"),
    ([dict(episode(), dataset_source="rxr")], "r2r"),
])
def test_join_fails_closed_on_missing_ambiguous_or_cross_source(api, rows, source):
    with pytest.raises(ValueError):
        api.join_episode(record(), rows, source)


@pytest.mark.parametrize("field,value", [
    ("start_position", [0., 0.]), ("start_position", [0., np.nan, 0.]),
    ("start_rotation", [0., 0., 0., 0.]),
    ("start_rotation", [0., 0., np.inf, 1.]),
    ("start_rotation", [0., 0., 1.]),
])
def test_join_rejects_invalid_initial_state(api, field, value):
    row = episode()
    row[field] = value
    with pytest.raises(ValueError):
        api.join_episode(record(), [row], "r2r")


@pytest.mark.parametrize("field", ["start_position", "start_rotation"])
def test_join_rejects_missing_initial_state(api, field):
    row = episode()
    del row[field]
    with pytest.raises(ValueError):
        api.join_episode(record(), [row], "r2r")


def test_local_coordinates_are_cumulative_metres_forward_and_left(api):
    pos = np.array([[4., 2., 3.], [4., 2., 2.], [3., 2., 1.]])
    q = quats(0, 0, 0)
    before = pos.copy(), q.copy()
    actual = api.local_trajectory(pos, q)
    np.testing.assert_allclose(actual, [[0, 0, 0], [1, 0, 0], [2, 1, 0]], atol=1e-12)
    np.testing.assert_array_equal(pos, before[0])
    np.testing.assert_array_equal(q, before[1])


def test_origin_rotation_and_translation_use_local_frame(api):
    pos = np.array([[3., 2., 4.], [2., 2., 4.], [1., 2., 5.]])
    np.testing.assert_allclose(api.local_trajectory(pos, quats(*([np.pi / 2] * 3))),
                               [[0, 0, 0], [1, 0, 0], [2, 1, 0]], atol=1e-12)


def test_pure_turn_then_forward_retains_true_heading(api):
    pos = [[0., 0., 0.], [0., 0., 0.], [-1., 0., 0.]]
    result = api.local_trajectory(pos, quats(0, np.pi / 2, np.pi / 2))
    np.testing.assert_allclose(result, [[0, 0, 0], [0, 0, np.pi / 2], [0, 1, np.pi / 2]], atol=1e-12)


def test_yaw_wrap_and_quaternion_double_cover(api):
    angles = np.deg2rad([179., -179.])
    q = quats(*angles)
    pos = np.zeros((2, 3))
    expected = [[0, 0, 0], [0, 0, np.deg2rad(2.)]]
    np.testing.assert_allclose(api.local_trajectory(pos, q), expected, atol=1e-12)
    q[1] *= -1
    np.testing.assert_allclose(api.local_trajectory(pos, q), expected, atol=1e-12)


def test_nonzero_origin_is_independent_of_previous_history(api):
    pos = [[5., 0., 5.], [0., 0., 0.], [-2., 0., 0.]]
    q = quats(0, np.pi / 2, np.pi / 2)
    got = api.local_trajectory(pos, q, origin_index=1)
    np.testing.assert_allclose(got[1:], [[0, 0, 0], [2, 0, 0]], atol=1e-12)


def test_height_is_excluded_from_lwm_planar_trajectory(api):
    np.testing.assert_allclose(api.local_trajectory([[0, 0, 0], [0, 2, 0]], quats(0, 0)), np.zeros((2, 3)))


@pytest.mark.parametrize("positions,q,origin", [
    ([[0, 0]], [[0, 0, 0, 1]], 0),
    ([[0, 0, 0]], [[0, 0, 1]], 0),
    ([[0, 0, 0], [0, 0, 1]], [[0, 0, 0, 1]], 0),
    ([[0, 0, 0]], [[0, 0, 0, 0]], 0),
    ([[0, 0, 0]], [[0, np.nan, 0, 1]], 0),
    ([[np.inf, 0, 0]], [[0, 0, 0, 1]], 0),
    ([], [], 0),
    ([[0, 0, 0]], [[0, 0, 0, 1]], -1),
    ([[0, 0, 0]], [[0, 0, 0, 1]], 1),
])
def test_geometry_rejects_invalid_pose_alignment(api, positions, q, origin):
    with pytest.raises(ValueError):
        api.local_trajectory(positions, q, origin_index=origin)


def test_keyframes_use_distance_from_last_retained_not_path_length(api):
    positions = [[0, 0, 0], [.15, 0, 0], [0, 0, 0], [.2, 0, 0], [.25, 0, 0]]
    assert list(api.select_keyframes(positions)) == [0, 3]


def test_keyframes_are_horizontal_and_preserve_original_frame_indices(api):
    positions = [[0, 0, 0], [0, 10, 0], [0, 10, .2], [0, 10, .21], [0, 10, .5]]
    assert list(api.select_keyframes(positions)) == [0, 2, 4]


def test_keyframe_threshold_is_configurable(api):
    assert list(api.select_keyframes([[0, 0, 0], [0, 0, .25], [0, 0, .5]], min_distance=.5)) == [0, 2]


def test_keyframes_empty_or_stationary(api):
    assert list(api.select_keyframes(np.empty((0, 3)))) == []
    assert list(api.select_keyframes(np.zeros((4, 3)))) == [0]


@pytest.mark.parametrize("threshold", [0., -.2, np.nan, np.inf])
def test_keyframes_reject_invalid_threshold(api, threshold):
    with pytest.raises(ValueError):
        api.select_keyframes([[0, 0, 0]], min_distance=threshold)


@pytest.mark.parametrize("positions", [[[0, 0]], [[0, np.nan, 0]]])
def test_keyframes_reject_invalid_positions(api, positions):
    with pytest.raises(ValueError):
        api.select_keyframes(positions)
