import math

import numpy as np
import pytest

from rae_stream.action_decoder import decode_first_delta, unnormalize_xy


def test_normalized_forward_decodes_to_forward():
    # Official affine normalization maps one waypoint unit to 1/64 in [-1, 1].
    decoded = decode_first_delta([1.0 / 64.0, 0.0, 0.0])
    assert decoded.action_name == "FWD"
    assert decoded.action == 1
    np.testing.assert_allclose(decoded.metric_delta, [0.25, 0.0, 0.0], atol=1e-7)


def test_zero_delta_is_deterministically_stop():
    decoded = decode_first_delta([0.0, 0.0, 0.0])
    assert decoded.action_name == "STOP"
    assert decoded.action == 0


def test_tie_break_order_is_fixed_and_threshold_is_fail_closed():
    # Exactly equidistant between STOP and FWD in metric space; STOP wins by registered order.
    decoded = decode_first_delta([0.5 / 64.0, 0.0, 0.0])
    assert decoded.action_name == "STOP"
    with pytest.raises(ValueError, match="decode distance"):
        decode_first_delta([100.0, 100.0, 0.0], max_distance=1.0)


def test_xy_unnormalization_does_not_scale_yaw():
    xy = unnormalize_xy([0.0, 0.0], spacing=0.25)
    np.testing.assert_allclose(xy, [0.0, 0.0])
    with pytest.raises(ValueError, match="three"):
        decode_first_delta([0.0, 0.0])


def test_decoder_rejects_batched_continuous_command_instead_of_flattening():
    with pytest.raises(ValueError, match="single"):
        decode_first_delta(np.zeros((1, 3), dtype=np.float32))
