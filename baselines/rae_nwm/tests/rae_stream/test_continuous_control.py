"""Diagnostic planar SE2 decoding preserves lateral motion and physical units."""
import math
import pytest
from rae_stream.continuous_control import decode_continuous_command


def test_full_planar_axes_and_time_units_without_quantization():
    out=decode_continuous_command([1/64,2/64,math.pi/12],dt=.5,max_translation=10,max_rotation=10)
    assert out["metric_delta"]==pytest.approx([.25,.5,math.pi/12])
    assert out["executed_metric_delta"]==pytest.approx(out["metric_delta"])
    assert out["linear_velocity"]==pytest.approx([-1.,0.,-.5])
    assert out["angular_velocity"]==pytest.approx([0.,math.pi/6,0.])
    assert out["is_stop"] is False and out["name"]=="CONTINUOUS"


def test_norm_clip_preserves_translation_direction_and_reports_requested_delta():
    out=decode_continuous_command([1/64,2/64,1.])
    dx,dy,yaw=out["executed_metric_delta"]
    assert math.hypot(dx,dy)==pytest.approx(.25)
    assert dy/dx==pytest.approx(2.)
    assert yaw==pytest.approx(math.pi/12)
    assert out["metric_delta"]==pytest.approx([.25,.5,1.])


@pytest.mark.parametrize("command",[[0,0,0],[.0001,0,0],[0,.0001,.001]])
def test_small_translation_and_rotation_dispatch_native_stop(command):
    out=decode_continuous_command(command)
    assert out["is_stop"] is True
    assert out["habitat_payload"]=={"action":"stop"}


@pytest.mark.parametrize("command",[[0,1/64,0],[0,0,math.pi/12],[1/128,0,0]])
def test_lateral_rotation_and_translation_each_prevent_false_stop(command):
    assert decode_continuous_command(command)["is_stop"] is False


def test_clipping_cannot_turn_requested_motion_into_stop():
    out=decode_continuous_command([1/64,0,0],max_translation=.001)
    assert out["is_stop"] is False
    assert out["executed_metric_delta"][0]==pytest.approx(.001)


@pytest.mark.parametrize("kwargs",[{"dt":0},{"dt":-1},{"dt":float('nan')},{"dt":True},
    {"translation_stop_speed":-1},{"rotation_stop_speed":float('inf')},
    {"max_translation":0},{"max_rotation":float('nan')}])
def test_invalid_physical_parameters_reject(kwargs):
    with pytest.raises((ValueError,TypeError)):decode_continuous_command([0,0,0],**kwargs)


@pytest.mark.parametrize("command",[[0,0],[float('nan'),0,0],[0,float('inf'),0],[0,0,True]])
def test_malformed_nonfinite_commands_reject(command):
    with pytest.raises((ValueError,TypeError)):decode_continuous_command(command)
