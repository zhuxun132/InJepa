"""Provisional metric waypoint interface to native original-frame velocity control."""
import math
import numbers


def waypoint_to_velocity(waypoint, *, dt, max_translation, max_yaw):
    try:values=list(waypoint)
    except TypeError as exc:raise ValueError('waypoint must be a triplet') from exc
    if len(values)!=3:
        raise ValueError('waypoint must be a triplet')
    for v in [*values,dt,max_translation,max_yaw]:
        if isinstance(v,bool) or not isinstance(v,numbers.Real) or not math.isfinite(v):
            raise ValueError('finite real values required')
    if min(dt,max_translation,max_yaw)<=0:
        raise ValueError('time and bounds must be positive')
    x,y,yaw=map(float,values)
    norm=math.hypot(x,y);scale=min(1.,max_translation/norm) if norm else 1.
    ex,ey=x*scale,y*scale;theta=max(-max_yaw,min(max_yaw,yaw))
    # Provisional robot frame: x forward/y left, positive yaw left.
    # Habitat-Sim v0.2.4 integrates local translation with the ORIGINAL rotation.
    linear=[-ey/dt,0.,-ex/dt]
    angular=[0.,theta/dt,0.]
    return {'linear_velocity':linear,'angular_velocity':angular,'time_step':float(dt),
            'metric_waypoint':[x,y,yaw],'executed_metric_waypoint':[ex,ey,theta],
            'clipped_translation':scale<1.,'clipped_yaw':theta!=yaw,
            'coordinate_convention':'provisional_x_forward_y_left_yaw_left',
            'integration_convention':'native_original_rotation_for_translation'}
