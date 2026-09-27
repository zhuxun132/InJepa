"""Pure task-budget and path-pose contracts; no Habitat/model imports."""
import math
import random
from collections import deque


def allocate_scene_bin_quotas(scene_ids, *, total_episodes, bin_totals, seed):
    if (not isinstance(scene_ids, (list, tuple)) or not scene_ids
            or any(not isinstance(s, str) or not s for s in scene_ids)
            or len(set(scene_ids)) != len(scene_ids)):
        raise ValueError('scene identities must be unique nonempty strings')
    if (type(total_episodes) is not int or total_episodes <= 0
            or type(seed) is not int or not isinstance(bin_totals, (list, tuple))
            or not bin_totals or any(type(n) is not int or n < 0 for n in bin_totals)
            or sum(bin_totals) != total_episodes):
        raise ValueError('invalid exact episode budget, bin totals or seed')
    scenes = sorted(scene_ids)
    n = len(scenes)
    rng = random.Random(seed)
    extra = set(rng.sample(scenes, total_episodes % n))
    base = [count // n for count in bin_totals]
    row_need = {s: total_episodes // n + int(s in extra) - sum(base) for s in scenes}
    # Integral bipartite max-flow: each bin can add at most one per scene.
    source, sink = ('source',), ('sink',)
    capacity = {}
    neighbors = {}
    def edge(a, b, cap):
        capacity[a, b] = cap
        capacity[b, a] = 0
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)
    for j, count in enumerate(bin_totals):
        edge(source, ('bin', j), count % n)
        for scene in scenes:
            edge(('bin', j), ('scene', scene), 1)
    for scene in scenes:
        edge(('scene', scene), sink, row_need[scene])
    required = sum(row_need.values())
    for _ in range(required):
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            a = queue.popleft()
            for b in neighbors[a]:
                if b not in parent and capacity[a, b] > 0:
                    parent[b] = a
                    queue.append(b)
        if sink not in parent:
            raise ValueError('no balanced scene/bin allocation satisfies the exact budgets')
        b = sink
        while b != source:
            a = parent[b]
            capacity[a, b] -= 1
            capacity[b, a] += 1
            b = a
    return {s: [base[j] + capacity[('scene', s), ('bin', j)]
                for j in range(len(base))] for s in scenes}


def vector3(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('expected three finite numeric coordinates')
    return [float(x) for x in value]


def validated_path(path_points):
    if not isinstance(path_points, (list, tuple)) or len(path_points) < 2:
        raise ValueError('at least two path vertices required')
    return [vector3(p) for p in path_points]


def anchor_on_path(path_points, distance, from_end=False):
    points = validated_path(path_points)
    if type(distance) not in (int, float) or not math.isfinite(distance) or distance <= 0:
        raise ValueError('arc distance must be finite positive')
    if from_end:
        points = list(reversed(points))
    remaining = float(distance)
    for a, b in zip(points, points[1:]):
        length = math.dist(a, b)
        if length > 0 and remaining <= length:
            return [x + (y-x)*remaining/length for x, y in zip(a, b)]
        remaining -= length
    return points[-1]


def final_approach_rotation(path_points, approach_arc_length=1.):
    points = validated_path(path_points)
    anchor = anchor_on_path(points, approach_arc_length, from_end=True)
    goal = points[-1]
    dx, dz = goal[0] - anchor[0], goal[2] - anchor[2]
    if math.hypot(dx, dz) <= 1e-10:
        raise ValueError('path has no nonzero horizontal terminal approach')
    angle = math.atan2(-dx, -dz)
    return [0., math.sin(angle/2), 0., math.cos(angle/2)]


def initial_approach_rotation(path_points, approach_arc_length=1.):
    points = validated_path(path_points)
    return final_approach_rotation([points[0], anchor_on_path(points, approach_arc_length)], approach_arc_length)


def path_geometry(path_points, *, geodesic_distance):
    points = validated_path(path_points)
    if type(geodesic_distance) not in (int, float) or not math.isfinite(geodesic_distance) or geodesic_distance <= 0:
        raise ValueError('finite positive geodesic distance required')
    lengths, vectors = [], []
    for a, b in zip(points, points[1:]):
        v = [y-x for x, y in zip(a, b)]
        length = math.sqrt(sum(x*x for x in v))
        if length > 0:
            lengths.append(length)
        horizontal = math.hypot(v[0], v[2])
        if horizontal > 0:
            vectors.append([v[0]/horizontal, v[2]/horizontal])
    chord = math.dist(points[0], points[-1])
    if chord <= 1e-10 or not vectors:
        raise ValueError('degenerate path')
    turns = [math.degrees(math.acos(max(-1., min(1., sum(x*y for x,y in zip(a,b))))))
             for a,b in zip(vectors,vectors[1:])]
    return dict(ratio=float(geodesic_distance)/chord, euclidean_distance=chord,
                polyline_length=sum(lengths), nonzero_segments=len(vectors),
                corner_turns_deg=turns,cumulative_turn_deg=sum(turns),
                corner_definition='raw_nonzero_xz_segments_no_smoothing')


def validate_route_evidence(evidence, *, limits):
    def finite(name):
        value=evidence[name]
        if type(value) not in (int,float) or not math.isfinite(value):
            raise ValueError('nonfinite evidence: '+name)
        return value
    if not 1.-1e-5 <= finite('ratio') <= limits['ratio_max']:
        raise ValueError('route_ratio')
    if not 0 <= finite('cumulative_turn_deg') <= limits['cumulative_turn_max_deg']:
        raise ValueError('route_turn')
    clearance=evidence['endpoint_clearance']
    if len(clearance)!=2 or any(type(v) not in (int,float) or not math.isfinite(v)
            or v < limits['endpoint_extra_clearance'] for v in clearance):
        raise ValueError('endpoint_center_space_clearance')
    if evidence['initial_visible'] is not True:
        raise ValueError('initial_visibility')
    if evidence['initial_fwd_collided'] is not False or finite('initial_fwd_progress') < limits['start_fwd_min_projection_m']:
        raise ValueError('initial_forward_blocked')
    if not 0 <= finite('initial_fwd_lateral_error') <= limits['start_fwd_max_lateral_m']:
        raise ValueError('initial_forward_lateral')
    if not 0 <= finite('follower_turn_degrees') <= limits['follower_turn_max_deg']:
        raise ValueError('reference_turn_budget')
    if type(evidence['follower_steps']) is not int or not 0 <= evidence['follower_steps'] <= limits['max_replay_steps']:
        raise ValueError('reference_step_budget')
    if not 0 <= finite('follower_final_geodesic') < limits['arrival_radius']:
        raise ValueError('reference_controller_not_arrived')
    return dict(nonstraight=finite('ratio')>=limits['nonstraight_ratio_min'] and
                finite('cumulative_turn_deg')>=limits['nonstraight_turn_min_deg'])


def rgb_asset_quality(rgb, *, max_exact_black_fraction):
    """Narrow uniform asset-fault proxy; never alter the rendered RGB array."""
    import numpy as np
    threshold=max_exact_black_fraction
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('max_exact_black_fraction must be finite in [0,1]')
    if (not isinstance(rgb,np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0):
        raise ValueError('photo quality requires nonempty original HxWx3 uint8 RGB')
    black=int(np.count_nonzero(np.all(rgb == 0,axis=2)))
    total=int(rgb.shape[0]*rgb.shape[1]);fraction=black/total
    return dict(exact_black_pixels=black,total_pixels=total,exact_black_fraction=fraction,
                max_exact_black_fraction=float(threshold),accepted=fraction <= threshold)
