"""Diagnostic whole-branch ranking; consistency is not collision probability."""

import math
from dataclasses import replace

import torch


def validate_whole_branch_options(options):
    required = {"risk_weight", "epsilon"}
    if (not isinstance(options, dict) or not required <= set(options)
            or set(options) - required - {"score_mode"}):
        raise ValueError("whole branch options require risk_weight and epsilon")
    mode = options.get("score_mode", "l1_consistency")
    if not isinstance(mode, str) or mode != "l1_consistency":
        raise ValueError("invalid whole branch score_mode")
    for name in required:
        value = options[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (name == "epsilon" and value == 0)):
            raise ValueError("invalid whole branch " + name)
    result = {name: float(options[name]) for name in required}
    if "score_mode" in options:
        result["score_mode"] = mode
    return result


def rank_whole_branches(ranking, *, current_grid, goal_grid,
                        risk_weight=1.0, epsilon=1e-6, score_mode="l1_consistency"):
    options = validate_whole_branch_options(dict(risk_weight=risk_weight,
        epsilon=epsilon, score_mode=score_mode))
    epsilon, risk_weight = options["epsilon"], options["risk_weight"]
    q, f = ranking.proposal_tape, ranking.endpoints
    if q.ndim != 5 or q.shape != f.shape or min(q.shape) < 1:
        raise ValueError("whole branch requires matching [B,K,H,M,D] grids")
    batch, modes, horizon, spatial, width = q.shape
    if (current_grid.shape != (batch, spatial, width) or goal_grid.shape != current_grid.shape
            or ranking.actions.shape != (batch, modes, horizon)
            or ranking.mode_indices.shape != (batch, modes)):
        raise ValueError("whole branch tensor axes disagree")
    values = (q, f, current_grid, goal_grid, ranking.actions, ranking.mode_indices,
              ranking.consistency, ranking.goal_distance, ranking.log_mass)
    if any(value is None for value in values):
        raise ValueError("whole branch requires full rollout evidence")
    if len({value.device for value in values}) != 1:
        raise ValueError("whole branch tensors must share a device")
    for value in values:
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("nonfinite whole branch input")
    if ranking.mode_indices.dtype != torch.int64 or ranking.actions.dtype != torch.int64:
        raise ValueError("whole branch indices/actions must be int64")
    mode_order = torch.argsort(ranking.mode_indices, stable=True, dim=1)
    sorted_modes = ranking.mode_indices.gather(1, mode_order)
    if bool((sorted_modes < 0).any()) or bool((sorted_modes[:, 1:] == sorted_modes[:, :-1]).any()):
        raise ValueError("mode indices must be unique and nonnegative")
    if not bool(((ranking.actions >= 1) & (ranking.actions <= 3)).all()):
        raise ValueError("whole branch supports only motion action sequences")
    previous = torch.cat((current_grid[:, None, None].expand(-1, modes, 1, -1, -1),
                          f[:, :, :-1]), dim=2).float()
    request = (q.float() - previous).abs().mean((-1, -2))
    residual = (q.float() - f.float()).abs().mean((-1, -2))
    inconclusive = request <= epsilon
    ratio = residual / request.clamp_min(epsilon)
    ratio = torch.where(inconclusive, ratio.clamp_min(1.0), ratio)
    worst = ratio.amax(dim=2)
    initial = (current_grid.float() - goal_grid.float()).abs().mean((-1, -2))
    terminal = (f[:, :, -1].float() - goal_grid[:, None].float()).abs().mean((-1, -2))
    normalized = terminal / initial[:, None].clamp_min(epsilon)
    score = normalized + risk_weight * worst
    if not all(bool(torch.isfinite(value).all()) for value in
               (request, residual, initial, terminal, ratio, worst, normalized, score)):
        raise FloatingPointError("nonfinite whole branch score")
    primary = torch.argsort(score.gather(1, mode_order), dim=1, stable=True)
    order = mode_order.gather(1, primary)
    ordered_modes = ranking.mode_indices.gather(1, order)
    rows = torch.arange(batch, device=q.device)
    winner_local = order[:, 0]
    selected = replace(ranking,
        ordered_indices=torch.stack((ordered_modes, torch.full_like(ordered_modes, horizon - 1)), -1),
        winner_k=ordered_modes[:, 0],
        winner_h=torch.full((batch,), horizon - 1, dtype=torch.long, device=q.device),
        first_action=ranking.actions[rows, winner_local, 0])
    return selected, dict(request_distance=request, relative_residual=ratio,
        inconclusive=inconclusive, worst_relative_residual=worst,
        terminal_normalized=normalized, branch_score=score,
        ordered_mode_indices=ordered_modes, goal_normalization_inconclusive=initial <= epsilon)
