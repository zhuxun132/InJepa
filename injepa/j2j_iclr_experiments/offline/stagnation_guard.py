"""Opt-in, inference-only factual stagnation veto; no collision oracle or state."""
from dataclasses import replace
import math
import torch
from j2j.adapter import ActionId, Raw4Adapter


def validate_stagnation_options(options):
    if not isinstance(options, dict) or set(options) != {'frame_count', 'max_pair_mae', 'empty_policy'}:
        raise ValueError('stagnation guard requires frame_count/max_pair_mae/empty_policy')
    count, threshold = options['frame_count'], options['max_pair_mae']
    if type(count) is not int or count < 2:
        raise ValueError('stagnation frame_count must be an integer >=2')
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold < 0):
        raise ValueError('stagnation max_pair_mae must be finite and nonnegative')
    if options['empty_policy'] != 'f_turns':
        raise ValueError('stagnation empty_policy must be f_turns')
    return dict(frame_count=count, max_pair_mae=float(threshold), empty_policy='f_turns')


@torch.no_grad()
def apply_stagnation_guard(ranking, scores, *, factual_grids, factual_valid,
        outgoing_raw4, goal_grid, forward_step, options):
    """One online decision; returns original/filtered ranking, or None for fallback.

    Detection uses only the most recent contiguous factual grids. All primary
    branches already received unchanged Q/G/F and J=C+lambda*maxR. The veto is
    applied after scoring, to the first primitive only. Fallback has no Q/R:
    two one-step turn actions are scored by F endpoint-goal MAE in one batch.
    """
    options = validate_stagnation_options(options)
    if factual_grids.ndim != 3 or min(factual_grids.shape) < 1:
        raise ValueError('factual grids must be nonempty [history,spatial,width]')
    length, spatial, width = factual_grids.shape
    if (factual_valid.shape != (1, length) or factual_valid.dtype != torch.bool
            or outgoing_raw4.shape != (length, 4) or goal_grid.shape != (spatial, width)):
        raise ValueError('stagnation factual history/action/mask/goal axes disagree')
    if len({v.device for v in (factual_grids, factual_valid, outgoing_raw4, goal_grid)}) != 1:
        raise ValueError('stagnation tensors must share a device')
    if not all(bool(torch.isfinite(v).all()) for v in (factual_grids, outgoing_raw4, goal_grid)):
        raise FloatingPointError('nonfinite stagnation factual input')
    if ranking.actions.ndim != 3 or ranking.actions.shape[0] != 1:
        raise ValueError('stagnation guard accepts one online decision')
    if scores['branch_score'].shape != ranking.actions.shape[:2]:
        raise ValueError('stagnation score/candidate axes disagree')
    count = options['frame_count']
    ready = length >= count and bool(factual_valid[0, -count:].all())
    changes = ((factual_grids[-count+1:].float() - factual_grids[-count:-1].float())
               .abs().mean((-1,-2))) if ready else None
    triggered = ready and bool((changes <= options['max_pair_mae']).all())
    info = dict(options, metric='native_grid_mean_absolute_change', ready=ready,
        pair_mae=[] if changes is None else changes.cpu().tolist(), triggered=triggered,
        selection_source='original_candidates', selected_action=int(ranking.first_action[0]),
        pre_guard_winner_k=int(ranking.winner_k[0]), extra_f_calls=0, extra_f_rows=0)
    allowed = torch.ones_like(ranking.actions[0,:,0], dtype=torch.bool)
    if triggered:
        allowed = ranking.actions[0,:,0] != int(ActionId.FWD)
    info['eligible_mask'] = allowed.cpu().tolist()
    if not triggered:
        return ranking, info
    if bool(allowed.any()):
        # Keep the existing stable whole-branch order; never compare a later
        # primitive against another branch's first primitive.
        ordered = ranking.ordered_indices[0,:,0]
        local = {int(mode):i for i,mode in enumerate(ranking.mode_indices[0])}
        kept = [int(mode) for mode in ordered if bool(allowed[local[int(mode)]])]
        banned = [int(mode) for mode in ordered if not bool(allowed[local[int(mode)]])]
        winner = local[kept[0]]
        new_order = torch.tensor(kept+banned,device=ordered.device,dtype=ordered.dtype)[None]
        horizon = ranking.actions.shape[-1]
        result = replace(ranking,
            ordered_indices=torch.stack((new_order,torch.full_like(new_order,horizon-1)),-1),
            winner_k=ranking.mode_indices[:,winner],
            winner_h=torch.full_like(ranking.winner_h,horizon-1),
            first_action=ranking.actions[:,winner,0])
        info['selected_action'] = int(result.first_action[0])
        return result, info
    if not callable(forward_step):
        raise ValueError('no allowed candidate: action-conditioned F is required')
    turn_ids = (int(ActionId.LEFT), int(ActionId.RIGHT))
    actions = torch.stack([Raw4Adapter.encode(ActionId(a)) for a in turn_ids]).to(outgoing_raw4)
    histories = factual_grids[None].expand(len(turn_ids),-1,-1,-1).clone()
    conditioned = outgoing_raw4[None].expand(len(turn_ids),-1,-1).clone()
    conditioned[:,-1] = actions
    valid = factual_valid.expand(len(turn_ids),-1).clone()
    predicted = forward_step(histories,conditioned,valid)
    if not isinstance(predicted,torch.Tensor) or predicted.shape != (len(turn_ids),spatial,width):
        raise ValueError('fallback F must return [turns,spatial,width]')
    if predicted.device != goal_grid.device or not bool(torch.isfinite(predicted).all()):
        raise FloatingPointError('fallback F prediction is nonfinite or on wrong device')
    cost = (predicted.float()-goal_grid.float()[None]).abs().mean((-1,-2))
    if not bool(torch.isfinite(cost).all()):
        raise FloatingPointError('fallback turn costs must be finite')
    winner = int(cost.argmin())  # Deterministic tie: LEFT precedes RIGHT.
    info.update(selection_source='fallback_turns', selected_action=turn_ids[winner],
        extra_f_calls=1, extra_f_rows=len(turn_ids),
        fallback=dict(actions=list(turn_ids), horizon=1, goal_mae=cost.cpu().tolist(),
                      selected_index=winner, criterion='minimum_F_goal_MAE', q_used=False, r_used=False))
    return None, info
