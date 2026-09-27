"""Batched candidate rollout through INTACT's public G/F seams."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from j2j.adapter import ActionId, Raw4Adapter


@dataclass(frozen=True)
class CandidateRollout:
    """Disposable predictions and prefix-consistency errors for each tape."""

    action_id: Tensor
    predicted: Tensor
    valid_prefix: Tensor
    path_error: Tensor
    endpoint_error: Tensor


@torch.inference_mode()
def rollout_candidates(
    intact_model,
    tape: Tensor,
    embedding_history: Tensor,
    raw4_history: Tensor,
    *,
    history_size: int,
) -> CandidateRollout:
    """Roll every candidate once, retaining each valid motion prefix."""

    batch_size, candidate_count, horizon, _, latent_dim = tape.shape
    owner_count = batch_size * candidate_count
    pooled_tape = tape.float().mean(dim=-2)
    targets = pooled_tape.reshape(owner_count, horizon, latent_dim)

    embeddings = (
        embedding_history.float()
        .unsqueeze(1)
        .expand(-1, candidate_count, -1, -1)
        .reshape(owner_count, embedding_history.size(1), latent_dim)
        .clone()
    )
    master = (
        raw4_history.float()
        .unsqueeze(1)
        .expand(-1, candidate_count, -1, -1)
        .reshape(owner_count, raw4_history.size(1), 4)
        .clone()
    )

    device = targets.device
    action_id = torch.full(
        (owner_count, horizon),
        -1,
        dtype=torch.int64,
        device=device,
    )
    predicted = torch.zeros(
        (owner_count, horizon, latent_dim),
        dtype=torch.float32,
        device=device,
    )
    valid_prefix = torch.zeros(
        (owner_count, horizon),
        dtype=torch.bool,
        device=device,
    )
    path_error = torch.full(
        (owner_count, horizon),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    endpoint_error = torch.full_like(path_error, float("inf"))
    cumulative_error = torch.zeros(
        owner_count,
        dtype=torch.float32,
        device=device,
    )

    live_owner = torch.arange(owner_count, device=device)
    for step in range(horizon):
        if live_owner.numel() == 0:
            break

        current = embeddings[:, -1]
        target = targets.index_select(0, live_owner)[:, step]
        logits = intact_model.action_logits(
            current,
            target - current,
            master[:, -1],
        )
        decoded = Raw4Adapter.decode_logits(logits)
        raw4 = torch.stack(
            [
                Raw4Adapter.encode(ActionId(int(value.item())))
                for value in decoded
            ],
            dim=0,
        ).to(device=logits.device, dtype=torch.float32)

        action_id[live_owner, step] = decoded.to(
            device=device,
            dtype=torch.int64,
        )
        motion_index = torch.nonzero(
            decoded != int(ActionId.STOP),
            as_tuple=False,
        ).flatten()
        if motion_index.numel() == 0:
            break

        motion_owner = live_owner.index_select(0, motion_index)
        motion_embeddings = embeddings.index_select(0, motion_index)
        motion_master = master.index_select(0, motion_index)
        motion_raw4 = raw4.index_select(0, motion_index)
        corrected_master = motion_master.clone()
        corrected_master[:, -1] = motion_raw4.to(corrected_master)

        embeddings, master = intact_model.rollout_one_step(
            motion_embeddings,
            motion_raw4.to(motion_embeddings),
            corrected_master,
            history_size,
        )
        next_prediction = embeddings[:, -1].float()
        motion_target = target.index_select(0, motion_index).float()
        error = (next_prediction - motion_target).abs().mean(dim=-1)
        cumulative_error[motion_owner] = (
            cumulative_error.index_select(0, motion_owner) + error
        )

        predicted[motion_owner, step] = next_prediction
        valid_prefix[motion_owner, step] = True
        endpoint_error[motion_owner, step] = error
        path_error[motion_owner, step] = (
            cumulative_error.index_select(0, motion_owner) / float(step + 1)
        )
        live_owner = motion_owner

    return CandidateRollout(
        action_id=action_id.reshape(batch_size, candidate_count, horizon),
        predicted=predicted.reshape(
            batch_size,
            candidate_count,
            horizon,
            latent_dim,
        ),
        valid_prefix=valid_prefix.reshape(
            batch_size,
            candidate_count,
            horizon,
        ),
        path_error=path_error.reshape(
            batch_size,
            candidate_count,
            horizon,
        ),
        endpoint_error=endpoint_error.reshape(
            batch_size,
            candidate_count,
            horizon,
        ),
    )
