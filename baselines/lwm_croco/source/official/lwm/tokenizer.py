"""Discrete action tokenizer.

Quantizes per-step action deltas (dx, dy) against a fixed codebook of K cluster
centers, then decodes a token sequence back into an absolute (x, y, yaw)
trajectory. The special tokens are:

    BOS = K, EOS = K + 1, PAD = K + 2
"""

import json

import numpy as np
import torch


class ActionTokenizer:
    def __init__(self, center_path: str, max_len: int = 63 + 2):
        with open(center_path, "r", encoding="utf-8") as f:
            data_raw = json.load(f)
        cluster_centers = np.array(data_raw)
        self.centers = cluster_centers
        self.centers_tensor = torch.tensor(cluster_centers, dtype=torch.float32)
        self.K = cluster_centers.shape[0]

        # special tokens
        self.BOS = self.K
        self.EOS = self.K + 1
        self.PAD = self.K + 2

        self.vocab_size = self.K + 3
        self.max_len = max_len

    # ------------------------------------------------------------
    # encode (torch, used to turn a GT trajectory into tokens)
    # ------------------------------------------------------------
    def actions_to_deltas_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: (T, 2) -> deltas: (T, 2), prepended with a zero step."""
        zero = torch.zeros((1, actions.shape[1]), dtype=actions.dtype, device=actions.device)
        padded = torch.cat([zero, actions], dim=0)
        deltas = padded[1:] - padded[:-1]
        return deltas

    def quantize_tensor(self, deltas):
        centers = self.centers_tensor.to(deltas.device)
        dist = torch.norm(deltas[:, None, :] - centers[None, :, :], dim=-1)
        topk = 5
        topk_dist, topk_idx = torch.topk(dist, k=topk, dim=1, largest=False)
        probs = torch.softmax(-topk_dist * 0.1, dim=1)
        rand_choice = torch.multinomial(probs, num_samples=1).squeeze(1)
        result = topk_idx[torch.arange(deltas.shape[0], device=deltas.device), rand_choice]
        return result

    def encode_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        deltas = self.actions_to_deltas_tensor(actions)
        tokens = self.quantize_tensor(deltas)
        tokens = torch.cat(
            [
                torch.tensor([self.BOS], device=actions.device),
                tokens,
                torch.tensor([self.EOS], device=actions.device),
            ]
        )
        if tokens.shape[0] < self.max_len:
            pad = torch.full(
                (self.max_len - tokens.shape[0],),
                self.PAD,
                dtype=torch.long,
                device=actions.device,
            )
            tokens = torch.cat([tokens, pad])
        else:
            tokens = tokens[: self.max_len]
        return tokens.long()

    # ------------------------------------------------------------
    # decode
    # ------------------------------------------------------------
    def decode(self, token_ids: torch.Tensor, return_deltas: bool = False, smooth_alpha=0.4):
        """Decode a single token sequence ``(T,)`` into an absolute trajectory.

        Returns:
            trajectory: (N + 1, 3) torch.FloatTensor, or (trajectory, deltas)
        """
        device = token_ids.device
        start_pos = torch.zeros(3, device=device)

        bos_mask = token_ids == self.BOS
        eos_mask = token_ids == self.EOS

        if bos_mask.any():
            bos_idx = torch.nonzero(bos_mask, as_tuple=False)[0, 0] + 1
        else:
            bos_idx = 0

        eos_after_bos = torch.nonzero(
            eos_mask & (torch.arange(len(token_ids), device=device) >= bos_idx),
            as_tuple=False,
        )
        if len(eos_after_bos) > 0:
            eos_idx = eos_after_bos[0, 0]
        else:
            eos_idx = len(token_ids) - 1

        valid_tokens = token_ids[bos_idx:eos_idx]
        valid_tokens = valid_tokens[valid_tokens != self.PAD]

        if valid_tokens.numel() == 0:
            traj = start_pos.unsqueeze(0)
            return (traj, None) if return_deltas else traj

        centers_tensor = self.centers_tensor.to(device)
        deltas = centers_tensor[valid_tokens]  # (N, 2)
        yaw = torch.arctan2(deltas[:, 1], deltas[:, 0])

        # delta EMA smoothing
        if deltas.shape[0] > 1:
            smooth_deltas = torch.zeros_like(deltas)
            smooth_deltas[0] = deltas[0]
            smooth_yaw = torch.zeros_like(yaw)
            smooth_yaw[0] = yaw[0]
            for t in range(1, deltas.shape[0]):
                smooth_deltas[t] = smooth_alpha * smooth_deltas[t - 1] + (1 - smooth_alpha) * deltas[t]
                smooth_yaw[t] = smooth_alpha * smooth_yaw[t - 1] + (1 - smooth_alpha) * yaw[t]
            deltas = smooth_deltas
            yaw = smooth_yaw

        trajectory = torch.zeros((deltas.shape[0] + 1, 3), device=device, dtype=deltas.dtype)
        trajectory[0] = start_pos
        trajectory[1:, :2] = torch.cumsum(deltas, dim=0)
        trajectory[1:, 2] = yaw

        if return_deltas:
            return trajectory, deltas
        return trajectory

    def batch_decode(self, token_ids_batch: torch.Tensor):
        """Decode a batch ``(B, T)``.

        Returns:
            action_batch: (B, max_len - 2, 3), zero-padded absolute trajectories
            lengths: (B,) number of valid steps per sample
        """
        device = token_ids_batch.device
        B = token_ids_batch.size(0)

        traj_list = []
        lengths = []
        for i in range(B):
            traj, deltas = self.decode(token_ids_batch[i], return_deltas=True)
            traj_list.append(traj)
            if deltas is not None:
                lengths.append(deltas.size(0))
            else:
                lengths.append(0)
        lengths = torch.tensor(lengths, device=device, dtype=torch.long)

        action_batch = torch.zeros((B, self.max_len - 2, 3), device=device, dtype=torch.float32)
        for i, traj_now in enumerate(traj_list):
            if traj_now is not None:
                action_batch[i, : lengths[i]] = traj_now[1:]

        return action_batch, lengths
