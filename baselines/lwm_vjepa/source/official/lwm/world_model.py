"""Latent world model for goal-conditioned navigation.

Given the current observation image, the goal observation image, and a set of
candidate action trajectories, it predicts a scalar ``distance-to-goal`` for
every step of every candidate. The best candidate/step is the one minimizing
that distance.

Inputs are 224x224 RGB images normalized with ImageNet mean/std, and action
trajectories expressed as ``(x, y, yaw)`` in *normalized* units (real units
multiplied by 0.1).
"""

import torch
import torch.nn as nn

from .croco import CroCoNet
from .utils import MLP, SinusoidalPositionalEncoding

# Architecture of the vision backbone (CroCo V2 ViT-Base BaseDecoder); these are
# the values stored under the ``croco_kwargs`` key of the released backbone
# checkpoint. Encoder hyper-parameters are the CroCoNet defaults
# (enc_embed_dim=768, enc_depth=12, enc_num_heads=12).
BACKBONE_KWARGS = {
    "dec_embed_dim": 768,
    "dec_num_heads": 12,
    "dec_depth": 12,
    "pos_embed": "RoPE100",
}


class LatentWorldModel(nn.Module):
    def __init__(self, output_dim=384, backbone_kwargs=None):
        super().__init__()
        self.croco = CroCoNet(**(backbone_kwargs or BACKBONE_KWARGS))

        self.dim = 384
        self.num_a = 63  # number of steps in a trajectory
        self.Np = 196    # number of image patches (224 / 16) ** 2

        # The CroCo encoder is frozen; only the projector + head are trained.
        for param in self.croco.enc_blocks.parameters():
            param.requires_grad = False

        self.cro_proj = nn.Linear(768, output_dim, bias=True)
        self.acton_encoder = MLP(3, 512, output_dim)

        decoder_layer1 = nn.TransformerDecoderLayer(d_model=output_dim, nhead=6, batch_first=True)
        self.decoder1 = nn.TransformerDecoder(decoder_layer1, num_layers=6)
        self.output_sim = nn.Linear(output_dim, 1)
        self.position = SinusoidalPositionalEncoding(d_model=self.dim, max_len=self.num_a)

    def encode_images(self, img1, img2):
        """Fuse two images through the CroCo encoder-decoder.

        Returns ``(B, Np, 768)`` cross-view features.
        """
        feat1, pos1, _ = self.croco._encode_image(img1, do_mask=False)
        feat2, pos2, _ = self.croco._encode_image(img2, do_mask=False)
        decfeat = self.croco._decoder(feat1, pos1, None, feat2, pos2)
        out = self.croco.prediction_head(decfeat)
        return out

    @torch.no_grad()
    def score(self, img_now, img_obj, actions_kmeans):
        """Score every step of every candidate trajectory.

        Args:
            img_now: current observation, ``(B, 3, 224, 224)``.
            img_obj: goal observation, ``(B, 3, 224, 224)``.
            actions_kmeans: candidate trajectories ``(K, 63, 3)`` in normalized units.

        Returns:
            pred_sim: ``(B, K, 63)`` distance-to-goal for each (candidate, step).
        """
        n_kmeans = actions_kmeans.shape[0]
        num_a = self.num_a
        bsz = img_now.shape[0]

        fusion_feature = self.encode_images(img_now, img_obj)          # (B, 196, 768)
        fusion_feature = self.cro_proj(fusion_feature)                # (B, 196, 384)
        fusion_feature = torch.repeat_interleave(fusion_feature, n_kmeans, dim=0)  # (B*K, 196, 384)

        action_feats = self.acton_encoder(actions_kmeans)             # (K, 63, 384)
        action_feats = action_feats.repeat(bsz, 1, 1)                 # (B*K, 63, 384)
        action_feats = self.position(action_feats)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(num_a)
        pred_feats = self.decoder1(
            tgt=action_feats,
            memory=fusion_feature,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
        )
        pred_sim = self.output_sim(pred_feats).squeeze().view(bsz, n_kmeans, num_a)
        return pred_sim

    @torch.no_grad()
    def eval_wm(self, img_now, img_obj, actions_kmeans):
        """Pick the best candidate trajectory and end point.

        Returns:
            (kmeans_idx, point_idx): ``(B,)`` tensors selecting, for each sample,
            the candidate and the step minimizing the predicted distance.
        """
        num_a = self.num_a
        pred_sim = self.score(img_now, img_obj, actions_kmeans)       # (B, K, 63)
        bsz = pred_sim.shape[0]

        pred_sim = pred_sim.view(bsz, -1)
        best_idx = pred_sim.argmin(dim=1)                             # (B,)
        kmeans_idx = best_idx // num_a
        point_idx = best_idx % num_a
        return kmeans_idx, point_idx

    @torch.no_grad()
    def eval_wm_topk(self, img_now, img_obj, actions_kmeans, k=5):
        """Top-k variant of :meth:`eval_wm` (returns flattened indices)."""
        num_a = self.num_a
        pred_sim = self.score(img_now, img_obj, actions_kmeans)       # (B, K, 63)
        bsz = pred_sim.shape[0]

        value, best_idx = torch.topk(-pred_sim.view(bsz, -1), k=k, dim=-1)
        kmeans_idx = best_idx // num_a
        point_idx = best_idx % num_a
        return kmeans_idx.view(-1), point_idx.view(-1)

    @torch.no_grad()
    def get_reward(self, img_now, img_obj, actions, lengths):
        """Score a batch of full-length candidate trajectories.

        Args:
            actions: ``(B, S, 63, 3)`` candidate trajectories in normalized units.
            lengths: ``(B*S,)`` number of valid steps per candidate.

        Returns:
            reward_mat: ``(B, S)`` reward = negative predicted distance.
        """
        bsz, num_sample, _, _ = actions.shape
        num_a = self.num_a

        fusion_feature = self.encode_images(img_now, img_obj)          # (B, 196, 768)
        fusion_feature = self.cro_proj(fusion_feature)                # (B, 196, 384)
        fusion_feature = fusion_feature.repeat_interleave(num_sample, dim=0)  # (B*S, 196, 384)

        actions = actions.view(bsz * num_sample, num_a, 3)
        action_feats = self.acton_encoder(actions)
        action_feats = self.position(action_feats)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(num_a)
        pred_feats = self.decoder1(
            tgt=action_feats,
            memory=fusion_feature,
            tgt_mask=tgt_mask,
            tgt_is_causal=True,
        )
        pred_dis = self.output_sim(pred_feats).squeeze().view(bsz * num_sample, num_a)

        idx = torch.clip(lengths - 1, 0)
        pred_dis = pred_dis[torch.arange(bsz * num_sample), idx]      # (B*S,)
        reward_mat = -1 * pred_dis.view(bsz, num_sample)
        return reward_mat
