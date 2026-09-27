"""Goal-conditioned navigation policy.

This is the model referred to as ``AR_plus`` in the training code. Given the
current observation image and the goal observation image, it auto-regressively
generates a sequence of action tokens that decode into a (x, y, yaw) trajectory.

Inputs are 224x224 RGB images normalized with ImageNet mean/std.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .croco import CroCoNet
from .trajectory_decoder import TrajectoryDecoder
from .world_model import BACKBONE_KWARGS


class ARPlusPolicy(nn.Module):
    def __init__(self, max_len=65, decoder_layer=6, feats_seq_len=196, dim=384,
                 PAD_token=66, backbone_kwargs=None):
        super().__init__()
        self.max_len = max_len
        self.PAD_token = PAD_token

        self.croco = CroCoNet(**(backbone_kwargs or BACKBONE_KWARGS))
        # The CroCo encoder is frozen; the decoder is fine-tuned with the policy.
        for param in self.croco.enc_blocks.parameters():
            param.requires_grad = False

        self.dim = dim
        self.num_a = 63
        self.cro_proj = nn.Linear(768, dim, bias=True)

        self.feats_seq_len = feats_seq_len
        self.waypoint_decoder = TrajectoryDecoder(
            PAD_token=66,
            feats_seq_len=self.feats_seq_len,
            decoder_layer=decoder_layer,
            max_len=max_len,
            dim=dim,
        )

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
    def predict(self, img_now, img_obj):
        """Greedy (argmax) auto-regressive decoding of a single trajectory.

        Returns a token sequence ``(B, max_len)``.
        """
        B = img_now.shape[0]
        # 64: BOS, 65: EOS, 66: PAD
        auto_token = torch.tensor([[64]], device=img_now.device).repeat(B, 1)

        fusion_feature = self.encode_images(img_now, img_obj)          # (B, 196, 768)
        fusion_feature = self.cro_proj(fusion_feature)                # (B, 196, 384)

        for _ in range(self.max_len - 1):
            pred_traj_point = self.waypoint_decoder.predict(fusion_feature, auto_token, ret_all=True)
            pred_traj_point[:, 64] = -999999  # BOS
            pred_traj_point[:, 66] = -999999  # PAD
            next_token = pred_traj_point.argmax(dim=-1).view(-1, 1)
            auto_token = torch.cat([auto_token, next_token], dim=1)
        return auto_token

    @torch.no_grad()
    def roll_out(self, img_now, img_obj, num_sample=5, temperature=1.0):
        """Sample ``num_sample`` trajectories per image via auto-regressive decoding.

        Returns token sequences ``(B, num_sample, max_len)``.
        """
        bsz = img_now.shape[0]
        device = img_now.device
        EOS = 65
        PAD = self.PAD_token
        BOS = 64

        fusion_feature = self.encode_images(img_now, img_obj)          # (B, 196, 768)
        fusion_feature = self.cro_proj(fusion_feature)                # (B, 196, 384)
        fusion_feature = torch.repeat_interleave(fusion_feature, num_sample, dim=0)

        B = bsz * num_sample

        auto_token = torch.full((B, 1), BOS, device=device, dtype=torch.long)
        alive = torch.ones(B, device=device, dtype=torch.bool)

        for _ in range(self.max_len - 1):
            if not alive.any():
                break

            logits = self.waypoint_decoder.predict(fusion_feature, auto_token, ret_all=True)  # (B, vocab)

            logits[:, BOS] = -1e9
            logits[:, PAD] = -1e9

            probs = F.softmax(logits / temperature, dim=-1)

            next_token = torch.full((B, 1), PAD, device=device, dtype=torch.long)
            sampled = torch.multinomial(probs[alive], num_samples=1)
            next_token[alive] = sampled

            auto_token = torch.cat([auto_token, next_token], dim=1)
            alive = alive & (next_token.squeeze(-1) != EOS)

        if auto_token.shape[1] < self.max_len:
            pad_len = self.max_len - auto_token.shape[1]
            pad = torch.full((B, pad_len), PAD, device=device, dtype=torch.long)
            auto_token = torch.cat([auto_token, pad], dim=1)

        auto_token = auto_token.view(bsz, num_sample, self.max_len)
        return auto_token


# Alias kept for compatibility with the original training code.
AR_plus = ARPlusPolicy
