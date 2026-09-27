"""Auto-regressive trajectory (waypoint) decoder used by the policy."""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_


class TrajectoryDecoder(nn.Module):
    def __init__(self, PAD_token, feats_seq_len, max_len, dim=384, decoder_layer=6):
        super().__init__()
        self.PAD_token = PAD_token
        self.num_waypoints = max_len
        self.embedding = nn.Embedding(64 + 3, dim)
        self.pos_drop = nn.Dropout(0.05)

        item_cnt = max_len - 1

        self.pos_embed_waypoint = nn.Parameter(torch.randn(1, item_cnt, dim) * 0.02)
        self.pos_embed_img = nn.Parameter(torch.randn(1, feats_seq_len, dim) * 0.02)

        tf_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=4)
        self.tf_decoder = nn.TransformerDecoder(tf_layer, decoder_layer)
        self.output = nn.Linear(dim, 64 + 3)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if "pos_embed_waypoint" in name or "pos_embed_img" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed_waypoint, std=0.02)
        trunc_normal_(self.pos_embed_img, std=0.02)

    def create_mask(self, tgt):
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=tgt.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float("-inf")).masked_fill(tgt_mask == 1, float(0.0))
        tgt_padding_mask = tgt == self.PAD_token
        return tgt_mask, tgt_padding_mask

    def decoder(self, feats, tgt_embedding, tgt_mask, tgt_padding_mask):
        feats = feats.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        pred_traj_points = self.tf_decoder(
            tgt=tgt_embedding,
            memory=feats,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        pred_traj_points = pred_traj_points.transpose(0, 1)
        return pred_traj_points

    def forward(self, feats, tgt):
        tgt = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed_waypoint)
        feats = self.pos_drop(feats + self.pos_embed_img)
        pred_points = self.decoder(feats, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_points = self.output(pred_points)  # bsz, num_waypoints-1, 64+3
        return pred_points

    def predict(self, feats, tgt, ret_all=False):
        length = tgt.size(1)
        padding_num = self.num_waypoints - 1 - length
        offset = 1
        if padding_num > 0:
            padding = torch.ones(tgt.size(0), padding_num).fill_(self.PAD_token).long().to(tgt.device)
            tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)
        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + self.pos_embed_waypoint
        feats = feats + self.pos_embed_img

        pred_traj_points = self.decoder(feats, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_traj_points = self.output(pred_traj_points)[:, length - offset, :]
        if ret_all:
            return pred_traj_points
        else:
            pred_traj_points = torch.softmax(pred_traj_points, dim=-1)
            pred_traj_points = pred_traj_points.argmax(dim=-1).view(-1, 1)
            return pred_traj_points
