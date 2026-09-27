# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed
from RAE.src.stage2.models.model_utils import RMSNorm, NormAttention, SwiGLUFFN, VisionRotaryEmbeddingFast


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    B, Lx, D = x.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    return x * (1 + scale) + shift


def DDTGate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    B, Lx, D = x.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        gate = gate.repeat_interleave(repeat, dim=1)
    return x * gate


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class GaussianFourierEmbedding(nn.Module):
    """
    Gaussian Fourier Embedding for timesteps, suitable for inputs in [0, 1] or [-0.5, 0.5].
    """
    def __init__(self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale
        self.W = nn.Parameter(torch.normal(0, self.scale, (embedding_size,)), requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        with torch.no_grad():
            W = self.W  # stop gradient manually
        # t: (B, 1) or (B,) -> (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_proj = t * W[None, :] * 2 * torch.pi
        t_embed = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class ActionEmbedder(nn.Module):
    """
    Unified ActionEmbedder using GaussianFourierEmbedding for all continuous components.
    """
    def __init__(self, hidden_size, embedding_size=256):
        super().__init__()
        hsize = hidden_size // 3

        self.x_emb = GaussianFourierEmbedding(hsize, embedding_size, scale=1.0)
        self.y_emb = GaussianFourierEmbedding(hsize, embedding_size, scale=1.0)

        self.angle_emb = GaussianFourierEmbedding(hidden_size - 2 * hsize, embedding_size, scale=1.0)

    def forward(self, xya):
        # xya: [Batch, 3] -> (x, y, angle)
        x = self.x_emb(xya[..., 0])
        y = self.y_emb(xya[..., 1])
        a = self.angle_emb(xya[..., 2])
        return torch.cat([x, y, a], dim=-1)


#################################################################################
#                                 Core CDiT Model                               #
#################################################################################

class CDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_low_rank_adaln=False, use_qknorm: bool = False, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=bool(use_qknorm),
            use_rmsnorm=True,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.norm_cond = RMSNorm(hidden_size)
        self.cttn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, add_bias_kv=True, bias=True, batch_first=True, **block_kwargs)
        if use_low_rank_adaln:
            rank_dim = min(hidden_size // 3, 512)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, rank_dim, bias=True),
                nn.SiLU(),
                nn.Linear(rank_dim, 11 * hidden_size, bias=True),
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 11 * hidden_size, bias=True),
            )
        self.norm3 = RMSNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        self.feat_rope = None

    def forward(self, x, c, x_cond, rope=None):
        shift_msa, scale_msa, gate_msa, shift_ca_xcond, scale_ca_xcond, shift_ca_x, scale_ca_x, gate_ca_x, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(11, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(attn_in, rope=rope)

        x_cond_norm = modulate(self.norm_cond(x_cond), shift_ca_xcond, scale_ca_xcond)
        x = x + gate_ca_x.unsqueeze(1) * self.cttn(
            query=modulate(self.norm2(x), shift_ca_x, scale_ca_x),
            key=x_cond_norm,
            value=x_cond_norm,
            need_weights=False
        )[0]

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class DDTHeadBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_qknorm: bool = False, **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=bool(use_qknorm),
            use_rmsnorm=True,
            **block_kwargs,
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope=None) -> torch.Tensor:
        if c.dim() == 2:
            c = c.unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x), shift_msa, scale_msa), rope=rope), gate_msa)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        if c.dim() == 2:
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class CDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        context_size=2,
        patch_size=1,
        in_channels=768,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        head_width=None,
        head_depth=2,
        head_num_heads=16,
        use_low_rank_adaln_head=False,
        use_qknorm: bool = False,
    ):
        super().__init__()
        self.context_size = context_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels

        # DiT-DH head width: default to token channels (in_channels) to match RAE latent
        self.head_width = head_width if head_width is not None else in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.head_depth = head_depth
        self.head_num_heads = head_num_heads
        self.use_low_rank_adaln_head = bool(use_low_rank_adaln_head)

        self._init_core_components(
            input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio,
            use_qknorm=bool(use_qknorm)
        )

        self.initialize_weights()

    def _init_core_components(self, input_size, patch_size, in_channels, hidden_size, depth, num_heads, mlp_ratio, use_qknorm: bool = False):

        # -------------------------
        # Encoder (base DiT): token dim = hidden_size
        # -------------------------
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        # Use GaussianFourierEmbedding for flow matching time t in [0, 1]
        self.t_embedder = GaussianFourierEmbedding(hidden_size)
        self.y_embedder = ActionEmbedder(hidden_size)
        # Use GaussianFourierEmbedding for relative time rel_t in [-0.5, 0.5]
        self.time_embedder = GaussianFourierEmbedding(hidden_size)

        self.dyn_fuse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.dyn_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(self.context_size + 1, num_patches, hidden_size),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            CDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_low_rank_adaln=False, use_qknorm=use_qknorm)
            for _ in range(depth)
        ])

        half_head = hidden_size // num_heads // 2
        grid = int(self.x_embedder.num_patches ** 0.5)
        self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head, pt_seq_len=grid)

        # -------------------------
        # Head (DDT / DiT-DH style):
        #   - NEW: head re-embeds raw x_t into head_width (query stream)
        #   - z_t from encoder (token-wise) projected to head_width as conditioning (key/value stream)
        # -------------------------
        self.x_embedder_head = PatchEmbed(input_size, patch_size, in_channels, self.head_width, bias=True)

        # Project encoder token features to head_width (token-wise condition)
        self.head_projector = nn.Linear(hidden_size, self.head_width, bias=True)

        # Keep these projections (not strictly needed, but keep for compatibility / cleanliness)
        self.head_cond_projector = nn.Linear(hidden_size, self.head_width, bias=True)
        self.final_cond_proj = nn.Linear(hidden_size, self.head_width, bias=True)

        self.head_blocks = nn.ModuleList([
            DDTHeadBlock(self.head_width, self.head_num_heads, mlp_ratio=mlp_ratio, use_qknorm=use_qknorm)
            for _ in range(self.head_depth)
        ])

        half_head2 = self.head_width // self.head_num_heads // 2
        grid2 = int(self.x_embedder.num_patches ** 0.5)
        self.head_feat_rope = VisionRotaryEmbeddingFast(dim=half_head2, pt_seq_len=grid2)

        self.final_layer = DDTFinalLayer(self.head_width, patch_size, self.out_channels)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        grid_size = int(self.x_embedder.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
        pos_embed = pos_embed.repeat(self.context_size + 1, 1, 1)
        self.pos_embed.data.copy_(pos_embed)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize head patch_embed too:
        w2 = self.x_embedder_head.proj.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder_head.proj.bias, 0)

        # Initialize action embedding:
        nn.init.normal_(self.y_embedder.x_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.x_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.y_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_emb.mlp[2].weight, std=0.02)

        nn.init.normal_(self.y_embedder.angle_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.y_embedder.angle_emb.mlp[2].weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        for block in self.head_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, num_patches, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        num_patches = x.shape[1]
        h = w = int(num_patches ** 0.5)
        assert h * w == num_patches, f"num_patches {num_patches} is not a square"
        # derive patch size dynamically from the last dimension
        patch_area_times_c = x.shape[2]
        if patch_area_times_c % c != 0:
            # handle swapped token layout: (N, Ctok, num_patches)
            if x.shape[2] == num_patches and x.shape[1] != num_patches:
                x = x.transpose(1, 2)
                patch_area_times_c = x.shape[2]
            # if still not divisible, raise
        # decide effective channel count
        if patch_area_times_c % c == 0:
            c_eff = c
        elif patch_area_times_c % (2 * c) == 0:
            c_eff = 2 * c
        else:
            raise AssertionError(f"channel mismatch: {patch_area_times_c} not divisible by {c} or {2*c}")
        p = int(((patch_area_times_c // c_eff)) ** 0.5)
        assert p * p * c_eff == patch_area_times_c, f"cannot factor last dim {patch_area_times_c} into p^2*c_eff with c_eff={c_eff}"

        x = x.reshape(x.shape[0], h, w, p, p, c_eff)
        x = x.permute(0, 5, 3, 4, 1, 2)  # (N, C_eff, p, p, H, W)
        imgs = x.reshape(x.shape[0], c_eff, h * p, w * p)
        if c_eff == 2 * c:
            imgs_mu, imgs_sigma = imgs.chunk(2, dim=1)
            return imgs_mu
        return imgs

    def forward(self, x, t, y, x_cond, rel_t):
        # Keep raw x_t for DDT head (RAE-style: head re-embeds raw x_t)
        x_raw = x

        # Prepare encoder token inputs and global condition
        x_tok, x_cond_tok, c, t_emb = self._prepare_inputs(x, t, y, x_cond, rel_t)

        # -------------------------
        # Encoder (base DiT): identical to your original design
        # -------------------------
        z = self._process_blocks(x_tok, c, x_cond_tok)

        # RAE-style: fuse time embedding into token features before head (token-wise)
        # z_t := SiLU(z + t_emb)
        z = F.silu(z + t_emb[:, None, :])

        # Project token-wise z to head width as conditioning stream
        z_head = self.head_projector(z)  # (B, L, head_width)

        # -------------------------
        # DDT Head (DiT-DH style):
        #   Query stream: re-embed raw x_t -> x_head
        #   Key/Value stream: z_head (token-wise condition)
        # -------------------------
        x_head = self.x_embedder_head(x_raw)  # (B, L, head_width)  (no APE in head; RoPE inside attention)

        for blk in self.head_blocks:
            x_head = blk(x_head, z_head, rope=self.head_feat_rope)

        x_out = self.final_layer(x_head, z_head)

        num_patches = self.x_embedder.num_patches
        N = x_out.shape[0]
        if x_out.dim() != 3:
            x_out = x_out.reshape(N, num_patches, -1)
        x_out = self.unpatchify(x_out)
        return x_out

    def _prepare_inputs(self, x, t, y, x_cond, rel_t):
        x = self.x_embedder(x) + self.pos_embed[self.context_size:]
        x_cond = self.x_embedder(x_cond.flatten(0, 1)).unflatten(0, (x_cond.shape[0], x_cond.shape[1])) + self.pos_embed[:self.context_size]
        x_cond = x_cond.flatten(1, 2)

        t_emb = self.t_embedder(t[..., None])
        y_emb = self.y_embedder(y)
        span_emb = self.time_embedder(rel_t[..., None])

        c_dyn = self.dyn_fuse(torch.cat([y_emb, span_emb], dim=-1))
        gate = torch.sigmoid(self.dyn_gate(t_emb))
        c = t_emb + gate * c_dyn

        return x, x_cond, c, t_emb

    def _process_blocks(self, x, c, x_cond):
        for block in self.blocks:
            x = block(x, c, x_cond, rope=self.feat_rope)
        return x


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   CDiT Configs                                #
#################################################################################

def CDiT_XL_2(**kwargs):
    return CDiT(depth=28, hidden_size=1152, patch_size=1, num_heads=16, **kwargs)

def CDiT_L_2(**kwargs):
    return CDiT(depth=24, hidden_size=1024, patch_size=1, num_heads=16, **kwargs)

def CDiT_B_2(**kwargs):
    return CDiT(depth=12, hidden_size=768, patch_size=1, num_heads=12, **kwargs)

def CDiT_S_2(**kwargs):
    return CDiT(depth=12, hidden_size=384, patch_size=1, num_heads=6, **kwargs)


CDiT_models = {
    'CDiT-XL/2': CDiT_XL_2,
    'CDiT-L/2':  CDiT_L_2,
    'CDiT-B/2':  CDiT_B_2,
    'CDiT-S/2':  CDiT_S_2
}
