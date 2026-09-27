"""Frozen native V-JEPA image features feeding the unchanged CroCo fusion code."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from lwm.croco.croco import CroCoNet
from lwm.croco.blocks import PositionGetter
from . import vendor_vjepa as official


def vision_identity(vision):
    """Scientific fields only; moving identical assets cannot change the method."""
    fixed = dict(kind='vjepa21_vitb384_native',
                 source_commit=official.OFFICIAL_SOURCE_COMMIT,
                 source_tree=official.OFFICIAL_SOURCE_TREE,
                 preprocessing='official_vjepa384_shortside_centercrop_imagenet',
                 image_shape=[3, 1, 384, 384], grid_shape=[576, 768], frozen=True)
    allowed = set(fixed) | {'source_root', 'checkpoint', 'checkpoint_sha256', 'checkpoint_bytes'}
    if set(vision) - allowed:
        raise ValueError('unknown V-JEPA configuration fields')
    for key, value in fixed.items():
        if key in vision and (vision[key] != value or type(vision[key]) is not type(value)):
            raise ValueError(f'unsupported V-JEPA contract: {key}')
    digest = vision['checkpoint_sha256']
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('invalid V-JEPA SHA-256')
    if vision['checkpoint_bytes'] != official.OFFICIAL_CHECKPOINT_BYTES:
        raise ValueError('unexpected V-JEPA checkpoint size')
    return dict(fixed, checkpoint_sha256=digest, checkpoint_bytes=vision['checkpoint_bytes'])


def verify_vision_assets(vision):
    identity = vision_identity(vision)
    official._validate_source(Path(vision['source_root']))
    handle, digest = official._hash_open_checkpoint(
        Path(vision['checkpoint']), expected_sha256=identity['checkpoint_sha256'],
        expected_bytes=identity['checkpoint_bytes'])
    handle.close()
    return digest


def load_frozen_encoder(vision):
    identity = vision_identity(vision)
    loaded = official.load_frozen_vjepa2(
        source_root=vision['source_root'], checkpoint=vision['checkpoint'],
        expected_sha256=identity['checkpoint_sha256'], expected_bytes=identity['checkpoint_bytes'])
    # The official predictor and the optional pooling helper are never used by LWM.
    return loaded.encoder


class VJEPAImageTransform:
    """Lazy official CPU transform; only a source path crosses worker spawn."""
    def __init__(self, source_root):
        self.source_root = str(Path(source_root).resolve())
        self._transform = None

    def __getstate__(self):
        return {'source_root': self.source_root, '_transform': None}

    def __call__(self, image):
        if self._transform is None:
            root = official._validate_source(Path(self.source_root))
            with official._official_modules(root) as (_, preprocessing):
                self._transform = preprocessing.vjepa2_preprocessor(crop_size=384)
        rgb = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.uint8, copy=True)).permute(2, 0, 1)
        with torch.autocast(device_type='cpu', enabled=False):
            sample = self._transform([rgb])[0]
        if sample.shape != (3, 1, 384, 384) or sample.dtype != torch.float32:
            raise ValueError('official image preprocessing ABI changed')
        return sample.contiguous()


class VJEPACroCo(CroCoNet):
    """Replace only E; inherit the actual upstream _decoder without rewriting it."""
    def __init__(self, original_croco, frozen_encoder):
        nn.Module.__init__(self)
        if original_croco.dec_pos_embed is not None or not original_croco.pos_embed.startswith('RoPE'):
            raise ValueError('native V-JEPA requires the original LWM RoPE fusion')
        for name in ('decoder_embed', 'dec_blocks', 'dec_norm', 'prediction_head', 'mask_token', 'rope'):
            setattr(self, name, getattr(original_croco, name))
        self.pos_embed = original_croco.pos_embed
        self.dec_pos_embed = None
        self.dec_embed_dim = original_croco.dec_embed_dim
        self.dec_depth = original_croco.dec_depth
        self.encoder = frozen_encoder.requires_grad_(False).eval()
        self._positions = {}
        self.train(original_croco.training)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        if do_mask or return_all_blocks:
            raise ValueError('LWM V-JEPA admits only unmasked final single-image features')
        if (not isinstance(image, torch.Tensor) or image.ndim != 5 or image.shape[0] < 1
                or tuple(image.shape[1:]) != (3, 1, 384, 384)):
            raise ValueError('expected independent [B,3,1,384,384] images')
        if image.dtype != torch.float32:
            raise TypeError('V-JEPA image input must be float32')
        self.encoder.eval()
        with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
            tokens = self.encoder(image)
        if not isinstance(tokens, torch.Tensor) or tokens.shape != (image.shape[0], 576, 768):
            raise ValueError('expected native [B,576,768] encoder output')
        if tokens.dtype != torch.float32 or tokens.device != image.device:
            raise TypeError('V-JEPA native output must preserve float32 and device')
        if not torch.isfinite(tokens).all():
            raise ValueError('non-finite V-JEPA features')
        # The original helper caches by grid only. Separate helpers per device
        # preserve its exact row-major convention across CPU/GPU model moves.
        getter = self._positions.setdefault(str(tokens.device), PositionGetter())
        positions = getter(tokens.shape[0], 24, 24, tokens.device)
        return tokens, positions, None


def install_frontend(model, encoder):
    if isinstance(model.croco, VJEPACroCo):
        raise ValueError('V-JEPA frontend is already installed')
    model.croco = VJEPACroCo(model.croco, encoder)
    if hasattr(model, 'Np'):
        model.Np = 576
    return model
