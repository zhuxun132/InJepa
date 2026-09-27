"""Immutable feature and training contracts used by model identity validation."""
from __future__ import annotations
import copy
import hashlib
import json

FAMILY = 'croco_v2_vitbase_single_image_224'
CHECKPOINT_SHA256 = 'f5f338ebca4257372d543941065801ef518cea9abced737f742f094a60f8a650'
OFFICIAL_LWM_COMMIT = '5d2e0fd6c8b46850e6c0ac528d23a2db863e86d2'
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
PREPROCESS_SHA256 = digest({'source':'LWM get_image_transform', 'operations':['ToTensor','Resize((224,224),bilinear,antialias=True)','ImageNet Normalize'], 'crop':None})
POOL_SHA256 = digest({'operation':'identity','representation':'native_spatial_tokens','spatial_shape':[196,768]})

def identities(source_sha256):
    return dict(encoder_family=FAMILY, encoder_source_commit=OFFICIAL_LWM_COMMIT,
                encoder_source_sha256=source_sha256, checkpoint_sha256=CHECKPOINT_SHA256,
                preprocess_sha256=PREPROCESS_SHA256, pool_sha256=POOL_SHA256, whitening_sha256=None)

def validate_identities(value):
    if not isinstance(value,dict): value=dict(value)
    sha=value.get('encoder_source_sha256')
    if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('invalid CroCo source SHA')
    if value != identities(sha): raise ValueError('CroCo224 encoding identity mismatch')
    return dict(value)

def training_config(base, *,source_sha256):
    from j2j.context4_variants import CROCO224_IDENTITY, variant_identity_dict
    cfg=copy.deepcopy(base)
    cfg['experiment_id']=CROCO224_IDENTITY.variant_id
    cfg['variant_identity']=variant_identity_dict(CROCO224_IDENTITY)
    cfg['model']['grid_side']=14
    cfg['data']['expected_identities']=identities(source_sha256)
    cfg['encoder_contract']=dict(family=FAMILY, frozen=True, spatial_shape=[196,768], fusion=False)
    return cfg

def validate_training_config(cfg):
    if cfg.get('encoder_contract')!=dict(family=FAMILY,frozen=True,spatial_shape=[196,768],fusion=False):
        raise ValueError('CroCo224 encoder contract mismatch')
    if cfg['model'].get('grid_side')!=14:raise ValueError('CroCo224 requires grid_side14')
    validate_identities(cfg['data']['expected_identities'])
