"""Immutable feature and training contracts used by model identity validation."""
import copy
import hashlib
import json

FAMILY = 'dinov3_vitb16_lvd1689m_256'
CHECKPOINT_SHA256 = '73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c'
OFFICIAL_COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

PREPROCESS_SHA256 = digest({'source':'Meta DINOv3 README LVD1689M make_transform',
    'operations':['v2.ToImage','v2.Resize((256,256),bilinear,antialias=True)',
        'v2.ToDtype(float32,scale=True)','ImageNet Normalize'], 'crop':None})
POOL_SHA256 = digest({'operation':'identity','representation':'native_spatial_tokens','spatial_shape':[256,768]})

def identities(source_sha256):
    return dict(encoder_family=FAMILY,encoder_source_commit=OFFICIAL_COMMIT,
        encoder_source_sha256=source_sha256,checkpoint_sha256=CHECKPOINT_SHA256,
        preprocess_sha256=PREPROCESS_SHA256,pool_sha256=POOL_SHA256,whitening_sha256=None)

def validate_identities(value):
    value=dict(value);sha=value.get('encoder_source_sha256')
    if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('invalid DINOv3 source SHA')
    if value!=identities(sha):raise ValueError('DINOv3 encoding identity mismatch')
    return value

def training_config(base,*,source_sha256):
    from j2j.context4_variants import DINOV3_256_IDENTITY,variant_identity_dict
    cfg=copy.deepcopy(base)
    cfg['experiment_id']=DINOV3_256_IDENTITY.variant_id
    cfg['variant_identity']=variant_identity_dict(DINOV3_256_IDENTITY)
    cfg['model']['grid_side']=16
    cfg['data']['expected_identities']=identities(source_sha256)
    cfg['encoder_contract']=dict(family=FAMILY,frozen=True,spatial_shape=[256,768],fusion=False)
    return cfg

def validate_training_config(cfg):
    from collections.abc import Mapping
    contract = cfg.get('encoder_contract')
    if not isinstance(contract, Mapping):
        raise ValueError('DINOv3 encoder contract mismatch')
    # Checkpoint admission freezes JSON lists into tuples; preserve the caller.
    contract = dict(contract)
    shape = contract.get('spatial_shape')
    if isinstance(shape, (list, tuple)):
        contract['spatial_shape'] = list(shape)
    if contract!=dict(family=FAMILY,frozen=True,spatial_shape=[256,768],fusion=False):raise ValueError('DINOv3 encoder contract mismatch')
    if cfg['model'].get('grid_side')!=16:raise ValueError('DINOv3-256 requires grid_side16')
    validate_identities(cfg['data']['expected_identities'])
