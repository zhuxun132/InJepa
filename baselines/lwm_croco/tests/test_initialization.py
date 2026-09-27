"""Original CroCo initialization integrity and ownership, tiny CPU fixtures."""
import hashlib
import sys
from pathlib import Path
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'source'))
from lwm_stream.initialization import initialize_croco

KWARGS = {'dec_embed_dim': 12, 'dec_num_heads': 3, 'dec_depth': 1, 'pos_embed': 'RoPE100'}

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.croco = nn.Sequential(nn.Linear(3,4),nn.Linear(4,4))
        self.croco[1].requires_grad_(False)
        self.croco.register_buffer('audit_buffer',torch.tensor([3.]))
        self.navigation_head = nn.Linear(4,1)

@pytest.fixture(autouse=True)
def rng_isolation():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(63)
        yield

def payload(model):
    return {'model': {k: torch.full_like(v, 0.25 + i) for i,(k,v) in enumerate(model.croco.state_dict().items())},
            'croco_kwargs': dict(KWARGS)}

def save(path, value):
    torch.save(value,path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot(model):
    return {k:v.clone() for k,v in model.state_dict().items()}

def assert_unchanged(model,before):
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value,before[key],rtol=0,atol=0)

def test_success_only_backbone_preserves_identity_flags_rng_and_safe_load(tmp_path,monkeypatch):
    model = TinyModel()
    before = snapshot(model)
    identity = {name:(id(p),p.requires_grad) for name,p in model.named_parameters()}
    data = payload(model)
    path = tmp_path/'croco.pth'
    sha = save(path,data)
    calls = []
    original = torch.load
    def checked_load(file,*args,**kwargs):
        assert kwargs.get('weights_only') is True
        assert hasattr(file,'read') and hasattr(file,'fileno')
        calls.append(file.fileno())
        return original(file,*args,**kwargs)
    monkeypatch.setattr(torch,'load',checked_load)
    rng = torch.get_rng_state().clone()
    receipt = initialize_croco(model,path,sha,KWARGS)
    assert isinstance(receipt,dict) and receipt
    assert len(calls) == 1
    assert torch.equal(rng,torch.get_rng_state())
    assert identity == {name:(id(p),p.requires_grad) for name,p in model.named_parameters()}
    for key,value in model.croco.state_dict().items():
        torch.testing.assert_close(value,data['model'][key],rtol=0,atol=0)
    for key,value in model.navigation_head.state_dict().items():
        torch.testing.assert_close(value,before['navigation_head.'+key],rtol=0,atol=0)

@pytest.mark.parametrize('mutation',['kwargs','missing','extra','shape','nan','inf'])
def test_rejected_asset_is_atomic(tmp_path,mutation):
    model = TinyModel()
    before = snapshot(model)
    data = payload(model)
    if mutation == 'kwargs': data['croco_kwargs']['dec_depth'] = 2
    elif mutation == 'missing': del data['model']['1.bias']
    elif mutation == 'extra': data['model']['extra.weight'] = torch.ones(1)
    elif mutation == 'shape': data['model']['1.bias'] = torch.ones(5)
    elif mutation == 'nan': data['model']['1.bias'][0] = float('nan')
    else: data['model']['1.bias'][0] = float('inf')
    path = tmp_path/'bad.pth'
    sha = save(path,data)
    with pytest.raises(ValueError):
        initialize_croco(model,path,sha,KWARGS)
    assert_unchanged(model,before)

def test_hash_rejection_happens_before_deserialization(tmp_path,monkeypatch):
    model = TinyModel()
    before = snapshot(model)
    path = tmp_path/'wrong.pth'
    save(path,payload(model))
    def forbidden(*args,**kwargs):
        pytest.fail('hash mismatch must be rejected before torch.load')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError):
        initialize_croco(model,path,'0'*64,KWARGS)
    assert_unchanged(model,before)

def test_load_uses_hashed_descriptor_despite_path_replacement(tmp_path,monkeypatch):
    model = TinyModel()
    original_payload = payload(model)
    path = tmp_path/'original.pth'
    sha = save(path,original_payload)
    replacement = tmp_path/'replacement.pth'
    substitute = payload(model)
    substitute['model']['0.weight'].fill_(99)
    save(replacement,substitute)
    load = torch.load
    def replace_then_load(file,*args,**kwargs):
        assert hasattr(file,'fileno'), 'must load the hashed open file'
        replacement.replace(path)
        return load(file,*args,**kwargs)
    monkeypatch.setattr(torch,'load',replace_then_load)
    initialize_croco(model,path,sha,KWARGS)
    torch.testing.assert_close(model.croco[0].weight,original_payload['model']['0.weight'],rtol=0,atol=0)

def test_trained_navigation_archive_is_not_an_initialization_source(tmp_path):
    model = TinyModel()
    before = snapshot(model)
    data = {'model':snapshot(model),'croco_kwargs':dict(KWARGS)}
    path = tmp_path/'trained_lwm.pth'
    sha = save(path,data)
    with pytest.raises(ValueError):
        initialize_croco(model,path,sha,KWARGS)
    assert_unchanged(model,before)
