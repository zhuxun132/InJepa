"""V2 stage wiring, scientific identity and dataset seam; CPU only."""
import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'source'))
sys.path.insert(0, str(HERE / 'source/official'))
from lwm_stream import entry
from test_vjepa_frontend import RecordingEncoder, native_croco


def vision_config():
    return {
        'kind': 'vjepa21_vitb384_native',
        'source_root': '/machine-a/official-vjepa',
        'checkpoint': '/machine-a/ema.pt',
        'checkpoint_sha256': '848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d',
        'checkpoint_bytes': 1664223428,
        'source_commit': '204698b45b3712590f06245fbfba32d3be539812',
        'source_tree': 'dd6cfc1e792158510b983d827cb2e84f47fd5706',
    }


def config():
    result = json.loads((HERE / 'configs/lwm_stream_train_batch8_scratch.json').read_text())
    result['identity'] = 'lwm_vjepa21_encoder_croco_fusion_v2'
    result['vision'] = vision_config()
    return result


def admission():
    return {'hashes': {'croco': 'a' * 64, 'wm_il': 'b' * 64},
            'predecessors': {}, 'pseudo': None}


def test_scientific_identity_binds_native_encoder_without_machine_paths():
    cfg = config()
    identity = entry.scientific_identity(cfg, 'wm', admission())
    assert 'vision' in identity
    assert identity['vision']['checkpoint_sha256'] == cfg['vision']['checkpoint_sha256']
    assert identity['vision']['source_commit'] == cfg['vision']['source_commit']
    assert identity['vision']['source_tree'] == cfg['vision']['source_tree']
    assert identity['vision']['preprocessing'] == 'official_vjepa384_shortside_centercrop_imagenet'
    assert identity['vision']['image_shape'] == [3, 1, 384, 384]
    assert identity['vision']['grid_shape'] == [576, 768]
    assert identity['vision']['frozen'] is True
    relocated = copy.deepcopy(cfg)
    relocated['vision'].update(source_root='/machine-b/source', checkpoint='/machine-b/ema.pt')
    assert entry.scientific_identity(relocated, 'wm', admission()) == identity
    assert '/machine-a' not in json.dumps(identity)
    changed = copy.deepcopy(cfg)
    changed['vision']['checkpoint_sha256'] = '1' * 64
    assert entry.scientific_identity(changed, 'wm', admission()) != identity


@pytest.mark.parametrize('field,value', [
    ('preprocessing', 'stretch224'), ('grid_shape', [196, 768]),
    ('image_shape', [3, 2, 384, 384]), ('frozen', False),
    ('source_commit', 'f' * 40), ('kind', 'rae'),
])
def test_unsupported_visual_science_fails_closed(field, value):
    cfg = config()
    cfg['vision'][field] = value
    with pytest.raises((ValueError, TypeError)):
        entry.scientific_identity(cfg, 'wm', admission())


def test_no_vision_route_retains_legacy_scientific_fields():
    cfg = config()
    del cfg['vision']
    identity = entry.scientific_identity(cfg, 'wm', admission())
    assert 'vision' not in identity
    assert identity['settings'] == cfg['wm'] and identity['adam'] == cfg['adam']


@pytest.fixture
def wiring(tmp_path, monkeypatch):
    frontend = importlib.import_module('lwm_stream.vjepa_frontend')
    from lwm import world_model, policy
    from lwm_stream import initialization
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    events, models, encoders = [], [], []

    class TinyModel(nn.Module):
        def __init__(self, kind, **kwargs):
            super().__init__()
            self.kind, self.constructor_kwargs = kind, kwargs
            self.croco = native_croco()
            self.cro_proj = nn.Linear(768, 384)
            self.Np = 196
            self.max_len = 65
            models.append(self)
            events.append(('construct', id(self), kind, kwargs))

    monkeypatch.setattr(world_model, 'LatentWorldModel', lambda **kw: TinyModel('wm', **kw))
    monkeypatch.setattr(policy, 'ARPlusPolicy', lambda **kw: TinyModel('policy', **kw))

    def generic(model, *args):
        assert not isinstance(model.croco, frontend.VJEPACroCo)
        with torch.no_grad():
            model.croco.prediction_head.bias.fill_(.125)
        model.generic_fusion = model.croco.prediction_head
        events.append(('generic', id(model)))

    def encoder(vision):
        assert vision == cfg['vision']
        result = RecordingEncoder()
        encoders.append(result)
        events.append(('encoder', id(result)))
        return result

    def completed(model, predecessor, epochs):
        assert isinstance(model.croco, frontend.VJEPACroCo), 'must install ABI before strict load'
        events.append(('completed', id(model), predecessor['stage'], epochs))

    monkeypatch.setattr(initialization, 'initialize_croco', generic)
    monkeypatch.setattr(frontend, 'load_frozen_encoder', encoder)
    if hasattr(entry, 'load_frozen_encoder'):
        monkeypatch.setattr(entry, 'load_frozen_encoder', encoder)
    monkeypatch.setattr(entry, 'load_completed', completed)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda *a, **k: pytest.fail('CPU tests'))
    cfg = config()
    books = tmp_path / 'books'
    books.mkdir()
    (books / 'action_centers.json').write_text(json.dumps(np.zeros((64, 2)).tolist()))
    (books / 'trajectory_centers.json').write_text(json.dumps(np.zeros((64, 63, 3)).tolist()))
    cfg['codebooks'] = str(books)
    admitted = admission()
    admitted['predecessors'] = {name: {'stage': name, 'sha256': name * 32} for name in ('wm', 'il')}
    try:
        yield cfg, admitted, frontend, events, models, encoders
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('stage', ['wm', 'il'])
def test_fresh_stages_initialize_generic_fusion_then_replace_frontend(wiring, stage):
    cfg, admitted, frontend, events, models, encoders = wiring
    model, objective = entry.build_model(cfg, stage, admitted, torch.device('cpu'))
    assert len(models) == len(encoders) == 1
    assert isinstance(model.croco, frontend.VJEPACroCo)
    assert model.croco.prediction_head is model.generic_fusion
    torch.testing.assert_close(model.croco.prediction_head.bias, torch.full((768,), .125))
    assert [event[0] for event in events] == ['construct', 'generic', 'encoder']
    assert objective.stage == stage and objective.model is model
    assert all(not parameter.requires_grad for parameter in encoders[0].parameters())
    if stage == 'il':
        assert model.constructor_kwargs['feats_seq_len'] == 576
    else:
        assert model.Np == 576 and objective.epsilon_m == cfg['wm']['epsilon_m']


def test_rl_loads_new_abi_before_each_completed_stage_and_preserves_owners(wiring):
    cfg, admitted, frontend, events, models, encoders = wiring
    policy, objective = entry.build_model(cfg, 'rl', admitted, torch.device('cpu'))
    assert len(models) == len(encoders) == 3
    assert all(isinstance(model.croco, frontend.VJEPACroCo) for model in models)
    assert not any(event[0] == 'generic' for event in events)
    assert sorted(event[2] for event in events if event[0] == 'completed') == ['il', 'il', 'wm']
    owners = [{id(p) for p in model.parameters()} for model in models]
    assert all(not owners[i] & owners[j] for i in range(3) for j in range(i))
    for model in models:
        if model.kind == 'policy':
            assert model.constructor_kwargs['feats_seq_len'] == 576
    assert objective.model is policy
    assert objective.wm in models and objective.reference in models
    for name in ('num_sample', 'temperature', 'beta', 'clip_epsilon'):
        assert getattr(objective, name) == cfg['rl'][name]
    assert all(not module.training for model in models for module in model.modules())


def test_pseudo_uses_v2_wm_and_same_trajectory_codebook(wiring):
    cfg, admitted, frontend, events, models, encoders = wiring
    wm, tokenizer, candidates = entry.build_pseudo_model(cfg, admitted, torch.device('cpu'))
    assert len(models) == len(encoders) == 1
    assert isinstance(wm.croco, frontend.VJEPACroCo)
    assert [event[0] for event in events] == ['construct', 'encoder', 'completed']
    assert events[-1][2:] == ('wm', cfg['wm']['epochs'])
    assert candidates.shape == (64, 63, 3) and tokenizer.K == 64
    assert not wm.training


@pytest.mark.parametrize('stage', ['wm', 'pseudo', 'rl'])
def test_dataset_only_changes_image_transform_not_rows_seed_or_candidate_rule(wiring, monkeypatch, stage):
    cfg, admitted, frontend, *_ = wiring
    cfg.pop('sampling')
    rows = [{'pose_key': 'train-physical-1', 'keyframe_indices': [0, 4, 9], 'split': 'train'}]
    partition = 'rl' if stage == 'rl' else 'wm_il'
    (Path(cfg['codebooks']) / (partition + '.jsonl')).write_text(json.dumps(rows[0]) + '\n')
    seen = {}

    class Dataset:
        def __init__(self, actual_rows, **kwargs):
            seen.update(rows=actual_rows, kwargs=kwargs)
            self.transform = 'old224'

    class Transform:
        def __init__(self, source_root):
            self.source_root = source_root

    monkeypatch.setattr(entry, 'ReplayDataset', Dataset)
    monkeypatch.setattr(frontend, 'VJEPAImageTransform', Transform)
    if hasattr(entry, 'VJEPAImageTransform'):
        monkeypatch.setattr(entry, 'VJEPAImageTransform', Transform)
    dataset = entry.build_dataset(cfg, stage, admitted)
    assert isinstance(dataset.transform, Transform)
    assert dataset.transform.source_root == cfg['vision']['source_root']
    assert seen['rows'] == rows
    assert seen['kwargs'] == dict(mode='wm' if stage == 'wm' else 'pair', seed=cfg['seed'],
                                  num_candidates=cfg['wm']['num_candidates'] if stage == 'wm' else 1)


@pytest.mark.parametrize('native', [True, False])
def test_navigation_transform_injection_keeps_t1_and_legacy_default(monkeypatch, native):
    from PIL import Image
    from lwm_stream import navigation
    from lwm.preprocess import get_image_transform
    policy, wm, tokenizer = nn.Linear(1, 1), nn.Linear(1, 1), object()
    current = np.full((12, 16, 3), 31, dtype=np.uint8)
    goal = np.full((12, 16, 3), 89, dtype=np.uint8)
    calls = []

    def native_transform(image):
        assert isinstance(image, Image.Image) and image.mode == 'RGB'
        value = float(np.asarray(image)[0, 0, 0])
        return torch.full((3, 1, 384, 384), value)

    def sample(p, w, t, now, goal_tensor, **kwargs):
        assert p is policy and w is wm and t is tokenizer
        calls.append((now, goal_tensor))
        return {'rewards': torch.tensor([[1.]]), 'lengths': torch.tensor([[1]]),
                'actions_m': torch.tensor([[[[.2, -.1, .0]]]])}

    class UnreadableHistory:
        def __iter__(self):
            raise AssertionError('no history in single-image LWM')

    monkeypatch.setattr(navigation, 'sample_rewards', sample)
    kwargs = {'transform': native_transform} if native else {}
    adapter = navigation.LWMNavigationAdapter(policy, wm, tokenizer, device='cpu', **kwargs)
    result = adapter.act(current, goal, UnreadableHistory())
    assert result['continuous_action']['stop'] is False
    assert result['continuous_action']['waypoint_m'] == pytest.approx([.2, -.1])
    transform = native_transform if native else get_image_transform()
    for actual, rgb in zip(calls[0], (current, goal)):
        torch.testing.assert_close(actual, transform(Image.fromarray(rgb))[None], rtol=0, atol=0)
        assert actual.shape == ((1, 3, 1, 384, 384) if native else (1, 3, 224, 224))


def test_actual_completed_v2_roundtrip_and_legacy_abi_rejection(tmp_path):
    frontend = importlib.import_module('lwm_stream.vjepa_frontend')
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        def model(native):
            result = nn.Module()
            result.croco = native_croco()
            result.head = nn.Linear(768, 3)
            if native:
                frontend.install_frontend(result, RecordingEncoder())
            return result

        source, target = model(True), model(True)
        scientific = entry.scientific_identity(config(), 'wm', admission())
        path = tmp_path / 'completed.pt'
        payload = {'format': 'LWM_STREAM_LOCAL_STATE_V1',
                   'identity': {'scientific': scientific},
                   'progress': {'training': {'epoch': 50, 'sampler_offset': 0}},
                   'model': source.state_dict()}
        torch.save(payload, path)
        predecessor = {'path': str(path), 'scientific': scientific}
        entry.load_completed(target, predecessor, 50)
        for name, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
        before = {name: value.clone() for name, value in target.state_dict().items()}
        # Even maliciously re-labelled V1 weights must fail the strict model ABI.
        payload['model'] = model(False).state_dict()
        torch.save(payload, path)
        with pytest.raises(ValueError, match='ABI'):
            entry.load_completed(target, predecessor, 50)
        for name, value in before.items():
            torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous)
