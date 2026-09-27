"""CPU scientific contract for the native single-image LWM V2 adapter."""
import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'source'))
sys.path.insert(0, str(HERE / 'source/official'))
HUB = HERE.parents[1]
SOURCE = Path(os.environ.get('LWM_VJEPA_SOURCE',
                            HUB / '08_research_assets/github_sources/v_jepa2_1/repo'))


@pytest.fixture
def frontend(monkeypatch):
    def no_cuda(*args, **kwargs):
        pytest.fail('V2 unit tests must not initialize CUDA')
    monkeypatch.setattr(torch.cuda, '_lazy_init', no_cuda)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield importlib.import_module('lwm_stream.vjepa_frontend')
    finally:
        torch.set_num_threads(previous)


class RecordingEncoder(nn.Module):
    def __init__(self, defect=None):
        super().__init__()
        self.weight = nn.Parameter(torch.linspace(-.4, .7, 768))
        self.dropout = nn.Dropout(.8)
        self.calls = []
        self.defect = defect

    def forward(self, image):
        self.calls.append((tuple(image.shape), image.detach().clone(), self.training,
                           torch.is_grad_enabled()))
        pixels = image[:, :, 0, ::16, ::16].mean(1).reshape(image.shape[0], 576, 1)
        result = self.dropout(pixels + self.weight[None, None])
        if self.defect == 'tokens':
            return result[:, :-1]
        if self.defect == 'width':
            return result[:, :, :-1]
        if self.defect == 'dtype':
            return result.double()
        if self.defect == 'nan':
            return result * float('nan')
        return result


class SpawnImageDataset(torch.utils.data.Dataset):
    def __init__(self, transform, rgb):
        self.transform, self.rgb = transform, rgb

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.rgb))


def native_croco():
    """Original decoder code on native tokens, with small test-only fusion width."""
    from lwm.croco.croco import CroCoNet
    return CroCoNet(enc_embed_dim=768, enc_depth=0, enc_num_heads=12,
                    dec_embed_dim=32, dec_num_heads=4, dec_depth=1,
                    mlp_ratio=2, pos_embed='RoPE100')


def test_native_features_positions_original_fusion_and_frozen_gradients(frontend):
    from lwm.croco.croco import CroCoNet
    torch.manual_seed(107)
    original = native_croco()
    encoder = RecordingEncoder()
    retained = {name: getattr(original, name) for name in
                ('decoder_embed', 'dec_blocks', 'dec_norm', 'prediction_head')}
    old_front_parameters = {id(p) for name in ('patch_embed', 'enc_blocks', 'enc_norm')
                            for p in getattr(original, name).parameters()}
    wrapper = frontend.VJEPACroCo(original, encoder)
    assert isinstance(wrapper, CroCoNet)
    assert type(wrapper)._decoder is CroCoNet._decoder
    for name, module in retained.items():
        assert getattr(wrapper, name) is module
    assert not (old_front_parameters & {id(p) for p in wrapper.parameters()})
    assert all(not p.requires_grad for p in encoder.parameters())
    wrapper.train()
    assert wrapper.training and not encoder.training and not encoder.dropout.training
    now = torch.randn(1, 3, 1, 384, 384, requires_grad=True)
    goal = torch.randn_like(now, requires_grad=True)
    now_features, now_pos, mask = wrapper._encode_image(now, do_mask=False)
    goal_features, goal_pos, goal_mask = wrapper._encode_image(goal, do_mask=False)
    assert mask is None and goal_mask is None
    assert now_features.shape == goal_features.shape == (1, 576, 768)
    assert not now_features.requires_grad and not goal_features.requires_grad
    expected = now.detach()[:, :, 0, ::16, ::16].mean(1).reshape(1, 576, 1) + encoder.weight.detach()[None, None]
    torch.testing.assert_close(now_features, expected, rtol=0, atol=0)
    positions = torch.cartesian_prod(torch.arange(24), torch.arange(24))[None]
    assert now_pos.dtype in (torch.int32, torch.int64)
    torch.testing.assert_close(now_pos, positions, check_dtype=False, rtol=0, atol=0)
    torch.testing.assert_close(goal_pos, positions, check_dtype=False, rtol=0, atol=0)
    assert len(encoder.calls) == 2
    for call, image in zip(encoder.calls, (now, goal)):
        assert call[0] == (1, 3, 1, 384, 384)
        assert not call[2] and not call[3]
        torch.testing.assert_close(call[1], image.detach(), rtol=0, atol=0)
    fused = wrapper._decoder(now_features, now_pos, None, goal_features, goal_pos)
    golden = original._decoder(now_features, now_pos, None, goal_features, goal_pos)
    torch.testing.assert_close(fused, golden, rtol=0, atol=0)
    result = wrapper.prediction_head(fused)
    assert result.shape == (1, 576, 768) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert now.grad is None and goal.grad is None and encoder.weight.grad is None
    for name, module in retained.items():
        gradients = [p.grad for p in module.parameters()]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients), name
        assert sum(float(g.abs().sum()) for g in gradients) > 0, name


@pytest.mark.parametrize('shape,dtype', [
    ((1, 3, 384, 384), torch.float32),
    ((1, 3, 2, 384, 384), torch.float32),
    ((1, 3, 1, 224, 224), torch.float32),
    ((1, 1, 1, 384, 384), torch.float32),
    ((1, 3, 1, 384, 384), torch.float64),
    ((1, 3, 1, 384, 384), torch.float16),
])
def test_wrong_image_time_or_native_abi_fails_before_encoder(frontend, shape, dtype):
    encoder = RecordingEncoder()
    wrapper = frontend.VJEPACroCo(native_croco(), encoder)
    with pytest.raises((ValueError, TypeError)):
        wrapper._encode_image(torch.zeros(shape, dtype=dtype), do_mask=False)
    assert encoder.calls == []


def test_masked_pretraining_is_not_silently_enabled(frontend):
    encoder = RecordingEncoder()
    wrapper = frontend.VJEPACroCo(native_croco(), encoder)
    with pytest.raises((ValueError, NotImplementedError)):
        wrapper._encode_image(torch.zeros(1, 3, 1, 384, 384), do_mask=True)
    assert encoder.calls == []


@pytest.mark.parametrize('defect', ['tokens', 'width', 'dtype', 'nan'])
def test_invalid_encoder_output_is_rejected(frontend, defect):
    wrapper = frontend.VJEPACroCo(native_croco(), RecordingEncoder(defect))
    with pytest.raises((ValueError, TypeError)):
        wrapper._encode_image(torch.zeros(1, 3, 1, 384, 384), do_mask=False)


def test_install_only_replaces_frontend_and_synchronizes_wm_metadata(frontend):
    model = nn.Module()
    model.croco = native_croco()
    model.Np = 196
    model.cro_proj = nn.Linear(768, 384)
    model.action_network = nn.Linear(3, 384)
    projection, actions = model.cro_proj, model.action_network
    encoder = RecordingEncoder()
    frontend.install_frontend(model, encoder)
    assert isinstance(model.croco, frontend.VJEPACroCo)
    assert model.Np == 576
    assert model.cro_proj is projection and model.action_network is actions


def test_official_non_square_preprocessing_and_pickle_roundtrip(frontend):
    transform = frontend.VJEPAImageTransform(str(SOURCE))
    # Non-square image and asymmetric patterns catch stretching, flip and token-axis errors.
    yy, xx = np.meshgrid(np.arange(480), np.arange(640), indexing='ij')
    rgb = np.stack((xx % 256, yy % 256, (2 * xx + yy) % 256), axis=-1).astype(np.uint8)
    image = Image.fromarray(rgb)
    actual = transform(image)
    sys.path.insert(0, str(SOURCE))
    try:
        from evals.hub.preprocessor import vjepa2_preprocessor
        expected = vjepa2_preprocessor(crop_size=384)([torch.from_numpy(rgb.copy()).permute(2, 0, 1)])[0]
    finally:
        sys.path.remove(str(SOURCE))
    assert actual.shape == (3, 1, 384, 384) and actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = pickle.loads(pickle.dumps(transform))
    torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
    loader = torch.utils.data.DataLoader(SpawnImageDataset(restored, rgb), batch_size=1,
                                         num_workers=1, multiprocessing_context='spawn')
    batches = list(loader)
    assert len(batches) == 1
    torch.testing.assert_close(batches[0], expected[None], rtol=0, atol=0)
