"""Exact data relocation and JPEG spool integrity; CPU-only deploy boundary."""
import copy
import hashlib
import importlib.util
import io
import os
import struct
from pathlib import Path

import pytest
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / 'receipts/deploy_4090_20260912/prepare_data.py'


@pytest.fixture(scope='module')
def deploy():
    assert SCRIPT.is_file(), 'EXPECTED RED: data relocation tool does not exist yet'
    spec = importlib.util.spec_from_file_location('lwm_v2_deploy_data', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_row():
    return {
        'source': 'r2r', 'video': 'images/train/scene-a/trajectory-a',
        'rgb_dir': '/old/rgb/r2r/images/train/scene-a/trajectory-a/rgb',
        'pose_path': '/old/replay/scene-a/physical-hash.npz',
        'frame_names': ['00001.jpg', '00002.jpg', '00003.jpg', '00004.jpg'],
        'keyframe_indices': [0, 2, 3], 'split': 'train', 'scene': 'scene-a',
        'actions': [1, 2, 3], 'aliases': [{'id': 'alias-a', 'source': 'r2r'}],
        'pose_sha256': '0' * 64, 'physical_trajectory_id': 'physical-hash',
    }


def test_relocation_changes_only_paths_and_does_not_alias_nested_data(deploy):
    original = original_row()
    before = copy.deepcopy(original)
    result = deploy.relocated_row(original, '/new/rgb', '/new/replay')
    expected = copy.deepcopy(before)
    expected['rgb_dir'] = '/new/rgb/r2r/images/train/scene-a/trajectory-a/rgb'
    expected['pose_path'] = '/new/replay/scene-a/physical-hash.npz'
    assert result == expected
    assert original == before
    result['aliases'][0]['id'] = 'mutated'
    result['keyframe_indices'].append(1)
    assert original == before


@pytest.mark.parametrize('source,source_id', [('r2r', 'R2R'), ('rxr', 'RxR')])
def test_requests_preserve_original_sparse_keyframes_and_final_goal(deploy, source, source_id):
    row = original_row()
    row['source'] = source
    assert deploy.frame_requests(row) == [
        (source_id, f'images/train/scene-a/trajectory-a/rgb/{name}')
        for name in ('00001.jpg', '00003.jpg', '00004.jpg')
    ]


@pytest.mark.parametrize('field,value', [
    ('source', 'mp3d'), ('video', '../outside'),
    ('frame_names', ['../outside.jpg', '2.jpg', '3.jpg', '4.jpg']),
    ('keyframe_indices', [-1, 2]), ('keyframe_indices', [0, 4]),
])
def test_requests_fail_closed_on_unadmitted_source_or_bad_frame_mapping(deploy, field, value):
    row = original_row()
    row[field] = value
    with pytest.raises((ValueError, TypeError, IndexError)):
        deploy.frame_requests(row)


def jpeg_record(color, offset):
    encoded = io.BytesIO()
    Image.new('RGB', (7, 5), color=color).save(encoded, format='JPEG')
    payload = encoded.getvalue()
    with Image.open(io.BytesIO(payload)) as image:
        decoded = image.convert('RGB').tobytes()
    return payload, {
        'offset': offset, 'length': len(payload), 'width': 7, 'height': 5,
        'compressed_jpeg_sha256': hashlib.sha256(payload).hexdigest(),
        'decoded_rgb_sha256': hashlib.sha256(decoded).hexdigest(),
    }


@pytest.fixture
def spool(tmp_path):
    first, _ = jpeg_record((20, 80, 120), 0)
    second, record = jpeg_record((240, 10, 80), 8 + len(first))
    path = tmp_path / 'rgb.spool'
    path.write_bytes(struct.pack('<Q', len(first)) + first + struct.pack('<Q', len(second)) + second)
    fd = os.open(path, os.O_RDONLY)
    try:
        yield fd, record, second, path
    finally:
        os.close(fd)


def test_jpeg_read_uses_record_offset_and_preserves_original_compressed_bytes(deploy, spool):
    fd, record, payload, _ = spool
    os.lseek(fd, 11, os.SEEK_SET)
    assert deploy.read_verified_jpeg(fd, record) == payload
    assert os.lseek(fd, 0, os.SEEK_CUR) == 11, 'pread must not race on shared file cursor'


@pytest.mark.parametrize('field,value', [
    ('length', 1), ('compressed_jpeg_sha256', '1' * 64),
    ('decoded_rgb_sha256', '2' * 64), ('width', 8), ('height', 6),
])
def test_jpeg_rejects_length_hash_and_pixel_shape_mismatch(deploy, spool, field, value):
    fd, record, _, _ = spool
    record = {**record, field: value}
    with pytest.raises((ValueError, RuntimeError, OSError)):
        deploy.read_verified_jpeg(fd, record)


def test_jpeg_rejects_truncated_record(deploy, spool):
    fd, record, _, path = spool
    with path.open('r+b') as writable:
        writable.truncate(path.stat().st_size - 5)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        deploy.read_verified_jpeg(fd, record)


def test_extract_job_is_spawn_picklable_and_preserves_verified_outputs(tmp_path, monkeypatch):
    import importlib
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    module = importlib.import_module('prepare_data')
    assert callable(getattr(module, 'extract_job', None)), 'EXPECTED RED: top-level process extract_job missing'
    first, record_one = jpeg_record((30, 180, 100), 0)
    second, record_two = jpeg_record((210, 50, 10), 8 + len(first))
    path = tmp_path / 'spawn.spool'
    path.write_bytes(struct.pack('<Q', len(first)) + first + struct.pack('<Q', len(second)) + second)
    outputs = [tmp_path / 'first.jpg', tmp_path / 'second.jpg']
    jobs = [(str(path), record, str(target)) for record, target in zip((record_one, record_two), outputs)]
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        list(pool.map(module.extract_job, jobs))
        assert [target.read_bytes() for target in outputs] == [first, second]
        list(pool.map(module.extract_job, jobs))
        assert [hashlib.sha256(target.read_bytes()).hexdigest() for target in outputs] == [
            record_one['compressed_jpeg_sha256'], record_two['compressed_jpeg_sha256']]
        outputs[0].write_bytes(b'corrupted existing JPEG')
        with pytest.raises((ValueError, RuntimeError, OSError)):
            pool.submit(module.extract_job, jobs[0]).result()
