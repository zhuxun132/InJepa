"""Published RGB and measured replay alignment for LWM datasets."""
import hashlib
import io
import numbers
import copy
import bisect
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .data import local_trajectory, select_keyframes, validate_annotation

def index_trajectory(record, pose_path, rgb_dir):
    pose_path, rgb_dir = Path(pose_path), Path(rgb_dir)
    if not pose_path.is_file() or not rgb_dir.is_dir():
        raise ValueError('required pose file or RGB directory is missing')
    names = validate_annotation(record, [p.name for p in rgb_dir.iterdir()
                                         if p.is_file() and p.suffix == '.jpg'])
    payload = pose_path.read_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        positions = arrays['positions']
        rotations = arrays['quaternions_xyzw']
        indices = arrays['state_index']
        if (len(positions) != len(names) or indices.dtype.kind not in 'iu'
                or not np.array_equal(indices, np.arange(len(names)))):
            raise ValueError('pose states do not match published RGB/action indices')
        local_trajectory(positions, rotations)
        selected = select_keyframes(positions)
    return dict(video=record['video'], rgb_dir=str(rgb_dir), frame_names=names,
                pose_path=str(pose_path), pose_sha256=hashlib.sha256(payload).hexdigest(),
                keyframe_indices=selected)


def window_geometry(positions, quaternions_xyzw, keyframe_indices, anchor_index, *, max_future=63):
    keys = np.asarray(keyframe_indices)
    if (keys.ndim != 1 or keys.dtype.kind not in 'iu' or len(keys) < 2
            or np.any(keys[1:] <= keys[:-1]) or keys[0] < 0
            or keys[-1] >= len(positions)):
        raise ValueError('keyframes must be increasing original frame indices with a future')
    if (isinstance(anchor_index, bool) or not isinstance(anchor_index, numbers.Integral)
            or not 0 <= anchor_index < len(keys) - 1):
        raise ValueError('anchor must have a real future keyframe')
    if (isinstance(max_future, bool) or not isinstance(max_future, numbers.Integral)
            or max_future <= 0):
        raise ValueError('future horizon must be a positive integer')
    now = int(keys[anchor_index])
    local = local_trajectory(positions, quaternions_xyzw, origin_index=now)
    future = keys[anchor_index + 1:anchor_index + 1 + max_future]
    count = len(future)
    actions = np.zeros((max_future, 3), dtype=np.float32)
    actions[:count] = local[future]
    valid = np.zeros(max_future, dtype=bool)
    valid[:count] = True
    indices = np.full(max_future, -1, dtype=np.int64)
    indices[:count] = future
    return dict(actions=actions, valid=valid, future_frame_indices=indices, now_frame_index=now)


def partition_rows(rows, *, seed):
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError('seed must be an integer')
    aliases, owners = set(), {}
    for row in rows:
        alias = (row['source'], row['video'])
        if alias in aliases or row['split'] not in ('train', 'dev'):
            raise ValueError('duplicate alias or unsupported split')
        aliases.add(alias)
        key = row['pose_key']
        if key in owners and owners[key] != row['split']:
            raise ValueError('physical trajectory crosses train/development boundary')
        owners[key] = row['split']
    keys = sorted((key for key in owners if owners[key] == 'train'),
                  key=lambda key: (hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest(), key))
    if len(keys) < 2:
        raise ValueError('two distinct training trajectories are required for the two halves')
    wm_keys = set(keys[:len(keys) // 2])
    result = dict(wm_il=[], rl=[], dev=[])
    for row in rows:
        section = 'dev' if row['split'] == 'dev' else ('wm_il' if row['pose_key'] in wm_keys else 'rl')
        result[section].append(copy.deepcopy(row))
    for section in result.values():
        section.sort(key=lambda row: (row['source'], row['video']))
    return result


def physical_rows(rows):
    groups, aliases = {}, set()
    for row in rows:
        alias = (row['source'], row['video'])
        if alias in aliases:
            raise ValueError('duplicate source alias')
        aliases.add(alias)
        group = groups.setdefault(row['pose_key'], [])
        if group:
            previous = group[0]
            if (any(previous[name] != row[name] for name in ('source', 'split', 'pose_sha256'))
                    or len(previous['frame_names']) != len(row['frame_names'])
                    or not np.array_equal(previous['keyframe_indices'], row['keyframe_indices'])):
                raise ValueError('physical aliases disagree about source, split or geometry')
        group.append(row)
    result = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: (row['source'], row['video']))
        representative = copy.deepcopy(group[0])
        representative['source_aliases'] = [dict(source=row['source'], video=row['video']) for row in group]
        result.append(representative)
    return result


class StratifiedReplayDataset:
    def __init__(self, dataset, *, anchor_bin_size, seed):
        for name, value, minimum in (('anchor_bin_size', anchor_bin_size, 1), ('seed', seed, 0)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
                raise ValueError(f'invalid {name}')
        keys = [row['pose_key'] for row in dataset.rows]
        if len(set(keys)) != len(keys):
            raise ValueError('stratified sampling requires unique physical trajectories')
        self.dataset, self.seed, self.anchor_bin_size = dataset, int(seed), int(anchor_bin_size)
        self._bins = []
        for row_index, row in enumerate(dataset.rows):
            length = len(row['keyframe_indices']) - 1
            # Stable physical identity: a row permutation must not change a bin's draw.
            key_seed = int.from_bytes(hashlib.sha256(row['pose_key'].encode()).digest()[:16], 'little')
            for start in range(0, length, self.anchor_bin_size):
                self._bins.append((dataset._offsets[row_index] + start,
                                   min(self.anchor_bin_size, length-start), key_seed, start))

    def __len__(self):
        return len(self._bins)

    def anchor_index(self, epoch, index):
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0 for v in (epoch, index)):
            raise ValueError('epoch and sample index must be nonnegative integers')
        if index >= len(self):
            raise IndexError('stratified sample index out of range')
        start, width, key_seed, local_start = self._bins[index]
        cycle, offset = divmod(int(epoch), width)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, key_seed, local_start, cycle]))
        return start + int(rng.permutation(width)[offset])

    def __getitem__(self, item):
        epoch, index = item if isinstance(item, tuple) and len(item) == 2 else (0, item)
        return self.dataset[(epoch, self.anchor_index(epoch, index))]


class ReplayDataset:
    def __init__(self, rows, *, mode, seed, num_candidates=1, max_future=63, pose_cache_size=128):
        from lwm.preprocess import get_image_transform
        if mode not in ('wm', 'pair'):
            raise ValueError('mode must be wm or pair')
        for value in (num_candidates, max_future, pose_cache_size):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError('candidate count, horizon and cache size must be positive integers')
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        self.rows = copy.deepcopy(list(rows))
        if not self.rows:
            raise ValueError('dataset is empty')
        self.mode, self.seed = mode, int(seed)
        self.num_candidates, self.max_future = num_candidates, max_future
        self.pose_cache_size, self._cache = pose_cache_size, OrderedDict()
        self._offsets, self._physical_rows = [0], {}
        aliases = set()
        splits = set()
        for i, row in enumerate(self.rows):
            alias = (row['source'], row['video'])
            if alias in aliases:
                raise ValueError('duplicate dataset alias')
            aliases.add(alias); splits.add(row['split'])
            keys = np.asarray(row['keyframe_indices'])
            if (keys.ndim != 1 or keys.dtype.kind not in 'iu' or len(keys) < 2
                    or keys[0] < 0 or keys[-1] >= len(row['frame_names'])
                    or np.any(keys[1:] <= keys[:-1])):
                raise ValueError('dataset row has invalid keyframes or no future')
            self._offsets.append(self._offsets[-1] + len(keys) - 1)
            self._physical_rows.setdefault(row['pose_key'], []).append(i)
        if len(splits) != 1 or not splits <= {'train', 'dev'}:
            raise ValueError('dataset must use one admitted split')
        if mode == 'wm' and num_candidates > 1 and len(self._physical_rows) < 2:
            raise ValueError('counterfactual candidates require a different physical trajectory')
        self.transform = get_image_transform()

    def __len__(self):
        return self._offsets[-1]

    def _window(self, row_index, anchor):
        row = self.rows[row_index]
        key = (row['pose_path'], row['pose_sha256'])
        if key not in self._cache:
            payload = Path(row['pose_path']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != row['pose_sha256']:
                raise ValueError('pose checksum differs from RGB index')
            with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                p, q = arrays['positions'], arrays['quaternions_xyzw']
                if len(p) != len(row['frame_names']) or not np.array_equal(arrays['state_index'], np.arange(len(p))):
                    raise ValueError('pose states differ from RGB index')
            self._cache[key] = (p, q)
            if len(self._cache) > self.pose_cache_size:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        p, q = self._cache[key]
        return window_geometry(p, q, row['keyframe_indices'], anchor, max_future=self.max_future)

    def _image(self, row, frame):
        from PIL import Image
        with Image.open(Path(row['rgb_dir']) / row['frame_names'][frame]) as image:
            return self.transform(image.convert('RGB'))

    def __getitem__(self, item):
        import torch
        epoch, index = item if isinstance(item, tuple) and len(item) == 2 else (0, item)
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0 for v in (epoch, index)):
            raise ValueError('epoch and flat index must be nonnegative integers')
        if index >= len(self):
            raise IndexError('dataset index is out of range')
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(epoch), int(index)]))
        row_index = bisect.bisect_right(self._offsets, index) - 1
        row = self.rows[row_index]
        window = self._window(row_index, index - self._offsets[row_index])
        goal_step = int(rng.integers(int(window['valid'].sum())))
        result = dict(now=self._image(row, window['now_frame_index']),
                      goal=self._image(row, int(window['future_frame_indices'][goal_step])))
        if self.mode == 'pair':
            return result
        actions, masks = [window['actions']], [window['valid']]
        excluded = self._physical_rows[row['pose_key']]
        for _ in range(self.num_candidates - 1):
            # Map one draw onto the complement, preserving uniform donor-row weights.
            donor = int(rng.integers(len(self.rows) - len(excluded)))
            for excluded_index in excluded:
                if excluded_index <= donor:
                    donor += 1
                else:
                    break
            anchor = int(rng.integers(len(self.rows[donor]['keyframe_indices']) - 1))
            other = self._window(donor, anchor)
            actions.append(other['actions']); masks.append(other['valid'])
        result.update(actions_m=torch.from_numpy(np.stack(actions)),
                      valid_mask=torch.from_numpy(np.stack(masks)),
                      goal_xy_m=torch.from_numpy(window['actions'][goal_step, :2].copy()))
        return result


def codebook_samples(rows, *, horizon=63):
    if isinstance(horizon, bool) or not isinstance(horizon, numbers.Integral) or horizon < 1:
        raise ValueError('codebook horizon must be a positive integer')
    groups, aliases = {}, set()
    for row in rows:
        alias = (row['source'], row['video'])
        if alias in aliases or row['split'] != 'train':
            raise ValueError('codebooks require unique training aliases only')
        aliases.add(alias)
        key = row['pose_key']
        if key in groups:
            previous, count = groups[key]
            if (previous['pose_sha256'] != row['pose_sha256']
                    or not np.array_equal(previous['keyframe_indices'], row['keyframe_indices'])):
                raise ValueError('physical aliases disagree about pose or keyframes')
            groups[key] = (previous, count + 1)
        else:
            groups[key] = (row, 1)
    deltas, trajectories, dw, tw = [], [], [], []
    for key in sorted(groups):
        row, weight = groups[key]
        payload = Path(row['pose_path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['pose_sha256']:
            raise ValueError('codebook pose checksum differs')
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            p, q = arrays['positions'], arrays['quaternions_xyzw']
            if not np.array_equal(arrays['state_index'], np.arange(len(p))):
                raise ValueError('codebook pose indices differ')
        for anchor in range(len(row['keyframe_indices']) - 1):
            window = window_geometry(p, q, row['keyframe_indices'], anchor, max_future=horizon)
            actions = window['actions'][window['valid']]
            delta = np.diff(np.concatenate([np.zeros((1, 2), dtype=np.float32), actions[:, :2]]), axis=0)
            deltas.append(delta); dw.append(np.full(len(delta), weight, dtype=np.float64))
            if len(actions) == horizon:
                trajectories.append(actions); tw.append(weight)
    if not trajectories:
        raise ValueError('no complete training trajectory for the trajectory codebook')
    return dict(deltas=np.concatenate(deltas), trajectories=np.stack(trajectories),
                delta_weights=np.concatenate(dw), trajectory_weights=np.asarray(tw, dtype=np.float64))


class PseudoDataset:
    def __init__(self, pair_dataset, token_path, *, expected_sha256):
        path = Path(token_path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError('pseudo token checksum differs')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.shape != (len(pair_dataset), 65) or values.dtype not in (np.dtype('int16'), np.dtype('int32'), np.dtype('int64')):
            raise ValueError('pseudo tokens must have integer shape (N,65)')
        # Validate in bounded chunks; keep the actual cache memory-mapped and read-only.
        for start in range(0, len(values), 4096):
            chunk = values[start:start + 4096]
            eos = chunk == 65
            if np.any(chunk[:, 0] != 64) or np.any(eos.sum(axis=1) != 1):
                raise ValueError('pseudo sequence requires one initial BOS and one EOS')
            ends = eos.argmax(axis=1)
            positions = np.arange(65)[None, :]
            motion = (positions > 0) & (positions < ends[:, None])
            padding = positions > ends[:, None]
            if np.any((chunk[motion] < 0) | (chunk[motion] >= 64)) or np.any(chunk[padding] != 66):
                raise ValueError('pseudo motion or padding tokens are invalid')
        self.pairs, self.tokens = pair_dataset, values
        self._token_path = str(path)
        self._token_shape, self._token_dtype = values.shape, values.dtype.str

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('tokens')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokens = np.load(self._token_path, mmap_mode='r', allow_pickle=False)
        if self.tokens.shape != self._token_shape or self.tokens.dtype.str != self._token_dtype:
            raise ValueError('worker pseudo cache layout differs from admitted cache')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        import torch
        epoch, index = item if isinstance(item, tuple) and len(item) == 2 else (0, item)
        if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0 for v in (epoch, index)):
            raise ValueError('epoch and index must be nonnegative integers')
        if index >= len(self):
            raise IndexError('pseudo dataset index is out of range')
        pair = dict(self.pairs[(0, index)])
        pair['tokens'] = torch.from_numpy(np.asarray(self.tokens[index], dtype=np.int64).copy())
        return pair
