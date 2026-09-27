"""Executable stage admission and orchestration for the original LWM models."""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from .engine import make_adam, save_checkpoint, load_checkpoint
from .trainer import StageObjective, train_epochs
from .batching import EpochBatches
from .dataset import ReplayDataset, PseudoDataset, StratifiedReplayDataset


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'asset SHA-256 mismatch: {path}')
    return expected


def _sampling(config):
    rule = config['sampling']
    if (not isinstance(rule, dict) or rule.get('strategy') != 'physical_stratified_cycle_v1'
            or type(rule.get('anchor_bin_size')) is not int or rule['anchor_bin_size'] < 1):
        raise ValueError('unknown sampling strategy or invalid anchor bin size')
    return rule


def sampling_budget(config, partitions):
    size = _sampling(config)['anchor_bin_size']
    summaries, owners = {}, set()
    for name in ('wm_il', 'rl', 'dev'):
        rows = partitions[name]
        anchors, samples = 0, 0
        for row in rows:
            key = row['pose_key']
            if key in owners or row['split'] != ('dev' if name == 'dev' else 'train'):
                raise ValueError('physical trajectory repeated or assigned to incompatible partitions')
            owners.add(key)
            keys = np.asarray(row['keyframe_indices'])
            if (keys.ndim != 1 or keys.dtype.kind not in 'iu' or len(keys) < 2
                    or keys[0] < 0 or np.any(keys[1:] <= keys[:-1])):
                raise ValueError('invalid physical trajectory keyframes')
            length = len(keys) - 1
            anchors += length
            samples += (length + size - 1) // size
        summaries[name] = dict(physical_trajectories=len(rows), all_anchors=anchors, samples_per_epoch=samples)
    stages = {}
    for stage in ('wm', 'il', 'rl'):
        settings = config[stage]
        for field, minimum in (('global_batch', 1), ('epochs', 1), ('warmup_epochs', 0)):
            if type(settings[field]) is not int or settings[field] < minimum:
                raise ValueError(f'invalid {stage} {field}')
        samples = summaries['rl' if stage == 'rl' else 'wm_il']['samples_per_epoch']
        updates, dropped = divmod(samples, settings['global_batch'])
        if updates == 0:
            raise ValueError('sample budget cannot supply one complete global batch')
        stages[stage] = dict(samples_per_epoch=samples, updates_per_epoch=updates,
                             total_updates=updates*settings['epochs'], warmup_updates=updates*settings['warmup_epochs'],
                             dropped_samples_per_epoch=dropped)
    return {'partitions': summaries, 'stages': stages}


def pseudo_identity(config, admission):
    identity = dict(wm=admission['predecessors']['wm']['sha256'],
                **{name: admission['hashes'][name] for name in
                   ('wm_il', 'action_centers', 'trajectory_centers')}, seed=config['seed'])
    if 'sampling' in config:
        identity['sampling'] = dict(_sampling(config))
    return identity


def preflight(config, stage):
    if stage not in ('wm', 'pseudo', 'il', 'rl'):
        raise ValueError('unknown LWM stage')
    books, output = Path(config['codebooks']), Path(config['output'])
    status = read_json(books / 'STATUS.json')
    if status['status'] != 'CPU_CODEBOOKS_READY':
        raise ValueError('CPU codebooks are incomplete')
    hashes = {}
    for name in ('wm_il', 'rl', 'dev'):
        hashes[name] = checked(books / (name + '.jsonl'), status['partitions'][name]['sha256'])
    for name in ('action_centers', 'trajectory_centers'):
        hashes[name] = checked(books / (name + '.json'), status['books'][name]['sha256'])
    hashes['croco'] = checked(config['croco'], config['croco_sha256'])
    if 'vision' in config:
        from .vjepa_frontend import verify_vision_assets
        hashes['vjepa21'] = verify_vision_assets(config['vision'])
    result = {'hashes': hashes, 'predecessors': {}, 'pseudo': None}
    if 'sampling' in config:
        if status.get('representation') != 'physical_trajectories_v2':
            raise ValueError('stratified sampling requires admitted physical v2 codebooks')
        partitions = {}
        for name in ('wm_il', 'rl', 'dev'):
            partitions[name] = [json.loads(line) for line in (books / (name + '.jsonl')).read_text().splitlines()]
        result['data_budget'] = sampling_budget(config, partitions)
        for name, summary in result['data_budget']['partitions'].items():
            recorded = status['partitions'][name]
            if (recorded['rows'] != summary['physical_trajectories']
                    or recorded['physical_trajectories'] != summary['physical_trajectories']
                    or recorded['windows'] != summary['all_anchors']):
                raise ValueError('physical dataset counts differ from the admitted status')
    def admit_pseudo():
        ready = read_json(output / 'pseudo' / 'READY.json')
        if ready['status'] != 'READY' or ready['identity'] != pseudo_identity(config, result):
            raise ValueError('pseudo cache does not match this completed training chain')
        if ready['tokens'] != 'tokens.npy':
            raise ValueError('unexpected pseudo token file')
        checked(output / 'pseudo' / ready['tokens'], ready['sha256'])
        result['pseudo'] = ready
    for predecessor in (() if stage == 'wm' else ('wm', 'il') if stage == 'rl' else ('wm',)):
        if predecessor == 'il':
            admit_pseudo()
        directory = output / predecessor
        receipt = read_json(directory / 'COMPLETE.json')
        if receipt['status'] != 'COMPLETE' or receipt['scientific']['stage'] != predecessor:
            raise ValueError('a completed stage is required; smoke weights are not admitted')
        if receipt['scientific'] != scientific_identity(config, predecessor, result):
            raise ValueError('completed predecessor scientific identity differs from this run')
        if (receipt['progress']['epoch'] != config[predecessor]['epochs']
                or receipt['progress']['sampler_offset'] != 0):
            raise ValueError('predecessor receipt describes an incomplete epoch budget')
        path = directory / receipt['checkpoint']
        if path.resolve().parent != directory.resolve():
            raise ValueError('checkpoint must belong to its stage directory')
        checked(path, receipt['sha256'])
        result['predecessors'][predecessor] = dict(path=str(path), sha256=receipt['sha256'],
                                                  scientific=receipt['scientific'])
    if stage == 'il':
        admit_pseudo()
    return result


def construct_model(config, kind, *, initialize=False):
    from lwm.world_model import LatentWorldModel, BACKBONE_KWARGS
    from lwm.policy import ARPlusPolicy
    from .initialization import initialize_croco
    if kind not in ('wm', 'policy'):
        raise ValueError('unknown model kind')
    model = (LatentWorldModel() if kind == 'wm' else
             ARPlusPolicy(**({'feats_seq_len': 576} if 'vision' in config else {})))
    if initialize:
        initialize_croco(model, config['croco'], config['croco_sha256'], BACKBONE_KWARGS)
    if 'vision' in config:
        from .vjepa_frontend import install_frontend, load_frozen_encoder
        install_frontend(model, load_frozen_encoder(config['vision']))
    return model


def build_model(config, stage, admission, device):
    from lwm.tokenizer import ActionTokenizer
    model = construct_model(config, 'wm' if stage == 'wm' else 'policy',
                            initialize=stage in ('wm', 'il'))
    if stage == 'rl':
        load_completed(model, admission['predecessors']['il'], config['il']['epochs'])
    model.to(device)
    kwargs = {}
    if stage == 'wm':
        kwargs['epsilon_m'] = config['wm']['epsilon_m']
    elif stage == 'rl':
        wm, reference = construct_model(config, 'wm'), construct_model(config, 'policy')
        load_completed(wm, admission['predecessors']['wm'], config['wm']['epochs'])
        load_completed(reference, admission['predecessors']['il'], config['il']['epochs'])
        kwargs.update(wm=wm.to(device).eval(), reference=reference.to(device).eval(),
                      tokenizer=ActionTokenizer(Path(config['codebooks']) / 'action_centers.json'))
        kwargs.update({name: config['rl'][name] for name in
                       ('num_sample', 'temperature', 'beta', 'clip_epsilon')})
    return model, StageObjective(stage, model, **kwargs)


def load_completed(model, predecessor, epochs):
    payload = torch.load(predecessor['path'], map_location='cpu', weights_only=True)
    if (payload.get('format') != 'LWM_STREAM_LOCAL_STATE_V1'
            or payload['identity']['scientific'] != predecessor['scientific']
            or payload['progress']['training']['epoch'] != epochs
            or payload['progress']['training']['sampler_offset'] != 0):
        raise ValueError('predecessor payload does not describe a completed matching stage')
    state, current = payload['model'], model.state_dict()
    if state.keys() != current.keys() or any(
            tensor.shape != current[name].shape or tensor.dtype != current[name].dtype
            or not torch.isfinite(tensor).all() for name, tensor in state.items()):
        raise ValueError('predecessor model ABI or finite-value check failed')
    model.load_state_dict(state, strict=True)


def build_dataset(config, stage, admission):
    partition = 'rl' if stage == 'rl' else 'wm_il'
    with open(Path(config['codebooks']) / (partition + '.jsonl')) as stream:
        rows = [json.loads(line) for line in stream]
    dataset = ReplayDataset(rows, mode='wm' if stage == 'wm' else 'pair', seed=config['seed'],
                            num_candidates=config['wm']['num_candidates'] if stage == 'wm' else 1)
    if 'vision' in config:
        from .vjepa_frontend import VJEPAImageTransform
        dataset.transform = VJEPAImageTransform(config['vision']['source_root'])
    if 'sampling' in config:
        dataset = StratifiedReplayDataset(dataset, anchor_bin_size=_sampling(config)['anchor_bin_size'], seed=config['seed'])
    if stage == 'il':
        dataset = PseudoDataset(dataset, Path(config['output']) / 'pseudo' / 'tokens.npy',
                                expected_sha256=admission['pseudo']['sha256'])
    return dataset


def scientific_identity(config, stage, admission):
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in (source / 'official', source / 'lwm_stream'):
        for path in sorted(directory.rglob('*.py')):
            digest.update(str(path.relative_to(source)).encode())
            digest.update(path.read_bytes())
    identity = {'stage': stage, 'identity': config['identity'], 'settings': config[stage],
            'adam': config['adam'], 'seed': config['seed'], 'hashes': admission['hashes'],
            'source_sha256': digest.hexdigest(),
            'predecessors': {name: value['sha256'] for name, value in admission['predecessors'].items()},
            'pseudo_sha256': admission['pseudo']['sha256'] if stage == 'il' else None}
    if 'sampling' in config:
        identity['sampling'] = dict(_sampling(config))
    if 'vision' in config:
        from .vjepa_frontend import vision_identity
        identity['vision'] = vision_identity(config['vision'])
    return identity


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rank_zero(function, rank, world_size):
    message = [None]
    if rank == 0:
        try:
            message[0] = {'result': function()}
        except Exception as error:
            message[0] = {'error': f'{type(error).__name__}: {error}'}
    if world_size > 1:
        dist.broadcast_object_list(message, src=0)
    if 'error' in message[0]:
        raise RuntimeError(message[0]['error'])
    return message[0]['result']


def rng_state(device):
    state = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'], device)


class SmokeStop(Exception):
    pass


def build_pseudo_model(config, admission, device):
    from lwm.tokenizer import ActionTokenizer
    model = construct_model(config, 'wm')
    load_completed(model, admission['predecessors']['wm'], config['wm']['epochs'])
    books = Path(config['codebooks'])
    candidates = torch.tensor(read_json(books / 'trajectory_centers.json'), dtype=torch.float32, device=device)
    return model.to(device).eval(), ActionTokenizer(books / 'action_centers.json'), candidates


def run_pseudo(config, admission, device, *, rank=0, world_size=1):
    from .stage_data import pseudo_labels
    device = torch.device(device)
    directory = Path(config['output']) / 'pseudo'
    identity = pseudo_identity(config, admission)
    def prepare():
        if directory.exists():
            if not (directory / 'READY.json').is_file():
                raise ValueError('incomplete pseudo directory: use a fresh output, do not admit partial tokens')
            ready = read_json(directory / 'READY.json')
            if ready['status'] != 'READY' or ready['identity'] != identity or ready['tokens'] != 'tokens.npy':
                raise ValueError('existing pseudo cache belongs to a different training chain')
            checked(directory / 'tokens.npy', ready['sha256'])
            return ready
        directory.mkdir(parents=True, exist_ok=False)
        return None
    ready = rank_zero(prepare, rank, world_size)
    if ready is not None:
        return ready
    dataset = build_dataset(config, 'pseudo', admission)
    count = len(dataset)
    if count < 1:
        raise ValueError('pseudo dataset is empty')
    path = directory / 'tokens.npy'
    def allocate():
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.int16, shape=(count, 65))
        array[:] = -1
        array.flush()
    rank_zero(allocate, rank, world_size)
    model, tokenizer, candidates = build_pseudo_model(config, admission, device)
    seed = config['seed'] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    batch_size = config['pseudo']['batch_size']
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('pseudo batch_size must be a positive integer')
    indices = list(range(rank, count, world_size))
    sampler = [[(0, index) for index in indices[start:start+batch_size]]
               for start in range(0, len(indices), batch_size)]
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=config['workers'],
                                        generator=torch.Generator().manual_seed(seed),
                                        multiprocessing_context='spawn' if config['workers'] else None)
    array = np.lib.format.open_memmap(path, mode='r+')
    written = 0
    for batch in loader:
        tokens = pseudo_labels(model, tokenizer, batch['now'].to(device), batch['goal'].to(device), candidates)['tokens']
        selected = indices[written:written+len(tokens)]
        array[selected] = tokens.cpu().numpy().astype(np.int16)
        written += len(tokens)
    array.flush()
    del array
    counts = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(counts, written)
    else:
        counts[0] = written
    def finish():
        if counts != [len(range(r, count, world_size)) for r in range(world_size)]:
            raise ValueError('pseudo rank coverage is incomplete')
        digest = sha256(path)
        PseudoDataset(dataset, path, expected_sha256=digest)
        receipt = {'status': 'READY', 'tokens': 'tokens.npy', 'sha256': digest, 'identity': identity,
                   'shape': [count, 65], 'generation': {'world_size': world_size, 'batch_size': batch_size,
                                                      'seed': config['seed'], 'rank_counts': counts}}
        write_json(directory / 'READY.json', receipt)
        return receipt
    return rank_zero(finish, rank, world_size)


def run_training(config, stage, admission, device, *, rank=0, world_size=1,
                 resume=None, smoke_updates=None):
    device = torch.device(device)
    if smoke_updates is not None and (isinstance(smoke_updates, bool) or smoke_updates < 1):
        raise ValueError('smoke_updates must be positive')
    settings = config[stage]
    directory = Path(config['output']) / stage
    def prepare():
        if resume is None:
            directory.mkdir(parents=True, exist_ok=False)
        elif not directory.is_dir() or Path(resume).resolve().parent != directory.resolve():
            raise ValueError('resume must use a checkpoint in the existing stage directory')
        if (directory / 'COMPLETE.json').exists():
            raise ValueError('stage already complete')
    rank_zero(prepare, rank, world_size)
    seed = config['seed'] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model, objective = build_model(config, stage, admission, device)
    dataset = build_dataset(config, stage, admission)
    batches = EpochBatches(dataset, global_batch=settings['global_batch'], micro_batch=config['micro_batch'],
                           rank=rank, world_size=world_size, seed=config['seed'], workers=config['workers'],
                           device=device)
    if 'sampling' in config:
        budget = admission['data_budget']['stages'][stage]
        if len(dataset) != budget['samples_per_epoch'] or batches.total_updates != budget['updates_per_epoch']:
            raise ValueError('actual loader budget differs from admitted sampling budget')
        rank_zero(lambda: write_json(directory / 'DATA_BUDGET.json', admission['data_budget']), rank, world_size)
    optimizer = make_adam(model, lr=settings['lr'], **config['adam'])
    identity = {'scientific': scientific_identity(config, stage, admission),
                'execution': {'world_size': world_size, 'micro_batch': config['micro_batch'],
                              'workers': config['workers'], 'device_type': device.type}}
    training_model = objective.training_model
    if world_size > 1:
        training_model = torch.nn.parallel.DistributedDataParallel(
            training_model, device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=True)
    progress = None
    if resume is not None:
        restored = load_checkpoint(resume, model, optimizer, identity=identity)
        if len(restored['rank_rng']) != world_size:
            raise ValueError('rank RNG count differs from resume world size')
        restore_rng(restored['rank_rng'][rank], device)
        progress = restored['training']
    started_updates = progress['update'] if progress else 0
    last_time = time.monotonic()
    saved_receipts = {}

    def checkpoint(state):
        local = rng_state(device)
        states = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(states, local)
        else:
            states[0] = local
        def save():
            name = f"update-{state['update']:09d}-epoch-{state['epoch']:04d}.pt"
            path = directory / name
            if name in saved_receipts:
                return saved_receipts[name]
            save_checkpoint(path, model, optimizer, identity=identity,
                            progress={'training': state, 'rank_rng': states})
            receipt = {'checkpoint': name, 'sha256': sha256(path), 'scientific': identity['scientific']}
            write_json(directory / 'LATEST.json', receipt)
            saved_receipts[name] = receipt
            return receipt
        return rank_zero(save, rank, world_size)

    def update(state, metrics):
        nonlocal last_time
        now = time.monotonic()
        if rank == 0 and (state['update'] == 1 or state['update'] % config['log_updates'] == 0
                          or (smoke_updates is not None and state['update'] - started_updates >= smoke_updates)):
            with (directory / 'train.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(**state, **metrics, seconds=now-last_time)) + '\n')
                stream.flush()
        last_time = now
        if smoke_updates is not None and state['update'] - started_updates >= smoke_updates:
            receipt = checkpoint(state)
            rank_zero(lambda: write_json(directory / 'SMOKE_PARTIAL.json',
                                        dict(receipt, status='SMOKE_PARTIAL', progress=state)), rank, world_size)
            raise SmokeStop
        if state['update'] % config['checkpoint_updates'] == 0:
            checkpoint(state)
    try:
        state = train_epochs(objective, optimizer, batches, epochs=settings['epochs'], peak_lr=settings['lr'],
                             warmup_updates=settings['warmup_epochs'] * batches.total_updates,
                             progress=progress, training_model=training_model, on_update=update,
                             on_epoch=checkpoint)
    except SmokeStop:
        return {'status': 'SMOKE_PARTIAL'}
    receipt = checkpoint(state)
    rank_zero(lambda: write_json(directory / 'COMPLETE.json', dict(receipt, status='COMPLETE', progress=state)),
              rank, world_size)
    return dict(receipt, status='COMPLETE')
