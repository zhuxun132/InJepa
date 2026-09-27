"""LWM optimizer updates; native state keys and explicit numerical controls."""
from collections.abc import Sequence
from contextlib import nullcontext
import math
import numbers
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .reinforcement import _scalar


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _owners(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    ids = [id(p) for p in parameters]
    if not ids or len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("optimizer must own every trainable exactly once and no other parameter")
    return parameters


def make_adam(model, *, lr, betas, eps, weight_decay):
    _scalar(lr, 'lr')
    _scalar(eps, 'eps')
    _scalar(weight_decay, 'weight_decay', inclusive=True)
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError('Adam requires two betas')
    for beta in betas:
        _scalar(beta, 'beta', upper=1., inclusive=True)
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)


def set_warmup_lr(optimizer, *, peak_lr, warmup_updates, update_index):
    _scalar(peak_lr, 'peak_lr')
    _integer(warmup_updates, 'warmup_updates')
    _integer(update_index, 'update_index')
    lr = peak_lr * (min((update_index + 1) / warmup_updates, 1.) if warmup_updates else 1.)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


def optimizer_update(model, optimizer, batches, loss_terms):
    parameters = _owners(model, optimizer)
    if not isinstance(batches, Sequence) or not len(batches):
        raise ValueError('each rank requires at least one microbatch')
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and not isinstance(model, DDP):
        raise ValueError('an initialized distributed job requires the DDP training wrapper')
    if isinstance(model, DDP) and not model.find_unused_parameters:
        raise ValueError('LWM DDP requires find_unused_parameters for original mask_token')
    group = model.process_group if distributed else None
    world = dist.get_world_size(group) if distributed else 1
    device = parameters[0].device
    total = torch.zeros((), device=device, dtype=torch.float64)
    count = torch.zeros((), device=device, dtype=torch.int64)
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(batches):
        context = model.no_sync() if distributed and index + 1 < len(batches) else nullcontext()
        # no_sync must enclose both the forward and backward.
        with context:
            numerator, denominator = loss_terms(model, batch)
            if (not isinstance(numerator, torch.Tensor) or numerator.ndim != 0
                    or not numerator.requires_grad or not numerator.is_floating_point()
                    or numerator.device != device or not torch.isfinite(numerator)):
                raise ValueError('loss must be a finite differentiable scalar on model device')
            if (not isinstance(denominator, torch.Tensor) or denominator.ndim != 0
                    or denominator.dtype != torch.int64 or denominator.device != device
                    or denominator.item() < 0):
                raise ValueError('loss count must be nonnegative scalar int64 on model device')
            total.add_(numerator.detach().double())
            count.add_(denominator)
            numerator.backward()
    if distributed:
        dist.all_reduce(total, group=group)
        dist.all_reduce(count, group=group)
    if count.item() <= 0:
        raise ValueError('global loss count is zero; no optimizer update')
    finite = torch.tensor(int(bool(torch.isfinite(total))), device=device, dtype=torch.int32)
    squared_norm = torch.zeros((), device=device, dtype=torch.float64)
    factor = world / count.item()
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(factor)
            finite.mul_(torch.isfinite(parameter.grad).all().to(torch.int32))
            squared_norm.add_(parameter.grad.double().square().sum())
    finite.mul_(torch.isfinite(squared_norm).to(torch.int32))
    if distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
    if not finite.item():
        raise FloatingPointError('nonfinite loss or gradients; no optimizer update')
    optimizer.step()
    return {'loss': (total / count).item(), 'global_count': count.item(),
            'gradient_norm': squared_norm.sqrt().item()}


def _parameter_names(model, optimizer):
    _owners(model, optimizer)
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def save_checkpoint(path, model, optimizer, *, identity, progress):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    names = _parameter_names(model, optimizer)
    numpy_rng = np.random.get_state()
    payload = {'format': 'LWM_STREAM_LOCAL_STATE_V1', 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'parameter_names': names,
               'identity': identity, 'progress': progress,
               'torch_cpu_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
               'numpy_rng': [numpy_rng[0], numpy_rng[1].tolist(), int(numpy_rng[2]),
                             int(numpy_rng[3]), float(numpy_rng[4])],
               'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Publish atomically without overwriting an existing checkpoint.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path, model, optimizer, *, identity):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    names = _parameter_names(model, optimizer)
    if (not isinstance(payload, dict) or payload.get('format') != 'LWM_STREAM_LOCAL_STATE_V1'
            or payload.get('identity') != identity or payload.get('parameter_names') != names):
        raise ValueError('checkpoint format, identity or optimizer parameter order mismatch')
    current = model.state_dict()
    state = payload.get('model')
    if not isinstance(state, dict) or state.keys() != current.keys():
        raise ValueError('checkpoint model keys mismatch')
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != current[name].shape
                or tensor.dtype != current[name].dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f'checkpoint model shape/dtype/value mismatch: {name}')
    optimizer_state = payload.get('optimizer', {})
    if (not isinstance(optimizer_state, dict) or 'state' not in optimizer_state
            or 'param_groups' not in optimizer_state
            or [len(g['params']) for g in optimizer_state['param_groups']] != [len(g) for g in names]):
        raise ValueError('checkpoint optimizer groups mismatch')
    cuda_rng = payload.get('cuda_rng')
    if cuda_rng is not None and (not torch.cuda.is_initialized() or len(cuda_rng) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG restore requires matching initialized visible devices')
    numpy_rng = payload['numpy_rng']
    restored_numpy = (numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32),
                      numpy_rng[2], numpy_rng[3], numpy_rng[4])
    # Validate random states in independent generators before mutating model/RNG.
    random.Random().setstate(payload['python_rng'])
    np.random.RandomState().set_state(restored_numpy)
    torch.Generator().set_state(payload['torch_cpu_rng'])
    model.load_state_dict(state, strict=True, assign=False)
    optimizer.load_state_dict(optimizer_state)
    random.setstate(payload['python_rng'])
    np.random.set_state(restored_numpy)
    torch.set_rng_state(payload['torch_cpu_rng'])
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    return payload['progress']
