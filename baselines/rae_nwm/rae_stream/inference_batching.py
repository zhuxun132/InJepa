"""Memory-only batching of the unchanged deterministic official image encoder."""
import numbers
import torch


def install_endpoint_only_sampler(sampler):
    """Release unused ODE states at the official planner's endpoint-only seam.

    planning_eval's only consumer uses xs[-1]. Without a copy that tensor is
    a view holding the entire 50-state solver buffer across future horizons.
    The original solver, its settings and its initial noise are unchanged.
    """
    if getattr(sampler, '_rae_stream_endpoint_only', False):
        raise RuntimeError('endpoint-only sampler already installed')
    original_factory = sampler.sample_ode

    def sample_ode(**settings):
        original_sample = original_factory(**settings)
        def sample(*args, **kwargs):
            if torch.is_grad_enabled():
                raise RuntimeError('endpoint-only sampling requires no_grad')
            states = original_sample(*args, **kwargs)
            if not torch.is_tensor(states) or states.ndim < 2 or len(states) == 0:
                raise TypeError('official ODE sampler must return a nonempty time-first tensor')
            endpoint = states[-1].clone()
            return (endpoint,)
        return sample

    sampler.sample_ode = sample_ode
    sampler._rae_stream_endpoint_only = True


def make_batched_model(model, *, batch_size):
    """Slice independent model evaluations inside the unchanged ODE solver.

    The solver still draws its full noise batch and integrates every candidate
    using the original time grid. This wrapper introduces no random draws.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
        raise ValueError('model batch_size must be a positive integer')

    def forward(x, t, **kwargs):
        if torch.is_grad_enabled() or any(module.training for module in model.modules()):
            raise RuntimeError('model batching requires no_grad and all modules in eval mode')
        if not torch.is_tensor(x) or x.ndim < 1 or len(x) == 0:
            raise ValueError('model requires a nonempty state batch')
        count = len(x)
        def validate(value, name, scalar=False):
            if not torch.is_tensor(value):
                raise TypeError(f'{name} must be a tensor')
            if scalar and value.ndim == 0:
                return
            if value.ndim == 0 or len(value) != count:
                raise ValueError(f'{name} batch does not match states')
        validate(t, 'time', scalar=True)
        for key, value in kwargs.items():
            validate(value, key)
        if count <= batch_size:
            return model(x, t, **kwargs)
        results = []
        for start in range(0, count, batch_size):
            end = start + batch_size
            results.append(model(x[start:end], t if t.ndim == 0 else t[start:end],
                                 **{key: value[start:end] for key, value in kwargs.items()}))
        return torch.cat(results, dim=0)
    return forward


def install_encoder_batching(rae, *, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, numbers.Integral) or batch_size <= 0:
        raise ValueError('encoder batch_size must be a positive integer')
    if hasattr(rae, '_rae_stream_encoder_batch_size'):
        raise RuntimeError('encoder batching is already installed')
    original = rae.encode

    def encode(images):
        if torch.is_grad_enabled() or any(module.training for module in rae.modules()):
            raise RuntimeError('encoder batching requires no_grad and all modules in eval mode')
        if not torch.is_tensor(images) or images.ndim < 1 or len(images) == 0:
            raise ValueError('encoder requires a nonempty image batch')
        if len(images) <= batch_size:
            return original(images)
        return torch.cat([original(images[start:start+batch_size])
                          for start in range(0, len(images), batch_size)], dim=0)

    rae.encode = encode
    rae._rae_stream_encoder_batch_size = int(batch_size)
