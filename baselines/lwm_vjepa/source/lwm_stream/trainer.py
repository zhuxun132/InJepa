"""Three LWM objectives and checked progression through complete epochs."""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .training import WorldModelTraining, PolicyTraining, log_distance_targets, masked_mse_terms, cross_entropy_terms
from .reinforcement import PolicyLikelihood, group_advantages, grpo_terms, _scalar
from .stage_data import sample_rewards
from .engine import _integer, set_warmup_lr, optimizer_update


class StageObjective:
    def __init__(self, stage, model, *, wm=None, reference=None, tokenizer=None,
                 epsilon_m=None, num_sample=None, temperature=None, beta=None, clip_epsilon=None):
        if stage not in ('wm', 'il', 'rl'):
            raise ValueError('stage must be wm, il or rl')
        if stage == 'wm':
            _scalar(epsilon_m, 'epsilon_m')
        if stage == 'rl':
            if wm is None or reference is None or tokenizer is None:
                raise ValueError('RL requires separate WM, reference policy and tokenizer')
            sets = [{id(p) for p in m.parameters()} for m in (model, wm, reference)]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
                raise ValueError('trained, world and reference models must not share parameters')
            _integer(num_sample, 'num_sample')
            if num_sample < 2:
                raise ValueError('group advantages require at least two samples')
            _scalar(temperature, 'temperature')
            _scalar(beta, 'beta', inclusive=True)
            _scalar(clip_epsilon, 'clip_epsilon', upper=1.)
        self.stage, self.model = stage, model
        self.wm, self.reference, self.tokenizer = wm, reference, tokenizer
        self.epsilon_m, self.num_sample = epsilon_m, num_sample
        self.temperature, self.beta, self.clip_epsilon = temperature, beta, clip_epsilon
        wrapper = {'wm': WorldModelTraining, 'il': PolicyTraining, 'rl': PolicyLikelihood}[stage]
        self.training_model = wrapper(model)
        self.set_mode()

    def set_mode(self):
        self.training_model.train(self.stage != 'rl')
        if self.stage == 'rl':
            self.wm.eval()
            self.reference.eval()

    def terms(self, training_model, batch):
        actual = training_model.module if isinstance(training_model, DDP) else training_model
        if actual is not self.training_model:
            raise ValueError('training forward must use this objective wrapper or its DDP owner')
        now, goal = batch['now'], batch['goal']
        if self.stage == 'wm':
            actions, mask = batch['actions_m'], batch['valid_mask']
            predicted = training_model(now, goal, actions, mask)
            labels = log_distance_targets(actions, batch['goal_xy_m'], self.epsilon_m, mask)
            return masked_mse_terms(predicted, labels, mask)
        if self.stage == 'il':
            tokens = batch['tokens']
            return cross_entropy_terms(training_model(now, goal, tokens), tokens)
        sampled = sample_rewards(self.model, self.wm, self.tokenizer, now, goal,
                                 num_sample=self.num_sample, temperature=self.temperature)
        advantages = group_advantages(sampled['rewards'])
        tokens = sampled['tokens']
        with torch.no_grad():
            reference = PolicyLikelihood(self.reference)(now, goal, tokens, temperature=self.temperature)
        current = training_model(now, goal, tokens, temperature=self.temperature)
        return grpo_terms(current['log_probs'], reference['log_probs'], tokens, advantages,
                          beta=self.beta, clip_epsilon=self.clip_epsilon)


def train_epochs(objective, optimizer, epoch_batches, *, epochs, peak_lr, warmup_updates,
                 progress=None, training_model=None, on_update=None, on_epoch=None):
    _integer(epochs, 'epochs')
    _integer(warmup_updates, 'warmup_updates')
    _scalar(peak_lr, 'peak_lr')
    state = dict(progress) if progress is not None else {'epoch': 0, 'update': 0, 'sampler_offset': 0}
    if set(state) != {'epoch', 'update', 'sampler_offset'}:
        raise ValueError('progress requires epoch, update and sampler_offset')
    for key, value in state.items():
        _integer(value, key)
    if state['epoch'] > epochs or (state['epoch'] == epochs and state['sampler_offset'] != 0):
        raise ValueError('progress is outside requested epoch range')
    model = objective.training_model if training_model is None else training_model
    while state['epoch'] < epochs:
        objective.set_mode()
        total_updates, batches = epoch_batches(state['epoch'], state['sampler_offset'])
        _integer(total_updates, 'total_updates')
        if total_updates < 1 or state['sampler_offset'] > total_updates:
            raise ValueError('empty epoch or sampler cursor exceeds actual epoch length')
        for microbatches in batches:
            if state['sampler_offset'] >= total_updates:
                raise ValueError('factory yielded more updates than declared; no extra update')
            lr = set_warmup_lr(optimizer, peak_lr=peak_lr, warmup_updates=warmup_updates,
                               update_index=state['update'])
            metrics = optimizer_update(model, optimizer, microbatches, objective.terms)
            state['update'] += 1
            state['sampler_offset'] += 1
            metrics['lr'] = lr
            if on_update is not None:
                on_update(dict(state), metrics)
        if state['sampler_offset'] != total_updates:
            raise ValueError('factory ended before declared epoch length; epoch is incomplete')
        state['epoch'] += 1
        state['sampler_offset'] = 0
        if on_epoch is not None:
            on_epoch(dict(state))
    return state
