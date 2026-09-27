"""Conditional trajectory CVAE with a causal divided full-grid decoder."""

from __future__ import annotations

from j2j.compat import zip_compatible as zip

from dataclasses import dataclass
import math
import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import DecoderBlock, DividedDecoderBlock, FeedForward, _attention_context, run_block
from .initialization import initialize_proposal_, _fqn_seed
from .model import ProposalJEPA, _validate_constructor, _require_positive_integer
from .positions import age_sincos, fixed_2d_sincos
from .sampling import keyed_normal
from .stability import StabilityConfig, BoundedLayerNorm, BoundedMultiheadAttention


@dataclass(frozen=True)
class StochasticProposalOutput:
    tape: Tensor
    log_mass: Tensor
    trajectory_latent: Tensor
    prior_mean: Tensor
    prior_logvar: Tensor
    posterior_mean: Tensor | None = None
    posterior_logvar: Tensor | None = None
    sampling_namespace: str | None = None
    posterior_standardized_delta: Tensor | None = None
    posterior_log_scale_ratio: Tensor | None = None


class AttentionReadout(nn.Module):
    """Learned summary for Gaussian parameters; full decoder memory is retained."""

    def __init__(self, hidden: int, heads: int, ffn: int, dropout: float, backend: str):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, hidden))
        self.memory_norm = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.cross_output_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = FeedForward(hidden, ffn, dropout)
        self.backend = backend
        self.residual_scale = 1.0

    def forward(self, memory: Tensor, valid: Tensor) -> Tensor:
        memory = self.memory_norm(memory.masked_fill(~valid[..., None], 0.0))
        query = self.query.expand(memory.shape[0], -1, -1)
        with _attention_context(self.backend):
            value, _ = self.cross_attn(query, memory, memory, need_weights=False,
                                      key_padding_mask=None if bool(valid.all()) else ~valid)
        value = query + self.residual_scale * self.cross_output_dropout(value)
        return (value + self.residual_scale * self.ffn(self.ffn_norm(value))).squeeze(1)


class GaussianHead(nn.Module):
    def __init__(self, inputs: int, hidden: int, width: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(inputs)
        self.fc = nn.Linear(inputs, hidden)
        self.output = nn.Linear(hidden, 2 * width)

    def forward(self, value: Tensor) -> Tensor:
        return self.output(F.gelu(self.fc(self.input_norm(value)))).float()


class StochasticFutureDecoder(nn.Module):
    def __init__(self, latent_dim: int, grid_side: int, hidden: int, heads: int,
                 ffn_dim: int, depth: int, dropout: float, trajectory_width: int,
                 *, attention_backend: str, memory_norm: bool, activation_checkpointing: bool):
        super().__init__()
        self.type_embedding = nn.Embedding(6, hidden)
        self.register_buffer("spatial_position", fixed_2d_sincos(dim=hidden, grid_side=grid_side))
        self.latent_projection = nn.Linear(trajectory_width, hidden)
        self.latent_biases = nn.ModuleList(nn.Linear(trajectory_width, hidden) for _ in range(depth))
        self.blocks = nn.ModuleList(DividedDecoderBlock(
            hidden, heads, ffn_dim, dropout,
            attention_backend=attention_backend, memory_norm=memory_norm,
        ) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, latent_dim)
        self.activation_checkpointing = activation_checkpointing

    def forward(self, memory: Tensor, memory_valid: Tensor, trajectory_latent: Tensor,
                horizon: int) -> Tensor:
        batch, samples, _ = trajectory_latent.shape
        spatial, hidden = self.spatial_position.shape
        z = trajectory_latent.reshape(batch * samples, -1)
        time = age_sincos(torch.arange(1, horizon + 1, dtype=torch.int64), dim=hidden).to(memory)
        position = self.spatial_position.to(memory)
        query = (self.latent_projection(z)[:, None, None]
                 + time[None, :, None] + position[None, None]
                 + self.type_embedding.weight[5])
        valid = torch.ones(batch * samples, horizon, spatial, dtype=torch.bool, device=memory.device)
        expanded_memory = memory[:, None].expand(-1, samples, -1, -1).reshape(
            batch * samples, memory.shape[1], hidden)
        expanded_valid = memory_valid[:, None].expand(-1, samples, -1).reshape(batch * samples, -1)
        for block, latent_bias in zip(self.blocks, self.latent_biases, strict=True):
            query = query + block.residual_scale * latent_bias(z)[:, None, None]
            query = run_block(block, query, valid, expanded_memory, expanded_valid,
                              activation_checkpointing=self.activation_checkpointing)
        return self.output_proj(self.final_norm(query)).reshape(
            batch, samples, horizon, spatial, -1)


class RecurrentFutureDecoder(nn.Module):
    """One full-grid conditional cell, shared over a disposable proposed future."""

    def __init__(self, latent_dim: int, grid_side: int, hidden: int, heads: int,
                 ffn_dim: int, depth: int, dropout: float, trajectory_width: int,
                 *, attention_backend: str, memory_norm: bool, activation_checkpointing: bool):
        super().__init__()
        self.type_embedding = nn.Embedding(6, hidden)
        self.register_buffer("spatial_position", fixed_2d_sincos(dim=hidden, grid_side=grid_side))
        self.state_proj = nn.Linear(latent_dim, hidden)
        self.latent_projection = nn.Linear(trajectory_width, hidden)
        self.latent_biases = nn.ModuleList(nn.Linear(trajectory_width, hidden) for _ in range(depth))
        self.blocks = nn.ModuleList(DecoderBlock(
            hidden, heads, ffn_dim, dropout, attention_backend=attention_backend,
            memory_norm=memory_norm) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Linear(hidden, latent_dim)
        self.activation_checkpointing = activation_checkpointing

    def step(self, previous_grid: Tensor, memory: Tensor, memory_valid: Tensor,
             trajectory_latent: Tensor) -> Tensor:
        query = (self.state_proj(previous_grid) + self.spatial_position.to(memory)[None]
                 + self.latent_projection(trajectory_latent)[:, None]
                 + self.type_embedding.weight[5])
        valid = torch.ones(query.shape[:-1], dtype=torch.bool, device=query.device)
        for block, latent_bias in zip(self.blocks, self.latent_biases, strict=True):
            query = query + block.residual_scale * latent_bias(trajectory_latent)[:, None]
            query = run_block(block, query, valid, memory, memory_valid,
                              activation_checkpointing=self.activation_checkpointing)
        return self.output_proj(self.final_norm(query))

    def forward(self, memory: Tensor, memory_valid: Tensor, trajectory_latent: Tensor,
                horizon: int, *, current_grid: Tensor) -> Tensor:
        _require_positive_integer("horizon", horizon)
        if memory.ndim != 3 or trajectory_latent.ndim != 3:
            raise ValueError("recurrent memory/latent must have ranks three")
        batch, samples, _ = trajectory_latent.shape
        spatial = self.spatial_position.shape[0]
        if min(batch, samples, memory.shape[1]) < 1 or memory.shape[0] != batch:
            raise ValueError("recurrent batch/sample/memory axes must agree and be nonempty")
        if current_grid.shape != (batch, spatial, self.state_proj.in_features):
            raise ValueError("current grid must match recurrent full-grid coordinates")
        if memory_valid.shape != memory.shape[:-1] or memory_valid.dtype != torch.bool:
            raise ValueError("memory validity must be boolean [batch,tokens]")
        if not bool(memory_valid.any(-1).all()):
            raise ValueError("each recurrent row needs valid factual/goal memory")
        if trajectory_latent.shape[-1] != self.latent_projection.in_features:
            raise ValueError("recurrent latent width mismatch")
        z = trajectory_latent.reshape(batch * samples, -1)
        expanded_memory = memory[:, None].expand(-1, samples, -1, -1).reshape(
            batch * samples, memory.shape[1], memory.shape[-1])
        expanded_valid = memory_valid[:, None].expand(-1, samples, -1).reshape(batch * samples, -1)
        previous = current_grid[:, None].expand(-1, samples, -1, -1).reshape(batch * samples, spatial, -1)
        future = []
        for _ in range(horizon):
            previous = self.step(previous, expanded_memory, expanded_valid, z)
            future.append(previous)
        return torch.stack(future, dim=1).reshape(batch, samples, horizon, spatial, -1)


class StochasticProposalJEPA(ProposalJEPA):
    """Prior-only deployment and posterior-only factual recognition interfaces.

    Reuses the existing strict factual record contract, with no selector or
    learned mode/horizon table. K/H defaults are runtime budgets, not capacity.
    """

    is_stochastic = True

    def __init__(self, *, latent_dim: int = 768, grid_side: int = 24,
                 hidden_dim: int = 768, heads: int = 16, ffn_dim: int = 2048,
                 future_depth: int = 6, dropout: float = 0.1, modes: int = 4,
                 horizon: int = 4, global_seed: int, trajectory_latent_dim: int = 128,
                 sigma_epsilon: float = 1e-4, posterior_residual_init_std: float = 1e-3,
                 attention_backend: str = "math", memory_norm: bool = True,
                 activation_checkpointing: bool = False, stability: dict | None = None,
                 future_architecture: str = "divided", training_horizon: int | None = None):
        _validate_constructor(latent_dim, grid_side, hidden_dim, heads, ffn_dim, 1,
                              future_depth, dropout, modes, horizon, global_seed)
        _require_positive_integer("trajectory_latent_dim", trajectory_latent_dim)
        if not 0 < sigma_epsilon < 1 or not math.isfinite(posterior_residual_init_std) or posterior_residual_init_std <= 0:
            raise ValueError("Gaussian scale floor and residual initialization must be positive")
        if type(activation_checkpointing) is not bool:
            raise TypeError("activation_checkpointing must be bool")
        if future_architecture not in {"divided", "recurrent"}:
            raise ValueError("unknown stochastic future architecture")
        if training_horizon is not None:
            _require_positive_integer("training_horizon", training_horizon)
        if future_architecture == "recurrent" and training_horizon != 1:
            raise ValueError("recurrent Q requires explicit one-step training_horizon")
        if future_architecture == "divided" and training_horizon is not None:
            raise ValueError("divided Q retains its existing trained horizon")
        nn.Module.__init__(self)
        self.is_recurrent = future_architecture == "recurrent"
        self.stability = None if stability is None else StabilityConfig.from_mapping(stability)
        self.global_seed = global_seed
        self.default_samples, self.default_horizon = modes, horizon
        self.trained_horizon = horizon if training_horizon is None else training_horizon
        self.trajectory_latent_dim = trajectory_latent_dim
        self.sigma_epsilon = float(sigma_epsilon)
        self.sampling_namespace = "train/epoch-1"
        self.selector = None
        with torch.random.fork_rng(devices=[]):
            self.input_proj = nn.Linear(latent_dim, hidden_dim)
            decoder_type = RecurrentFutureDecoder if self.is_recurrent else StochasticFutureDecoder
            self.future = decoder_type(
                latent_dim, grid_side, hidden_dim, heads, ffn_dim, future_depth,
                dropout, trajectory_latent_dim, attention_backend=attention_backend,
                memory_norm=memory_norm, activation_checkpointing=activation_checkpointing)
            self.condition_readout = AttentionReadout(hidden_dim, heads, ffn_dim, dropout, attention_backend)
            self.future_readout = AttentionReadout(hidden_dim, heads, ffn_dim, dropout, attention_backend)
            self.prior_head = GaussianHead(hidden_dim, hidden_dim, trajectory_latent_dim)
            self.posterior_head = GaussianHead(2 * hidden_dim, hidden_dim, trajectory_latent_dim)
            initialize_proposal_(self, global_seed=global_seed)
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    module.reset_parameters()
            with torch.no_grad():
                # Four residual branches per divided block, instead of three.
                for index, block in enumerate(self.future.blocks, 1):
                    if hasattr(block, "temporal_attn"):
                        block.temporal_attn.out_proj.weight.div_(math.sqrt(4 * index))
                    for weight in (block.self_attn.out_proj.weight, block.cross_attn.out_proj.weight,
                                   block.ffn.fc2.weight):
                        weight.mul_(math.sqrt(3 / 4))
                self.prior_head.output.weight.zero_()
                self.prior_head.output.bias.zero_()
                self.prior_head.output.bias[trajectory_latent_dim:].fill_(math.log(math.expm1(1 - self.sigma_epsilon)))
                generator = torch.Generator(device="cpu").manual_seed(_fqn_seed(
                    global_seed, "j2j.stochastic.posterior_head.output.weight"))
                nn.init.normal_(self.posterior_head.output.weight, std=posterior_residual_init_std, generator=generator)
                self.posterior_head.output.bias.zero_()
            if self.stability is not None:
                self._enable_stability()

    def _enable_stability(self):
        config = self.stability
        def replace_modules(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.LayerNorm):
                    bounded = BoundedLayerNorm(child.normalized_shape, eps=config.ln_eps,
                                               gain_bound=config.ln_gain_bound)
                    parent.add_module(name, bounded)
                elif isinstance(child, nn.MultiheadAttention):
                    bounded = BoundedMultiheadAttention(child.embed_dim, child.num_heads,
                        dropout=child.dropout, batch_first=True, stability=config)
                    bounded.load_state_dict(child.state_dict(), strict=False)
                    parent.add_module(name, bounded)
                else:
                    replace_modules(child)
        replace_modules(self)
        with torch.no_grad():
            # Undo both existing initialization divisions; the same factor is
            # now applied every forward, not merely at constructor time.
            for index, block in enumerate(self.future.blocks, 1):
                block.residual_scale = 1 / math.sqrt(4 * index)
                block.memory_norm_epsilon = config.ln_eps
                attentions = [block.self_attn, block.cross_attn]
                if hasattr(block, "temporal_attn"):
                    attentions.append(block.temporal_attn)
                for attention in attentions:
                    attention.out_proj.weight.mul_(math.sqrt(4 * index))
                block.ffn.fc2.weight.mul_(math.sqrt(4 * index))
            for readout in (self.condition_readout, self.future_readout):
                readout.residual_scale = 1 / math.sqrt(2)
            self.prior_head.output.bias.zero_()

    def _horizon(self, horizon: int) -> int:
        _require_positive_integer("horizon", horizon)
        if horizon > self.trained_horizon:
            warnings.warn("Requested horizon exceeds the trained horizon; extrapolation is unvalidated.", UserWarning)
        return horizon

    def _memory(self, record_grid, incoming_action_embedding, record_age,
                record_type, record_valid, goal_grid):
        batch, records, spatial, latent = self._validate_records(
            record_grid, incoming_action_embedding, record_age, record_type, record_valid)
        if goal_grid.shape != (batch, spatial, latent) or goal_grid.dtype != torch.float32:
            raise ValueError("goal must match the factual float32 spatial grid")
        if goal_grid.device != record_grid.device or not bool(torch.isfinite(goal_grid).all()):
            raise ValueError("goal must be finite on the model device")
        grid, action, age, kind = self._sanitize_records(
            record_grid, incoming_action_embedding, record_age, record_type, record_valid)
        visual = self.input_proj(grid)
        action = self.input_proj(action)
        goal = self.input_proj(goal_grid.detach())
        hidden = visual.shape[-1]
        ages = age_sincos(age, dim=hidden).to(visual)
        position = self.future.spatial_position.to(visual)
        current = kind == 2
        visual_kind = torch.where(current, 2, 0)
        action_kind = torch.where(current, 3, 1)
        visual = visual + ages[:, :, None] + position[None, None] + self.future.type_embedding(visual_kind)[:, :, None]
        action = action + ages + self.future.type_embedding(action_kind)
        visual = visual.masked_fill(~record_valid[:, :, None, None], 0)
        action = action.masked_fill(~record_valid[:, :, None], 0)
        fact_memory = torch.cat((visual, action[:, :, None]), dim=2).reshape(batch, -1, hidden)
        goal = goal + position[None] + self.future.type_embedding.weight[4]
        memory = torch.cat((fact_memory, goal), dim=1)
        valid = torch.cat((record_valid[:, :, None].expand(-1, -1, spatial + 1).reshape(batch, -1),
                           torch.ones(batch, spatial, dtype=torch.bool, device=record_grid.device)), dim=1)
        return memory, valid

    def _gaussian(self, raw: Tensor) -> tuple[Tensor, Tensor]:
        if self.stability is not None and not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("raw prior Gaussian parameters are nonfinite")
        with torch.autocast(device_type=raw.device.type, enabled=False):
            mean, raw_scale = raw.float().chunk(2, dim=-1)
            if self.stability is None:
                logvar = 2 * (F.softplus(raw_scale) + self.sigma_epsilon).log()
            else:
                config = self.stability
                mean = config.prior_mean_bound * torch.tanh(mean / config.prior_mean_bound)
                logvar = 2 * config.prior_log_scale_bound * torch.tanh(raw_scale / config.prior_log_scale_bound)
        if not bool(torch.isfinite(mean).all() & torch.isfinite(logvar).all()):
            raise ValueError("conditional Gaussian parameters are nonfinite")
        return mean, logvar

    def decode(self, *facts, trajectory_latent: Tensor, horizon: int) -> Tensor:
        horizon = self._horizon(horizon)
        memory, valid = self._memory(*facts)
        self._validate_latent(trajectory_latent, memory.shape[0])
        return self._decode_future(memory, valid, trajectory_latent, horizon, facts)

    def _decode_future(self, memory, valid, z, horizon, facts):
        if self.is_recurrent:
            record_grid, _, _, record_type, record_valid, _ = facts
            current = record_grid.detach()[record_valid & (record_type == 2)]
            return self.future(memory, valid, z, horizon, current_grid=current)
        return self.future(memory, valid, z, horizon)

    def _validate_latent(self, value: Tensor, batch: int, samples: int | None = None):
        if value.ndim != 3 or value.shape[0] != batch or value.shape[2] != self.trajectory_latent_dim or value.shape[1] < 1:
            raise ValueError("trajectory latent/noise must have shape [B,K,L]")
        if samples is not None and value.shape[1] != samples:
            raise ValueError("posterior training requires exactly one trajectory sample")
        if not value.is_floating_point() or value.device != self.input_proj.weight.device or not bool(torch.isfinite(value).all()):
            raise ValueError("trajectory latent/noise must be finite floating point on the model device")

    def posterior_forward(self, record_grid, incoming_action_embedding, record_age,
                          record_type, record_valid, goal_grid, future_target,
                          future_label_valid, *, horizon: int, sample_noise: Tensor):
        if self.is_recurrent and horizon != self.trained_horizon:
            raise ValueError("recurrent posterior requires one-step factual targets")
        horizon = self._horizon(horizon)
        memory, valid = self._memory(record_grid, incoming_action_embedding, record_age,
                                     record_type, record_valid, goal_grid)
        batch, _, hidden = memory.shape
        spatial = self.future.spatial_position.shape[0]
        self._validate_latent(sample_noise, batch, samples=1)
        if future_target.shape != (batch, horizon, spatial, self.input_proj.in_features):
            raise ValueError("future target must match [B,H,M,D]")
        if future_label_valid.shape != (batch, horizon) or future_label_valid.dtype != torch.bool:
            raise ValueError("future label validity must be boolean [B,H]")
        if not bool(future_label_valid[:, 0].all()) or bool(((~future_label_valid[:, :-1]) & future_label_valid[:, 1:]).any()):
            raise ValueError("future labels must form a nonempty factual prefix")
        safe_target = future_target.detach().masked_fill(~future_label_valid[:, :, None, None], 0)
        if not bool(torch.isfinite(safe_target).all()):
            raise ValueError("valid future labels must be finite")
        summary = self.condition_readout(memory, valid)
        prior_raw = self.prior_head(summary)
        pm, pv = self._gaussian(prior_raw)
        target_memory = self.input_proj(safe_target)
        target_memory = (target_memory + self.future.spatial_position.to(target_memory)[None, None]
                         + age_sincos(torch.arange(1, horizon + 1, dtype=torch.int64), dim=hidden).to(target_memory)[None, :, None])
        target_valid = future_label_valid[:, :, None].expand(-1, -1, spatial).reshape(batch, -1)
        target_summary = self.future_readout(target_memory.reshape(batch, -1, hidden), target_valid)
        residual_raw = self.posterior_head(torch.cat((summary, target_summary), dim=-1))
        if self.stability is not None and not bool(torch.isfinite(residual_raw).all()):
            raise FloatingPointError("raw posterior Gaussian parameters are nonfinite")
        delta = ratio = None
        if self.stability is None:
            qm, qv = self._gaussian(prior_raw + residual_raw)
        else:
            with torch.autocast(device_type=memory.device.type, enabled=False):
                raw_delta, raw_ratio = residual_raw.float().chunk(2, dim=-1)
                config = self.stability
                delta = config.posterior_mean_delta_bound * torch.tanh(raw_delta / config.posterior_mean_delta_bound)
                ratio = config.posterior_log_scale_ratio_bound * torch.tanh(raw_ratio / config.posterior_log_scale_ratio_bound)
                qm, qv = pm + (.5 * pv).exp() * delta, pv + 2 * ratio
        z = qm[:, None] + (0.5 * qv).exp()[:, None] * sample_noise.float()
        tape = self._decode_future(memory, valid, z, horizon,
            (record_grid, incoming_action_embedding, record_age, record_type, record_valid, goal_grid))
        return StochasticProposalOutput(tape, torch.zeros(batch, 1, device=tape.device), z, pm, pv, qm, qv,
            posterior_standardized_delta=delta, posterior_log_scale_ratio=ratio)

    def sample_prior(self, record_grid, incoming_action_embedding, record_age,
                     record_type, record_valid, goal_grid, *, horizon: int,
                     samples: int, sample_keys, seed: int, namespace: str):
        horizon = self._horizon(horizon)
        _require_positive_integer("samples", samples)
        memory, valid = self._memory(record_grid, incoming_action_embedding, record_age,
                                     record_type, record_valid, goal_grid)
        pm, pv = self._gaussian(self.prior_head(self.condition_readout(memory, valid)))
        if len(sample_keys) != memory.shape[0]:
            raise ValueError("one sample key is required per occurrence")
        noise = keyed_normal(sample_keys, seed=seed, namespace=namespace, samples=samples,
                             latent_dim=self.trajectory_latent_dim, device=memory.device, dtype=torch.float32)
        z = pm[:, None] + (0.5 * pv).exp()[:, None] * noise
        tape = self._decode_future(memory, valid, z, horizon,
            (record_grid, incoming_action_embedding, record_age, record_type, record_valid, goal_grid))
        return StochasticProposalOutput(tape, torch.full((memory.shape[0], samples), -math.log(samples),
                                        dtype=torch.float32, device=memory.device), z, pm, pv,
                                        sampling_namespace=namespace)

    def recurrent_step_from_state(
        self,
        record_grid,
        incoming_action_embedding,
        record_age,
        record_type,
        record_valid,
        goal_grid,
        *,
        previous_grid: Tensor,
        trajectory_latent: Tensor,
    ) -> Tensor:
        """Decode one Q step from an externally supplied predicted state."""
        if not self.is_recurrent:
            raise RuntimeError("external-state Q stepping requires recurrent Q")
        memory, valid = self._memory(
            record_grid,
            incoming_action_embedding,
            record_age,
            record_type,
            record_valid,
            goal_grid,
        )
        batch, samples = trajectory_latent.shape[:2]
        spatial = self.future.spatial_position.shape[0]
        if previous_grid.shape != (batch, samples, spatial, self.input_proj.in_features):
            raise ValueError("previous_grid must be [B,K,M,D]")
        self._validate_latent(trajectory_latent, batch)
        expanded_memory = memory[:, None].expand(-1, samples, -1, -1).reshape(
            batch * samples, memory.shape[1], memory.shape[2]
        )
        expanded_valid = valid[:, None].expand(-1, samples, -1).reshape(
            batch * samples, valid.shape[1]
        )
        decoded = self.future.step(
            previous_grid.reshape(batch * samples, spatial, -1),
            expanded_memory,
            expanded_valid,
            trajectory_latent.reshape(batch * samples, -1),
        )
        return decoded.reshape(batch, samples, spatial, -1)

    def forward(self, *args, **kwargs):
        raise RuntimeError("stochastic Q requires explicit posterior_forward or prior-only sample_prior")
