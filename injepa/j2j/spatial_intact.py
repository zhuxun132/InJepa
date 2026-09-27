"""Full-grid INTACT actor/forward modules for the context-four variant."""

from __future__ import annotations

import hashlib
import math
import struct
import unicodedata

import torch
from torch import Tensor, nn

from j2j.proposal.blocks import DecoderBlock, EncoderBlock, run_block, validate_retained_activation_blocks
from j2j.proposal.model import ProposalJEPA
from j2j.proposal.positions import age_sincos, fixed_2d_sincos
from module import Embedder, MLP, SingletonSafeBatchNorm1d


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_transformer_shape(
    *,
    latent_dim: object,
    action_emb_dim: object,
    hidden_dim: object,
    heads: object,
    ffn_dim: object,
    depth: object,
    dropout: object,
    grid_side: object,
) -> tuple[int, int, int, int, int, int, float, int]:
    latent = _positive_integer("latent_dim", latent_dim)
    action = _positive_integer("action_emb_dim", action_emb_dim)
    hidden = _positive_integer("hidden_dim", hidden_dim)
    attention_heads = _positive_integer("heads", heads)
    feed_forward = _positive_integer("ffn_dim", ffn_dim)
    layers = _positive_integer("depth", depth)
    side = _positive_integer("grid_side", grid_side)
    if action != latent:
        raise ValueError("action_emb_dim must equal latent_dim for the shared INTACT coordinates")
    if hidden % attention_heads != 0:
        raise ValueError("hidden_dim must be divisible by heads")
    if hidden % 4 != 0:
        raise ValueError("hidden_dim must be divisible by four")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise TypeError("dropout must be a real scalar")
    probability = float(dropout)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout must be finite and in [0, 1)")
    return latent, action, hidden, attention_heads, feed_forward, layers, probability, side


def _semantic_seed(global_seed: int, semantic_fqn: str) -> int:
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")
    if global_seed < 0 or global_seed >= 2**64:
        raise ValueError("global_seed must be an unsigned 64-bit integer")
    encoded = unicodedata.normalize("NFC", semantic_fqn).encode("utf-8")
    preimage = b"J2J_CONTEXT4_INIT_V1\x00" + struct.pack("<Q", global_seed) + encoded
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") % (2**63)


def _initialize_module_(
    module: nn.Module,
    *,
    semantic_root: str,
    global_seed: int,
    residual_family: str | None = None,
) -> None:
    """Apply the frozen per-FQN CPU initialization without touching global RNG."""

    with torch.no_grad():
        for relative_name, parameter in sorted(module.named_parameters()):
            fqn = f"{semantic_root}.{relative_name}"
            if relative_name.endswith((".bias", "in_proj_bias")):
                parameter.zero_()
                continue
            if relative_name.endswith(".weight") and (
                "norm" in relative_name or "running" in relative_name
            ):
                parameter.fill_(1.0)
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(_semantic_seed(global_seed, fqn))
            nn.init.trunc_normal_(
                parameter,
                mean=0.0,
                std=0.02,
                a=-2.0,
                b=2.0,
                generator=generator,
            )
            pieces = relative_name.split(".")
            if residual_family == "actor" and len(pieces) >= 4 and pieces[0] == "blocks":
                layer = int(pieces[1]) + 1
                tail = ".".join(pieces[2:])
                if tail in {"self_attn.out_proj.weight", "ffn.fc2.weight"}:
                    parameter.div_(math.sqrt(2 * layer))
            elif residual_family == "forward" and len(pieces) >= 4 and pieces[0] == "blocks":
                layer = int(pieces[1]) + 1
                tail = ".".join(pieces[2:])
                if tail in {
                    "self_attn.out_proj.weight",
                    "cross_attn.out_proj.weight",
                    "ffn.fc2.weight",
                }:
                    parameter.div_(math.sqrt(3 * layer))


class SpatialIntentActionActor(nn.Module):
    """The single full-grid categorical action generator ``G``."""

    def __init__(
        self,
        *,
        latent_dim: int = 768,
        action_emb_dim: int = 768,
        hidden_dim: int = 768,
        heads: int = 16,
        ffn_dim: int = 2048,
        depth: int = 3,
        dropout: float = 0.1,
        grid_side: int = 6,
        global_seed: int,
        attention_backend: str = "math",
        activation_checkpointing: bool = False,
    ) -> None:
        latent, action, hidden, attention_heads, feed_forward, layers, probability, side = (
            _validate_transformer_shape(
                latent_dim=latent_dim,
                action_emb_dim=action_emb_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                ffn_dim=ffn_dim,
                depth=depth,
                dropout=dropout,
                grid_side=grid_side,
            )
        )
        super().__init__()
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        self.activation_checkpointing = activation_checkpointing
        with torch.random.fork_rng(devices=[]):
            self.input_proj = nn.Linear(
                3 * latent + action,
                hidden,
                bias=True,
                device="cpu",
                dtype=torch.float32,
            )
            self.action_token = nn.Parameter(torch.empty(1, 1, hidden, dtype=torch.float32))
            self.blocks = nn.ModuleList(
                EncoderBlock(hidden, attention_heads, feed_forward, probability,
                             attention_backend=attention_backend)
                for _ in range(layers)
            )
            self.final_norm = nn.LayerNorm(hidden, eps=1e-5, dtype=torch.float32)
            self.head = nn.Linear(hidden, 4, bias=True, dtype=torch.float32)
            self.register_buffer(
                "spatial_position",
                fixed_2d_sincos(dim=hidden, grid_side=side),
                persistent=True,
            )
            self.latent_dim = latent
            self.action_emb_dim = action
            self.grid_side = side
            _initialize_module_(
                self,
                semantic_root="j2j.spatial_intact.SpatialIntentActionActor",
                global_seed=global_seed,
                residual_family="actor",
            )

    def forward(self, z: Tensor, intent: Tensor, previous_action_embedding: Tensor) -> Tensor:
        spatial = self.grid_side * self.grid_side
        if z.ndim != 3 or z.shape[1:] != (spatial, self.latent_dim):
            raise ValueError("z must have shape [batch, spatial, latent_dim]")
        if intent.shape != z.shape:
            raise ValueError("intent must have the same shape as z")
        if previous_action_embedding.shape != (z.shape[0], self.action_emb_dim):
            raise ValueError("previous action embedding shape is inconsistent")
        if not (z.device == intent.device == previous_action_embedding.device == self.input_proj.weight.device):
            raise ValueError("actor inputs and module must share one device")
        if not bool(torch.isfinite(z).all() and torch.isfinite(intent).all() and torch.isfinite(previous_action_embedding).all()):
            raise ValueError("actor inputs must be finite")

        previous = previous_action_embedding[:, None, :].expand(-1, spatial, -1)
        features = torch.cat((z, intent, z * intent, previous), dim=-1)
        position = self.spatial_position.to(device=features.device, dtype=features.dtype)
        visual = self.input_proj(features) + position[None]
        action_token = self.action_token.to(dtype=visual.dtype).expand(z.shape[0], -1, -1)
        tokens = torch.cat((action_token, visual), dim=1)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        for block in self.blocks:
            tokens = run_block(block, tokens, valid,
                               activation_checkpointing=self.activation_checkpointing)
        return self.head(self.final_norm(tokens[:, 0]))


class SpatialForwardPredictor(nn.Module):
    """The single action-conditioned full-grid forward core ``F``."""

    def __init__(
        self,
        *,
        latent_dim: int = 768,
        action_emb_dim: int = 768,
        hidden_dim: int = 768,
        heads: int = 16,
        ffn_dim: int = 2048,
        depth: int = 6,
        dropout: float = 0.1,
        grid_side: int = 6,
        max_context: int = 4,
        global_seed: int,
        attention_backend: str = "math",
        activation_checkpointing: bool = False,
        memory_norm: bool = False,
    ) -> None:
        latent, action, hidden, attention_heads, feed_forward, layers, probability, side = (
            _validate_transformer_shape(
                latent_dim=latent_dim,
                action_emb_dim=action_emb_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                ffn_dim=ffn_dim,
                depth=depth,
                dropout=dropout,
                grid_side=grid_side,
            )
        )
        context = _positive_integer("max_context", max_context)
        if context != 4:
            raise ValueError("max_context must equal the RAE-NWM source value 4")
        super().__init__()
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        self.activation_checkpointing = activation_checkpointing
        with torch.random.fork_rng(devices=[]):
            self.input_proj = nn.Linear(latent, hidden, bias=True, dtype=torch.float32)
            self.type_embedding = nn.Embedding(3, hidden, dtype=torch.float32)
            self.blocks = nn.ModuleList(
                DecoderBlock(hidden, attention_heads, feed_forward, probability,
                             attention_backend=attention_backend, memory_norm=memory_norm)
                for _ in range(layers)
            )
            self.final_norm = nn.LayerNorm(hidden, eps=1e-5, dtype=torch.float32)
            self.register_buffer(
                "spatial_position",
                fixed_2d_sincos(dim=hidden, grid_side=side),
                persistent=True,
            )
            self.latent_dim = latent
            self.action_emb_dim = action
            self.hidden_dim = hidden
            self.grid_side = side
            self.max_context = context
            _initialize_module_(
                self,
                semantic_root="j2j.spatial_intact.SpatialForwardPredictor",
                global_seed=global_seed,
                residual_family="forward",
            )

    def forward(
        self,
        history_grid: Tensor,
        outgoing_action_embedding: Tensor,
        history_valid: Tensor,
    ) -> Tensor:
        spatial = self.grid_side * self.grid_side
        if history_grid.ndim != 4:
            raise ValueError("history_grid must have shape [batch, context, spatial, latent]")
        batch, context, seen_spatial, latent = history_grid.shape
        if not 1 <= context <= self.max_context:
            raise ValueError("history context length must be in [1, 4]")
        if (seen_spatial, latent) != (spatial, self.latent_dim):
            raise ValueError("history_grid spatial/latent shape is inconsistent")
        if outgoing_action_embedding.shape != (batch, context, self.action_emb_dim):
            raise ValueError("outgoing action embedding shape is inconsistent")
        if history_valid.shape != (batch, context) or history_valid.dtype != torch.bool:
            raise ValueError("history_valid must be bool [batch, context]")
        if not bool(history_valid[:, -1].all()):
            raise ValueError("the current context slot must be valid")
        if context > 1 and bool((history_valid[:, 1:] < history_valid[:, :-1]).any()):
            raise ValueError("history_valid must be a contiguous right-aligned suffix")
        if not (
            history_grid.device
            == outgoing_action_embedding.device
            == history_valid.device
            == self.input_proj.weight.device
        ):
            raise ValueError("forward inputs and module must share one device")
        valid_grid = history_grid[history_valid]
        valid_action = outgoing_action_embedding[history_valid]
        if not bool(torch.isfinite(valid_grid).all() and torch.isfinite(valid_action).all()):
            raise ValueError("valid forward inputs must be finite")

        position = self.spatial_position.to(device=history_grid.device, dtype=history_grid.dtype)
        ages = torch.arange(context - 1, -1, -1, dtype=torch.int64, device=history_grid.device)
        ages = ages[None].expand(batch, -1)
        age = age_sincos(ages, dim=self.hidden_dim).to(
            device=history_grid.device,
            dtype=history_grid.dtype,
        )
        visual_type = self.type_embedding(
            torch.zeros((batch, context), dtype=torch.int64, device=history_grid.device)
        )
        action_type = self.type_embedding(
            torch.ones((batch, context), dtype=torch.int64, device=history_grid.device)
        )
        visual = (
            self.input_proj(history_grid)
            + position[None, None]
            + age[:, :, None]
            + visual_type[:, :, None]
        )
        action = self.input_proj(outgoing_action_embedding) + age + action_type
        visual = visual.masked_fill(~history_valid[:, :, None, None], 0.0)
        action = action.masked_fill(~history_valid[:, :, None], 0.0)
        memory = torch.cat((visual, action[:, :, None]), dim=2).reshape(
            batch, context * (spatial + 1), self.hidden_dim
        )
        memory_valid = history_valid[:, :, None].expand(
            batch, context, spatial + 1
        ).reshape(batch, context * (spatial + 1))

        query_type = self.type_embedding(
            torch.full((batch, spatial), 2, dtype=torch.int64, device=history_grid.device)
        )
        query = position[None] + query_type
        query_valid = torch.ones((batch, spatial), dtype=torch.bool, device=history_grid.device)
        for block in self.blocks:
            query = run_block(block, query, query_valid, memory, memory_valid,
                              activation_checkpointing=self.activation_checkpointing)
        return self.final_norm(query)


class Context4SpatialJointModel(nn.Module):
    """One uniquely owned trainable Q/A/G/F module tree."""

    def __init__(
        self,
        *,
        global_seed: int,
        modes: int = 4,
        horizon: int = 4,
        latent_dim: int = 768,
        grid_side: int = 6,
        hidden_dim: int = 768,
        heads: int = 16,
        ffn_dim: int = 2048,
        proposal_depth: int = 6,
        actor_depth: int = 3,
        forward_depth: int = 6,
        dropout: float = 0.1,
        attention_backend: str = "math",
        activation_checkpointing: bool = False,
        memory_norm: bool = False,
        prediction_norm: str = "batchnorm",
        proposal_architecture: str = "fixed_mixture",
        trajectory_latent_dim: int = 128,
        kl_beta: float = 0.05,
        sigma_epsilon: float = 1e-4,
        posterior_residual_init_std: float = 1e-3,
        stability: dict | None = None,
        retain_activation_blocks: tuple[str, ...] | list[str] = (),
        training_qkv_normalization_backend: str = "eager",
        training_q_horizon: int | None = None,
    ) -> None:
        super().__init__()
        stochastic = proposal_architecture in {"stochastic_divided", "stochastic_recurrent"}
        recurrent = proposal_architecture == "stochastic_recurrent"
        if proposal_architecture != "fixed_mixture" and not stochastic:
            raise ValueError("unknown proposal architecture")
        if recurrent:
            if type(training_q_horizon) is not int or training_q_horizon != 1:
                raise ValueError("recurrent Q requires training_q_horizon=1")
        elif training_q_horizon is not None:
            raise ValueError("training_q_horizon requires a recurrent Q")
        if stability is not None and not stochastic:
            raise ValueError("stability controls require a stochastic proposal")
        from j2j.proposal.stability import BoundedMultiheadAttention, validate_training_qkv_normalization_backend
        self.training_qkv_normalization_backend = validate_training_qkv_normalization_backend(
            training_qkv_normalization_backend,
            stable=stability is not None and stochastic)
        if isinstance(kl_beta, bool) or not math.isfinite(kl_beta) or kl_beta <= 0:
            raise ValueError("kl_beta must be finite and positive")
        self.proposal_architecture = proposal_architecture
        self.kl_beta = float(kl_beta)
        if prediction_norm not in {"batchnorm", "layernorm"}:
            raise ValueError("prediction_norm must be batchnorm or layernorm")
        with torch.random.fork_rng(devices=[]):
            from j2j.proposal.stochastic import StochasticProposalJEPA
            proposal_type = StochasticProposalJEPA if stochastic else ProposalJEPA
            proposal_options = (
                dict(trajectory_latent_dim=trajectory_latent_dim, sigma_epsilon=sigma_epsilon,
                     stability=stability,
                     posterior_residual_init_std=posterior_residual_init_std,
                     attention_backend=attention_backend, memory_norm=memory_norm,
                     activation_checkpointing=activation_checkpointing)
                if stochastic else dict(selector_depth=3)
            )
            if recurrent:
                proposal_options.update(future_architecture="recurrent", training_horizon=training_q_horizon)
            self.proposal = proposal_type(
                latent_dim=latent_dim,
                grid_side=grid_side,
                hidden_dim=hidden_dim,
                heads=heads,
                ffn_dim=ffn_dim,
                future_depth=proposal_depth,
                dropout=dropout,
                modes=modes,
                horizon=horizon,
                global_seed=global_seed,
                **proposal_options,
            )
            # Execution policy belongs to this Context4 owner; preserve the
            # legacy ProposalJEPA constructor and deterministic tensor state.
            self.proposal.future.activation_checkpointing = activation_checkpointing
            for block in self.proposal.future.blocks:
                block.attention_backend = attention_backend
                block.memory_norm = memory_norm
            # Context4 reuses the established Q input projection and future
            # decoder exactly, but this distinct model has no selector owner.
            # Removing the submodule here preserves the legacy ProposalJEPA
            # public API/state while excluding every selector parameter from
            # this model, optimizer, checkpoint, and parameter ledger.
            self.proposal.selector = None
            self.action_encoder = Embedder(input_dim=4, emb_dim=latent_dim)
            _initialize_module_(
                self.action_encoder,
                semantic_root="j2j.spatial_intact.Context4SpatialJointModel.action_encoder",
                global_seed=global_seed,
            )
            self.actor = SpatialIntentActionActor(
                latent_dim=latent_dim,
                action_emb_dim=latent_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                ffn_dim=ffn_dim,
                depth=actor_depth,
                dropout=dropout,
                grid_side=grid_side,
                global_seed=global_seed,
                attention_backend=attention_backend,
                activation_checkpointing=activation_checkpointing,
            )
            self.forward_core = SpatialForwardPredictor(
                latent_dim=latent_dim,
                action_emb_dim=latent_dim,
                hidden_dim=hidden_dim,
                heads=heads,
                ffn_dim=ffn_dim,
                depth=forward_depth,
                dropout=dropout,
                grid_side=grid_side,
                max_context=4,
                global_seed=global_seed,
                attention_backend=attention_backend,
                activation_checkpointing=activation_checkpointing,
                memory_norm=memory_norm,
            )
            self.pred_proj = MLP(
                hidden_dim,
                2048,
                latent_dim,
                norm_fn=SingletonSafeBatchNorm1d if prediction_norm == "batchnorm" else nn.LayerNorm,
            )
            _initialize_module_(
                self.pred_proj,
                semantic_root="j2j.spatial_intact.Context4SpatialJointModel.pred_proj",
                global_seed=global_seed,
            )
            if prediction_norm == "layernorm":
                # Numeric Sequential names do not identify norms to the
                # inherited per-FQN initializer; retain LayerNorm's identity.
                self.pred_proj.net[1].reset_parameters()

        for module in self.proposal.modules():
            if isinstance(module, BoundedMultiheadAttention):
                module.training_qkv_normalization_backend = self.training_qkv_normalization_backend

        owners = {"proposal.future.blocks": self.proposal.future.blocks,
                  "forward_core.blocks": self.forward_core.blocks,
                  "actor.blocks": self.actor.blocks}
        self.retain_activation_blocks = validate_retained_activation_blocks(
            retain_activation_blocks, activation_checkpointing=activation_checkpointing,
            depths={prefix: len(blocks) for prefix, blocks in owners.items()})
        selected = frozenset(self.retain_activation_blocks)
        for prefix, blocks in owners.items():
            for index, block in enumerate(blocks):
                # Execution metadata only; tensor state and unique owners stay unchanged.
                block.retain_activations = f"{prefix}.{index}" in selected

    def embed_actions(self, raw4: Tensor) -> Tensor:
        if raw4.ndim == 2:
            return self.action_encoder(raw4[:, None]).squeeze(1)
        if raw4.ndim == 3:
            return self.action_encoder(raw4)
        raise ValueError("raw actions must have shape [batch,4] or [batch,time,4]")

    def actor_logits(self, z: Tensor, intent: Tensor, previous_raw4: Tensor) -> Tensor:
        return self.actor(z, intent, self.embed_actions(previous_raw4))

    def predict_next_grid(
        self,
        history_grid: Tensor,
        outgoing_raw4: Tensor,
        history_valid: Tensor,
    ) -> Tensor:
        hidden = self.forward_core(
            history_grid,
            self.embed_actions(outgoing_raw4),
            history_valid,
        )
        batch, spatial, width = hidden.shape
        return self.pred_proj(hidden.reshape(batch * spatial, width)).reshape(
            batch, spatial, -1
        )
