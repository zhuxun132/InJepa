"""Explicit, configuration-bound local scale controls for stochastic Q v2."""

from dataclasses import dataclass, fields
from collections.abc import Mapping
from functools import lru_cache
import math
from numbers import Real

import torch
from torch import nn, Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class StabilityConfig:
    ln_eps: float = 1e-4
    ln_gain_bound: float = 2.0
    qk_eps: float = 1e-4
    attention_temperature_max_factor: float = 2.0
    attention_temperature_init_factor: float = 0.3072
    value_rms_bound: float = 2.0
    prior_mean_bound: float = 5.0
    prior_log_scale_bound: float = math.log(2)
    posterior_mean_delta_bound: float = 4.0
    posterior_log_scale_ratio_bound: float = math.log(2)

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"stability {field.name} must be finite and positive")
        if self.ln_gain_bound <= 1:
            raise ValueError("LayerNorm gain bound must admit initial gamma one")
        if self.attention_temperature_init_factor >= self.attention_temperature_max_factor:
            raise ValueError("initial attention temperature must be below its bound")

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping):
            raise TypeError("stability must be a mapping")
        if set(value) - {field.name for field in fields(cls)}:
            raise ValueError("unknown stability control")
        return cls(**dict(value))


class BoundedLayerNorm(nn.LayerNorm):
    """Same affine owners, smoothly bounded effective gamma; at least FP32."""

    def __init__(self, normalized_shape, *, eps=1e-4, gain_bound=2.0):
        if not math.isfinite(gain_bound) or gain_bound <= 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("invalid bounded LayerNorm controls")
        self.gain_bound = float(gain_bound)
        super().__init__(normalized_shape, eps=eps)

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.constant_(self.weight, self.gain_bound * math.atanh(1 / self.gain_bound))
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    @property
    def effective_weight(self):
        return self.gain_bound * torch.tanh(self.weight / self.gain_bound)

    def forward(self, value):
        dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=value.device.type, enabled=False):
            output = F.layer_norm(value.to(dtype), self.normalized_shape,
                                  self.effective_weight.to(dtype), self.bias.to(dtype), self.eps)
        return output.to(value.dtype)


def _normalize_bounded_qkv(
    q: Tensor, k: Tensor, v: Tensor, temperature: Tensor, *,
    head_dim: int, qk_eps: float, value_rms_bound: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Pure normalization seam; retain precision and all input gradients."""
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=q.device.type, enabled=False):
        qn, kn, vn = (part.to(compute_dtype) for part in (q, k, v))
        qn = qn / torch.sqrt(qn.square().sum(-1, keepdim=True) + head_dim * qk_eps)
        kn = kn / torch.sqrt(kn.square().sum(-1, keepdim=True) + head_dim * qk_eps)
        vn = vn / torch.sqrt(1 + vn.square().mean(-1, keepdim=True) / value_rms_bound**2)
        qn = qn * temperature.to(compute_dtype)[None, :, None, None]
        return qn.to(q.dtype), kn.to(k.dtype), vn.to(v.dtype)


def validate_training_qkv_normalization_backend(value, *, stable=True):
    if not isinstance(value, str) or value not in {"eager", "inductor"}:
        raise ValueError("training_qkv_normalization_backend must be eager or inductor")
    if value != "eager" and not stable:
        raise ValueError("compiled QKV normalization requires stable bounded Q")
    return value


@lru_cache(maxsize=1)
def _compiled_training_qkv_normalizer():
    """Cache only a pure callable; attention, dropout and model state stay eager."""
    return torch.compile(_normalize_bounded_qkv, fullgraph=True, dynamic=True,
                         options={"triton.cudagraphs": False})


def _use_split_key_padding_attention(q, k, v, dropout_p):
    if q.device.type != "cuda" or not torch.backends.cuda.flash_sdp_enabled():
        return False
    parameters = torch.backends.cuda.SDPAParams(q, k, v, None, dropout_p, False, False)
    return torch.backends.cuda.can_use_flash_attention(parameters)


def _key_padding_bounded_sdpa(q, k, v, key_padding_mask, *, dropout_p):
    """Keep each row's allowed keys while separating unpadded SDPA work."""
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape or q.shape[0] < 1:
        raise ValueError("bounded SDPA requires nonempty batch/head/query/key tensors")
    if (key_padding_mask.dtype != torch.bool or
            key_padding_mask.shape != (q.shape[0], k.shape[2])):
        raise ValueError("key padding mask must be boolean [B,S]")
    complete = ~key_padding_mask.any(dim=-1)
    if bool(complete.all()):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=1.0)
    if not bool(complete.any()):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=~key_padding_mask[:, None, None, :],
            dropout_p=dropout_p, scale=1.0)
    full_indices = complete.nonzero(as_tuple=False).flatten()
    partial_indices = (~complete).nonzero(as_tuple=False).flatten()
    full = F.scaled_dot_product_attention(
        *(part.index_select(0, full_indices) for part in (q, k, v)),
        dropout_p=dropout_p, scale=1.0)
    partial = F.scaled_dot_product_attention(
        *(part.index_select(0, partial_indices) for part in (q, k, v)),
        attn_mask=~key_padding_mask.index_select(0, partial_indices)[:, None, None, :],
        dropout_p=dropout_p, scale=1.0)
    output = q.new_empty(q.shape)
    output.index_copy_(0, full_indices, full)
    output.index_copy_(0, partial_indices, partial)
    return output


class BoundedMultiheadAttention(nn.MultiheadAttention):
    """Packed QKV attention with regularized cosine logits and bounded V RMS.

    Masks retain the MultiheadAttention True=blocked public convention. This
    module only implements the batch-first, equal-width, no-weight-return ABI
    actually used by Q. SDPA receives True=allowed and explicit scale one.
    """

    def __init__(self, embed_dim, num_heads, *, stability: StabilityConfig,
                 dropout=0.0, batch_first=True,
                 training_qkv_normalization_backend="eager", **kwargs):
        backend = validate_training_qkv_normalization_backend(training_qkv_normalization_backend)
        if not isinstance(stability, StabilityConfig) or not batch_first:
            raise ValueError("bounded attention requires validated controls and batch_first")
        super().__init__(embed_dim, num_heads, dropout=dropout, batch_first=True, **kwargs)
        if not self._qkv_same_embed_dim or self.bias_k is not None or self.bias_v is not None or self.add_zero_attn:
            raise ValueError("bounded attention requires ordinary packed QKV")
        self.stability = stability
        self.training_qkv_normalization_backend = backend
        self.temperature_logits = nn.Parameter(torch.empty(num_heads))
        self.reset_temperature()

    def reset_temperature(self):
        fraction = self.stability.attention_temperature_init_factor / self.stability.attention_temperature_max_factor
        nn.init.constant_(self.temperature_logits, math.log(fraction / (1 - fraction)))

    @property
    def effective_temperature(self):
        return self.stability.attention_temperature_max_factor * math.sqrt(self.head_dim) * self.temperature_logits.sigmoid()

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if need_weights or is_causal:
            raise ValueError("bounded Q attention requires explicit masks and need_weights=False")
        if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
            raise ValueError("bounded attention requires batch-first query/key/value tensors")
        batch, length, hidden = query.shape
        source = key.shape[1]
        if hidden != self.embed_dim or key.shape[0] != batch or key.shape[2] != hidden:
            raise ValueError("bounded attention projection shape mismatch")
        blocked = None
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                raise TypeError("bounded Q attention mask must be boolean")
            if attn_mask.shape == (length, source):
                blocked = attn_mask[None, None]
            elif attn_mask.shape == (batch * self.num_heads, length, source):
                blocked = attn_mask.reshape(batch, self.num_heads, length, source)
            else:
                raise ValueError("attention mask shape mismatch")
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (batch, source):
                raise ValueError("key padding mask must be boolean [B,S]")
            padding = key_padding_mask[:, None, None, :]
            blocked = padding if blocked is None else blocked | padding
        if query is key and key is value:
            parts = F.linear(query, self.in_proj_weight, self.in_proj_bias).split(hidden, dim=-1)
        elif key is value:
            q_bias = None if self.in_proj_bias is None else self.in_proj_bias[:hidden]
            kv_bias = None if self.in_proj_bias is None else self.in_proj_bias[hidden:]
            q_part = F.linear(query, self.in_proj_weight[:hidden], q_bias)
            k_part, v_part = F.linear(key, self.in_proj_weight[hidden:], kv_bias).split(hidden, dim=-1)
            parts = (q_part, k_part, v_part)
        else:
            # Equal values or aliased storage do not imply identical autograd inputs.
            parts = []
            for index, tensor in enumerate((query, key, value)):
                sl = slice(index * hidden, (index + 1) * hidden)
                bias = None if self.in_proj_bias is None else self.in_proj_bias[sl]
                parts.append(F.linear(tensor, self.in_proj_weight[sl], bias))
        q, k, v = (part.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
                   for part in parts)
        with torch.autocast(device_type=query.device.type, enabled=False):
            normalizer = (_compiled_training_qkv_normalizer()
                          if self.training_qkv_normalization_backend == "inductor"
                          and self.training and torch.is_grad_enabled()
                          else _normalize_bounded_qkv)
            qn, kn, vn = normalizer(
                q, k, v, self.effective_temperature, head_dim=self.head_dim,
                qk_eps=self.stability.qk_eps,
                value_rms_bound=self.stability.value_rms_bound,
            )
            dropout_p = self.dropout if self.training else 0.0
            if (attn_mask is None and key_padding_mask is not None
                    and _use_split_key_padding_attention(qn, kn, vn, dropout_p)):
                attended = _key_padding_bounded_sdpa(
                    qn, kn, vn, key_padding_mask, dropout_p=dropout_p)
            else:
                attended = F.scaled_dot_product_attention(
                    qn, kn, vn, attn_mask=None if blocked is None else ~blocked,
                    dropout_p=dropout_p, scale=1.0)
        attended = attended.transpose(1, 2).reshape(batch, length, hidden)
        return F.linear(attended, self.out_proj.weight, self.out_proj.bias), None
