"""Small building blocks shared by the world model and the policy."""

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with ReLU, used to embed actions."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to a sequence."""

    def __init__(self, d_model, max_len=2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d_model] -> broadcast along the batch dimension
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """x: [batch_size, seq_len, d_model]."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
