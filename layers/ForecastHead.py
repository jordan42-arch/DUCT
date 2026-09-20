"""Forecast heads."""
import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """Flatten N*d_model -> pred_len (channel-independent)."""

    def __init__(self, d_model: int, n_patches: int, pred_len: int):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(n_patches * d_model, pred_len)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B*V, N, d_model) -> (B*V, pred_len)
        x = self.flatten(z)
        x = self.linear(x)
        return x
