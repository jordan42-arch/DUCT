"""Naive persistence baseline."""
import torch
import torch.nn as nn


class Naive(nn.Module):
    """Persistence: predict the last lookback value for all horizons."""

    def __init__(self, n_vars: int, lookback: int, pred_len: int):
        super().__init__()
        self.pred_len = pred_len
        # Keeps the optimizer and the backward pass happy; contributes nothing.
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[..., -1:].expand(-1, -1, self.pred_len)
        return last + 0.0 * self._dummy
