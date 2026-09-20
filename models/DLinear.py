"""DLinear baseline (Zeng et al. 2023, arXiv:2205.13504)."""
import torch
import torch.nn as nn


class DLinear(nn.Module):
    """Split into trend (moving average) and seasonal, project each, sum."""

    def __init__(self, n_vars: int, lookback: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.kernel_size = kernel_size
        # padding for moving average so output length = lookback
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        # per-channel linear (channel-independent)
        self.linear_trend = nn.Linear(lookback, pred_len)
        self.linear_seasonal = nn.Linear(lookback, pred_len)

    def _moving_avg(self, x):
        L = x.size(-1)
        avg = self.avg_pool(x)
        # padding=k//2 yields L+1 for even k; truncate back to L
        if avg.size(-1) != L:
            avg = avg[..., :L]
        return avg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, L)
        trend = self._moving_avg(x)
        seasonal = x - trend
        y_t = self.linear_trend(trend)
        y_s = self.linear_seasonal(seasonal)
        return y_t + y_s  # (B, V, pred_len)
