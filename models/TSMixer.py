"""TSMixer baseline (arXiv 2023, arXiv:2303.06053)."""
import torch
import torch.nn as nn


class _MixerBlock(nn.Module):
    def __init__(
        self, n_vars: int, lookback: int, d_model: int, dropout: float,
    ):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(lookback, d_model),
            nn.ReLU(),
            nn.Linear(d_model, lookback),
            nn.Dropout(dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(n_vars, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_vars),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        return x + self.channel(x)


class TSMixer(nn.Module):
    def __init__(
        self,
        n_vars: int,
        lookback: int,
        pred_len: int,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            _MixerBlock(n_vars, lookback, d_model, dropout)
            for _ in range(n_layers)
        ])
        self.projection = nn.Linear(lookback, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, L) -> (B, V, pred_len)."""
        value = x.transpose(1, 2)
        for block in self.blocks:
            value = block(value)
        return self.projection(value.transpose(1, 2))


__all__ = ["TSMixer"]
