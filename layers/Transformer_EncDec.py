"""Transformer encoder stack shared by the patch-based models."""
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Pre-norm GELU Transformer encoder over a token sequence."""

    def __init__(self, d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B*V, N, d_model)
        return self.encoder(z)
