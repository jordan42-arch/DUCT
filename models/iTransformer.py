"""iTransformer baseline (arXiv:2310.06625): each variate is one token."""
import torch
import torch.nn as nn

from layers import RevIN


class iTransformer(nn.Module):
    def __init__(self, n_vars: int, lookback: int, pred_len: int,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.n_vars = n_vars
        self.lookback = lookback
        self.pred_len = pred_len
        self.d_model = d_model

        self.revin = RevIN()
        self.value_emb = nn.Linear(lookback, d_model)
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
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n, mu, sd = self.revin.norm(x)
        # x_n: (B, V, L)
        z = self.value_emb(x_n)  # (B, V, d_model)
        z = self.encoder(z)
        y = self.head(z)  # (B, V, pred_len)
        return self.revin.denorm(y, mu, sd)


# Pre-refactor name, kept so old checkpoints and notebooks still resolve.
iTransformerBaseline = iTransformer
