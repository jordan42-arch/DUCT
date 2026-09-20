"""PatchTST baseline (Nie et al., ICLR 2023, arXiv:2211.14730)."""
import torch
import torch.nn as nn

from layers import ForecastHead, PatchEmbed, RevIN, TransformerEncoder


class PatchTST(nn.Module):
    """Channel-independent: every channel's patches share one set of weights."""

    def __init__(
        self,
        n_vars: int,
        lookback: int = 96,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        revin: bool = True,
    ):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.revin = revin

        self.n_patches = (lookback - patch_len) // stride + 1

        self.revin_module = RevIN()
        self.patch_embed = PatchEmbed(patch_len=patch_len, stride=stride, d_model=d_model)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.forecast_head = ForecastHead(d_model=d_model, n_patches=self.n_patches, pred_len=pred_len)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B*V, N, d_model) patch encodings (post-Transformer)."""
        # x: (B, V, L)
        z = self.patch_embed(x)
        z = self.encoder(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, L) -> (B, V, pred_len)."""
        if self.revin:
            x_n, mu, sd = self.revin_module.norm(x)
        else:
            x_n = x
        B, V, _ = x_n.shape
        z = self.encode(x_n)  # (B*V, N, d_model)
        y = self.forecast_head(z)  # (B*V, pred_len)
        y = y.view(B, V, self.pred_len)
        if self.revin:
            y = self.revin_module.denorm(y, mu, sd)
        return y
