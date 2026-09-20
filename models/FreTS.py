"""FreTS baseline (NeurIPS 2023, arXiv:2311.06184)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FreTS(nn.Module):
    """Frequency-domain temporal MLP with the released channel-independent path."""

    def __init__(
        self,
        n_vars: int,
        lookback: int,
        pred_len: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        sparsity_threshold: float = 0.01,
    ):
        super().__init__()
        del n_vars  # Weights are shared across variables in this protocol.
        self.lookback = lookback
        self.pred_len = pred_len
        self.embed_size = embed_size
        self.sparsity_threshold = sparsity_threshold
        scale = 0.02

        self.embedding = nn.Parameter(torch.randn(1, embed_size))
        self.real_weight = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.imag_weight = nn.Parameter(scale * torch.randn(embed_size, embed_size))
        self.real_bias = nn.Parameter(scale * torch.randn(embed_size))
        self.imag_bias = nn.Parameter(scale * torch.randn(embed_size))
        self.head = nn.Sequential(
            nn.Linear(lookback * embed_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, pred_len),
        )

    def _frequency_mlp(self, x: torch.Tensor) -> torch.Tensor:
        real = F.relu(
            torch.einsum("bntd,dd->bntd", x.real, self.real_weight)
            - torch.einsum("bntd,dd->bntd", x.imag, self.imag_weight)
            + self.real_bias
        )
        imag = F.relu(
            torch.einsum("bntd,dd->bntd", x.imag, self.real_weight)
            + torch.einsum("bntd,dd->bntd", x.real, self.imag_weight)
            + self.imag_bias
        )
        value = torch.stack([real, imag], dim=-1)
        return torch.view_as_complex(
            F.softshrink(value, lambd=self.sparsity_threshold)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, L) -> (B, V, pred_len)."""
        embedded = x.unsqueeze(-1) * self.embedding
        residual = embedded
        spectrum = torch.fft.rfft(embedded, dim=2, norm="ortho")
        spectrum = self._frequency_mlp(spectrum)
        encoded = torch.fft.irfft(
            spectrum, n=self.lookback, dim=2, norm="ortho"
        )
        encoded = encoded + residual
        return self.head(encoded.flatten(start_dim=2))


__all__ = ["FreTS"]
