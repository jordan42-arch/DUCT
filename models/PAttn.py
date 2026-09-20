"""PAttn baseline from TSLib (NeurIPS 2024, arXiv:2406.16964)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PAttnEncoderLayer(nn.Module):
    """The single post-norm attention block used by the released PAttn."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        y = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class PAttn(nn.Module):
    """Patch attention with channel-independent shared weights."""

    def __init__(
        self,
        n_vars: int,
        lookback: int,
        pred_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        del n_vars  # The released model shares all weights across variables.
        if lookback < patch_len:
            raise ValueError("PAttn requires lookback >= patch_len")
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.patch_num = (lookback - patch_len) // stride + 2

        self.pad = nn.ReplicationPad1d((0, stride))
        self.in_layer = nn.Linear(patch_len, d_model)
        self.encoder = _PAttnEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.out_layer = nn.Linear(d_model * self.patch_num, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, L) -> (B, V, pred_len)."""
        means = x.mean(dim=-1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(
            torch.var(centered, dim=-1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        normalized = centered / stdev

        batch, n_vars, _ = normalized.shape
        patches = self.pad(normalized).unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )
        encoded = self.in_layer(patches).reshape(
            batch * n_vars, self.patch_num, -1
        )
        encoded = self.final_norm(self.encoder(encoded))
        forecast = self.out_layer(encoded.reshape(batch, n_vars, -1))
        return forecast * stdev + means


__all__ = ["PAttn"]
