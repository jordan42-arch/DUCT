"""Embedding layers shared by the patch-based models."""
import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Channel-independent patch embedding. (B, V, L) -> (B*V, N, d_model)."""

    def __init__(self, patch_len: int = 16, stride: int = 8, d_model: int = 128):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.value_emb = nn.Linear(patch_len, d_model)
        self._cached_pe_n = 0
        self._cached_pe = None

    def _make_pos_emb(self, n_patches: int, device, dtype):
        if self._cached_pe is not None and self._cached_pe_n == n_patches:
            return self._cached_pe
        position = torch.arange(n_patches, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(n_patches, self.d_model, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self._cached_pe = pe
        self._cached_pe_n = n_patches
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, V, L)
        B, V, L = x.shape
        # unfold into patches
        x = x.unfold(-1, self.patch_len, self.stride)  # (B, V, N, patch_len)
        N = x.shape[-2]
        x = x.reshape(B * V, N, self.patch_len)
        z = self.value_emb(x)  # (B*V, N, d_model)
        pe = self._make_pos_emb(N, z.device, z.dtype)
        z = z + pe.unsqueeze(0)
        return z  # (B*V, N, d_model)
