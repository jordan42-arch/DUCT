"""Reversible Instance Normalization."""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Per-channel normalization over the last dim; denorm restores the stats."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def norm(self, x: torch.Tensor):
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True) + self.eps
        return (x - mu) / sd, mu, sd

    def denorm(self, y: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor):
        return y * sd + mu
