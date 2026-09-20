"""Augmentations for contrastive pretraining. All act on (B, V, L) per
sample-and-channel, so they are model-agnostic."""
import math

import torch


def aug_gaussian_noise(x: torch.Tensor, sigma_range=(0.01, 0.05)) -> torch.Tensor:
    """Add per-sample-per-channel Gaussian noise with random sigma."""
    B, V, L = x.shape
    sigma = torch.empty(B, V, 1, device=x.device).uniform_(*sigma_range)
    return x + sigma * torch.randn_like(x)


def aug_amplitude_scaling(x: torch.Tensor, scale_range=(0.8, 1.2)) -> torch.Tensor:
    """Per-sample-per-channel amplitude scaling."""
    B, V, L = x.shape
    s = torch.empty(B, V, 1, device=x.device).uniform_(*scale_range)
    return x * s


def aug_time_warp(x: torch.Tensor, warp_strength: float = 0.2) -> torch.Tensor:
    """Non-linear time warping via random sinusoidal time map + linear interp."""
    B, V, L = x.shape
    device = x.device
    t = torch.arange(L, device=device, dtype=x.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,L)
    freq = torch.empty(B, 1, 1, device=device).uniform_(1.0, 3.0)
    phase = torch.empty(B, 1, 1, device=device).uniform_(0, 2 * math.pi)
    offset = warp_strength * L * torch.sin(2 * math.pi * t / L * freq + phase)  # (B,1,L)
    t_warp = (t + offset).clamp(0, L - 1)  # (B,1,L)
    idx_lo = t_warp.floor().long()
    idx_hi = (idx_lo + 1).clamp_max(L - 1)
    w_hi = (t_warp - idx_lo.float())
    w_lo = 1 - w_hi
    # gather per channel
    idx_lo_exp = idx_lo.expand(B, V, L)
    idx_hi_exp = idx_hi.expand(B, V, L)
    x_lo = torch.gather(x, dim=-1, index=idx_lo_exp)
    x_hi = torch.gather(x, dim=-1, index=idx_hi_exp)
    return w_lo * x_lo + w_hi * x_hi


def parse_augmentation_mode(mode: str):
    """Resolve a mode string into the list of enabled augmentation names."""
    aliases = {
        "all": ["noise", "scaling", "time_warp"],
        "none": [],
        "no": [],
        "identity": [],
        "gaussian": ["noise"],
        "gaussian_noise": ["noise"],
        "scale": ["scaling"],
        "amplitude": ["scaling"],
        "amplitude_scaling": ["scaling"],
        "warp": ["time_warp"],
        "time-warp": ["time_warp"],
        "time_warping": ["time_warp"],
    }
    if mode in aliases:
        return aliases[mode]
    parts = [p.strip() for p in mode.split(",") if p.strip()]
    resolved = []
    for part in parts:
        resolved.extend(aliases.get(part, [part]))
    valid = {"noise", "scaling", "time_warp"}
    unknown = [p for p in resolved if p not in valid]
    if unknown:
        raise ValueError(f"unknown augmentation mode(s): {unknown}; valid={sorted(valid)}")
    return resolved


def apply_random_aug(x: torch.Tensor, mode: str = "all") -> torch.Tensor:
    """Pick one enabled augmentation uniformly at random."""
    choices = parse_augmentation_mode(mode)
    if not choices:
        return x
    idx = torch.randint(len(choices), (), device=x.device).item()
    aug = choices[idx]
    if aug == "noise":
        return aug_gaussian_noise(x)
    if aug == "scaling":
        return aug_amplitude_scaling(x)
    return aug_time_warp(x)
