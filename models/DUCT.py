"""DUCT: channel-independent patch branch + variate-as-token branch, mixed by
learned fusion weights. Pretrained with masked reconstruction + InfoNCE, both
toggleable via use_mae / use_contrastive for the ablations."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    ForecastHead,
    PatchEmbed,
    RevIN,
    TransformerEncoder,
    apply_random_aug,
)


class PatchBranch(nn.Module):
    """Channel-independent patch-based temporal branch."""

    def __init__(
        self,
        lookback: int = 96,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback = lookback
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (lookback - patch_len) // stride + 1

        self.patch_embed = PatchEmbed(patch_len=patch_len, stride=stride, d_model=d_model)
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.time_head = ForecastHead(d_model=d_model, n_patches=self.n_patches, pred_len=pred_len)

    def encode(self, x: torch.Tensor):
        """(B, V, L) -> tokens (B*V, N, D) and per-channel rep (B, V, D)."""
        B, V, _ = x.shape
        z = self.encoder(self.patch_embed(x))
        rep = z.reshape(B, V, self.n_patches, -1).mean(dim=2)
        return z, rep

    def forward(self, x: torch.Tensor):
        B, V, _ = x.shape
        z, rep = self.encode(x)
        y = self.time_head(z).view(B, V, self.pred_len)
        return y, rep


class VariateBranch(nn.Module):
    """Variate-as-token interaction branch."""

    def __init__(
        self,
        lookback: int,
        pred_len: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
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

    def encode(self, x: torch.Tensor):
        return self.encoder(self.value_emb(x))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.head(z), z


class DUCT(nn.Module):
    """Dual-path DUCT paper model."""

    branch_names = ("patch", "var")

    def __init__(
        self,
        n_vars: int,
        lookback: int = 96,
        pred_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        proj_dim: int = 64,
        mask_ratio: float = 0.4,
        mae_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07,
        revin: bool = True,
        use_contrastive: bool = True,
        use_mae: bool = True,
        use_augmentation: bool = True,
        augmentation_mode: str = "all",
    ):
        super().__init__()
        self.n_vars = n_vars
        self.lookback = lookback
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.mae_weight = mae_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.revin = revin
        self.use_contrastive = use_contrastive
        self.use_mae = use_mae
        self.use_augmentation = use_augmentation
        self.augmentation_mode = augmentation_mode if use_augmentation else "none"

        self.revin_module = RevIN()
        self.patch_branch = PatchBranch(
            lookback=lookback,
            pred_len=pred_len,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.variate_branch = VariateBranch(
            lookback=lookback,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.rep_fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.mae_decoder = nn.Linear(d_model, lookback)
        self.pool_score = nn.Linear(d_model, 1)
        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )
        self.fusion_logits = nn.Parameter(torch.zeros(2))

    def _encode_branch_reps(self, x: torch.Tensor):
        _, patch_rep = self.patch_branch.encode(x)
        var_rep = self.variate_branch.encode(x)
        return patch_rep, var_rep

    def _fuse_rep(self, patch_rep: torch.Tensor, var_rep: torch.Tensor):
        return self.rep_fusion(torch.cat([patch_rep, var_rep], dim=-1))

    def _encode_fused_rep(self, x: torch.Tensor):
        patch_rep, var_rep = self._encode_branch_reps(x)
        return self._fuse_rep(patch_rep, var_rep)

    def _pool_global(self, rep: torch.Tensor):
        weights = torch.softmax(self.pool_score(rep), dim=1)
        return (weights * rep).sum(dim=1)

    def _masked_reconstruction_loss(self, x: torch.Tensor):
        mask = (torch.rand_like(x) < self.mask_ratio).float()
        x_masked = x * (1.0 - mask)
        fused_rep = self._encode_fused_rep(x_masked)
        x_recon = self.mae_decoder(fused_rep)
        err = ((x_recon - x) ** 2) * mask
        return err.sum() / (mask.sum() + 1e-8)

    def pretrain_forward(self, x: torch.Tensor):
        device = x.device
        if self.revin:
            x_n, _mu, _sd = self.revin_module.norm(x)
        else:
            x_n = x

        x_v1 = apply_random_aug(x_n, self.augmentation_mode)
        x_v2 = apply_random_aug(x_n, self.augmentation_mode)

        if self.use_mae:
            loss_mae = self._masked_reconstruction_loss(x_v1)
        else:
            loss_mae = torch.tensor(0.0, device=device)

        if self.use_contrastive:
            r1 = self._pool_global(self._encode_fused_rep(x_v1))
            r2 = self._pool_global(self._encode_fused_rep(x_v2))
            B = r1.shape[0]
            p1 = F.normalize(self.proj_head(r1), dim=-1)
            p2 = F.normalize(self.proj_head(r2), dim=-1)
            allp = torch.cat([p1, p2], dim=0)
            sim = allp @ allp.t() / self.temperature
            mask_self = torch.eye(2 * B, device=device, dtype=torch.bool)
            sim.masked_fill_(mask_self, -1e9)
            pos_idx = torch.arange(2 * B, device=device)
            pos_idx = (pos_idx + B) % (2 * B)
            loss_contrast = F.cross_entropy(sim, pos_idx)
        else:
            loss_contrast = torch.tensor(0.0, device=device)

        loss_total = self.mae_weight * loss_mae + self.contrastive_weight * loss_contrast
        return {
            "loss_mae": loss_mae,
            "loss_contrast": loss_contrast,
            "loss_total": loss_total,
        }

    def forward(self, x: torch.Tensor, return_components: bool = False):
        if self.revin:
            x_n, mu, sd = self.revin_module.norm(x)
        else:
            x_n = x

        y_patch, _ = self.patch_branch(x_n)
        y_var, _ = self.variate_branch(x_n)
        weights = torch.softmax(self.fusion_logits, dim=0)
        y = weights[0] * y_patch + weights[1] * y_var

        if self.revin:
            y = self.revin_module.denorm(y, mu, sd)
            y_patch = self.revin_module.denorm(y_patch, mu, sd)
            y_var = self.revin_module.denorm(y_var, mu, sd)
        if return_components:
            return {
                "y": y,
                "y_patch": y_patch,
                "y_var": y_var,
                "patch_weight": weights[0],
                "var_weight": weights[1],
            }
        return y
