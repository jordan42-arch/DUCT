"""Pretraining: masked reconstruction + contrastive learning."""
import json
import os
import time

import torch

from data_provider import build_loader, build_pretrain_pool
from exp.exp_basic import Exp_Basic
from utils import results as R

# Excluded from the saved backbone: cell-specific, re-initialized at finetune.
FORECAST_PREFIXES = (
    "forecast_head.",
    "patch_branch.time_head.",
    "variate_branch.head.",
)
FORECAST_KEYS = {"fusion_logits"}


class Exp_Pretrain(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.contrastive_weight = (
            0.5 if args.contrastive_weight is None else args.contrastive_weight
        )
        self.augmentation_mode = "none" if args.no_augmentation else args.augmentation_mode
        self.loss_curve = []

    # ------------------------------------------------------------------- data

    def get_data(self):
        args = self.args
        pool, n_vars = build_pretrain_pool(
            args.data_dir, args.datasets.split(","), args.lookback
        )
        self.n_vars = n_vars
        self.pool = pool
        self.loader = build_loader(pool, args.batch, shuffle=True, drop_last=True)

    # ------------------------------------------------------------------ model

    def build_model(self, n_vars):
        args = self.args
        return self.build_duct(
            n_vars,
            pred_len=1,  # unused in pretrain; only x is consumed
            mask_ratio=args.mask_ratio,
            mae_weight=args.mae_weight,
            contrastive_weight=self.contrastive_weight,
            temperature=args.temperature,
            use_contrastive=not args.no_contrastive,
            use_mae=not args.no_mae,
            use_augmentation=not args.no_augmentation,
            augmentation_mode=self.augmentation_mode,
        )

    # --------------------------------------------------------------- training

    def train(self):
        args = self.args
        opt, sched = self.build_optimizer(self.model, args.epochs)
        print(f"Starting pretrain: {args.epochs} epochs over {len(self.pool):,} samples, "
              f"batch={args.batch}")

        t0 = time.time()
        for epoch in range(args.epochs):
            self.model.train()
            sums = {"loss_total": 0.0, "loss_mae": 0.0, "loss_contrast": 0.0}
            n = 0
            for step, (x, _y) in enumerate(self.loader):
                x = x.to(self.device, non_blocking=True)
                out = self.model.pretrain_forward(x)
                self.optimizer_step(opt, out["loss_total"], self.model)
                for key in sums:
                    sums[key] += float(out[key].detach())
                n += 1
                if step % args.log_every == 0:
                    print(f"  ep {epoch+1:3d} step {step:4d}: "
                          f"total={float(out['loss_total'].detach()):.4f} "
                          f"mae={float(out['loss_mae'].detach()):.4f} "
                          f"contrast={float(out['loss_contrast'].detach()):.4f} "
                          f"lr={opt.param_groups[0]['lr']:.6g}")
            sched.step()
            elapsed = time.time() - t0
            row = {
                "epoch": epoch + 1,
                "learning_rate": opt.param_groups[0]["lr"],
                "elapsed_min": elapsed / 60,
            }
            row.update({key: value / max(n, 1) for key, value in sums.items()})
            self.loss_curve.append(row)
            print(f"[ep {epoch+1:3d}/{args.epochs}] avg "
                  f"total={row['loss_total']:.4f} mae={row['loss_mae']:.4f} "
                  f"contrast={row['loss_contrast']:.4f} "
                  f"lr={row['learning_rate']:.6g} | elapsed {elapsed/60:.1f}min")

    # ---------------------------------------------------------------- results

    def backbone_state_dict(self):
        return {
            k: v for k, v in self.model.state_dict().items()
            if not k.startswith(FORECAST_PREFIXES) and k not in FORECAST_KEYS
        }

    def save(self):
        args = self.args
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save({
            "state_dict": self.backbone_state_dict(),
            "model": args.model,
            "n_vars": self.n_vars,
            "lookback": args.lookback,
            "patch_len": args.patch_len,
            "stride": args.stride,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ff": args.d_ff,
            "dropout": args.dropout,
            "mask_ratio": args.mask_ratio,
            "mae_weight": args.mae_weight,
            "contrastive_weight": self.contrastive_weight,
            "temperature": args.temperature,
            "augmentation_mode": self.augmentation_mode,
            "pretrain_args": vars(args),
        }, args.save_path)
        print(f"Saved pretrained backbone to {args.save_path}")

        out_dir = os.path.dirname(args.save_path)
        config_path = os.path.join(out_dir, "pretrain_config.json")
        with open(config_path, "w") as f:
            json.dump({"args": vars(args)}, f, indent=2)
        R.write_curve(out_dir, self.loss_curve, "pretrain_loss_curve")
        print(f"Saved pretrain config to {config_path}")
        print(f"Saved pretrain loss curve to {os.path.join(out_dir, 'pretrain_loss_curve.csv')}")

    # -------------------------------------------------------------------- run

    def run(self):
        print(f"Device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.get_data()
        self.model = self.build_model(self.n_vars).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Pretrain model params: {n_params:,}")
        self.train()
        self.save()
