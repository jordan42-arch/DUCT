"""Finetuning one cell. A baseline is the same path without a pretrained
backbone and without auxiliary branch losses."""
import time

import torch
import torch.nn as nn

from data_provider import build_forecast_splits, build_loader
from exp.exp_basic import Exp_Basic
from models import build_baseline
from utils import EarlyStopping, evaluate, evaluate_branch_components
from utils import results as R

# The backbone transfers across horizons; heads and fusion do not.
FORECAST_HEAD_PREFIXES = ("patch_branch.time_head.", "variate_branch.head.")
FORECAST_HEAD_KEYS = {"fusion_logits"}
# A checkpoint missing any of these came from a different architecture.
REQUIRED_BACKBONE_PREFIXES = ("patch_branch.", "variate_branch.", "rep_fusion.")


class Exp_Finetune(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.model = None
        self.n_params = 0
        self.epochs_run = 0
        self.train_time_min = 0.0
        self.val_curve = []
        self.best_val = float("inf")
        self.best_val_mae = None
        self.component_metrics = None

    # ------------------------------------------------------------------- data

    def get_data(self):
        args = self.args
        ds_tr, ds_va, ds_te = build_forecast_splits(
            args.data_dir, args.dataset, args.lookback, args.pred_len
        )
        print(f"  Train={len(ds_tr)} Val={len(ds_va)} Test={len(ds_te)} V={ds_tr.n_vars}")
        self.n_vars = ds_tr.n_vars
        self.loader_tr = build_loader(ds_tr, args.batch, shuffle=True, drop_last=True)
        self.loader_va = build_loader(ds_va, args.batch, shuffle=False)
        self.loader_te = build_loader(ds_te, args.batch, shuffle=False)

    # ------------------------------------------------------------------ model

    def _configure_fusion(self, model):
        args = self.args
        if not hasattr(model, "fusion_logits"):
            return
        if args.fusion_init_patch_weight is not None:
            w = float(args.fusion_init_patch_weight)
            if not 0.0 <= w <= 1.0:
                raise ValueError("--fusion_init_patch_weight must be in [0, 1]")
            eps = 1e-4
            w = min(max(w, eps), 1.0 - eps)
            logits = torch.log(torch.tensor([w, 1.0 - w], dtype=model.fusion_logits.dtype))
            with torch.no_grad():
                model.fusion_logits.copy_(logits.to(model.fusion_logits.device))
        if args.freeze_fusion:
            model.fusion_logits.requires_grad_(False)

    def _load_pretrained(self, model, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_model = ckpt.get("model")
        if ckpt_model and ckpt_model != self.args.model:
            raise ValueError(
                f"checkpoint model mismatch: checkpoint={ckpt_model}, requested={self.args.model}"
            )
        sd = ckpt["state_dict"]
        missing_prefixes = [
            prefix for prefix in REQUIRED_BACKBONE_PREFIXES
            if not any(k.startswith(prefix) for k in sd)
        ]
        if missing_prefixes:
            raise ValueError(
                "checkpoint is not compatible with the current dual-path DUCT "
                f"(missing prefixes: {missing_prefixes}). Re-run pretraining."
            )
        backbone_sd = {
            k: v for k, v in sd.items()
            if not k.startswith(FORECAST_HEAD_PREFIXES) and k not in FORECAST_HEAD_KEYS
        }
        missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
        print(f"Loaded pretrained backbone from {path}")
        print(f"  missing keys: {len(missing)} (forecast heads expected to be missing)")
        print(f"  unexpected keys: {len(unexpected)}")

    def build_model(self, n_vars):
        args = self.args
        if not self.uses_pretraining():
            return build_baseline(args.model, n_vars, args.lookback, args.pred_len)
        model = self.build_duct(
            n_vars,
            args.pred_len,
            use_contrastive=True,
            use_mae=True,
            use_augmentation=True,
        )
        if args.pretrained_path:
            self._load_pretrained(model, args.pretrained_path)
        if self.is_dual_branch():
            self._configure_fusion(model)
        return model

    # --------------------------------------------------------------- training

    def _batch_loss(self, x, y, crit):
        if self.is_dual_branch():
            out = self.model(x, return_components=True)
            loss_fused = crit(out["y"], y)
            loss_patch = crit(out["y_patch"], y)
            loss_var = crit(out["y_var"], y)
            return loss_fused + self.args.branch_loss_weight * (loss_patch + loss_var)
        return crit(self.model(x), y)

    def train(self):
        args = self.args
        opt, sched = self.build_optimizer(self.model, args.epochs)
        crit = nn.MSELoss()
        stopper = EarlyStopping(args.patience)
        train_metric_name = "train_obj" if self.is_dual_branch() else "train_mse"

        t0 = time.time()
        for epoch in range(args.epochs):
            self.model.train()
            sum_loss = 0.0
            n = 0
            for x, y in self.loader_tr:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                loss = self._batch_loss(x, y, crit)
                self.optimizer_step(opt, loss, self.model)
                sum_loss += loss.item()
                n += 1
            sched.step()
            train_loss = sum_loss / n
            val_mse, val_mae, _, _ = evaluate(self.model, self.loader_va, self.device)
            print(f"  ep {epoch+1:3d}/{args.epochs}: {train_metric_name}={train_loss:.4f} "
                  f"val_mse={val_mse:.4f} val_mae={val_mae:.4f}")
            self.epochs_run += 1
            self.val_curve.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "forecast_mse": val_mse,
                "forecast_mae": val_mae,
                "learning_rate": opt.param_groups[0]["lr"],
            })
            if stopper.step(val_mse, self.model, aux=val_mae):
                print(f"  early stopping at ep {epoch+1}")
                break

        self.train_time_min = (time.time() - t0) / 60.0
        stopper.restore(self.model)
        self.best_val = stopper.best_score
        self.best_val_mae = stopper.best_aux

    # ------------------------------------------------------------------- test

    def test(self):
        collect = self.args.save_predictions
        if self.is_dual_branch():
            self.component_metrics = evaluate_branch_components(
                self.model, self.loader_te, self.device
            )
            mse = self.component_metrics["mse"]
            mae = self.component_metrics["mae"]
            preds = trues = None
            if collect:
                mse, mae, preds, trues = evaluate(
                    self.model, self.loader_te, self.device, collect_predictions=True
                )
            return mse, mae, preds, trues
        return evaluate(self.model, self.loader_te, self.device, collect_predictions=collect)

    # ---------------------------------------------------------------- results

    def _common_fields(self, mse, mae):
        dual = self.is_dual_branch()
        return {
            "mse": mse,
            "mae": mae,
            "params": self.n_params,
            "epochs_run": self.epochs_run,
            "train_time_min": self.train_time_min,
            "best_val_mse": None if self.best_val == float("inf") else self.best_val,
            "best_val_mae": self.best_val_mae,
            "lookback": self.args.lookback,
            "branch_loss_weight": self.args.branch_loss_weight if dual else None,
            "fusion_init_patch_weight": self.args.fusion_init_patch_weight if dual else None,
            "freeze_fusion": self.args.freeze_fusion if dual else None,
        }

    def _with_components(self, record):
        if self.is_dual_branch() and self.component_metrics is not None:
            record.update({
                k: v for k, v in self.component_metrics.items() if k not in {"mse", "mae"}
            })
        return record

    def cell_record(self, mse, mae):
        """Self-describing per-cell record. Key order sets the result.csv header."""
        args = self.args
        c = self._common_fields(mse, mae)
        return self._with_components({
            "mse": c["mse"],
            "mae": c["mae"],
            "params": c["params"],
            "epochs_run": c["epochs_run"],
            "train_time_min": c["train_time_min"],
            "best_val_mse": c["best_val_mse"],
            "best_val_mae": c["best_val_mae"],
            "lookback": c["lookback"],
            "pred_len": args.pred_len,
            "dataset": args.dataset,
            "seed": args.seed,
            "branch_loss_weight": c["branch_loss_weight"],
            "fusion_init_patch_weight": c["fusion_init_patch_weight"],
            "freeze_fusion": c["freeze_fusion"],
        })

    def aggregate_record(self, mse, mae):
        """Row merged into <results_dir>/metrics.json."""
        args = self.args
        c = self._common_fields(mse, mae)
        return self._with_components({
            "model": args.model,
            "dataset": args.dataset,
            "pred_len": args.pred_len,
            "seed": args.seed,
            **c,
        })

    def save(self, mse, mae, preds, trues):
        args = self.args
        cell_dir = R.cell_dir_for(args.results_dir, args.dataset, args.pred_len, args.model)

        record = self.cell_record(mse, mae)
        R.write_cell_metrics(cell_dir, args.model, record)
        result_csv_path = R.write_cell_csv(cell_dir, args.model, record)

        if self.val_curve:
            R.write_curve(cell_dir, self.val_curve, "finetune_val_curve")
            R.write_checkpoint(cell_dir, self.model, args, self.best_val)

        npz_path = None
        if args.save_predictions:
            npz_path = R.write_predictions(cell_dir, args, preds, trues, mse, mae)

        agg_record = self.aggregate_record(mse, mae)
        agg_key = R.aggregate_key(args.model, args.dataset, args.pred_len)
        agg_path = R.update_aggregate(args.results_dir, agg_key, agg_record)

        print(f"  Wrote cell metrics to {cell_dir}/metrics.json")
        print(f"  Wrote cell result CSV to {result_csv_path}")
        if npz_path:
            print(f"  Wrote cell preds to {npz_path}")
        else:
            # Off by default: traffic/PEMS produce tens of GB per cell.
            print("  Skipped cell preds npz (--save_predictions not set)")
        print(f"  Updated aggregate metrics: {agg_path} (key={agg_key})")

    # -------------------------------------------------------------------- run

    def run(self):
        args = self.args
        print(f"[{args.model} | {args.dataset} | pl={args.pred_len}] Device={self.device}")
        self.get_data()
        R.cell_dir_for(args.results_dir, args.dataset, args.pred_len, args.model)

        self.model = self.build_model(self.n_vars).to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Model params: {self.n_params:,}")

        if args.model == "Naive":
            # Closed form; training it would only fit the dummy parameter.
            mse, mae, preds, trues = evaluate(
                self.model, self.loader_te, self.device,
                collect_predictions=args.save_predictions,
            )
            print(f"  Naive Test: MSE={mse:.4f} MAE={mae:.4f}")
        else:
            self.train()
            mse, mae, preds, trues = self.test()
            print(f"  Test: MSE={mse:.4f} MAE={mae:.4f}  "
                  f"(train_time={self.train_time_min:.1f}min)")

        self.save(mse, mae, preds, trues)
