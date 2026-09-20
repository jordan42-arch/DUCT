"""Summarize cell-level search results. Selection uses best_val_mse, falling
back to test MSE for older runs; both are reported."""
import argparse
import csv
import json
from pathlib import Path


def read_env(path: Path):
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def read_metric(metrics_path: Path):
    with metrics_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if "DUCT" in payload:
        return payload["DUCT"]
    for value in payload.values():
        if isinstance(value, dict) and value.get("model") == "DUCT":
            return value
    return None


def iter_rows(root: Path):
    for run_config in sorted(root.glob("*/*/*/run_config.env")):
        run_dir = run_config.parent
        cfg = read_env(run_config)
        dataset = cfg.get("dataset", run_dir.parents[1].name)
        horizon = int(cfg.get("horizon", run_dir.parent.name.replace("pl", "")))
        cell_metrics = run_dir / "cells" / f"{dataset}_pl{horizon}__DUCT" / "metrics.json"
        if not cell_metrics.exists():
            continue
        metric = read_metric(cell_metrics)
        if not metric:
            continue
        best_val_mse = metric.get("best_val_mse")
        selector_mse = best_val_mse if best_val_mse is not None else metric.get("mse")
        yield {
            "dataset": dataset,
            "horizon": horizon,
            "config_name": cfg.get("config_name", run_dir.name),
            "pretrain_tag": cfg.get("pretrain_tag", ""),
            "selector_mse": selector_mse,
            "selector": "best_val_mse" if best_val_mse is not None else "test_mse",
            "best_val_mse": best_val_mse,
            "best_val_mae": metric.get("best_val_mae"),
            "test_mse": metric.get("mse"),
            "test_mae": metric.get("mae"),
            "epochs_run": metric.get("epochs_run"),
            "train_time_min": metric.get("train_time_min"),
            "branch_loss_weight": metric.get("branch_loss_weight"),
            "fusion_init_patch_weight": metric.get("fusion_init_patch_weight"),
            "freeze_fusion": metric.get("freeze_fusion"),
            "patch_mse": metric.get("patch_mse"),
            "patch_mae": metric.get("patch_mae"),
            "var_mse": metric.get("var_mse"),
            "var_mae": metric.get("var_mae"),
            "fusion_patch_weight": metric.get("fusion_patch_weight"),
            "fusion_var_weight": metric.get("fusion_var_weight"),
            "pretrain_flags": cfg.get("pretrain_flags", ""),
            "finetune_flags": cfg.get("finetune_flags", ""),
            "run_dir": str(run_dir),
        }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "horizon",
        "config_name",
        "pretrain_tag",
        "selector_mse",
        "selector",
        "best_val_mse",
        "best_val_mae",
        "test_mse",
        "test_mae",
        "epochs_run",
        "train_time_min",
        "branch_loss_weight",
        "fusion_init_patch_weight",
        "freeze_fusion",
        "patch_mse",
        "patch_mae",
        "var_mse",
        "var_mae",
        "fusion_patch_weight",
        "fusion_var_weight",
        "pretrain_flags",
        "finetune_flags",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/parameter_analysis/duct_cell_hp_proxy")
    parser.add_argument("--out-all", default=None)
    parser.add_argument("--out-best", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = list(iter_rows(root))
    rows.sort(key=lambda r: (r["dataset"], int(r["horizon"]), float(r["selector_mse"])))

    best = {}
    for row in rows:
        key = (row["dataset"], row["horizon"])
        if key not in best:
            best[key] = row

    all_path = Path(args.out_all) if args.out_all else root / "cell_search_all.csv"
    best_path = Path(args.out_best) if args.out_best else root / "cell_search_best.csv"
    write_csv(all_path, rows)
    write_csv(best_path, [best[k] for k in sorted(best)])

    print(f"wrote {len(rows)} finished runs to {all_path}")
    print(f"wrote {len(best)} best cells to {best_path}")
    for key in sorted(best):
        row = best[key]
        print(
            f"{row['dataset']} pl{row['horizon']}: {row['config_name']} "
            f"selector={float(row['selector_mse']):.6f} "
            f"test_mse={float(row['test_mse']):.6f} test_mae={float(row['test_mae']):.6f}"
        )


if __name__ == "__main__":
    main()
