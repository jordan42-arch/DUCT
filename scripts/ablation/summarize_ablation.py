#!/usr/bin/env python3
"""Summarize DUCT ablation metrics into a CSV file."""
import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "suite",
    "seed",
    "section",
    "variant",
    "dataset",
    "horizon",
    "mse",
    "mae",
    "best_val_mse",
    "best_val_mae",
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
    "run_dir",
]


def read_metric(path: Path):
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    for value in payload.values():
        if isinstance(value, dict) and value.get("model") == "DUCT":
            return value
    value = payload.get("DUCT")
    if isinstance(value, dict) and "mse" in value:
        return value
    return None


def iter_rows(root: Path):
    for metrics_path in root.glob("*/seed_*/*/*/*/pl*/cells/*/metrics.json"):
        metric = read_metric(metrics_path)
        if not metric:
            continue
        # root/suite/seed/section/variant/dataset/plH/cells/key/metrics.json
        rel = metrics_path.relative_to(root)
        suite = rel.parts[0]
        seed = rel.parts[1].replace("seed_", "")
        section = rel.parts[2]
        variant = rel.parts[3]
        dataset = rel.parts[4]
        horizon = rel.parts[5].replace("pl", "")
        run_dir = metrics_path.parents[2]
        yield {
            "suite": suite,
            "seed": seed,
            "section": section,
            "variant": variant,
            "dataset": dataset,
            "horizon": horizon,
            "mse": metric.get("mse"),
            "mae": metric.get("mae"),
            "best_val_mse": metric.get("best_val_mse"),
            "best_val_mae": metric.get("best_val_mae"),
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
            "run_dir": str(run_dir),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/ablation/DUCT")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = sorted(
        iter_rows(root),
        key=lambda r: (
            r["suite"],
            r["section"],
            r["variant"],
            r["dataset"],
            int(r["horizon"]),
        ),
    )
    out = Path(args.out) if args.out else root / "ablation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
