"""Summarize parameter-analysis metrics into a flat CSV."""
import argparse
import csv
import json
from pathlib import Path


def read_env(path: Path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key] = value
    return out


def iter_rows(root: Path):
    for metrics_path in sorted(root.glob("*/seed_*/*/metrics.json")):
        run_dir = metrics_path.parent
        config = read_env(run_dir / "run_config.env")
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            if not isinstance(value, dict) or value.get("is_aggregate"):
                continue
            if "dataset" not in value or "pred_len" not in value:
                continue
            yield {
                "suite": config.get("suite", metrics_path.parents[2].name),
                "seed": config.get("seed", run_dir.parents[1].name.replace("seed_", "")),
                "parameter": config.get("parameter", ""),
                "value": config.get("value", run_dir.name),
                "dataset": value.get("dataset"),
                "pred_len": value.get("pred_len"),
                "mse": value.get("mse"),
                "mae": value.get("mae"),
                "epochs_run": value.get("epochs_run"),
                "train_time_min": value.get("train_time_min"),
                "metric_key": key,
                "run_dir": str(run_dir),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="logs/parameter_analysis")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out) if args.out else root / "summary.csv"
    rows = list(iter_rows(root))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "suite", "seed", "parameter", "value", "dataset", "pred_len",
        "mse", "mae", "epochs_run", "train_time_min", "metric_key", "run_dir",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
