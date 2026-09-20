"""Result artifacts: aggregate metrics.json plus one directory per cell."""
import csv
import fcntl
import json
import os

import numpy as np
import torch


def cell_key(dataset, pred_len, model):
    return f"{dataset}_pl{pred_len}__{model}"


def aggregate_key(model, dataset, pred_len):
    return f"{model}__{dataset}__pl{pred_len}"


def cell_dir_for(results_dir, dataset, pred_len, model):
    path = os.path.join(results_dir, "cells", cell_key(dataset, pred_len, model))
    os.makedirs(path, exist_ok=True)
    return path


def write_cell_metrics(cell_dir, model_name, record):
    with open(os.path.join(cell_dir, "metrics.json"), "w") as f:
        json.dump({model_name: record}, f, indent=2)


def write_cell_csv(cell_dir, model_name, record):
    path = os.path.join(cell_dir, "result.csv")
    row = dict(record)
    row["model"] = model_name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return path


def write_curve(cell_dir, curve, stem):
    """Write a per-epoch curve as both json and csv."""
    if not curve:
        return
    with open(os.path.join(cell_dir, f"{stem}.json"), "w") as f:
        json.dump(curve, f, indent=2)
    with open(os.path.join(cell_dir, f"{stem}.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)


def write_checkpoint(cell_dir, model, args, best_val):
    torch.save({
        "state_dict": model.state_dict(),
        "model": args.model,
        "dataset": args.dataset,
        "pred_len": args.pred_len,
        "best_val_mse": best_val,
        "args": vars(args),
    }, os.path.join(cell_dir, "best_validation_checkpoint.pt"))


def write_predictions(cell_dir, args, preds, trues, mse, mae):
    path = os.path.join(
        cell_dir, f"pred_{args.dataset}_pl{args.pred_len}__{args.model}.npz"
    )
    np.savez_compressed(
        path,
        predictions=preds.astype(np.float32),
        trues=trues.astype(np.float32),
        mse=mse, mae=mae,
        model=args.model, dataset=args.dataset, pred_len=args.pred_len,
    )
    return path


def update_aggregate(results_dir, key, record):
    """Merge one cell into metrics.json. Locked and atomic: workers run in parallel."""
    agg_path = os.path.join(results_dir, "metrics.json")
    lock_path = agg_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        agg = {}
        if os.path.exists(agg_path):
            try:
                with open(agg_path) as f:
                    agg = json.load(f)
            except Exception:
                agg = {}
        agg[key] = record

        # Recomputed from scratch: incremental updates would leave stale
        # averages behind when cells are deleted or rerun.
        by_model = {}
        for k, v in agg.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            mdl = v.get("model")
            if not mdl:
                continue
            slot = by_model.setdefault(mdl, {"mse_sum": 0.0, "mae_sum": 0.0, "n": 0})
            slot["mse_sum"] += v["mse"]
            slot["mae_sum"] += v["mae"]
            slot["n"] += 1
        for mdl, d in by_model.items():
            agg[mdl] = {
                "mse": d["mse_sum"] / d["n"],
                "mae": d["mae_sum"] / d["n"],
                "n_cells": d["n"],
                "is_aggregate": True,
            }

        tmp_path = agg_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(agg, f, indent=2)
        os.replace(tmp_path, agg_path)
        fcntl.flock(lock_f, fcntl.LOCK_UN)
    return agg_path
