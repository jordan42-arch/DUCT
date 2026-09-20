"""Figures from DUCT outputs. Takes a run directory or the repository root,
preferring logs/comparison and falling back to archived round_2 runs."""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def find_metrics_path(run_dir):
    return first_existing([
        os.path.join(run_dir, "metrics.json"),
        os.path.join(run_dir, "logs", "comparison", "ett", "metrics.json"),
        os.path.join(run_dir, "logs", "metrics.json"),
        os.path.join(run_dir, "experiments", "results", "round_2", "metrics.json"),
    ])


def find_log_path(run_dir):
    return first_existing([
        os.path.join(run_dir, "run.log"),
        os.path.join(run_dir, "logs", "comparison", "ett", "run.log"),
        os.path.join(run_dir, "logs", "run.log"),
        os.path.join(run_dir, "experiments", "results", "round_2", "run.log"),
    ])


def find_prediction_path(run_dir):
    candidates = [
        os.path.join(run_dir, "cells", "ETTh1_pl96__DUCT", "pred_ETTh1_pl96__DUCT.npz"),
        os.path.join(run_dir, "logs", "comparison", "ett", "cells", "ETTh1_pl96__DUCT", "pred_ETTh1_pl96__DUCT.npz"),
        os.path.join(run_dir, "experiments", "results", "round_2", "cells", "ETTh1_pl96__DUCT", "pred_ETTh1_pl96__DUCT.npz"),
    ]
    return first_existing(candidates)


def load_metrics(metrics_path):
    with open(metrics_path) as f:
        return json.load(f)


def plot_horizon_curves(metrics, out_path):
    models = ["Naive", "DLinear", "iTransformer", "PatchTST", "DUCT"]
    horizons = [96, 192, 336, 720]
    avg = {m: [] for m in models}
    for model in models:
        for horizon in horizons:
            cells = [
                v["mse"] for v in metrics.values()
                if isinstance(v, dict) and v.get("model") == model and v.get("pred_len") == horizon
            ]
            avg[model].append(float(np.mean(cells)) if cells else np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {
        "Naive": "#888888",
        "DLinear": "#1f77b4",
        "iTransformer": "#2ca02c",
        "PatchTST": "#ff7f0e",
        "DUCT": "#d62728",
    }
    markers = {"Naive": "o", "DLinear": "s", "iTransformer": "^", "PatchTST": "D", "DUCT": "*"}
    for model in models:
        ax.plot(horizons, avg[model], color=colors[model], marker=markers[model], label=model, linewidth=1.5)
    ax.set_xlabel("Prediction horizon H")
    ax.set_ylabel("MSE")
    ax.set_title("Forecasting MSE vs horizon")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pretrain_loss(log_path, out_path):
    pattern = re.compile(r"\[ep\s+(\d+)/(\d+)\] avg total=([\d.]+) mae=([\d.]+) (?:ctr|contrast)=([\d.]+)")
    eps, total, mae, contrast = [], [], [], []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                eps.append(int(match.group(1)))
                total.append(float(match.group(3)))
                mae.append(float(match.group(4)))
                contrast.append(float(match.group(5)))
    if not eps:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(eps, total, label="Total", color="#d62728", linewidth=2)
    ax.plot(eps, mae, label="MAE reconstruction", color="#1f77b4", linewidth=1.5, linestyle="--")
    ax.plot(eps, contrast, label="InfoNCE contrastive", color="#2ca02c", linewidth=1.5, linestyle=":")
    ax.set_xlabel("Pretraining epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DUCT pretraining loss decomposition")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tsne_channels(npz_path, out_path):
    from sklearn.manifold import TSNE

    data = np.load(npz_path)
    preds = data["predictions"]
    n_samples, n_vars, horizon = preds.shape
    reps = preds.reshape(n_samples * n_vars, horizon)
    labels = np.repeat(np.arange(n_vars), n_samples).reshape(n_vars, n_samples).T.reshape(-1)
    rng = np.random.default_rng(0)
    if reps.shape[0] > 1000:
        idx = rng.choice(reps.shape[0], 1000, replace=False)
        reps = reps[idx]
        labels = labels[idx]

    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=0, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=0, n_iter=1000)
    proj = tsne.fit_transform(reps)
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = plt.cm.tab10(np.linspace(0, 1, n_vars))
    for ch in range(n_vars):
        mask = labels == ch
        ax.scatter(proj[mask, 0], proj[mask, 1], c=[palette[ch]], label=f"ch{ch}", s=14, alpha=0.6)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("DUCT test prediction signatures per channel")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = find_metrics_path(args.run_dir)
    if metrics_path:
        plot_horizon_curves(load_metrics(metrics_path), os.path.join(out_dir, "fig_horizon_curves.pdf"))
        print(f"saved {os.path.join(out_dir, 'fig_horizon_curves.pdf')}")
    else:
        print(f"missing metrics.json under {args.run_dir}")

    log_path = find_log_path(args.run_dir)
    if log_path:
        out_path = os.path.join(out_dir, "fig_pretrain_loss.pdf")
        plot_pretrain_loss(log_path, out_path)
        if os.path.exists(out_path):
            print(f"saved {out_path}")

    npz_path = find_prediction_path(args.run_dir)
    if npz_path:
        out_path = os.path.join(out_dir, "fig_tsne_channels.pdf")
        plot_tsne_channels(npz_path, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
