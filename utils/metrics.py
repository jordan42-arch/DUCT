"""Forecast evaluation. Errors sum over elements, not batches, so batch size
does not affect the result."""
import numpy as np
import torch


def unpack_prediction(out):
    """Models may return a tensor or a component dict; take the forecast."""
    if isinstance(out, dict):
        return out["y"]
    return out


@torch.no_grad()
def evaluate(model, loader, device, collect_predictions=False):
    """Return (mse, mae, preds, trues) over a loader."""
    model.eval()
    preds, trues = [], []
    sum_sq = 0.0
    sum_abs = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        yhat = unpack_prediction(model(x))
        diff = yhat - y
        sum_sq += torch.sum(diff * diff).item()
        sum_abs += torch.sum(torch.abs(diff)).item()
        count += diff.numel()
        if collect_predictions:
            preds.append(yhat.cpu().numpy())
            trues.append(y.cpu().numpy())
    mse = float(sum_sq / count)
    mae = float(sum_abs / count)
    if collect_predictions:
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
    else:
        preds = None
        trues = None
    return mse, mae, preds, trues


@torch.no_grad()
def evaluate_branch_components(model, loader, device):
    """Fused + per-branch metrics and fusion weights; read by the branch ablations."""
    model.eval()
    first_branch, second_branch = getattr(model, "branch_names", ("patch", "var"))
    sums = {
        "fused_sq": 0.0,
        "fused_abs": 0.0,
        f"{first_branch}_sq": 0.0,
        f"{first_branch}_abs": 0.0,
        f"{second_branch}_sq": 0.0,
        f"{second_branch}_abs": 0.0,
        "count": 0,
    }
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, return_components=True)
        for name, pred_key in (
            ("fused", "y"),
            (first_branch, f"y_{first_branch}"),
            (second_branch, f"y_{second_branch}"),
        ):
            diff = out[pred_key] - y
            sums[f"{name}_sq"] += torch.sum(diff * diff).item()
            sums[f"{name}_abs"] += torch.sum(torch.abs(diff)).item()
        sums["count"] += y.numel()
    count = sums["count"]
    weights = torch.softmax(model.fusion_logits.detach().cpu(), dim=0)
    return {
        "mse": float(sums["fused_sq"] / count),
        "mae": float(sums["fused_abs"] / count),
        f"{first_branch}_mse": float(sums[f"{first_branch}_sq"] / count),
        f"{first_branch}_mae": float(sums[f"{first_branch}_abs"] / count),
        f"{second_branch}_mse": float(sums[f"{second_branch}_sq"] / count),
        f"{second_branch}_mae": float(sums[f"{second_branch}_abs"] / count),
        f"fusion_{first_branch}_weight": float(weights[0]),
        f"fusion_{second_branch}_weight": float(weights[1]),
    }
