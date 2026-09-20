"""Forecasting datasets. ETT uses the standard fixed split, everything else a
ratio split, so all of them run through the same entrypoints."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# (train, val, test) lengths per the iTransformer paper: 12/4/4 months.
SPLITS = {
    "ETTh1": (8640, 2880, 2880),
    "ETTh2": (8640, 2880, 2880),
    "ETTm1": (34560, 11520, 11520),
    "ETTm2": (34560, 11520, 11520),
}

DATASET_REGISTRY = {
    "ETTh1": {"path": "ETT-small/ETTh1.csv", "format": "csv", "split": "ett"},
    "ETTh2": {"path": "ETT-small/ETTh2.csv", "format": "csv", "split": "ett"},
    "ETTm1": {"path": "ETT-small/ETTm1.csv", "format": "csv", "split": "ett"},
    "ETTm2": {"path": "ETT-small/ETTm2.csv", "format": "csv", "split": "ett"},
    "electricity": {"path": "electricity/electricity.csv", "format": "csv", "split": "ratio_7_1_2"},
    "traffic": {"path": "traffic/traffic.csv", "format": "csv", "split": "ratio_7_1_2"},
    "weather": {"path": "weather/weather.csv", "format": "csv", "split": "ratio_7_1_2"},
    "exchange_rate": {"path": "exchange_rate/exchange_rate.csv", "format": "csv", "split": "ratio_7_1_2"},
    "solar": {"path": "Solar/solar_AL.txt", "format": "txt", "split": "ratio_7_1_2"},
    "PEMS03": {"path": "PEMS/PEMS03.npz", "format": "npz", "split": "ratio_6_2_2"},
    "PEMS04": {"path": "PEMS/PEMS04.npz", "format": "npz", "split": "ratio_6_2_2"},
    "PEMS07": {"path": "PEMS/PEMS07.npz", "format": "npz", "split": "ratio_6_2_2"},
    "PEMS08": {"path": "PEMS/PEMS08.npz", "format": "npz", "split": "ratio_6_2_2"},
    # ByteDance cloud workload, 10-minute granularity; see scripts/data/prepare_cloudts.py.
    "FaaS_Small": {"path": "CloudTS/FaaS_Small.csv", "format": "csv", "split": "ratio_7_1_2"},
    "FaaS_Medium": {"path": "CloudTS/FaaS_Medium.csv", "format": "csv", "split": "ratio_7_1_2"},
    "FaaS_Large": {"path": "CloudTS/FaaS_Large.csv", "format": "csv", "split": "ratio_7_1_2"},
    "IaaS_Small": {"path": "CloudTS/IaaS_Small.csv", "format": "csv", "split": "ratio_7_1_2"},
    "IaaS_Medium": {"path": "CloudTS/IaaS_Medium.csv", "format": "csv", "split": "ratio_7_1_2"},
    "IaaS_Large": {"path": "CloudTS/IaaS_Large.csv", "format": "csv", "split": "ratio_7_1_2"},
}

SUPPORTED_DATASETS = tuple(DATASET_REGISTRY.keys())


def resolve_data_file(data_dir: str, dataset_name: str) -> str:
    """Resolve a dataset file while preserving the old ETT-small data_dir API."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"unknown dataset: {dataset_name}")
    if os.path.isfile(data_dir):
        return data_dir

    rel = DATASET_REGISTRY[dataset_name]["path"]
    candidates = [
        os.path.join(data_dir, rel),
        os.path.join(data_dir, os.path.basename(rel)),
        os.path.join("datasets", rel),
    ]
    if dataset_name in SPLITS:
        candidates.insert(0, os.path.join(data_dir, f"{dataset_name}.csv"))

    for path in candidates:
        if os.path.exists(path):
            return path
    tried = "\n  ".join(candidates)
    raise FileNotFoundError(f"could not locate {dataset_name}; tried:\n  {tried}")


def _load_array(csv_path: str, dataset_name: str, features: str) -> np.ndarray:
    spec = DATASET_REGISTRY[dataset_name]
    if spec["format"] == "csv":
        df = pd.read_csv(csv_path)
        df = df.drop(columns=["date"], errors="ignore")
        if features == "S" and "OT" in df.columns:
            df = df[["OT"]]
        return df.values.astype(np.float32)
    if spec["format"] == "txt":
        df = pd.read_csv(csv_path, header=None)
        return df.values.astype(np.float32)
    if spec["format"] == "npz":
        data = np.load(csv_path)["data"].astype(np.float32)
        if data.ndim == 3:
            data = data[..., 0]
        if data.ndim != 2:
            raise ValueError(f"expected 2D time-series array for {dataset_name}, got shape {data.shape}")
        return data
    raise ValueError(f"unsupported dataset format: {spec['format']}")


def _split_bounds(dataset_name: str, n_time: int):
    split = DATASET_REGISTRY[dataset_name]["split"]
    if split == "ett":
        train_size, val_size, test_size = SPLITS[dataset_name]
        train_end = train_size
        val_end = train_end + val_size
        test_end = val_end + test_size
    elif split == "ratio_6_2_2":
        train_end = int(n_time * 0.6)
        val_end = int(n_time * 0.8)
        test_end = n_time
    else:
        train_end = int(n_time * 0.7)
        val_end = int(n_time * 0.8)
        test_end = n_time
    return train_end, val_end, test_end


class ForecastDataset(Dataset):
    """Sliding windows in channel-independent layout: x (V, lookback), y (V, pred_len).

    features: 'M' multivariate, 'S' the OT column only.
    """

    def __init__(self, csv_path: str, dataset_name: str, split: str = "train",
                 lookback: int = 96, pred_len: int = 96, scale: bool = True,
                 features: str = "M"):
        assert split in ("train", "val", "test")
        assert dataset_name in DATASET_REGISTRY

        data = _load_array(csv_path, dataset_name, features)
        self.n_vars = data.shape[1]
        train_end, val_end, test_end = _split_bounds(dataset_name, len(data))

        # Fit scaler on train only
        if scale:
            mu = data[:train_end].mean(axis=0, keepdims=True)
            sd = data[:train_end].std(axis=0, keepdims=True) + 1e-5
            data = (data - mu) / sd
            self.mu = mu
            self.sd = sd
        else:
            self.mu = np.zeros((1, self.n_vars), dtype=np.float32)
            self.sd = np.ones((1, self.n_vars), dtype=np.float32)

        # iTransformer convention: val/test slices carry lookback context overlap.
        if split == "train":
            slice_data = data[:train_end]
        elif split == "val":
            slice_data = data[train_end - lookback : val_end]
        else:
            slice_data = data[val_end - lookback : test_end]
        self.data = slice_data  # (T_split, V)
        self.lookback = lookback
        self.pred_len = pred_len

    def __len__(self):
        n = len(self.data) - self.lookback - self.pred_len + 1
        return max(0, n)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.lookback]  # (L, V)
        y = self.data[idx + self.lookback : idx + self.lookback + self.pred_len]  # (pred_len, V)
        # convert to (V, L) and (V, pred_len) -- channel-independent layout
        x = torch.from_numpy(x).T.contiguous()
        y = torch.from_numpy(y).T.contiguous()
        return x, y


# Backward-compatible alias used by older training scripts and notebooks.
ETTDataset = ForecastDataset
