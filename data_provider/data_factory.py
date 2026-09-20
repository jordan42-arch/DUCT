"""Dataset/DataLoader construction, kept in one place so the two training
stages cannot drift apart."""
from torch.utils.data import ConcatDataset, DataLoader

from .data_loader import ForecastDataset, resolve_data_file

NUM_WORKERS = 2


def build_dataset(data_dir, dataset_name, split, lookback, pred_len,
                  scale=True, features="M"):
    """One split of one dataset."""
    csv_path = resolve_data_file(data_dir, dataset_name)
    return ForecastDataset(
        csv_path=csv_path,
        dataset_name=dataset_name,
        split=split,
        lookback=lookback,
        pred_len=pred_len,
        scale=scale,
        features=features,
    )


def build_forecast_splits(data_dir, dataset_name, lookback, pred_len):
    """Train/val/test splits for a single forecasting cell."""
    return tuple(
        build_dataset(data_dir, dataset_name, split, lookback, pred_len)
        for split in ("train", "val", "test")
    )


def build_loader(dataset, batch_size, shuffle, drop_last=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=drop_last,
    )


def build_pretrain_pool(data_dir, dataset_names, lookback, verbose=True):
    """Unlabeled pretraining pool. pred_len=1 since only x is consumed, and the
    pool must share a channel count because one backbone spans it."""
    dsets = []
    n_vars = None
    for name in dataset_names:
        name = name.strip()
        ds = build_dataset(data_dir, name, "train", lookback, pred_len=1)
        if n_vars is None:
            n_vars = ds.n_vars
        elif ds.n_vars != n_vars:
            raise ValueError(
                f"channel count mismatch in pretrain pool: {name} has {ds.n_vars}, expected {n_vars}"
            )
        dsets.append(ds)
        if verbose:
            print(f"  {name}: {len(ds)} samples, V={ds.n_vars}")
    return ConcatDataset(dsets), n_vars
