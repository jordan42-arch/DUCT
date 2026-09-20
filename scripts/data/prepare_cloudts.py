#!/usr/bin/env python3
"""Convert ByteDance CloudTimeSeriesData into the wide CSVs this repo loads.

Source: https://huggingface.co/datasets/ByteDance/CloudTimeSeriesData
Paper:  E3Former, arXiv:2508.12773

The release is TFB long format (date, data, cols) at 10-minute granularity.
This pivots it to one column per instance and slices the Small/Medium/Large
variate subsets used in the paper.
"""
import argparse
import os
import re

import pandas as pd

# Variate counts from Table 3 of the paper. Large equals the full release, so
# only Small and Medium are actually subsets.
SUBSETS = {
    "faas": {"FaaS_Small": 7, "FaaS_Medium": 93, "FaaS_Large": 226},
    "iaas": {"IaaS_Small": 7, "IaaS_Medium": 58, "IaaS_Large": 93},
}


def natural_key(name):
    """instance_2 sorts before instance_10."""
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 0, name)


def load_wide(path):
    """TFB long format -> (T, V) frame indexed by date, columns naturally sorted."""
    df = pd.read_csv(path)
    missing = {"date", "data", "cols"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: expected TFB long format, missing {sorted(missing)}")
    wide = df.pivot(index="date", columns="cols", values="data")
    wide = wide[sorted(wide.columns, key=natural_key)]
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="directory holding faas.csv / iaas.csv")
    ap.add_argument("--out_dir", default="datasets/CloudTS")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for service, subsets in SUBSETS.items():
        wide = load_wide(os.path.join(args.raw_dir, f"{service}.csv"))
        step = wide.index.to_series().diff().dropna().mode()[0]
        print(f"{service}.csv -> {wide.shape[0]} steps x {wide.shape[1]} variates, "
              f"step={step}, {wide.index[0]} .. {wide.index[-1]}")

        for name, n_vars in subsets.items():
            if n_vars > wide.shape[1]:
                raise ValueError(f"{name} wants {n_vars} variates, release has {wide.shape[1]}")
            # Small/Medium are the first N instances. The release does not define
            # the subsets, so a deterministic prefix keeps them reproducible.
            sub = wide.iloc[:, :n_vars]
            out = os.path.join(args.out_dir, f"{name}.csv")
            sub.to_csv(out, index_label="date")
            print(f"  {name:14s} {sub.shape[0]} x {sub.shape[1]}  -> {out}")


if __name__ == "__main__":
    main()
