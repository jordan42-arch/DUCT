# Reproducibility Notes

## What this release contains

This release contains the DUCT training source, baseline adapters, and stored
experiment scripts. It does not bundle datasets, pretrained checkpoints, raw
training logs, private credentials, or the manuscript source.

Record the repository commit, environment, random seed, full command, and
dataset preprocessing alongside any new results. Smoke tests verify basic
execution and tensor contracts, not forecasting accuracy.

## Environment

The code dependency manifest specifies PyTorch >= 2.0. This is a code-runtime
requirement, not a claim about the exact environment that produced every
historical result. Use a CUDA-compatible PyTorch build for GPU training.

## Data and evaluation

- ETT uses fixed chronological 12/4/4-month splits.
- PEMS uses chronological 6:2:2 splits.
- Other registered datasets use chronological 7:1:2 splits.
- Scaling statistics are fit on training data only.
- Validation and test inputs include preceding lookback context.
- Metrics are computed on the shared loader's standardized targets.
- Forecast windows overlap; reproduce the same lookback and horizon.
- PEMS NPZ inputs use the first feature when the array has three dimensions.

Cloud Small/Medium subsets are deterministic prefixes defined by
`scripts/data/prepare_cloudts.py`, not separately downloaded datasets.

## Training

The dataset-specific DUCT runners call a 20-epoch pretraining stage by default.
The standalone `pretrain.py` parser defaults to 100 epochs; pass `--epochs 20`
when reproducing the runner setting. Encoder width, depth, patch length, stride,
dropout, and lookback must match the checkpoint being loaded.

The model has independent patch and variate encoders. During pretraining,
per-variable fused features are pooled using learned scores before contrastive
projection. During forecasting, branch predictions are combined with global
learned softmax weights. See `models/DUCT.py` for the exact operations.

Dataset-specific scripts contain different settings for different horizons.
Do not replace them with a single default configuration and expect identical
results. Skipping existing metrics avoids repeated computation but does not
verify that a previous run used the current hyperparameters. Use a fresh output
directory for a new configuration.

## Baseline and result coverage

The release includes nine baseline options; the manuscript does not necessarily
report every supported model or dataset. Exchange Rate and TSMixer are supported
by the exploratory scripts. The comparison tool may exclude incomplete cells;
inspect its report before averaging scores.

Crossformer is not part of the current registry. No matching Crossformer metrics
were found in the local historical logs or supplied result archives during
release preparation. Its paper scores must not be presented as reruns under
this repository's evaluation protocol. Add and run a compatible adapter before
adding a Crossformer result column.

## Source attribution

Consult `THIRD_PARTY_NOTICES.md` and the original model papers before modifying
or redistributing baseline components. Dataset rights are separate from code
rights.
