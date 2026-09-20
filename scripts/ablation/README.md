# DUCT Ablation Scripts

These scripts keep the tuned DUCT hyperparameters fixed for each dataset and
horizon, then change one methodological factor at a time.

## Recommended Design

Use `run_core_ablation.sh` as the main paper ablation:

- `full`: tuned DUCT.
- `mae_only`: remove contrastive pretraining.
- `contrastive_only`: remove masked reconstruction.
- `no_pretrain`: train the same architecture from scratch.
- `no_branch_aux`: remove auxiliary branch forecast losses.
- `fixed_equal_fusion`: freeze patch/variate fusion to 0.5/0.5.
- `patch_only`: use only the channel-independent patch branch at output.
- `var_only`: use only the variate-interaction branch at output.
- `no_augmentation`: disable contrastive augmentation.
- `no_time_warp`: keep noise and scaling, remove time warping.

Use `run_highvar_ablation.sh` as a focused table for the variate-interaction
claim on high-variate datasets.

Use `run_objective_ablation.sh` as a supplementary table for pretraining
objective and augmentation choices.

Use `run_broad_ablation.sh` when you want a wider search for datasets where the
ablation story is cleanest. It covers short and long horizons from ETT,
Electricity, Weather, Exchange, Solar, Traffic, and PEMS.

## Commands

From the repository root:

```bash
bash scripts/ablation/run_core_ablation.sh
bash scripts/ablation/run_highvar_ablation.sh
bash scripts/ablation/run_objective_ablation.sh
bash scripts/ablation/run_broad_ablation.sh
```

Override cells or variants when needed:

```bash
CORE_CELLS="ETTh1:96 ETTm1:720" \
ABLATION_VARIANTS="full mae_only no_pretrain patch_only var_only" \
bash scripts/ablation/run_core_ablation.sh
```

Summarize finished runs:

```bash
python scripts/ablation/summarize_ablation.py --root logs/ablation/DUCT
```
